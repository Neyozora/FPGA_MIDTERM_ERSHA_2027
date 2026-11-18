// =============================================================================
// Module: de10_standard_top_m2.v
// Description:
//   Method 2 Physical Hardware Top-Level Wrapper for Terasic DE10-Standard
//   (Intel Cyclone V SoC: 5CSXFC6D6F31C6) featuring:
//     1. 4-Layer Parallel CIFAR-10 RGB CNN (`cnn_top_m2.v`)
//     2. 10-Class CIFAR-10 On-Chip Test ROM (`board_img_rom`, 30,720 x 8b)
//     3. Live 640x480 @ 60 Hz Color RGB VGA HUD (`vga_cnn_hud_m2.v`)
//     4. 32-Pixel Row-Burst RGB USB-JTAG Bridge (`source_width = 272`)
//     5. SW[3:0] class select, SW[9] cycle counter, HEX5..0, LEDR[9:0]
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module de10_standard_top_m2 #(
    parameter DE10_IMG10_MEM_FILE = "mem_m2/de10_images_10_m2.mem"
) (
    input  wire       CLOCK_50,
    input  wire [3:0] KEY,
    input  wire [9:0] SW,
    output wire [9:0] LEDR,
    output wire [6:0] HEX0,
    output wire [6:0] HEX1,
    output wire [6:0] HEX2,
    output wire [6:0] HEX3,
    output wire [6:0] HEX4,
    output wire [6:0] HEX5,

    // ADV7123 24-bit VGA DAC Output Pins
    output wire       VGA_CLK,
    output wire       VGA_HS,
    output wire       VGA_VS,
    output wire       VGA_BLANK_N,
    output wire       VGA_SYNC_N,
    output wire [7:0] VGA_R,
    output wire [7:0] VGA_G,
    output wire [7:0] VGA_B,

    // Cyclone V HPS DDR3 Pins (Dedicated Hard-IP)
    output wire [14:0] HPS_DDR3_ADDR,
    output wire [2:0]  HPS_DDR3_BA,
    output wire        HPS_DDR3_CK_P,
    output wire        HPS_DDR3_CK_N,
    output wire        HPS_DDR3_CKE,
    output wire        HPS_DDR3_CS_N,
    output wire        HPS_DDR3_RAS_N,
    output wire        HPS_DDR3_CAS_N,
    output wire        HPS_DDR3_WE_N,
    output wire        HPS_DDR3_RESET_N,
    inout  wire [31:0] HPS_DDR3_DQ,
    inout  wire [3:0]  HPS_DDR3_DQS_P,
    inout  wire [3:0]  HPS_DDR3_DQS_N,
    output wire        HPS_DDR3_ODT,
    output wire [3:0]  HPS_DDR3_DM,
    input  wire        HPS_DDR3_RZQ
);

    // -------------------------------------------------------------------------
    // 1. Input Synchronization & LFSR Randomizer
    // -------------------------------------------------------------------------
    reg [9:0] sw_meta, sw_sync, sw_prev;
    reg [3:0] key_meta, key_sync, key_prev;
    reg [15:0] lfsr;

    always @(posedge CLOCK_50) begin
        sw_meta  <= SW;
        sw_sync  <= sw_meta;
        sw_prev  <= sw_sync;
        key_meta <= KEY;
        key_sync <= key_meta;
        key_prev <= key_sync;
        if (~key_sync[0])
            lfsr <= 16'hACE1;
        else
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    end

    wire rst        = ~key_sync[0];
    wire key1_trig  = (key_prev[1] && !key_sync[1]);
    wire key2_rand  = (key_prev[2] && !key_sync[2]);
    wire sw_changed = (sw_sync[3:0] != sw_prev[3:0]);
    wire [3:0] sel_img  = (sw_sync[3:0] <= 4'd9) ? sw_sync[3:0] : 4'd0;
    wire [3:0] rand_img = (lfsr[3:0] <= 4'd9) ? lfsr[3:0] : (lfsr[3:0] - 4'd6);

    // -------------------------------------------------------------------------
    // 2. On-Chip 10-Class CIFAR-10 Test ROM (10 x 3072 x 8b = 30,720 bytes)
    // -------------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg [7:0] board_img_rom [0:30719];
    initial $readmemh(DE10_IMG10_MEM_FILE, board_img_rom);

    // -------------------------------------------------------------------------
    // 3. USB-JTAG Bridge (272-bit In-System Sources & Probes)
    // -------------------------------------------------------------------------
    wire [271:0] jtag_src;
    reg  [127:0] jtag_probe;

    altsource_probe #(
        .enable_metastability ("YES"),
        .instance_id          ("CNN2"),
        .probe_width          (128),
        .sld_auto_instance_index ("YES"),
        .sld_instance_index   (0),
        .source_initial_value ("0"),
        .source_width         (272)
    ) u_jtag_isp (
        .source     (jtag_src),
        .source_clk (CLOCK_50),
        .probe      (jtag_probe)
    );

    reg  [271:0] jtag_src_d1, jtag_src_d2;
    always @(posedge CLOCK_50) begin
        if (rst) begin
            jtag_src_d1 <= 272'd0;
            jtag_src_d2 <= 272'd0;
        end else begin
            jtag_src_d1 <= jtag_src;
            jtag_src_d2 <= jtag_src_d1;
        end
    end

    wire        jtag_wr_edge = (jtag_src_d1[261] != jtag_src_d2[261]);
    wire        jtag_pc_mode = jtag_src_d1[263];
    wire        jtag_infer   = (jtag_src_d1[264] != jtag_src_d2[264]);
    wire [4:0]  jtag_row_y   = jtag_src_d1[260:256];
    wire [1:0]  jtag_chan    = jtag_src_d1[268:267]; // 0=R, 1=G, 2=B

    // -------------------------------------------------------------------------
    // 3B. Cyclone V HPS Lightweight AXI Bridge (0xFF200000)
    // -------------------------------------------------------------------------
    wire [31:0] avm_writedata;
    wire [7:0]  avm_address;
    wire        avm_write;
    wire        avm_read;
    reg  [31:0] avm_readdata;
    reg         avm_readdatavalid;

    hps_cnn_qsys u_hps_qsys (
        .clk_clk               (CLOCK_50),
        .reset_reset_n         (~rst),
        .cnn_avm_waitrequest   (1'b0),
        .cnn_avm_readdata      (avm_readdata),
        .cnn_avm_readdatavalid (avm_readdatavalid),
        .cnn_avm_burstcount    (),
        .cnn_avm_writedata     (avm_writedata),
        .cnn_avm_address       (avm_address),
        .cnn_avm_write         (avm_write),
        .cnn_avm_read          (avm_read),
        .cnn_avm_byteenable    (),
        .cnn_avm_debugaccess   (),
        .memory_mem_a          (HPS_DDR3_ADDR),
        .memory_mem_ba         (HPS_DDR3_BA),
        .memory_mem_ck         (HPS_DDR3_CK_P),
        .memory_mem_ck_n       (HPS_DDR3_CK_N),
        .memory_mem_cke        (HPS_DDR3_CKE),
        .memory_mem_cs_n       (HPS_DDR3_CS_N),
        .memory_mem_ras_n      (HPS_DDR3_RAS_N),
        .memory_mem_cas_n      (HPS_DDR3_CAS_N),
        .memory_mem_we_n       (HPS_DDR3_WE_N),
        .memory_mem_reset_n    (HPS_DDR3_RESET_N),
        .memory_mem_dq         (HPS_DDR3_DQ),
        .memory_mem_dqs        (HPS_DDR3_DQS_P),
        .memory_mem_dqs_n      (HPS_DDR3_DQS_N),
        .memory_mem_odt        (HPS_DDR3_ODT),
        .memory_mem_dm         (HPS_DDR3_DM),
        .memory_oct_rzqin      (HPS_DDR3_RZQ)
    );

    wire hps_pix_wr  = avm_write && (avm_address == 8'h08);
    wire hps_start_w = avm_write && (avm_address == 8'h00) && avm_writedata[0];

    // -------------------------------------------------------------------------
    // 4. Board Loader FSM & CNN Control
    // -------------------------------------------------------------------------
    localparam [2:0]
        B_INIT       = 3'd0,
        B_ROM_RD     = 3'd1,
        B_ROM_WR     = 3'd2,
        B_JTAG_BURST = 3'd3,
        B_CNN_START  = 3'd4,
        B_CNN_WAIT   = 3'd5,
        B_IDLE       = 3'd6;

    reg [2:0]   b_state;
    reg [3:0]   active_img_class;
    reg [11:0]  load_idx;
    reg [7:0]   rom_data_reg;
    reg [4:0]   burst_col;
    reg [255:0] burst_row_buf;
    reg [4:0]   burst_row_y;
    reg [1:0]   burst_chan;

    reg         cnn_start;
    wire        cnn_done;
    wire [3:0]  predicted_class;
    wire [79:0] dbg_fc_scores_flat;
    reg         ext_img_wr_en;
    reg  [11:0] ext_img_wr_addr;
    reg  [7:0]  ext_img_wr_data;

    reg [23:0]  cycle_count;
    reg         cnn_done_latched;

    always @(posedge CLOCK_50) begin
        if (rst) begin
            b_state          <= B_INIT;
            active_img_class <= sel_img;
            load_idx         <= 12'd0;
            cnn_start        <= 1'b0;
            ext_img_wr_en    <= 1'b0;
            ext_img_wr_addr  <= 12'd0;
            ext_img_wr_data  <= 8'd0;
            cycle_count      <= 24'd0;
            cnn_done_latched <= 1'b0;
        end else begin
            cnn_start     <= 1'b0;
            ext_img_wr_en <= 1'b0;

            if (hps_pix_wr) begin
                ext_img_wr_en   <= 1'b1;
                ext_img_wr_addr <= avm_writedata[11:0];
                ext_img_wr_data <= avm_writedata[19:12];
            end else if (hps_start_w) begin
                cnn_done_latched <= 1'b0;
                cycle_count      <= 24'd0;
                b_state          <= B_CNN_START;
            end else if (jtag_wr_edge) begin
                burst_row_buf <= jtag_src_d1[255:0];
                burst_row_y   <= jtag_row_y;
                burst_chan    <= jtag_chan;
                burst_col     <= 5'd0;
                b_state       <= B_JTAG_BURST;
            end else if (jtag_infer) begin
                cnn_done_latched <= 1'b0;
                cycle_count      <= 24'd0;
                b_state          <= B_CNN_START;
            end else if (sw_changed || key1_trig || key2_rand) begin
                active_img_class <= key2_rand ? rand_img : sel_img;
                load_idx         <= 12'd0;
                b_state          <= B_ROM_RD;
            end else begin
                case (b_state)
                    B_INIT: begin
                        active_img_class <= sel_img;
                        load_idx         <= 12'd0;
                        b_state          <= B_ROM_RD;
                    end

                    B_ROM_RD: begin
                        rom_data_reg <= board_img_rom[({11'd0, active_img_class} * 15'd3072) + {3'd0, load_idx}];
                        b_state      <= B_ROM_WR;
                    end

                    B_ROM_WR: begin
                        ext_img_wr_en   <= 1'b1;
                        ext_img_wr_addr <= load_idx;
                        ext_img_wr_data <= rom_data_reg;
                        if (load_idx == 12'd3071) begin
                            cnn_done_latched <= 1'b0;
                            cycle_count      <= 24'd0;
                            b_state          <= B_CNN_START;
                        end else begin
                            load_idx <= load_idx + 12'd1;
                            b_state  <= B_ROM_RD;
                        end
                    end

                    B_JTAG_BURST: begin
                        ext_img_wr_en   <= 1'b1;
                        ext_img_wr_addr <= {burst_chan, burst_row_y, burst_col};
                        ext_img_wr_data <= burst_row_buf[burst_col*8 +: 8];
                        if (burst_col == 5'd31)
                            b_state <= B_IDLE;
                        else
                            burst_col <= burst_col + 5'd1;
                    end

                    B_CNN_START: begin
                        cnn_start <= 1'b1;
                        b_state   <= B_CNN_WAIT;
                    end

                    B_CNN_WAIT: begin
                        cycle_count <= cycle_count + 24'd1;
                        if (cnn_done) begin
                            cnn_done_latched <= 1'b1;
                            b_state          <= B_IDLE;
                        end
                    end

                    B_IDLE: ;
                    default: b_state <= B_IDLE;
                endcase
            end
        end
    end

    always @(posedge CLOCK_50) begin
        avm_readdatavalid <= avm_read;
        if (avm_read) begin
            case (avm_address)
                8'h04: avm_readdata <= {27'd0, cnn_done_latched, predicted_class};
                8'h0C: avm_readdata <= {8'd0, cycle_count};
                8'h10: avm_readdata <= {{24{dbg_fc_scores_flat[7]}},  dbg_fc_scores_flat[7:0]};
                8'h14: avm_readdata <= {{24{dbg_fc_scores_flat[15]}}, dbg_fc_scores_flat[15:8]};
                8'h18: avm_readdata <= {{24{dbg_fc_scores_flat[23]}}, dbg_fc_scores_flat[23:16]};
                8'h1C: avm_readdata <= {{24{dbg_fc_scores_flat[31]}}, dbg_fc_scores_flat[31:24]};
                8'h20: avm_readdata <= {{24{dbg_fc_scores_flat[39]}}, dbg_fc_scores_flat[39:32]};
                8'h24: avm_readdata <= {{24{dbg_fc_scores_flat[47]}}, dbg_fc_scores_flat[47:40]};
                8'h28: avm_readdata <= {{24{dbg_fc_scores_flat[55]}}, dbg_fc_scores_flat[55:48]};
                8'h2C: avm_readdata <= {{24{dbg_fc_scores_flat[63]}}, dbg_fc_scores_flat[63:56]};
                8'h30: avm_readdata <= {{24{dbg_fc_scores_flat[71]}}, dbg_fc_scores_flat[71:64]};
                8'h34: avm_readdata <= {{24{dbg_fc_scores_flat[79]}}, dbg_fc_scores_flat[79:72]};
                8'h38: avm_readdata <= {14'd0, sw_sync, active_img_class, key_sync};
                default: avm_readdata <= 32'h0;
            endcase
        end
    end

    always @(posedge CLOCK_50) begin
        jtag_probe <= {
            16'd0,
            cycle_count,
            3'd0, cnn_done_latched,
            predicted_class,
            dbg_fc_scores_flat
        };
    end

    // -------------------------------------------------------------------------
    // 5. CNN Core Instance (Method 2)
    // -------------------------------------------------------------------------
    cnn_top_m2 u_cnn_top_m2 (
        .clk                (CLOCK_50),
        .rst                (rst),
        .start              (cnn_start),
        .done               (cnn_done),
        .predicted_class    (predicted_class),
        .ext_img_wr_en      (ext_img_wr_en),
        .ext_img_wr_addr    (ext_img_wr_addr),
        .ext_img_wr_data    (ext_img_wr_data),
        .dbg_fc_scores_flat (dbg_fc_scores_flat)
    );

    // -------------------------------------------------------------------------
    // 6. RGB Mirror BRAMs for Live Color VGA HUD
    // -------------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg [7:0] vga_r_ram [0:1023];
    (* ramstyle = "M10K" *) reg [7:0] vga_g_ram [0:1023];
    (* ramstyle = "M10K" *) reg [7:0] vga_b_ram [0:1023];

    wire [9:0] vga_img_rd_addr;
    reg  [7:0] vga_r_q, vga_g_q, vga_b_q;

    always @(posedge CLOCK_50) begin
        if (ext_img_wr_en) begin
            if (ext_img_wr_addr[11:10] == 2'd0) vga_r_ram[ext_img_wr_addr[9:0]] <= ext_img_wr_data;
            if (ext_img_wr_addr[11:10] == 2'd1) vga_g_ram[ext_img_wr_addr[9:0]] <= ext_img_wr_data;
            if (ext_img_wr_addr[11:10] == 2'd2) vga_b_ram[ext_img_wr_addr[9:0]] <= ext_img_wr_data;
        end
        vga_r_q <= vga_r_ram[vga_img_rd_addr];
        vga_g_q <= vga_g_ram[vga_img_rd_addr];
        vga_b_q <= vga_b_ram[vga_img_rd_addr];
    end

    vga_cnn_hud_m2 u_vga_hud (
        .clk_50             (CLOCK_50),
        .rst                (rst),
        .vga_img_rd_addr    (vga_img_rd_addr),
        .vga_img_rd_data    ({vga_b_q, vga_g_q, vga_r_q}),
        .predicted_class    (predicted_class),
        .active_img_class   (active_img_class),
        .pc_mode_active     (jtag_pc_mode),
        .dbg_fc_scores_flat (dbg_fc_scores_flat),
        .VGA_CLK            (VGA_CLK),
        .VGA_HS             (VGA_HS),
        .VGA_VS             (VGA_VS),
        .VGA_BLANK_N        (VGA_BLANK_N),
        .VGA_SYNC_N         (VGA_SYNC_N),
        .VGA_R              (VGA_R),
        .VGA_G              (VGA_G),
        .VGA_B              (VGA_B)
    );

    // -------------------------------------------------------------------------
    // 7. LEDs & 7-Segment Displays (HEX5..HEX0)
    // -------------------------------------------------------------------------
    assign LEDR = cnn_done_latched ? (10'd1 << predicted_class) : 10'd0;

    function [6:0] seg7_hex;
        input [3:0] nib;
        begin
            case (nib)
                4'h0: seg7_hex = 7'b1000000;
                4'h1: seg7_hex = 7'b1111001;
                4'h2: seg7_hex = 7'b0100100;
                4'h3: seg7_hex = 7'b0110000;
                4'h4: seg7_hex = 7'b0011001;
                4'h5: seg7_hex = 7'b0010010;
                4'h6: seg7_hex = 7'b0000010;
                4'h7: seg7_hex = 7'b1111000;
                4'h8: seg7_hex = 7'b0000000;
                4'h9: seg7_hex = 7'b0010000;
                4'hA: seg7_hex = 7'b0001000;
                4'hB: seg7_hex = 7'b0000011;
                4'hC: seg7_hex = 7'b1000110;
                4'hD: seg7_hex = 7'b0100001;
                4'hE: seg7_hex = 7'b0000110;
                4'hF: seg7_hex = 7'b0001110;
            endcase
        end
    endfunction

    wire [23:0] disp_val = sw_sync[9] ? cycle_count : {active_img_class, 16'h0000, predicted_class};
    assign HEX5 = seg7_hex(disp_val[23:20]);
    assign HEX4 = seg7_hex(disp_val[19:16]);
    assign HEX3 = seg7_hex(disp_val[15:12]);
    assign HEX2 = seg7_hex(disp_val[11:8]);
    assign HEX1 = seg7_hex(disp_val[7:4]);
    assign HEX0 = seg7_hex(disp_val[3:0]);

endmodule
