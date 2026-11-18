// =============================================================================
// Module: de10_standard_top.v
// Description:
//   Physical Hardware Top-Level Wrapper for the Terasic DE10-Standard Board
//   (Intel Cyclone V SoC: 5CSXFC6D6F31C6) featuring:
//     1. Ultra-Fast 32-Pixel Row-Burst USB-JTAG Bridge (`source_width = 272`):
//          - Transfers an entire 32-pixel row (256 bits) per JTAG write (32x
//            faster than single-pixel JTAG: full 32x32 image in 32 writes!).
//          - 1-Command Hardware Clear (`source[262]`) wipes all 1,024 pixels
//            in 20.5 microseconds!
//     2. Live 640x480 @ 60 Hz VGA Monitor Output (`vga_cnn_hud.v`)
//     3. PS/2 Mouse Controller (`ps2_mouse_ctrl.v`) + KEY[3] Clear + KEY[2]
//        Hardware LFSR Randomizer
//     4. Slide Switches `SW[3:0]` (ROM image 0..9) & `SW[9]` (302978 cycles)
//     5. 6 Seven-Segment Displays (`HEX5..HEX0`) & 10 Red LEDs (`LEDR[9:0]`)
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module de10_standard_top #(
    parameter DE10_IMG10_MEM_FILE = "mem/de10_images_10.mem"
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

    // PS/2 Mouse Port on DE10-Standard
    inout  wire       PS2_CLK,
    inout  wire       PS2_DAT,

    // ADV7123 24-bit VGA DAC Output Pins
    output wire       VGA_CLK,
    output wire       VGA_HS,
    output wire       VGA_VS,
    output wire       VGA_BLANK_N,
    output wire       VGA_SYNC_N,
    output wire [7:0] VGA_R,
    output wire [7:0] VGA_G,
    output wire [7:0] VGA_B
);

    // -------------------------------------------------------------------------
    // 1. Input Synchronization (Switches & Pushbuttons) + LFSR Randomizer
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

    wire rst        = ~key_sync[0];                    // KEY[0]: Hardware Reset
    wire key1_trig  = (key_prev[1] && !key_sync[1]);   // KEY[1]: Re-run current image
    wire key2_rand  = (key_prev[2] && !key_sync[2]);   // KEY[2]: Hardware Randomizer!
    wire key3_clear = (key_prev[3] && !key_sync[3]);   // KEY[3]: Clear Drawing Canvas!
    wire sw_changed = (sw_sync[3:0] != sw_prev[3:0]);  // Physical switch moved
    wire [3:0] sel_img  = (sw_sync[3:0] <= 4'd9) ? sw_sync[3:0] : 4'd0;
    wire [3:0] rand_img = (lfsr[3:0] <= 4'd9) ? lfsr[3:0] : (lfsr[3:0] - 4'd6);

    // -------------------------------------------------------------------------
    // 2. PS/2 Mouse Controller (`ps2_mouse_ctrl.v`)
    // -------------------------------------------------------------------------
    wire [7:0] mouse_x;
    wire [7:0] mouse_y;
    wire [4:0] grid_x;
    wire [4:0] grid_y;
    wire       mouse_left;
    wire       mouse_right;
    wire       mouse_middle;
    wire       mouse_pkt_valid;

    ps2_mouse_ctrl u_ps2_mouse (
        .clk_50       (CLOCK_50),
        .rst          (rst),
        .PS2_CLK      (PS2_CLK),
        .PS2_DAT      (PS2_DAT),
        .mouse_x      (mouse_x),
        .mouse_y      (mouse_y),
        .grid_x       (grid_x),
        .grid_y       (grid_y),
        .btn_left     (mouse_left),
        .btn_right    (mouse_right),
        .btn_middle   (mouse_middle),
        .packet_valid (mouse_pkt_valid)
    );

    // -------------------------------------------------------------------------
    // 3. High-Speed 272-Bit USB-JTAG Row-Burst Bridge (`altsource_probe`)
    // -------------------------------------------------------------------------
    // JTAG Source Layout (272 bits, PC -> FPGA):
    //   source[255:0]   : 32 x 8-bit signed INT8 pixels for one full row `row_y`
    //   source[260:256] : 5-bit `row_y` address (0..31)
    //   source[261]     : Toggle to burst-write all 32 pixels of `row_y` into RAM
    //   source[262]     : Toggle to clear all 1,024 pixels to 0 AND run CNN
    //   source[263]     : Toggle to run CNN on current `img_ram` contents
    //   source[267:264] : 4-bit `true_class` (0..9) for custom/Pinterest image
    //   source[268]     : Toggle to load on-chip ROM image `source[271:269]`
    //   source[271:268] : [268]=ROM toggle, reuse [267:264] as ROM slot 0..9
    wire [127:0] jtag_probe;
    wire [271:0] jtag_source;

    altsource_probe #(
        .enable_metastability    ("YES"),
        .instance_id             ("CNN0"),
        .probe_width             (128),
        .sld_auto_instance_index ("YES"),
        .sld_instance_index      (0),
        .source_initial_value    ("0"),
        .source_width            (272)
    ) u_jtag_bridge (
        .probe      (jtag_probe),
        .source     (jtag_source),
        .source_clk (CLOCK_50),
        .source_ena (1'b1)
    );

    reg [15:0] jtag_ctrl_d1, jtag_ctrl_d2;
    reg [255:0] jtag_row_buf;
    always @(posedge CLOCK_50) begin
        jtag_ctrl_d1 <= jtag_source[271:256];
        jtag_ctrl_d2 <= jtag_ctrl_d1;
        jtag_row_buf <= jtag_source[255:0];
    end

    wire [4:0] pc_row_y         = jtag_ctrl_d1[4:0];
    wire       pc_row_wr_trig   = (jtag_ctrl_d1[5]  ^ jtag_ctrl_d2[5]);
    wire       pc_clear_trig    = (jtag_ctrl_d1[6]  ^ jtag_ctrl_d2[6]);
    wire       pc_cust_run_trig = (jtag_ctrl_d1[7]  ^ jtag_ctrl_d2[7]);
    wire [3:0] pc_cls_or_rom    = (jtag_ctrl_d1[11:8] <= 4'd9) ? jtag_ctrl_d1[11:8] : 4'd0;
    wire       pc_rom_run_trig  = (jtag_ctrl_d1[12] ^ jtag_ctrl_d2[12]);

    // -------------------------------------------------------------------------
    // 4. On-Chip 10-Class Test Image ROM (10 images x 1024 pixels = 10,240 B)
    // -------------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg signed [7:0] board_img_rom [0:10239];

    initial begin
        $readmemh(DE10_IMG10_MEM_FILE, board_img_rom);
    end

    reg [13:0]       rom_rd_addr;
    reg signed [7:0] rom_rd_data;

    always @(posedge CLOCK_50) begin
        rom_rd_data <= board_img_rom[rom_rd_addr];
    end

    // -------------------------------------------------------------------------
    // 5. Board Loader, 32-Pixel Row Burst Writer, Clear, & Mouse Brush FSM
    // -------------------------------------------------------------------------
    localparam [2:0]
        B_INIT_LOAD    = 3'd0,
        B_ROM_ADDR     = 3'd1,
        B_ROM_WRITE    = 3'd2,
        B_START_CNN    = 3'd3,
        B_WAIT_CNN     = 3'd4,
        B_IDLE_DONE    = 3'd5,
        B_CLEAR_CANVAS = 3'd6,
        B_ROW_BURST    = 3'd7;

    reg [2:0]        b_state;
    reg [3:0]        active_img;
    reg [9:0]        load_idx;
    reg [4:0]        burst_row_y;
    reg [4:0]        burst_col_x;
    reg [255:0]      burst_shift;
    reg              ext_wr_en;
    reg [9:0]        ext_wr_addr;
    reg signed [7:0] ext_wr_data;
    reg              cnn_start;
    wire             cnn_done;
    wire [3:0]       cnn_pred_class;
    wire [79:0]      dbg_fc_scores_flat;

    reg [3:0]        latched_pred;
    reg              has_result;
    reg              pc_mode_active;
    reg              draw_mode_active;

    // 6-digit synchronous BCD cycle counter (000000 .. 999999)
    reg [3:0] bcd0, bcd1, bcd2, bcd3, bcd4, bcd5;

    always @(posedge CLOCK_50) begin
        if (rst) begin
            b_state          <= B_INIT_LOAD;
            active_img       <= 4'd0;
            load_idx         <= 10'd0;
            burst_row_y      <= 5'd0;
            burst_col_x      <= 5'd0;
            burst_shift      <= 256'd0;
            rom_rd_addr      <= 14'd0;
            ext_wr_en        <= 1'b0;
            ext_wr_addr      <= 10'd0;
            ext_wr_data      <= 8'sd0;
            cnn_start        <= 1'b0;
            latched_pred     <= 4'd0;
            has_result       <= 1'b0;
            pc_mode_active   <= 1'b0;
            draw_mode_active <= 1'b0;
            bcd0 <= 4'd0; bcd1 <= 4'd0; bcd2 <= 4'd0;
            bcd3 <= 4'd0; bcd4 <= 4'd0; bcd5 <= 4'd0;
        end else begin
            ext_wr_en <= 1'b0;
            cnn_start <= 1'b0;

            case (b_state)
                B_INIT_LOAD: begin
                    active_img       <= sel_img;
                    pc_mode_active   <= 1'b0;
                    draw_mode_active <= 1'b0;
                    load_idx         <= 10'd0;
                    rom_rd_addr      <= {sel_img, 10'd0};
                    b_state          <= B_ROM_ADDR;
                end

                B_ROM_ADDR: begin
                    b_state <= B_ROM_WRITE;
                end

                B_ROM_WRITE: begin
                    ext_wr_en   <= 1'b1;
                    ext_wr_addr <= load_idx;
                    ext_wr_data <= rom_rd_data;
                    if (load_idx == 10'd1023) begin
                        b_state <= B_START_CNN;
                    end else begin
                        load_idx    <= load_idx + 10'd1;
                        rom_rd_addr <= {active_img, load_idx + 10'd1};
                        b_state     <= B_ROM_ADDR;
                    end
                end

                // Clear all 1,024 pixels to 0 in 20.5 us!
                B_CLEAR_CANVAS: begin
                    ext_wr_en   <= 1'b1;
                    ext_wr_addr <= load_idx;
                    ext_wr_data <= 8'sd0;
                    if (load_idx == 10'd1023) begin
                        b_state <= B_IDLE_DONE;
                    end else begin
                        load_idx <= load_idx + 10'd1;
                    end
                end

                // Write all 32 pixels of row `burst_row_y` in 32 clock cycles (0.64 us)!
                B_ROW_BURST: begin
                    ext_wr_en   <= 1'b1;
                    ext_wr_addr <= {burst_row_y, burst_col_x};
                    ext_wr_data <= burst_shift[7:0];
                    burst_shift <= {8'd0, burst_shift[255:8]};
                    if (burst_col_x == 5'd31) begin
                        b_state <= B_IDLE_DONE;
                    end else begin
                        burst_col_x <= burst_col_x + 5'd1;
                    end
                end

                B_START_CNN: begin
                    cnn_start <= 1'b1;
                    bcd0 <= 4'd1; bcd1 <= 4'd0; bcd2 <= 4'd0;
                    bcd3 <= 4'd0; bcd4 <= 4'd0; bcd5 <= 4'd0;
                    b_state <= B_WAIT_CNN;
                end

                B_WAIT_CNN: begin
                    if (cnn_done) begin
                        latched_pred <= cnn_pred_class;
                        if (draw_mode_active)
                            active_img <= cnn_pred_class;
                        has_result   <= 1'b1;
                        b_state      <= B_IDLE_DONE;
                    end else begin
                        if (bcd0 == 4'd9) begin
                            bcd0 <= 4'd0;
                            if (bcd1 == 4'd9) begin
                                bcd1 <= 4'd0;
                                if (bcd2 == 4'd9) begin
                                    bcd2 <= 4'd0;
                                    if (bcd3 == 4'd9) begin
                                        bcd3 <= 4'd0;
                                        if (bcd4 == 4'd9) begin
                                            bcd4 <= 4'd0;
                                            if (bcd5 != 4'd9) bcd5 <= bcd5 + 4'd1;
                                        end else bcd4 <= bcd4 + 4'd1;
                                    end else bcd3 <= bcd3 + 4'd1;
                                end else bcd2 <= bcd2 + 4'd1;
                            end else bcd1 <= bcd1 + 4'd1;
                        end else bcd0 <= bcd0 + 4'd1;
                    end
                end

                B_IDLE_DONE: begin
                    if (key3_clear || pc_clear_trig || (mouse_pkt_valid && mouse_middle)) begin
                        // Instant 20us hardware clear from KEY[3], PC GUI, or Middle Mouse!
                        draw_mode_active <= 1'b1;
                        pc_mode_active   <= 1'b1;
                        load_idx         <= 10'd0;
                        b_state          <= B_CLEAR_CANVAS;
                    end else if (pc_row_wr_trig) begin
                        // Instant 32-pixel row burst from PC GUI!
                        pc_mode_active <= 1'b1;
                        burst_row_y    <= pc_row_y;
                        burst_col_x    <= 5'd0;
                        burst_shift    <= jtag_row_buf;
                        b_state        <= B_ROW_BURST;
                    end else if (pc_cust_run_trig) begin
                        active_img       <= pc_cls_or_rom;
                        pc_mode_active   <= 1'b1;
                        draw_mode_active <= 1'b0;
                        b_state          <= B_START_CNN;
                    end else if (pc_rom_run_trig) begin
                        active_img       <= pc_cls_or_rom;
                        pc_mode_active   <= 1'b1;
                        draw_mode_active <= 1'b0;
                        load_idx         <= 10'd0;
                        rom_rd_addr      <= {pc_cls_or_rom, 10'd0};
                        b_state          <= B_ROM_ADDR;
                    end else if (mouse_left || mouse_right) begin
                        draw_mode_active <= 1'b1;
                        pc_mode_active   <= 1'b0;
                        ext_wr_en        <= 1'b1;
                        ext_wr_addr      <= {grid_y, grid_x};
                        ext_wr_data      <= mouse_right ? 8'sd0 : 8'sd125;
                        b_state          <= B_START_CNN;
                    end else if (sw_changed || key1_trig) begin
                        b_state <= B_INIT_LOAD;
                    end else if (key2_rand) begin
                        active_img       <= rand_img;
                        pc_mode_active   <= 1'b0;
                        draw_mode_active <= 1'b0;
                        load_idx         <= 10'd0;
                        rom_rd_addr      <= {rand_img, 10'd0};
                        b_state          <= B_ROM_ADDR;
                    end
                end

                default: b_state <= B_INIT_LOAD;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // 6. Dual-Port Mirror of 32x32 Input Image for Live VGA Framebuffer
    // -------------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg [7:0] vga_img_ram [0:1023];
    wire [9:0] vga_img_rd_addr;
    reg  [7:0] vga_img_rd_data;

    initial begin
        $readmemh("mem/test_image.mem", vga_img_ram);
    end

    always @(posedge CLOCK_50) begin
        if (ext_wr_en) begin
            vga_img_ram[ext_wr_addr] <= ext_wr_data[7:0];
        end
        vga_img_rd_data <= vga_img_ram[vga_img_rd_addr];
    end

    // -------------------------------------------------------------------------
    // 7. Instantiate Two-Layer CNN Accelerator (`cnn_top.v`)
    // -------------------------------------------------------------------------
    cnn_top u_cnn_top (
        .clk                (CLOCK_50),
        .rst                (rst),
        .start              (cnn_start),
        .done               (cnn_done),
        .predicted_class    (cnn_pred_class),
        .ext_img_wr_en      (ext_wr_en),
        .ext_img_wr_addr    (ext_wr_addr),
        .ext_img_wr_data    (ext_wr_data),
        .dbg_fc_scores_flat (dbg_fc_scores_flat)
    );

    // -------------------------------------------------------------------------
    // 8. Instantiate Live 640x480 @ 60 Hz VGA Dashboard HUD (`vga_cnn_hud.v`)
    // -------------------------------------------------------------------------
    vga_cnn_hud u_vga_hud (
        .clk_50             (CLOCK_50),
        .rst                (rst),
        .vga_img_rd_addr    (vga_img_rd_addr),
        .vga_img_rd_data    (vga_img_rd_data),
        .predicted_class    (latched_pred),
        .active_img_class   (active_img),
        .pc_mode_active     (pc_mode_active),
        .dbg_fc_scores_flat (dbg_fc_scores_flat),
        .mouse_x            (mouse_x),
        .mouse_y            (mouse_y),
        .mouse_left         (mouse_left),
        .mouse_right        (mouse_right),
        .VGA_CLK            (VGA_CLK),
        .VGA_HS             (VGA_HS),
        .VGA_VS             (VGA_VS),
        .VGA_BLANK_N        (VGA_BLANK_N),
        .VGA_SYNC_N         (VGA_SYNC_N),
        .VGA_R              (VGA_R),
        .VGA_G              (VGA_G),
        .VGA_B              (VGA_B)
    );

    // Pack 128-bit JTAG probe vector for Python GUI
    assign jtag_probe = {
        dbg_fc_scores_flat,                           // [127:48] 10 x INT8 class scores
        bcd5, bcd4, bcd3, bcd2, bcd1, bcd0,           // [47:24]  6-digit BCD cycle count
        pc_mode_active,                               // [23]     1 = PC triggered, 0 = SW
        has_result,                                   // [22]     1 = Inference done
        active_img,                                   // [21:18]  True/selected class 0..9
        latched_pred,                                 // [17:14]  FPGA predicted_class 0..9
        key_sync,                                     // [13:10]  KEY[3:0]
        sw_sync                                       // [9:0]    SW[9:0]
    };

    // -------------------------------------------------------------------------
    // 9. LEDR[9:0] & Seven-Segment Display Formatting (HEX5 .. HEX0)
    // -------------------------------------------------------------------------
    assign LEDR = has_result ? (10'b1 << latched_pred) : 10'b0000000001;

    function [6:0] seg7_digit;
        input [3:0] val;
        begin
            case (val)
                4'h0: seg7_digit = 7'b1000000;
                4'h1: seg7_digit = 7'b1111001;
                4'h2: seg7_digit = 7'b0100100;
                4'h3: seg7_digit = 7'b0110000;
                4'h4: seg7_digit = 7'b0011001;
                4'h5: seg7_digit = 7'b0010010;
                4'h6: seg7_digit = 7'b0000010;
                4'h7: seg7_digit = 7'b1111000;
                4'h8: seg7_digit = 7'b0000000;
                4'h9: seg7_digit = 7'b0010000;
                default: seg7_digit = 7'b1111111;
            endcase
        end
    endfunction

    localparam [6:0]
        SEG_DASH = 7'b0111111,
        SEG_P    = 7'b0001100,
        SEG_A    = 7'b0001000,
        SEG_S    = 7'b0010010,
        SEG_E    = 7'b0000110,
        SEG_R    = 7'b0101111,
        SEG_D    = 7'b0100001;

    wire is_pass = (latched_pred == active_img);

    assign HEX5 = sw_sync[9] ? seg7_digit(bcd5) : (draw_mode_active ? SEG_D : seg7digit_or_0(active_img));
    assign HEX4 = sw_sync[9] ? seg7_digit(bcd4) : SEG_DASH;
    assign HEX3 = sw_sync[9] ? seg7_digit(bcd3) : (is_pass ? SEG_P : SEG_E);
    assign HEX2 = sw_sync[9] ? seg7_digit(bcd2) : (is_pass ? SEG_A : SEG_R);
    assign HEX1 = sw_sync[9] ? seg7_digit(bcd1) : (is_pass ? SEG_S : SEG_R);
    assign HEX0 = sw_sync[9] ? seg7_digit(bcd0) : seg7_digit(latched_pred);

    function [6:0] seg7digit_or_0;
        input [3:0] v;
        begin
            seg7digit_or_0 = seg7_digit(v);
        end
    endfunction

endmodule
