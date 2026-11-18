// =============================================================================
// DESim Virtual DE-Board Top-Level Wrapper (`desim_cnn/top.v`)
// Optimized for DESim's 0.0001x real-time simulation speed:
//   - Renders a complete 160x120 VGA Dashboard WITH TEXT LABELS at 1 pixel per
//     CLOCK_50 cycle (19,200 cycles = ~3.8 seconds in DESim!).
//   - Top Header: "DE10 FPGA CNN CLASSIFIER"
//   - Left (X=8..71, Y=24..87): Live 32x32 Fashion-MNIST image (2x zoom)
//   - Right (X=76..154, Y=15..104): Class labels ("0 TSHRT".."9 BOOT ") + bars
//   - Bottom Bar (Y=110..117): "WINNER: <CLASS>" in Emerald Green!
// =============================================================================
`timescale 1ns / 1ps

module top (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [9:0]  SW,
    output wire [9:0]  LEDR,
    output wire [6:0]  HEX0,
    output wire [6:0]  HEX1,
    output wire [6:0]  HEX2,
    output wire [6:0]  HEX3,
    output wire [6:0]  HEX4,
    output wire [6:0]  HEX5,
    output reg  [9:0]  VGA_X,
    output reg  [8:0]  VGA_Y,
    output reg  [23:0] VGA_COLOR,
    output reg         plot
);

    // -------------------------------------------------------------------------
    // 1. Key / Switch Edge Detection & LFSR Randomizer
    // -------------------------------------------------------------------------
    reg [9:0]  sw_prev;
    reg [3:0]  key_prev;
    reg [15:0] lfsr;

    always @(posedge CLOCK_50) begin
        sw_prev  <= SW;
        key_prev <= KEY;
        if (KEY[0])
            lfsr <= 16'hACE1;
        else
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    end

    // In DESim, Push Buttons start unchecked (0). Use active-high triggers!
    wire rst        = KEY[0];
    wire key1_trig  = (!key_prev[1] && KEY[1]);
    wire key2_rand  = (!key_prev[2] && KEY[2]);
    wire key3_clear = (!key_prev[3] && KEY[3]);
    wire sw_changed = (SW[3:0] != sw_prev[3:0]);
    wire [3:0] sel_img  = (SW[3:0] <= 4'd9) ? SW[3:0] : 4'd0;
    wire [3:0] rand_img = (lfsr[3:0] <= 4'd9) ? lfsr[3:0] : (lfsr[3:0] - 4'd6);

    // -------------------------------------------------------------------------
    // 2. 10-Class Test Image ROM
    // -------------------------------------------------------------------------
    reg signed [7:0] board_img_rom [0:10239];
    initial begin
        $readmemh("mem/de10_images_10.mem", board_img_rom);
    end

    reg [13:0]       rom_rd_addr;
    reg signed [7:0] rom_rd_data;
    always @(posedge CLOCK_50) begin
        rom_rd_data <= board_img_rom[rom_rd_addr];
    end

    // -------------------------------------------------------------------------
    // 3. Loader & CNN Control FSM
    // -------------------------------------------------------------------------
    localparam [2:0]
        B_INIT_LOAD    = 3'd0,
        B_ROM_ADDR     = 3'd1,
        B_ROM_WRITE    = 3'd2,
        B_START_CNN    = 3'd3,
        B_WAIT_CNN     = 3'd4,
        B_IDLE_DONE    = 3'd5,
        B_CLEAR_CANVAS = 3'd6;

    reg [2:0]        b_state = B_INIT_LOAD;
    reg [3:0]        active_img = 4'd0;
    reg [9:0]        load_idx = 10'd0;
    reg              ext_wr_en = 1'b0;
    reg [9:0]        ext_wr_addr = 10'd0;
    reg signed [7:0] ext_wr_data = 8'sd0;
    reg              cnn_start = 1'b0;
    wire             cnn_done;
    wire [3:0]       cnn_pred_class;
    wire [79:0]      dbg_fc_scores_flat;

    reg [3:0]        latched_pred = 4'd0;
    reg              has_result = 1'b0;
    reg [3:0]        bcd0 = 0, bcd1 = 0, bcd2 = 0, bcd3 = 0, bcd4 = 0, bcd5 = 0;

    always @(posedge CLOCK_50) begin
        if (rst) begin
            b_state      <= B_INIT_LOAD;
            active_img   <= 4'd0;
            load_idx     <= 10'd0;
            rom_rd_addr  <= 14'd0;
            ext_wr_en    <= 1'b0;
            ext_wr_addr  <= 10'd0;
            ext_wr_data  <= 8'sd0;
            cnn_start    <= 1'b0;
            latched_pred <= 4'd0;
            has_result   <= 1'b0;
            bcd0 <= 4'd0; bcd1 <= 4'd0; bcd2 <= 4'd0;
            bcd3 <= 4'd0; bcd4 <= 4'd0; bcd5 <= 4'd0;
        end else begin
            ext_wr_en <= 1'b0;
            cnn_start <= 1'b0;

            case (b_state)
                B_INIT_LOAD: begin
                    active_img  <= sel_img;
                    has_result  <= 1'b0;
                    load_idx    <= 10'd0;
                    rom_rd_addr <= {sel_img, 10'd0};
                    b_state     <= B_ROM_ADDR;
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

                B_CLEAR_CANVAS: begin
                    ext_wr_en   <= 1'b1;
                    ext_wr_addr <= load_idx;
                    ext_wr_data <= 8'sd0;
                    if (load_idx == 10'd1023) begin
                        b_state <= B_START_CNN;
                    end else begin
                        load_idx <= load_idx + 10'd1;
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
                    if (key3_clear) begin
                        has_result <= 1'b0;
                        load_idx   <= 10'd0;
                        b_state    <= B_CLEAR_CANVAS;
                    end else if (sw_changed || key1_trig) begin
                        b_state <= B_INIT_LOAD;
                    end else if (key2_rand) begin
                        has_result  <= 1'b0;
                        active_img  <= rand_img;
                        load_idx    <= 10'd0;
                        rom_rd_addr <= {rand_img, 10'd0};
                        b_state     <= B_ROM_ADDR;
                    end
                end

                default: b_state <= B_INIT_LOAD;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // 4. Mirror Image RAM for DESim 160x120 Fast VGA + CNN Accelerator
    // -------------------------------------------------------------------------
    reg [7:0]  vga_img_ram [0:1023];
    reg [7:0]  scan_x = 8'd0;
    reg [6:0]  scan_y = 7'd0;

    initial begin
        $readmemh("mem/test_image.mem", vga_img_ram);
    end

    always @(posedge CLOCK_50) begin
        if (ext_wr_en) begin
            vga_img_ram[ext_wr_addr] <= ext_wr_data[7:0];
        end
    end

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
    // 5. Crisp 4x5 Mini Bitmap Font (fits cleanly on 160x120 DESim VGA!)
    // -------------------------------------------------------------------------
    function [3:0] mini_font_row;
        input [7:0] ch;
        input [2:0] ry; // 0..4
        begin
            case (ch)
                "0": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1010; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "1": case (ry) 0:mini_font_row=4'b0100; 1:mini_font_row=4'b1100; 2:mini_font_row=4'b0100; 3:mini_font_row=4'b0100; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "2": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b0010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1000; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "3": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b0010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b0010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "4": case (ry) 0:mini_font_row=4'b1010; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b0010; 4:mini_font_row=4'b0010; default:mini_font_row=4'b0000; endcase
                "5": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b0010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "6": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "7": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b0010; 2:mini_font_row=4'b0100; 3:mini_font_row=4'b0100; 4:mini_font_row=4'b0100; default:mini_font_row=4'b0000; endcase
                "8": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "9": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b0010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "A": case (ry) 0:mini_font_row=4'b0100; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1010; default:mini_font_row=4'b0000; endcase
                "B": case (ry) 0:mini_font_row=4'b1100; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1100; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1100; default:mini_font_row=4'b0000; endcase
                "C": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1000; 3:mini_font_row=4'b1000; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "D": case (ry) 0:mini_font_row=4'b1100; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1010; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1100; default:mini_font_row=4'b0000; endcase
                "E": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1100; 3:mini_font_row=4'b1000; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "F": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1100; 3:mini_font_row=4'b1000; 4:mini_font_row=4'b1000; default:mini_font_row=4'b0000; endcase
                "G": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1010; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "H": case (ry) 0:mini_font_row=4'b1010; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1010; default:mini_font_row=4'b0000; endcase
                "I": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b0100; 2:mini_font_row=4'b0100; 3:mini_font_row=4'b0100; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "K": case (ry) 0:mini_font_row=4'b1010; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1100; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1010; default:mini_font_row=4'b0000; endcase
                "L": case (ry) 0:mini_font_row=4'b1000; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1000; 3:mini_font_row=4'b1000; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "N": case (ry) 0:mini_font_row=4'b1010; 1:mini_font_row=4'b1110; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1010; default:mini_font_row=4'b0000; endcase
                "O": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1010; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "P": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1000; 4:mini_font_row=4'b1000; default:mini_font_row=4'b0000; endcase
                "R": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1100; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1010; default:mini_font_row=4'b0000; endcase
                "S": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b1000; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b0010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "T": case (ry) 0:mini_font_row=4'b1110; 1:mini_font_row=4'b0100; 2:mini_font_row=4'b0100; 3:mini_font_row=4'b0100; 4:mini_font_row=4'b0100; default:mini_font_row=4'b0000; endcase
                "U": case (ry) 0:mini_font_row=4'b1010; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1010; 3:mini_font_row=4'b1010; 4:mini_font_row=4'b1110; default:mini_font_row=4'b0000; endcase
                "V": case (ry) 0:mini_font_row=4'b1010; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1010; 3:mini_font_row=4'b0100; 4:mini_font_row=4'b0100; default:mini_font_row=4'b0000; endcase
                "W": case (ry) 0:mini_font_row=4'b1010; 1:mini_font_row=4'b1010; 2:mini_font_row=4'b1110; 3:mini_font_row=4'b1110; 4:mini_font_row=4'b1010; default:mini_font_row=4'b0000; endcase
                ":": case (ry) 1:mini_font_row=4'b0100; 3:mini_font_row=4'b0100; default:mini_font_row=4'b0000; endcase
                ".": case (ry) 4:mini_font_row=4'b0100; default:mini_font_row=4'b0000; endcase
                default: mini_font_row = 4'b0000;
            endcase
        end
    endfunction

    function [7:0] label_char;
        input [3:0] cls_id;
        input [3:0] pos;
        reg [63:0] s8;
        begin
            case (cls_id)
                4'd0: s8 = "0 TSHIRT";
                4'd1: s8 = "1 TROUSR";
                4'd2: s8 = "2 PULLVR";
                4'd3: s8 = "3 DRESS ";
                4'd4: s8 = "4 COAT  ";
                4'd5: s8 = "5 SANDAL";
                4'd6: s8 = "6 SHIRT ";
                4'd7: s8 = "7 SNEAKR";
                4'd8: s8 = "8 BAG   ";
                4'd9: s8 = "9 BOOT  ";
                default: s8 = "        ";
            endcase
            label_char = (pos < 4'd8) ? s8[(7 - pos) * 8 +: 8] : " ";
        end
    endfunction

    function [7:0] hdr_char;
        input [4:0] pos;
        reg [191:0] hs;
        begin
            hs = "DE10 FPGA CNN CLASSIFIER";
            hdr_char = (pos < 5'd24) ? hs[(23 - pos) * 8 +: 8] : " ";
        end
    endfunction

    function [7:0] bot_char;
        input [4:0] pos;
        input [3:0] win_cls;
        input       ready;
        reg [63:0]  p_win;
        reg [127:0] p_run;
        begin
            p_win = "WINNER: ";
            p_run = "CNN RUNNING...  ";
            if (!ready)
                bot_char = (pos < 5'd16) ? p_run[(15 - pos) * 8 +: 8] : " ";
            else if (pos < 5'd8)
                bot_char = p_win[(7 - pos) * 8 +: 8];
            else if (pos < 5'd16)
                bot_char = label_char(win_cls, pos - 5'd8);
            else
                bot_char = " ";
        end
    endfunction

    // -------------------------------------------------------------------------
    // 6. Fast 160x120 Virtual VGA Scanner (1 pixel per clock = 3.8s/frame!)
    // -------------------------------------------------------------------------
    function signed [7:0] get_score;
        input [3:0] idx;
        begin
            case (idx)
                4'd0: get_score = dbg_fc_scores_flat[7:0];
                4'd1: get_score = dbg_fc_scores_flat[15:8];
                4'd2: get_score = dbg_fc_scores_flat[23:16];
                4'd3: get_score = dbg_fc_scores_flat[31:24];
                4'd4: get_score = dbg_fc_scores_flat[39:32];
                4'd5: get_score = dbg_fc_scores_flat[47:40];
                4'd6: get_score = dbg_fc_scores_flat[55:48];
                4'd7: get_score = dbg_fc_scores_flat[63:56];
                4'd8: get_score = dbg_fc_scores_flat[71:64];
                4'd9: get_score = dbg_fc_scores_flat[79:72];
                default: get_score = 8'sd0;
            endcase
        end
    endfunction

    wire [4:0] img_gx = (scan_x >= 8'd8 && scan_x < 8'd72) ? ((scan_x - 8'd8) >> 1) : 5'd0;
    wire [4:0] img_gy = (scan_y >= 7'd24 && scan_y < 7'd88) ? ((scan_y - 7'd24) >> 1) : 5'd0;
    wire [7:0] pix_raw = vga_img_ram[{img_gy, img_gx}];
    wire [7:0] gray_u8 = pix_raw[7] ? 8'd0 : {pix_raw[6:0], 1'b0};

    // 10-row score chart on right (X = 76..155, Y = 15..104 -> 9 px/row)
    wire in_chart = (scan_x >= 8'd76 && scan_x < 8'd156) && (scan_y >= 7'd15 && scan_y < 7'd105);
    wire [6:0] rel_cy = scan_y - 7'd15;
    wire [3:0] chart_row = (rel_cy < 7'd9)  ? 4'd0 :
                           (rel_cy < 7'd18) ? 4'd1 :
                           (rel_cy < 7'd27) ? 4'd2 :
                           (rel_cy < 7'd36) ? 4'd3 :
                           (rel_cy < 7'd45) ? 4'd4 :
                           (rel_cy < 7'd54) ? 4'd5 :
                           (rel_cy < 7'd63) ? 4'd6 :
                           (rel_cy < 7'd72) ? 4'd7 :
                           (rel_cy < 7'd81) ? 4'd8 : 4'd9;
    wire [3:0] row_sub_y = rel_cy - (chart_row * 4'd9);
    wire signed [7:0] row_sc = get_score(chart_row);
    wire signed [9:0] sc_off = $signed({{2{row_sc[7]}}, row_sc}) + 10'sd48;
    wire [5:0] bar_len = (sc_off <= 10'sd2)  ? 6'd2 :
                         (sc_off >= 10'sd84) ? 6'd42 : sc_off[6:1];

    // Header text (X = 32..127, Y = 4..8, 24 chars x 4px pitch)
    wire in_hdr_txt = (scan_y >= 7'd4 && scan_y < 7'd9) && (scan_x >= 8'd32 && scan_x < 8'd128);
    wire [7:0] hdr_dx = scan_x - 8'd32;
    wire [3:0] hdr_bits = mini_font_row(hdr_char(hdr_dx[6:2]), scan_y - 7'd4);
    wire       hdr_pix  = in_hdr_txt && hdr_bits[3 - hdr_dx[1:0]];

    // Chart row label text (X = 76..107, row_sub_y = 2..6, 8 chars x 4px pitch)
    wire in_lbl_txt = in_chart && (row_sub_y >= 4'd2 && row_sub_y < 4'd7) && (scan_x >= 8'd76 && scan_x < 8'd108);
    wire [7:0] lbl_dx = scan_x - 8'd76;
    wire [3:0] lbl_bits = mini_font_row(label_char(chart_row, lbl_dx[4:2]), row_sub_y - 4'd2);
    wire       lbl_pix  = in_lbl_txt && lbl_bits[3 - lbl_dx[1:0]];

    // Bottom banner text (X = 48..111, Y = 111..115, 16 chars x 4px pitch)
    wire in_bot_txt = (scan_y >= 7'd111 && scan_y < 7'd116) && (scan_x >= 8'd48 && scan_x < 8'd112);
    wire [7:0] bot_dx = scan_x - 8'd48;
    wire [3:0] bot_bits = mini_font_row(bot_char(bot_dx[5:2], latched_pred, has_result), scan_y - 7'd111);
    wire       bot_pix  = in_bot_txt && bot_bits[3 - bot_dx[1:0]];

    always @(posedge CLOCK_50) begin
        if (rst) begin
            scan_x    <= 8'd0;
            scan_y    <= 7'd0;
            VGA_X     <= 10'd0;
            VGA_Y     <= 9'd0;
            VGA_COLOR <= 24'h0F172A;
            plot      <= 1'b0;
        end else begin
            VGA_X <= {2'b00, scan_x};
            VGA_Y <= {2'b00, scan_y};
            plot  <= 1'b1;

            // Default dark slate background
            VGA_COLOR <= 24'h0F172A;

            // Top blue header bar with white text
            if (scan_y >= 7'd2 && scan_y < 7'd11) begin
                VGA_COLOR <= hdr_pix ? 24'hFFFFFF : 24'h1E3A8A;
            end
            // Left 32x32 image zoomed 2x (64x64) with cyan border
            else if (scan_x >= 8'd6 && scan_x < 8'd74 && scan_y >= 7'd22 && scan_y < 7'd90) begin
                if (scan_x >= 8'd8 && scan_x < 8'd72 && scan_y >= 7'd24 && scan_y < 7'd88)
                    VGA_COLOR <= {gray_u8, gray_u8, gray_u8};
                else
                    VGA_COLOR <= 24'h38BDF8;
            end
            // Right 10-class text labels + score bars
            else if (in_chart) begin
                if (lbl_pix) begin
                    VGA_COLOR <= (has_result && chart_row == latched_pred) ? 24'h34D399 : 24'hE2E8F0;
                end else if (row_sub_y >= 4'd2 && row_sub_y < 4'd7 && scan_x >= 8'd110 && scan_x < 8'd152) begin
                    if ((scan_x - 8'd110) < {2'b00, bar_len})
                        VGA_COLOR <= (has_result && chart_row == latched_pred) ? 24'h10B981 : 24'h38BDF8;
                    else
                        VGA_COLOR <= 24'h1E293B;
                end
            end
            // Bottom status bar with "WINNER: <CLASS>"
            else if (scan_y >= 7'd109 && scan_y < 7'd118 && scan_x >= 8'd6 && scan_x < 8'd154) begin
                if (bot_pix)
                    VGA_COLOR <= 24'hFFFFFF;
                else
                    VGA_COLOR <= has_result ? ((latched_pred == active_img) ? 24'h059669 : 24'hDC2626) : 24'h334155;
            end

            // Advance 160x120 raster
            if (scan_x == 8'd159) begin
                scan_x <= 8'd0;
                if (scan_y == 7'd119) scan_y <= 7'd0;
                else                  scan_y <= scan_y + 7'd1;
            end else begin
                scan_x <= scan_x + 8'd1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // 7. LEDR & Seven-Segment Displays (HEX5..HEX0)
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
        SEG_R    = 7'b0101111;

    wire is_pass = (latched_pred == active_img);

    assign HEX5 = SW[9] ? seg7_digit(bcd5) : seg7_digit(active_img);
    assign HEX4 = SW[9] ? seg7_digit(bcd4) : SEG_DASH;
    assign HEX3 = SW[9] ? seg7_digit(bcd3) : (is_pass ? SEG_P : SEG_E);
    assign HEX2 = SW[9] ? seg7_digit(bcd2) : (is_pass ? SEG_A : SEG_R);
    assign HEX1 = SW[9] ? seg7_digit(bcd1) : (is_pass ? SEG_S : SEG_R);
    assign HEX0 = SW[9] ? seg7_digit(bcd0) : seg7_digit(latched_pred);

endmodule
