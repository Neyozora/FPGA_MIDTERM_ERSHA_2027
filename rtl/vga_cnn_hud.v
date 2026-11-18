// =============================================================================
// Module: vga_cnn_hud.v
// Description:
//   Synthesizable 640x480 @ 60 Hz VGA Controller & Live CNN Dashboard HUD
//   for the Terasic DE10-Standard (ADV7123 24-bit Video DAC).
//
//   Renders in real-time hardware:
//     1. Top Header Banner: "DE10-STANDARD FPGA CNN CLASSIFIER"
//     2. Left Panel (256x256, 8x zoom): Live 32x32 INT8 input image from BRAM
//     3. Right Panel: 10-Class Horizontal Bar Chart driven by the live FPGA
//        `fc_score_ram[0..9]` INT8 scores, highlighting the winning class in
//        bright Emerald Green!
//     4. Bottom Banner: Large 2x text showing "WINNER: CLASS # <NAME>"
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module vga_cnn_hud (
    input  wire        clk_50,
    input  wire        rst,

    // Read port to 1024x8b mirror of the current 32x32 input image
    output wire [9:0]  vga_img_rd_addr,
    input  wire [7:0]  vga_img_rd_data,

    // Live CNN status & 10 hardware INT8 class scores
    input  wire [3:0]  predicted_class,
    input  wire [3:0]  active_img_class,
    input  wire        pc_mode_active,
    input  wire [79:0] dbg_fc_scores_flat,

    // PS/2 Mouse Cursor & Button States (0..255 inside 256x256 canvas)
    input  wire [7:0]  mouse_x,
    input  wire [7:0]  mouse_y,
    input  wire        mouse_left,
    input  wire        mouse_right,

    // ADV7123 VGA DAC pins
    output wire        VGA_CLK,
    output reg         VGA_HS,
    output reg         VGA_VS,
    output reg         VGA_BLANK_N,
    output wire        VGA_SYNC_N,
    output reg  [7:0]  VGA_R,
    output reg  [7:0]  VGA_G,
    output reg  [7:0]  VGA_B
);

    // -------------------------------------------------------------------------
    // 1. 25 MHz Pixel Clock Divider (from 50 MHz CLOCK_50)
    // -------------------------------------------------------------------------
    reg pix_clk;
    always @(posedge clk_50) begin
        if (rst) pix_clk <= 1'b0;
        else     pix_clk <= ~pix_clk;
    end

    assign VGA_CLK    = pix_clk;
    assign VGA_SYNC_N = 1'b0;

    // -------------------------------------------------------------------------
    // 2. 640x480 @ 60 Hz Timing Counters (800 x 525 total)
    // -------------------------------------------------------------------------
    localparam [9:0]
        H_ACTIVE = 10'd640,
        H_FP     = 10'd16,
        H_SYNC   = 10'd96,
        H_BP     = 10'd48,
        H_TOTAL  = 10'd800,

        V_ACTIVE = 10'd480,
        V_FP     = 10'd10,
        V_SYNC   = 10'd2,
        V_BP     = 10'd33,
        V_TOTAL  = 10'd525;

    reg [9:0] h_cnt;
    reg [9:0] v_cnt;

    always @(posedge clk_50) begin
        if (rst) begin
            h_cnt <= 10'd0;
            v_cnt <= 10'd0;
        end else if (pix_clk) begin
            if (h_cnt == H_TOTAL - 10'd1) begin
                h_cnt <= 10'd0;
                if (v_cnt == V_TOTAL - 10'd1)
                    v_cnt <= 10'd0;
                else
                    v_cnt <= v_cnt + 10'd1;
            end else begin
                h_cnt <= h_cnt + 10'd1;
            end
        end
    end

    // Prefetch 32x32 image pixel (X = 32..287, Y = 96..351 -> 256x256, 8x zoom)
    wire [9:0] next_h = (h_cnt == H_TOTAL - 10'd1) ? 10'd0 : (h_cnt + 10'd1);
    wire [4:0] img_x  = (next_h >= 10'd32 && next_h < 10'd288) ? ((next_h - 10'd32) >> 3) : 5'd0;
    wire [4:0] img_y  = (v_cnt  >= 10'd96 && v_cnt  < 10'd352) ? ((v_cnt  - 10'd96) >> 3) : 5'd0;
    assign vga_img_rd_addr = {img_y, img_x};

    // -------------------------------------------------------------------------
    // 3. Compact 8x8 Uppercase & Digit Bitmap Font Function
    // -------------------------------------------------------------------------
    function [7:0] font_row;
        input [7:0] ch;
        input [2:0] ry;
        begin
            case (ch)
                "0": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h6E; 3:font_row=8'h76; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "1": case (ry) 0:font_row=8'h18; 1:font_row=8'h38; 2:font_row=8'h18; 3:font_row=8'h18; 4:font_row=8'h18; 5:font_row=8'h18; 6:font_row=8'h7E; default:font_row=8'h00; endcase
                "2": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h06; 3:font_row=8'h1C; 4:font_row=8'h30; 5:font_row=8'h66; 6:font_row=8'h7E; default:font_row=8'h00; endcase
                "3": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h06; 3:font_row=8'h1C; 4:font_row=8'h06; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "4": case (ry) 0:font_row=8'h0C; 1:font_row=8'h1C; 2:font_row=8'h3C; 3:font_row=8'h6C; 4:font_row=8'h7E; 5:font_row=8'h0C; 6:font_row=8'h0C; default:font_row=8'h00; endcase
                "5": case (ry) 0:font_row=8'h7E; 1:font_row=8'h60; 2:font_row=8'h7C; 3:font_row=8'h06; 4:font_row=8'h06; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "6": case (ry) 0:font_row=8'h3C; 1:font_row=8'h60; 2:font_row=8'h7C; 3:font_row=8'h66; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "7": case (ry) 0:font_row=8'h7E; 1:font_row=8'h66; 2:font_row=8'h06; 3:font_row=8'h0C; 4:font_row=8'h18; 5:font_row=8'h18; 6:font_row=8'h18; default:font_row=8'h00; endcase
                "8": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h3C; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "9": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h3E; 4:font_row=8'h06; 5:font_row=8'h06; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "A": case (ry) 0:font_row=8'h18; 1:font_row=8'h3C; 2:font_row=8'h66; 3:font_row=8'h7E; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h66; default:font_row=8'h00; endcase
                "B": case (ry) 0:font_row=8'h7C; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h7C; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h7C; default:font_row=8'h00; endcase
                "C": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h60; 3:font_row=8'h60; 4:font_row=8'h60; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "D": case (ry) 0:font_row=8'h78; 1:font_row=8'h6C; 2:font_row=8'h66; 3:font_row=8'h66; 4:font_row=8'h66; 5:font_row=8'h6C; 6:font_row=8'h78; default:font_row=8'h00; endcase
                "E": case (ry) 0:font_row=8'h7E; 1:font_row=8'h60; 2:font_row=8'h60; 3:font_row=8'h78; 4:font_row=8'h60; 5:font_row=8'h60; 6:font_row=8'h7E; default:font_row=8'h00; endcase
                "F": case (ry) 0:font_row=8'h7E; 1:font_row=8'h60; 2:font_row=8'h60; 3:font_row=8'h78; 4:font_row=8'h60; 5:font_row=8'h60; 6:font_row=8'h60; default:font_row=8'h00; endcase
                "G": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h60; 3:font_row=8'h6E; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "H": case (ry) 0:font_row=8'h66; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h7E; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h66; default:font_row=8'h00; endcase
                "I": case (ry) 0:font_row=8'h3C; 1:font_row=8'h18; 2:font_row=8'h18; 3:font_row=8'h18; 4:font_row=8'h18; 5:font_row=8'h18; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "K": case (ry) 0:font_row=8'h66; 1:font_row=8'h6C; 2:font_row=8'h78; 3:font_row=8'h70; 4:font_row=8'h78; 5:font_row=8'h6C; 6:font_row=8'h66; default:font_row=8'h00; endcase
                "L": case (ry) 0:font_row=8'h60; 1:font_row=8'h60; 2:font_row=8'h60; 3:font_row=8'h60; 4:font_row=8'h60; 5:font_row=8'h60; 6:font_row=8'h7E; default:font_row=8'h00; endcase
                "M": case (ry) 0:font_row=8'h63; 1:font_row=8'h77; 2:font_row=8'h7F; 3:font_row=8'h6B; 4:font_row=8'h63; 5:font_row=8'h63; 6:font_row=8'h63; default:font_row=8'h00; endcase
                "N": case (ry) 0:font_row=8'h66; 1:font_row=8'h76; 2:font_row=8'h7E; 3:font_row=8'h7E; 4:font_row=8'h6E; 5:font_row=8'h66; 6:font_row=8'h66; default:font_row=8'h00; endcase
                "O": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h66; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "P": case (ry) 0:font_row=8'h7C; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h7C; 4:font_row=8'h60; 5:font_row=8'h60; 6:font_row=8'h60; default:font_row=8'h00; endcase
                "R": case (ry) 0:font_row=8'h7C; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h7C; 4:font_row=8'h6C; 5:font_row=8'h66; 6:font_row=8'h66; default:font_row=8'h00; endcase
                "S": case (ry) 0:font_row=8'h3C; 1:font_row=8'h66; 2:font_row=8'h60; 3:font_row=8'h3C; 4:font_row=8'h06; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "T": case (ry) 0:font_row=8'h7E; 1:font_row=8'h18; 2:font_row=8'h18; 3:font_row=8'h18; 4:font_row=8'h18; 5:font_row=8'h18; 6:font_row=8'h18; default:font_row=8'h00; endcase
                "U": case (ry) 0:font_row=8'h66; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h66; 4:font_row=8'h66; 5:font_row=8'h66; 6:font_row=8'h3C; default:font_row=8'h00; endcase
                "V": case (ry) 0:font_row=8'h66; 1:font_row=8'h66; 2:font_row=8'h66; 3:font_row=8'h66; 4:font_row=8'h66; 5:font_row=8'h3C; 6:font_row=8'h18; default:font_row=8'h00; endcase
                "W": case (ry) 0:font_row=8'h63; 1:font_row=8'h63; 2:font_row=8'h63; 3:font_row=8'h6B; 4:font_row=8'h7F; 5:font_row=8'h77; 6:font_row=8'h63; default:font_row=8'h00; endcase
                "X": case (ry) 0:font_row=8'h66; 1:font_row=8'h66; 2:font_row=8'h3C; 3:font_row=8'h18; 4:font_row=8'h3C; 5:font_row=8'h66; 6:font_row=8'h66; default:font_row=8'h00; endcase
                "-": case (ry) 3:font_row=8'h7E; default:font_row=8'h00; endcase
                ":": case (ry) 2:font_row=8'h18; 5:font_row=8'h18; default:font_row=8'h00; endcase
                default: font_row = 8'h00;
            endcase
        end
    endfunction

    // 10-character class label lookup ("0 T-SHIRT ", "1 TROUSER ", etc.)
    function [7:0] class_label_char;
        input [3:0] cls_id;
        input [3:0] char_pos;
        reg [79:0] str10;
        begin
            case (cls_id)
                4'd0: str10 = "0 T-SHIRT ";
                4'd1: str10 = "1 TROUSER ";
                4'd2: str10 = "2 PULLOVER";
                4'd3: str10 = "3 DRESS   ";
                4'd4: str10 = "4 COAT    ";
                4'd5: str10 = "5 SANDAL  ";
                4'd6: str10 = "6 SHIRT   ";
                4'd7: str10 = "7 SNEAKER ";
                4'd8: str10 = "8 BAG     ";
                4'd9: str10 = "9 BOOT    ";
                default: str10 = "          ";
            endcase
            if (char_pos < 4'd10)
                class_label_char = str10[(9 - char_pos) * 8 +: 8];
            else
                class_label_char = " ";
        end
    endfunction

    // Top banner text (32 chars): "DE10-STANDARD FPGA CNN CLASSIFIER"
    function [7:0] header_char;
        input [4:0] pos;
        reg [255:0] hstr;
        begin
            hstr = "DE10-STANDARD FPGA CNN ACCEL    ";
            header_char = hstr[(31 - pos) * 8 +: 8];
        end
    endfunction

    // Bottom winner banner (22 chars): "WINNER: " + class_label_char
    function [7:0] winner_char;
        input [4:0] pos;
        input [3:0] win_cls;
        reg [63:0] prefix;
        begin
            prefix = "WINNER: ";
            if (pos < 5'd8)
                winner_char = prefix[(7 - pos) * 8 +: 8];
            else if (pos < 5'd18)
                winner_char = class_label_char(win_cls, pos - 5'd8);
            else
                winner_char = " ";
        end
    endfunction

    // Extract signed INT8 score for a given row (0..9)
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

    // -------------------------------------------------------------------------
    // 4. Pixel Rendering Pipeline (Registered @ 25 MHz pix_clk)
    // -------------------------------------------------------------------------
    wire active_area = (h_cnt < H_ACTIVE) && (v_cnt < V_ACTIVE);
    wire hs_pulse    = (h_cnt >= H_ACTIVE + H_FP) && (h_cnt < H_ACTIVE + H_FP + H_SYNC);
    wire vs_pulse    = (v_cnt >= V_ACTIVE + V_FP) && (v_cnt < V_ACTIVE + V_FP + V_SYNC);

    // Chart row calculation (Y = 88 .. 367 -> 10 rows of 28 pixels each)
    wire in_chart_y  = (v_cnt >= 10'd88) && (v_cnt < 10'd368);
    wire [8:0] chart_rel_y = v_cnt - 10'd88;
    wire [3:0] chart_row   = (chart_rel_y < 9'd28)  ? 4'd0 :
                             (chart_rel_y < 9'd56)  ? 4'd1 :
                             (chart_rel_y < 9'd84)  ? 4'd2 :
                             (chart_rel_y < 9'd112) ? 4'd3 :
                             (chart_rel_y < 9'd140) ? 4'd4 :
                             (chart_rel_y < 9'd168) ? 4'd5 :
                             (chart_rel_y < 9'd196) ? 4'd6 :
                             (chart_rel_y < 9'd224) ? 4'd7 :
                             (chart_rel_y < 9'd252) ? 4'd8 : 4'd9;
    wire [4:0] row_sub_y   = chart_rel_y - (chart_row * 5'd28);

    wire signed [7:0] row_score = get_score(chart_row);
    wire signed [9:0] shifted_sc = $signed({{2{row_score[7]}}, row_score}) + 10'sd48;
    wire [7:0] bar_width = (shifted_sc <= 10'sd2)  ? 8'd6 :
                           (shifted_sc >= 10'sd80) ? 8'd160 :
                           (shifted_sc[6:0] << 1);

    // Font pixel checks
    // A. Top Header (X = 64..575, Y = 16..31, 2x scale -> 32 chars x 16px)
    wire in_hdr = (v_cnt >= 10'd16 && v_cnt < 10'd32) && (h_cnt >= 10'd64 && h_cnt < 10'd576);
    wire [4:0] hdr_cpos = (h_cnt - 10'd64) >> 4;
    wire [2:0] hdr_cx   = ((h_cnt - 10'd64) >> 1) & 3'd7;
    wire [2:0] hdr_cy   = (v_cnt - 10'd16) >> 1;
    wire [7:0] hdr_bits = font_row(header_char(hdr_cpos), hdr_cy);
    wire       hdr_pix  = in_hdr && hdr_bits[3'd7 - hdr_cx];

    // B. Chart Row Label (X = 336..415, Y = row_sub_y 6..13, 1x scale -> 10 chars x 8px)
    wire in_lbl = in_chart_y && (row_sub_y >= 5'd6 && row_sub_y < 5'd14) && (h_cnt >= 10'd336 && h_cnt < 10'd416);
    wire [3:0] lbl_cpos = (h_cnt - 10'd336) >> 3;
    wire [2:0] lbl_cx   = (h_cnt - 10'd336) & 3'd7;
    wire [2:0] lbl_cy   = row_sub_y - 5'd6;
    wire [7:0] lbl_bits = font_row(class_label_char(chart_row, lbl_cpos), lbl_cy);
    wire       lbl_pix  = in_lbl && lbl_bits[3'd7 - lbl_cx];

    // C. Bottom Winner Banner (X = 144..431, Y = 412..427, 2x scale -> 18 chars x 16px)
    wire in_win = (v_cnt >= 10'd412 && v_cnt < 10'd428) && (h_cnt >= 10'd144 && h_cnt < 10'd432);
    wire [4:0] win_cpos = (h_cnt - 10'd144) >> 4;
    wire [2:0] win_cx   = ((h_cnt - 10'd144) >> 1) & 3'd7;
    wire [2:0] win_cy   = (v_cnt - 10'd412) >> 1;
    wire [7:0] win_bits = font_row(winner_char(win_cpos, predicted_class), win_cy);
    wire       win_pix  = in_win && win_bits[3'd7 - win_cx];

    // Greyscale value of 32x32 input pixel
    wire [7:0] gray_u8 = vga_img_rd_data[7] ? 8'd0 : {vga_img_rd_data[6:0], 1'b0};

    always @(posedge clk_50) begin
        if (rst) begin
            VGA_HS      <= 1'b1;
            VGA_VS      <= 1'b1;
            VGA_BLANK_N <= 1'b0;
            VGA_R       <= 8'd0;
            VGA_G       <= 8'd0;
            VGA_B       <= 8'd0;
        end else if (pix_clk) begin
            VGA_HS      <= ~hs_pulse;
            VGA_VS      <= ~vs_pulse;
            VGA_BLANK_N <= active_area;

            if (!active_area) begin
                VGA_R <= 8'd0;
                VGA_G <= 8'd0;
                VGA_B <= 8'd0;
            end else begin
                // Default dark slate background (#0F172A)
                VGA_R <= 8'd15;
                VGA_G <= 8'd23;
                VGA_B <= 8'd42;

                // 1. Top Header Bar (Y = 8..40)
                if (v_cnt >= 10'd8 && v_cnt < 10'd40) begin
                    VGA_R <= 8'd30;
                    VGA_G <= 8'd58;
                    VGA_B <= 8'd138;
                    if (hdr_pix) begin
                        VGA_R <= 8'd255;
                        VGA_G <= 8'd255;
                        VGA_B <= 8'd255;
                    end
                end

                // 2. Left 32x32 Input Image Frame (X = 28..291, Y = 92..355)
                else if (h_cnt >= 10'd28 && h_cnt < 10'd292 && v_cnt >= 10'd92 && v_cnt < 10'd356) begin
                    if (h_cnt >= 10'd32 && h_cnt < 10'd288 && v_cnt >= 10'd96 && v_cnt < 10'd352) begin
                        VGA_R <= gray_u8;
                        VGA_G <= gray_u8;
                        VGA_B <= gray_u8;

                        // Highlight active 8x8 brush cell outline + crosshair cursor
                        if ((((h_cnt - 10'd32) >> 3) == {5'd0, mouse_x[7:3]}) &&
                            (((v_cnt - 10'd96) >> 3) == {5'd0, mouse_y[7:3]}) &&
                            (((h_cnt - 10'd32) & 10'd7) == 10'd0 || ((h_cnt - 10'd32) & 10'd7) == 10'd7 ||
                             ((v_cnt - 10'd96) & 10'd7) == 10'd0 || ((v_cnt - 10'd96) & 10'd7) == 10'd7)) begin
                            if (mouse_left) begin
                                VGA_R <= 8'd16;  VGA_G <= 8'd255; VGA_B <= 8'd129; // Green draw cursor
                            end else if (mouse_right) begin
                                VGA_R <= 8'd255; VGA_G <= 8'd120; VGA_B <= 8'd30;  // Orange erase cursor
                            end else begin
                                VGA_R <= 8'd250; VGA_G <= 8'd204; VGA_B <= 8'd21;  // Yellow hover box
                            end
                        end
                        if (((h_cnt - 10'd32) == {2'b00, mouse_x} &&
                             (v_cnt - 10'd96) >= ({2'b00, mouse_y} - 10'd3) &&
                             (v_cnt - 10'd96) <= ({2'b00, mouse_y} + 10'd3)) ||
                            ((v_cnt - 10'd96) == {2'b00, mouse_y} &&
                             (h_cnt - 10'd32) >= ({2'b00, mouse_x} - 10'd3) &&
                             (h_cnt - 10'd32) <= ({2'b00, mouse_x} + 10'd3))) begin
                            VGA_R <= 8'd239;
                            VGA_G <= 8'd68;
                            VGA_B <= 8'd68;
                        end
                    end else begin
                        // Glowing cyan border around 32x32 image
                        VGA_R <= 8'd56;
                        VGA_G <= 8'd189;
                        VGA_B <= 8'd248;
                    end
                end

                // 3. Right 10-Class Bar Chart (X = 328..608, Y = 88..368)
                else if (in_chart_y && h_cnt >= 10'd328 && h_cnt < 10'd608) begin
                    if (chart_row == predicted_class) begin
                        // Highlight row background for Winner
                        VGA_R <= 8'd6;
                        VGA_G <= 8'd78;
                        VGA_B <= 8'd59;
                    end else begin
                        VGA_R <= 8'd30;
                        VGA_G <= 8'd41;
                        VGA_B <= 8'd59;
                    end

                    // Class name text
                    if (lbl_pix) begin
                        if (chart_row == predicted_class) begin
                            VGA_R <= 8'd167;
                            VGA_G <= 8'd243;
                            VGA_B <= 8'd208;
                        end else begin
                            VGA_R <= 8'd226;
                            VGA_G <= 8'd232;
                            VGA_B <= 8'd240;
                        end
                    end

                    // Horizontal Score Bar (X = 432 .. 592)
                    if (row_sub_y >= 5'd5 && row_sub_y < 5'd17 && h_cnt >= 10'd432 && h_cnt < 10'd592) begin
                        if ((h_cnt - 10'd432) < {2'd0, bar_width}) begin
                            if (chart_row == predicted_class) begin
                                // Bright Emerald Green Winner Bar (#10B981)
                                VGA_R <= 8'd16;
                                VGA_G <= 8'd185;
                                VGA_B <= 8'd129;
                            end else begin
                                // Steel Cyan Bar (#38BDF8)
                                VGA_R <= 8'd56;
                                VGA_G <= 8'd189;
                                VGA_B <= 8'd248;
                            end
                        end else begin
                            // Dark bar track
                            VGA_R <= 8'd15;
                            VGA_G <= 8'd23;
                            VGA_B <= 8'd42;
                        end
                    end
                end

                // 4. Bottom Winner Banner (Y = 400..440, X = 32..608)
                else if (v_cnt >= 10'd400 && v_cnt < 10'd440 && h_cnt >= 10'd32 && h_cnt < 10'd608) begin
                    VGA_R <= 8'd6;
                    VGA_G <= 8'd95;
                    VGA_B <= 8'd70;
                    if (win_pix) begin
                        VGA_R <= 8'd255;
                        VGA_G <= 8'd255;
                        VGA_B <= 8'd255;
                    end
                end
            end
        end
    end

endmodule
