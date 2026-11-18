// =============================================================================
// Module: conv_engine.v
// Description:
//   Reusable sequential 3x3 VALID 2D convolution + INT32 bias + dyadic
//   requantization engine for the Cyclone V CNN accelerator.
//
//   Time-multiplexed between:
//     - Conv1: IN = 1 x 32 x 32, OUT = 8 x 30 x 30  (9 MACs/output,  64,800 MACs total)
//     - Conv2: IN = 8 x 15 x 15, OUT = 16 x 13 x 13 (72 MACs/output, 194,688 MACs total)
//
//   Arithmetic (matches `docs/quantization_spec.md` and `sw/refmodel/int8_ref_model.py`):
//     1. Signed INT8 input x Signed INT8 weight -> INT32 accumulation + INT32 bias
//     2. Dyadic fixed-point requantization:
//            prod_64  = int64(acc_reg) * int64(cfg_mult)
//            rnd_64   = 64'sd1 <<< (cfg_shift - 1)
//            scaled   = (prod_64 + rnd_64) >>> cfg_shift
//            req_int8 = clamp(scaled, -128, 127)
//     3. Emits both `out_acc_int32` and `out_req_int8` with a valid/ready
//        handshake (`out_valid` && `out_ready`) so downstream `relu.v` and
//        feature-map BRAMs can capture every output element cleanly.
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module conv_engine #(
    parameter IN_ADDR_W  = 11,  // max(1024, 1800) -> 11 bits
    parameter W_ADDR_W   = 11,  // max(72, 1152)   -> 11 bits
    parameter B_ADDR_W   = 4,   // max(8, 16)      -> 4 bits
    parameter OUT_ADDR_W = 13   // max(7200, 2704) -> 13 bits
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,
    output reg                      busy,
    output reg                      done,

    // Runtime layer configuration
    input  wire [4:0]               cfg_in_c,    // 1 (Conv1) or 8 (Conv2)
    input  wire [5:0]               cfg_in_h,    // 32 (Conv1) or 15 (Conv2)
    input  wire [5:0]               cfg_in_w,    // 32 (Conv1) or 15 (Conv2)
    input  wire [4:0]               cfg_out_c,   // 8 (Conv1) or 16 (Conv2)
    input  wire [5:0]               cfg_out_h,   // 30 (Conv1) or 13 (Conv2)
    input  wire [5:0]               cfg_out_w,   // 30 (Conv1) or 13 (Conv2)
    input  wire [15:0]              cfg_mult,    // 23202 (Conv1) or 16868 (Conv2)
    input  wire [5:0]               cfg_shift,   // 19 (Conv1) or 20 (Conv2)

    // Synchronous 1-cycle read port to input feature map BRAM
    output reg  [IN_ADDR_W-1:0]     in_rd_addr,
    input  wire signed [7:0]        in_rd_data,

    // Synchronous 1-cycle read port to weight ROM
    output reg  [W_ADDR_W-1:0]      w_rd_addr,
    input  wire signed [7:0]        w_rd_data,

    // Synchronous 1-cycle read port to bias ROM
    output reg  [B_ADDR_W-1:0]      b_rd_addr,
    input  wire signed [31:0]       b_rd_data,

    // Valid/ready output interface
    output reg                      out_valid,
    input  wire                     out_ready,
    output reg  [OUT_ADDR_W-1:0]    out_addr,
    output reg  [4:0]               out_ch,
    output reg  [5:0]               out_y,
    output reg  [5:0]               out_x,
    output reg  signed [31:0]       out_acc_int32,
    output reg  signed [7:0]        out_req_int8
);

    localparam [2:0]
        S_IDLE    = 3'd0,
        S_PRE     = 3'd1,  // Issue read addr 0 for (ic=0, ky=0, kx=0) and bias[oc]
        S_PIPE    = 3'd2,  // Wait 1 cycle for synchronous BRAM/ROM; issue read addr 1
        S_MAC     = 3'd3,  // 1 MAC/cycle across all cfg_in_c * 9 kernel elements
        S_REQUANT = 3'd4,  // Dyadic multiply-shift-round-saturate & handshake out_valid
        S_DONE    = 3'd5;

    reg [2:0]            state;

    // Output position counters
    reg [4:0]            oc_idx;
    reg [5:0]            oy_idx;
    reg [5:0]            ox_idx;
    reg [OUT_ADDR_W-1:0] out_addr_cnt;

    // Kernel request counters (ic, ky, kx) and flat counters
    reg [4:0]            req_ic;
    reg [1:0]            req_ky;
    reg [1:0]            req_kx;
    reg [7:0]            req_cnt;     // 0 .. total_macs
    reg [7:0]            mac_cnt;     // 0 .. total_macs - 1
    reg [W_ADDR_W-1:0]   w_base_oc;   // oc * (cfg_in_c * 9)
    reg [W_ADDR_W-1:0]   w_req_ptr;   // w_base_oc + req_cnt

    reg signed [31:0]    acc_reg;

    // Total MACs per output pixel = cfg_in_c * 9 = (cfg_in_c << 3) + cfg_in_c
    wire [7:0] total_macs = {cfg_in_c, 3'b000} + {3'b000, cfg_in_c};

    // Current input read address for (req_ic, oy_idx + req_ky, ox_idx + req_kx):
    //   addr = ((req_ic * cfg_in_h + (oy_idx + req_ky)) * cfg_in_w) + (ox_idx + req_kx)
    wire [5:0]           in_y_cur   = oy_idx + {4'd0, req_ky};
    wire [5:0]           in_x_cur   = ox_idx + {4'd0, req_kx};
    wire [10:0]          row_idx    = (req_ic * cfg_in_h) + {5'd0, in_y_cur};
    wire [IN_ADDR_W-1:0] cur_in_addr = (row_idx * cfg_in_w) + {5'd0, in_x_cur};

    // Signed 8x8 -> 16-bit product sign-extended to 32 bits
    wire signed [15:0] prod_16 = $signed(in_rd_data) * $signed(w_rd_data);
    wire signed [31:0] prod_32 = {{16{prod_16[15]}}, prod_16};

    // 64-bit dyadic rounding and saturation to signed INT8 [-128, 127]
    wire signed [63:0] mult_64   = {48'd0, cfg_mult};
    wire signed [63:0] prod_64   = $signed(acc_reg) * mult_64;
    wire signed [63:0] rnd_64    = 64'sd1 <<< (cfg_shift - 6'd1);
    wire signed [63:0] scaled_64 = (prod_64 + rnd_64) >>> cfg_shift;

    wire signed [7:0]  sat_int8  = (scaled_64 < -64'sd128) ? -8'sd128 :
                                   (scaled_64 >  64'sd127) ?  8'sd127 :
                                   scaled_64[7:0];

    // Helper task to advance (req_ic, req_ky, req_kx) in (ic -> ky -> kx) order
    task advance_kernel_coords;
        begin
            if (req_kx == 2'd2) begin
                req_kx <= 2'd0;
                if (req_ky == 2'd2) begin
                    req_ky <= 2'd0;
                    req_ic <= req_ic + 1'b1;
                end else begin
                    req_ky <= req_ky + 1'b1;
                end
            end else begin
                req_kx <= req_kx + 1'b1;
            end
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            state         <= S_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            in_rd_addr    <= {IN_ADDR_W{1'b0}};
            w_rd_addr     <= {W_ADDR_W{1'b0}};
            b_rd_addr     <= {B_ADDR_W{1'b0}};
            out_valid     <= 1'b0;
            out_addr      <= {OUT_ADDR_W{1'b0}};
            out_ch        <= 5'd0;
            out_y         <= 6'd0;
            out_x         <= 6'd0;
            out_acc_int32 <= 32'sd0;
            out_req_int8  <= 8'sd0;
            oc_idx        <= 5'd0;
            oy_idx        <= 6'd0;
            ox_idx        <= 6'd0;
            out_addr_cnt  <= {OUT_ADDR_W{1'b0}};
            req_ic        <= 5'd0;
            req_ky        <= 2'd0;
            req_kx        <= 2'd0;
            req_cnt       <= 8'd0;
            mac_cnt       <= 8'd0;
            w_base_oc     <= {W_ADDR_W{1'b0}};
            w_req_ptr     <= {W_ADDR_W{1'b0}};
            acc_reg       <= 32'sd0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    out_valid <= 1'b0;
                    busy      <= 1'b0;
                    if (start) begin
                        busy         <= 1'b1;
                        oc_idx       <= 5'd0;
                        oy_idx       <= 6'd0;
                        ox_idx       <= 6'd0;
                        out_addr_cnt <= {OUT_ADDR_W{1'b0}};
                        w_base_oc    <= {W_ADDR_W{1'b0}};
                        req_ic       <= 5'd0;
                        req_ky       <= 2'd0;
                        req_kx       <= 2'd0;
                        state        <= S_PRE;
                    end
                end

                S_PRE: begin
                    out_valid  <= 1'b0;
                    // Request element 0: (req_ic=0, req_ky=0, req_kx=0) and bias[oc]
                    b_rd_addr  <= oc_idx[B_ADDR_W-1:0];
                    in_rd_addr <= cur_in_addr;
                    w_rd_addr  <= w_base_oc;
                    w_req_ptr  <= w_base_oc + 1'b1;
                    req_cnt    <= 8'd1;
                    mac_cnt    <= 8'd0;
                    req_ic     <= 5'd0;
                    req_ky     <= 2'd0;
                    req_kx     <= 2'd1; // next coordinate is (0, 0, 1)
                    state      <= S_PIPE;
                end

                S_PIPE: begin
                    // Synchronous BRAM/ROM latches element 0 on this edge; request element 1
                    in_rd_addr <= cur_in_addr;
                    w_rd_addr  <= w_req_ptr;
                    w_req_ptr  <= w_req_ptr + 1'b1;
                    req_cnt    <= 8'd2;
                    advance_kernel_coords();
                    state      <= S_MAC;
                end

                S_MAC: begin
                    // Accumulate element `mac_cnt` returned from synchronous BRAM/ROM
                    if (mac_cnt == 8'd0) begin
                        acc_reg <= b_rd_data + prod_32;
                    end else begin
                        acc_reg <= acc_reg + prod_32;
                    end

                    // Issue read address for element `req_cnt` if remaining
                    if (req_cnt < total_macs) begin
                        in_rd_addr <= cur_in_addr;
                        w_rd_addr  <= w_req_ptr;
                        w_req_ptr  <= w_req_ptr + 1'b1;
                        req_cnt    <= req_cnt + 1'b1;
                        advance_kernel_coords();
                    end

                    if (mac_cnt + 1 == total_macs) begin
                        state <= S_REQUANT;
                    end else begin
                        mac_cnt <= mac_cnt + 1'b1;
                    end
                end

                S_REQUANT: begin
                    out_valid     <= 1'b1;
                    out_addr      <= out_addr_cnt;
                    out_ch        <= oc_idx;
                    out_y         <= oy_idx;
                    out_x         <= ox_idx;
                    out_acc_int32 <= acc_reg;
                    out_req_int8  <= sat_int8;

                    if (out_ready) begin
                        out_addr_cnt <= out_addr_cnt + 1'b1;
                        req_ic       <= 5'd0;
                        req_ky       <= 2'd0;
                        req_kx       <= 2'd0;

                        if (ox_idx + 1 == cfg_out_w) begin
                            ox_idx <= 6'd0;
                            if (oy_idx + 1 == cfg_out_h) begin
                                oy_idx <= 6'd0;
                                if (oc_idx + 1 == cfg_out_c) begin
                                    state <= S_DONE;
                                end else begin
                                    oc_idx    <= oc_idx + 1'b1;
                                    w_base_oc <= w_base_oc + total_macs;
                                    state     <= S_PRE;
                                end
                            end else begin
                                oy_idx <= oy_idx + 1'b1;
                                state  <= S_PRE;
                            end
                        end else begin
                            ox_idx <= ox_idx + 1'b1;
                            state  <= S_PRE;
                        end
                    end
                end

                S_DONE: begin
                    out_valid <= 1'b0;
                    busy      <= 1'b0;
                    done      <= 1'b1;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
