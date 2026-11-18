// =============================================================================
// Module: fully_connected.v
// Description:
//   16-input -> 10-output Fully Connected (FC) layer with signed INT8 weights
//   and inputs, INT32 bias addition and accumulation, and dyadic fixed-point
//   requantization to signed INT8 class scores in [-128, 127]:
//       acc[oc]   = bias[oc] + sum_{ic=0..15} int32(in[ic]) * int32(weight[oc, ic])
//       prod_64   = int64(acc[oc]) * int64(cfg_mult)
//       rnd_64    = 64'sd1 <<< (cfg_shift - 1)
//       scaled    = (prod_64 + rnd_64) >>> cfg_shift
//       score[oc] = clamp(scaled, -128, 127)
//
//   Weight memory layout (matches `docs/memory_map.md`):
//       w_addr(oc, ic) = (oc << 4) | ic   (depth = 160, width = 8b)
//   Bias memory layout:
//       b_addr(oc)     = oc               (depth = 10,  width = 32b)
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module fully_connected #(
    parameter IN_FEATURES   = 16,
    parameter OUT_CLASSES   = 10,
    parameter DEFAULT_MULT  = 23793,
    parameter DEFAULT_SHIFT = 21
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    output reg                  busy,
    output reg                  done,

    // Dyadic requantization parameters (from quantization_params.mem indices 6 & 7)
    input  wire [15:0]          cfg_mult,
    input  wire [5:0]           cfg_shift,

    // Synchronous 1-cycle read port to 16-element GAP output buffer
    output reg  [3:0]           in_rd_addr,
    input  wire signed [7:0]    in_rd_data,

    // Synchronous 1-cycle read port to FC weight ROM (160 x 8b)
    output reg  [7:0]           w_rd_addr,
    input  wire signed [7:0]    w_rd_data,

    // Synchronous 1-cycle read port to FC bias ROM (10 x 32b)
    output reg  [3:0]           b_rd_addr,
    input  wire signed [31:0]   b_rd_data,

    // Output port (1-cycle valid pulse per class oc = 0..9)
    output reg                  out_valid,
    output reg  [3:0]           out_idx,
    output reg  signed [31:0]   out_acc_int32,
    output reg  signed [7:0]    out_data
);

    localparam [2:0]
        S_IDLE    = 3'd0,
        S_PRE     = 3'd1,  // Issue read addr for ic = 0 and bias[oc]
        S_PIPE    = 3'd2,  // Wait 1 cycle for synchronous BRAM/ROM; issue addr for ic = 1
        S_MAC     = 3'd3,  // Accumulate ic = 0..15 & issue addresses for ic = 2..15
        S_REQUANT = 3'd4,  // Dyadic multiply-shift-round-saturate & emit out_valid
        S_DONE    = 3'd5;

    reg [2:0]        state;
    reg [3:0]        oc_idx;
    reg [4:0]        req_ic;
    reg [4:0]        mac_ic;
    reg signed [31:0] acc_reg;

    // Signed 8x8 -> 16-bit product sign-extended to 32 bits
    wire signed [15:0] prod_16 = $signed(in_rd_data) * $signed(w_rd_data);
    wire signed [31:0] prod_32 = {{16{prod_16[15]}}, prod_16};

    // 64-bit dyadic rounding and saturation to signed INT8 [-128, 127]
    wire [15:0]        eff_mult  = (cfg_mult == 16'd0) ? DEFAULT_MULT[15:0] : cfg_mult;
    wire [5:0]         eff_shift = (cfg_shift == 6'd0) ? DEFAULT_SHIFT[5:0] : cfg_shift;
    wire signed [63:0] mult_64   = {48'd0, eff_mult};
    wire signed [63:0] prod_64   = $signed(acc_reg) * mult_64;
    wire signed [63:0] rnd_64    = 64'sd1 <<< (eff_shift - 6'd1);
    wire signed [63:0] scaled_64 = (prod_64 + rnd_64) >>> eff_shift;

    wire signed [7:0]  sat_int8  = (scaled_64 < -64'sd128) ? -8'sd128 :
                                   (scaled_64 >  64'sd127) ?  8'sd127 :
                                   scaled_64[7:0];

    always @(posedge clk) begin
        if (rst) begin
            state         <= S_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            in_rd_addr    <= 4'd0;
            w_rd_addr     <= 8'd0;
            b_rd_addr     <= 4'd0;
            out_valid     <= 1'b0;
            out_idx       <= 4'd0;
            out_acc_int32 <= 32'sd0;
            out_data      <= 8'sd0;
            oc_idx        <= 4'd0;
            req_ic        <= 5'd0;
            mac_ic        <= 5'd0;
            acc_reg       <= 32'sd0;
        end else begin
            out_valid <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy   <= 1'b1;
                        oc_idx <= 4'd0;
                        state  <= S_PRE;
                    end
                end

                S_PRE: begin
                    // Request bias[oc] and ic = 0
                    b_rd_addr  <= oc_idx;
                    in_rd_addr <= 4'd0;
                    w_rd_addr  <= {oc_idx, 4'd0};
                    req_ic     <= 5'd1;
                    mac_ic     <= 5'd0;
                    state      <= S_PIPE;
                end

                S_PIPE: begin
                    // Synchronous ROM/RAM latches ic = 0 on this edge; request ic = 1
                    in_rd_addr <= req_ic[3:0];
                    w_rd_addr  <= {oc_idx, req_ic[3:0]};
                    req_ic     <= 5'd2;
                    state      <= S_MAC;
                end

                S_MAC: begin
                    // in_rd_data & w_rd_data now hold feature `mac_ic`
                    if (mac_ic == 5'd0) begin
                        acc_reg <= b_rd_data + prod_32;
                    end else begin
                        acc_reg <= acc_reg + prod_32;
                    end

                    if (req_ic < IN_FEATURES) begin
                        in_rd_addr <= req_ic[3:0];
                        w_rd_addr  <= {oc_idx, req_ic[3:0]};
                        req_ic     <= req_ic + 1'b1;
                    end

                    if (mac_ic + 1 == IN_FEATURES) begin
                        state <= S_REQUANT;
                    end else begin
                        mac_ic <= mac_ic + 1'b1;
                    end
                end

                S_REQUANT: begin
                    out_valid     <= 1'b1;
                    out_idx       <= oc_idx;
                    out_acc_int32 <= acc_reg;
                    out_data      <= sat_int8;

                    if (oc_idx + 1 == OUT_CLASSES) begin
                        state <= S_DONE;
                    end else begin
                        oc_idx <= oc_idx + 1'b1;
                        state  <= S_PRE;
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
