// =============================================================================
// Module: gap.v
// Description:
//   Global Average Pooling (GAP) over 13 x 13 = 169 spatial positions for
//   16 channels, with fixed-point dyadic multiply-shift rounding and INT8
//   saturation:
//       sum_c    = sum_{idx=0..168} int32(in_data[c, idx])
//       prod_64  = int64(sum_c) * int64(cfg_mult)
//       rnd_64   = 64'sd1 <<< (cfg_shift - 1)
//       scaled   = (prod_64 + rnd_64) >>> cfg_shift
//       out_data = clamp(scaled, 0, 127)
//
//   Default parameters match Milestone 4/5/6:
//       CHANNELS = 16, SPATIAL_N = 169, DEFAULT_MULT = 16780, DEFAULT_SHIFT = 19.
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module gap #(
    parameter CHANNELS      = 16,
    parameter SPATIAL_N     = 169,   // 13 * 13
    parameter IN_ADDR_W     = 12,    // ceil(log2(16 * 169 = 2704)) = 12
    parameter CH_W          = 4,
    parameter DEFAULT_MULT  = 16780,
    parameter DEFAULT_SHIFT = 19
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  start,
    output reg                   busy,
    output reg                   done,

    // Dyadic scaling parameters (from quantization_params.mem indices 4 & 5)
    input  wire [15:0]           cfg_mult,
    input  wire [5:0]            cfg_shift,

    // Synchronous 1-cycle read port to Conv2+ReLU2 feature map BRAM (2704 x 8b)
    output reg  [IN_ADDR_W-1:0]  in_rd_addr,
    input  wire signed [7:0]     in_rd_data,

    // Output port (1-cycle valid pulse per channel c = 0..15)
    output reg                   out_valid,
    output reg  [CH_W-1:0]       out_ch,
    output reg  signed [31:0]    out_sum_int32,
    output reg  signed [7:0]     out_data
);

    localparam [2:0]
        S_IDLE    = 3'd0,
        S_PRE     = 3'd1,  // Issue read address 0 for current channel
        S_PIPE    = 3'd2,  // Wait 1 cycle for synchronous BRAM to latch data 0; issue address 1
        S_ACCUM   = 3'd3,  // Accumulate elements 0..168 & issue addresses 2..168
        S_REQUANT = 3'd4,  // Dyadic multiply-shift-round-saturate & emit out_valid
        S_DONE    = 3'd5;

    reg [2:0]          state;
    reg [CH_W-1:0]     ch_idx;
    reg [7:0]          sp_idx;      // 0..168
    reg [IN_ADDR_W-1:0] rd_ptr;
    reg signed [31:0]  sum_acc;

    // 64-bit dyadic rounding arithmetic for S_REQUANT
    wire [15:0]        eff_mult  = (cfg_mult == 16'd0) ? DEFAULT_MULT[15:0] : cfg_mult;
    wire [5:0]         eff_shift = (cfg_shift == 6'd0) ? DEFAULT_SHIFT[5:0] : cfg_shift;
    wire signed [63:0] mult_64   = {48'd0, eff_mult};
    wire signed [63:0] prod_64   = $signed(sum_acc) * mult_64;
    wire signed [63:0] rnd_64    = 64'sd1 <<< (eff_shift - 6'd1);
    wire signed [63:0] scaled_64 = (prod_64 + rnd_64) >>> eff_shift;

    wire signed [7:0]  sat_uint7 = (scaled_64 < 64'sd0)   ? 8'sd0 :
                                   (scaled_64 > 64'sd127) ? 8'sd127 :
                                   scaled_64[7:0];

    always @(posedge clk) begin
        if (rst) begin
            state         <= S_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            in_rd_addr    <= {IN_ADDR_W{1'b0}};
            out_valid     <= 1'b0;
            out_ch        <= {CH_W{1'b0}};
            out_sum_int32 <= 32'sd0;
            out_data      <= 8'sd0;
            ch_idx        <= {CH_W{1'b0}};
            sp_idx        <= 8'd0;
            rd_ptr        <= {IN_ADDR_W{1'b0}};
            sum_acc       <= 32'sd0;
        end else begin
            out_valid <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy   <= 1'b1;
                        ch_idx <= {CH_W{1'b0}};
                        rd_ptr <= {IN_ADDR_W{1'b0}};
                        state  <= S_PRE;
                    end
                end

                S_PRE: begin
                    // Present address for spatial element 0 of current channel
                    in_rd_addr <= rd_ptr;
                    rd_ptr     <= rd_ptr + 1'b1;
                    sum_acc    <= 32'sd0;
                    state      <= S_PIPE;
                end

                S_PIPE: begin
                    // BRAM latches element 0 on this edge; present address for element 1
                    in_rd_addr <= rd_ptr;
                    rd_ptr     <= rd_ptr + 1'b1;
                    sp_idx     <= 8'd1;
                    state      <= S_ACCUM;
                end

                S_ACCUM: begin
                    // in_rd_data now holds element (sp_idx - 1)
                    sum_acc <= sum_acc + {{24{in_rd_data[7]}}, in_rd_data};
                    if (sp_idx < SPATIAL_N - 1) begin
                        in_rd_addr <= rd_ptr;
                        rd_ptr     <= rd_ptr + 1'b1;
                    end
                    if (sp_idx < SPATIAL_N) begin
                        sp_idx <= sp_idx + 1'b1;
                    end else begin
                        state <= S_REQUANT;
                    end
                end

                S_REQUANT: begin
                    out_valid     <= 1'b1;
                    out_ch        <= ch_idx;
                    out_sum_int32 <= sum_acc;
                    out_data      <= sat_uint7;

                    if (ch_idx + 1 == CHANNELS) begin
                        state <= S_DONE;
                    end else begin
                        ch_idx <= ch_idx + 1'b1;
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
