// =============================================================================
// Module: maxpool.v
// Description:
//   2x2 Max Pooling with stride 2 and signed INT8 comparison.
//   Reads a 3D feature map of shape (CHANNELS, IN_H, IN_W) from a synchronous
//   1-cycle-latency BRAM (`in_rd_addr` -> `in_rd_data`) in CHW layout:
//       in_addr  = (c * IN_H + y) * IN_W + x
//   and writes the pooled feature map of shape (CHANNELS, OUT_H, OUT_W) where
//   OUT_H = IN_H / 2, OUT_W = IN_W / 2:
//       out_addr = (c * OUT_H + py) * OUT_W + px
//
//   Default parameters match Pool1: 8 channels, 30x30 input -> 8 channels, 15x15 output.
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module maxpool #(
    parameter CHANNELS   = 8,
    parameter IN_H       = 30,
    parameter IN_W       = 30,
    parameter OUT_H      = 15,
    parameter OUT_W      = 15,
    parameter IN_ADDR_W  = 13,  // ceil(log2(8*30*30 = 7200)) = 13
    parameter OUT_ADDR_W = 11   // ceil(log2(8*15*15 = 1800)) = 11
) (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    output reg                       busy,
    output reg                       done,

    // Synchronous 1-cycle read port to input feature map BRAM
    output reg  [IN_ADDR_W-1:0]      in_rd_addr,
    input  wire signed [7:0]         in_rd_data,

    // Streaming output port (1-cycle valid strobe per 2x2 pooled output)
    output reg                       out_valid,
    output reg  [OUT_ADDR_W-1:0]     out_addr,
    output reg  signed [7:0]         out_data
);

    // FSM states
    localparam [2:0]
        S_IDLE = 3'd0,
        S_REQ0 = 3'd1,  // Issue addr for (2*py,   2*px)
        S_REQ1 = 3'd2,  // Issue addr for (2*py,   2*px+1); wait for v00
        S_REQ2 = 3'd3,  // Issue addr for (2*py+1, 2*px);   capture v00
        S_REQ3 = 3'd4,  // Issue addr for (2*py+1, 2*px+1); cmp v01 with max
        S_WAIT = 3'd5,  // cmp v10 with max; wait for v11
        S_EMIT = 3'd6,  // cmp v11 with max; emit out_valid & advance (c, py, px)
        S_DONE = 3'd7;

    reg [2:0] state;
    reg [3:0] ch_idx;
    reg [4:0] py_idx;
    reg [4:0] px_idx;
    reg [OUT_ADDR_W-1:0] out_addr_cnt;
    reg signed [7:0] cur_max;

    wire [5:0] y0 = {py_idx, 1'b0};          // 2 * py_idx
    wire [5:0] y1 = {py_idx, 1'b1};          // 2 * py_idx + 1
    wire [5:0] x0 = {px_idx, 1'b0};          // 2 * px_idx
    wire [5:0] x1 = {px_idx, 1'b1};          // 2 * px_idx + 1

    wire [IN_ADDR_W-1:0] row0_base = (ch_idx * IN_H + y0) * IN_W;
    wire [IN_ADDR_W-1:0] row1_base = (ch_idx * IN_H + y1) * IN_W;

    wire signed [7:0] final_max = ($signed(in_rd_data) > $signed(cur_max)) ? in_rd_data : cur_max;

    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            in_rd_addr   <= {IN_ADDR_W{1'b0}};
            out_valid    <= 1'b0;
            out_addr     <= {OUT_ADDR_W{1'b0}};
            out_data     <= 8'sd0;
            ch_idx       <= 4'd0;
            py_idx       <= 5'd0;
            px_idx       <= 5'd0;
            out_addr_cnt <= {OUT_ADDR_W{1'b0}};
            cur_max      <= -8'sd128;
        end else begin
            out_valid <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy         <= 1'b1;
                        ch_idx       <= 4'd0;
                        py_idx       <= 5'd0;
                        px_idx       <= 5'd0;
                        out_addr_cnt <= {OUT_ADDR_W{1'b0}};
                        state        <= S_REQ0;
                    end
                end

                S_REQ0: begin
                    in_rd_addr <= row0_base + x0;
                    state      <= S_REQ1;
                end

                S_REQ1: begin
                    in_rd_addr <= row0_base + x1;
                    state      <= S_REQ2;
                end

                S_REQ2: begin
                    // in_rd_data now holds v00 = feat[c, 2*py, 2*px]
                    cur_max    <= in_rd_data;
                    in_rd_addr <= row1_base + x0;
                    state      <= S_REQ3;
                end

                S_REQ3: begin
                    // in_rd_data now holds v01 = feat[c, 2*py, 2*px+1]
                    if ($signed(in_rd_data) > $signed(cur_max))
                        cur_max <= in_rd_data;
                    in_rd_addr <= row1_base + x1;
                    state      <= S_WAIT;
                end

                S_WAIT: begin
                    // in_rd_data now holds v10 = feat[c, 2*py+1, 2*px]
                    if ($signed(in_rd_data) > $signed(cur_max))
                        cur_max <= in_rd_data;
                    state <= S_EMIT;
                end

                S_EMIT: begin
                    // in_rd_data now holds v11 = feat[c, 2*py+1, 2*px+1]
                    out_valid    <= 1'b1;
                    out_addr     <= out_addr_cnt;
                    out_data     <= final_max;
                    out_addr_cnt <= out_addr_cnt + 1'b1;

                    if (px_idx + 1 == OUT_W) begin
                        px_idx <= 5'd0;
                        if (py_idx + 1 == OUT_H) begin
                            py_idx <= 5'd0;
                            if (ch_idx + 1 == CHANNELS) begin
                                state <= S_DONE;
                            end else begin
                                ch_idx <= ch_idx + 1'b1;
                                state  <= S_REQ0;
                            end
                        end else begin
                            py_idx <= py_idx + 1'b1;
                            state  <= S_REQ0;
                        end
                    end else begin
                        px_idx <= px_idx + 1'b1;
                        state  <= S_REQ0;
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
