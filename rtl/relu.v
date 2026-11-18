// =============================================================================
// Module: relu.v
// Description:
//   Signed INT8 Rectified Linear Unit (ReLU) for the Cyclone V CNN accelerator.
//   Replaces negative signed INT8 values (in_data < 0, i.e. sign bit in_data[7] == 1)
//   with 8'sd0 and preserves non-negative values in [0, 127].
//
//   Provides both a zero-latency combinational output (`out_data_comb`) and a
//   1-cycle registered pipeline output (`out_valid`, `out_addr`, `out_data`).
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module relu #(
    parameter DATA_W = 8,
    parameter ADDR_W = 13
) (
    input  wire                     clk,
    input  wire                     rst,

    // Streaming input interface
    input  wire                     in_valid,
    input  wire [ADDR_W-1:0]        in_addr,
    input  wire signed [DATA_W-1:0] in_data,

    // Combinational output (available in the same cycle as in_data)
    output wire signed [DATA_W-1:0] out_data_comb,

    // Registered output (1-cycle latency)
    output reg                      out_valid,
    output reg  [ADDR_W-1:0]        out_addr,
    output reg  signed [DATA_W-1:0] out_data
);

    // Sign bit check: in two's-complement signed INT8, in_data[DATA_W-1] == 1'b1
    // iff in_data < 0.
    assign out_data_comb = in_data[DATA_W-1] ? {DATA_W{1'b0}} : in_data;

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_addr  <= {ADDR_W{1'b0}};
            out_data  <= {DATA_W{1'b0}};
        end else begin
            out_valid <= in_valid;
            out_addr  <= in_addr;
            out_data  <= out_data_comb;
        end
    end

endmodule
