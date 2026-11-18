// =============================================================================
// Module: maxpool_m2.v
// Description:
//   4-Bank Parallel 2x2 Stride-2 MaxPooling Engine for Method 2.
//   Reads 4 channels simultaneously from fmap_a[0..3] and writes max-pooled
//   INT8 results to fmap_b[0..3].
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module maxpool_m2 (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire        pool_id,    // 0: Pool1 (16ch, 32x32->16x16), 1: Pool2 (32ch, 16x16->8x8)
    output reg         done,

    // Read port (1-cycle synchronous read latency from 4 banks)
    output reg  [11:0] rd_addr,
    input  wire [7:0]  rd_data0,
    input  wire [7:0]  rd_data1,
    input  wire [7:0]  rd_data2,
    input  wire [7:0]  rd_data3,

    // Write port (to 4 banks)
    output reg         wr_en,
    output reg  [11:0] wr_addr,
    output reg  [7:0]  wr_data0,
    output reg  [7:0]  wr_data1,
    output reg  [7:0]  wr_data2,
    output reg  [7:0]  wr_data3
);

    localparam [2:0]
        ST_IDLE = 3'd0,
        ST_RD0  = 3'd1,
        ST_RD1  = 3'd2,
        ST_RD2  = 3'd3,
        ST_RD3  = 3'd4,
        ST_WAIT = 3'd5,
        ST_DONE = 3'd6;

    reg [2:0] state;
    reg [3:0] og;       // 0..3 (Pool1) or 0..7 (Pool2)
    reg [3:0] r_out, c_out;

    wire [3:0] og_max   = pool_id ? 4'd7 : 4'd3;
    wire [3:0] out_hw_m = pool_id ? 4'd7 : 4'd15;

    // Address helpers
    wire [11:0] in_hw2  = pool_id ? 12'd256 : 12'd1024;
    wire [11:0] out_hw2 = pool_id ? 12'd64  : 12'd256;
    wire [4:0]  r2      = {r_out, 1'b0};
    wire [4:0]  c2      = {c_out, 1'b0};
    wire [11:0] base_in = (og * in_hw2) + (pool_id ? ({r2, 4'd0} + c2) : ({r2, 5'd0} + c2));
    wire [11:0] in_row  = pool_id ? 12'd16 : 12'd32;

    reg signed [7:0] m0, m1, m2, m3;

    function signed [7:0] smax;
        input signed [7:0] a;
        input signed [7:0] b;
        begin
            smax = (a > b) ? a : b;
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state    <= ST_IDLE;
            done     <= 1'b0;
            wr_en    <= 1'b0;
            rd_addr  <= 12'd0;
            wr_addr  <= 12'd0;
            wr_data0 <= 8'd0;
            wr_data1 <= 8'd0;
            wr_data2 <= 8'd0;
            wr_data3 <= 8'd0;
            og       <= 4'd0;
            r_out    <= 4'd0;
            c_out    <= 4'd0;
        end else begin
            done  <= 1'b0;
            wr_en <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        og    <= 4'd0;
                        r_out <= 4'd0;
                        c_out <= 4'd0;
                        state <= ST_RD0;
                    end
                end

                ST_RD0: begin
                    rd_addr <= base_in;
                    state   <= ST_RD1;
                end

                ST_RD1: begin
                    rd_addr <= base_in + 12'd1;
                    state   <= ST_RD2;
                end

                ST_RD2: begin
                    // rd_data from RD0 is valid now
                    m0 <= $signed(rd_data0);
                    m1 <= $signed(rd_data1);
                    m2 <= $signed(rd_data2);
                    m3 <= $signed(rd_data3);
                    rd_addr <= base_in + in_row;
                    state   <= ST_RD3;
                end

                ST_RD3: begin
                    // rd_data from RD1 is valid now
                    m0 <= smax(m0, $signed(rd_data0));
                    m1 <= smax(m1, $signed(rd_data1));
                    m2 <= smax(m2, $signed(rd_data2));
                    m3 <= smax(m3, $signed(rd_data3));
                    rd_addr <= base_in + in_row + 12'd1;
                    state   <= ST_WAIT;
                end

                ST_WAIT: begin
                    // rd_data from RD2 is valid now; RD3 arrives next cycle
                    m0 <= smax(m0, $signed(rd_data0));
                    m1 <= smax(m1, $signed(rd_data1));
                    m2 <= smax(m2, $signed(rd_data2));
                    m3 <= smax(m3, $signed(rd_data3));
                    state <= 3'd7;
                end

                3'd7: begin
                    // rd_data from RD3 is valid now -> write max!
                    wr_en    <= 1'b1;
                    wr_addr  <= (og * out_hw2) + (pool_id ? ({r_out, 3'd0} + c_out) : ({r_out, 4'd0} + c_out));
                    wr_data0 <= smax(m0, $signed(rd_data0));
                    wr_data1 <= smax(m1, $signed(rd_data1));
                    wr_data2 <= smax(m2, $signed(rd_data2));
                    wr_data3 <= smax(m3, $signed(rd_data3));

                    if (c_out == out_hw_m) begin
                        c_out <= 4'd0;
                        if (r_out == out_hw_m) begin
                            r_out <= 4'd0;
                            if (og == og_max) begin
                                state <= ST_DONE;
                            end else begin
                                og    <= og + 4'd1;
                                state <= ST_RD0;
                            end
                        end else begin
                            r_out <= r_out + 4'd1;
                            state <= ST_RD0;
                        end
                    end else begin
                        c_out <= c_out + 4'd1;
                        state <= ST_RD0;
                    end
                end

                ST_DONE: begin
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
