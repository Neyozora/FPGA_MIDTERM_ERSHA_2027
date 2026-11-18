// =============================================================================
// Module: argmax.v
// Description:
//   Signed INT8 Argmax across 10 output class scores (classes 0..9).
//   Reads each class score sequentially from a 1-cycle-latency score buffer/array
//   (`in_rd_addr` -> `in_rd_data`) and uses strict `>` comparison:
//       if ($signed(in_rd_data) > $signed(max_score))
//   so ties always break to the lowest class index (matching Python `np.argmax`).
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module argmax #(
    parameter NUM_CLASSES = 10,
    parameter IDX_W       = 4
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    output reg                  busy,
    output reg                  done,

    // Read interface to 10-element signed INT8 score buffer
    output reg  [IDX_W-1:0]     in_rd_addr,
    input  wire signed [7:0]    in_rd_data,

    // Final classification result
    output reg  [IDX_W-1:0]     predicted_class,
    output reg  signed [7:0]    max_score
);

    localparam [1:0]
        S_IDLE = 2'd0,
        S_PRE  = 2'd1,
        S_SCAN = 2'd2,
        S_DONE = 2'd3;

    reg [1:0]       state;
    reg [IDX_W-1:0] req_idx;
    reg [IDX_W-1:0] cmp_idx;

    always @(posedge clk) begin
        if (rst) begin
            state           <= S_IDLE;
            busy            <= 1'b0;
            done            <= 1'b0;
            in_rd_addr      <= {IDX_W{1'b0}};
            predicted_class <= {IDX_W{1'b0}};
            max_score       <= -8'sd128;
            req_idx         <= {IDX_W{1'b0}};
            cmp_idx         <= {IDX_W{1'b0}};
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy       <= 1'b1;
                        in_rd_addr <= {IDX_W{1'b0}}; // request score[0]
                        req_idx    <= 4'd1;
                        cmp_idx    <= 4'd0;
                        state      <= S_PRE;
                    end
                end

                S_PRE: begin
                    // BRAM/reg-file latches score[0] on this edge; request score[1]
                    in_rd_addr <= req_idx;
                    req_idx    <= req_idx + 1'b1;
                    state      <= S_SCAN;
                end

                S_SCAN: begin
                    // Issue next read address if any remain
                    if (req_idx < NUM_CLASSES) begin
                        in_rd_addr <= req_idx;
                        req_idx    <= req_idx + 1'b1;
                    end

                    // Compare data returned from previous cycle's address (`cmp_idx`)
                    if (cmp_idx == 4'd0) begin
                        max_score       <= in_rd_data;
                        predicted_class <= 4'd0;
                    end else if ($signed(in_rd_data) > $signed(max_score)) begin
                        max_score       <= in_rd_data;
                        predicted_class <= cmp_idx;
                    end

                    if (cmp_idx + 1 == NUM_CLASSES) begin
                        state <= S_DONE;
                    end else begin
                        cmp_idx <= cmp_idx + 1'b1;
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
