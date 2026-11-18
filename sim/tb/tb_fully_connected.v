// =============================================================================
// Testbench: tb_fully_connected.v
// Description:
//   Self-checking ModelSim unit testbench for `fully_connected.v`:
//   Loads `sim/vectors/img0_gap_out.mem` (16 inputs), `mem/fc_weights.mem`
//   (160 weights), and `mem/fc_bias.mem` (10 biases), runs `fully_connected.v`
//   (`M_fc = 23793, n_fc = 21`), and compares all 10 `out_acc_int32` values
//   against `sim/vectors/img0_fc_acc.mem` AND all 10 `out_data` (`INT8`) class
//   scores against `sim/vectors/img0_fc_scores.mem`.
// =============================================================================
`timescale 1ns / 1ps

module tb_fully_connected;

    reg                clk;
    reg                rst;
    reg                start;
    wire               busy;
    wire               done;

    wire [3:0]         in_rd_addr;
    reg  signed [7:0]  in_rd_data;
    wire [7:0]         w_rd_addr;
    reg  signed [7:0]  w_rd_data;
    wire [3:0]         b_rd_addr;
    reg  signed [31:0] b_rd_data;

    wire               out_valid;
    wire [3:0]         out_idx;
    wire signed [31:0] out_acc_int32;
    wire signed [7:0]  out_data;

    reg signed [7:0]   gap_mem        [0:15];
    reg signed [7:0]   fc_w_mem       [0:159];
    reg signed [31:0]  fc_b_mem       [0:9];
    reg signed [31:0]  gold_acc_mem   [0:9];
    reg signed [7:0]   gold_score_mem [0:9];

    always @(posedge clk) begin
        in_rd_data <= gap_mem[in_rd_addr];
        w_rd_data  <= fc_w_mem[w_rd_addr];
        b_rd_data  <= fc_b_mem[b_rd_addr];
    end

    fully_connected #(
        .IN_FEATURES   (16),
        .OUT_CLASSES   (10),
        .DEFAULT_MULT  (23793),
        .DEFAULT_SHIFT (21)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .start         (start),
        .busy          (busy),
        .done          (done),
        .cfg_mult      (16'd23793),
        .cfg_shift     (6'd21),
        .in_rd_addr    (in_rd_addr),
        .in_rd_data    (in_rd_data),
        .w_rd_addr     (w_rd_addr),
        .w_rd_data     (w_rd_data),
        .b_rd_addr     (b_rd_addr),
        .b_rd_data     (b_rd_data),
        .out_valid     (out_valid),
        .out_idx       (out_idx),
        .out_acc_int32 (out_acc_int32),
        .out_data      (out_data)
    );

    initial clk = 1'b0;
    always #10 clk = ~clk;

    integer errors;
    integer max_abs_err;
    integer diff;
    integer out_cnt;
    integer cycles;

    initial begin
        $readmemh("sim/vectors/img0_gap_out.mem",   gap_mem);
        $readmemh("mem/fc_weights.mem",             fc_w_mem);
        $readmemh("mem/fc_bias.mem",                fc_b_mem);
        $readmemh("sim/vectors/img0_fc_acc.mem",    gold_acc_mem);
        $readmemh("sim/vectors/img0_fc_scores.mem", gold_score_mem);

        rst         = 1'b1;
        start       = 1'b0;
        errors      = 0;
        max_abs_err = 0;
        out_cnt     = 0;
        cycles      = 0;

        @(posedge clk);
        @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        while (!done) begin
            @(posedge clk);
            cycles = cycles + 1;
            #1;
            if (out_valid) begin
                out_cnt = out_cnt + 1;
                diff = (out_data >= gold_score_mem[out_idx]) ?
                       (out_data - gold_score_mem[out_idx]) :
                       (gold_score_mem[out_idx] - out_data);
                if (diff > max_abs_err) max_abs_err = diff;

                $display("[TB_FC] class=%1d | acc_exp=%6d acc_got=%6d | score_exp=%4d score_got=%4d",
                         out_idx, gold_acc_mem[out_idx], out_acc_int32, gold_score_mem[out_idx], out_data);

                if (out_acc_int32 !== gold_acc_mem[out_idx] || out_data !== gold_score_mem[out_idx]) begin
                    $display("[FAIL] tb_fully_connected class=%0d mismatch!", out_idx);
                    errors = errors + 1;
                end
            end
        end

        if (out_cnt != 10) begin
            $display("[FAIL] tb_fully_connected expected 10 outputs, got %0d", out_cnt);
            errors = errors + 1;
        end

        $display("=================================================================");
        $display("[TB_FC] Classes Checked: %0d | Cycles: %0d | Max Abs Error: %0d | Mismatches: %0d",
                 out_cnt, cycles, max_abs_err, errors);
        if (errors == 0)
            $display("[TB_FC] STATUS: PASS");
        else
            $display("[TB_FC] STATUS: FAIL (%0d errors)", errors);
        $display("=================================================================");
        $finish;
    end

endmodule
