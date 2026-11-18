// =============================================================================
// Testbench: tb_argmax.v
// Description:
//   Self-checking ModelSim unit testbench for `argmax.v`:
//     1. Golden vector verification on `sim/vectors/img0_fc_scores.mem`
//        (`[7, -28, -5, -2, -19, -25, -2, -36, -7, -14]`) -> verifies
//        `predicted_class == 0` (matching `sim/vectors/img0_expected_class.mem`)
//        and `max_score == 7`.
//     2. Tie-breaking test (`[-10, 25, 25, 10, -128, 25, 0, 1, 2, 3]`) ->
//        verifies lowest index `1` wins.
//     3. All-negative score vector (`[-120, -95, -80, -45, -12, -88, -30, -15, -12, -100]`) ->
//        verifies signed comparison selects class `4` (`max_score == -12`).
//     4. Highest-index winner (`[-128, -100, -50, 0, 10, 20, 30, 40, 50, 127]`) ->
//        verifies class `9` (`max_score == 127`).
// =============================================================================
`timescale 1ns / 1ps

module tb_argmax;

    reg                clk;
    reg                rst;
    reg                start;
    wire               busy;
    wire               done;

    wire [3:0]         in_rd_addr;
    reg  signed [7:0]  in_rd_data;
    wire [3:0]         predicted_class;
    wire signed [7:0]  max_score;

    reg signed [7:0]   score_mem [0:9];
    reg signed [7:0]   gold_cls  [0:0];

    always @(posedge clk) begin
        in_rd_data <= score_mem[in_rd_addr];
    end

    argmax #(
        .NUM_CLASSES (10),
        .IDX_W       (4)
    ) dut (
        .clk             (clk),
        .rst             (rst),
        .start           (start),
        .busy            (busy),
        .done            (done),
        .in_rd_addr      (in_rd_addr),
        .in_rd_data      (in_rd_data),
        .predicted_class (predicted_class),
        .max_score       (max_score)
    );

    initial clk = 1'b0;
    always #10 clk = ~clk;

    integer errors;
    integer cycles;

    task run_one_argmax_case;
        input [3:0]        exp_cls;
        input signed [7:0] exp_max;
        input [8*24-1:0]   label;
        integer c;
        begin
            c = 0;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            while (!done) begin
                @(posedge clk);
                c = c + 1;
            end
            #1;
            cycles = c;
            $display("[TB_ARGMAX] %-24s | exp_cls=%0d got_cls=%0d | exp_max=%4d got_max=%4d | cycles=%0d",
                     label, exp_cls, predicted_class, exp_max, max_score, c);
            if (predicted_class !== exp_cls || max_score !== exp_max) begin
                $display("[FAIL] tb_argmax mismatch on %s", label);
                errors = errors + 1;
            end
            @(posedge clk);
        end
    endtask

    initial begin
        $readmemh("sim/vectors/img0_fc_scores.mem",      score_mem);
        $readmemh("sim/vectors/img0_expected_class.mem", gold_cls);

        rst    = 1'b1;
        start  = 1'b0;
        errors = 0;
        cycles = 0;

        @(posedge clk);
        @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // Case 1: Image 0 golden FC scores
        run_one_argmax_case(gold_cls[0][3:0], 8'sd7, "Image0 Golden Scores");

        // Case 2: Tie-breaking to lowest class index (class 1, 2, 5 tied at +25)
        score_mem[0] = -8'sd10;  score_mem[1] =  8'sd25;  score_mem[2] =  8'sd25;
        score_mem[3] =  8'sd10;  score_mem[4] = -8'sd128; score_mem[5] =  8'sd25;
        score_mem[6] =  8'sd0;   score_mem[7] =  8'sd1;   score_mem[8] =  8'sd2;
        score_mem[9] =  8'sd3;
        run_one_argmax_case(4'd1, 8'sd25, "Tie-Break Lowest Idx");

        // Case 3: All-negative scores (class 4 and 8 tied at -12)
        score_mem[0] = -8'sd120; score_mem[1] = -8'sd95;  score_mem[2] = -8'sd80;
        score_mem[3] = -8'sd45;  score_mem[4] = -8'sd12;  score_mem[5] = -8'sd88;
        score_mem[6] = -8'sd30;  score_mem[7] = -8'sd15;  score_mem[8] = -8'sd12;
        score_mem[9] = -8'sd100;
        run_one_argmax_case(4'd4, -8'sd12, "All-Negative Scores");

        // Case 4: Last class index 9 wins with +127
        score_mem[0] = -8'sd128; score_mem[1] = -8'sd100; score_mem[2] = -8'sd50;
        score_mem[3] =  8'sd0;   score_mem[4] =  8'sd10;  score_mem[5] =  8'sd20;
        score_mem[6] =  8'sd30;  score_mem[7] =  8'sd40;  score_mem[8] =  8'sd50;
        score_mem[9] =  8'sd127;
        run_one_argmax_case(4'd9, 8'sd127, "Last Class Index 9");

        $display("=================================================================");
        $display("[TB_ARGMAX] Cases Checked: 4 | Cycles/Run: %0d | Mismatches: %0d",
                 cycles, errors);
        if (errors == 0)
            $display("[TB_ARGMAX] STATUS: PASS");
        else
            $display("[TB_ARGMAX] STATUS: FAIL (%0d errors)", errors);
        $display("=================================================================");
        $finish;
    end

endmodule
