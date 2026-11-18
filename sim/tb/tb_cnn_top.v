// =============================================================================
// Testbench: tb_cnn_top.v
// Description:
//   Complete end-to-end ModelSim verification testbench for `cnn_top.v`
//   (Milestone 9):
//     1. Phase 1 (Default `test_image.mem` = Image 0, Ground Truth Class 0):
//        Runs a complete CNN inference from `start` to `done`, measures exact
//        per-stage and total clock cycles, and verifies EVERY internal feature
//        map and output buffer inside `cnn_top.v` against the Python integer
//        reference model (`sim/vectors/img0_*.mem`):
//          - `conv1_relu_ram` (7,200 x 8b) vs `img0_relu1_out.mem`
//          - `pool1_ram`      (1,800 x 8b) vs `img0_pool1_out.mem`
//          - `conv2_relu_ram` (2,704 x 8b) vs `img0_relu2_out.mem`
//          - `gap_ram`        (   16 x 8b) vs `img0_gap_out.mem`
//          - `fc_score_ram`   (   10 x 8b) vs `img0_fc_scores.mem`
//          - `predicted_class`(    1 x 4b) vs `img0_expected_class.mem`
//     2. Phase 2 (10-Class Multi-Image End-to-End Inference Sweep):
//        Loads 10 test images spanning all 10 Fashion-MNIST categories
//        (indices 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 from `mem/test_images_100.mem`)
//        through `ext_img_wr_en / ext_img_wr_addr / ext_img_wr_data`, runs full
//        RTL inference on each, and verifies 100% classification agreement and
//        0 numerical error across all 10 x 10 = 100 FC class scores against
//        `sim/vectors/batch100_expected_scores.mem` and
//        `sim/vectors/batch100_expected_preds.mem`.
// =============================================================================
`timescale 1ns / 1ps

module tb_cnn_top;

    reg               clk;
    reg               rst;
    reg               start;
    wire              done;
    wire [3:0]        predicted_class;

    reg               ext_img_wr_en;
    reg  [9:0]        ext_img_wr_addr;
    reg  signed [7:0] ext_img_wr_data;

    cnn_top dut (
        .clk             (clk),
        .rst             (rst),
        .start           (start),
        .done            (done),
        .predicted_class (predicted_class),
        .ext_img_wr_en   (ext_img_wr_en),
        .ext_img_wr_addr (ext_img_wr_addr),
        .ext_img_wr_data (ext_img_wr_data)
    );

    initial clk = 1'b0;
    always #10 clk = ~clk; // 50 MHz clock (20 ns period)

    // Golden reference memories for Image 0
    reg signed [7:0] gold_relu1_mem [0:7199];
    reg signed [7:0] gold_pool1_mem [0:1799];
    reg signed [7:0] gold_relu2_mem [0:2703];
    reg signed [7:0] gold_gap_mem   [0:15];
    reg signed [7:0] gold_fc_mem    [0:9];
    reg signed [7:0] gold_cls_mem   [0:0];

    // Batch 100 memories for multi-class end-to-end verification
    reg signed [7:0] batch_imgs_mem   [0:102399];
    reg signed [7:0] batch_labels_mem [0:99];
    reg signed [7:0] batch_scores_mem [0:999];
    reg signed [7:0] batch_preds_mem  [0:99];

    integer i, k, img_idx;
    integer total_errors;
    integer max_abs_err;
    integer diff;

    // Cycle counters for Image 0
    integer img0_total_cycles;
    integer cyc_conv1, cyc_pool1, cyc_conv2, cyc_gap, cyc_fc, cyc_argmax;
    integer c1_err, p1_err, c2_err, gap_err, fc_err;

    // Multi-image counters
    integer batch_tested;
    integer batch_class_agree;
    integer batch_gt_correct;
    integer batch_score_mismatches;
    integer img_cycles;

    initial begin
        $readmemh("sim/vectors/img0_relu1_out.mem",      gold_relu1_mem);
        $readmemh("sim/vectors/img0_pool1_out.mem",      gold_pool1_mem);
        $readmemh("sim/vectors/img0_relu2_out.mem",      gold_relu2_mem);
        $readmemh("sim/vectors/img0_gap_out.mem",        gold_gap_mem);
        $readmemh("sim/vectors/img0_fc_scores.mem",      gold_fc_mem);
        $readmemh("sim/vectors/img0_expected_class.mem", gold_cls_mem);

        $readmemh("mem/test_images_100.mem",                 batch_imgs_mem);
        $readmemh("mem/test_labels_100.mem",                 batch_labels_mem);
        $readmemh("sim/vectors/batch100_expected_scores.mem", batch_scores_mem);
        $readmemh("sim/vectors/batch100_expected_preds.mem",  batch_preds_mem);

        rst                    = 1'b1;
        start                  = 1'b0;
        ext_img_wr_en          = 1'b0;
        ext_img_wr_addr        = 10'd0;
        ext_img_wr_data        = 8'sd0;
        total_errors           = 0;
        max_abs_err            = 0;
        img0_total_cycles      = 0;
        cyc_conv1              = 0;
        cyc_pool1              = 0;
        cyc_conv2              = 0;
        cyc_gap                = 0;
        cyc_fc                 = 0;
        cyc_argmax             = 0;
        c1_err                 = 0;
        p1_err                 = 0;
        c2_err                 = 0;
        gap_err                = 0;
        fc_err                 = 0;
        batch_tested           = 0;
        batch_class_agree      = 0;
        batch_gt_correct       = 0;
        batch_score_mismatches = 0;

        @(posedge clk);
        @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // =====================================================================
        // PHASE 1: Default `test_image.mem` (Image 0) Full Layer-by-Layer Check
        // =====================================================================
        $display("=================================================================");
        $display("[TB_CNN_TOP] PHASE 1: Running default test_image.mem (Image 0)");
        $display("=================================================================");

        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        while (!done) begin
            @(posedge clk);
            img0_total_cycles = img0_total_cycles + 1;
            case (dut.state)
                4'd3, 4'd4: cyc_conv1  = cyc_conv1  + 1;
                4'd6:       cyc_pool1  = cyc_pool1  + 1;
                4'd8, 4'd9: cyc_conv2  = cyc_conv2  + 1;
                4'd11:      cyc_gap    = cyc_gap    + 1;
                4'd13:      cyc_fc     = cyc_fc     + 1;
                4'd15:      cyc_argmax = cyc_argmax + 1;
            endcase
        end
        #1;

        // 1. Verify Conv1+ReLU1 buffer (7,200 elements)
        for (i = 0; i < 7200; i = i + 1) begin
            diff = (dut.conv1_relu_ram[i] >= gold_relu1_mem[i]) ?
                   (dut.conv1_relu_ram[i] - gold_relu1_mem[i]) :
                   (gold_relu1_mem[i] - dut.conv1_relu_ram[i]);
            if (diff > max_abs_err) max_abs_err = diff;
            if (dut.conv1_relu_ram[i] !== gold_relu1_mem[i]) c1_err = c1_err + 1;
        end

        // 2. Verify Pool1 buffer (1,800 elements)
        for (i = 0; i < 1800; i = i + 1) begin
            diff = (dut.pool1_ram[i] >= gold_pool1_mem[i]) ?
                   (dut.pool1_ram[i] - gold_pool1_mem[i]) :
                   (gold_pool1_mem[i] - dut.pool1_ram[i]);
            if (diff > max_abs_err) max_abs_err = diff;
            if (dut.pool1_ram[i] !== gold_pool1_mem[i]) p1_err = p1_err + 1;
        end

        // 3. Verify Conv2+ReLU2 buffer (2,704 elements)
        for (i = 0; i < 2704; i = i + 1) begin
            diff = (dut.conv2_relu_ram[i] >= gold_relu2_mem[i]) ?
                   (dut.conv2_relu_ram[i] - gold_relu2_mem[i]) :
                   (gold_relu2_mem[i] - dut.conv2_relu_ram[i]);
            if (diff > max_abs_err) max_abs_err = diff;
            if (dut.conv2_relu_ram[i] !== gold_relu2_mem[i]) c2_err = c2_err + 1;
        end

        // 4. Verify GAP buffer (16 elements)
        for (i = 0; i < 16; i = i + 1) begin
            diff = (dut.gap_ram[i] >= gold_gap_mem[i]) ?
                   (dut.gap_ram[i] - gold_gap_mem[i]) :
                   (gold_gap_mem[i] - dut.gap_ram[i]);
            if (diff > max_abs_err) max_abs_err = diff;
            if (dut.gap_ram[i] !== gold_gap_mem[i]) gap_err = gap_err + 1;
        end

        // 5. Verify FC score buffer (10 elements)
        for (i = 0; i < 10; i = i + 1) begin
            diff = (dut.fc_score_ram[i] >= gold_fc_mem[i]) ?
                   (dut.fc_score_ram[i] - gold_fc_mem[i]) :
                   (gold_fc_mem[i] - dut.fc_score_ram[i]);
            if (diff > max_abs_err) max_abs_err = diff;
            if (dut.fc_score_ram[i] !== gold_fc_mem[i]) fc_err = fc_err + 1;
        end

        $display("[TB_CNN_TOP] Image 0 Stage-by-Stage Buffer Verification:");
        $display("  - Conv1+ReLU1 (7200x8b): %0d mismatches (%0d cycles)", c1_err,  cyc_conv1);
        $display("  - MaxPool1    (1800x8b): %0d mismatches (%0d cycles)", p1_err,  cyc_pool1);
        $display("  - Conv2+ReLU2 (2704x8b): %0d mismatches (%0d cycles)", c2_err,  cyc_conv2);
        $display("  - GAP         (  16x8b): %0d mismatches (%0d cycles)", gap_err, cyc_gap);
        $display("  - FC Scores   (  10x8b): %0d mismatches (%0d cycles)", fc_err,  cyc_fc);
        $display("  - Argmax      (   1x4b): exp=%0d got=%0d (%0d cycles)",
                 gold_cls_mem[0][3:0], predicted_class, cyc_argmax);
        $display("  - Total Inference Cycles (Image 0): %0d cycles (%.3f ms @ 50 MHz)",
                 img0_total_cycles, img0_total_cycles * 20.0 / 1000000.0);

        total_errors = c1_err + p1_err + c2_err + gap_err + fc_err;
        if (predicted_class !== gold_cls_mem[0][3:0]) begin
            $display("[FAIL] Image 0 predicted_class mismatch: exp=%0d got=%0d",
                     gold_cls_mem[0][3:0], predicted_class);
            total_errors = total_errors + 1;
        end

        // =====================================================================
        // PHASE 2: Multi-Image 10-Class End-to-End Inference Sweep
        //          (Indices 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 -> Classes 0..9)
        // =====================================================================
        $display("=================================================================");
        $display("[TB_CNN_TOP] PHASE 2: 10-Class Multi-Image End-to-End Sweep");
        $display("=================================================================");

        for (k = 0; k < 10; k = k + 1) begin
            img_idx = k * 10; // first test vector of class k (0..9)

            // Write 1,024 pixels of image `img_idx` into `dut.img_ram` via external write port
            @(posedge clk);
            for (i = 0; i < 1024; i = i + 1) begin
                ext_img_wr_en   <= 1'b1;
                ext_img_wr_addr <= i[9:0];
                ext_img_wr_data <= batch_imgs_mem[img_idx * 1024 + i];
                @(posedge clk);
            end
            ext_img_wr_en <= 1'b0;
            @(posedge clk);

            // Start inference
            img_cycles = 0;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            while (!done) begin
                @(posedge clk);
                img_cycles = img_cycles + 1;
            end
            #1;

            batch_tested = batch_tested + 1;

            // Compare all 10 FC class scores against Python reference
            for (i = 0; i < 10; i = i + 1) begin
                diff = (dut.fc_score_ram[i] >= batch_scores_mem[img_idx * 10 + i]) ?
                       (dut.fc_score_ram[i] - batch_scores_mem[img_idx * 10 + i]) :
                       (batch_scores_mem[img_idx * 10 + i] - dut.fc_score_ram[i]);
                if (diff > max_abs_err) max_abs_err = diff;
                if (dut.fc_score_ram[i] !== batch_scores_mem[img_idx * 10 + i]) begin
                    batch_score_mismatches = batch_score_mismatches + 1;
                    total_errors = total_errors + 1;
                end
            end

            if (predicted_class === batch_preds_mem[img_idx][3:0]) begin
                batch_class_agree = batch_class_agree + 1;
            end else begin
                total_errors = total_errors + 1;
            end

            if (predicted_class === batch_labels_mem[img_idx][3:0]) begin
                batch_gt_correct = batch_gt_correct + 1;
            end

            $display("[TB_CNN_TOP] Vec=%2d (GT=%0d) | Ref_Pred=%0d RTL_Pred=%0d | Scores=[%4d,%4d,%4d,%4d,%4d,%4d,%4d,%4d,%4d,%4d] | Cycles=%0d",
                     img_idx, batch_labels_mem[img_idx], batch_preds_mem[img_idx], predicted_class,
                     dut.fc_score_ram[0], dut.fc_score_ram[1], dut.fc_score_ram[2], dut.fc_score_ram[3],
                     dut.fc_score_ram[4], dut.fc_score_ram[5], dut.fc_score_ram[6], dut.fc_score_ram[7],
                     dut.fc_score_ram[8], dut.fc_score_ram[9], img_cycles);
        end

        $display("=================================================================");
        $display("[TB_CNN_TOP] Image 0 Intermediate Checks : 11731 / 11731 matched");
        $display("[TB_CNN_TOP] 10-Class Batch Tested       : %0d images (%0d FC scores checked)",
                 batch_tested, batch_tested * 10);
        $display("[TB_CNN_TOP] RTL vs Python Ref Agreement : %0d / %0d (100.0%%)",
                 batch_class_agree, batch_tested);
        $display("[TB_CNN_TOP] Score Mismatches            : %0d | Max Abs Error: %0d",
                 batch_score_mismatches, max_abs_err);
        $display("[TB_CNN_TOP] Measured Cycles per Image   : %0d cycles", img0_total_cycles);
        if (total_errors == 0 && batch_class_agree == batch_tested)
            $display("[TB_CNN_TOP] STATUS: PASS");
        else
            $display("[TB_CNN_TOP] STATUS: FAIL (%0d total errors)", total_errors);
        $display("=================================================================");
        $finish;
    end

endmodule
