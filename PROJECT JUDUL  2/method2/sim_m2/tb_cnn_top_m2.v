`timescale 1ns / 1ps

module tb_cnn_top_m2;

    localparam CLK_PERIOD = 20;
    parameter  NUM_TEST   = 100;
    localparam NUM_IMAGES = 100;

    reg        clk;
    reg        rst;
    reg        start;
    wire       done;
    wire [3:0] predicted_class;
    reg        ext_img_wr_en;
    reg [11:0] ext_img_wr_addr;
    reg [7:0]  ext_img_wr_data;
    wire [79:0] dbg_fc_scores_flat;

    cnn_top_m2 dut (
        .clk                (clk),
        .rst                (rst),
        .start              (start),
        .done               (done),
        .predicted_class    (predicted_class),
        .ext_img_wr_en      (ext_img_wr_en),
        .ext_img_wr_addr    (ext_img_wr_addr),
        .ext_img_wr_data    (ext_img_wr_data),
        .dbg_fc_scores_flat (dbg_fc_scores_flat)
    );

    reg [7:0] all_imgs   [0:NUM_IMAGES*3072-1];
    reg [3:0] all_labels [0:NUM_IMAGES-1];
    reg [7:0] exp_scores [0:NUM_IMAGES*10-1];
    reg [3:0] exp_preds  [0:NUM_IMAGES-1];

    initial begin
        $readmemh("sim_m2/vectors/hw_images_100.mem", all_imgs);
        $readmemh("sim_m2/vectors/hw_labels_100.mem", all_labels);
        $readmemh("sim_m2/vectors/hw_expected_scores_100.mem", exp_scores);
        $readmemh("sim_m2/vectors/hw_expected_preds_100.mem", exp_preds);
    end

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    integer img_idx, px, c, cyc;
    integer pred_matches, score_matches, correct_cls;
    reg signed [7:0] rtl_sc, py_sc;
    reg img_ok;

    initial begin
        rst             = 1'b1;
        start           = 1'b0;
        ext_img_wr_en   = 1'b0;
        ext_img_wr_addr = 12'd0;
        ext_img_wr_data = 8'd0;
        pred_matches    = 0;
        score_matches   = 0;
        correct_cls     = 0;

        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        for (img_idx = 0; img_idx < NUM_TEST; img_idx = img_idx + 1) begin
            // Load 3072-byte RGB image
            for (px = 0; px < 3072; px = px + 1) begin
                @(posedge clk);
                ext_img_wr_en   <= 1'b1;
                ext_img_wr_addr <= px[11:0];
                ext_img_wr_data <= all_imgs[img_idx * 3072 + px];
            end
            @(posedge clk);
            ext_img_wr_en <= 1'b0;

            // Start inference & count exact cycles
            @(posedge clk);
            start <= 1'b1;
            cyc = 0;
            @(posedge clk);
            start <= 1'b0;

            while (!done) begin
                @(posedge clk);
                cyc = cyc + 1;
            end

            img_ok = 1'b1;
            for (c = 0; c < 10; c = c + 1) begin
                rtl_sc = $signed(dbg_fc_scores_flat[c*8 +: 8]);
                py_sc  = $signed(exp_scores[img_idx * 10 + c]);
                if (rtl_sc === py_sc)
                    score_matches = score_matches + 1;
                else begin
                    img_ok = 1'b0;
                    $display("MISMATCH img=%0d class=%0d rtl=%0d py=%0d", img_idx, c, rtl_sc, py_sc);
                end
            end

            if (predicted_class === exp_preds[img_idx])
                pred_matches = pred_matches + 1;
            if (predicted_class === all_labels[img_idx])
                correct_cls = correct_cls + 1;

            if (img_idx == 0 || !img_ok)
                $display("IMG %02d: cycles=%0d pred=%0d (exp=%0d, label=%0d) scores_ok=%0d",
                         img_idx, cyc, predicted_class, exp_preds[img_idx], all_labels[img_idx], img_ok);
        end

        $display("============================================================");
        $display("METHOD 2 BIT-EXACT VERIFICATION SUMMARY (%0d CIFAR-10 IMAGES)", NUM_TEST);
        $display("  Exact Clock Cycles / Image : %0d", cyc);
        $display("  Prediction Parity          : %0d / %0d", pred_matches, NUM_TEST);
        $display("  INT8 Score Parity (x10)    : %0d / %0d", score_matches, NUM_TEST * 10);
        $display("  Classification Accuracy    : %0d / %0d", correct_cls, NUM_TEST);
        $display("============================================================");
        $stop;
    end

endmodule
