`timescale 1ns / 1ps

module tb_gui_single_m2;
    localparam CLK_PERIOD = 20;

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

    reg [7:0] single_img [0:3071];
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    integer px, c, cyc;
    reg signed [7:0] sc;

    initial begin
        $readmemh("sim_m2/vectors/gui_single_img.mem", single_img);

        rst             = 1'b1;
        start           = 1'b0;
        ext_img_wr_en   = 1'b0;
        ext_img_wr_addr = 12'd0;
        ext_img_wr_data = 8'd0;

        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        for (px = 0; px < 3072; px = px + 1) begin
            @(posedge clk);
            ext_img_wr_en   <= 1'b1;
            ext_img_wr_addr <= px[11:0];
            ext_img_wr_data <= single_img[px];
        end
        @(posedge clk);
        ext_img_wr_en <= 1'b0;

        @(posedge clk);
        start <= 1'b1;
        cyc = 0;
        @(posedge clk);
        start <= 1'b0;

        while (!done) begin
            @(posedge clk);
            cyc = cyc + 1;
        end

        $display("GUI_RTL_CYCLES=%0d", cyc);
        $display("GUI_RTL_PRED=%0d", predicted_class);
        $write("GUI_RTL_SCORES=");
        for (c = 0; c < 10; c = c + 1) begin
            sc = $signed(dbg_fc_scores_flat[c*8 +: 8]);
            $write("%0d%s", sc, (c < 9) ? "," : "\n");
        end
        $stop;
    end
endmodule
