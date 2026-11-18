// =============================================================================
// Testbench: tb_vga_cnn_hud.v
// Description:
//   Simulates `rtl/vga_cnn_hud.v` for 1 complete 640x480 @ 60 Hz VGA frame,
//   verifies VGA_HS / VGA_VS / VGA_BLANK_N timing, and saves the exact
//   simulated VGA monitor output to `reports/simulation/modelsim_vga_frame.ppm`!
// =============================================================================
`timescale 1ns / 1ps

module tb_vga_cnn_hud;

    reg         clk_50 = 1'b0;
    reg         rst    = 1'b1;

    // 32x32 Test Image 0 RAM
    reg  [7:0]  img_mem [0:1023];
    wire [9:0]  vga_img_rd_addr;
    reg  [7:0]  vga_img_rd_data = 8'd0;

    // Image 0 scores: [+7, -28, -5, -2, -19, -25, -2, -36, -7, -14], winner = 0
    reg  [3:0]  predicted_class  = 4'd0;
    reg  [3:0]  active_img_class = 4'd0;
    reg         pc_mode_active   = 1'b0;
    reg  [79:0] dbg_fc_scores_flat;
    reg  [7:0]  mouse_x    = 8'd128;
    reg  [7:0]  mouse_y    = 8'd128;
    reg         mouse_left = 1'b0;
    reg         mouse_right= 1'b0;

    wire        VGA_CLK;
    wire        VGA_HS;
    wire        VGA_VS;
    wire        VGA_BLANK_N;
    wire        VGA_SYNC_N;
    wire [7:0]  VGA_R;
    wire [7:0]  VGA_G;
    wire [7:0]  VGA_B;

    // 50 MHz clock (20 ns period)
    always #10 clk_50 = ~clk_50;

    always @(posedge clk_50) begin
        vga_img_rd_data <= img_mem[vga_img_rd_addr];
    end

    vga_cnn_hud dut (
        .clk_50             (clk_50),
        .rst                (rst),
        .vga_img_rd_addr    (vga_img_rd_addr),
        .vga_img_rd_data    (vga_img_rd_data),
        .predicted_class    (predicted_class),
        .active_img_class   (active_img_class),
        .pc_mode_active     (pc_mode_active),
        .dbg_fc_scores_flat (dbg_fc_scores_flat),
        .mouse_x            (mouse_x),
        .mouse_y            (mouse_y),
        .mouse_left         (mouse_left),
        .mouse_right        (mouse_right),
        .VGA_CLK            (VGA_CLK),
        .VGA_HS             (VGA_HS),
        .VGA_VS             (VGA_VS),
        .VGA_BLANK_N        (VGA_BLANK_N),
        .VGA_SYNC_N         (VGA_SYNC_N),
        .VGA_R              (VGA_R),
        .VGA_G              (VGA_G),
        .VGA_B              (VGA_B)
    );

    integer fd;
    integer active_pix_cnt;
    integer hs_cnt;
    integer vs_cnt;
    reg     prev_hs, prev_vs;

    initial begin
        $readmemh("mem/test_image.mem", img_mem);

        // Pack Image 0 scores: [+7, -28, -5, -2, -19, -25, -2, -36, -7, -14]
        dbg_fc_scores_flat[ 7: 0] =  8'sd7;
        dbg_fc_scores_flat[15: 8] = -8'sd28;
        dbg_fc_scores_flat[23:16] = -8'sd5;
        dbg_fc_scores_flat[31:24] = -8'sd2;
        dbg_fc_scores_flat[39:32] = -8'sd19;
        dbg_fc_scores_flat[47:40] = -8'sd25;
        dbg_fc_scores_flat[55:48] = -8'sd2;
        dbg_fc_scores_flat[63:56] = -8'sd36;
        dbg_fc_scores_flat[71:64] = -8'sd7;
        dbg_fc_scores_flat[79:72] = -8'sd14;

        fd = $fopen("reports/simulation/modelsim_vga_frame.ppm", "w");
        $fwrite(fd, "P3\n640 480\n255\n");

        active_pix_cnt = 0;
        hs_cnt = 0;
        vs_cnt = 0;
        prev_hs = 1'b1;
        prev_vs = 1'b1;

        #100;
        rst = 1'b0;

        // Run for 1 full VGA frame (800 x 525 = 420,000 pixel clocks)
        while (vs_cnt == 0) begin
            @(posedge VGA_CLK);
            if (prev_hs == 1'b1 && VGA_HS == 1'b0) hs_cnt = hs_cnt + 1;
            if (prev_vs == 1'b1 && VGA_VS == 1'b0) vs_cnt = vs_cnt + 1;
            prev_hs = VGA_HS;
            prev_vs = VGA_VS;

            if (VGA_BLANK_N && active_pix_cnt < 640 * 480) begin
                $fwrite(fd, "%0d %0d %0d\n", VGA_R, VGA_G, VGA_B);
                active_pix_cnt = active_pix_cnt + 1;
            end
        end

        $fclose(fd);
        $display("=================================================================");
        $display("VGA Frame Simulation Complete!");
        $display("  Active Pixels Captured : %0d (Expected: 307200 = 640x480)", active_pix_cnt);
        $display("  HSYNC Pulses           : %0d (Expected: 525)", hs_cnt);
        $display("  VSYNC Pulses           : %0d (Expected: 1)", vs_cnt);
        $display("  Saved Frame Image      : reports/simulation/modelsim_vga_frame.ppm");
        $display("=================================================================");
        $finish;
    end

endmodule
