// =============================================================================
// Testbench: tb_gap.v
// Description:
//   Self-checking ModelSim unit testbench for `gap.v`:
//   Runs Global Average Pooling over the 16x13x13 (`2,704`-element) golden
//   ReLU2 tensor (`sim/vectors/img0_relu2_out.mem`) and verifies all 16
//   `out_sum_int32` spatial sums against `sim/vectors/img0_gap_sum.mem` AND
//   all 16 `out_data` (`INT8`) outputs against `sim/vectors/img0_gap_out.mem`.
// =============================================================================
`timescale 1ns / 1ps

module tb_gap;

    reg                clk;
    reg                rst;
    reg                start;
    wire               busy;
    wire               done;

    wire [11:0]        in_rd_addr;
    reg  signed [7:0]  in_rd_data;

    wire               out_valid;
    wire [3:0]         out_ch;
    wire signed [31:0] out_sum_int32;
    wire signed [7:0]  out_data;

    reg signed [7:0]   relu2_mem    [0:2703];
    reg signed [31:0]  gold_sum_mem [0:15];
    reg signed [7:0]   gold_out_mem [0:15];

    always @(posedge clk) begin
        in_rd_data <= relu2_mem[in_rd_addr];
    end

    gap #(
        .CHANNELS      (16),
        .SPATIAL_N     (169),
        .IN_ADDR_W     (12),
        .CH_W          (4),
        .DEFAULT_MULT  (16780),
        .DEFAULT_SHIFT (19)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .start         (start),
        .busy          (busy),
        .done          (done),
        .cfg_mult      (16'd16780),
        .cfg_shift     (6'd19),
        .in_rd_addr    (in_rd_addr),
        .in_rd_data    (in_rd_data),
        .out_valid     (out_valid),
        .out_ch        (out_ch),
        .out_sum_int32 (out_sum_int32),
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
        $readmemh("sim/vectors/img0_relu2_out.mem", relu2_mem);
        $readmemh("sim/vectors/img0_gap_sum.mem",   gold_sum_mem);
        $readmemh("sim/vectors/img0_gap_out.mem",   gold_out_mem);

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
                diff = (out_data >= gold_out_mem[out_ch]) ?
                       (out_data - gold_out_mem[out_ch]) :
                       (gold_out_mem[out_ch] - out_data);
                if (diff > max_abs_err) max_abs_err = diff;

                $display("[TB_GAP] ch=%2d | sum_exp=%5d sum_got=%5d | out_exp=%3d out_got=%3d",
                         out_ch, gold_sum_mem[out_ch], out_sum_int32, gold_out_mem[out_ch], out_data);

                if (out_sum_int32 !== gold_sum_mem[out_ch] || out_data !== gold_out_mem[out_ch]) begin
                    $display("[FAIL] tb_gap ch=%0d mismatch!", out_ch);
                    errors = errors + 1;
                end
            end
        end

        if (out_cnt != 16) begin
            $display("[FAIL] tb_gap expected 16 channel outputs, got %0d", out_cnt);
            errors = errors + 1;
        end

        $display("=================================================================");
        $display("[TB_GAP] Channels Checked: %0d | Cycles: %0d | Max Abs Error: %0d | Mismatches: %0d",
                 out_cnt, cycles, max_abs_err, errors);
        if (errors == 0)
            $display("[TB_GAP] STATUS: PASS");
        else
            $display("[TB_GAP] STATUS: FAIL (%0d errors)", errors);
        $display("=================================================================");
        $finish;
    end

endmodule
