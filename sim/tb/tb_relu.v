// =============================================================================
// Testbench: tb_relu.v
// Description:
//   Self-checking ModelSim unit testbench for `relu.v`:
//     1. Exhaustive sweep across all 256 signed INT8 values (-128 .. +127),
//        checking both combinational (`out_data_comb`) and registered (`out_data`)
//        outputs for negative, zero, and positive inputs.
//     2. Golden vector verification on all 7,200 Conv1 pre-ReLU values
//        (`sim/vectors/img0_conv1_req.mem` -> `sim/vectors/img0_relu1_out.mem`)
//        and all 2,704 Conv2 pre-ReLU values
//        (`sim/vectors/img0_conv2_req.mem` -> `sim/vectors/img0_relu2_out.mem`).
// =============================================================================
`timescale 1ns / 1ps

module tb_relu;

    reg                clk;
    reg                rst;
    reg                in_valid;
    reg  [12:0]        in_addr;
    reg  signed [7:0]  in_data;

    wire signed [7:0]  out_data_comb;
    wire               out_valid;
    wire [12:0]        out_addr;
    wire signed [7:0]  out_data;

    relu #(
        .DATA_W (8),
        .ADDR_W (13)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .in_valid      (in_valid),
        .in_addr       (in_addr),
        .in_data       (in_data),
        .out_data_comb (out_data_comb),
        .out_valid     (out_valid),
        .out_addr      (out_addr),
        .out_data      (out_data)
    );

    initial clk = 1'b0;
    always #10 clk = ~clk; // 50 MHz clock

    reg signed [7:0] c1_req_mem  [0:7199];
    reg signed [7:0] c1_gold_mem [0:7199];
    reg signed [7:0] c2_req_mem  [0:2703];
    reg signed [7:0] c2_gold_mem [0:2703];

    integer i;
    integer errors;
    integer max_abs_err;
    integer diff;
    integer cycle_cnt;
    reg signed [7:0] exp_val;

    initial begin
        $readmemh("sim/vectors/img0_conv1_req.mem", c1_req_mem);
        $readmemh("sim/vectors/img0_relu1_out.mem", c1_gold_mem);
        $readmemh("sim/vectors/img0_conv2_req.mem", c2_req_mem);
        $readmemh("sim/vectors/img0_relu2_out.mem", c2_gold_mem);

        rst         = 1'b1;
        in_valid    = 1'b0;
        in_addr     = 13'd0;
        in_data     = 8'sd0;
        errors      = 0;
        max_abs_err = 0;
        cycle_cnt   = 0;

        @(posedge clk);
        @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // ---------------------------------------------------------------------
        // Test 1: Exhaustive signed INT8 sweep (-128 .. +127)
        // ---------------------------------------------------------------------
        for (i = -128; i <= 127; i = i + 1) begin
            in_valid <= 1'b1;
            in_addr  <= i + 128;
            in_data  <= i[7:0];
            exp_val  = (i < 0) ? 8'sd0 : i[7:0];

            #1; // check combinational output before clock edge
            if (out_data_comb !== exp_val) begin
                $display("[FAIL] tb_relu comb: in=%0d exp=%0d got=%0d", i, exp_val, out_data_comb);
                errors = errors + 1;
            end

            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            #1; // check registered output after clock edge
            if (out_valid !== 1'b1 || out_data !== exp_val || out_addr !== (i + 128)) begin
                $display("[FAIL] tb_relu reg: in=%0d exp=%0d got=%0d valid=%b",
                         i, exp_val, out_data, out_valid);
                errors = errors + 1;
            end
        end

        // ---------------------------------------------------------------------
        // Test 2: Golden Conv1 -> ReLU1 tensor (7,200 elements)
        // ---------------------------------------------------------------------
        for (i = 0; i < 7200; i = i + 1) begin
            in_valid <= 1'b1;
            in_addr  <= i[12:0];
            in_data  <= c1_req_mem[i];
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            #1;
            diff = (out_data >= c1_gold_mem[i]) ? (out_data - c1_gold_mem[i]) : (c1_gold_mem[i] - out_data);
            if (diff > max_abs_err) max_abs_err = diff;
            if (out_valid !== 1'b1 || out_data !== c1_gold_mem[i] || out_addr !== i[12:0]) begin
                if (errors < 10)
                    $display("[FAIL] tb_relu Conv1 idx=%0d in=%0d exp=%0d got=%0d",
                             i, c1_req_mem[i], c1_gold_mem[i], out_data);
                errors = errors + 1;
            end
        end

        // ---------------------------------------------------------------------
        // Test 3: Golden Conv2 -> ReLU2 tensor (2,704 elements)
        // ---------------------------------------------------------------------
        for (i = 0; i < 2704; i = i + 1) begin
            in_valid <= 1'b1;
            in_addr  <= i[12:0];
            in_data  <= c2_req_mem[i];
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            #1;
            diff = (out_data >= c2_gold_mem[i]) ? (out_data - c2_gold_mem[i]) : (c2_gold_mem[i] - out_data);
            if (diff > max_abs_err) max_abs_err = diff;
            if (out_valid !== 1'b1 || out_data !== c2_gold_mem[i] || out_addr !== i[12:0]) begin
                if (errors < 10)
                    $display("[FAIL] tb_relu Conv2 idx=%0d in=%0d exp=%0d got=%0d",
                             i, c2_req_mem[i], c2_gold_mem[i], out_data);
                errors = errors + 1;
            end
        end

        in_valid <= 1'b0;
        @(posedge clk);

        $display("=================================================================");
        $display("[TB_RELU] Checks: 10160 (256 exhaustive + 7200 ReLU1 + 2704 ReLU2)");
        $display("[TB_RELU] Cycles: %0d | Max Abs Error: %0d | Mismatches: %0d",
                 cycle_cnt, max_abs_err, errors);
        if (errors == 0)
            $display("[TB_RELU] STATUS: PASS");
        else
            $display("[TB_RELU] STATUS: FAIL (%0d errors)", errors);
        $display("=================================================================");
        $finish;
    end

endmodule
