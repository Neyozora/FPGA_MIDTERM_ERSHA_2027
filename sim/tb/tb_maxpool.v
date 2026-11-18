// =============================================================================
// Testbench: tb_maxpool.v
// Description:
//   Self-checking ModelSim unit testbench for `maxpool.v`:
//     1. Directed signed INT8 test on a 1x4x4 input containing negative, zero,
//        and positive values (including an all-negative 2x2 window) using a
//        second parameterized instance `dut_signed`.
//     2. Full Pool1 tensor verification (8x30x30 -> 8x15x15 = 1,800 outputs)
//        against `sim/vectors/img0_relu1_out.mem` and `sim/vectors/img0_pool1_out.mem`.
// =============================================================================
`timescale 1ns / 1ps

module tb_maxpool;

    reg clk;
    reg rst;

    initial clk = 1'b0;
    always #10 clk = ~clk;

    // -------------------------------------------------------------------------
    // Instance 1: Directed 1x4x4 -> 1x2x2 signed test (includes negative windows)
    // -------------------------------------------------------------------------
    reg               s_start;
    wire              s_busy;
    wire              s_done;
    wire [3:0]        s_in_rd_addr;
    reg  signed [7:0] s_in_rd_data;
    wire              s_out_valid;
    wire [1:0]        s_out_addr;
    wire signed [7:0] s_out_data;

    reg signed [7:0]  s_in_mem   [0:15];
    reg signed [7:0]  s_gold_mem [0:3];

    always @(posedge clk) begin
        s_in_rd_data <= s_in_mem[s_in_rd_addr];
    end

    maxpool #(
        .CHANNELS   (1),
        .IN_H       (4),
        .IN_W       (4),
        .OUT_H      (2),
        .OUT_W      (2),
        .IN_ADDR_W  (4),
        .OUT_ADDR_W (2)
    ) dut_signed (
        .clk        (clk),
        .rst        (rst),
        .start      (s_start),
        .busy       (s_busy),
        .done       (s_done),
        .in_rd_addr (s_in_rd_addr),
        .in_rd_data (s_in_rd_data),
        .out_valid  (s_out_valid),
        .out_addr   (s_out_addr),
        .out_data   (s_out_data)
    );

    // -------------------------------------------------------------------------
    // Instance 2: Full Pool1 (8x30x30 -> 8x15x15 = 1,800 elements)
    // -------------------------------------------------------------------------
    reg               p_start;
    wire              p_busy;
    wire              p_done;
    wire [12:0]       p_in_rd_addr;
    reg  signed [7:0] p_in_rd_data;
    wire              p_out_valid;
    wire [10:0]       p_out_addr;
    wire signed [7:0] p_out_data;

    reg signed [7:0]  p_in_mem   [0:7199];
    reg signed [7:0]  p_gold_mem [0:1799];

    always @(posedge clk) begin
        p_in_rd_data <= p_in_mem[p_in_rd_addr];
    end

    maxpool #(
        .CHANNELS   (8),
        .IN_H       (30),
        .IN_W       (30),
        .OUT_H      (15),
        .OUT_W      (15),
        .IN_ADDR_W  (13),
        .OUT_ADDR_W (11)
    ) dut_pool1 (
        .clk        (clk),
        .rst        (rst),
        .start      (p_start),
        .busy       (p_busy),
        .done       (p_done),
        .in_rd_addr (p_in_rd_addr),
        .in_rd_data (p_in_rd_data),
        .out_valid  (p_out_valid),
        .out_addr   (p_out_addr),
        .out_data   (p_out_data)
    );

    integer errors;
    integer max_abs_err;
    integer diff;
    integer s_count;
    integer p_count;
    integer p_cycles;

    initial begin
        // Directed 4x4 matrix with negative, zero, and positive values:
        // Window (0,0): [-128, -50, -10, -99] -> max = -10
        // Window (0,1): [   0,  12, 127,  -1] -> max = 127
        // Window (1,0): [  -5,  -6, -20,  -2] -> max =  -2
        // Window (1,1): [  -7,  -8,  -9,  -3] -> max =  -3
        s_in_mem[0]  = -8'sd128; s_in_mem[1]  = -8'sd50;  s_in_mem[2]  =  8'sd0;   s_in_mem[3]  =  8'sd12;
        s_in_mem[4]  = -8'sd10;  s_in_mem[5]  = -8'sd99;  s_in_mem[6]  =  8'sd127; s_in_mem[7]  = -8'sd1;
        s_in_mem[8]  = -8'sd5;   s_in_mem[9]  = -8'sd6;   s_in_mem[10] = -8'sd7;   s_in_mem[11] = -8'sd8;
        s_in_mem[12] = -8'sd20;  s_in_mem[13] = -8'sd2;   s_in_mem[14] = -8'sd9;   s_in_mem[15] = -8'sd3;

        s_gold_mem[0] = -8'sd10;
        s_gold_mem[1] =  8'sd127;
        s_gold_mem[2] = -8'sd2;
        s_gold_mem[3] = -8'sd3;

        $readmemh("sim/vectors/img0_relu1_out.mem", p_in_mem);
        $readmemh("sim/vectors/img0_pool1_out.mem", p_gold_mem);

        rst         = 1'b1;
        s_start     = 1'b0;
        p_start     = 1'b0;
        errors      = 0;
        max_abs_err = 0;
        s_count     = 0;
        p_count     = 0;
        p_cycles    = 0;

        @(posedge clk);
        @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // Run Test 1: Directed signed 4x4 -> 2x2
        s_start <= 1'b1;
        @(posedge clk);
        s_start <= 1'b0;

        while (!s_done) begin
            @(posedge clk);
            #1;
            if (s_out_valid) begin
                s_count = s_count + 1;
                if (s_out_data !== s_gold_mem[s_out_addr]) begin
                    $display("[FAIL] tb_maxpool signed idx=%0d exp=%0d got=%0d",
                             s_out_addr, s_gold_mem[s_out_addr], s_out_data);
                    errors = errors + 1;
                end
            end
        end

        // Run Test 2: Full Pool1 (8x30x30 -> 8x15x15 = 1,800 outputs)
        @(posedge clk);
        p_start <= 1'b1;
        @(posedge clk);
        p_start <= 1'b0;

        while (!p_done) begin
            @(posedge clk);
            p_cycles = p_cycles + 1;
            #1;
            if (p_out_valid) begin
                p_count = p_count + 1;
                diff = (p_out_data >= p_gold_mem[p_out_addr]) ?
                       (p_out_data - p_gold_mem[p_out_addr]) :
                       (p_gold_mem[p_out_addr] - p_out_data);
                if (diff > max_abs_err) max_abs_err = diff;
                if (p_out_data !== p_gold_mem[p_out_addr]) begin
                    if (errors < 10)
                        $display("[FAIL] tb_maxpool Pool1 idx=%0d exp=%0d got=%0d",
                                 p_out_addr, p_gold_mem[p_out_addr], p_out_data);
                    errors = errors + 1;
                end
            end
        end

        if (s_count != 4 || p_count != 1800) begin
            $display("[FAIL] tb_maxpool output count mismatch: s_count=%0d p_count=%0d", s_count, p_count);
            errors = errors + 1;
        end

        $display("=================================================================");
        $display("[TB_MAXPOOL] Checks: %0d (4 directed signed + %0d Pool1 golden)", s_count + p_count, p_count);
        $display("[TB_MAXPOOL] Pool1 Cycles: %0d | Max Abs Error: %0d | Mismatches: %0d",
                 p_cycles, max_abs_err, errors);
        if (errors == 0)
            $display("[TB_MAXPOOL] STATUS: PASS");
        else
            $display("[TB_MAXPOOL] STATUS: FAIL (%0d errors)", errors);
        $display("=================================================================");
        $finish;
    end

endmodule
