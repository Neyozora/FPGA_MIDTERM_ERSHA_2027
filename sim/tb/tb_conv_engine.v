// =============================================================================
// Testbench: tb_conv_engine.v
// Description:
//   Self-checking ModelSim unit testbench for `conv_engine.v`:
//   Tests the single reusable `conv_engine` instance across BOTH layers:
//     1. Conv1 (1x32x32 -> 8x30x30 = 7,200 outputs, M1=23202, n1=19):
//        Compares all 7,200 `out_acc_int32` values against
//        `sim/vectors/img0_conv1_acc.mem` AND all 7,200 `out_req_int8` values
//        against `sim/vectors/img0_conv1_req.mem`.
//     2. Conv2 (8x15x15 -> 16x13x13 = 2,704 outputs, M2=16868, n2=20):
//        Compares all 2,704 `out_acc_int32` values against
//        `sim/vectors/img0_conv2_acc.mem` AND all 2,704 `out_req_int8` values
//        against `sim/vectors/img0_conv2_req.mem`.
// =============================================================================
`timescale 1ns / 1ps

module tb_conv_engine;

    reg                clk;
    reg                rst;
    reg                start;
    wire               busy;
    wire               done;

    reg                is_conv2;

    wire [10:0]        in_rd_addr;
    wire signed [7:0]  in_rd_data;
    wire [10:0]        w_rd_addr;
    wire signed [7:0]  w_rd_data;
    wire [3:0]         b_rd_addr;
    wire signed [31:0] b_rd_data;

    wire               out_valid;
    wire [12:0]        out_addr;
    wire [4:0]         out_ch;
    wire [5:0]         out_y;
    wire [5:0]         out_x;
    wire signed [31:0] out_acc_int32;
    wire signed [7:0]  out_req_int8;

    // Memories for Conv1 & Conv2
    reg signed [7:0]   c1_in_mem       [0:1023];
    reg signed [7:0]   c1_w_mem        [0:71];
    reg signed [31:0]  c1_b_mem        [0:7];
    reg signed [31:0]  c1_gold_acc_mem [0:7199];
    reg signed [7:0]   c1_gold_req_mem [0:7199];

    reg signed [7:0]   c2_in_mem       [0:1799];
    reg signed [7:0]   c2_w_mem        [0:1151];
    reg signed [31:0]  c2_b_mem        [0:15];
    reg signed [31:0]  c2_gold_acc_mem [0:2703];
    reg signed [7:0]   c2_gold_req_mem [0:2703];

    reg signed [7:0]   c1_in_rd_reg;
    reg signed [7:0]   c1_w_rd_reg;
    reg signed [31:0]  c1_b_rd_reg;

    reg signed [7:0]   c2_in_rd_reg;
    reg signed [7:0]   c2_w_rd_reg;
    reg signed [31:0]  c2_b_rd_reg;

    always @(posedge clk) begin
        c1_in_rd_reg <= c1_in_mem[in_rd_addr[9:0]];
        c1_w_rd_reg  <= c1_w_mem[w_rd_addr[6:0]];
        c1_b_rd_reg  <= c1_b_mem[b_rd_addr[2:0]];

        c2_in_rd_reg <= c2_in_mem[in_rd_addr];
        c2_w_rd_reg  <= c2_w_mem[w_rd_addr];
        c2_b_rd_reg  <= c2_b_mem[b_rd_addr];
    end

    assign in_rd_data = is_conv2 ? c2_in_rd_reg : c1_in_rd_reg;
    assign w_rd_data  = is_conv2 ? c2_w_rd_reg  : c1_w_rd_reg;
    assign b_rd_data  = is_conv2 ? c2_b_rd_reg  : c1_b_rd_reg;

    wire [4:0]  cfg_in_c  = is_conv2 ? 5'd8      : 5'd1;
    wire [5:0]  cfg_in_h  = is_conv2 ? 6'd15     : 6'd32;
    wire [5:0]  cfg_in_w  = is_conv2 ? 6'd15     : 6'd32;
    wire [4:0]  cfg_out_c = is_conv2 ? 5'd16     : 5'd8;
    wire [5:0]  cfg_out_h = is_conv2 ? 6'd13     : 6'd30;
    wire [5:0]  cfg_out_w = is_conv2 ? 6'd13     : 6'd30;
    wire [15:0] cfg_mult  = is_conv2 ? 16'd16868 : 16'd23202;
    wire [5:0]  cfg_shift = is_conv2 ? 6'd20     : 6'd19;

    conv_engine #(
        .IN_ADDR_W  (11),
        .W_ADDR_W   (11),
        .B_ADDR_W   (4),
        .OUT_ADDR_W (13)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .start         (start),
        .busy          (busy),
        .done          (done),
        .cfg_in_c      (cfg_in_c),
        .cfg_in_h      (cfg_in_h),
        .cfg_in_w      (cfg_in_w),
        .cfg_out_c     (cfg_out_c),
        .cfg_out_h     (cfg_out_h),
        .cfg_out_w     (cfg_out_w),
        .cfg_mult      (cfg_mult),
        .cfg_shift     (cfg_shift),
        .in_rd_addr    (in_rd_addr),
        .in_rd_data    (in_rd_data),
        .w_rd_addr     (w_rd_addr),
        .w_rd_data     (w_rd_data),
        .b_rd_addr     (b_rd_addr),
        .b_rd_data     (b_rd_data),
        .out_valid     (out_valid),
        .out_ready     (1'b1),
        .out_addr      (out_addr),
        .out_ch        (out_ch),
        .out_y         (out_y),
        .out_x         (out_x),
        .out_acc_int32 (out_acc_int32),
        .out_req_int8  (out_req_int8)
    );

    initial clk = 1'b0;
    always #10 clk = ~clk;

    integer c1_errors, c2_errors;
    integer c1_max_err, c2_max_err;
    integer c1_cnt, c2_cnt;
    integer c1_cycles, c2_cycles;
    integer diff;

    initial begin
        $readmemh("sim/vectors/img0_input.mem",     c1_in_mem);
        $readmemh("mem/conv1_weights.mem",          c1_w_mem);
        $readmemh("mem/conv1_bias.mem",             c1_b_mem);
        $readmemh("sim/vectors/img0_conv1_acc.mem", c1_gold_acc_mem);
        $readmemh("sim/vectors/img0_conv1_req.mem", c1_gold_req_mem);

        $readmemh("sim/vectors/img0_pool1_out.mem", c2_in_mem);
        $readmemh("mem/conv2_weights.mem",          c2_w_mem);
        $readmemh("mem/conv2_bias.mem",             c2_b_mem);
        $readmemh("sim/vectors/img0_conv2_acc.mem", c2_gold_acc_mem);
        $readmemh("sim/vectors/img0_conv2_req.mem", c2_gold_req_mem);

        rst        = 1'b1;
        start      = 1'b0;
        is_conv2   = 1'b0;
        c1_errors  = 0;
        c2_errors  = 0;
        c1_max_err = 0;
        c2_max_err = 0;
        c1_cnt     = 0;
        c2_cnt     = 0;
        c1_cycles  = 0;
        c2_cycles  = 0;

        @(posedge clk);
        @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // ---------------------------------------------------------------------
        // Test 1: Conv1 (1x32x32 -> 8x30x30 = 7,200 elements)
        // ---------------------------------------------------------------------
        is_conv2 <= 1'b0;
        start    <= 1'b1;
        @(posedge clk);
        start    <= 1'b0;

        while (!done) begin
            @(posedge clk);
            c1_cycles = c1_cycles + 1;
            #1;
            if (out_valid) begin
                c1_cnt = c1_cnt + 1;
                diff = (out_req_int8 >= c1_gold_req_mem[out_addr]) ?
                       (out_req_int8 - c1_gold_req_mem[out_addr]) :
                       (c1_gold_req_mem[out_addr] - out_req_int8);
                if (diff > c1_max_err) c1_max_err = diff;

                if (out_acc_int32 !== c1_gold_acc_mem[out_addr] ||
                    out_req_int8  !== c1_gold_req_mem[out_addr]) begin
                    if (c1_errors < 10)
                        $display("[FAIL] tb_conv_engine Conv1 addr=%0d acc_exp=%0d acc_got=%0d req_exp=%0d req_got=%0d",
                                 out_addr, c1_gold_acc_mem[out_addr], out_acc_int32,
                                 c1_gold_req_mem[out_addr], out_req_int8);
                    c1_errors = c1_errors + 1;
                end
            end
        end

        // ---------------------------------------------------------------------
        // Test 2: Conv2 (8x15x15 -> 16x13x13 = 2,704 elements)
        // ---------------------------------------------------------------------
        @(posedge clk);
        is_conv2 <= 1'b1;
        start    <= 1'b1;
        @(posedge clk);
        start    <= 1'b0;

        while (!done) begin
            @(posedge clk);
            c2_cycles = c2_cycles + 1;
            #1;
            if (out_valid) begin
                c2_cnt = c2_cnt + 1;
                diff = (out_req_int8 >= c2_gold_req_mem[out_addr]) ?
                       (out_req_int8 - c2_gold_req_mem[out_addr]) :
                       (c2_gold_req_mem[out_addr] - out_req_int8);
                if (diff > c2_max_err) c2_max_err = diff;

                if (out_acc_int32 !== c2_gold_acc_mem[out_addr] ||
                    out_req_int8  !== c2_gold_req_mem[out_addr]) begin
                    if (c2_errors < 10)
                        $display("[FAIL] tb_conv_engine Conv2 addr=%0d acc_exp=%0d acc_got=%0d req_exp=%0d req_got=%0d",
                                 out_addr, c2_gold_acc_mem[out_addr], out_acc_int32,
                                 c2_gold_req_mem[out_addr], out_req_int8);
                    c2_errors = c2_errors + 1;
                end
            end
        end

        if (c1_cnt != 7200 || c2_cnt != 2704) begin
            $display("[FAIL] tb_conv_engine count mismatch: c1_cnt=%0d c2_cnt=%0d", c1_cnt, c2_cnt);
            c1_errors = c1_errors + 1;
        end

        $display("=================================================================");
        $display("[TB_CONV_ENGINE] Conv1: %0d outputs | Cycles: %0d | Max Abs Err: %0d | Mismatches: %0d",
                 c1_cnt, c1_cycles, c1_max_err, c1_errors);
        $display("[TB_CONV_ENGINE] Conv2: %0d outputs | Cycles: %0d | Max Abs Err: %0d | Mismatches: %0d",
                 c2_cnt, c2_cycles, c2_max_err, c2_errors);
        if (c1_errors == 0 && c2_errors == 0)
            $display("[TB_CONV_ENGINE] STATUS: PASS");
        else
            $display("[TB_CONV_ENGINE] STATUS: FAIL (c1_err=%0d, c2_err=%0d)", c1_errors, c2_errors);
        $display("=================================================================");
        $finish;
    end

endmodule
