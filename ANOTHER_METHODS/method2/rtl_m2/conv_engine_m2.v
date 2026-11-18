// =============================================================================
// Module: conv_engine_m2.v
// Description:
//   Method 2 Parallel Convolution Engine for CIFAR-10 4-Layer CNN:
//     - 2-Line Shift-Register Line Buffer (`lb0`, `lb1`) for 3x3 sliding window
//       with automatic zero-padding (PAD=1)
//     - 4x9 = 36 Parallel INT8 MAC Array (Po = 4 output channels in parallel)
//     - 4-bank on-chip partial-sum BRAM (`psum_ram0..3`, 1024 x 32b each)
//     - Per-channel INT32 bias + dyadic requantization + ReLU clamp [0, 127]
//
// Standard: Verilog-2001 (synthesizable in Quartus Prime 18.1)
// =============================================================================
`timescale 1ns / 1ps

module conv_engine_m2 #(
    parameter W_BANK0_FILE = "mem_m2/w_bank0.mem",
    parameter W_BANK1_FILE = "mem_m2/w_bank1.mem",
    parameter W_BANK2_FILE = "mem_m2/w_bank2.mem",
    parameter W_BANK3_FILE = "mem_m2/w_bank3.mem",
    parameter Q_BANK0_FILE = "mem_m2/q_bank0.mem",
    parameter Q_BANK1_FILE = "mem_m2/q_bank1.mem",
    parameter Q_BANK2_FILE = "mem_m2/q_bank2.mem",
    parameter Q_BANK3_FILE = "mem_m2/q_bank3.mem"
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [1:0]  layer_id,   // 0:Conv1, 1:Conv2, 2:Conv3, 3:Conv4
    output reg         done,

    // Input feature map read port (1-cycle synchronous read latency)
    output reg  [1:0]  in_rd_bank,
    output reg  [11:0] in_rd_addr,
    input  wire [7:0]  in_rd_data,

    // Output feature map write port (4 parallel banks)
    output reg         out_wr_en,
    output reg  [11:0] out_wr_addr,
    output reg  [7:0]  out_wr_data0,
    output reg  [7:0]  out_wr_data1,
    output reg  [7:0]  out_wr_data2,
    output reg  [7:0]  out_wr_data3
);

    // -------------------------------------------------------------------------
    // Weight ROMs (1676 x 72b) & Quantization/Bias ROMs (44 x 53b)
    // -------------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg [71:0] w_bank0 [0:1675];
    (* ramstyle = "M10K" *) reg [71:0] w_bank1 [0:1675];
    (* ramstyle = "M10K" *) reg [71:0] w_bank2 [0:1675];
    (* ramstyle = "M10K" *) reg [71:0] w_bank3 [0:1675];

    (* ramstyle = "MLAB" *) reg [52:0] q_bank0 [0:43];
    (* ramstyle = "MLAB" *) reg [52:0] q_bank1 [0:43];
    (* ramstyle = "MLAB" *) reg [52:0] q_bank2 [0:43];
    (* ramstyle = "MLAB" *) reg [52:0] q_bank3 [0:43];

    initial begin
        $readmemh(W_BANK0_FILE, w_bank0);
        $readmemh(W_BANK1_FILE, w_bank1);
        $readmemh(W_BANK2_FILE, w_bank2);
        $readmemh(W_BANK3_FILE, w_bank3);
        $readmemh(Q_BANK0_FILE, q_bank0);
        $readmemh(Q_BANK1_FILE, q_bank1);
        $readmemh(Q_BANK2_FILE, q_bank2);
        $readmemh(Q_BANK3_FILE, q_bank3);
    end

    // 4-bank partial-sum BRAMs (max 32x32 = 1024 pixels)
    (* ramstyle = "M10K" *) reg signed [31:0] psum_ram0 [0:1023];
    (* ramstyle = "M10K" *) reg signed [31:0] psum_ram1 [0:1023];
    (* ramstyle = "M10K" *) reg signed [31:0] psum_ram2 [0:1023];
    (* ramstyle = "M10K" *) reg signed [31:0] psum_ram3 [0:1023];

    // 2-line shift-register line buffers (max width = 32)
    reg signed [7:0] lb0 [0:31];
    reg signed [7:0] lb1 [0:31];

    // Layer configuration decode
    reg [5:0]  cfg_hw;      // H = W (32, 16, or 8)
    reg [6:0]  cfg_cin;     // 3, 16, 32, 64
    reg [4:0]  cfg_og_cnt;  // 4, 8, 16, 16
    reg [10:0] cfg_w_base;  // 0, 12, 140, 652
    reg [5:0]  cfg_q_base;  // 0, 4, 12, 28
    reg [10:0] cfg_hw2;     // H*W (1024, 256, 64)
    reg [3:0]  cfg_w_shift; // log2(W): 5, 4, 3

    always @(*) begin
        case (layer_id)
            2'd0: begin // Conv1: 3->16, 32x32
                cfg_hw = 6'd32; cfg_cin = 7'd3;  cfg_og_cnt = 5'd4;
                cfg_w_base = 11'd0;   cfg_q_base = 6'd0;  cfg_hw2 = 11'd1024; cfg_w_shift = 4'd5;
            end
            2'd1: begin // Conv2: 16->32, 16x16
                cfg_hw = 6'd16; cfg_cin = 7'd16; cfg_og_cnt = 5'd8;
                cfg_w_base = 11'd12;  cfg_q_base = 6'd4;  cfg_hw2 = 11'd256;  cfg_w_shift = 4'd4;
            end
            2'd2: begin // Conv3: 32->64, 8x8
                cfg_hw = 6'd8;  cfg_cin = 7'd32; cfg_og_cnt = 5'd16;
                cfg_w_base = 11'd140; cfg_q_base = 6'd12; cfg_hw2 = 11'd64;   cfg_w_shift = 4'd3;
            end
            default: begin // Conv4: 64->64, 8x8
                cfg_hw = 6'd8;  cfg_cin = 7'd64; cfg_og_cnt = 5'd16;
                cfg_w_base = 11'd652; cfg_q_base = 6'd28; cfg_hw2 = 11'd64;   cfg_w_shift = 4'd3;
            end
        endcase
    end

    // FSM states
    localparam [2:0]
        ST_IDLE      = 3'd0,
        ST_LOAD_PASS = 3'd1,
        ST_STREAM    = 3'd2,
        ST_DRAIN     = 3'd3,
        ST_NEXT_PASS = 3'd4,
        ST_DONE      = 3'd5;

    reg [2:0]  state;
    reg [4:0]  og;
    reg [6:0]  ic;
    reg [5:0]  r_in, c_in;
    reg [3:0]  drain_cnt;

    // Active pass weights & qparams
    reg [71:0] cur_w0, cur_w1, cur_w2, cur_w3;
    reg [52:0] cur_q0, cur_q1, cur_q2, cur_q3;

    wire signed [31:0] bias0 = $signed(cur_q0[31:0]);
    wire signed [31:0] bias1 = $signed(cur_q1[31:0]);
    wire signed [31:0] bias2 = $signed(cur_q2[31:0]);
    wire signed [31:0] bias3 = $signed(cur_q3[31:0]);

    wire [15:0] mult0 = cur_q0[47:32];
    wire [15:0] mult1 = cur_q1[47:32];
    wire [15:0] mult2 = cur_q2[47:32];
    wire [15:0] mult3 = cur_q3[47:32];

    wire [4:0]  shft0 = cur_q0[52:48];
    wire [4:0]  shft1 = cur_q1[52:48];
    wire [4:0]  shft2 = cur_q2[52:48];
    wire [4:0]  shft3 = cur_q3[52:48];

    // Pipeline registers
    reg        s0_valid;
    reg [5:0]  s0_r, s0_c;
    reg        s1_valid;
    reg [5:0]  s1_r, s1_c;

    reg [23:0] col0, col1, col2; // {row2[7:0], row1[7:0], row0[7:0]}
    reg        s2_valid;
    reg [9:0]  s2_pix_idx;
    reg        s2_is_last_ic;
    reg        s2_is_first_ic;

    reg signed [31:0] psum_rd0, psum_rd1, psum_rd2, psum_rd3;

    // Stage 2 -> Stage 3: 4x9 products
    reg        s3_valid;
    reg [9:0]  s3_pix_idx;
    reg        s3_is_last_ic;
    reg        s3_is_first_ic;
    reg signed [31:0] s3_psum_old0, s3_psum_old1, s3_psum_old2, s3_psum_old3;
    reg signed [15:0] prod0 [0:8];
    reg signed [15:0] prod1 [0:8];
    reg signed [15:0] prod2 [0:8];
    reg signed [15:0] prod3 [0:8];

    // Stage 3 -> Stage 4: accumulated + biased
    reg        s4_valid;
    reg [9:0]  s4_pix_idx;
    reg signed [31:0] s4_acc0, s4_acc1, s4_acc2, s4_acc3;

    // Combinational window pixels from col0, col1, col2
    wire signed [7:0] win [0:8];
    assign win[0] = $signed(col0[7:0]);
    assign win[1] = $signed(col1[7:0]);
    assign win[2] = $signed(col2[7:0]);
    assign win[3] = $signed(col0[15:8]);
    assign win[4] = $signed(col1[15:8]);
    assign win[5] = $signed(col2[15:8]);
    assign win[6] = $signed(col0[23:16]);
    assign win[7] = $signed(col1[23:16]);
    assign win[8] = $signed(col2[23:16]);

    // Helper function: dyadic requant + ReLU [0, 127]
    function [7:0] requant_relu;
        input signed [31:0] acc_val;
        input [15:0]        m_val;
        input [4:0]         s_val;
        reg signed [48:0]   prod_w;
        reg signed [48:0]   rnd_w;
        reg signed [48:0]   shf_w;
        begin
            prod_w = acc_val * $signed({1'b0, m_val});
            rnd_w  = prod_w + ($signed(49'd1) << (s_val - 5'd1));
            shf_w  = rnd_w >>> s_val;
            if (shf_w <= 49'sd0)
                requant_relu = 8'd0;
            else if (shf_w >= 49'sd127)
                requant_relu = 8'd127;
            else
                requant_relu = shf_w[7:0];
        end
    endfunction

    integer k_i;
    reg signed [7:0] row0_v, row1_v, row2_v;
    reg [9:0]        s1_pix_idx;
    wire signed [31:0] sum0 = s3_psum_old0 + prod0[0] + prod0[1] + prod0[2] + prod0[3] + prod0[4] + prod0[5] + prod0[6] + prod0[7] + prod0[8];
    wire signed [31:0] sum1 = s3_psum_old1 + prod1[0] + prod1[1] + prod1[2] + prod1[3] + prod1[4] + prod1[5] + prod1[6] + prod1[7] + prod1[8];
    wire signed [31:0] sum2 = s3_psum_old2 + prod2[0] + prod2[1] + prod2[2] + prod2[3] + prod2[4] + prod2[5] + prod2[6] + prod2[7] + prod2[8];
    wire signed [31:0] sum3 = s3_psum_old3 + prod3[0] + prod3[1] + prod3[2] + prod3[3] + prod3[4] + prod3[5] + prod3[6] + prod3[7] + prod3[8];
    reg signed [31:0] p0_old, p1_old, p2_old, p3_old;

    wire [9:0] psum_rd_addr = ((s1_r - 6'd1) << cfg_w_shift) + (s1_c - 6'd1);

    always @(posedge clk) begin
        // Unconditional synchronous BRAM read/write for psum_ram0..3 -> M10K inference
        psum_rd0 <= psum_ram0[psum_rd_addr];
        psum_rd1 <= psum_ram1[psum_rd_addr];
        psum_rd2 <= psum_ram2[psum_rd_addr];
        psum_rd3 <= psum_ram3[psum_rd_addr];
        if (s3_valid && !s3_is_last_ic) begin
            psum_ram0[s3_pix_idx] <= sum0;
            psum_ram1[s3_pix_idx] <= sum1;
            psum_ram2[s3_pix_idx] <= sum2;
            psum_ram3[s3_pix_idx] <= sum3;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state         <= ST_IDLE;
            done          <= 1'b0;
            og            <= 5'd0;
            ic            <= 7'd0;
            r_in          <= 6'd0;
            c_in          <= 6'd0;
            drain_cnt     <= 4'd0;
            in_rd_bank    <= 2'd0;
            in_rd_addr    <= 12'd0;
            out_wr_en     <= 1'b0;
            out_wr_addr   <= 12'd0;
            out_wr_data0  <= 8'd0;
            out_wr_data1  <= 8'd0;
            out_wr_data2  <= 8'd0;
            out_wr_data3  <= 8'd0;
            s0_valid      <= 1'b0;
            s1_valid      <= 1'b0;
            s2_valid      <= 1'b0;
            s3_valid      <= 1'b0;
            s4_valid      <= 1'b0;
            col0          <= 24'd0;
            col1          <= 24'd0;
            col2          <= 24'd0;
        end else begin
            done      <= 1'b0;
            out_wr_en <= 1'b0;

            // Default pipeline valid shift
            s0_valid <= 1'b0;
            s1_valid <= s0_valid;
            s1_r     <= s0_r;
            s1_c     <= s0_c;
            s2_valid <= 1'b0;
            s3_valid <= 1'b0;
            s4_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        og    <= 5'd0;
                        ic    <= 7'd0;
                        state <= ST_LOAD_PASS;
                    end
                end

                ST_LOAD_PASS: begin
                    // Fetch weights & qparams for (og, ic)
                    cur_w0 <= w_bank0[cfg_w_base + og * cfg_cin + ic];
                    cur_w1 <= w_bank1[cfg_w_base + og * cfg_cin + ic];
                    cur_w2 <= w_bank2[cfg_w_base + og * cfg_cin + ic];
                    cur_w3 <= w_bank3[cfg_w_base + og * cfg_cin + ic];

                    cur_q0 <= q_bank0[cfg_q_base + og];
                    cur_q1 <= q_bank1[cfg_q_base + og];
                    cur_q2 <= q_bank2[cfg_q_base + og];
                    cur_q3 <= q_bank3[cfg_q_base + og];

                    r_in  <= 6'd0;
                    c_in  <= 6'd0;
                    col0  <= 24'd0;
                    col1  <= 24'd0;
                    col2  <= 24'd0;
                    state <= ST_STREAM;
                end

                ST_STREAM: begin
                    // Issue input BRAM read for (r_in, c_in)
                    in_rd_bank <= ic[1:0];
                    in_rd_addr <= (ic[6:2] * cfg_hw2) + (r_in << cfg_w_shift) + c_in;

                    s0_valid <= 1'b1;
                    s0_r     <= r_in;
                    s0_c     <= c_in;

                    if (c_in == cfg_hw) begin
                        c_in <= 6'd0;
                        if (r_in == cfg_hw) begin
                            drain_cnt <= 4'd0;
                            state     <= ST_DRAIN;
                        end else begin
                            r_in <= r_in + 6'd1;
                        end
                    end else begin
                        c_in <= c_in + 6'd1;
                    end
                end

                ST_DRAIN: begin
                    if (drain_cnt == 4'd6)
                        state <= ST_NEXT_PASS;
                    else
                        drain_cnt <= drain_cnt + 4'd1;
                end

                ST_NEXT_PASS: begin
                    if (ic + 7'd1 < cfg_cin) begin
                        ic    <= ic + 7'd1;
                        state <= ST_LOAD_PASS;
                    end else begin
                        ic <= 7'd0;
                        if (og + 5'd1 < cfg_og_cnt) begin
                            og    <= og + 5'd1;
                            state <= ST_LOAD_PASS;
                        end else begin
                            state <= ST_DONE;
                        end
                    end
                end

                ST_DONE: begin
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase

            // -----------------------------------------------------------------
            // Pipeline Stage 1: Line Buffer & Window Shift + psum_ram read
            // -----------------------------------------------------------------
            if (s1_valid) begin
                row0_v = (s1_r >= 6'd2 && s1_c < cfg_hw) ? lb0[s1_c[4:0]] : 8'sd0;
                row1_v = (s1_r >= 6'd1 && s1_c < cfg_hw) ? lb1[s1_c[4:0]] : 8'sd0;
                row2_v = (s1_r < cfg_hw && s1_c < cfg_hw) ? $signed(in_rd_data) : 8'sd0;

                if (s1_c < cfg_hw) begin
                    lb0[s1_c[4:0]] <= row1_v;
                    lb1[s1_c[4:0]] <= row2_v;
                end

                if (s1_c == 6'd0) begin
                    col0 <= 24'd0;
                    col1 <= 24'd0;
                    col2 <= {row2_v, row1_v, row0_v};
                end else begin
                    col0 <= col1;
                    col1 <= col2;
                    col2 <= {row2_v, row1_v, row0_v};
                end

                if (s1_r >= 6'd1 && s1_c >= 6'd1) begin
                    s1_pix_idx     = psum_rd_addr;
                    s2_pix_idx     <= psum_rd_addr;
                    s2_is_first_ic <= (ic == 7'd0);
                    s2_is_last_ic  <= (ic == cfg_cin - 7'd1);
                    s2_valid       <= 1'b1;
                end
            end

            // -----------------------------------------------------------------
            // Pipeline Stage 2: 4x9 Parallel INT8 Multiplications
            // -----------------------------------------------------------------
            if (s2_valid) begin
                for (k_i = 0; k_i < 9; k_i = k_i + 1) begin
                    prod0[k_i] <= win[k_i] * $signed(cur_w0[k_i*8 +: 8]);
                    prod1[k_i] <= win[k_i] * $signed(cur_w1[k_i*8 +: 8]);
                    prod2[k_i] <= win[k_i] * $signed(cur_w2[k_i*8 +: 8]);
                    prod3[k_i] <= win[k_i] * $signed(cur_w3[k_i*8 +: 8]);
                end
                s3_psum_old0   <= s2_is_first_ic ? 32'sd0 : psum_rd0;
                s3_psum_old1   <= s2_is_first_ic ? 32'sd0 : psum_rd1;
                s3_psum_old2   <= s2_is_first_ic ? 32'sd0 : psum_rd2;
                s3_psum_old3   <= s2_is_first_ic ? 32'sd0 : psum_rd3;
                s3_pix_idx     <= s2_pix_idx;
                s3_is_first_ic <= s2_is_first_ic;
                s3_is_last_ic  <= s2_is_last_ic;
                s3_valid       <= 1'b1;
            end

            // -----------------------------------------------------------------
            // Pipeline Stage 3: 9-Input Adder Tree + Accumulate / Bias
            // -----------------------------------------------------------------
            if (s3_valid) begin
                if (s3_is_last_ic) begin
                    s4_acc0    <= sum0 + bias0;
                    s4_acc1    <= sum1 + bias1;
                    s4_acc2    <= sum2 + bias2;
                    s4_acc3    <= sum3 + bias3;
                    s4_pix_idx <= s3_pix_idx;
                    s4_valid   <= 1'b1;
                end
            end

            // -----------------------------------------------------------------
            // Pipeline Stage 4: Per-Channel Dyadic Requant + ReLU -> Output BRAM
            // -----------------------------------------------------------------
            if (s4_valid) begin
                out_wr_en    <= 1'b1;
                out_wr_addr  <= (og * cfg_hw2) + s4_pix_idx;
                out_wr_data0 <= requant_relu(s4_acc0, mult0, shft0);
                out_wr_data1 <= requant_relu(s4_acc1, mult1, shft1);
                out_wr_data2 <= requant_relu(s4_acc2, mult2, shft2);
                out_wr_data3 <= requant_relu(s4_acc3, mult3, shft3);
            end
        end
    end

endmodule
