// =============================================================================
// Module: gap_fc_argmax_m2.v
// Description:
//   Method 2 Global Average Pooling (8x8 -> >> 6, 0 DSPs!) +
//   Fully Connected (64 -> 10) + Per-Channel Dyadic Requantization + Argmax.
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module gap_fc_argmax_m2 #(
    parameter FC_W_FILE = "mem_m2/fc_weights_m2.mem",
    parameter FC_Q_FILE = "mem_m2/fc_qparams_m2.mem"
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,
    output reg  [3:0]  predicted_class,
    output reg  [79:0] dbg_fc_scores_flat,

    // Read port to Conv4 output (fmap_b[0..3], 16 groups * 64 pixels = 1024 words)
    output reg  [11:0] fmap_rd_addr,
    input  wire [7:0]  fmap_rd_data0,
    input  wire [7:0]  fmap_rd_data1,
    input  wire [7:0]  fmap_rd_data2,
    input  wire [7:0]  fmap_rd_data3
);

    (* ramstyle = "M10K" *) reg signed [7:0] fc_w_rom [0:639];
    (* ramstyle = "MLAB" *) reg [52:0]       fc_q_rom [0:9];

    initial begin
        $readmemh(FC_W_FILE, fc_w_rom);
        $readmemh(FC_Q_FILE, fc_q_rom);
    end

    reg signed [7:0] gap_vec [0:63];

    localparam [2:0]
        ST_IDLE     = 3'd0,
        ST_GAP_RD   = 3'd1,
        ST_GAP_WAIT = 3'd2,
        ST_GAP_ACC  = 3'd3,
        ST_FC_INIT  = 3'd4,
        ST_FC_MAC   = 3'd5,
        ST_FC_REQ   = 3'd6,
        ST_DONE     = 3'd7;

    reg [2:0]  state;
    reg [3:0]  og;
    reg [6:0]  pix;
    reg signed [15:0] sum0, sum1, sum2, sum3;

    reg [3:0]  fc_oc;
    reg [6:0]  fc_ic;
    reg signed [31:0] fc_acc;
    reg [52:0] cur_fq;

    reg signed [7:0] best_score;
    reg [3:0]        best_cls;

    function signed [7:0] clamp_gap;
        input signed [15:0] s;
        reg signed [15:0] q;
        begin
            q = (s + 16'sd32) >>> 6;
            if (q <= 16'sd0)       clamp_gap = 8'sd0;
            else if (q >= 16'sd127) clamp_gap = 8'sd127;
            else                   clamp_gap = q[7:0];
        end
    endfunction

    function signed [7:0] requant_fc;
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
            if (shf_w <= -49'sd128)     requant_fc = -8'sd128;
            else if (shf_w >= 49'sd127) requant_fc = 8'sd127;
            else                        requant_fc = shf_w[7:0];
        end
    endfunction

    reg signed [7:0] sc_val;

    always @(posedge clk) begin
        if (rst) begin
            state              <= ST_IDLE;
            done               <= 1'b0;
            predicted_class    <= 4'd0;
            dbg_fc_scores_flat <= 80'd0;
            fmap_rd_addr       <= 12'd0;
            og                 <= 4'd0;
            pix                <= 7'd0;
            fc_oc              <= 4'd0;
            fc_ic              <= 7'd0;
        end else begin
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        og    <= 4'd0;
                        pix   <= 7'd0;
                        sum0  <= 16'sd0;
                        sum1  <= 16'sd0;
                        sum2  <= 16'sd0;
                        sum3  <= 16'sd0;
                        state <= ST_GAP_RD;
                    end
                end

                // Issue read for (og, pix)
                ST_GAP_RD: begin
                    fmap_rd_addr <= {og, pix[5:0]};
                    state        <= ST_GAP_WAIT;
                end

                ST_GAP_WAIT: begin
                    state <= ST_GAP_ACC;
                end

                ST_GAP_ACC: begin
                    sum0 <= sum0 + $signed(fmap_rd_data0);
                    sum1 <= sum1 + $signed(fmap_rd_data1);
                    sum2 <= sum2 + $signed(fmap_rd_data2);
                    sum3 <= sum3 + $signed(fmap_rd_data3);
                    if (pix == 7'd63) begin
                        // Finalize 4 channels into gap_vec (on next cycle? No, use sum + rd_data!)
                        gap_vec[{og, 2'd0}] <= clamp_gap(sum0 + $signed(fmap_rd_data0));
                        gap_vec[{og, 2'd1}] <= clamp_gap(sum1 + $signed(fmap_rd_data1));
                        gap_vec[{og, 2'd2}] <= clamp_gap(sum2 + $signed(fmap_rd_data2));
                        gap_vec[{og, 2'd3}] <= clamp_gap(sum3 + $signed(fmap_rd_data3));
                        sum0 <= 16'sd0;
                        sum1 <= 16'sd0;
                        sum2 <= 16'sd0;
                        sum3 <= 16'sd0;
                        pix  <= 7'd0;
                        if (og == 4'd15) begin
                            fc_oc <= 4'd0;
                            state <= ST_FC_INIT;
                        end else begin
                            og    <= og + 4'd1;
                            state <= ST_GAP_RD;
                        end
                    end else begin
                        pix   <= pix + 7'd1;
                        state <= ST_GAP_RD;
                    end
                end

                ST_FC_INIT: begin
                    cur_fq <= fc_q_rom[fc_oc];
                    fc_ic  <= 7'd0;
                    fc_acc <= 32'sd0;
                    state  <= ST_FC_MAC;
                end

                ST_FC_MAC: begin
                    if (fc_ic == 7'd0)
                        fc_acc <= $signed(cur_fq[31:0]) + ($signed(gap_vec[fc_ic[5:0]]) * fc_w_rom[fc_oc * 64 + fc_ic[5:0]]);
                    else
                        fc_acc <= fc_acc + ($signed(gap_vec[fc_ic[5:0]]) * fc_w_rom[fc_oc * 64 + fc_ic[5:0]]);

                    if (fc_ic == 7'd63) begin
                        state <= ST_FC_REQ;
                    end else begin
                        fc_ic <= fc_ic + 7'd1;
                    end
                end

                ST_FC_REQ: begin
                    sc_val = requant_fc(fc_acc, cur_fq[47:32], cur_fq[52:48]);
                    dbg_fc_scores_flat[fc_oc*8 +: 8] <= sc_val;
                    if (fc_oc == 4'd0) begin
                        best_score <= sc_val;
                        best_cls   <= 4'd0;
                    end else if (sc_val > best_score) begin
                        best_score <= sc_val;
                        best_cls   <= fc_oc;
                    end

                    if (fc_oc == 4'd9) begin
                        state <= ST_DONE;
                    end else begin
                        fc_oc <= fc_oc + 4'd1;
                        state <= ST_FC_INIT;
                    end
                end

                ST_DONE: begin
                    predicted_class <= best_cls;
                    done            <= 1'b1;
                    state           <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
