// =============================================================================
// Module: cnn_top_m2.v
// Description:
//   Method 2 Top-Level 4-Layer CNN Accelerator for CIFAR-10 (32x32x3 RGB):
//     1. img_ram (3072 x 8b) -> Conv1 (3->16, 32x32) -> fmap_a[0..3]
//     2. Pool1 (2x2 max)     -> fmap_b[0..3] (16x16x16)
//     3. Conv2 (16->32)      -> fmap_a[0..3] (32x16x16)
//     4. Pool2 (2x2 max)     -> fmap_b[0..3] (32x8x8)
//     5. Conv3 (32->64)      -> fmap_a[0..3] (64x8x8)
//     6. Conv4 (64->64)      -> fmap_b[0..3] (64x8x8)
//     7. GAP (8x8) + FC (64->10) + Argmax -> predicted_class & dbg_fc_scores_flat
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module cnn_top_m2 #(
    parameter W_BANK0_FILE      = "mem_m2/w_bank0.mem",
    parameter W_BANK1_FILE      = "mem_m2/w_bank1.mem",
    parameter W_BANK2_FILE      = "mem_m2/w_bank2.mem",
    parameter W_BANK3_FILE      = "mem_m2/w_bank3.mem",
    parameter Q_BANK0_FILE      = "mem_m2/q_bank0.mem",
    parameter Q_BANK1_FILE      = "mem_m2/q_bank1.mem",
    parameter Q_BANK2_FILE      = "mem_m2/q_bank2.mem",
    parameter Q_BANK3_FILE      = "mem_m2/q_bank3.mem",
    parameter FC_W_FILE         = "mem_m2/fc_weights_m2.mem",
    parameter FC_Q_FILE         = "mem_m2/fc_qparams_m2.mem",
    parameter TEST_IMG_MEM_FILE = "mem_m2/test_image_m2.mem"
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,
    output wire [3:0]  predicted_class,

    // External image write port (3072 x 8b)
    input  wire        ext_img_wr_en,
    input  wire [11:0] ext_img_wr_addr,
    input  wire [7:0]  ext_img_wr_data,

    // Debug 10 x 8-bit INT8 scores
    output wire [79:0] dbg_fc_scores_flat
);

    // -------------------------------------------------------------------------
    // On-Chip Memories (M10K BRAMs)
    // -------------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg [7:0] img_ram [0:3071];
    initial $readmemh(TEST_IMG_MEM_FILE, img_ram);

    (* ramstyle = "M10K" *) reg [7:0] fmap_a0 [0:4095];
    (* ramstyle = "M10K" *) reg [7:0] fmap_a1 [0:4095];
    (* ramstyle = "M10K" *) reg [7:0] fmap_a2 [0:4095];
    (* ramstyle = "M10K" *) reg [7:0] fmap_a3 [0:4095];

    (* ramstyle = "M10K" *) reg [7:0] fmap_b0 [0:2047];
    (* ramstyle = "M10K" *) reg [7:0] fmap_b1 [0:2047];
    (* ramstyle = "M10K" *) reg [7:0] fmap_b2 [0:2047];
    (* ramstyle = "M10K" *) reg [7:0] fmap_b3 [0:2047];

    // -------------------------------------------------------------------------
    // Submodule Signals
    // -------------------------------------------------------------------------
    reg        conv_start;
    reg  [1:0] conv_layer_id;
    wire       conv_done;
    wire [1:0] conv_in_bank;
    wire [11:0] conv_in_addr;
    reg  [7:0] conv_in_data;
    wire       conv_out_wr_en;
    wire [11:0] conv_out_wr_addr;
    wire [7:0] conv_out_d0, conv_out_d1, conv_out_d2, conv_out_d3;

    reg        pool_start;
    reg        pool_id;
    wire       pool_done;
    wire [11:0] pool_rd_addr;
    wire       pool_wr_en;
    wire [11:0] pool_wr_addr;
    wire [7:0] pool_wr_d0, pool_wr_d1, pool_wr_d2, pool_wr_d3;

    reg        gfa_start;
    wire       gfa_done;
    wire [11:0] gfa_rd_addr;

    // -------------------------------------------------------------------------
    // FSM Controller
    // -------------------------------------------------------------------------
    localparam [3:0]
        S_IDLE        = 4'd0,
        S_CONV1_START = 4'd1,
        S_CONV1_WAIT  = 4'd2,
        S_POOL1_START = 4'd3,
        S_POOL1_WAIT  = 4'd4,
        S_CONV2_START = 4'd5,
        S_CONV2_WAIT  = 4'd6,
        S_POOL2_START = 4'd7,
        S_POOL2_WAIT  = 4'd8,
        S_CONV3_START = 4'd9,
        S_CONV3_WAIT  = 4'd10,
        S_CONV4_START = 4'd11,
        S_CONV4_WAIT  = 4'd12,
        S_GFA_START   = 4'd13,
        S_GFA_WAIT    = 4'd14,
        S_DONE        = 4'd15;

    reg [3:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            conv_start    <= 1'b0;
            conv_layer_id <= 2'd0;
            pool_start    <= 1'b0;
            pool_id       <= 1'b0;
            gfa_start     <= 1'b0;
        end else begin
            done       <= 1'b0;
            conv_start <= 1'b0;
            pool_start <= 1'b0;
            gfa_start  <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) state <= S_CONV1_START;
                end

                S_CONV1_START: begin
                    conv_layer_id <= 2'd0;
                    conv_start    <= 1'b1;
                    state         <= S_CONV1_WAIT;
                end
                S_CONV1_WAIT: if (conv_done) state <= S_POOL1_START;

                S_POOL1_START: begin
                    pool_id    <= 1'b0;
                    pool_start <= 1'b1;
                    state      <= S_POOL1_WAIT;
                end
                S_POOL1_WAIT: if (pool_done) state <= S_CONV2_START;

                S_CONV2_START: begin
                    conv_layer_id <= 2'd1;
                    conv_start    <= 1'b1;
                    state         <= S_CONV2_WAIT;
                end
                S_CONV2_WAIT: if (conv_done) state <= S_POOL2_START;

                S_POOL2_START: begin
                    pool_id    <= 1'b1;
                    pool_start <= 1'b1;
                    state      <= S_POOL2_WAIT;
                end
                S_POOL2_WAIT: if (pool_done) state <= S_CONV3_START;

                S_CONV3_START: begin
                    conv_layer_id <= 2'd2;
                    conv_start    <= 1'b1;
                    state         <= S_CONV3_WAIT;
                end
                S_CONV3_WAIT: if (conv_done) state <= S_CONV4_START;

                S_CONV4_START: begin
                    conv_layer_id <= 2'd3;
                    conv_start    <= 1'b1;
                    state         <= S_CONV4_WAIT;
                end
                S_CONV4_WAIT: if (conv_done) state <= S_GFA_START;

                S_GFA_START: begin
                    gfa_start <= 1'b1;
                    state     <= S_GFA_WAIT;
                end
                S_GFA_WAIT: if (gfa_done) state <= S_DONE;

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Synchronous BRAM Read/Write Routing
    // -------------------------------------------------------------------------
    reg [7:0]  img_rd_q;
    reg [7:0]  fa_rd_q0, fa_rd_q1, fa_rd_q2, fa_rd_q3;
    reg [7:0]  fb_rd_q0, fb_rd_q1, fb_rd_q2, fb_rd_q3;
    reg [1:0]  conv_in_bank_d1;

    wire       fa_wr_en   = conv_out_wr_en && (conv_layer_id != 2'd3);
    wire [11:0] fa_wr_addr = conv_out_wr_addr;
    wire [11:0] fa_rd_addr = (state == S_POOL1_WAIT || state == S_POOL2_WAIT) ? pool_rd_addr : conv_in_addr;

    wire       fb_wr_en   = (conv_out_wr_en && (conv_layer_id == 2'd3)) || pool_wr_en;
    wire [11:0] fb_wr_addr = pool_wr_en ? pool_wr_addr : conv_out_wr_addr;
    wire [7:0] fb_wr_d0   = pool_wr_en ? pool_wr_d0 : conv_out_d0;
    wire [7:0] fb_wr_d1   = pool_wr_en ? pool_wr_d1 : conv_out_d1;
    wire [7:0] fb_wr_d2   = pool_wr_en ? pool_wr_d2 : conv_out_d2;
    wire [7:0] fb_wr_d3   = pool_wr_en ? pool_wr_d3 : conv_out_d3;
    wire [11:0] fb_rd_addr = (state == S_GFA_WAIT) ? gfa_rd_addr : conv_in_addr;

    always @(posedge clk) begin
        // img_ram
        if (ext_img_wr_en)
            img_ram[ext_img_wr_addr] <= ext_img_wr_data;
        img_rd_q <= img_ram[{conv_in_bank, conv_in_addr[9:0]}];

        // fmap_a0..3
        if (fa_wr_en) begin
            fmap_a0[fa_wr_addr] <= conv_out_d0;
            fmap_a1[fa_wr_addr] <= conv_out_d1;
            fmap_a2[fa_wr_addr] <= conv_out_d2;
            fmap_a3[fa_wr_addr] <= conv_out_d3;
        end
        fa_rd_q0 <= fmap_a0[fa_rd_addr];
        fa_rd_q1 <= fmap_a1[fa_rd_addr];
        fa_rd_q2 <= fmap_a2[fa_rd_addr];
        fa_rd_q3 <= fmap_a3[fa_rd_addr];

        // fmap_b0..3
        if (fb_wr_en) begin
            fmap_b0[fb_wr_addr[10:0]] <= fb_wr_d0;
            fmap_b1[fb_wr_addr[10:0]] <= fb_wr_d1;
            fmap_b2[fb_wr_addr[10:0]] <= fb_wr_d2;
            fmap_b3[fb_wr_addr[10:0]] <= fb_wr_d3;
        end
        fb_rd_q0 <= fmap_b0[fb_rd_addr[10:0]];
        fb_rd_q1 <= fmap_b1[fb_rd_addr[10:0]];
        fb_rd_q2 <= fmap_b2[fb_rd_addr[10:0]];
        fb_rd_q3 <= fmap_b3[fb_rd_addr[10:0]];

        conv_in_bank_d1 <= conv_in_bank;
    end

    always @(*) begin
        if (conv_layer_id == 2'd0) begin
            conv_in_data = img_rd_q;
        end else if (conv_layer_id == 2'd3) begin
            case (conv_in_bank_d1)
                2'd0: conv_in_data = fa_rd_q0;
                2'd1: conv_in_data = fa_rd_q1;
                2'd2: conv_in_data = fa_rd_q2;
                2'd3: conv_in_data = fa_rd_q3;
            endcase
        end else begin
            case (conv_in_bank_d1)
                2'd0: conv_in_data = fb_rd_q0;
                2'd1: conv_in_data = fb_rd_q1;
                2'd2: conv_in_data = fb_rd_q2;
                2'd3: conv_in_data = fb_rd_q3;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Submodule Instances
    // -------------------------------------------------------------------------
    conv_engine_m2 #(
        .W_BANK0_FILE(W_BANK0_FILE), .W_BANK1_FILE(W_BANK1_FILE),
        .W_BANK2_FILE(W_BANK2_FILE), .W_BANK3_FILE(W_BANK3_FILE),
        .Q_BANK0_FILE(Q_BANK0_FILE), .Q_BANK1_FILE(Q_BANK1_FILE),
        .Q_BANK2_FILE(Q_BANK2_FILE), .Q_BANK3_FILE(Q_BANK3_FILE)
    ) u_conv (
        .clk          (clk),
        .rst          (rst),
        .start        (conv_start),
        .layer_id     (conv_layer_id),
        .done         (conv_done),
        .in_rd_bank   (conv_in_bank),
        .in_rd_addr   (conv_in_addr),
        .in_rd_data   (conv_in_data),
        .out_wr_en    (conv_out_wr_en),
        .out_wr_addr  (conv_out_wr_addr),
        .out_wr_data0 (conv_out_d0),
        .out_wr_data1 (conv_out_d1),
        .out_wr_data2 (conv_out_d2),
        .out_wr_data3 (conv_out_d3)
    );

    maxpool_m2 u_pool (
        .clk      (clk),
        .rst      (rst),
        .start    (pool_start),
        .pool_id  (pool_id),
        .done     (pool_done),
        .rd_addr  (pool_rd_addr),
        .rd_data0 (fa_rd_q0),
        .rd_data1 (fa_rd_q1),
        .rd_data2 (fa_rd_q2),
        .rd_data3 (fa_rd_q3),
        .wr_en    (pool_wr_en),
        .wr_addr  (pool_wr_addr),
        .wr_data0 (pool_wr_d0),
        .wr_data1 (pool_wr_d1),
        .wr_data2 (pool_wr_d2),
        .wr_data3 (pool_wr_d3)
    );

    gap_fc_argmax_m2 #(
        .FC_W_FILE(FC_W_FILE),
        .FC_Q_FILE(FC_Q_FILE)
    ) u_gfa (
        .clk                (clk),
        .rst                (rst),
        .start              (gfa_start),
        .done               (gfa_done),
        .predicted_class    (predicted_class),
        .dbg_fc_scores_flat (dbg_fc_scores_flat),
        .fmap_rd_addr       (gfa_rd_addr),
        .fmap_rd_data0      (fb_rd_q0),
        .fmap_rd_data1      (fb_rd_q1),
        .fmap_rd_data2      (fb_rd_q2),
        .fmap_rd_data3      (fb_rd_q3)
    );

endmodule
