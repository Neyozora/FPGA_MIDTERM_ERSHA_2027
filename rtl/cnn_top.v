// =============================================================================
// Module: cnn_top.v
// Description:
//   Top-level Two-Layer CNN Image Classification Accelerator for the Terasic
//   DE10-Standard (Intel Cyclone V SoC FPGA).
//
//   Integrates all processing modules and on-chip M10K memories under a single
//   deterministic FSM controller:
//     1. Input loading (`img_ram`, 1024 x 8b = 1 M10K)
//     2. Conv1 (`conv_engine.v`, reused: 1x32x32 -> 8x30x30)
//     3. ReLU1 (`relu.v`, reused -> `conv1_relu_ram`, 7200 x 8b)
//     4. MaxPool (`maxpool.v`: 8x30x30 -> 8x15x15 -> `pool1_ram`, 1800 x 8b)
//     5. Conv2 (`conv_engine.v`, reused: 8x15x15 -> 16x13x13)
//     6. ReLU2 (`relu.v`, reused -> `conv2_relu_ram`, 2704 x 8b)
//     7. Global Average Pooling (`gap.v`: 16x13x13 -> 16 -> `gap_ram`, 16 x 8b)
//     8. Fully Connected (`fully_connected.v`: 16 -> 10 -> `fc_score_ram`, 10 x 8b)
//     9. Argmax (`argmax.v`: 10 -> `predicted_class` 0..9)
//    10. Completion signal (`done`)
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module cnn_top #(
    parameter CONV1_W_MEM_FILE  = "mem/conv1_weights.mem",
    parameter CONV1_B_MEM_FILE  = "mem/conv1_bias.mem",
    parameter CONV2_W_MEM_FILE  = "mem/conv2_weights.mem",
    parameter CONV2_B_MEM_FILE  = "mem/conv2_bias.mem",
    parameter FC_W_MEM_FILE     = "mem/fc_weights.mem",
    parameter FC_B_MEM_FILE     = "mem/fc_bias.mem",
    parameter QPARAMS_MEM_FILE  = "mem/quantization_params.mem",
    parameter TEST_IMG_MEM_FILE = "mem/test_image.mem",

    // Default dyadic quantization parameters (matching quantization_params.mem)
    parameter [15:0] CONV1_MULT  = 16'd23202,
    parameter [5:0]  CONV1_SHIFT = 6'd19,
    parameter [15:0] CONV2_MULT  = 16'd16868,
    parameter [5:0]  CONV2_SHIFT = 6'd20,
    parameter [15:0] GAP_MULT    = 16'd16780,
    parameter [5:0]  GAP_SHIFT   = 6'd19,
    parameter [15:0] FC_MULT     = 16'd23793,
    parameter [5:0]  FC_SHIFT    = 6'd21
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    output reg               done,
    output reg  [3:0]        predicted_class,

    // Optional external image write port (for loading new 32x32 images at runtime
    // or during multi-image ModelSim verification; tie `ext_img_wr_en = 1'b0` to
    // classify the pre-initialized `test_image.mem` ROM/BRAM image).
    input  wire              ext_img_wr_en,
    input  wire [9:0]        ext_img_wr_addr,
    input  wire signed [7:0] ext_img_wr_data,

    // Optional hardware debug/telemetry port exposing the 10 INT8 class scores
    // (10 x 8-bit = 80 bits) for JTAG In-System Probes & board display
    output reg  [79:0]       dbg_fc_scores_flat
);

    // -------------------------------------------------------------------------
    // Top-level FSM states
    // -------------------------------------------------------------------------
    localparam [3:0]
        S_IDLE        = 4'd0,
        S_LOAD_IMG    = 4'd1,
        S_CONV1_START = 4'd2,
        S_CONV1_RUN   = 4'd3,
        S_RELU1_DRAIN = 4'd4,
        S_POOL1_START = 4'd5,
        S_POOL1_RUN   = 4'd6,
        S_CONV2_START = 4'd7,
        S_CONV2_RUN   = 4'd8,
        S_RELU2_DRAIN = 4'd9,
        S_GAP_START   = 4'd10,
        S_GAP_RUN     = 4'd11,
        S_FC_START    = 4'd12,
        S_FC_RUN      = 4'd13,
        S_ARGMAX_START= 4'd14,
        S_ARGMAX_RUN  = 4'd15;

    reg [3:0] state;
    reg       is_conv2;  // 0 = Conv1+ReLU1 active, 1 = Conv2+ReLU2 active

    // -------------------------------------------------------------------------
    // On-chip Weight & Bias ROMs and Feature-Map M10K BRAMs
    // -------------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg signed [7:0]  img_ram        [0:1023];
    (* ramstyle = "M10K" *) reg signed [7:0]  conv1_w_rom    [0:71];
    (* ramstyle = "MLAB" *) reg signed [31:0] conv1_b_rom    [0:7];
    (* ramstyle = "M10K" *) reg signed [7:0]  conv2_w_rom    [0:1151];
    (* ramstyle = "MLAB" *) reg signed [31:0] conv2_b_rom    [0:15];
    (* ramstyle = "M10K" *) reg signed [7:0]  fc_w_rom       [0:159];
    (* ramstyle = "MLAB" *) reg signed [31:0] fc_b_rom       [0:9];

    (* ramstyle = "M10K" *) reg signed [7:0]  conv1_relu_ram [0:7199];
    (* ramstyle = "M10K" *) reg signed [7:0]  pool1_ram      [0:1799];
    (* ramstyle = "M10K" *) reg signed [7:0]  conv2_relu_ram [0:2703];
    (* ramstyle = "MLAB" *) reg signed [7:0]  gap_ram        [0:15];
    (* ramstyle = "MLAB" *) reg signed [7:0]  fc_score_ram   [0:9];

    initial begin
        $readmemh(TEST_IMG_MEM_FILE, img_ram);
        $readmemh(CONV1_W_MEM_FILE,  conv1_w_rom);
        $readmemh(CONV1_B_MEM_FILE,  conv1_b_rom);
        $readmemh(CONV2_W_MEM_FILE,  conv2_w_rom);
        $readmemh(CONV2_B_MEM_FILE,  conv2_b_rom);
        $readmemh(FC_W_MEM_FILE,     fc_w_rom);
        $readmemh(FC_B_MEM_FILE,     fc_b_rom);
    end

    // -------------------------------------------------------------------------
    // Interconnect wires for processing modules
    // -------------------------------------------------------------------------
    // Shared conv_engine wires
    reg                conv_start;
    wire               conv_busy;
    wire               conv_done;
    wire [10:0]        conv_in_rd_addr;
    wire signed [7:0]  conv_in_rd_data;
    wire [10:0]        conv_w_rd_addr;
    wire signed [7:0]  conv_w_rd_data;
    wire [3:0]         conv_b_rd_addr;
    wire signed [31:0] conv_b_rd_data;
    wire               conv_out_valid;
    wire [12:0]        conv_out_addr;
    wire [4:0]         conv_out_ch;
    wire [5:0]         conv_out_y;
    wire [5:0]         conv_out_x;
    wire signed [31:0] conv_out_acc_int32;
    wire signed [7:0]  conv_out_req_int8;

    // Shared relu.v wires
    wire signed [7:0]  relu_out_comb;
    wire               relu_out_valid;
    wire [12:0]        relu_out_addr;
    wire signed [7:0]  relu_out_data;

    // MaxPool wires
    reg                pool_start;
    wire               pool_busy;
    wire               pool_done;
    wire [12:0]        pool_in_rd_addr;
    reg  signed [7:0]  pool_in_rd_data;
    wire               pool_out_valid;
    wire [10:0]        pool_out_addr;
    wire signed [7:0]  pool_out_data;

    // GAP wires
    reg                gap_start;
    wire               gap_busy;
    wire               gap_done;
    wire [11:0]        gap_in_rd_addr;
    reg  signed [7:0]  gap_in_rd_data;
    wire               gap_out_valid;
    wire [3:0]         gap_out_ch;
    wire signed [31:0] gap_out_sum_int32;
    wire signed [7:0]  gap_out_data;

    // Fully Connected wires
    reg                fc_start;
    wire               fc_busy;
    wire               fc_done;
    wire [3:0]         fc_in_rd_addr;
    reg  signed [7:0]  fc_in_rd_data;
    wire [7:0]         fc_w_rd_addr;
    reg  signed [7:0]  fc_w_rd_data;
    wire [3:0]         fc_b_rd_addr;
    reg  signed [31:0] fc_b_rd_data;
    wire               fc_out_valid;
    wire [3:0]         fc_out_idx;
    wire signed [31:0] fc_out_acc_int32;
    wire signed [7:0]  fc_out_data;

    // Argmax wires
    reg                argmax_start;
    wire               argmax_busy;
    wire               argmax_done;
    wire [3:0]         argmax_in_rd_addr;
    reg  signed [7:0]  argmax_in_rd_data;
    wire [3:0]         argmax_pred_class;
    wire signed [7:0]  argmax_max_score;

    // -------------------------------------------------------------------------
    // Synchronous 1-Cycle BRAM / ROM Read & Write Blocks
    // -------------------------------------------------------------------------
    reg signed [7:0]  img_rd_data_reg;
    reg signed [7:0]  pool1_rd_data_reg;
    reg signed [7:0]  conv1_w_rd_reg;
    reg signed [7:0]  conv2_w_rd_reg;
    reg signed [31:0] conv1_b_rd_reg;
    reg signed [31:0] conv2_b_rd_reg;

    // 1. Input image BRAM (1024 x 8b)
    always @(posedge clk) begin
        if (ext_img_wr_en) begin
            img_ram[ext_img_wr_addr] <= ext_img_wr_data;
        end
        img_rd_data_reg <= img_ram[conv_in_rd_addr[9:0]];
    end

    // 2. Conv1 & Conv2 Weight / Bias ROMs
    always @(posedge clk) begin
        conv1_w_rd_reg <= conv1_w_rom[conv_w_rd_addr[6:0]];
        conv2_w_rd_reg <= conv2_w_rom[conv_w_rd_addr[10:0]];
        conv1_b_rd_reg <= conv1_b_rom[conv_b_rd_addr[2:0]];
        conv2_b_rd_reg <= conv2_b_rom[conv_b_rd_addr[3:0]];
    end

    // 3. Conv1+ReLU1 output BRAM (7200 x 8b) -> read by MaxPool
    always @(posedge clk) begin
        if (!is_conv2 && relu_out_valid) begin
            conv1_relu_ram[relu_out_addr] <= relu_out_data;
        end
        pool_in_rd_data <= conv1_relu_ram[pool_in_rd_addr];
    end

    // 4. MaxPool1 output BRAM (1800 x 8b) -> read by Conv2
    always @(posedge clk) begin
        if (pool_out_valid) begin
            pool1_ram[pool_out_addr] <= pool_out_data;
        end
        pool1_rd_data_reg <= pool1_ram[conv_in_rd_addr];
    end

    // 5. Conv2+ReLU2 output BRAM (2704 x 8b) -> read by GAP
    always @(posedge clk) begin
        if (is_conv2 && relu_out_valid) begin
            conv2_relu_ram[relu_out_addr[11:0]] <= relu_out_data;
        end
        gap_in_rd_data <= conv2_relu_ram[gap_in_rd_addr];
    end

    // 6. GAP output RAM (16 x 8b) -> read by FC
    always @(posedge clk) begin
        if (gap_out_valid) begin
            gap_ram[gap_out_ch] <= gap_out_data;
        end
        fc_in_rd_data <= gap_ram[fc_in_rd_addr];
    end

    // 7. FC Weight & Bias ROMs + FC Score RAM (10 x 8b) -> read by Argmax
    always @(posedge clk) begin
        fc_w_rd_data <= fc_w_rom[fc_w_rd_addr];
        fc_b_rd_data <= fc_b_rom[fc_b_rd_addr];
        if (fc_out_valid) begin
            fc_score_ram[fc_out_idx] <= fc_out_data;
            dbg_fc_scores_flat[fc_out_idx * 8 +: 8] <= fc_out_data;
        end
        argmax_in_rd_data <= fc_score_ram[argmax_in_rd_addr];
    end

    // Multiplexed inputs to the reusable `conv_engine` instance
    assign conv_in_rd_data = is_conv2 ? pool1_rd_data_reg : img_rd_data_reg;
    assign conv_w_rd_data  = is_conv2 ? conv2_w_rd_reg    : conv1_w_rd_reg;
    assign conv_b_rd_data  = is_conv2 ? conv2_b_rd_reg    : conv1_b_rd_reg;

    wire [4:0]  cfg_conv_in_c  = is_conv2 ? 5'd8  : 5'd1;
    wire [5:0]  cfg_conv_in_h  = is_conv2 ? 6'd15 : 6'd32;
    wire [5:0]  cfg_conv_in_w  = is_conv2 ? 6'd15 : 6'd32;
    wire [4:0]  cfg_conv_out_c = is_conv2 ? 5'd16 : 5'd8;
    wire [5:0]  cfg_conv_out_h = is_conv2 ? 6'd13 : 6'd30;
    wire [5:0]  cfg_conv_out_w = is_conv2 ? 6'd13 : 6'd30;
    wire [15:0] cfg_conv_mult  = is_conv2 ? CONV2_MULT  : CONV1_MULT;
    wire [5:0]  cfg_conv_shift = is_conv2 ? CONV2_SHIFT : CONV1_SHIFT;

    // -------------------------------------------------------------------------
    // Submodule Instantiations
    // -------------------------------------------------------------------------
    conv_engine #(
        .IN_ADDR_W  (11),
        .W_ADDR_W   (11),
        .B_ADDR_W   (4),
        .OUT_ADDR_W (13)
    ) u_conv_engine (
        .clk           (clk),
        .rst           (rst),
        .start         (conv_start),
        .busy          (conv_busy),
        .done          (conv_done),
        .cfg_in_c      (cfg_conv_in_c),
        .cfg_in_h      (cfg_conv_in_h),
        .cfg_in_w      (cfg_conv_in_w),
        .cfg_out_c     (cfg_conv_out_c),
        .cfg_out_h     (cfg_conv_out_h),
        .cfg_out_w     (cfg_conv_out_w),
        .cfg_mult      (cfg_conv_mult),
        .cfg_shift     (cfg_conv_shift),
        .in_rd_addr    (conv_in_rd_addr),
        .in_rd_data    (conv_in_rd_data),
        .w_rd_addr     (conv_w_rd_addr),
        .w_rd_data     (conv_w_rd_data),
        .b_rd_addr     (conv_b_rd_addr),
        .b_rd_data     (conv_b_rd_data),
        .out_valid     (conv_out_valid),
        .out_ready     (1'b1),
        .out_addr      (conv_out_addr),
        .out_ch        (conv_out_ch),
        .out_y         (conv_out_y),
        .out_x         (conv_out_x),
        .out_acc_int32 (conv_out_acc_int32),
        .out_req_int8  (conv_out_req_int8)
    );

    relu #(
        .DATA_W (8),
        .ADDR_W (13)
    ) u_relu (
        .clk           (clk),
        .rst           (rst),
        .in_valid      (conv_out_valid),
        .in_addr       (conv_out_addr),
        .in_data       (conv_out_req_int8),
        .out_data_comb (relu_out_comb),
        .out_valid     (relu_out_valid),
        .out_addr      (relu_out_addr),
        .out_data      (relu_out_data)
    );

    maxpool #(
        .CHANNELS   (8),
        .IN_H       (30),
        .IN_W       (30),
        .OUT_H      (15),
        .OUT_W      (15),
        .IN_ADDR_W  (13),
        .OUT_ADDR_W (11)
    ) u_maxpool (
        .clk        (clk),
        .rst        (rst),
        .start      (pool_start),
        .busy       (pool_busy),
        .done       (pool_done),
        .in_rd_addr (pool_in_rd_addr),
        .in_rd_data (pool_in_rd_data),
        .out_valid  (pool_out_valid),
        .out_addr   (pool_out_addr),
        .out_data   (pool_out_data)
    );

    gap #(
        .CHANNELS      (16),
        .SPATIAL_N     (169),
        .IN_ADDR_W     (12),
        .CH_W          (4),
        .DEFAULT_MULT  (GAP_MULT),
        .DEFAULT_SHIFT (GAP_SHIFT)
    ) u_gap (
        .clk           (clk),
        .rst           (rst),
        .start         (gap_start),
        .busy          (gap_busy),
        .done          (gap_done),
        .cfg_mult      (GAP_MULT),
        .cfg_shift     (GAP_SHIFT),
        .in_rd_addr    (gap_in_rd_addr),
        .in_rd_data    (gap_in_rd_data),
        .out_valid     (gap_out_valid),
        .out_ch        (gap_out_ch),
        .out_sum_int32 (gap_out_sum_int32),
        .out_data      (gap_out_data)
    );

    fully_connected #(
        .IN_FEATURES   (16),
        .OUT_CLASSES   (10),
        .DEFAULT_MULT  (FC_MULT),
        .DEFAULT_SHIFT (FC_SHIFT)
    ) u_fc (
        .clk           (clk),
        .rst           (rst),
        .start         (fc_start),
        .busy          (fc_busy),
        .done          (fc_done),
        .cfg_mult      (FC_MULT),
        .cfg_shift     (FC_SHIFT),
        .in_rd_addr    (fc_in_rd_addr),
        .in_rd_data    (fc_in_rd_data),
        .w_rd_addr     (fc_w_rd_addr),
        .w_rd_data     (fc_w_rd_data),
        .b_rd_addr     (fc_b_rd_addr),
        .b_rd_data     (fc_b_rd_data),
        .out_valid     (fc_out_valid),
        .out_idx       (fc_out_idx),
        .out_acc_int32 (fc_out_acc_int32),
        .out_data      (fc_out_data)
    );

    argmax #(
        .NUM_CLASSES (10),
        .IDX_W       (4)
    ) u_argmax (
        .clk             (clk),
        .rst             (rst),
        .start           (argmax_start),
        .busy            (argmax_busy),
        .done            (argmax_done),
        .in_rd_addr      (argmax_in_rd_addr),
        .in_rd_data      (argmax_in_rd_data),
        .predicted_class (argmax_pred_class),
        .max_score       (argmax_max_score)
    );

    // -------------------------------------------------------------------------
    // Top-level FSM Controller
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state           <= S_IDLE;
            is_conv2        <= 1'b0;
            conv_start      <= 1'b0;
            pool_start      <= 1'b0;
            gap_start       <= 1'b0;
            fc_start        <= 1'b0;
            argmax_start    <= 1'b0;
            done            <= 1'b0;
            predicted_class <= 4'd0;
        end else begin
            conv_start   <= 1'b0;
            pool_start   <= 1'b0;
            gap_start    <= 1'b0;
            fc_start     <= 1'b0;
            argmax_start <= 1'b0;
            done         <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        is_conv2 <= 1'b0;
                        state    <= S_LOAD_IMG;
                    end
                end

                S_LOAD_IMG: begin
                    // Image RAM is loaded; trigger Conv1
                    is_conv2   <= 1'b0;
                    conv_start <= 1'b1;
                    state      <= S_CONV1_RUN;
                end

                S_CONV1_RUN: begin
                    if (conv_done) begin
                        state <= S_RELU1_DRAIN;
                    end
                end

                S_RELU1_DRAIN: begin
                    // Wait 1 cycle for the last registered ReLU1 write into conv1_relu_ram
                    pool_start <= 1'b1;
                    state      <= S_POOL1_RUN;
                end

                S_POOL1_RUN: begin
                    if (pool_done) begin
                        state <= S_CONV2_START;
                    end
                end

                S_CONV2_START: begin
                    is_conv2   <= 1'b1;
                    conv_start <= 1'b1;
                    state      <= S_CONV2_RUN;
                end

                S_CONV2_RUN: begin
                    if (conv_done) begin
                        state <= S_RELU2_DRAIN;
                    end
                end

                S_RELU2_DRAIN: begin
                    // Wait 1 cycle for the last registered ReLU2 write into conv2_relu_ram
                    gap_start <= 1'b1;
                    state     <= S_GAP_RUN;
                end

                S_GAP_RUN: begin
                    if (gap_done) begin
                        state <= S_FC_START;
                    end
                end

                S_FC_START: begin
                    fc_start <= 1'b1;
                    state    <= S_FC_RUN;
                end

                S_FC_RUN: begin
                    if (fc_done) begin
                        state <= S_ARGMAX_START;
                    end
                end

                S_ARGMAX_START: begin
                    argmax_start <= 1'b1;
                    state        <= S_ARGMAX_RUN;
                end

                S_ARGMAX_RUN: begin
                    if (argmax_done) begin
                        predicted_class <= argmax_pred_class;
                        done            <= 1'b1;
                        state           <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
