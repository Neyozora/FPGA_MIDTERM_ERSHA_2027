// =============================================================================
// Module: ps2_mouse_ctrl.v
// Description:
//   Bidirectional PS/2 Mouse Controller for Terasic DE10-Standard (50 MHz).
//   - Automatically transmits 0xF4 ("Enable Data Reporting") to the PS/2 mouse
//     after reset (and upon hot-plug BAT 0xAA 0x00 detection).
//   - Decodes 3-byte PS/2 movement & button packets with bit[3] frame sync.
//   - Outputs smooth 8-bit canvas coordinates `mouse_x` (0..255), `mouse_y`
//     (0..255), 5-bit 32x32 grid coordinates `grid_x` (0..31), `grid_y` (0..31),
//     and debounced button states (`btn_left`, `btn_right`, `btn_middle`).
//
// Standard: Verilog-2001 (synthesizable)
// =============================================================================
`timescale 1ns / 1ps

module ps2_mouse_ctrl (
    input  wire       clk_50,
    input  wire       rst,
    inout  wire       PS2_CLK,
    inout  wire       PS2_DAT,

    output reg  [7:0] mouse_x,      // 0..255 (256x256 VGA canvas coordinate)
    output reg  [7:0] mouse_y,      // 0..255 (256x256 VGA canvas coordinate)
    output wire [4:0] grid_x,       // 0..31  (32x32 CNN pixel X = mouse_x[7:3])
    output wire [4:0] grid_y,       // 0..31  (32x32 CNN pixel Y = mouse_y[7:3])
    output reg        btn_left,
    output reg        btn_right,
    output reg        btn_middle,
    output reg        packet_valid
);

    assign grid_x = mouse_x[7:3];
    assign grid_y = mouse_y[7:3];

    // -------------------------------------------------------------------------
    // 1. Open-Drain Tri-State Drivers & Digital Glitch Filter on PS2_CLK/DAT
    // -------------------------------------------------------------------------
    reg ps2_clk_out_low;
    reg ps2_dat_out_low;

    assign PS2_CLK = ps2_clk_out_low ? 1'b0 : 1'bz;
    assign PS2_DAT = ps2_dat_out_low ? 1'b0 : 1'bz;

    reg [7:0] clk_filter;
    reg       ps2_clk_clean, ps2_clk_prev;
    reg [2:0] dat_sync;

    always @(posedge clk_50) begin
        if (rst) begin
            clk_filter    <= 8'hFF;
            ps2_clk_clean <= 1'b1;
            ps2_clk_prev  <= 1'b1;
            dat_sync      <= 3'b111;
        end else begin
            clk_filter   <= {clk_filter[6:0], PS2_CLK};
            ps2_clk_prev <= ps2_clk_clean;
            if (clk_filter == 8'hFF)      ps2_clk_clean <= 1'b1;
            else if (clk_filter == 8'h00) ps2_clk_clean <= 1'b0;
            dat_sync <= {dat_sync[1:0], PS2_DAT};
        end
    end

    wire ps2_clk_fall = (ps2_clk_prev == 1'b1) && (ps2_clk_clean == 1'b0);
    wire ps2_dat_in   = dat_sync[2];

    // -------------------------------------------------------------------------
    // 2. Host-to-Mouse Init (Send 0xF4) & 3-Byte Packet Receiver FSM
    // -------------------------------------------------------------------------
    localparam [2:0]
        S_BOOT_WAIT = 3'd0,
        S_RTS_CLK   = 3'd1,
        S_RTS_DAT   = 3'd2,
        S_TX_BITS   = 3'd3,
        S_RX_STREAM = 3'd4;

    reg [2:0]  state;
    reg [19:0] timer;
    reg [3:0]  bit_cnt;
    reg [9:0]  tx_shift;   // {Stop=1, OddParity=0, 8'hF4}
    reg [10:0] rx_shift;
    reg [1:0]  byte_idx;
    reg [7:0]  b0_status, b1_dx;
    reg [21:0] idle_timeout;

    wire [7:0] rx_byte = rx_shift[8:1];

    // Signed movement deltas (clamped to avoid overflow jumps)
    wire signed [8:0] dx_signed = {b0_status[4], b1_dx};
    wire signed [8:0] dy_signed = {b0_status[5], rx_byte};
    wire signed [9:0] new_x = $signed({2'b00, mouse_x}) + (dx_signed >>> 1);
    wire signed [9:0] new_y = $signed({2'b00, mouse_y}) - (dy_signed >>> 1);

    always @(posedge clk_50) begin
        if (rst) begin
            state           <= S_BOOT_WAIT;
            timer           <= 20'd0;
            bit_cnt         <= 4'd0;
            tx_shift        <= 10'b1_0_11110100; // Stop=1, Parity(0xF4)=0, Data=0xF4
            rx_shift        <= 11'd0;
            byte_idx        <= 2'd0;
            b0_status       <= 8'd0;
            b1_dx           <= 8'd0;
            idle_timeout    <= 22'd0;
            ps2_clk_out_low <= 1'b0;
            ps2_dat_out_low <= 1'b0;
            mouse_x         <= 8'd128; // Start cursor in center of 256x256 canvas
            mouse_y         <= 8'd128;
            btn_left        <= 1'b0;
            btn_right       <= 1'b0;
            btn_middle      <= 1'b0;
            packet_valid    <= 1'b0;
        end else begin
            packet_valid <= 1'b0;

            case (state)
                // Wait ~15 ms after reset before sending 0xF4
                S_BOOT_WAIT: begin
                    ps2_clk_out_low <= 1'b0;
                    ps2_dat_out_low <= 1'b0;
                    if (timer == 20'd750_000) begin
                        timer <= 20'd0;
                        state <= S_RTS_CLK;
                    end else begin
                        timer <= timer + 20'd1;
                    end
                end

                // Hold PS2_CLK low for >120 us (6,500 cycles @ 50 MHz)
                S_RTS_CLK: begin
                    ps2_clk_out_low <= 1'b1;
                    ps2_dat_out_low <= 1'b0;
                    if (timer == 20'd6_500) begin
                        timer <= 20'd0;
                        state <= S_RTS_DAT;
                    end else begin
                        timer <= timer + 20'd1;
                    end
                end

                // Pull PS2_DAT low (Start bit) and release PS2_CLK
                S_RTS_DAT: begin
                    ps2_dat_out_low <= 1'b1;
                    ps2_clk_out_low <= 1'b0;
                    tx_shift        <= 10'b1_0_11110100; // 0xF4 + odd parity 0 + stop 1
                    bit_cnt         <= 4'd0;
                    state           <= S_TX_BITS;
                end

                // Shift out 8 data bits + 1 parity + 1 stop + 1 ACK on PS2_CLK falling edges
                S_TX_BITS: begin
                    if (ps2_clk_fall) begin
                        if (bit_cnt < 4'd10) begin
                            ps2_dat_out_low <= ~tx_shift[0];
                            tx_shift        <= {1'b1, tx_shift[9:1]};
                            bit_cnt         <= bit_cnt + 4'd1;
                        end else begin
                            // ACK cycle finished -> release bus and enter RX mode
                            ps2_dat_out_low <= 1'b0;
                            bit_cnt         <= 4'd0;
                            byte_idx        <= 2'd0;
                            state           <= S_RX_STREAM;
                        end
                    end
                end

                // Receive 11-bit PS/2 frames and assemble 3-byte mouse packets
                S_RX_STREAM: begin
                    if (ps2_clk_fall) begin
                        idle_timeout <= 22'd0;
                        rx_shift     <= {ps2_dat_in, rx_shift[10:1]};
                        if (bit_cnt == 4'd10) begin
                            bit_cnt <= 4'd0;
                            // Check if mouse was just hot-plugged and sent 0x00 (Mouse ID after 0xAA)
                            if (byte_idx == 2'd0 && ({ps2_dat_in, rx_shift[10:1]} & 11'h1FE) == 11'h000) begin
                                timer <= 20'd0;
                                state <= S_BOOT_WAIT;
                            end else if (byte_idx == 2'd0) begin
                                // Byte 0 of mouse packet must have bit[3] == 1 and not be 0xFA (ACK)
                                if (rx_shift[4] == 1'b1 && rx_shift[8:1] != 8'hFA) begin
                                    b0_status <= rx_shift[8:1];
                                    byte_idx  <= 2'd1;
                                end
                            end else if (byte_idx == 2'd1) begin
                                b1_dx    <= rx_shift[8:1];
                                byte_idx <= 2'd2;
                            end else begin
                                // Byte 2 (dY) complete! Update buttons & cursor
                                byte_idx     <= 2'd0;
                                btn_left     <= b0_status[0];
                                btn_right    <= b0_status[1];
                                btn_middle   <= b0_status[2];
                                packet_valid <= 1'b1;

                                if (!b0_status[6]) begin
                                    if (new_x < 10'sd0)        mouse_x <= 8'd0;
                                    else if (new_x > 10'sd255) mouse_x <= 8'd255;
                                    else                       mouse_x <= new_x[7:0];
                                end
                                if (!b0_status[7]) begin
                                    if (new_y < 10'sd0)        mouse_y <= 8'd0;
                                    else if (new_y > 10'sd255) mouse_y <= 8'd255;
                                    else                       mouse_y <= new_y[7:0];
                                end
                            end
                        end else begin
                            bit_cnt <= bit_cnt + 4'd1;
                        end
                    end else begin
                        // Reset bit alignment if PS/2 clock is idle for >2 ms (100,000 cycles)
                        if (idle_timeout < 22'd100_000) begin
                            idle_timeout <= idle_timeout + 22'd1;
                        end else begin
                            bit_cnt <= 4'd0;
                        end
                    end
                end

                default: state <= S_BOOT_WAIT;
            endcase
        end
    end

endmodule
