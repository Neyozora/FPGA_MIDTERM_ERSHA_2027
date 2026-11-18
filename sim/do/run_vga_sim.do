# ModelSim Simulation Script for VGA Monitor HUD (`rtl/vga_cnn_hud.v`)
# Run in ModelSim GUI:   do sim/do/run_vga_sim.do
# Run headless CLI:      vsim -c -do "do sim/do/run_vga_sim.do; quit -f"

if {![file exists work]} {
    vlib work
}

vlog -work work rtl/vga_cnn_hud.v sim/tb/tb_vga_cnn_hud.v
vsim -t 1ps work.tb_vga_cnn_hud
onfinish stop

# Add VGA signals to Wave window if running in GUI
if {[batch_mode] == 0} {
    add wave -divider "CLOCK & RESET"
    add wave -color Yellow /tb_vga_cnn_hud/clk_50
    add wave -color Cyan   /tb_vga_cnn_hud/VGA_CLK
    add wave -color Red    /tb_vga_cnn_hud/rst

    add wave -divider "VGA SYNC SIGNALS (640x480 @ 60Hz)"
    add wave -color Green  /tb_vga_cnn_hud/VGA_HS
    add wave -color Green  /tb_vga_cnn_hud/VGA_VS
    add wave -color Orange /tb_vga_cnn_hud/VGA_BLANK_N

    add wave -divider "VGA RGB PIXEL OUTPUT (24-bit)"
    add wave -radix unsigned -color Red   /tb_vga_cnn_hud/VGA_R
    add wave -radix unsigned -color Green /tb_vga_cnn_hud/VGA_G
    add wave -radix unsigned -color Blue  /tb_vga_cnn_hud/VGA_B

    add wave -divider "CNN INPUT & WINNER"
    add wave -radix unsigned /tb_vga_cnn_hud/vga_img_rd_addr
    add wave -radix unsigned /tb_vga_cnn_hud/vga_img_rd_data
    add wave -radix unsigned /tb_vga_cnn_hud/predicted_class
}

run -all
wave zoom full
