# Synopsys Design Constraints (SDC) for Terasic DE10-Standard (50 MHz CLOCK_50)
create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]
derive_clock_uncertainty
set_false_path -from [get_ports {KEY[*] SW[*] PS2_CLK PS2_DAT}]
set_false_path -to [get_ports {LEDR[*] HEX0[*] HEX1[*] HEX2[*] HEX3[*] HEX4[*] HEX5[*] PS2_CLK PS2_DAT VGA_CLK VGA_HS VGA_VS VGA_BLANK_N VGA_SYNC_N VGA_R[*] VGA_G[*] VGA_B[*]}]
