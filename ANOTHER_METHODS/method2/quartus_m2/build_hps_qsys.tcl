package require -exact qsys 18.1

create_system hps_cnn_qsys

set_project_property DEVICE_FAMILY {Cyclone V}
set_project_property DEVICE {5CSXFC6D6F31C6}

add_instance clk_0 clock_source
set_instance_parameter_value clk_0 {clockFrequency} {50000000}
set_instance_parameter_value clk_0 {resetSynchronousEdges} {NONE}

add_instance hps_0 altera_hps
set_instance_parameter_value hps_0 {F2S_Width} {0}
set_instance_parameter_value hps_0 {S2F_Width} {0}
set_instance_parameter_value hps_0 {LWH2F_Enable} {true}
set_instance_parameter_value hps_0 {MPU_EVENTS_Enable} {false}
set_instance_parameter_value hps_0 {HPS_PROTOCOL} {DDR3}
set_instance_parameter_value hps_0 {MEM_VENDOR} {Micron}
set_instance_parameter_value hps_0 {MEM_FORMAT} {DISCRETE}
set_instance_parameter_value hps_0 {MEM_CLK_FREQ} {400.0}
set_instance_parameter_value hps_0 {MEM_DQ_WIDTH} {32}
set_instance_parameter_value hps_0 {MEM_ROW_ADDR_WIDTH} {15}
set_instance_parameter_value hps_0 {MEM_COL_ADDR_WIDTH} {10}
set_instance_parameter_value hps_0 {MEM_BANKADDR_WIDTH} {3}
set_instance_parameter_value hps_0 {MEM_IF_DM_PINS_EN} {true}

add_instance mm_bridge_0 altera_avalon_mm_bridge
set_instance_parameter_value mm_bridge_0 {DATA_WIDTH} {32}
set_instance_parameter_value mm_bridge_0 {SYMBOL_WIDTH} {8}
set_instance_parameter_value mm_bridge_0 {ADDRESS_WIDTH} {8}
set_instance_parameter_value mm_bridge_0 {ADDRESS_UNITS} {SYMBOLS}

add_connection clk_0.clk hps_0.h2f_lw_axi_clock
add_connection clk_0.clk hps_0.f2h_sdram0_clock
add_connection clk_0.clk mm_bridge_0.clk
add_connection clk_0.clk_reset mm_bridge_0.reset
add_connection hps_0.h2f_lw_axi_master mm_bridge_0.s0
set_connection_parameter_value hps_0.h2f_lw_axi_master/mm_bridge_0.s0 baseAddress {0x0000}

add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_0.clk_in
add_interface reset reset sink
set_interface_property reset EXPORT_OF clk_0.clk_in_reset
add_interface memory conduit end
set_interface_property memory EXPORT_OF hps_0.memory
add_interface cnn_avm avalon master
set_interface_property cnn_avm EXPORT_OF mm_bridge_0.m0

save_system hps_cnn_qsys.qsys
