# ModelSim .do script: Compile and run tb_conv_engine
if {![file exists work]} { vlib work }
vlog -work work rtl/conv_engine.v sim/tb/tb_conv_engine.v
vsim -c -t 1ps work.tb_conv_engine -do "onfinish stop; run -all; quit -f"
