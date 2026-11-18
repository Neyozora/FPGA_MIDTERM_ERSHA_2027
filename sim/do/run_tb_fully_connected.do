# ModelSim .do script: Compile and run tb_fully_connected
if {![file exists work]} { vlib work }
vlog -work work rtl/fully_connected.v sim/tb/tb_fully_connected.v
vsim -c -t 1ps work.tb_fully_connected -do "onfinish stop; run -all; quit -f"
