# ModelSim .do script: Compile and run tb_maxpool
if {![file exists work]} { vlib work }
vlog -work work rtl/maxpool.v sim/tb/tb_maxpool.v
vsim -c -t 1ps work.tb_maxpool -do "onfinish stop; run -all; quit -f"
