# ModelSim .do script: Compile and run tb_relu
if {![file exists work]} { vlib work }
vlog -work work rtl/relu.v sim/tb/tb_relu.v
vsim -c -t 1ps work.tb_relu -do "onfinish stop; run -all; quit -f"
