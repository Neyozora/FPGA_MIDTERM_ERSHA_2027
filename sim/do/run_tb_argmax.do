# ModelSim .do script: Compile and run tb_argmax
if {![file exists work]} { vlib work }
vlog -work work rtl/argmax.v sim/tb/tb_argmax.v
vsim -c -t 1ps work.tb_argmax -do "onfinish stop; run -all; quit -f"
