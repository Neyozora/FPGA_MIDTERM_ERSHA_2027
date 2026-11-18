# ModelSim .do script: Compile and run tb_gap
if {![file exists work]} { vlib work }
vlog -work work rtl/gap.v sim/tb/tb_gap.v
vsim -c -t 1ps work.tb_gap -do "onfinish stop; run -all; quit -f"
