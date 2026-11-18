# ModelSim .do script: Compile all RTL & unit testbenches and execute all 6 unit tests
onbreak {resume}
onerror {quit -code 1 -f}
if {![file exists work]} { vlib work }

vlog -work work \
    rtl/relu.v \
    rtl/maxpool.v \
    rtl/gap.v \
    rtl/argmax.v \
    rtl/fully_connected.v \
    rtl/conv_engine.v \
    rtl/cnn_top.v \
    sim/tb/tb_relu.v \
    sim/tb/tb_maxpool.v \
    sim/tb/tb_gap.v \
    sim/tb/tb_fully_connected.v \
    sim/tb/tb_argmax.v \
    sim/tb/tb_conv_engine.v

vsim -c -t 1ps work.tb_relu
onfinish stop
run -all

vsim -c -t 1ps work.tb_maxpool
onfinish stop
run -all

vsim -c -t 1ps work.tb_gap
onfinish stop
run -all

vsim -c -t 1ps work.tb_fully_connected
onfinish stop
run -all

vsim -c -t 1ps work.tb_argmax
onfinish stop
run -all

vsim -c -t 1ps work.tb_conv_engine
onfinish stop
run -all

quit -f
