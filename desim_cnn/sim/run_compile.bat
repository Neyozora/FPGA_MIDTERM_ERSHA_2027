@echo off
if not exist mem mkdir mem
%WINDIR%\System32\xcopy /y /c /q ..\..\mem\*.mem mem\ >nul

if exist work rmdir /S /Q work
vlib work
vlog ../tb/*.v
vlog ../../rtl/conv_engine.v ../../rtl/relu.v ../../rtl/maxpool.v ../../rtl/gap.v ../../rtl/fully_connected.v ../../rtl/argmax.v ../../rtl/cnn_top.v ../../rtl/vga_cnn_hud.v
vlog ../*.v
