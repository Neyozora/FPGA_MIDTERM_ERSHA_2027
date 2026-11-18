@echo off
cd /d "%~dp0"
echo ============================================================================
echo  Step 1: Programming Terasic DE10-Standard Board (VGA + JTAG + CNN)...
echo ============================================================================
"C:\intelFPGA_lite\18.1\quartus\bin64\quartus_pgm.exe" -c "DE-SoC [USB-1]" -m JTAG -o "p;quartus/output_files/de10_standard.sof@2"

echo.
echo ============================================================================
echo  Step 2: Launching Live Python + FPGA + VGA Interactive GUI...
echo ============================================================================
python -m sw.gui_app
pause
