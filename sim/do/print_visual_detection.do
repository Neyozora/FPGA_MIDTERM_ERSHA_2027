# =============================================================================
# ModelSim Visual ASCII Image & Detection Display (`sim/do/print_visual_detection.do`)
# Reads the actual 32x32 input pixels from `/tb_cnn_top/dut/img_ram` and the 10
# class scores from `/tb_cnn_top/dut/fc_score_ram` inside the live ModelSim
# simulation and prints the image + detection bar chart in the Transcript!
# =============================================================================
quietly set class_names [list \
    "T-shirt/top" "Trouser" "Pullover" "Dress" "Coat" \
    "Sandal" "Shirt" "Sneaker" "Bag" "Ankle boot"]

echo ""
echo "======================================================================"
echo "   1. INPUT IMAGE INSIDE FPGA BRAM (/tb_cnn_top/dut/img_ram 32x32)    "
echo "======================================================================"
for {set y 0} {$y < 32} {incr y} {
    quietly set row_str [format "Row %2d | " $y]
    for {set x 0} {$x < 32} {incr x} {
        quietly set addr [expr {$y * 32 + $x}]
        quietly set p [examine -radix decimal "/tb_cnn_top/dut/img_ram($addr)"]
        if {$p > 95} {
            append row_str "##"
        } elseif {$p > 60} {
            append row_str "**"
        } elseif {$p > 25} {
            append row_str ".."
        } elseif {$p > 5} {
            append row_str "--"
        } else {
            append row_str "  "
        }
    }
    echo "$row_str|"
}

echo "======================================================================"
echo "   2. FPGA CNN CLASS SCORES (/tb_cnn_top/dut/fc_score_ram 0..9)       "
echo "======================================================================"
quietly set pred_cls [examine -radix unsigned "/tb_cnn_top/predicted_class"]
for {set c 0} {$c < 10} {incr c} {
    quietly set sc [examine -radix decimal "/tb_cnn_top/dut/fc_score_ram($c)"]
    quietly set cname [lindex $class_names $c]
    quietly set bar_len [expr {($sc + 128) / 5}]
    if {$bar_len < 0} { quietly set bar_len 0 }
    quietly set bar [string repeat "=" $bar_len]
    if {$c == $pred_cls} {
        echo [format "  Class %d (%-11s): %4d |%s|  <-- FPGA WINNER!" $c $cname $sc $bar]
    } else {
        echo [format "  Class %d (%-11s): %4d |%s" $c $cname $sc $bar]
    }
}

quietly set win_name [lindex $class_names $pred_cls]
quietly set win_score [examine -radix decimal "/tb_cnn_top/dut/fc_score_ram($pred_cls)"]
echo "======================================================================"
echo [format "   >>> FINAL FPGA DETECTION OUTPUT: CLASS %d = %s (Score: %d) <<<" $pred_cls $win_name $win_score]
echo "======================================================================"
