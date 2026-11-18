"""
Milestone 7 automated tests: Verify all 7 Verilog-2001 RTL modules (`rtl/*.v`),
their required port interfaces, synthesizability rules, and ModelSim 18.1
compilation status (`work/_info`).

Run:
    python -m sw.tests.test_m7_rtl
    python -m pytest sw/tests/test_m7_rtl.py
"""
import re
import sys

from sw.config import REPO_ROOT
from sw.tests._runner import run_all

RTL_DIR = REPO_ROOT / "rtl"
WORK_INFO = REPO_ROOT / "work" / "_info"
ARCH_DOC = REPO_ROOT / "docs" / "rtl_architecture.md"

REQUIRED_MODULES = {
    "conv_engine.v": (
        "conv_engine",
        ["clk", "rst", "start", "busy", "done", "cfg_in_c", "cfg_in_h", "cfg_in_w",
         "cfg_out_c", "cfg_out_h", "cfg_out_w", "cfg_mult", "cfg_shift",
         "in_rd_addr", "in_rd_data", "w_rd_addr", "w_rd_data", "b_rd_addr", "b_rd_data",
         "out_valid", "out_ready", "out_addr", "out_acc_int32", "out_req_int8"],
    ),
    "relu.v": (
        "relu",
        ["clk", "rst", "in_valid", "in_addr", "in_data", "out_data_comb",
         "out_valid", "out_addr", "out_data"],
    ),
    "maxpool.v": (
        "maxpool",
        ["clk", "rst", "start", "busy", "done", "in_rd_addr", "in_rd_data",
         "out_valid", "out_addr", "out_data"],
    ),
    "gap.v": (
        "gap",
        ["clk", "rst", "start", "busy", "done", "cfg_mult", "cfg_shift",
         "in_rd_addr", "in_rd_data", "out_valid", "out_ch", "out_sum_int32", "out_data"],
    ),
    "fully_connected.v": (
        "fully_connected",
        ["clk", "rst", "start", "busy", "done", "cfg_mult", "cfg_shift",
         "in_rd_addr", "in_rd_data", "w_rd_addr", "w_rd_data", "b_rd_addr", "b_rd_data",
         "out_valid", "out_idx", "out_acc_int32", "out_data"],
    ),
    "argmax.v": (
        "argmax",
        ["clk", "rst", "start", "busy", "done", "in_rd_addr", "in_rd_data",
         "predicted_class", "max_score"],
    ),
    "cnn_top.v": (
        "cnn_top",
        ["clk", "rst", "start", "done", "predicted_class",
         "ext_img_wr_en", "ext_img_wr_addr", "ext_img_wr_data"],
    ),
}


def test_all_required_rtl_files_and_ports_exist():
    """Verify every required Verilog module file exists and declares all expected ports."""
    assert ARCH_DOC.is_file(), f"Missing {ARCH_DOC}"
    for fname, (mod_name, ports) in REQUIRED_MODULES.items():
        fpath = RTL_DIR / fname
        assert fpath.is_file(), f"Missing RTL file: {fpath}"
        text = fpath.read_text(encoding="utf-8")
        assert re.search(rf"\bmodule\s+{mod_name}\b", text), (
            f"{fname} does not declare `module {mod_name}`"
        )
        assert re.search(r"\bendmodule\b", text), f"{fname} missing `endmodule`"
        for p in ports:
            assert re.search(rf"\b{p}\b", text), f"{fname} missing required port `{p}`"


def test_synthesizable_verilog2001_conventions():
    """Ensure RTL files do not use floating-point types or simulation delays inside RTL."""
    for fname in REQUIRED_MODULES:
        text = (RTL_DIR / fname).read_text(encoding="utf-8")
        # Strip comments
        code = re.sub(r"//.*", "", text)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        assert not re.search(r"\b(real|shortreal|realtime)\b", code), (
            f"{fname} uses unsynthesizable floating-point type"
        )
        # Ensure `#<number>` delay is not used outside `timescale
        no_timescale = re.sub(r"`timescale.*", "", code)
        # Parameter instantiation `#(` is allowed, but `#10` delay is not
        assert not re.search(r"#\s*\d+", no_timescale), (
            f"{fname} contains unsynthesizable `#<delay>` statement"
        )
        assert "signed" in code, f"{fname} must use explicit `signed` declarations"


def test_cnn_top_instantiates_all_six_submodules():
    """Verify `cnn_top.v` instantiates `conv_engine`, `relu`, `maxpool`, `gap`, `fully_connected`, and `argmax`."""
    top_text = (RTL_DIR / "cnn_top.v").read_text(encoding="utf-8")
    for submod in ("conv_engine", "relu", "maxpool", "gap", "fully_connected", "argmax"):
        assert re.search(rf"\b{submod}\s+#\(", top_text), (
            f"cnn_top.v does not instantiate `{submod}`"
        )


def test_modelsim_work_library_compiled_cleanly():
    """Verify ModelSim 18.1 compiled all 7 modules into `work/_info`."""
    assert WORK_INFO.is_file(), "ModelSim `work/_info` not found — run `vlog` first"
    info_text = WORK_INFO.read_text(encoding="latin-1")
    for _, (mod_name, _) in REQUIRED_MODULES.items():
        assert f"v{mod_name}" in info_text, (
            f"Module `{mod_name}` not found in compiled ModelSim `work/_info`"
        )


if __name__ == "__main__":
    sys.exit(run_all(globals()))
