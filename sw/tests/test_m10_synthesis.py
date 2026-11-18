"""Milestone 10 automated verification: Quartus Prime FPGA synthesis & timing."""

from pathlib import Path
from sw.tests._runner import run_all

ROOT = Path(__file__).resolve().parents[2]


def test_quartus_project_and_bitstream_exist() -> None:
    """Verify Quartus project files and compiled .sof bitstream exist."""
    required = [
        ROOT / "rtl" / "de10_standard_top.v",
        ROOT / "mem" / "de10_images_10.mem",
        ROOT / "quartus" / "de10_standard.qpf",
        ROOT / "quartus" / "de10_standard.qsf",
        ROOT / "quartus" / "de10_standard.sdc",
        ROOT / "quartus" / "cnn_top.qpf",
        ROOT / "quartus" / "cnn_top.qsf",
        ROOT / "quartus" / "cnn_top.sdc",
        ROOT / "quartus" / "output_files" / "de10_standard.sof",
        ROOT / "quartus" / "output_files" / "de10_standard.map.rpt",
        ROOT / "quartus" / "output_files" / "de10_standard.fit.rpt",
        ROOT / "quartus" / "output_files" / "de10_standard.sta.rpt",
    ]
    for path in required:
        assert path.exists(), f"Missing Milestone 10 artifact: {path}"
        assert path.stat().st_size > 0, f"Empty Milestone 10 artifact: {path}"


def test_quartus_reports_and_synthesis_summary() -> None:
    """Verify Fitter, Timing Analyzer, and Markdown synthesis report metrics."""
    fit_rpt = (ROOT / "quartus" / "output_files" / "de10_standard.fit.rpt").read_text(
        encoding="utf-8", errors="ignore"
    )
    sta_rpt = (ROOT / "quartus" / "output_files" / "de10_standard.sta.rpt").read_text(
        encoding="utf-8", errors="ignore"
    )
    syn_md = (ROOT / "reports" / "synthesis" / "synthesis_report.md").read_text(
        encoding="utf-8"
    )

    assert "Fitter Status                   ; Successful" in fit_rpt
    assert "5CSXFC6D6F31C6" in fit_rpt
    assert "804 / 41,910" in fit_rpt
    assert "22 / 112" in fit_rpt
    assert "67.82 MHz" in sta_rpt
    assert "804" in syn_md and "67.82 MHz" in syn_md and "104,272" in syn_md


if __name__ == "__main__":
    run_all(globals())
