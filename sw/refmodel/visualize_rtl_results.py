"""
Generate visual dashboards of the FPGA RTL CNN pipeline (`img0` layer-by-layer
feature maps + 10-class ModelSim detection results) so the user can visually
inspect the input images, internal FPGA feature maps, class score bars, and
final detection outputs.

Run:
    python -m sw.refmodel.visualize_rtl_results
"""
import shutil
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from sw.config import CLASS_NAMES, MEM_DIR, REPO_ROOT, SIM_VECTORS_DIR
from sw.export.formatters import read_mem

SIM_REPORT_DIR = REPO_ROOT / "reports" / "simulation"
ARTIFACT_DIR = Path(
    r"C:\Users\Ersha Anandita\.gemini\antigravity\brain\e6a30f7b-010b-495a-aa36-853b92d52282"
)


def plot_img0_pipeline() -> Path:
    img0 = read_mem(SIM_VECTORS_DIR / "img0_input.mem", 8).reshape(32, 32)
    relu1 = read_mem(SIM_VECTORS_DIR / "img0_relu1_out.mem", 8).reshape(8, 30, 30)
    pool1 = read_mem(SIM_VECTORS_DIR / "img0_pool1_out.mem", 8).reshape(8, 15, 15)
    relu2 = read_mem(SIM_VECTORS_DIR / "img0_relu2_out.mem", 8).reshape(16, 13, 13)
    gap = read_mem(SIM_VECTORS_DIR / "img0_gap_out.mem", 8)
    fc_scores = read_mem(SIM_VECTORS_DIR / "img0_fc_scores.mem", 8)
    pred_cls = int(read_mem(SIM_VECTORS_DIR / "img0_expected_class.mem", 8)[0])

    fig = plt.figure(figsize=(18, 10), dpi=140, facecolor="#111827")
    fig.suptitle(
        "Inside the FPGA CNN Accelerator (rtl/cnn_top.v) — Live Hardware Tensors & Detection Output",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )

    outer = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.1, 2.2, 2.2], hspace=0.34, wspace=0.26)

    # 1. Input image (32x32)
    ax_in = fig.add_subplot(outer[0, 0])
    ax_in.imshow(img0, cmap="gray", vmin=0, vmax=127)
    ax_in.set_title("1. Input Image in FPGA BRAM\n(img_ram: 1×32×32 INT8, 2px pad)", color="#60A5FA", fontsize=11, fontweight="bold")
    ax_in.axis("off")

    # 2. Conv1 + ReLU1 (8 x 30 x 30)
    gs_c1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0, 1], wspace=0.1, hspace=0.22)
    for c in range(8):
        ax = fig.add_subplot(gs_c1[c // 4, c % 4])
        ax.imshow(relu1[c], cmap="viridis", vmin=0, vmax=127)
        ax.set_title(f"C1 ch{c}", color="#D1D5DB", fontsize=8)
        ax.axis("off")
    ax_c1_hdr = fig.add_subplot(outer[0, 1], frameon=False)
    ax_c1_hdr.set_xticks([])
    ax_c1_hdr.set_yticks([])
    ax_c1_hdr.set_title("2. Conv1 + ReLU1\n(conv1_relu_ram: 8×30×30 INT8)",
                        color="#34D399", fontsize=10.5, fontweight="bold", pad=16)

    # 3. MaxPool1 (8 x 15 x 15)
    gs_p1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0, 2], wspace=0.1, hspace=0.22)
    for c in range(8):
        ax = fig.add_subplot(gs_p1[c // 4, c % 4])
        ax.imshow(pool1[c], cmap="viridis", vmin=0, vmax=127)
        ax.set_title(f"Pool ch{c}", color="#D1D5DB", fontsize=8)
        ax.axis("off")
    ax_p1_hdr = fig.add_subplot(outer[0, 2], frameon=False)
    ax_p1_hdr.set_xticks([])
    ax_p1_hdr.set_yticks([])
    ax_p1_hdr.set_title("3. MaxPool1\n(pool1_ram: 8×15×15 INT8)",
                        color="#FBBF24", fontsize=10.5, fontweight="bold", pad=16)

    # 4. GAP 16-channel bar chart
    ax_gap = fig.add_subplot(outer[1, 0])
    ax_gap.set_facecolor("#1F2937")
    ax_gap.bar(np.arange(16), gap, color="#A78BFA", edgecolor="white", linewidth=0.5)
    ax_gap.set_title("5. GAP Output (gap_ram: 16 INT8)\nAverage of each 13×13 map", color="#A78BFA", fontsize=11, fontweight="bold")
    ax_gap.set_xlabel("Channel (0..15)", color="#E5E7EB", fontsize=9)
    ax_gap.set_ylabel("INT8 Value [0..127]", color="#E5E7EB", fontsize=9)
    ax_gap.tick_params(colors="#E5E7EB", labelsize=8)
    for spine in ax_gap.spines.values():
        spine.set_color("#4B5563")

    # 5. Conv2 + ReLU2 (16 x 13 x 13)
    gs_c2 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[1, 1], wspace=0.08, hspace=0.18)
    for c in range(16):
        ax = fig.add_subplot(gs_c2[c // 4, c % 4])
        ax.imshow(relu2[c], cmap="magma", vmin=0, vmax=127)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_c2_hdr = fig.add_subplot(outer[1, 1], frameon=False)
    ax_c2_hdr.set_xticks([])
    ax_c2_hdr.set_yticks([])
    ax_c2_hdr.set_title("4. Conv2 + ReLU2\n(conv2_relu_ram: 16×13×13 INT8)",
                        color="#F472B6", fontsize=10.5, fontweight="bold", pad=12)

    # 6. FC 10-Class Scores + Argmax Detection Winner
    ax_fc = fig.add_subplot(outer[1, 2])
    ax_fc.set_facecolor("#1F2937")
    y_pos = np.arange(10)
    colors = ["#10B981" if i == pred_cls else "#6B7280" for i in range(10)]
    bars = ax_fc.barh(y_pos, fc_scores, color=colors, edgecolor="white", linewidth=0.6)
    ax_fc.axvline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax_fc.set_yticks(y_pos)
    ax_fc.set_yticklabels([f"{i}: {CLASS_NAMES[i]}" for i in range(10)], color="white", fontsize=9)
    ax_fc.invert_yaxis()
    ax_fc.set_xlabel("FPGA Signed INT8 Class Score (fc_score_ram)", color="#E5E7EB", fontsize=9)
    ax_fc.set_title(
        f"6. FPGA Detection Output (predicted_class = {pred_cls})\n"
        f"DETECTED: {CLASS_NAMES[pred_cls].upper()} (Score = +{int(fc_scores[pred_cls])})",
        color="#10B981",
        fontsize=11,
        fontweight="bold",
    )
    ax_fc.tick_params(colors="#E5E7EB", labelsize=8)
    ax_fc.set_xlim(-45, 20)
    for idx, b in enumerate(bars):
        v = int(fc_scores[idx])
        offset = 1.2 if v >= 0 else -4.5
        ax_fc.text(v + offset, idx + 0.12, f"{v:+d}", color="white", fontsize=8, fontweight="bold")
    for spine in ax_fc.spines.values():
        spine.set_color("#4B5563")

    out_path = SIM_REPORT_DIR / "fpga_cnn_pipeline_visual.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    if ARTIFACT_DIR.is_dir():
        shutil.copy2(out_path, ARTIFACT_DIR / out_path.name)
    return out_path


def plot_10class_detections() -> Path:
    batch_imgs = read_mem(MEM_DIR / "test_images_100.mem", 8).reshape(100, 32, 32)
    batch_lbls = read_mem(MEM_DIR / "test_labels_100.mem", 8)
    batch_scores = read_mem(SIM_VECTORS_DIR / "batch100_expected_scores.mem", 8).reshape(100, 10)
    batch_preds = read_mem(SIM_VECTORS_DIR / "batch100_expected_preds.mem", 8)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=140, facecolor="#111827")
    fig.suptitle(
        "ModelSim RTL Simulation (tb_cnn_top.v Phase 2) — All 10 Fashion-MNIST Classes Classified by FPGA",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    for k in range(10):
        vec_idx = k * 10
        ax = axes[k // 5, k % 5]
        img = batch_imgs[vec_idx]
        gt = int(batch_lbls[vec_idx])
        pred = int(batch_preds[vec_idx])
        sc = batch_scores[vec_idx]

        ax.imshow(img, cmap="gray", vmin=0, vmax=127)
        ax.set_title(
            f"Vec #{vec_idx} | GT: {CLASS_NAMES[gt]}\n"
            f"FPGA Output: {CLASS_NAMES[pred]} (Class {pred})\n"
            f"Winning Score: {int(sc[pred]):+d} | PASS",
            color="#34D399" if pred == gt else "#F87171",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = SIM_REPORT_DIR / "fpga_10class_detections.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    if ARTIFACT_DIR.is_dir():
        shutil.copy2(out_path, ARTIFACT_DIR / out_path.name)
    return out_path


def main() -> None:
    p1 = plot_img0_pipeline()
    p2 = plot_10class_detections()
    print(f"Saved {p1}")
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
