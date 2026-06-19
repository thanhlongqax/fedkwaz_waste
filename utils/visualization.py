"""
Utilities: Visualization + Metrics
====================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional
import logging
from PIL import Image
logger = logging.getLogger(__name__)

# Color scheme cho 4 clients
CLIENT_COLORS = {
    "client_0": "#2196F3",  # Blue - ZeroWaste
    "client_1": "#4CAF50",  # Green - SpectralWaste
    "client_2": "#FF9800",  # Orange - TACO
    "client_3": "#9C27B0",  # Purple - MJU-Waste
}

CLIENT_LABELS = {
    "client_0": "ZeroWaste\n(Industrial RGB)",
    "client_1": "SpectralWaste\n(RGB+HSI)",
    "client_2": "TACO\n(Mobile cams)",
    "client_3": "MJU-Waste\n(RGBD)",
}


def plot_training_curves(history: List[Dict], output_dir: str = "./outputs"):
    """Vẽ training curves tổng hợp cho toàn bộ FL training"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history:
        logger.warning("No history to plot")
        return

    rounds = [h["round"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "FedKWAZ Training Dashboard\nIndustrial Waste Management (Multi-Factory FL)",
        fontsize=14, fontweight="bold", y=0.98
    )

    # ── Plot 1: Total Loss per Client ────────────────────────────────────────
    ax = axes[0, 0]
    client_ids = list(CLIENT_COLORS.keys())

    for cid in client_ids:
        losses = []
        for h in history:
            stats_list = h.get("client_stats", [])
            for stats in stats_list:
                if stats.get("client_id") == cid:
                    losses.append(stats.get("avg_total_loss", float("nan")))
                    break
            else:
                losses.append(float("nan"))

        valid_rounds = [r for r, l in zip(rounds, losses) if not np.isnan(l)]
        valid_losses = [l for l in losses if not np.isnan(l)]
        if valid_rounds:
            ax.plot(valid_rounds, valid_losses,
                    color=CLIENT_COLORS[cid], linewidth=2,
                    label=CLIENT_LABELS.get(cid, cid), marker="o", markersize=3)

    ax.set_title("Total Loss per Client", fontweight="bold")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    # ── Plot 2: KWAZ Loss per Client ─────────────────────────────────────────
    ax = axes[0, 1]
    for cid in client_ids:
        kwaz_losses = []
        for h in history:
            for stats in h.get("client_stats", []):
                if stats.get("client_id") == cid:
                    kwaz_losses.append(stats.get("avg_kwaz_loss", float("nan")))
                    break
            else:
                kwaz_losses.append(float("nan"))

        valid_r = [r for r, l in zip(rounds, kwaz_losses) if not np.isnan(l)]
        valid_l = [l for l in kwaz_losses if not np.isnan(l)]
        if valid_r:
            ax.plot(valid_r, valid_l, color=CLIENT_COLORS[cid], linewidth=2,
                    label=CLIENT_LABELS.get(cid, cid), linestyle="--", marker="s", markersize=3)

    ax.set_title("KWAZ Loss (Knowledge Alignment)", fontweight="bold")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("KWAZ Loss")
    ax.plot(losses, label="Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    # ── Plot 3: Accuracy over Rounds ─────────────────────────────────────────
    ax = axes[0, 2]
    eval_rounds = [h["round"] for h in history if h.get("eval_metrics")]
    avg_accs = [h.get("avg_accuracy", 0) for h in history if h.get("eval_metrics")]

    if eval_rounds:
        ax.plot(eval_rounds, avg_accs, color="#E91E63", linewidth=2.5,
                marker="D", markersize=5, label="Global Avg Accuracy")

        for cid in client_ids:
            client_accs = []
            for h in history:
                if h.get("eval_metrics") and cid in h["eval_metrics"]:
                    client_accs.append(h["eval_metrics"][cid].get("accuracy", 0))

            if len(client_accs) == len(eval_rounds):
                ax.plot(eval_rounds, client_accs,
                        color=CLIENT_COLORS[cid], linewidth=1.5,
                        linestyle=":", alpha=0.7,
                        label=CLIENT_LABELS.get(cid, cid))

    ax.set_title("Accuracy over Rounds", fontweight="bold")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    # ── Plot 4: Proxy Loss ────────────────────────────────────────────────────
    ax = axes[1, 0]
    proxy_losses = [h.get("proxy_loss", float("nan")) for h in history]
    valid_r = [r for r, l in zip(rounds, proxy_losses) if not np.isnan(l)]
    valid_l = [l for l in proxy_losses if not np.isnan(l)]
    if valid_r:
        ax.fill_between(valid_r, valid_l, alpha=0.3, color="#607D8B")
        ax.plot(valid_r, valid_l, color="#607D8B", linewidth=2, label="Proxy Loss")
    ax.set_title("Server Proxy Model Loss", fontweight="bold")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    # ── Plot 5: Round Time ────────────────────────────────────────────────────
    ax = axes[1, 1]
    round_times = [h.get("round_time_sec", 0) for h in history]
    ax.bar(rounds, round_times, color="#795548", alpha=0.7, width=0.8)
    if round_times:
        avg_time = np.mean(round_times)
        ax.axhline(avg_time, color="red", linestyle="--", linewidth=1.5,
                   label=f"Avg: {avg_time:.1f}s")
    ax.set_title("Round Duration (Communication Overhead)", fontweight="bold")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Time (seconds)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f8f9fa")

    # ── Plot 6: Camera Heterogeneity Legend ───────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")

    camera_info = [
        ("client_0", "ZeroWaste", "Industrial RGB\n(Teledyne Dalsa)", "1920×1080", "ResNet-50"),
        ("client_1", "SpectralWaste", "RGB + HSI\n(Specim FX17)", "1024×1024", "EfficientNet-B3"),
        ("client_2", "TACO", "Mobile Cameras\n(Crowdsourced)", "Variable", "YOLOv8-Nano"),
        ("client_3", "MJU-Waste", "RGBD Kinect\n(Depth sensor)", "640×480", "MobileNetV3"),
    ]

    y_pos = 0.95
    ax.text(0.5, 1.02, "System Architecture", ha="center", va="top",
            transform=ax.transAxes, fontsize=11, fontweight="bold")

    for cid, dataset, camera, res, model in camera_info:
        color = CLIENT_COLORS[cid]
        rect = mpatches.FancyBboxPatch(
            (0.02, y_pos - 0.18), 0.96, 0.17,
            boxstyle="round,pad=0.01",
            facecolor=color + "22", edgecolor=color, linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        ax.text(0.05, y_pos - 0.04, f"● {cid.upper()}: {dataset}",
                transform=ax.transAxes, fontsize=8, fontweight="bold", color=color)
        ax.text(0.05, y_pos - 0.09, f"  Camera: {camera}",
                transform=ax.transAxes, fontsize=7, color="#333")
        ax.text(0.05, y_pos - 0.14, f"  Resolution: {res} | Model: {model}",
                transform=ax.transAxes, fontsize=7, color="#333")

        y_pos -= 0.22

    plt.tight_layout()
    out_path = output_dir / "fedkwaz_training_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"📊 Training dashboard saved: {out_path}")


def print_round_summary(summary: Dict):
    """In summary của một FL round ra console"""
    print(f"\n{'─'*55}")
    print(f"  Round {summary['round']:3d} | "
          f"Time: {summary.get('round_time_sec', 0):.1f}s | "
          f"Agg: {summary.get('aggregation', 'N/A')}")
    print(f"  Proxy Loss: {summary.get('proxy_loss', 0):.4f}")

    for stats in summary.get("client_stats", []):
        print(
            f"  [{stats['client_id']}] "
            f"Loss={stats.get('avg_total_loss', 0):.4f} "
            f"KWAZ={stats.get('avg_kwaz_loss', 0):.4f} "
            f"n={stats.get('num_samples', 0)}"
        )

    if summary.get("eval_metrics"):
        avg_acc = summary.get("avg_accuracy", 0)
        print(f"  Avg Accuracy: {avg_acc:.4f}")
    print(f"{'─'*55}")


def compute_per_class_metrics(preds: list, targets: list, class_names: List[str]) -> Dict:
    """Tính per-class accuracy và confusion matrix"""
    from collections import defaultdict
    n_classes = len(class_names)
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for pred, target in zip(preds, targets):
        per_class_total[target] += 1
        if pred == target:
            per_class_correct[target] += 1

    metrics = {}
    for i, name in enumerate(class_names):
        total = per_class_total.get(i, 0)
        correct = per_class_correct.get(i, 0)
        metrics[name] = correct / max(total, 1)

    return metrics

def create_attention_progress_figure(
    monitor_dir,
    client_id,
    first_round=1,
    last_round=100
):
    """
    Tạo figure:

    (a) Input Image
    (b) Activation Map Round 1
    (c) Activation Map Round N
    """

    monitor_dir = Path(monitor_dir)

    input_img = monitor_dir / f"{client_id}_input.png"
    round1_img = monitor_dir / f"{client_id}_round_{first_round}.png"
    roundN_img = monitor_dir / f"{client_id}_round_{last_round}.png"

    if not (
        input_img.exists()
        and round1_img.exists()
        and roundN_img.exists()
    ):
        print("Missing monitor images")
        return

    fig = plt.figure(figsize=(15,5))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(Image.open(input_img))
    ax1.set_title("(a) Input Image")

    ax2.imshow(Image.open(round1_img))
    ax2.set_title(f"(b) Activation Map Round {first_round}")

    ax3.imshow(Image.open(roundN_img))
    ax3.set_title(f"(c) Activation Map Round {last_round}")

    for ax in [ax1, ax2, ax3]:
        ax.axis("off")

    plt.tight_layout()

    save_path = monitor_dir / f"{client_id}_attention_progress.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved -> {save_path}")
