import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_monitor_history(monitor_dir):

    monitor_dir = Path(monitor_dir)

    plt.figure(figsize=(12,6))

    for json_file in monitor_dir.glob("*_history.json"):

        with open(json_file) as f:
            history = json.load(f)

        rounds = [x["round"] for x in history]
        confs = [x["confidence"] for x in history]

        plt.plot(
            rounds,
            confs,
            marker="o",
            label=json_file.stem.replace("_history","")
        )

    plt.xlabel("FL Round")
    plt.ylabel("Confidence")

    plt.title(
        "FedKWAZ Confidence Evolution"
    )

    plt.grid(True)
    plt.legend()

    save_path = monitor_dir / "confidence_curves.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved -> {save_path}")