import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="?",
        default="output/alexnet/metrics.json",
        help="Path to the newline-delimited metrics.json file",
    )
    parser.add_argument(
        "--save",
        help="Output image path (default: training_curves.png beside metrics.json)",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    with metrics_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise f"Yo nga no epoch records found in {metrics_path}"

    epochs = [record["epoch"] + 1 for record in records]
    fig, (accuracy_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, train_key, val_key, title, ylabel in (
        (accuracy_ax, "train_acc", "val_acc", "Accuracy", "Accuracy (%)"),
        (loss_ax, "train_loss", "val_loss", "Loss", "Cross-entropy loss"),
    ):
        ax.plot(epochs, [r[train_key] for r in records], marker="o", label="Train")
        ax.plot(epochs, [r[val_key] for r in records], marker="o", label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    output_path = Path(args.save) if args.save else metrics_path.with_name("training_curves.png")
    fig.savefig(output_path, dpi=160)
    print(f"Saved {len(records)} epoch points to {output_path}")


if __name__ == "__main__":
    main()
