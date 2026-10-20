import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_RATES = (1e-4, 3e-4, 1e-3, 3e-3)


def read_metrics(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--output", type=Path, default=Path("output/alexnet_lr_sweep"))
    args = parser.parse_args()

    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    runs = []
    for lr in args.rates:
        run_dir = output_root / f"lr_{lr:g}"
        metrics_path = run_dir / "alexnet" / "metrics.json"
        if metrics_path.exists():
            metrics_path.unlink()
        cmd = [
            sys.executable,
            str(ROOT / "main.py"),
            "--cfg",
            str(ROOT / "configs" / "lenet_base.yaml"),
            "--output",
            str(run_dir),
            "--opts",
            "MODEL.NAME",
            "alexnet",
            "MODEL.NUM_CLASSES",
            "10",
            "DATA.IMG_SIZE",
            "70",
            "TRAIN.EPOCHS",
            str(args.epochs),
            "TRAIN.LR",
            str(lr),
        ]
        print(f"Running AlexNet with lr={lr:g}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        records = read_metrics(metrics_path)
        runs.append((lr, records))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lr, records in runs:
        epochs = [r["epoch"] + 1 for r in records]
        color = None
        line, = ax.plot(epochs, [r["train_loss"] for r in records], label=f"{lr:g} train")
        color = line.get_color()
        ax.plot(
            epochs,
            [r["val_loss"] for r in records],
            linestyle="--",
            color=color,
            alpha=0.8,
            label=f"{lr:g} val",
        )

    ax.set(title="AlexNet CIFAR-10 loss by learning rate", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output_root.mkdir(parents=True, exist_ok=True)
    plot_path = output_root / "loss_by_learning_rate.png"
    fig.savefig(plot_path, dpi=160)

    best_lr, best_val = min(
        ((lr, records[-1]["val_loss"]) for lr, records in runs),
        key=lambda item: item[1],
    )
    print(f"Saved plot: {plot_path}")
    print(f"Lowest final validation loss: lr={best_lr:g} (loss={best_val:.4f})")


if __name__ == "__main__":
    main()
