"""Plot Week 2 training curves and summary comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def load_history(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curves(history_paths: Iterable[Path], output_path: Path) -> None:
    """Create one figure containing train and test loss curves for each run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for path in history_paths:
        history = load_history(path)
        epochs = [row["epoch"] for row in history]
        train_loss = [row["train_loss"] for row in history]
        test_loss = [row["test_loss"] for row in history]
        label = path.name.replace("_history.json", "")

        plt.plot(epochs, train_loss, linestyle="-", label=f"{label} train")
        plt.plot(epochs, test_loss, linestyle="--", label=f"{label} test")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Train/test loss curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved figure to {output_path}")


def plot_test_accuracy_curves(history_paths: Iterable[Path], output_path: Path) -> None:
    """Plot test accuracy versus epoch for each run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for path in history_paths:
        history = load_history(path)
        epochs = [row["epoch"] for row in history]
        test_accuracy = [row["test_accuracy"] for row in history]
        label = path.name.replace("_history.json", "")

        plt.plot(epochs, test_accuracy, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved figure to {output_path}")


def plot_best_accuracy_bar(summary_paths: Iterable[Path], output_path: Path) -> None:
    """Bar plot of best test accuracy for each run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = [load_summary(path) for path in summary_paths]
    summaries = sorted(
        summaries,
        key=lambda s: (
            s["optimizer"],
            s["batch_size"],
            s["learning_rate"],
            s["seed"],
        ),
    )

    labels = [
        f'{s["optimizer"]}\nbs={s["batch_size"]}\nlr={s["learning_rate"]}'
        for s in summaries
    ]
    values = [s["best_test_accuracy"] for s in summaries]

    plt.figure(figsize=(max(10, len(labels) * 0.7), 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Best test accuracy")
    plt.title("Best test accuracy by optimizer, batch size, and learning rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--figures", type=str, default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    figures_dir = Path(args.figures)

    histories = sorted(results_dir.glob("*_history.json"))
    summaries = sorted(results_dir.glob("*_summary.json"))

    if not histories:
        raise FileNotFoundError(f"No *_history.json files found in {results_dir}")

    if not summaries:
        raise FileNotFoundError(f"No *_summary.json files found in {results_dir}")

    plot_loss_curves(histories, figures_dir / "loss_curves.png")
    plot_test_accuracy_curves(histories, figures_dir / "test_accuracy_curves.png")
    plot_best_accuracy_bar(summaries, figures_dir / "best_test_accuracy.png")


if __name__ == "__main__":
    main()
