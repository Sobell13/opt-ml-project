"""Plot train/test loss curves from saved JSON histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def load_history(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curves(history_paths: Iterable[Path], output_path: Path) -> None:
    """Create one figure containing train and test loss curves for each run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
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
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures/loss_curves.png")
    args = parser.parse_args()

    results_dir = Path(args.results)
    histories = sorted(results_dir.glob("*_history.json"))
    if not histories:
        raise FileNotFoundError(f"No *_history.json files found in {results_dir}")
    plot_loss_curves(histories, Path(args.output))


if __name__ == "__main__":
    main()
