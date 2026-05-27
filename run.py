"""Run one or more Week 1 baseline experiments.

Examples:
    python run.py --config configs/sgd_small.yaml
    python run.py --config configs/adam_small.yaml
    python run.py --config configs/sgd_small.yaml --config configs/adam_small.yaml --make-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from plots import plot_loss_curves
from train import train_from_config


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a YAML config. Can be passed multiple times.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="After training, save loss curves for the histories just produced.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Folder where results/ and figures/ will be created.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"

    produced_histories = []

    for config_path in args.config:
        config = load_yaml(config_path)

        # Override the config output_dir from the command line
        config["output_dir"] = str(results_dir)

        result = train_from_config(config)
        produced_histories.append(Path(result["summary"]["history_path"]))

    if args.make_plots:
        plot_loss_curves(
            produced_histories,
            figures_dir / "loss_curves.png",
        )


if __name__ == "__main__":
    main()
