"""Generate V2 experiment configs.

This creates the grid:

SGD + momentum:
    batch size: 64, 512
    learning rate: 0.01, 0.05, 0.1

Adam:
    batch size: 64, 512
    learning rate: 1e-4, 5e-4, 1e-3
"""

from __future__ import annotations

from pathlib import Path

import yaml


def base_config() -> dict:
    return {
        "seed": 0,
        "device": "auto",
        "output_dir": "results",
        "model": {
            "name": "small_cnn",
            "input_channels": 1,
            "num_classes": 10,
        },
        "data": {
            "dataset": "FashionMNIST",
            "data_dir": "./data",
            "num_workers": 2,
        },
        "training": {
            "epochs": 20,
            "batch_size": 64,
        },
    }


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    output_dir = Path("v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0]
    batch_sizes = [64, 512]

    sgd_lrs = [0.01, 0.05, 0.1]
    adam_lrs = [0.0001, 0.0005, 0.001]

    for seed in seeds:
        for batch_size in batch_sizes:
            for lr in sgd_lrs:
                config = base_config()
                config["seed"] = seed
                config["training"]["batch_size"] = batch_size
                config["run_name"] = f"sgd_bs{batch_size}_lr{lr}_seed{seed}"
                config["optimizer"] = {
                    "name": "sgd",
                    "learning_rate": lr,
                    "momentum": 0.9,
                    "weight_decay": 0.0005,
                }

                path = output_dir / f"sgd_bs{batch_size}_lr{lr}_seed{seed}.yaml"
                write_config(path, config)

            for lr in adam_lrs:
                config = base_config()
                config["seed"] = seed
                config["training"]["batch_size"] = batch_size
                config["run_name"] = f"adam_bs{batch_size}_lr{lr}_seed{seed}"
                config["optimizer"] = {
                    "name": "adam",
                    "learning_rate": lr,
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.0005,
                }

                path = output_dir / f"adam_bs{batch_size}_lr{lr}_seed{seed}.yaml"
                write_config(path, config)

    print(f"Saved V2 configs to {output_dir}")


if __name__ == "__main__":
    main()
