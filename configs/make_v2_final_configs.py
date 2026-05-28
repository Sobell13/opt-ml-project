from pathlib import Path
import yaml

out = Path("v2_final")
out.mkdir(parents=True, exist_ok=True)

settings = [
    ("sgd", 64, 0.01),
    ("sgd", 512, 0.05),
    ("adam", 64, 0.001),
    ("adam", 512, 0.001),
]

for opt, bs, lr in settings:
    for seed in [0, 1, 2]:
        cfg = {
            "seed": seed,
            "device": "auto",
            "output_dir": "results",
            "run_name": f"{opt}_bs{bs}_lr{lr}_seed{seed}",
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
                "epochs": 50,
                "batch_size": bs,
            },
        }

        if opt == "sgd":
            cfg["optimizer"] = {
                "name": "sgd",
                "learning_rate": lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            }
        else:
            cfg["optimizer"] = {
                "name": "adam",
                "learning_rate": lr,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0005,
            }

        path = out / f"{opt}_bs{bs}_lr{lr}_seed{seed}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Saved configs to {out}")
