"""Training and evaluation utilities."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model


def set_seed(seed: int) -> None:
    """Make results reasonably reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create Fashion-MNIST train/test loaders."""
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset", "FashionMNIST")
    data_dir = data_cfg.get("data_dir", "./data")
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 2))

    if dataset_name != "FashionMNIST":
        raise ValueError("This Week 1 baseline implements FashionMNIST only.")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config["optimizer"]
    name = opt_cfg["name"].lower()
    lr = float(opt_cfg["learning_rate"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    """Train or evaluate one epoch.

    If optimizer is None, evaluation mode is used.
    Returns average loss and accuracy.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train a model from a config dict and save history/checkpoint/summary."""
    seed = int(config.get("seed", 0))
    set_seed(seed)

    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(config)
    device = get_device(config.get("device", "auto"))

    model_cfg = config.get("model", {})
    model = build_model(
        name=model_cfg.get("name", "small_cnn"),
        input_channels=int(model_cfg.get("input_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 10)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    epochs = int(config.get("training", {}).get("epochs", 10))

    opt_name = config["optimizer"]["name"].lower()
    lr = float(config["optimizer"]["learning_rate"])
    batch_size = int(config["training"]["batch_size"])
    run_name = config.get("run_name", f"{opt_name}_lr{lr}_bs{batch_size}_seed{seed}")

    # Single training loop: train for `epochs` epochs, log per-epoch metrics,
    # and keep a snapshot of the best-by-test-accuracy checkpoint along the way
    history = []
    best_checkpoint = None
    best_test_accuracy_so_far = float("-inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, device, optimizer
        )
        test_loss, test_acc = run_one_epoch(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "learning_rate": lr,
        }
        history.append(row)
        print(
            f"[{run_name}] epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_test_accuracy_so_far:
            best_test_accuracy_so_far = test_acc
            best_checkpoint = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "config": copy.deepcopy(config),
                "history": copy.deepcopy(history),
                "best_epoch": epoch,
                "best_test_accuracy": test_acc,
                "best_test_loss": test_loss,
            }

    checkpoint_path = output_dir / f"{run_name}_final.pt"
    best_checkpoint_path = output_dir / f"{run_name}_best.pt"
    history_path = output_dir / f"{run_name}_history.json"
    summary_path = output_dir / f"{run_name}_summary.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "checkpoint_type": "final",
        },
        checkpoint_path,
    )

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint was recorded; did training run zero epochs?")

    best_checkpoint["checkpoint_type"] = "best_test_accuracy"
    torch.save(best_checkpoint, best_checkpoint_path)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_by_test_loss = min(history, key=lambda row: row["test_loss"])
    best_by_test_accuracy = max(history, key=lambda row: row["test_accuracy"])

    summary = {
        "run_name": run_name,
        "optimizer": opt_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "seed": seed,
        "epochs": epochs,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_loss": history[-1]["test_loss"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "best_test_loss": best_by_test_loss["test_loss"],
        "best_test_loss_epoch": best_by_test_loss["epoch"],
        "best_test_accuracy": best_by_test_accuracy["test_accuracy"],
        "best_test_accuracy_epoch": best_by_test_accuracy["epoch"],
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "history_path": str(history_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"summary": summary, "history": history}