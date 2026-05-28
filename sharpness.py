"""Sharpness measurements for trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from models import build_model
from train import build_dataloaders, get_device, set_seed


def limited_batches(loader: Iterable, max_batches: int | None):
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        yield batch


@torch.no_grad()
def average_loss(model, loader, criterion, device, max_batches=None) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for x, y in limited_batches(loader, max_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = criterion(model(x), y)
        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("No examples were evaluated.")
    return total_loss / total_examples


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model_cfg = config.get("model", {})
    model = build_model(
        name=model_cfg.get("name", "small_cnn"),
        input_channels=int(model_cfg.get("input_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 10)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def estimate_random_direction_sharpness(
    model,
    loader,
    device,
    rho: float = 0.01,
    directions: int = 10,
    max_batches: int | None = 10,
    seed: int = 0,
) -> dict:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    criterion = nn.CrossEntropyLoss()
    original = parameters_to_vector(model.parameters()).detach().clone()
    weight_norm = torch.linalg.vector_norm(original).clamp_min(1e-12)
    perturbation_norm = rho * weight_norm

    base_loss = average_loss(model, loader, criterion, device, max_batches)
    perturbed_losses = []

    for _ in range(directions):
        direction = torch.randn(original.numel(), generator=generator, device=device)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)

        vector_to_parameters(
            original + perturbation_norm * direction, model.parameters()
        )
        perturbed_losses.append(
            average_loss(model, loader, criterion, device, max_batches)
        )

    vector_to_parameters(original, model.parameters())

    worst_loss = max(perturbed_losses)
    return {
        "base_loss": base_loss,
        "worst_perturbed_loss": worst_loss,
        "sharpness": worst_loss - base_loss,
        "mean_perturbed_loss": sum(perturbed_losses) / len(perturbed_losses),
        "rho": rho,
        "directions": directions,
        "max_batches": max_batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure random-direction sharpness for trained checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing .pt checkpoint files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.01,
        help="Relative perturbation size.",
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=10,
        help="Number of random directions per checkpoint.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Maximum number of batches used for loss evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sharpness directions.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    checkpoint_paths = sorted(args.checkpoint_dir.glob("*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pt checkpoints found in {args.checkpoint_dir}")

    rows = []

    for checkpoint_path in checkpoint_paths:
        print(f"Measuring sharpness: {checkpoint_path}")

        model, config = load_checkpoint_model(checkpoint_path, device)

        training_cfg = config.get("training", {})
        optimizer_cfg = config.get("optimizer", {})

        train_loader, test_loader = build_dataloaders(config)

        result = estimate_random_direction_sharpness(
            model=model,
            loader=test_loader,
            device=device,
            rho=args.rho,
            directions=args.directions,
            max_batches=args.max_batches,
            seed=args.seed,
        )

        row = {
            "checkpoint": checkpoint_path.name,
            "optimizer": optimizer_cfg.get("name", ""),
            "batch_size": training_cfg.get("batch_size", ""),
            "learning_rate": optimizer_cfg.get("learning_rate", ""),
            **result,
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sharpness results to {args.output}")


if __name__ == "__main__":
    main()
