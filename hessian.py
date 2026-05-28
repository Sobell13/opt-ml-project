"""Top-Hessian-eigenvalue estimation for trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from sharpness import load_checkpoint_model
from train import build_dataloaders, get_device, set_seed


def take_batches(loader: Iterable, max_batches: int):
    batches = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(batch)

    if not batches:
        raise ValueError("No batches available for Hessian estimation.")
    return batches


def flat_grad(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.contiguous().view(-1) for g in grads])


def hessian_vector_product(model, criterion, batches, vector, device):
    params = [p for p in model.parameters() if p.requires_grad]
    hvp_sum = torch.zeros_like(vector)

    for x, y in batches:
        model.zero_grad(set_to_none=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss = criterion(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_vector = flat_grad(grads)

        grad_dot_vector = torch.dot(grad_vector, vector)
        hvp = torch.autograd.grad(grad_dot_vector, params)

        hvp_sum += flat_grad(hvp).detach()

    return hvp_sum / len(batches)


def estimate_top_hessian_eigenvalue(
    model,
    loader,
    device,
    max_batches: int = 5,
    power_iters: int = 20,
    seed: int = 0,
) -> dict:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    batches = take_batches(loader, max_batches)

    n_params = parameters_to_vector(
        [p for p in model.parameters() if p.requires_grad]
    ).numel()

    vector = torch.randn(n_params, generator=generator, device=device)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)

    eigenvalue = None
    for _ in range(power_iters):
        hvp = hessian_vector_product(model, criterion, batches, vector, device)
        vector = hvp / torch.linalg.vector_norm(hvp).clamp_min(1e-12)
        eigenvalue = torch.dot(
            vector,
            hessian_vector_product(model, criterion, batches, vector, device),
        )

    return {
        "top_hessian_eigenvalue": float(eigenvalue.item()),
        "max_batches": max_batches,
        "power_iters": power_iters,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate top Hessian eigenvalue for trained checkpoints."
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
        "--max-batches",
        type=int,
        default=5,
        help="Number of batches used for Hessian estimation.",
    )
    parser.add_argument(
        "--power-iters",
        type=int,
        default=20,
        help="Number of power-iteration steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    checkpoint_paths = sorted(args.checkpoint_dir.glob("*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pt checkpoints found in {args.checkpoint_dir}")

    rows = []

    for checkpoint_path in checkpoint_paths:
        print(f"Estimating Hessian eigenvalue: {checkpoint_path}")

        model, config = load_checkpoint_model(checkpoint_path, device)

        training_cfg = config.get("training", {})
        optimizer_cfg = config.get("optimizer", {})

        train_loader, test_loader = build_dataloaders(config)

        result = estimate_top_hessian_eigenvalue(
            model=model,
            loader=test_loader,
            device=device,
            max_batches=args.max_batches,
            power_iters=args.power_iters,
            seed=args.seed,
        )

        row = {
            "checkpoint": checkpoint_path.name,
            "optimizer": optimizer_cfg.get("name", ""),
            "batch_size": training_cfg.get("batch_size", ""),
            "learning_rate": optimizer_cfg.get(
                "learning_rate", optimizer_cfg.get("lr", "")
            ),
            **result,
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Hessian results to {args.output}")


if __name__ == "__main__":
    main()
