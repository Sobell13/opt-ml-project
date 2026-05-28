import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

out_dir = Path("v2_final")
results_dir = out_dir / "results"
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

geometry = pd.read_csv(out_dir / "geometry_results.csv")

rows = []
for path in sorted(results_dir.glob("*_summary.json")):
    with path.open() as f:
        s = json.load(f)

    rows.append(
        {
            "checkpoint": Path(s["checkpoint_path"]).name,
            "run_name": s["run_name"],
            "optimizer": s["optimizer"],
            "batch_size": s["batch_size"],
            "learning_rate": s["learning_rate"],
            "seed": s["seed"],
            "best_test_accuracy": s["best_test_accuracy"],
            "final_test_accuracy": s["final_test_accuracy"],
            "best_test_loss": s["best_test_loss"],
        }
    )

accuracy = pd.DataFrame(rows)

df = accuracy.merge(
    geometry[["checkpoint", "sharpness", "top_hessian_eigenvalue"]],
    on="checkpoint",
    how="inner",
)

df.to_csv(out_dir / "accuracy_geometry_results.csv", index=False)

summary = df.groupby(["optimizer", "batch_size", "learning_rate"])[
    ["best_test_accuracy", "final_test_accuracy", "sharpness", "top_hessian_eigenvalue"]
].agg(["mean", "std"])

summary.to_csv(out_dir / "accuracy_geometry_summary.csv")

print("\nPer-checkpoint accuracy + geometry:")
print(
    df[
        [
            "run_name",
            "best_test_accuracy",
            "sharpness",
            "top_hessian_eigenvalue",
        ]
    ]
)

print("\nSummary by setting:")
print(summary)

# Accuracy vs sharpness
plt.figure()
plt.scatter(df["sharpness"], df["best_test_accuracy"])
for _, row in df.iterrows():
    label = f'{row["optimizer"]}_bs{row["batch_size"]}_s{row["seed"]}'
    plt.annotate(label, (row["sharpness"], row["best_test_accuracy"]), fontsize=7)
plt.xlabel("Random-direction sharpness")
plt.ylabel("Best test accuracy")
plt.title("Best test accuracy vs sharpness")
plt.tight_layout()
plt.savefig(fig_dir / "accuracy_vs_sharpness.png", dpi=200)
plt.close()

# Accuracy vs Hessian
plt.figure()
plt.scatter(df["top_hessian_eigenvalue"], df["best_test_accuracy"])
for _, row in df.iterrows():
    label = f'{row["optimizer"]}_bs{row["batch_size"]}_s{row["seed"]}'
    plt.annotate(
        label, (row["top_hessian_eigenvalue"], row["best_test_accuracy"]), fontsize=7
    )
plt.xlabel("Top Hessian eigenvalue")
plt.ylabel("Best test accuracy")
plt.title("Best test accuracy vs Hessian curvature")
plt.tight_layout()
plt.savefig(fig_dir / "accuracy_vs_hessian.png", dpi=200)
plt.close()

print("\nSaved:")
print(out_dir / "accuracy_geometry_results.csv")
print(out_dir / "accuracy_geometry_summary.csv")
print(fig_dir / "accuracy_vs_sharpness.png")
print(fig_dir / "accuracy_vs_hessian.png")
