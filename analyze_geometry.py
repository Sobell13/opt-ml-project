import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

out_dir = Path("v2_final")
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

sharp = pd.read_csv(out_dir / "sharpness_results.csv")
hess = pd.read_csv(out_dir / "hessian_results.csv")

df = sharp.merge(
    hess[["checkpoint", "top_hessian_eigenvalue"]],
    on="checkpoint",
    how="inner",
)

print("\nCombined per-checkpoint geometry results:")
print(df)

df.to_csv(out_dir / "geometry_results.csv", index=False)
print("\nSaved:", out_dir / "geometry_results.csv")

summary = df.groupby(["optimizer", "batch_size", "learning_rate"])[
    ["base_loss", "sharpness", "top_hessian_eigenvalue"]
].agg(["mean", "std"])

print("\nGeometry summary by setting:")
print(summary)

summary.to_csv(out_dir / "geometry_summary.csv")
print("Saved:", out_dir / "geometry_summary.csv")

# Plot 1: sharpness by checkpoint
plt.figure()
df_sorted = df.sort_values("sharpness")
plt.bar(df_sorted["checkpoint"], df_sorted["sharpness"])
plt.xticks(rotation=90)
plt.ylabel("Random-direction sharpness")
plt.title("Sharpness by checkpoint")
plt.tight_layout()
plt.savefig(fig_dir / "sharpness_by_checkpoint.png", dpi=200)
plt.close()

# Plot 2: top Hessian eigenvalue by checkpoint
plt.figure()
df_sorted = df.sort_values("top_hessian_eigenvalue")
plt.bar(df_sorted["checkpoint"], df_sorted["top_hessian_eigenvalue"])
plt.xticks(rotation=90)
plt.ylabel("Top Hessian eigenvalue")
plt.title("Top Hessian eigenvalue by checkpoint")
plt.tight_layout()
plt.savefig(fig_dir / "hessian_by_checkpoint.png", dpi=200)
plt.close()

# Plot 3: sharpness vs Hessian
plt.figure()
plt.scatter(df["sharpness"], df["top_hessian_eigenvalue"])
for _, row in df.iterrows():
    label = f'{row["optimizer"]}_bs{row["batch_size"]}'
    plt.annotate(label, (row["sharpness"], row["top_hessian_eigenvalue"]), fontsize=7)
plt.xlabel("Random-direction sharpness")
plt.ylabel("Top Hessian eigenvalue")
plt.title("Sharpness vs Hessian curvature")
plt.tight_layout()
plt.savefig(fig_dir / "sharpness_vs_hessian.png", dpi=200)
plt.close()

print("\nSaved figures:")
print(fig_dir / "sharpness_by_checkpoint.png")
print(fig_dir / "hessian_by_checkpoint.png")
print(fig_dir / "sharpness_vs_hessian.png")
