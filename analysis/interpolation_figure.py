"""Reproduce Figure 5: training loss along the linear interpolation between a small-batch
SGD minimum (alpha = 0) and a large-batch Adam minimum (alpha = 1).

Run from the repository root:
    python analysis/interpolation_figure.py

Reads : results_geometry_part2_FINAL.zip   (auto-extracted; provides interpolation.json)
Writes: results/figures/5_interpolation.png

Dependencies: matplotlib (no GPU).
"""
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "results_geometry_part2_FINAL.zip"
WORK = ROOT / "results_geometry_part2"          # the archive is extracted here on first run
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

if not (WORK / "interpolation.json").exists():
    print("Extracting", ZIP.name, "->", WORK)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(WORK)

d = json.load(open(WORK / "interpolation.json"))
alphas, losses = d["alphas"], d["losses"]

plt.figure(figsize=(6, 4))
plt.plot(alphas, losses, marker="o", lw=2, color="#6a3d9a")
plt.axvline(0.0, ls="--", c="gray", lw=1)
plt.axvline(1.0, ls="--", c="gray", lw=1)
plt.text(0.0, min(losses), " SGD (B=32)", va="bottom", ha="left", fontsize=8)
plt.text(1.0, min(losses), " Adam (B=1024)", va="bottom", ha="left", fontsize=8)
plt.yscale("log")
plt.xlabel(r"interpolation coefficient $\alpha$")
plt.ylabel("training loss (log scale)")
plt.title("Loss along SGD-to-Adam interpolation")
plt.grid(alpha=.3, which="both")
plt.savefig(OUT / "5_interpolation.png", dpi=150, bbox_inches="tight")
print("Saved figure 5 to", OUT / "5_interpolation.png")
