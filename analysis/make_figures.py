"""Reproduce Figures 1-3 and the full results table (Table 2) from the merged scalar results.

Run from the repository root:
    python analysis/make_figures.py

Reads : results/results_full.csv          (the 36-run merged scalars, committed to the repo)
Writes: results/figures/1_sharpness_vs_batchsize.png
        results/figures/2_accuracy_vs_batchsize.png
        results/figures/3_accuracy_vs_sharpness.png
Also prints the LaTeX body of the full results table (Table 2).

Dependencies: numpy, matplotlib (no GPU, runs instantly).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "results_full.csv"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- load the merged per-run scalars ---
rows = list(csv.DictReader(open(CSV_PATH)))
data = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (r["optimizer"], int(r["batch_size"]))
    data[key]["acc"].append(float(r["final_test_acc"]) * 100.0)   # store accuracy in %
    data[key]["th"].append(float(r["top_hessian"]))
    data[key]["sh"].append(float(r["sharpness"]))

BSS = [32, 64, 128, 256, 512, 1024]
COLORS = {"sgd": "#1f77b4", "adam": "#d62728"}


def series(opt, key):
    return (np.array([np.mean(data[(opt, b)][key]) for b in BSS]),
            np.array([np.std(data[(opt, b)][key]) for b in BSS]))


# --- Figure 1: top Hessian eigenvalue vs batch size (log y so both scales are visible) ---
plt.figure(figsize=(6, 4))
for opt, lab in [("sgd", "SGD"), ("adam", "Adam")]:
    m, s = series(opt, "th")
    plt.errorbar(BSS, m, yerr=s, marker="o", capsize=4, lw=2, label=lab, color=COLORS[opt])
plt.xscale("log", base=2); plt.yscale("log"); plt.xticks(BSS, BSS)
plt.xlabel("batch size"); plt.ylabel("top Hessian eigenvalue (log scale)")
plt.title("Sharpness vs batch size (SGD and Adam)")
plt.legend(); plt.grid(alpha=.3, which="both")
plt.savefig(OUT / "1_sharpness_vs_batchsize.png", dpi=150, bbox_inches="tight"); plt.close()

# --- Figure 2: test accuracy vs batch size ---
plt.figure(figsize=(6, 4))
for opt, lab in [("sgd", "SGD"), ("adam", "Adam")]:
    m, s = series(opt, "acc")
    plt.errorbar(BSS, m, yerr=s, marker="o", capsize=4, lw=2, label=lab, color=COLORS[opt])
plt.xscale("log", base=2); plt.xticks(BSS, BSS)
plt.xlabel("batch size"); plt.ylabel("test accuracy (%)")
plt.title("Test accuracy vs batch size (SGD and Adam)")
plt.legend(); plt.grid(alpha=.3)
plt.savefig(OUT / "2_accuracy_vs_batchsize.png", dpi=150, bbox_inches="tight"); plt.close()

# --- Figure 3: per-run test accuracy vs top Hessian eigenvalue (log x) ---
plt.figure(figsize=(6, 4))
for opt, lab in [("sgd", "SGD"), ("adam", "Adam")]:
    th = [v for b in BSS for v in data[(opt, b)]["th"]]
    ac = [v for b in BSS for v in data[(opt, b)]["acc"]]
    plt.scatter(th, ac, alpha=.75, label=lab, color=COLORS[opt], s=40)
plt.xscale("log")
plt.xlabel("top Hessian eigenvalue (log scale)"); plt.ylabel("test accuracy (%)")
plt.title("Test accuracy vs sharpness (SGD and Adam)")
plt.legend(); plt.grid(alpha=.3, which="both")
plt.savefig(OUT / "3_accuracy_vs_sharpness.png", dpi=150, bbox_inches="tight"); plt.close()

print("Saved figures 1-3 to", OUT)

# --- Table 2 (LaTeX body) for reference ---
def ms(v):
    return float(np.mean(v)), float(np.std(v))

print("\n% --- Table 2: full results (mean +/- std over 3 seeds) ---")
for opt, name in [("sgd", "SGD"), ("adam", "Adam")]:
    for b in BSS:
        th_m, th_s = ms(data[(opt, b)]["th"])
        sh_m, sh_s = ms(data[(opt, b)]["sh"])
        a_m, a_s = ms(data[(opt, b)]["acc"])
        print(f"{name} & {b} & {th_m:.2f} $\\pm$ {th_s:.2f} & "
              f"{sh_m * 1e3:.2f} $\\pm$ {sh_s * 1e3:.2f} & {a_m:.2f} $\\pm$ {a_s:.2f} \\\\")
