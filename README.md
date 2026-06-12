# Sharpness, Batch Size, and Generalization: A Controlled Study of SGD and Adam

A controlled study of how **mini-batch size** and **optimizer choice** (SGD vs. Adam) shape the
*sharpness* (loss curvature) of the minima a small CNN reaches on Fashion-MNIST, and whether
sharpness predicts generalization.

EPFL CS-439 *Optimization for Machine Learning* mini-project.
**Authors:** Alexandre Carel, Ljubomir Ceranic, Hervé Sérandour.

## What this project studies

We fix a small convolutional network and Fashion-MNIST, then vary two factors one at a time:

- **Batch size:** 32, 64, 128, 256, 512, 1024
- **Optimizer:** SGD (lr 0.05, momentum 0.9) vs. Adam (lr 1e-3)

For each of the 36 runs (6 batch sizes x 2 optimizers x 3 seeds) we measure, at the trained
model:

- the **top Hessian eigenvalue**, via power iteration on Hessian-vector products,
- a **random-direction sharpness** probe,
- **test accuracy**,

and, for a representative SGD/Adam pair, the **loss along the linear interpolation** between the
two minima, plus Adam's curvature in its **preconditioned geometry** (following Cohen et al.).

The questions: does batch size or optimizer change the sharpness of the minimum reached, and
does sharpness track generalization?

## Key findings

- **Sharpness rises with batch size for both optimizers.** The top Hessian eigenvalue grows
  about 23x for SGD (~1.5 to ~33 from batch 32 to 1024) and similarly for Adam (~23 to ~141),
  reproducing the Keskar et al. large-batch / sharp-minima effect.
- **Adam reaches sharper minima than SGD** at every batch size (roughly 4x to 30x the raw top
  eigenvalue, the gap widest at small batch), yet
- **test accuracy stays in a narrow band (~89 to 92%)** across the entire sharpness range. Raw
  top-Hessian sharpness is therefore a *weak* predictor of generalization within an optimizer
  and an *unreliable* one across optimizers.
- **The raw Hessian is the wrong ruler for Adam.** In Adam's own preconditioned geometry the top
  eigenvalue is ~50 to 90x larger than the raw value and climbs toward the edge-of-stability
  scale 38/eta, so cross-optimizer raw-Hessian comparisons are not meaningful (Cohen et al.).

The full write-up is in [`report/`](report/).

## Repository layout

The final pipeline is the two notebooks at the repository root. Everything else is either
supporting data or earlier pilot exploration kept for history.

```text
.
├── optgeo_experiments.ipynb              # Main pipeline: full 36-run sweep + geometry + interpolation
├── optgeo_experiments_second_part.ipynb  # Remaining Adam runs + a flat-minimum checkpoint (has a TEST_MODE smoke test)
├── analysis/                             # Scripts that regenerate the paper's figures and tables from the results
├── report/                               # Final 3-page report (PDF + LaTeX source)
├── results/
│   ├── results_full.csv                  # Merged 36-run scalar results (SGD + Adam) used for the figures/tables
│   ├── results_salvaged.csv / .json      # Scalars recovered after a Colab disconnect (see note below)
│   └── figures/                          # The five figures used in the report
├── requirements.txt
│
├── run.py, train.py, models.py,          # Week-1 pilot framework (standalone scripts),
│   hessian.py, sharpness.py,             #   superseded by the notebooks above
│   plots.py, comparison.py, analyze_*.py
├── configs/                              # YAML configs for the pilot framework
└── v1/ v2/ v2_final/ v2_small/ v3/       # Early exploration: pilot runs, figures, and notes
```

The `results_geometry_part2_FINAL.zip` archive at the root bundles the trained models from the
second-notebook run (model weights with optimizer state), kept so the geometry measurements can
be reproduced without retraining.

## Reproducing the results

The experiments were run on Google Colab with a GPU. The geometry measurements (Hessian-vector
products and power iteration) are the expensive part, so a GPU is strongly recommended.

### 1. Environment

```bash
pip install -r requirements.txt
```

`requirements.txt` pins PyTorch with CUDA; on Colab the GPU build is already present. The core
dependencies are `torch`, `torchvision`, `numpy`, `pandas`, and `matplotlib`. Fashion-MNIST
downloads automatically on first run.

### 2. Run the experiments

Open the notebooks in Colab (or Jupyter with a GPU) and run them top to bottom:

1. **`optgeo_experiments.ipynb`** runs the full sweep, computes the geometry measurements, and
   saves scalar results, model checkpoints (with optimizer state), and an incremental backup as
   it goes.
2. **`optgeo_experiments_second_part.ipynb`** runs the remaining Adam configurations and the
   interpolation checkpoint. Set `TEST_MODE = False` for the real run; the default `TEST_MODE`
   path is a fast synthetic CPU smoke test that checks the pipeline end to end. The notebook
   downloads a results archive to your machine when it finishes.

> **Note on the salvaged data.** During the original full run, a Colab disconnect wiped the
> checkpoints mid-job. The scalar results for the affected runs were recovered from the notebook
> output and stored as `results/results_salvaged.*`. `results/results_full.csv` is the merge of
> those salvaged SGD scalars with the Adam results from the second notebook, and is what the
> figures and report tables are built from.

### 3. Regenerating the figures and tables

The figures and tables in the report are produced from the stored results by the scripts in
`analysis/`, run from the repository root:

```bash
python analysis/make_figures.py              # Figures 1-3 and the full results table (Table 2)
python analysis/preconditioned_curvature.py  # Figure 4 and the raw/preconditioned numbers (Table 3)
python analysis/interpolation_figure.py      # Figure 5
```

`make_figures.py` needs only `results/results_full.csv`. The other two read the trained
checkpoints, auto-extracting `results_geometry_part2_FINAL.zip` on first run, and
`preconditioned_curvature.py` additionally downloads Fashion-MNIST via torchvision. All three
write into `results/figures/`. The five figures map to the report as:

| File | Content |
| --- | --- |
| `1_sharpness_vs_batchsize.png` | Top Hessian eigenvalue vs. batch size (SGD and Adam) |
| `2_accuracy_vs_batchsize.png`  | Test accuracy vs. batch size |
| `3_accuracy_vs_sharpness.png`  | Test accuracy vs. top Hessian eigenvalue |
| `4_raw_vs_preconditioned.png`  | Adam's raw vs. preconditioned curvature |
| `5_interpolation.png`          | Training loss along the SGD-to-Adam interpolation |

## References

1. N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, P. T. P. Tang.
   *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR, 2017.
2. J. M. Cohen, B. Ghorbani, S. Krishnan, N. Agarwal, et al.
   *Adaptive Gradient Methods at the Edge of Stability.* arXiv:2207.14484, 2022.
