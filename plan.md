# Mini-Project Plan — Optimizer Geometry & Generalization

**Course:** CS-439 Optimization for Machine Learning (EPFL), Spring 2026
**Deliverable:** 3-page report (LaTeX) + accessible code repository
**Deadline:** June 12, 2026, 16:00 *(the handout has one stray mention of June 13 — confirm with the TA, but plan for the 12th)*

---

## 1. Research question

> **How do mini-batch size and optimizer adaptivity (SGD vs Adam) shape the curvature ("sharpness") of the minima a small CNN converges to, and how does that curvature relate to generalization?**

This matches a topic from the project handout — the *shape of local minima found by different optimizers, depending on step-size or mini-batch size, vs Adam or SGD* — and the *"optimization variants vs generalization"* theme.

We follow a **baseline → contribution** structure (what the grading criteria reward):

- **Baseline:** a well-tuned small-batch SGD run.
- **Axis 1 — batch size** (Keskar question): does larger batch → sharper minima → worse test accuracy?
- **Axis 2 — adaptivity** (Cohen question): does Adam reach higher-curvature regions, and is that "sharpness" genuine or an artifact of its preconditioning?

Every claim is a **with-vs-without** comparison: SGD vs Adam, small vs large batch, raw vs preconditioned curvature.

---

## 2. Background — the two papers we build on

- **Keskar et al., 2017 — *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.*** Large-batch training tends to converge to *sharper* minima, which correlate with *worse* generalization. → motivates the batch-size axis and the sharpness metric.
- **Cohen et al., 2022 — *Adaptive Gradient Methods at the Edge of Stability.*** For adaptive optimizers like Adam, the curvature that actually matters is that of the *preconditioned* loss, not the raw Hessian — so raw sharpness can be misleading for Adam. → motivates the adaptivity axis and our raw-vs-preconditioned comparison (the novelty).

**Combined story:** test the Keskar sharpness–generalization link across batch sizes, then check whether it survives the switch to Adam once we apply Cohen's preconditioning correction.

---

## 3. Hypotheses (predicted results)

- **H1:** as batch size grows, top Hessian curvature rises and test accuracy drops (Keskar trend).
- **H2:** at matched batch size, Adam reaches low train loss quickly but sits at higher *raw* curvature than SGD.
- **H3:** Adam's *preconditioned* curvature is much closer to SGD's — part of Adam's apparent sharpness is the preconditioner, not a genuinely sharper basin (Cohen).
- **H4:** sharpness vs test accuracy is negative but *imperfect / non-monotonic* — a nuanced "partially supported" result, not a clean law. (The handout explicitly values reporting what does not fully work.)

---

## 4. Tasks & expected outputs

### Core (committed)

| # | Task | Expected output |
|---|------|-----------------|
| 0 | **Fix & harden the code** — repair training bugs, save best + final checkpoints, measure geometry on a fixed *training* subset | A trustworthy, reproducible pipeline (single `run.py` / notebook) |
| 1 | **Batch-size sweep × {SGD, Adam}** — small CNN on Fashion-MNIST, ~5 batch sizes (e.g. 32–1024), 3 seeds, tuned LR; log train/test loss & accuracy | Results table + training curves + accuracy-vs-batch-size trend |
| 2 | **Sharpness + top Hessian eigenvalue** — power iteration via Hessian-vector products on a fixed train subset | CSVs + plots: curvature vs batch size, and curvature vs test accuracy |
| 3 | **Loss-interpolation plot** — loss along `w(α) = (1−α)·w₁ + α·w₂` between a flat (small-batch SGD) and a sharp (large-batch Adam) minimum | The "basin shape" figure |
| 4 | **Raw vs Adam-preconditioned curvature** (novelty) — recompute top curvature through Adam's preconditioner | Figure showing how much of Adam's "sharpness" is preconditioning |

> Task 4 is moderate difficulty. If it proves unreliable, fall back to a simpler proxy + a qualitative discussion of the preconditioning caveat.

### Optional (only if agreed + time allows)

| # | Task | Expected output |
|---|------|-----------------|
| 5 | **Second dataset — CIFAR-10** for breadth | Replicated trend plots on a harder dataset *(extra runtime + small model tweak)* |
| 6 | **Extra sharpness metric** (e.g. ε-sharpness at several radii) | A more robust geometry section |

---

## 5. Deliverables (mapped to the grading criteria)

- **3-page LaTeX report** using the official template: question → setup → results organized by claim (with the 4 figures) → conclusion + limitations + relation to Keskar/Cohen. Self-contained; references and appendix don't count toward the page limit.
- **Code + accessible GitHub repo:** the fixed pipeline, **one `run.py` / notebook that reproduces every figure and table** in the paper, and a rewritten **README** describing the pipeline and how to run it.
- **Reproducibility:** all hyperparameters reported and justified, seeds fixed, dataset download documented.

---

## 6. Compute

Experiments run on **Google Colab (free GPU)** — the small CNN on Fashion-MNIST trains quickly there, far faster than on a laptop CPU. The optional CIFAR-10 runs need more time. A single notebook reproduces all results and figures end to end.
