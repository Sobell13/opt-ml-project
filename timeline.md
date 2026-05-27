Below is a complete timeline from **zero to final submission** for your mini-project on:

> **Sharpness and generalization in local minima found by SGD vs Adam, under different batch sizes and learning rates.**

This is designed for the EPFL mini-project format: a **3-page report**, code repository, reproducible experiments, fair baselines, and clear empirical evidence.

---

# Week 1 — Define the project and build the baseline

## Step 1 — Finalize the research question

Use a focused question:

> **How do optimizer choice, batch size, and learning rate affect the sharpness and generalization of neural-network minima?**

Your variables:

| Factor        | Values                                                       |
| ------------- | ------------------------------------------------------------ |
| Optimizer     | SGD, SGD + momentum, Adam                                    |
| Batch size    | small batch, large batch                                     |
| Learning rate | low, medium, high                                            |
| Metrics       | train loss, test accuracy, sharpness, Hessian top eigenvalue |

Do **not** try to reproduce both papers fully. Your goal is a smaller controlled study inspired by them.

---

## Step 2 — Choose dataset and model

Recommended simple setup:

| Component | Choice                                 |
| --------- | -------------------------------------- |
| Dataset   | Fashion-MNIST or CIFAR-10              |
| Model     | Small CNN                              |
| Framework | PyTorch                                |
| Runs      | 3 random seeds per setting if possible |

Best practical choice:

> **Fashion-MNIST + small CNN**

Why: fast, nontrivial, easy to train, and lets you run many optimizer/batch-size experiments.

Possible CNN:

```text
Conv(32) → ReLU → MaxPool
Conv(64) → ReLU → MaxPool
Flatten
Linear(128) → ReLU
Linear(10)
```

---

## Step 3 — Implement baseline training code

Create a repo structure like:

```text
project/
│
├── README.md
├── requirements.txt
├── run.py
├── train.py
├── models.py
├── sharpness.py
├── hessian.py
├── plots.py
├── configs/
│   ├── sgd_small.yaml
│   ├── sgd_large.yaml
│   ├── adam_small.yaml
│   └── adam_large.yaml
├── results/
└── figures/
```

Your first goal is simple:

> Train one CNN with SGD and one CNN with Adam, and save train/test loss curves.

Save every run with:

```text
optimizer
learning_rate
batch_size
seed
final_train_loss
final_test_loss
final_test_accuracy
checkpoint_path
```

---

# Week 2 — Run core experiments

## Step 4 — Define your main experiment grid

Start with a manageable grid:

| Optimizer      | Batch size | Learning rates   |
| -------------- | ---------: | ---------------- |
| SGD + momentum |         64 | 0.01, 0.05, 0.1  |
| SGD + momentum |        512 | 0.01, 0.05, 0.1  |
| Adam           |         64 | 1e-4, 5e-4, 1e-3 |
| Adam           |        512 | 1e-4, 5e-4, 1e-3 |

Use maybe **20 epochs** at first.

Then choose the best learning rates and rerun for **50 epochs**.

A reasonable final grid:

```text
SGD small batch
SGD large batch
Adam small batch
Adam large batch
```

with **3 seeds each**.

That gives 12 final runs, which is enough for a mini-project.

---

## Step 5 — Track standard metrics

For every epoch, log:

```text
train loss
train accuracy
test loss
test accuracy
learning rate
```

Your first plots should be:

1. train loss vs epoch;
2. test accuracy vs epoch;
3. final test accuracy by optimizer and batch size.

This gives you the baseline comparison required by the project instructions. The handout explicitly asks for solid baselines, clear quality measurements, and reproducible hyperparameter choices.

---

# Week 3 — Measure sharpness and geometry

## Step 6 — Implement Keskar-style sharpness

For a trained model with parameters ( w ), define sharpness approximately as:

[
\text{sharpness}(w) =
\max_{|\epsilon| \leq \rho}
L(w + \epsilon) - L(w)
]

Practical version:

1. take the final trained model;
2. sample random perturbations of the weights;
3. evaluate loss after perturbation;
4. record the maximum loss increase.

Use several radii:

```text
rho = 1e-4, 5e-4, 1e-3, 5e-3
```

For each trained model, compute:

```text
base train loss
max perturbed train loss
sharpness = max perturbed loss - base loss
```

Important: use the **same subset of training data** for all sharpness measurements, for example 2,000 examples.

---

## Step 7 — Implement top Hessian eigenvalue

Use PyTorch Hessian-vector products.

You do not need a full Hessian matrix. Use power iteration:

1. initialize random vector ( v );
2. compute Hessian-vector product ( Hv );
3. normalize;
4. repeat 20–50 times;
5. estimate:

[
\lambda_{\max} \approx v^T H v
]

Compute this on a small fixed subset, for example:

```text
1,000 or 2,000 training examples
```

This is directly aligned with the project description, which even points students toward Hessian-vector products as a useful tool.

---

## Step 8 — Optional but strong: interpolation plots

Train two models from the same initialization if possible:

```text
small-batch SGD model
large-batch Adam model
```

Then interpolate:

[
w(\alpha) = (1-\alpha)w_1 + \alpha w_2
]

for:

```text
alpha = 0.0, 0.05, 0.10, ..., 1.0
```

Plot:

```text
training loss along interpolation path
test loss along interpolation path
```

This gives a very visual “basin shape” result, similar in spirit to Keskar et al.

---

# Week 4 — Analyze results and refine

## Step 9 — Create final result tables

You want one compact table like:

| Optimizer | Batch size |  LR | Train loss | Test acc | Sharpness | Top Hessian eigenvalue |
| --------- | ---------: | --: | ---------: | -------: | --------: | ---------------------: |
| SGD       |         64 | ... |        ... |      ... |       ... |                    ... |
| SGD       |        512 | ... |        ... |      ... |       ... |                    ... |
| Adam      |         64 | ... |        ... |      ... |       ... |                    ... |
| Adam      |        512 | ... |        ... |      ... |       ... |                    ... |

Use mean ± standard deviation over seeds.

---

## Step 10 — Make the key plots

Aim for **4 final figures**:

### Figure 1 — Training dynamics

Train loss and test accuracy curves for the main settings.

Purpose:

> Show that all methods trained successfully.

---

### Figure 2 — Sharpness comparison

Bar plot:

```text
x-axis: optimizer + batch size
y-axis: sharpness
```

Purpose:

> Show whether large batch or Adam leads to sharper minima.

---

### Figure 3 — Generalization vs sharpness

Scatter plot:

```text
x-axis: sharpness
y-axis: test accuracy or test loss
```

Purpose:

> Show whether sharper solutions generalize worse in your setup.

---

### Figure 4 — Hessian top eigenvalue or interpolation plot

Choose one:

- Hessian top eigenvalue by optimizer/batch size;
- or loss interpolation between two minima.

Purpose:

> Give geometric evidence about the local minima.

---

# Week 5 — Write the report

The report must be short: **maximum 3 pages**, excluding references and appendix.

Use this structure:

## Title

**Sharpness and Generalization in SGD and Adam Minima**

---

## 1. Introduction

Say:

- deep networks can reach many different local minima;
- previous work suggests large-batch training finds sharper minima;
- Adam’s adaptive geometry may also affect curvature;
- your goal is to empirically compare SGD and Adam under controlled batch-size and learning-rate settings.

End with your main question:

> Do optimizer choice and batch size change the sharpness of the final solution, and does this correlate with generalization?

---

## 2. Experimental setup

Include:

```text
dataset
model architecture
optimizers
batch sizes
learning rates
number of epochs
number of seeds
sharpness metric
Hessian eigenvalue method
hardware, if relevant
```

This section is very important for reproducibility. The project handout explicitly emphasizes that the report and code should allow readers to reproduce your results.

---

## 3. Results

Organize by claim:

### Claim 1

Large batch reaches similar training loss but different test accuracy.

### Claim 2

Large batch / Adam finds sharper minima under your metric.

### Claim 3

Sharpness correlates with worse generalization, or does not.

Be honest if the result is mixed.

For example:

> We observe that large-batch Adam has the highest sharpness, but the relationship between sharpness and test accuracy is not perfectly monotonic. This suggests that sharpness is informative but not sufficient by itself to explain generalization.

That is a strong scientific conclusion.

---

## 4. Conclusion

Summarize:

- what you tested;
- what you found;
- how it relates to Keskar et al. and Cohen et al.;
- one limitation.

Example:

> Our experiments support the view that optimizer and batch size influence the geometry of the final solution. In our setting, sharper minima often correspond to worse generalization, although the relationship depends on the sharpness metric and learning rate.

---

# Final week — Clean code and prepare submission

## Step 11 — Make the code reproducible

Your `README.md` should include:

```bash
pip install -r requirements.txt
python run.py --config configs/sgd_small.yaml
python run.py --config configs/adam_large.yaml
python plots.py --results results/
```

Also include:

```text
Python version
PyTorch version
dataset download instructions
expected runtime
random seeds
```

The project requires documented executable code and a GitHub repository link.

---

## Step 12 — Final checklist

Before submission, verify:

| Item                                               | Done? |
| -------------------------------------------------- | ----- |
| Research question is clear                         | ☐     |
| Baselines are fair                                 | ☐     |
| Same model and dataset across methods              | ☐     |
| Same number of epochs or comparable compute budget | ☐     |
| Hyperparameters are reported                       | ☐     |
| At least 3 strong plots                            | ☐     |
| Mean ± std over seeds                              | ☐     |
| Sharpness metric clearly defined                   | ☐     |
| Hessian or interpolation analysis included         | ☐     |
| Code runs from README                              | ☐     |
| Report is ≤ 3 pages                                | ☐     |
| References and appendix separate                   | ☐     |
| GitHub repo accessible                             | ☐     |

---

# Suggested calendar

Assuming you start now:

| Period     | Goal                                              |
| ---------- | ------------------------------------------------- |
| Day 1      | Finalize question, dataset, model, repo structure |
| Days 2–3   | Implement training loop and baselines             |
| Days 4–5   | Run first SGD/Adam experiments                    |
| Days 6–7   | Tune learning rates and batch sizes               |
| Days 8–10  | Run final experiments with seeds                  |
| Days 11–12 | Implement sharpness metric                        |
| Days 13–14 | Implement Hessian eigenvalue or interpolation     |
| Days 15–16 | Generate final plots and tables                   |
| Days 17–18 | Write report draft                                |
| Days 19–20 | Clean code and README                             |
| Final day  | Proofread, check reproducibility, submit          |

---

# Minimal viable version

If time becomes tight, do this:

1. Fashion-MNIST.
2. Small CNN.
3. Four settings only:
   - SGD small batch;
   - SGD large batch;
   - Adam small batch;
   - Adam large batch.

4. Three metrics:
   - test accuracy;
   - perturbation sharpness;
   - train loss.

5. Three plots:
   - training curves;
   - sharpness bar plot;
   - sharpness vs test accuracy scatter.

That is already enough for a coherent mini-project.

# Strong version

If you have more time, add:

1. Hessian top eigenvalue.
2. Loss interpolation plots.
3. Three random seeds.
4. Adam preconditioned Hessian approximation.

The best final story would be:

> We compare SGD and Adam across batch sizes and learning rates. We find that optimizer choice and batch size affect the geometry of the final solution. In particular, sharper solutions often have worse generalization, but Adam’s adaptive preconditioning complicates the interpretation of raw Hessian sharpness.
