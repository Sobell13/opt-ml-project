Yes — this is a **strong base**, and it fits the project topic very well. Your chosen theme, **“Local minima for deep learning,”** is explicitly listed as a suggested direction in the mini-project description: comparing the “shape” of local minima found by different optimizers, step sizes, and mini-batch sizes.

Your two papers also connect nicely:

**Keskar et al. 2017** gives the classic story:
large batches tend to converge to **sharper minima**, and this correlates with worse generalization.

**Cohen et al. 2022** gives a more modern optimizer-specific story:
Adam can enter/stay near high-curvature regions because its adaptive preconditioning changes the relevant notion of curvature.

Together, they let you ask a focused and interesting question:

> Do batch size and optimizer adaptivity control the sharpness of the final minimum, and does this explain generalization differences?

That is a good project question because it is empirical, optimization-focused, and feasible in PyTorch.

## What you can do from this

I would not try to fully reproduce both papers. That is too large for a 3-page mini-project. Instead, use them as motivation and run a smaller controlled experiment.

A good project could be:

### Main experiment

Train the same small neural network on the same dataset using:

| Optimizer             |     Batch size | Learning rate variants |
| --------------------- | -------------: | ---------------------- |
| SGD or SGD + momentum | small vs large | low / medium / high    |
| Adam                  | small vs large | low / medium / high    |

For each trained model, measure:

1. **Train loss**
2. **Test accuracy / test loss**
3. **Sharpness of the final solution**
4. Possibly **top Hessian eigenvalue** or an approximation

This directly addresses the mini-project requirement to provide empirical evidence about optimizer behavior and compare baselines fairly. The project instructions emphasize empirical comparison, baselines, reproducibility, and explaining what insight your experiments add beyond existing work.

## A concrete research question

Use something like:

> How do batch size and adaptive preconditioning affect the sharpness and generalization of neural-network minima?

Or more sharply:

> Does Adam find sharper minima than SGD under matched training loss, and how does this depend on batch size and learning rate?

This is better than just saying “we study local minima,” because it gives you a testable claim.

## Recommended setup

Keep the setup simple so you can run enough repetitions.

Good choices:

**Dataset:** MNIST, Fashion-MNIST, CIFAR-10 subset, or full CIFAR-10 if compute allows.
**Model:** small CNN or MLP.
**Optimizers:** SGD with momentum and Adam.
**Batch sizes:** for example 32 and 512, or 64 and 1024.
**Runs:** ideally 3 seeds per setting.

For sharpness, use one or two practical metrics:

1. **Keskar-style ε-sharpness approximation**
   Perturb parameters inside a small box or ball and measure the maximum increase in loss.

2. **Top Hessian eigenvalue**
   Use Hessian-vector products and power iteration or Lanczos. This is directly relevant because the project handout even points students toward Hessian-vector products.

3. **Loss interpolation plots**
   Interpolate between two trained solutions, e.g. small-batch SGD and large-batch Adam, and plot the loss along the line. This is visually strong and close to Keskar et al.

## Best project angle

The most interesting version would be:

> Compare raw sharpness and preconditioned sharpness.

For Adam, Cohen et al. suggests that the raw Hessian may not be the whole story; the relevant geometry is affected by Adam’s preconditioner. So you can measure:

- raw Hessian top eigenvalue, approximately;
- Adam-preconditioned Hessian top eigenvalue, if feasible;
- simpler fallback: compare raw sharpness for SGD vs Adam and discuss the limitation.

That gives your project more novelty than simply “large batch gives sharp minima.”

## Possible plots for the report

You only have 3 pages, so aim for 3–4 strong figures:

1. **Train/test loss curves** for SGD vs Adam, small vs large batch.
2. **Test accuracy vs sharpness scatter plot.**
3. **Sharpness bar plot** by optimizer and batch size.
4. **1D loss interpolation plot** between two minima.

The report should tell a clear story:

> Large batch and/or Adam can reach solutions with similar training loss but different curvature. We test whether sharper solutions correlate with worse test performance, and whether Adam behaves differently from SGD.

## What to avoid

Avoid making the project too broad. Do **not** try to cover 8 optimizers, many architectures, and full Hessian analysis. The grading asks for rigorous and fair empirical comparison, not a huge reproduction study.

Also avoid claiming “sharp minima always generalize worse.” That statement is controversial because sharpness depends on parameter scaling and the metric used. A safer claim is:

> Under our chosen sharpness measure and controlled setup, we observe whether sharper solutions correlate with worse generalization.

## Suggested final project title

**“Batch Size, Adam, and the Geometry of Neural Network Minima”**

or

**“Sharpness and Generalization in SGD and Adam: A Small-Scale Empirical Study”**

Overall: yes, this is a good base. The best next step is to turn the two papers into one focused empirical question, then run a small but clean comparison of **SGD vs Adam × small batch vs large batch × sharpness/generalization**.
