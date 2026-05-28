## Main results

Grouped by optimizer and batch size:

| Setting                    | Mean best test accuracy | Std. dev. | Mean final test accuracy |
| -------------------------- | ----------------------: | --------: | -----------------------: |
| **SGD, batch 64, lr 0.01** |              **92.52%** |     0.09% |               **92.03%** |
| Adam, batch 512, lr 0.001  |                  92.31% |     0.17% |                   91.45% |
| Adam, batch 64, lr 0.001   |                  92.21% |     0.05% |                   91.53% |
| SGD, batch 512, lr 0.05    |                  92.16% |     0.30% |                   91.71% |

The best individual run is:

```text
sgd_bs64_lr0.01_seed1
best test accuracy: 92.62%
best epoch: 41
final test accuracy: 92.16%
```

## Comments on v2_final

The most important conclusion is that **SGD with small batch size is your strongest and most stable configuration**. It has the highest average best accuracy and very low seed-to-seed variation.

Adam performs well, but it does not clearly beat SGD. In fact, both Adam settings end around the same level or slightly below SGD. This is useful for your project because it gives you a clean comparison:

```text
SGD small batch: best generalization
Adam small/large batch: competitive but slightly worse
SGD large batch: decent but less stable
```

Another important pattern is that many runs peak before epoch 50. For example:

```text
adam_bs512_lr0.001_seed0: best epoch 34, final accuracy lower
adam_bs64_lr0.001_seed0: best epoch 30, final accuracy lower
sgd_bs64_lr0.01_seed1: best epoch 41, final accuracy lower
```

So our final checkpoints are not always the best-performing checkpoints. This matters for Week 3.

## One weakness to fix

Your `.pt` files appear to be final-epoch checkpoints, not best-epoch checkpoints.

That is acceptable if you explicitly say:

> Sharpness and Hessian are measured at the final trained model after 50 epochs.

But I would recommend saving best checkpoints too, because your summaries already show that best accuracy often happens before epoch 50.

Ideal files would be:

```text
sgd_bs64_lr0.01_seed1_final.pt
sgd_bs64_lr0.01_seed1_best.pt
```

If changing the training code is too much right now, continue with the existing final checkpoints and document that choice.

Best V2 model family is:

```text
SGD, batch size 64, learning rate 0.01
```

## Final note

SGD with small batch size gives the best generalization.
Next, we test whether this solution is flatter by measuring sharpness and Hessian curvature.
