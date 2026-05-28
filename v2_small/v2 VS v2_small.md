## Comparison between `v2_small` and the full `v2` grid

### 1. Checking whether `v2_small` is representative

`v2_small` is a **4-run subset** of the full `v2` grid. The matching runs are identical in both experiments:

| Run                        | In `v2_small` | In full `v2` | Difference |
| -------------------------- | ------------: | -----------: | ---------: |
| `sgd_bs64_lr0.01_seed0`    |        92.21% |       92.21% |          0 |
| `sgd_bs512_lr0.05_seed0`   |        91.72% |       91.72% |          0 |
| `adam_bs64_lr0.001_seed0`  |        92.14% |       92.14% |          0 |
| `adam_bs512_lr0.001_seed0` |        91.56% |       91.56% |          0 |

This confirms that `v2_small` is not a separate experiment with different outcomes. It is a smaller selection from the same `v2` grid.

### 2. Comparing `v2_small` to the full hyperparameter grid

The full `v2` grid contains 12 runs:

| Optimizer | Batch size | Learning rates tested |
| --------- | ---------: | --------------------- |
| SGD       |         64 | 0.01, 0.05, 0.1       |
| SGD       |        512 | 0.01, 0.05, 0.1       |
| Adam      |         64 | 0.0001, 0.0005, 0.001 |
| Adam      |        512 | 0.0001, 0.0005, 0.001 |

The best full-grid results are:

| Rank | Run                        | Best test accuracy |
| ---: | -------------------------- | -----------------: |
|    1 | `sgd_bs64_lr0.01_seed0`    |         **92.21%** |
|    2 | `adam_bs64_lr0.001_seed0`  |         **92.14%** |
|    3 | `adam_bs64_lr0.0005_seed0` |         **92.03%** |
|    4 | `sgd_bs512_lr0.05_seed0`   |         **91.72%** |
|    5 | `adam_bs512_lr0.001_seed0` |         **91.56%** |

This shows that `v2_small` already contains the **best overall run**, the **second-best run**, and the best large-batch configurations for both SGD and Adam. The only notable strong configuration missing from `v2_small` is:

```text
adam_bs64_lr0.0005_seed0
```

which reached **92.03%** best test accuracy. This run is close to the top, but it still does not outperform the best configuration already included in `v2_small`.

### 3. Main trends observed in the full `v2` grid

The full grid confirms that **batch size 64 generally performs better than batch size 512**. The top three configurations all use batch size 64, while the best batch size 512 run reaches only **91.72%**.

The results also show that **SGD and Adam are very close when well tuned**. The best SGD configuration achieves **92.21%**, while the best Adam configuration achieves **92.14%**. The difference is only **0.07 percentage points**, so it is too small to claim a strong advantage for one optimizer over the other based on this single seed.

However, the full grid also shows that both optimizers are **sensitive to the learning rate**. For example, SGD with batch size 64 and learning rate 0.1 performs poorly, reaching only **89.75%** best test accuracy. Similarly, Adam with batch size 512 and learning rate 0.0001 reaches only **89.10%**, suggesting that this learning rate is too small for that setting.

There is also evidence that the **final epoch is not always the best checkpoint**. For example, `sgd_bs512_lr0.05_seed0` reaches **91.72%** best test accuracy but ends at only **90.41%** final test accuracy. This suggests some late-training degradation or overfitting, and it supports selecting the best checkpoint rather than simply using the final epoch.

### 4. Suggested report text

> The `v2_small` subset appears to be a good reduced representation of the full `v2` grid. It contains the best-performing configuration from the full grid, SGD with batch size 64 and learning rate 0.01, which achieved 92.21% best test accuracy. It also contains the strongest Adam configuration, Adam with batch size 64 and learning rate 0.001, which achieved 92.14%. The full grid adds several additional learning rates, including SGD learning rates 0.05 and 0.1 and Adam learning rates 0.0001 and 0.0005, but these additional runs do not change the main conclusion. Overall, the full `v2` grid confirms the pattern observed in `v2_small`: smaller batch size generally performs better, and the best SGD and Adam configurations are very close.

> The full grid also shows that hyperparameter tuning matters. Poorly chosen learning rates can noticeably reduce performance, as seen with SGD using learning rate 0.1 and Adam using learning rate 0.0001 with batch size 512. In addition, several runs achieve their best test accuracy before the final epoch, suggesting that checkpoint selection or early stopping is preferable to using the final epoch by default.

### Key conclusion

**`v2_small` did not miss the best configuration.** It already included the two strongest configurations from the full `v2` grid. Therefore, the full `v2` grid mainly strengthens the conclusions from `v2_small` rather than changing them.
