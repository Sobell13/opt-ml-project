| Rank | Run                        | Optimizer | Batch size |    LR | Best test acc. | Best epoch | Final test acc. | Comment                                                        |
| ---: | -------------------------- | --------: | ---------: | ----: | -------------: | ---------: | --------------: | -------------------------------------------------------------- |
|    1 | `sgd_bs64_lr0.01_seed0`    |       SGD |         64 |  0.01 |     **92.21%** |         17 |          91.90% | Best overall; strong but starts overfitting after mid-training |
|    2 | `adam_bs64_lr0.001_seed0`  |      Adam |         64 | 0.001 |     **92.14%** |         12 |          91.68% | Nearly tied with SGD; reaches peak earlier                     |
|    3 | `sgd_bs512_lr0.05_seed0`   |       SGD |        512 |  0.05 |         91.72% |         17 |          90.41% | Good peak, but largest late degradation                        |
|    4 | `adam_bs512_lr0.001_seed0` |      Adam |        512 | 0.001 |         91.56% |         17 |          91.25% | Most stable large-batch run, but lower peak                    |

## Main comments on `v2_small`

The best result is **SGD with batch size 64 and learning rate 0.01**, reaching **92.21% test accuracy**. Adam with batch size 64 is essentially tied at **92.14%**, so the difference is very small and should not be overinterpreted from a single seed.

The **small batch size 64 clearly performs better than batch size 512** for both optimizers. For SGD, batch size 64 beats batch size 512 by about **0.49 percentage points** in best test accuracy. For Adam, batch size 64 beats batch size 512 by about **0.58 percentage points**.

There is visible **overfitting or late-training degradation**. In every run, the final training accuracy keeps rising, but the final test accuracy is below the best test accuracy. The clearest case is `sgd_bs512_lr0.05_seed0`, which drops from **91.72% best** to **90.41% final**, a **1.31 point** decrease.

The best test losses happen earlier than or around the best accuracies. For example, `sgd_bs64_lr0.01_seed0` has its best test loss at epoch **9**, but best test accuracy at epoch **17**. This suggests the model may keep improving classification decisions while becoming less well-calibrated or more confident, so accuracy and loss do not peak at exactly the same time.

## Suggested interpretation

> In the `v2_small` experiments, the best-performing configuration was SGD with batch size 64 and learning rate 0.01, achieving 92.21% best test accuracy. Adam with batch size 64 and learning rate 0.001 was very close at 92.14%, indicating that both optimizers can reach similar performance when the batch size is small. Larger batch size 512 produced slightly worse best accuracy for both optimizers. All runs show some degree of late-training overfitting: training accuracy continues to increase while test loss worsens and final test accuracy falls below the best checkpoint. This suggests that checkpoint selection or early stopping around epochs 12–17 is preferable to simply using the final epoch.
