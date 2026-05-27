# Sharpness and generalization mini-project — Week 1 baseline

Goal for Week 1 Step 3:

> Train one CNN with SGD and one CNN with Adam, and save train/test loss curves.

This baseline uses Fashion-MNIST and a small CNN:

```text
Conv(32) -> ReLU -> MaxPool
Conv(64) -> ReLU -> MaxPool
Flatten
Linear(128) -> ReLU
Linear(10)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the two required baselines

```bash
python run.py --config configs/sgd_small.yaml --config configs/adam_small.yaml --make-plots
```

This creates:

```text
results/*_history.json
results/*_summary.json
results/*.pt
figures/loss_curves.png
```

Each summary JSON stores:

```text
optimizer
learning_rate
batch_size
seed
final_train_loss
final_test_loss
final_test_accuracy
checkpoint_path
history_path
```

## Run individual experiments

```bash
python run.py --config configs/sgd_small.yaml
python run.py --config configs/adam_small.yaml
python plots.py --results results/ --output figures/loss_curves.png
```

## Notes

- Fashion-MNIST downloads automatically into `./data`.
- Start with 10 epochs for quick verification. Increase `epochs` in the YAML files once everything runs.
- `sharpness.py` and `hessian.py` are placeholders for Week 3.
