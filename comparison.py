import json
import glob
import os
import pandas as pd

base = "./"


def load_summaries(exp):
    rows = []
    for path in glob.glob(f"{base}/{exp}/results/*_summary.json"):
        with open(path) as f:
            row = json.load(f)
        row["experiment"] = exp
        row["run"] = os.path.basename(path).replace("_summary.json", "")
        rows.append(row)
    return pd.DataFrame(rows)


small = load_summaries("v2_small")
full = load_summaries("v2")

# Compare matching runs
keys = ["optimizer", "batch_size", "learning_rate", "seed"]

matched = small.merge(full, on=keys, suffixes=("_small", "_v2"))

matched["best_acc_delta"] = (
    matched["best_test_accuracy_small"] - matched["best_test_accuracy_v2"]
)

print("Matching v2_small runs inside full v2:")
print(
    matched[
        keys
        + [
            "best_test_accuracy_small",
            "best_test_accuracy_v2",
            "best_acc_delta",
        ]
    ]
)

# Rank full grid
ranking = full.sort_values("best_test_accuracy", ascending=False)

print("\nFull v2 ranking:")
print(
    ranking[
        [
            "run",
            "optimizer",
            "batch_size",
            "learning_rate",
            "best_test_accuracy",
            "best_test_accuracy_epoch",
            "final_test_accuracy",
        ]
    ]
)
