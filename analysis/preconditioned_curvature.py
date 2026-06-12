"""Reproduce Figure 4 and the Table 3 numbers: Adam's raw vs preconditioned top eigenvalue.

Following Cohen et al. (2022), Adam moves in a preconditioned geometry
    P = diag(sqrt(v_hat) + eps),
where v_hat is Adam's bias-corrected second moment. The curvature Adam actually navigates is
the top eigenvalue of the *preconditioned* Hessian P^{-1/2} H P^{-1/2}, not the raw Hessian.
This script computes both for the trained Adam models and plots them against Adam's
edge-of-stability scale 38/lr.

Run from the repository root:
    python analysis/preconditioned_curvature.py

Reads : results_geometry_part2_FINAL.zip   (auto-extracted; provides the Adam checkpoints)
        Fashion-MNIST                      (downloaded via torchvision on first run)
Writes: results/figures/4_raw_vs_preconditioned.png
Also prints the raw / preconditioned eigenvalues and their ratio (Table 3) per batch size.

Dependencies: torch, torchvision, matplotlib. Runs on CPU in a few minutes (the probe batch
is kept small on purpose so the Hessian-vector products stay tractable).
"""
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import grad
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "results_geometry_part2_FINAL.zip"
WORK = ROOT / "results_geometry_part2"          # the archive is extracted here on first run
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
N = 128                       # probe-batch size; small so the CPU HVPs stay tractable
LR_SGD, LR_ADAM = 0.05, 1e-3
torch.manual_seed(0)          # deterministic power-iteration start

if not (WORK / "checkpoints").exists():
    print("Extracting", ZIP.name, "->", WORK)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(WORK)
CKPT = WORK / "checkpoints"


class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def tparams(m):
    return [p for p in m.parameters() if p.requires_grad]


def raw_hvp(model, crit, x, y, vec, params):
    g = grad(crit(model(x), y), params, create_graph=True)
    flat = torch.cat([gi.reshape(-1) for gi in g])
    hv = grad((flat * vec).sum(), params)
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def top_eig(model, crit, x, y, iters=10, dinv_half=None):
    """Top eigenvalue of H (dinv_half=None) or of the preconditioned Hessian
    P^{-1/2} H P^{-1/2} (dinv_half = diag(P)^{-1/2}), via power iteration."""
    params = tparams(model)
    n = sum(p.numel() for p in params)
    v = torch.randn(n); v /= v.norm()
    for _ in range(iters):
        w = dinv_half * v if dinv_half is not None else v
        Hw = raw_hvp(model, crit, x, y, w, params)
        u = dinv_half * Hw if dinv_half is not None else Hw
        nv = u.norm()
        if nv == 0:
            break
        v = u / nv
    w = dinv_half * v if dinv_half is not None else v
    Hw = raw_hvp(model, crit, x, y, w, params)
    u = dinv_half * Hw if dinv_half is not None else Hw
    return float((v * u).sum().item())


def build_dinv_half(model, optim_state):
    """diag(P)^{-1/2} from Adam's saved second moment: P = diag(sqrt(v_hat) + eps)."""
    pg = optim_state["param_groups"][0]
    beta2, eps = pg["betas"][1], pg["eps"]
    state = optim_state["state"]
    Ds = []
    for i, _ in enumerate(tparams(model)):
        st = state[i]
        v = st["exp_avg_sq"].detach().cpu().reshape(-1).float()
        step = st["step"]
        step = float(step.item()) if torch.is_tensor(step) else float(step)
        vhat = v / (1 - beta2 ** step)
        Ds.append(torch.sqrt(vhat) + eps)
    D = torch.cat(Ds)
    return (1.0 / torch.sqrt(D)).float()


# --- probe batch: first N Fashion-MNIST training images, normalized exactly as in training ---
tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
ds = torchvision.datasets.FashionMNIST(root=str(ROOT / "data"), train=True, download=True, transform=tf)
gx = torch.stack([ds[i][0] for i in range(N)]).to(DEVICE)
gy = torch.tensor([ds[i][1] for i in range(N)]).to(DEVICE)
crit = nn.CrossEntropyLoss()
print(f"probe batch {tuple(gx.shape)};  EoS refs -> SGD 2/lr={2 / LR_SGD:.0f}, "
      f"Adam 38/lr={38 / LR_ADAM:.0f}\n")

print(f"{'bs':>5} | {'raw top_H':>10} | {'precond top_H':>14} | {'ratio':>6}")
print("-" * 46)
out = {}
for bs in [64, 128, 256, 512, 1024]:
    d = torch.load(CKPT / f"FashionMNIST_adam_bs{bs}_seed0.pt", map_location="cpu", weights_only=False)
    model = SmallCNN(1).to(DEVICE); model.load_state_dict(d["model_final"]); model.eval()
    dinv_half = build_dinv_half(model, d["optim_state"])
    raw = top_eig(model, crit, gx, gy)
    pre = top_eig(model, crit, gx, gy, dinv_half=dinv_half)
    out[bs] = dict(raw=raw, precond=pre)
    print(f"{bs:>5} | {raw:10.2f} | {pre:14.1f} | {pre / raw:5.0f}x")

# --- Figure 4 ---
bss = [64, 128, 256, 512, 1024]
plt.figure(figsize=(6, 4))
plt.plot(bss, [out[b]["raw"] for b in bss], marker="o", lw=2,
         label="raw Hessian", color="#1f77b4")
plt.plot(bss, [out[b]["precond"] for b in bss], marker="s", lw=2,
         label="preconditioned (Adam's geometry)", color="#d62728")
plt.axhline(38 / LR_ADAM, ls="--", c="gray", lw=1.5, label="Adam edge-of-stability (38/lr)")
plt.xscale("log", base=2); plt.yscale("log"); plt.xticks(bss, bss)
plt.xlabel("batch size"); plt.ylabel("top eigenvalue (log scale)")
plt.title("Adam: raw vs preconditioned curvature")
plt.legend(fontsize=8); plt.grid(alpha=.3, which="both")
plt.savefig(OUT / "4_raw_vs_preconditioned.png", dpi=150, bbox_inches="tight")
print("\nSaved figure 4 to", OUT / "4_raw_vs_preconditioned.png")
