#!/usr/bin/env python3
"""
Experiment A/B: is a real model class twin-rich, and does scaling shrink it?

For a fixed record (a random subset of the input universe with the intended
labels), train S independent models of a given size from scratch.  Two models
are twins on the record when they agree there but disagree off it.  The
measurable quantity is

    pi_hat = mean over model pairs of  Pr_{x off the record}[f(x) != g(x)],

the empirical worst-case separation of the trained class (pi_min in the
paper).  If pi_hat stays positive as the record grows, the trained class is
twin-rich on this task; if pi_hat collapses to zero quickly, the record
determines the function and the premise fails on this task.  Either outcome
is a result.

Design:
  * task: (a + b) mod 10 over a in 0..15, b in 0..15 (256 inputs), which is
    larger than the 0..9 universe used in the main text's demonstration and
    therefore a step toward an open input domain;
  * record sizes N in {16, 64, 128, 192}, three independent random record
    subsets per N (data splits, not just seeds);
  * model sizes small/medium/large (width/depth), five seeds each;
  * measured on the off-record inputs (the complement of the record).

Outputs: ../data/twin_richness.csv and ../data/twin_richness_summary.csv,
plus ../charts/fig7_twin_richness.{pdf,png}.

Run: python3 twin_richness_experiment.py
CPU only; deterministic given the seed grid.
"""
import csv
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
CHARTS = BASE / "charts"

PLUS, EQ = 16, 17
VOCAB = 18
BLOCK = 4
VMAX = 16
UNIVERSE = [(a, b) for a in range(VMAX) for b in range(VMAX)]
SIGMA = {(a, b): (a + b) % 10 for a, b in UNIVERSE}

RECORD_SIZES = [16, 64, 128, 192]
SPLITS = [0, 1, 2]
SEEDS = [0, 1, 2, 3, 4]
SIZES = {
    "small": dict(d=32, layers=1, heads=2, steps=1000),
    "medium": dict(d=64, layers=2, heads=4, steps=600),
    "large": dict(d=96, layers=3, heads=4, steps=600),
}


def record_mask(n, split):
    rng = np.random.default_rng(1000 * n + split)
    idx = rng.choice(len(UNIVERSE), size=n, replace=False)
    mask = np.zeros(len(UNIVERSE), dtype=bool)
    mask[idx] = True
    return mask


def train_one(job):
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(1)
    n, split, size_label, seed = job
    cfg = SIZES[size_label]
    seed_all = 10_000_000 + 100_000 * n + 1000 * split + 10 * SEEDS.index(seed) + list(SIZES).index(size_label)
    torch.manual_seed(seed_all)
    np.random.seed(seed_all % (2**32 - 1))
    random.seed(seed_all)

    mask = record_mask(n, split)
    rec = [UNIVERSE[i] for i in range(len(UNIVERSE)) if mask[i]]
    off = [UNIVERSE[i] for i in range(len(UNIVERSE)) if not mask[i]]
    xs = torch.tensor([[a, PLUS, b, EQ] for a, b in rec])
    ys = torch.tensor([SIGMA[p] for p in rec])

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg["d"]
            self.tok = nn.Embedding(VOCAB, d)
            self.pos = nn.Embedding(BLOCK, d)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=cfg["heads"], dim_feedforward=2 * d,
                    batch_first=True, dropout=0.0, activation="gelu")
                for _ in range(cfg["layers"])
            ])
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, VOCAB)

        def forward(self, x):
            B, T = x.shape
            h = self.tok(x) + self.pos(torch.arange(T))
            for blk in self.blocks:
                h = blk(h)
            return self.head(self.ln(h))[:, -1, :]

    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bs = min(64, n)
    for _ in range(cfg["steps"]):
        idx = torch.randint(0, n, (bs,))
        opt.zero_grad()
        loss = F.cross_entropy(model(xs[idx]), ys[idx])
        loss.backward()
        opt.step()

    with torch.no_grad():
        rec_acc = float((model(xs).argmax(-1) == ys).float().mean())
        xo = torch.tensor([[a, PLUS, b, EQ] for a, b in off])
        pred_off = model(xo).argmax(-1).numpy()
        true_off = np.array([SIGMA[p] for p in off])
        off_acc = float((pred_off == true_off).mean())

    return dict(n=n, split=split, size=size_label,
                params=sum(p.numel() for p in model.parameters()),
                seed=seed, rec_acc=rec_acc, off_acc=off_acc,
                pred_off=pred_off.tolist(), off=[list(p) for p in off])


def pairwise_stats(preds):
    preds = np.asarray(preds)
    m = preds.shape[0]
    pis = []
    for i, j in itertools.combinations(range(m), 2):
        pis.append(float((preds[i] != preds[j]).mean()))
    support = float((preds != preds[0]).any(axis=0).mean())
    distinct = float(np.mean([len(set(preds[:, k])) for k in range(preds.shape[1])]))
    return np.mean(pis), float(np.median(pis)), np.max(pis), support, distinct


def main():
    jobs = [(n, s, z, seed) for n in RECORD_SIZES for s in SPLITS
            for z in SIZES for seed in SEEDS]
    print(f"training {len(jobs)} models ...")
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        results = list(ex.map(train_one, jobs))
    print("training done")

    rows = []
    for n in RECORD_SIZES:
        for s in SPLITS:
            for z in SIZES:
                group = [r for r in results
                         if r["n"] == n and r["split"] == s and r["size"] == z]
                consistent = [r for r in group if r["rec_acc"] >= 0.99]
                if len(consistent) < 2:
                    continue
                group = consistent
                mean_pi, med_pi, max_pi, support, distinct = pairwise_stats(
                    [r["pred_off"] for r in group])
                rows.append(dict(
                    n=n, split=s, size=z, params=group[0]["params"],
                    models=len(group),
                    rec_acc_min=min(r["rec_acc"] for r in group),
                    off_acc_mean=float(np.mean([r["off_acc"] for r in group])),
                    pi_mean=mean_pi, pi_median=med_pi, pi_max=max_pi,
                    off_support=support, mean_distinct=distinct))

    np.savez_compressed(
        DATA / "twin_richness_models.npz",
        **{f"pred_{i}": np.asarray(r["pred_off"]) for i, r in enumerate(results)},
        meta=np.asarray([[r["n"], r["split"], list(SIZES).index(r["size"]),
                          SEEDS.index(r["seed"]), r["params"], r["rec_acc"]]
                         for r in results], dtype=float))

    with open(DATA / "twin_richness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    summary = []
    for n in RECORD_SIZES:
        for z in SIZES:
            g = [r for r in rows if r["n"] == n and r["size"] == z]
            summary.append(dict(
                n=n, size=z,
                params=next(r["params"] for r in g),
                pi_mean=float(np.mean([r["pi_mean"] for r in g])),
                pi_sd=float(np.std([r["pi_mean"] for r in g])),
                pi_max=float(np.max([r["pi_max"] for r in g])),
                support_mean=float(np.mean([r["off_support"] for r in g])),
                rec_acc_min=float(np.min([r["rec_acc_min"] for r in g]))))
    with open(DATA / "twin_richness_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    print(f"wrote {len(rows)} group rows and {len(summary)} summary rows")
    for r in summary:
        print(f"  N={r['n']:3d} {r['size']:6s} params={r['params']:6d} "
              f"pi={r['pi_mean']:.4f}+-{r['pi_sd']:.4f} "
              f"support={r['support_mean']:.3f} rec_acc_min={r['rec_acc_min']:.3f}")

    try:
        plot(summary)
    except Exception as e:
        print("plot skipped:", e)


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for z, mk in [("small", "o-"), ("medium", "s-"), ("large", "^-")]:
        g = [r for r in summary if r["size"] == z]
        x = [r["n"] for r in g]
        y = [r["pi_mean"] for r in g]
        e = [r["pi_sd"] for r in g]
        ax.errorbar(x, y, yerr=e, fmt=mk, ms=7, capsize=3,
                    label=f"{z} model ({g[0]['params']:,} params)")
    ax.set_xlabel("record size $N$")
    ax.set_ylabel(r"off-record separation $\hat{\pi}$")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.05)
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=1, fontsize=8, frameon=False)
    fig.savefig(CHARTS / "fig7_twin_richness.pdf", bbox_inches="tight")
    fig.savefig(CHARTS / "fig7_twin_richness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig7_twin_richness.pdf/.png")


if __name__ == "__main__":
    main()
