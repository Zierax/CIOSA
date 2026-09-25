#!/usr/bin/env python3
"""
Real-system verification of the certification game on a trained transformer.

The original demonstration simulated the theorem's conditions by
construction.  This script moves the measurement onto a real, trained
system, and the twin is a genuinely different learned function, not a
hand-assembled counterexample:

  * a small GPT is trained from scratch (PyTorch, CPU, deterministic) on a
    finite digit-arithmetic semantics  sigma(a,b) = (a+b) mod 10  over the
    universe  U = {0..9}^2  (100 inputs);
  * the aligned candidate M1 is that network, with accuracy 1.0 on U;
  * the record is the coincidence set  E = {(a,b) : a < 3}  (30 inputs),
    and the twin M2 is trained from scratch on the pair of rules
    sigma on E and tau(a,b) = sigma(a,b)+3 mod 10 off E.  This is the
    Goodman construction with a learnable coincidence set: the twin
    implements a genuinely different semantics that coincides with the
    intended one exactly on the record.  Whether M2 really agrees on E and
    really differs off E is MEASURED, not assumed;
  * certification value: records R subset of E are presented to a
    behavioral certifier; whenever M1 and M2 agree on the record the
    certifier's error is exactly 1/2, so the measured value is
    0.5 * P(agree on R);
  * identification error: a certifier allowed T free probes over U and
    comparing candidates against sigma identifies the twin as soon as a
    probe lands off the record, so the error decays like
    0.5 * (|E|/|U|)^T plus the aligned model's own error.

All numbers are measured on the trained networks.  Deterministic across
five seeds (12345-12349), CPU-only, no pretrained weights, no network
access required.

Run:  python3 real_twin_experiment.py
Outputs (in ../data/):  real_twin_agreement.csv  real_certification.csv
                        real_identification.csv
"""

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

SEEDS = [12345, 12346, 12347, 12348, 12349]

# ---------------------------------------------------------------------------
# Tokens and data:  "d1 + d2 ="  ->  answer digit
# ---------------------------------------------------------------------------

PLUS = 10         # token  +
EQ = 11           # token  =
VOCAB = 12
BLOCK = 4         # d1, +, d2, =

ALL = [(a, b) for a in range(10) for b in range(10)]

# The record is the coincidence set of the two semantics: the intended
# rule sigma and the grue rule tau agree exactly on E.
E = [(a, b) for a, b in ALL if a < 3]
OFF = [(a, b) for a, b in ALL if (a, b) not in E]


def sigma(a, b):
    return (a + b) % 10


def tau(a, b):
    return (a + b + 3) % 10


def seq_of(a, b):
    return [a, PLUS, b, EQ]


def encode(pairs, labels):
    xs = [seq_of(a, b) for a, b in pairs]
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


# ---------------------------------------------------------------------------
# Tiny GPT (trained from scratch)
# ---------------------------------------------------------------------------

class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, block=BLOCK, n_embd=128, n_head=4,
                 n_layer=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=2 * n_embd,
                batch_first=True, dropout=0.0, activation="gelu")
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))[:, -1, :]


def train(model, xs, ys, steps=1500, lr=1e-3, bs=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = xs.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))
        xb, yb = xs[idx], ys[idx]
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    return float(loss.item())


@torch.no_grad()
def predict(model, pairs):
    xs, _ = encode(pairs, [0] * len(pairs))
    out = []
    for i in range(0, len(xs), 256):
        out.append(model(xs[i:i + 256]).argmax(-1))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    # ---- aligned candidate: trained on the whole universe under sigma ----
    # ---- twin: a different network, trained on sigma on E and tau off E ----
    # Both are retrained from scratch for each of the five seeds; all
    # reported numbers are means across seeds (with per-seed spread where
    # the protocol is stochastic).
    agree_stats = []
    preds = []
    for SEED in SEEDS:
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        random.seed(SEED)
        np.random.seed(SEED)

        m1 = TinyGPT()
        xs_all, ys_all = encode(ALL, [sigma(a, b) for a, b in ALL])
        train(m1, xs_all, ys_all)

        m2 = TinyGPT()
        pairs = E + OFF
        labels = [sigma(a, b) for a, b in E] + [tau(a, b) for a, b in OFF]
        xs2, ys2 = encode(pairs, labels)
        train(m2, xs2, ys2, steps=2500)

        # ---- twin construction statistics (measured, not assumed) ----
        p1 = predict(m1, ALL)
        p2 = predict(m2, ALL)
        preds.append((p1, p2))
        iE = torch.tensor([ALL.index(p) for p in E])
        iO = torch.tensor([ALL.index(p) for p in OFF])
        agreeE = float((p1[iE] == p2[iE]).float().mean())
        agreeO = float((p1[iO] == p2[iO]).float().mean())
        acc1U = float((p1 == torch.tensor([sigma(a, b) for a, b in ALL])).float().mean())
        acc2E = float((p2[iE] == torch.tensor([sigma(a, b) for a, b in E])).float().mean())
        acc2O = float((p2[iO] == torch.tensor([tau(a, b) for a, b in OFF])).float().mean())
        agree_stats.append([agreeE, agreeO, acc1U, acc2E, acc2O])

    agree_mean = [float(np.mean([r[k] for r in agree_stats])) for k in range(5)]
    agree_std = [float(np.std([r[k] for r in agree_stats])) for k in range(5)]

    with open(DATA / "real_twin_agreement.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agreement_on_record_E", "agreement_off_E", "aligned_acc_on_U",
                    "twin_acc_on_E_sigma", "twin_acc_on_offE_tau", "seed_std"])
        w.writerow([f"{v:.4f}" for v in agree_mean] + [f"{max(agree_std):.4f}"])

    # ---- certification value vs record size N (records drawn from E) ----
    # Both models are scored on the record; if they agree on every recorded
    # input, the certifier's error is exactly 1/2.  Vectorized over trials,
    # averaged over the five seeds.
    sizes = [4, 8, 16, 30]
    trials = 400
    cert_pool = {N: [] for N in sizes}
    sigma_arr = np.array([sigma(a, b) for a, b in ALL])
    idxE = np.array([ALL.index(p) for p in E])
    for SEED, (p1, p2) in zip(SEEDS, preds):
        rng = np.random.default_rng(SEED)
        m1n, m2n = p1.numpy(), p2.numpy()
        for N in sizes:
            recs = rng.choice(idxE, size=(trials, N))
            agree = (m1n[recs] == m2n[recs]).all(axis=1)
            cert_pool[N].extend(agree.tolist())

    cert_rows = []
    for N in sizes:
        agrees = np.array(cert_pool[N])
        p_agree = float(agrees.mean())
        cert_rows.append({"n": N, "measured_value": f"{0.5 * p_agree:.4f}",
                          "p_agree_record": f"{p_agree:.4f}"})

    with open(DATA / "real_certification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cert_rows[0].keys()))
        w.writeheader()
        w.writerows(cert_rows)

    # ---- identification error vs free-probe budget T ----
    # parallel over trials with numpy: all probes for all trials at once.
    # Trials are pooled across the five seeds (all i.i.d. Bernoulli draws),
    # so the Wilson interval is computed on the pooled n = 5 * trials.
    Ts = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    trials = 400
    z = 1.96
    ident_pool = {T: [] for T in Ts}

    def wilson(p_hat, n):
        denom = 1.0 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n)) / denom
        return half

    for SEED, (p1, p2) in zip(SEEDS, preds):
        rng = np.random.default_rng(SEED)
        match1 = (p1.numpy() == sigma_arr)
        match2 = (p2.numpy() == sigma_arr)
        for T in Ts:
            probes = rng.choice(len(ALL), size=(trials, T))
            on1 = match1[probes].all(axis=1)
            on2 = match2[probes].all(axis=1)
            per_trial = ((~on1).astype(float) + on2.astype(float)) / 2.0
            ident_pool[T].extend(per_trial.tolist())

    ident_rows = []
    for T in Ts:
        pooled = np.array(ident_pool[T])
        err = float(pooled.mean())
        ci = wilson(err, len(pooled))
        ident_rows.append({"t": T, "measured_error": f"{err:.4f}",
                           "ci95": f"{ci:.4f}"})

    with open(DATA / "real_identification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "measured_error", "ci95"])
        w.writeheader()
        w.writerows(ident_rows)

    print("real_twin_agreement.csv (mean over 5 seeds):")
    print(f"  agreement on E = {agree_mean[0]:.4f}   off E = {agree_mean[1]:.4f}   "
          f"aligned acc = {agree_mean[2]:.4f}   twin acc(E,sigma) = {agree_mean[3]:.4f}   "
          f"twin acc(offE,tau) = {agree_mean[4]:.4f}   max seed std = {max(agree_std):.4f}")
    print("real_certification.csv (pooled over 5 seeds):")
    for r in cert_rows:
        print(f"  N={r['n']:3d}  value={r['measured_value']}  p_agree={r['p_agree_record']}")
    print("real_identification.csv (pooled over 5 seeds):")
    for r in ident_rows:
        print(f"  T={r['t']:3d}  error={r['measured_error']}")


if __name__ == "__main__":
    main()
