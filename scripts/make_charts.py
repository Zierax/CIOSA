#!/usr/bin/env python3
"""
Generate publication-quality standalone charts (PDF + 300dpi PNG) for the
six figures of the paper from the CSV data files.

Run: python3 make_charts.py   (anywhere; data/ and charts/ are resolved
relative to this script)
Outputs: fig1_game_value, fig2_budget, fig3_rate, fig4_logodds,
         fig5_real_certification, fig6_real_identification (pdf+png)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA, CHARTS = BASE / "data", BASE / "charts"
CHARTS.mkdir(exist_ok=True)
import numpy as np


def load(path):
    with open(DATA / path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS / f"{name}.pdf")
    fig.savefig(CHARTS / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def style(ax):
    ax.grid(True, which="major", alpha=0.35)
    ax.set_xlim(left=0)
    return ax


# ---- Figure 1: the decomposition (value flat, identification decays) ----
gv = load("game_value.csv")
ide = load("id_error.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xscale("log")
ax.set_xlabel("evidence size $N$")
ax.set_ylabel("error")
ax.plot(gv["n"], gv["measured_error"], "b-*", label="certification value $v(\\mathcal{E})$ (measured, exact $=1/2$)")
ax.plot(ide["n"], ide["measured_id_error"], "r-s", label="identification error under Occam prior (decays)")
ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
ax.set_ylim(0, 0.6)
ax.grid(True, which="major", alpha=0.35)
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig1_game_value")

# ---- Figure 2: the budget function (static linear, relocating flat) ----
b = load("budget_value.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xlabel("active query budget $T$")
ax.set_ylabel("value $v(T)$")
ax.plot(b["t"], b["static_value"], "b-*", label="static twin: $v(T)=\\frac{1}{2}(1-T/M)$ (measured)")
ax.plot(b["t"], b["relocating_value"], "r-s", label="relocating twin: $v(T)=\\frac{1}{2}$ for $T<M$ (measured)")
ax.set_ylim(0, 0.6)
ax.grid(True, which="major", alpha=0.35)
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig2_budget")

# ---- Figure 3: the contrast of rates (separable decay vs 1/2 floor) ----
sep = load("separable_rate.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xscale("log")
ax.set_xlabel("queries $N$ (log scale)")
ax.set_ylabel("error")
ax.plot(sep["n"], sep["measured_id_error"], "b-*", label="separable class: measured decay")
ax.plot(sep["n"], sep["theory"], "b--", label="exact expectation E[1-1/(1+B)], B~Bin(K-1, ALPH^-N)")
ax.axhline(0.5, color="0.3", ls="--", lw=1.0, label="value floor $1/2$ (twin-rich)")
ax.set_ylim(0, 1)
ax.grid(True, which="major", alpha=0.35)
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig3_rate")

# ---- Figure 4: the log-odds floor (flat at prior ratio vs O(log N)) ----
lo = load("logodds.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xscale("log")
ax.set_xlabel("evidence size $N$ (log scale)")
ax.set_ylabel("log-odds")
ax.plot(lo["n"], lo["measured_logodds"], "r-s", label="log-odds of $\\sigma$ vs frontier twin (measured)")
ax.plot(lo["n"], lo["logN_bound"], "b--", label="upper bound $O(\\log N)$")
ax.set_ylim(0, 12)
ax.grid(True, which="major", alpha=0.35)
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig4_logodds")

# ---- Figure 5: real-system certification value (trained transformer) ----
rc = load("real_certification.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xscale("log")
ax.set_xlabel("record size $N$")
ax.set_ylabel("measured certification value")
ax.plot(rc["n"], rc["measured_value"], "b-*", ms=8, alpha=0.9,
        label="real twin (two trained transformers, chain-MLE record)")
ax.axhline(0.5, color="0.5", ls="--", lw=0.8, label="theory $1/2$")
ax.set_ylim(0, 0.6)
ax.grid(True, which="major", alpha=0.35)
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig5_real_certification")

# ---- Figure 6: real-system identification error (free probes) ----
ri = load("real_identification.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xscale("log")
ax.set_xlabel("free probes $T$ (log scale)")
ax.set_ylabel("identification error")
ax.errorbar(ri["t"], ri["measured_error"], yerr=ri.get("ci95", 0.0),
            fmt="r-s", ms=5, lw=1.2, capsize=3,
            label="real trained model (measured)")
ax.set_ylim(0, 0.5)
ax.grid(True, which="major", alpha=0.35)
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig6_real_identification")

print("Wrote fig1..fig6 (pdf+png)")
