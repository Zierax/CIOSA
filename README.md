# CIOSA: Computational Impossibility of Semantic Alignment

**Scaling cannot certify semantic understanding in artificial systems: an
exact bound on the certification of meaning from finite behavior.**

This repository contains the full reproducible research package for the
manuscript: measurement scripts, data, charts, and the manuscript sources.

## The result in one paragraph

The paper formalizes *certification of semantic understanding* as a game:
an adversary presents one of two computationally indistinguishable
candidate functions, one realizing an intended semantics and one merely
matching the observable record, and a behavioral certifier that sees only
finite input-output evidence must decide. The value of this game is
**exactly 1/2 for every finite evidence size**: no behavioral certifier
can score below 1/2 in the worst case, and scaling the evidence does not
move it. Active probing cannot revive certification (the budget function
is a linear census, never a search), and the result decomposes into an
identification component, which scaling can reduce to zero (exponentially
on separable classes), and an alignment component, bounded below by Rice's
theorem, which scale cannot touch.

The impossibility is epistemic and paradigm-scoped, not ontic: the paper
neither asserts nor denies that a system may understand without being
certifiable.

## Repository layout

```
repo/
├── scripts/
│   ├── twin_models.py           # synthetic measurements (deterministic)
│   ├── real_twin_experiment.py  # real-transformer verification (5 seeds, pooled)
│   └── make_charts.py           # figure generators (PDF + PNG; six figures)
├── data/
│   ├── game_value.csv           # certification value vs evidence size N
│   ├── id_error.csv             # identification error under Occam prior
│   ├── budget_value.csv         # budget function: static vs relocating twin
│   ├── logodds.csv              # log-odds floor vs O(log N) bound
│   ├── separable_rate.csv       # exponential identification on separable class
│   ├── real_twin_agreement.csv  # trained twin: agreement on/off the record
│   ├── real_certification.csv   # certification value on trained transformers
│   └── real_identification.csv  # identification with T free probes (real models)
├── charts/
│   ├── fig1_game_value.{pdf,png}
│   ├── fig2_budget.{pdf,png}
│   ├── fig3_rate.{pdf,png}
│   ├── fig4_logodds.{pdf,png}
│   ├── fig5_real_certification.{pdf,png}
│   └── fig6_real_identification.{pdf,png}
└── results/
    └── results_summary.md    # measured numbers and how to read them
```

## Reproduce everything

Requires Python 3 with numpy, matplotlib, and torch (torch only for the
real-transformer verification).

```bash
python3 scripts/twin_models.py            # writes data/*.csv (five synthetic files)
python3 scripts/real_twin_experiment.py   # trains 2 small transformers; writes real_*.csv
python3 scripts/make_charts.py            # writes charts/fig1..fig6 (.pdf and .png)
```

Everything is deterministic (single fixed procedure; five seeds for the real transformer, pooled); the CSV values in `data/`
are exactly what the scripts produce, and the figures in `charts/` are
exactly what `make_charts.py` produces from them.

## The measurements

| Data file | What it measures | The theorem it confirms |
|---|---|---|
| `game_value.csv` | Behavioral certifier error vs evidence size N | Value = 1/2, saturated in N |
| `id_error.csv` | Bayes identification error under Occam prior | Identification decays with coverage |
| `budget_value.csv` | Static vs relocating twin, query budget T | v(T) = 1/2(1 − T/M) linear; relocating flat |
| `logodds.csv` | Log-odds of aligned vs frontier twin | O(log N) bound; no exponential verdict |
| `separable_rate.csv` | Identification on separable class | (B−1)(1−π)^N exponential decay |
| `real_twin_agreement.csv` | Trained twin: agreement on/off record (E vs off-E) | A real Goodman twin is trainable: agree=1 on E, 0 off |
| `real_certification.csv` | Certification value on two trained transformers | Value flat at exactly 1/2 for N = 4..30 |
| `real_identification.csv` | Identification with T free probes (real models) | Error decays with T, floor once probes leave E |

The real-transformer experiment (`real_twin_experiment.py`) trains a small
transformer (~0.5M params) on digit arithmetic $a+b \bmod 10$ (aligned),
and a twin trained on the record $a<3$ under $\sigma$ and the shifted rule
$a+b+3 \bmod 10$ off it; it then measures the game value and the
identification curve on the trained models (Figures 5--6).

## Manuscript

Manuscript sources are at the project root, built with `elsarticle.cls` + `pgfplots` from `repo/data/*.csv`:

- `manuscript.pdf` / `manuscript.tex` — main manuscript
- `appendix.pdf` / `appendix.tex` — supplementary appendix with expanded proofs and reproducibility details

See `manuscript.tex` and `appendix.tex` at the workspace root.

## Citation

If you use this repository, please cite:

```bibtex
@misc{salah2026scaling,
  title        = {Scaling cannot certify semantic understanding in artificial systems: an exact bound on the certification of meaning from finite behavior},
  author       = {Salah, Ziad},
  year         = {2026},
  howpublished = {\url{https://github.com/Zierax/CIOSA}},
  note         = {ORCID: \url{https://orcid.org/0009-0002-6813-2416}}
}
```

## License

MIT, see [LICENSE](LICENSE).
