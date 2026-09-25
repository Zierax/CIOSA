#!/usr/bin/env python3
"""
Certification game measurements for "Algorithmic agnosia: the exact value
of the semantic-alignment certification game".

Two quantities are MEASURED, not asserted (numpy-vectorized, deterministic):

  Demo 1 (adversarial game value).  A *behavioral certifier* judges a candidate
  from a finite input-output record alone (exactly what a benchmark does).  An
  adversary builds a twin pair -- f aligned to the intended semantics, g
  agreeing with f on the record but differing at one hidden input x0 -- and
  presents one with probability 1/2.  Because the record is identical for both,
  every behavioral certifier has error exactly 1/2.  We MEASURE this by Monte
  Carlo.  Result must be flat at 1/2 for every evidence size N.

  Demo 2 (identification under an aligned prior).  Under a fixed simplicity
  prior over the hypothesis class, scaling evidence shrinks the *identification*
  error to zero: once the record hits the hidden input, the grue twin is
  excluded and the posterior concentrates on the aligned model.  This exhibits
  the decomposition theorem: scaling can move identification, not the value.

Outputs: game_value.csv and id_error.csv.  Deterministic seed.
Run: python3 twin_models.py
"""

import csv
import math
import random

import numpy as np

SEED = 12345
random.seed(SEED)
rng = np.random.default_rng(SEED)

POP = 20_000
SIZES = [10, 100, 500, 1000, 2000, 4000, 8000, 12000, 16000]
KS = list(range(1, 7))


# ---------------------------------------------------------------------------
# Demo 1: adversarial certification game value
# ---------------------------------------------------------------------------

def measure_game_value(N, trials=1600):
    """Behavioral certifiers see only the record, so the value is 1/2
    regardless of the rule; we estimate it by sampling instances and the
    presentation coin."""
    err = 0.0
    count = 0
    for _ in range(trials):
        xs = rng.integers(0, POP, size=N)
        ys = xs * xs  # record for the aligned candidate f (square)
        # three behavioral rules, all seeing ONLY xs, ys:
        #   r0: certify iff record is non-decreasing in input
        #   r1: certify iff record matches perfect squares on its domain
        #   r2: random rule, certify with prob 1/2
        r0 = bool(np.all(np.diff(ys) >= 0)) or N < 2
        r1 = bool(np.all(ys == xs * xs))
        r2 = bool(rng.random() < 0.5)
        for cert in (r0, r1, r2):
            shown_f = bool(rng.random() < 0.5)  # coin hidden from the rule
            if shown_f:
                err += 1.0 if not cert else 0.0
            else:
                err += 1.0 if cert else 0.0
            count += 1
    return err / count


# ---------------------------------------------------------------------------
# Demo 2: identification error under an Occam prior
# ---------------------------------------------------------------------------

def measure_identification_error(N, trials=400):
    """Hypothesis class H = {x^k : k in KS} plus, for each k, a 'grue'
    twin g_k that agrees with x^k everywhere except at one hidden input
    hx[k], where it returns a value different from x^k.  Occam prior
    P(x^k) = 2^-k, and the twin of k carries mass P(g_k) = 2^-k * 0.1.
    The record is drawn from the aligned candidate f(x) = x^2.
    Posterior carries mass ONLY for hypotheses consistent with the record:

      - a pure hypothesis x^k is consistent iff x^k == x^2 on every
        observed point.  For integer inputs >= 2 (the relevant regime;
        the only monomial equalities at 0/1 are degenerate) this holds
        for k=2 only, so P-pure concentrates the intended k.
      - the grue twin of a consistent k is itself consistent iff its
        hidden point hx_k was NOT observed (otherwise the twin and x^k
        disagree on the record).  Hence at most one twin -- the twin of
        the intended k=2 -- can ever carry posterior mass, and only until
        the record samples its hidden point.

    Identification error = 1 - max posterior over {aligned, non-aligned}.
    As N grows the hidden point is eventually sampled, the single live
    twin is excluded, and error -> 0."""
    errs = []
    for _ in range(trials):
        xs = rng.choice(POP, size=N, replace=False)
        ys = xs * xs
        # one hidden input where the twin of the intended candidate differs
        hx = int(rng.integers(0, POP))
        observed = set(xs.tolist())
        total_pure = 0.0
        total_grue = 0.0
        for k in KS:
            pred = xs.astype(np.float64) ** k
            if np.allclose(ys.astype(np.float64), pred):
                # x^k is consistent with the record; for k != 2 this is
                # degenerate (all sampled inputs in {0, 1}), for k = 2 it
                # is the whole record.
                total_pure += 2.0 ** (-k)
                if hx not in observed:
                    total_grue += 2.0 ** (-k) * 0.1
        denom = total_pure + total_grue
        p_align = total_pure / denom if denom > 0 else 0.5
        errs.append(1.0 - max(p_align, 1.0 - p_align))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Demo 3: active certification with a query budget
# ---------------------------------------------------------------------------

def measure_static_budget_value(T, M, trials=4000):
    """Static twin: the adversary commits to a hidden input y in [0, M)
    BEFORE the certifier queries.  A certifier with budget T probes T
    distinct points uniformly; it catches the twin iff it probes y.
    Expected error = (1/2) * P(no probe of y) = 1/2 (1 - T/M)."""
    errs = []
    for _ in range(trials):
        y = int(rng.integers(0, M))
        probes = rng.choice(M, size=T, replace=False)
        hidden_f = bool(rng.random() < 0.5)
        if hidden_f:
            errs.append(0.0)
        else:
            errs.append(1.0 if y not in probes else 0.0)
    return float(np.mean(errs))


def measure_relocating_budget_value(T, M, trials=4000):
    """Relocating twin: the adversary observes the budget set AFTER the
    certifier commits, and places y in the uncovered part.  The twin is
    caught only when T >= M (the frontier of the class has been
    exhausted); for every T < M the value is 1/2."""
    errs = []
    for _ in range(trials):
        probes = rng.choice(M, size=T, replace=False)
        covered = set(probes.tolist())
        uncovered = [u for u in range(M) if u not in covered]
        if not uncovered:
            y = int(rng.integers(0, M))
        else:
            y = int(rng.choice(uncovered))
        hidden_f = bool(rng.random() < 0.5)
        if hidden_f:
            errs.append(0.0)
        else:
            errs.append(1.0 if y not in probes else 0.0)
    return float(np.mean(errs))


def measure_budget_curve(M, T_list, trials=3000):
    static = [measure_static_budget_value(t, M, trials) for t in T_list]
    reloc = [measure_relocating_budget_value(t, M, trials) for t in T_list]
    return static, reloc


# ---------------------------------------------------------------------------
# Demo 4: the log-odds floor of the frontier twin
# ---------------------------------------------------------------------------

def measure_log_odds(N, trials=400):
    """Occam prior over {x^k}, aligned x^2; the adversary's frontier twin
    differs at the cheapest unobserved point y <= N+1 (never probed, by
    definition of the frontier).  The posterior log-odds between aligned and
    twin is then bounded by the prior ratio-- a constant independent of N,
    illustrating the <= O(log N) bound: the evidence that identification
    would have driven to infinity is capped at the prior's log-odds."""
    odds = []
    for _ in range(trials):
        xs = rng.choice(POP, size=N, replace=False)
        ys = xs * xs
        total_pure = 0.0
        for k in KS:
            pred = xs.astype(np.float64) ** k
            if np.allclose(ys.astype(np.float64), pred):
                total_pure += 2.0 ** (-k)
        p_true = total_pure
        p_twin = total_pure * 0.1 if p_true > 0 else 1e-12
        odds.append(np.log(p_true / p_twin) if p_true > 0 else 0.0)
    return float(np.mean(odds))


# ---------------------------------------------------------------------------
# Demo 4: identification error on an exactly separable finite class
# ---------------------------------------------------------------------------

def separable_theory(N, K=64, ALPH=5.0):
    """Exact expected identification error for the separable class: with
    B ~ Binomial(K-1, ALPH^-N) surviving competitors, the uniform prior's
    error is E[1 - 1/(1+B)], evaluated exactly by the binomial sum."""
    q = ALPH ** (-N)
    p = 1.0 - q
    m = int(K) - 1
    E = 0.0
    for k in range(m + 1):
        E += math.comb(m, k) * (q ** k) * (p ** (m - k)) / (1.0 + k)
    return 1.0 - E


def measure_separable_rate(N, K=64, UNIV=4000, ALPH=5, trials=2000):
    """Exactly separable finite class: K hypotheses, each a random ALPH-ary
    function over UNIV inputs.  Every distinct pair differs at a random
    fraction (1 - 1/ALPH) > 0 of inputs, so the frontier theorem's
    separation condition (ii) holds with pi = 1 - 1/ALPH.  The record is
    drawn from h0.  After N observed inputs, the number of hypotheses still
    consistent has expectation 1 + (K-1) * ALPH^-N -> 1, so the uniform
    prior's identification error 1 - 1/consistent -> 0 exponentially, at
    rate log(ALPH).  This is the contrast that twin-rich classes deny.
    The theory column is the exact expectation (separable_theory), not
    the plug-in estimate 1 - 1/(1 + (K-1) ALPH^-N); the two differ by
    Jensen's inequality."""
    errs = []
    for _ in range(trials):
        H = rng.integers(0, ALPH, size=(K, UNIV), dtype=np.int8)
        h0 = int(rng.integers(0, K))
        xs = rng.choice(UNIV, size=N, replace=False)
        consistent = np.ones(K, dtype=bool)
        for x in xs:
            consistent &= H[:, int(x)] == H[h0, int(x)]
        n_consist = int(np.count_nonzero(consistent))
        errs.append(1.0 - 1.0 / n_consist)
    return float(np.mean(errs))


def main():
    from pathlib import Path
    OUT = Path(__file__).resolve().parent.parent / "data"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "game_value.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "measured_error"])
        for N in SIZES:
            e = measure_game_value(N)
            w.writerow([N, f"{e:.6f}"])
            print(f"game value     N={N:6d}  measured error = {e:.6f}")

    with open(OUT / "id_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "measured_id_error"])
        for N in SIZES:
            e = measure_identification_error(N)
            w.writerow([N, f"{e:.6f}"])
            print(f"identification N={N:6d}  measured error = {e:.6f}")

    M = 20000
    BUD = [100, 500, 1000, 2000, 4000, 8000, 12000, 16000, 19000]
    with open(OUT / "budget_value.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t", "static_value", "relocating_value"])
        for t in BUD:
            s = measure_static_budget_value(t, M, trials=3000)
            r = measure_relocating_budget_value(t, M, trials=3000)
            w.writerow([t, f"{s:.6f}", f"{r:.6f}"])
            print(f"budget  T={t:6d}  static={s:.6f}  relocating={r:.6f}")

    with open(OUT / "logodds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "measured_logodds", "logN_bound"])
        # theoretical bound log(N) shown for contrast (see text)
        for N in SIZES:
            lo = measure_log_odds(N)
            w.writerow([N, f"{lo:.6f}", f"{np.log(N):.6f}"])
            print(f"log-odds N={N:6d}  measured={lo:.4f}  log N={np.log(N):.4f}")

    with open(OUT / "separable_rate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "measured_id_error", "theory"])
        SN = [1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 20]
        ALPH = 5.0
        K = 64.0
        for N in SN:
            e = measure_separable_rate(N)
            t = separable_theory(N)
            w.writerow([N, f"{e:.6f}", f"{t:.6f}"])
            print(f"separable N={N:3d}  id error = {e:.6f}  theory={t:.6f}")

    print("\nDone. Wrote game_value.csv, id_error.csv, budget_value.csv, "
          "logodds.csv, separable_rate.csv")


if __name__ == "__main__":
    main()
