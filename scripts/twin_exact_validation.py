"""Numerical check of Theorem 8: v_N(pi) = (1/2)(1-pi)^N.

The certification game on a twin pair (f, g) with d_D(f,g) = pi:
- f is aligned, g differs from f on a set S of density pi;
- the presentation is uniform on {f, g};
- the certifier sees N i.i.d. uniform queries and their labels.

The Bayes-optimal certifier identifies the presentation exactly when some
query lands in S, and guesses otherwise, so its error is
(1/2) * P(no query in S) = (1/2)(1-pi)^N.

Run:  python3 twin_exact_validation.py
Writes ../data/twin_exact_validation.csv and prints the table.
"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "data", "twin_exact_validation.csv")
TRIALS = 80000


def simulated_error(M, pi, N, trials=TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    k = int(round(pi * M))
    pi = k / M
    S = np.zeros(M, dtype=bool)
    S[:k] = True
    err = 0
    for _ in range(trials):
        presented_is_g = bool(rng.integers(2))
        xs = rng.integers(0, M, N)
        hit = S[xs].any()
        if hit:
            y = 1 if presented_is_g else 0
            pred_g = (y == 1)
        else:
            pred_g = False
        err += int(pred_g != presented_is_g)
    return err / trials, pi


def main():
    rows = []
    for M, pi, N in [(100, 0.30, 1), (100, 0.30, 5), (100, 0.30, 10),
                     (1000, 0.01, 300), (10000, 0.001, 2000)]:
        sim, pi_exact = simulated_error(M, pi, N)
        rows.append((M, pi_exact, N, sim, 0.5 * (1 - pi_exact) ** N))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["universe_size", "pi", "queries", "simulated_error", "formula_half_pow"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6f}", r[2], f"{r[3]:.6f}", f"{r[4]:.6f}"])

    print(f"{'|D|':>6} {'pi':>6} {'N':>6} {'sim':>9} {'(1-pi)^N/2':>12}")
    for M, pi, N, sim, form in rows:
        print(f"{M:6d} {pi:6.3f} {N:6d} {sim:9.6f} {form:12.6f}")

    # sample complexity at pi=0.1, delta=0.1
    delta = 0.1
    for N in range(1, 200):
        e, _ = simulated_error(20000, 0.1, N, trials=40000)
        if e <= delta:
            formula = np.log(1 / (2 * delta)) / np.log(1 / (1 - 0.1))
            print(f"\nN*(pi=0.1, delta=0.1): simulated={N}, "
                  f"closed form={formula:.2f}")
            break


if __name__ == "__main__":
    main()
