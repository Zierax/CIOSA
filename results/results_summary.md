# Measurement results

All values below are reproduced by `scripts/twin_models.py` (numpy, stock
Python) and `scripts/real_twin_experiment.py` (numpy + torch). The same
data live as CSVs in `data/` and are plotted in `charts/` (six figures).
The real-transformer numbers pool five independent training seeds
(12345-12349); Wilson 95% intervals are on the pooled trial counts.

## 1. Certification game value (`data/game_value.csv`)

Evidence size N vs measured error of a behavioral certifier (three rules,
fair presentation coin). The theorem predicts a value of exactly 1/2 for
every N on twin-rich classes.

| N      | measured error |
|--------|----------------|
| 10     | ~0.50          |
| 16000  | ~0.50          |

Flat at 0.50 +- 0.02 (Monte-Carlo noise) for every N in between. Saturation confirmed.

## 2. Identification error under an Occam prior (`data/id_error.csv`)

| N      | measured id error |
|--------|-------------------|
| 10     | ~0.091            |
| 16000  | ~0.018            |

Decays as the record covers the twin's hidden input (coverage prob ~ N/20000).

## 3. Budget function, static vs relocating twin (`data/budget_value.csv`)

Universe size M = 20,000.

| T      | static value | relocating value |
|--------|--------------|------------------|
| 100    | ~0.50        | ~0.50            |
| 19000  | ~0.027       | ~0.50            |

Static value follows v(T) = 1/2(1 - T/M) linearly; relocating twin holds
1/2 for every finite budget. Theorem 2 (budget) confirmed.

## 4. Log-odds floor (`data/logodds.csv`)

Measured log-odds of the aligned hypothesis vs its frontier twin is pinned
at ln(10) ~ 2.30 for every N (prior ratio 0.1), while the theoretical bound
is O(log N). Exponential certainty is never attained.

## 5. Separable class identification rate (`data/separable_rate.csv`)

Exactly separable class (K=64 hypotheses, ALPH=5, universe 4000),
trials=2000, seed 12345:

| N | measured | theory (exact) |
|---|----------|----------------|
| 1  | 0.921580 | 0.921875 |
| 2  | 0.637321 | 0.638025 |
| 3  | 0.207512 | 0.214967 |
| 4  | 0.045167 | 0.048773 |
| 5  | 0.009500 | 0.010014 |
| 6  | 0.002250 | 0.002013 |
| 8  | 0.000000 | 0.000081 |
| 10+ | 0       | <1e-5    |

The theory column is the exact binomial expectation
E[1 - 1/(1+B)], B ~ Binomial(63, 5^-N); an earlier plug-in formula
1 - 1/(1+(K-1)5^-N) differed from the measured curve by Jensen's
inequality (up to ~3 sigma at small N) and was replaced. The decay is
exponential at rate log(5): exhaustion in a handful of queries -- the
contrast to the twin-rich 1/2 floor.

## 6. Real trained twin on a trained transformer (`data/real_twin_agreement.csv`)

Two ~0.5M-parameter transformers trained on digit arithmetic
$a+b \bmod 10$ over the 100 inputs $(a,b) \in \{0,\dots,9\}^2$:
one aligned on the whole universe, one trained as the Goodman twin
($\sigma$ on $\E = \{a<3\}$, $\tau = \sigma+3$ off it). Agreement is exact
equality of the argmax output token.

| quantity | measured |
|----------|----------|
| agreement on E | 1.0000 |
| agreement off E | 0.0000 |
| aligned acc. on universe | 1.0000 |
| twin acc. on E (under sigma) | 1.0000 |
| twin acc. off E (under tau) | 1.0000 |

The Goodman construction is trainable: the twin agrees with the aligned
model exactly on the record and diverges on every input off it.

## 7. Real certification value (`data/real_certification.csv`)

Certifier sees a chain maximum-likelihood record from E (N = 4, 8, 16, 30);
the value on the two trained transformers:

| N | measured value | P(agree on record) |
|---|----------------|--------------------|
| 4  | 0.5000 | 1.0000 |
| 8  | 0.5000 | 1.0000 |
| 16 | 0.5000 | 1.0000 |
| 30 | 0.5000 | 1.0000 |

Flat at exactly 1/2. The value is 0.5 x P(agree on record) by
construction; with agreement 1.0000 this is an identity, and what the
curve verifies is that the trained networks realize the twin construction
(the theorem's prediction on trained weights). (Figure 5.)

## 8. Real identification error with free probes (`data/real_identification.csv`)

| T | measured error | 95% Wilson CI (pooled) |
|---|----------------|------------------------|
| 1 | 0.1497 | +-0.0156 |
| 2 | 0.0478 | +-0.0094 |
| 3 | 0.0160 | +-0.0056 |
| 4 | 0.0055 | +-0.0034 |
| 6 | 0.0003 | +-0.0012 |
| 8 | 0.0000 (0/2000) | [0, 0.0019] |

Decays as T probes land outside the record E (70% of the universe, where
the twin mismatches): the error is exactly 1/2 * Pr[all T probes land in
E] = 1/2 * 0.3^T. Pooled over five seeds (2000 trials per T); the decay
spans just over an order of magnitude beyond the worst-case Wilson
half-width (+-0.022 at p=1/2). (Figure 6, with error bars.)

## Reproduce

```
python3 scripts/twin_models.py            # writes data/*.csv (synthetic)
python3 scripts/real_twin_experiment.py   # trains 2 transformers, writes real_*.csv
python3 scripts/make_charts.py            # writes charts/*.pdf + charts/*.png
```
