# Per-conf-bucket calibration

**Window:** 2026-04-20 → 2026-04-26 (1,666 filled outcomes of 1,666 total tracked)

## HEADLINE: Per-conf-bucket realized rate (vs-SP outcomes)

**vs-SP** = scoped to PAs where the batter actually faced the predicted starter. This is the apples-to-apples eval (C3 cleanup). Whole-game outcomes are kept as a secondary section for back-compat.
The High-conf bucket is the betting deliverable; monotonic Realized across (Very Low → High) is the goal.

### HR

| Bucket | n | Hits | Realized | 95% CI | Mean Score | Mean Raw P | Lift | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Very Low | 860 | 81 | 0.094 | [0.076, 0.116] | 0.0221 | 0.0345 | 0.95x | 507 excluded (no SP PA) |
| Low | 136 | 17 | 0.125 | [0.080, 0.191] | 0.0402 | 0.0498 | 1.26x | 34 excluded (no SP PA) |
| Medium | 97 | 11 | 0.113 | [0.065, 0.192] | 0.0464 | 0.0497 | 1.14x | 19 excluded (no SP PA) |
| High | 12 | 1 | 0.083 | [0.015, 0.354] | 0.0751 | 0.0682 | 0.84x | 1 excluded (no SP PA) |

_Monotonicity (Very Low ≤ Low ≤ Medium ≤ High): **FAIL** — Low(0.125) > Medium(0.113); Medium(0.113) > High(0.083)_
_Baseline (all picks): 0.0995_

### Hit

| Bucket | n | Hits | Realized | 95% CI | Mean Score | Mean Raw P | Lift | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Very Low | 773 | 374 | 0.484 | [0.449, 0.519] | 0.1506 | 0.2307 | 1.00x | 475 excluded (no SP PA) |
| Low | 242 | 123 | 0.508 | [0.446, 0.571] | 0.1985 | 0.2470 | 1.05x | 69 excluded (no SP PA) |
| Medium | 90 | 36 | 0.400 | [0.305, 0.503] | 0.2401 | 0.2593 | 0.83x | 17 excluded (no SP PA) |
| High | 0 | 0 | — | — | — | — | — | EMPTY |

_Monotonicity (Very Low ≤ Low ≤ Medium ≤ High): **FAIL** — Low(0.508) > Medium(0.400)_
_Baseline (all picks): 0.4824_

### XBH

| Bucket | n | Hits | Realized | 95% CI | Mean Score | Mean Raw P | Lift | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Very Low | 829 | 182 | 0.220 | [0.193, 0.249] | 0.0566 | 0.0866 | 1.04x | 498 excluded (no SP PA) |
| Low | 200 | 39 | 0.195 | [0.146, 0.255] | 0.0753 | 0.0939 | 0.92x | 51 excluded (no SP PA) |
| Medium | 75 | 12 | 0.160 | [0.094, 0.259] | 0.1005 | 0.1085 | 0.76x | 12 excluded (no SP PA) |
| High | 1 | 0 | 0.000 | [0.000, 0.793] | 0.1179 | 0.1096 | 0.00x |  |

_Monotonicity (Very Low ≤ Low ≤ Medium ≤ High): **FAIL** — Very Low(0.220) > Low(0.195); Low(0.195) > Medium(0.160); Medium(0.160) > High(0.000)_
_Baseline (all picks): 0.2109_

> **WARNING**: at least one target has non-monotonic bucket realized rates. This indicates a structural issue in the conf labeling system (see [docs/conf_label_audit.md](../../docs/conf_label_audit.md)).

## Secondary: Per-conf-bucket realized rate (whole-game outcomes)

Includes bullpen + late-game PAs we never predicted. Useful as a cross-check; vs-SP above is the primary lens.

### HR

| Bucket | n | Hits | Realized | 95% CI | Mean Score | Mean Raw P | Lift | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Very Low | 1367 | 125 | 0.091 | [0.077, 0.108] | 0.0214 | 0.0339 | 0.95x |  |
| Low | 170 | 21 | 0.124 | [0.082, 0.181] | 0.0405 | 0.0501 | 1.28x |  |
| Medium | 116 | 13 | 0.112 | [0.067, 0.182] | 0.0462 | 0.0496 | 1.16x |  |
| High | 13 | 2 | 0.154 | [0.043, 0.422] | 0.0750 | 0.0680 | 1.59x |  |

_Monotonicity (Very Low ≤ Low ≤ Medium ≤ High): **FAIL** — Low(0.124) > Medium(0.112)_
_Baseline (all picks): 0.0966_

### Hit

| Bucket | n | Hits | Realized | 95% CI | Mean Score | Mean Raw P | Lift | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Very Low | 1248 | 553 | 0.443 | [0.416, 0.471] | 0.1485 | 0.2296 | 0.96x |  |
| Low | 311 | 162 | 0.521 | [0.465, 0.576] | 0.1987 | 0.2477 | 1.12x |  |
| Medium | 107 | 57 | 0.533 | [0.439, 0.624] | 0.2404 | 0.2595 | 1.15x |  |
| High | 0 | 0 | — | — | — | — | — | EMPTY |

_Monotonicity (Very Low ≤ Low ≤ Medium ≤ High): PASS — OK_
_Baseline (all picks): 0.4634_

### XBH

| Bucket | n | Hits | Realized | 95% CI | Mean Score | Mean Raw P | Lift | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Very Low | 1327 | 265 | 0.200 | [0.179, 0.222] | 0.0553 | 0.0856 | 0.98x |  |
| Low | 251 | 56 | 0.223 | [0.176, 0.279] | 0.0760 | 0.0948 | 1.10x |  |
| Medium | 87 | 18 | 0.207 | [0.135, 0.304] | 0.0990 | 0.1073 | 1.02x |  |
| High | 1 | 0 | 0.000 | [0.000, 0.793] | 0.1179 | 0.1096 | 0.00x |  |

_Monotonicity (Very Low ≤ Low ≤ Medium ≤ High): **FAIL** — Low(0.223) > Medium(0.207); Medium(0.207) > High(0.000)_
_Baseline (all picks): 0.2035_

## Drift check: last 7d vs 8-30d (per High bucket)

_Not enough rolling data for drift check yet._

## Conf factor means by HR bucket

| Bucket | n | pitcher_data | bvp_hr | bvp_hit | staleness | convergence_hr | convergence_hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Very Low | 1367 | 0.962 | 0.886 | 0.925 | 0.850 | 0.871 | 0.890 |
| Low | 170 | 0.995 | 0.962 | 0.968 | 0.850 | 0.998 | 0.926 |
| Medium | 116 | 0.999 | 1.055 | 1.013 | 0.850 | 1.038 | 0.947 |
| High | 13 | 1.000 | 1.181 | 1.015 | 0.850 | 1.100 | 0.981 |

## Top-decile lift by score (secondary)

| Target | Top 10% n | Top 10% realized | Bottom 10% realized | Overall | Top-decile lift |
|---|---:|---:|---:|---:|---:|
| HR | 105 | 0.143 | 0.054 | 0.100 | 1.44x |
| Hit | 111 | 0.459 | 0.387 | 0.482 | 0.95x |
| XBH | 111 | 0.180 | 0.170 | 0.211 | 0.85x |

## Overall metrics (secondary)

| Target | n | Realized | Mean P (raw) | Mean P (score) | Brier (raw) | Brier (score) | AUC (score) | Log Loss (raw) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HR | 1105 | 0.100 | 0.038 | 0.027 | 0.0928 | 0.0945 | 0.572 | 0.3555 |
| Hit | 1105 | 0.482 | 0.237 | 0.168 | 0.3096 | 0.3482 | 0.533 | 0.8346 |
| XBH | 1105 | 0.211 | 0.089 | 0.063 | 0.1810 | 0.1884 | 0.519 | 0.5835 |
