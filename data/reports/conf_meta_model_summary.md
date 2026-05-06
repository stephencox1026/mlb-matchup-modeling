# M1: Confidence Meta-Model Summary (probability stacker)

Stacked HGBR Classifier on the 2025 val parquet with `beast` base predictions + context features (BvP, rolling rates, month, pitcher data). The meta-model outputs a refined probability; Lock = top 10% by meta-prob.

## HR
- Train rows: 146,596  |  Test rows: 36,649
- Baseline realized rate (test): 0.0305
- Thresholds (meta-prob): Lock ≥ 0.0670, Strong ≥ 0.0484, Lean ≥ 0.0260, else Avoid
- Base model Brier: 0.02887 → Meta Brier: 0.02886  (Δ +0.000011)
- Base AUC: 0.7376  →  Meta AUC: 0.7367
- Monotonic realized rate (Lock > Strong > Lean > Avoid): PASS

| Bucket | n | Realized | Mean Meta-Prob | Mean Base-Prob | Lift |
|---|---:|---:|---:|---:|---:|
| Lock | 3533 | 0.0849 | 0.0838 | 0.0761 | 2.79x |
| Strong | 5467 | 0.0573 | 0.0568 | 0.0596 | 1.88x |
| Lean | 9263 | 0.0337 | 0.0365 | 0.0400 | 1.11x |
| Avoid | 18386 | 0.0104 | 0.0101 | 0.0115 | 0.34x |

## HIT
- Train rows: 146,596  |  Test rows: 36,649
- Baseline realized rate (test): 0.2162
- Thresholds (meta-prob): Lock ≥ 0.3404, Strong ≥ 0.2923, Lean ≥ 0.2429, else Avoid
- Base model Brier: 0.16076 → Meta Brier: 0.16073  (Δ +0.000033)
- Base AUC: 0.6573  →  Meta AUC: 0.6572
- Monotonic realized rate (Lock > Strong > Lean > Avoid): PASS

| Bucket | n | Realized | Mean Meta-Prob | Mean Base-Prob | Lift |
|---|---:|---:|---:|---:|---:|
| Lock | 3548 | 0.3441 | 0.3621 | 0.3649 | 1.59x |
| Strong | 5428 | 0.3113 | 0.3162 | 0.3228 | 1.44x |
| Lean | 9281 | 0.2659 | 0.2651 | 0.2740 | 1.23x |
| Avoid | 18392 | 0.1384 | 0.1401 | 0.1473 | 0.64x |

## XBH
- Train rows: 146,596  |  Test rows: 36,649
- Baseline realized rate (test): 0.0759
- Thresholds (meta-prob): Lock ≥ 0.1351, Strong ≥ 0.1066, Lean ≥ 0.0798, else Avoid
- Base model Brier: 0.06824 → Meta Brier: 0.06819  (Δ +0.000053)
- Base AUC: 0.6779  →  Meta AUC: 0.6776
- Monotonic realized rate (Lock > Strong > Lean > Avoid): PASS

| Bucket | n | Realized | Mean Meta-Prob | Mean Base-Prob | Lift |
|---|---:|---:|---:|---:|---:|
| Lock | 3599 | 0.1542 | 0.1572 | 0.1585 | 2.03x |
| Strong | 5443 | 0.1146 | 0.1204 | 0.1292 | 1.51x |
| Lean | 9236 | 0.0943 | 0.0915 | 0.1003 | 1.24x |
| Avoid | 18371 | 0.0397 | 0.0404 | 0.0466 | 0.52x |
