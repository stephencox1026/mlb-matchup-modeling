# H4: Confidence Factor Driver Audit

**Window:** 2026-04-20 → 2026-04-26  (1105 vs-SP outcomes)

## HR target

Baseline (1105 picks, all): realized 0.0995
Baseline High bucket: n=12, realized=0.083, lift=0.84x, 95% CI [0.015, 0.354]

**Marginal contribution: strip each factor, recompute bucket assignments**

| Factor stripped | New High n | New High realized | New lift | Δ vs baseline |
|---|---:|---:|---:|---:|
| pitcher_data | 12 | 0.083 | 0.84x | 0.00x |
| bvp_hr | 0 | nan | nanx | nanx |
| bvp_hit | 12 | 0.083 | 0.84x | 0.00x |
| staleness | 108 | 0.111 | 1.12x | -0.28x |
| convergence_hr | 0 | nan | nanx | nanx |
| convergence_hit | 12 | 0.083 | 0.84x | 0.00x |

Interpretation: a factor whose Δ > 0 is **adding lift** to the High bucket (stripping it makes High realize worse). Δ ≤ 0 means the factor is **not earning its keep** for this target.

## HIT target

Baseline (1105 picks, all): realized 0.4824
Baseline High bucket: n=0, realized=nan, lift=nanx, 95% CI n/a

**Marginal contribution: strip each factor, recompute bucket assignments**

| Factor stripped | New High n | New High realized | New lift | Δ vs baseline |
|---|---:|---:|---:|---:|
| pitcher_data | 0 | nan | nanx | nanx |
| bvp_hr | 0 | nan | nanx | nanx |
| bvp_hit | 0 | nan | nanx | nanx |
| staleness | 82 | 0.415 | 0.86x | nanx |
| convergence_hr | 0 | nan | nanx | nanx |
| convergence_hit | 0 | nan | nanx | nanx |

Interpretation: a factor whose Δ > 0 is **adding lift** to the High bucket (stripping it makes High realize worse). Δ ≤ 0 means the factor is **not earning its keep** for this target.

## XBH target

Baseline (1105 picks, all): realized 0.2109
Baseline High bucket: n=1, realized=0.000, lift=0.00x, 95% CI [0.000, 0.793]

**Marginal contribution: strip each factor, recompute bucket assignments**

| Factor stripped | New High n | New High realized | New lift | Δ vs baseline |
|---|---:|---:|---:|---:|
| pitcher_data | 1 | 0.000 | 0.00x | 0.00x |
| bvp_hr | 0 | nan | nanx | nanx |
| bvp_hit | 0 | nan | nanx | nanx |
| staleness | 76 | 0.158 | 0.75x | -0.75x |
| convergence_hr | 0 | nan | nanx | nanx |
| convergence_hit | 0 | nan | nanx | nanx |

Interpretation: a factor whose Δ > 0 is **adding lift** to the High bucket (stripping it makes High realize worse). Δ ≤ 0 means the factor is **not earning its keep** for this target.
