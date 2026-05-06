# Confidence label audit (CB2)

## Symptom (2026-04-24 → 2026-04-26 live data, n=1172)

HR realized rate per `conf_hr_label` was **non-monotonic** at Medium:

| Bucket | n | Realized | 95% CI |
|---|---:|---:|---|
| Very Low | 942 | 9.9% | [8.1%, 11.9%] |
| Low | 132 | 12.1% | [7.6%, 18.8%] |
| **Medium** | 87 | **8.0%** | [4.0%, 15.7%] |
| High | 11 | 18.2% | [5.1%, 47.7%] |

Medium realized **below** Very Low — a confidence label that's supposed to be more confident than Very Low predicted worse outcomes.

## Diagnosis

`compute_confidence()` in [src/narrative_engine.py](../src/narrative_engine.py) was applying a +15-20% boost to `bvp_hr` for any batter with `bvp_hr_count >= 2` **regardless of `bvp_pa`**. Six of the 24 boosted picks had ≤13 BvP PA — small-sample HR clusters (e.g., Buxton 3 HR in 5 PA vs Rasmussen) were treated identically to high-confidence BvP histories (Riley 6 HR in 60 PA vs Nola).

Cross-tab of the original Medium HR bucket by `(bvp_hr_factor, convergence_hr_factor)`:

| bvp_hr | convergence_hr | n | realized | mean p_hr |
|---:|---:|---:|---:|---:|
| 1.00 | 1.10 | 23 | 13.0% | 0.060 |
| 1.05 | 1.00 | 40 | 7.5% | 0.044 |
| 1.05 | 1.10 | 11 | 9.1% | 0.061 |
| **1.15** | 1.00 | **8** | **0.0%** | **0.043** |
| **1.20** | 1.00 | **5** | **0.0%** | **0.042** |

The 13 picks with `bvp_hr ∈ [1.15, 1.20]` and `convergence_hr = 1.0` went **0-for-13**. They were inflated into Medium by BvP signal alone, while the per-PA model itself was saying "no power expected" (mean p_hr 0.043). The model was right; the BvP boost was anti-predictive.

## Fix

Added PA-size gates that mirror the existing pattern for `bvp_hr == 1`:

```python
elif bvp_hr >= 3 and bvp_pa >= 25:
    factors["bvp_hr"] = 1.20
    factors["bvp_hit"] = 1.05 if bvp_ba >= 0.250 else 1.0
elif bvp_hr >= 2 and bvp_pa >= 20:
    factors["bvp_hr"] = 1.15
    factors["bvp_hit"] = 1.05 if bvp_ba >= 0.250 else 1.0
elif bvp_hr >= 2:
    # 2-3 HR but small sample (PA < 20): modest signal only.
    factors["bvp_hr"] = 1.05
    factors["bvp_hit"] = 1.0
```

## Replay against historical data

After applying the new logic to the existing 1172-row sample:

| Bucket | n (old) | realized (old) | n (new) | realized (new) |
|---|---:|---:|---:|---:|
| Very Low | 942 | 9.9% | 951 | 9.9% |
| Low | 132 | 12.1% | 159 | 10.7% |
| Medium | 87 | **8.0%** | 56 | **10.7%** |
| High | 11 | 18.2% | 6 | 16.7% |

**Monotonicity restored.** 31 picks moved out of Medium (mostly into Low or Very Low), reflecting that small-sample BvP shouldn't have inflated their confidence in the first place. The High bucket lost 5 picks but kept its lift (~1.7× baseline).

## Caveats

- New High bucket has n=6 — too small to be statistically confident in the 16.7% realized rate (Wilson CI [3%, 56%]).
- Wider issue: BvP appears to be weakly predictive overall in this 3-day window. Phase 4 M1 will replace the entire hand-tuned conf system with a learned residual meta-model. CB2 is a structural-bug fix, not a final solution.

## Related work

- M1 (Phase 4): replaces the hand-tuned conf factors with a learned residual model (Lock / Strong / Lean / Avoid labels).
- H4 (Phase 4): tests whether each conf sub-factor adds High-bucket lift on the H2 backtest data; drops factors that don't pull weight.
