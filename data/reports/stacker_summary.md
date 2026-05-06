# H5: Stacker Ensemble Summary

Logistic regression stacker over per-PA member predictions. Acceptance gate: ship only if the stacker's Brier vs best individual member has a 95% bootstrap CI **excluding 0** (i.e., stat-significantly lower).

| Target | Members | Stack Brier | Best Member | Best Brier | Δ Brier (Stack - Best) | 95% CI | Ships? |
|---|---|---:|---|---:|---:|---|---|
| hr | beast, exp, prod, recency, recent365, lr_interactions | 0.02959 | beast | 0.02959 | +0.000006 | [-0.000048, +0.000057] | no (CI includes 0) |
| hit | beast, exp, prod, recency, recent365, lr_interactions | 0.16116 | beast | 0.16124 | -0.000075 | [-0.000226, +0.000060] | no (CI includes 0) |
| xbh | beast, exp, prod, recency, recent365, lr_interactions | 0.06899 | beast | 0.06897 | +0.000020 | [-0.000076, +0.000124] | no (CI includes 0) |

## Stacker weights per target

- **hr**: beast: +7.605, exp: +6.192, prod: +6.017, recency: +5.709, recent365: +6.340, lr_interactions: +1.096, intercept -4.775
- **hit**: beast: +3.632, exp: +0.651, prod: +0.249, recency: +0.208, recent365: +2.354, lr_interactions: +0.075, intercept -3.013
- **xbh**: beast: +3.574, exp: +3.127, prod: +3.061, recency: +3.011, recent365: +2.821, lr_interactions: -0.050, intercept -3.957