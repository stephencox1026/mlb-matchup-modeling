# M2: Per-Game Head Summary

Stacked HGBR Classifier on per-(batter, game) aggregations of `beast` per-PA predictions. Compares the per-game model to the `1-exp(-sum p_PA)` independence-approximation baseline.

| Target | n_test | Baseline rate | Base Brier | PG Brier | Δ Brier | Base AUC | PG AUC | Base Top-Dec | PG Top-Dec | Base Lift | PG Lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hr | 9,772 | 0.1070 | 0.09129 | 0.09079 | +0.00050 | 0.6924 | 0.6979 | 0.2311 | 0.2555 | 2.16x | 2.39x |
| hit | 9,772 | 0.5781 | 0.20899 | 0.20587 | +0.00312 | 0.7191 | 0.7231 | 0.8415 | 0.8507 | 1.46x | 1.47x |
| xbh | 9,772 | 0.2539 | 0.17578 | 0.17402 | +0.00176 | 0.6794 | 0.6859 | 0.4591 | 0.4908 | 1.81x | 1.93x |

Δ Brier > 0 = per-game head improves over independence baseline.