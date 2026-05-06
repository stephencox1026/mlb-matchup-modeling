## Beast HR — calibration snapshot

_Generated 2026-05-06T04:08:29+00:00 UTC_

**Drift window:** 14 days | **filled tracking rows:** 2060

### Section 0 — HR buckets only

| Bucket | n | n PA | Predicted | Realized | Lift | Status | Note |
|---|---:|---:|---:|---:|---:|---|---|
| Very Low | 1064 | 2646 | 0.0348 | 0.0385 | 0.96x | CRITICAL | drift 10.7% > 10% |
| Low | 149 | 393 | 0.0512 | 0.0483 | 1.21x | WARN | drift 5.6% > 5%; refit triggered |
| Medium | 97 | 254 | 0.0499 | 0.0433 | 1.08x | CRITICAL | drift 13.1% > 10% |
| High | 12 | 33 | 0.0674 | 0.0303 | 0.76x | CRITICAL | High bucket lift 0.76x < 1.0x floor |

### C1 calibration — HR target only

_Trained 2026-05-06T04:08:28.595313+00:00 | rows=2060 | slates ['2026-04-20', '2026-04-29']_

| Bucket | Method | n_rows | n_pa | mean_pred | mean_real | detail |
|---|---|---:|---:|---:|---:|---|
| Very Low | isotonic | 555 | 1368.0 | 0.0325 | 0.0431 | 8 knots |
| Low | isotonic | 142 | 358.0 | 0.0355 | 0.0363 | 8 knots |
| Medium | isotonic | 348 | 867.0 | 0.0371 | 0.0334 | 10 knots |
| High | isotonic | 277 | 733.0 | 0.0516 | 0.0437 | 14 knots |
