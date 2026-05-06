## Section 14: Conviction Picks (High-conf only)

**Slate:** 2026-05-05 | **Total picks scanned:** 390

High-conf bucket sizes: HR=0, Hit=0, XBH=0

_Filter rule: M1 **`Lock`** only if hand-tuned **`conf_*_label` is High or Medium**; else **`conf_*_label == "High"`** fallback. Caps **15** picks per target. Sort: Score (raw P × confidence)._

### HR Locks (n=0) — _M1 meta-model (Lock)_

_No Lock picks for this target tonight._

### Hit Locks (n=0) — _M1 meta-model (Lock)_

_No Lock picks for this target tonight._

### XBH Locks (n=0) — _M1 meta-model (Lock)_

_No Lock picks for this target tonight._

### Bucket Health Footer (rolling 14d)

| Target | Bucket | n | n PA | Predicted | Realized | Lift | Status | Note |
|---|---|---:|---:|---:|---:|---:|---|---|
| HR | Very Low | 1064 | 2646 | 0.0348 | 0.0385 | 0.96x | CRITICAL | drift 10.7% > 10% |
| HR | Low | 149 | 393 | 0.0512 | 0.0483 | 1.21x | WARN | drift 5.6% > 5%; refit triggered |
| HR | Medium | 97 | 254 | 0.0499 | 0.0433 | 1.08x | CRITICAL | drift 13.1% > 10% |
| HR | High | 12 | 33 | 0.0674 | 0.0303 | 0.76x | CRITICAL | High bucket lift 0.76x < 1.0x floor |
| Hit | Very Low | 938 | 2324 | 0.2313 | 0.2336 | 1.00x | HEALTHY | drift 1.0% |
| Hit | Low | 283 | 731 | 0.2480 | 0.2531 | 1.08x | HEALTHY | drift 2.1% |
| Hit | Medium | 101 | 271 | 0.2604 | 0.1845 | 0.79x | CRITICAL | drift 29.2% > 10% |
| Hit | High | 0 | 0 | — | — | — | INSUFFICIENT | n=0 |
| XBH | Very Low | 1031 | 2571 | 0.0871 | 0.0902 | 1.03x | HEALTHY | drift 3.6% |
| XBH | Low | 215 | 546 | 0.0959 | 0.0842 | 0.97x | CRITICAL | drift 12.2% > 10% |
| XBH | Medium | 75 | 206 | 0.1085 | 0.0583 | 0.67x | CRITICAL | drift 46.3% > 10% |
| XBH | High | 1 | 3 | 0.1096 | 0.0000 | 0.00x | CRITICAL | High bucket lift 0.00x < 1.0x floor |

_Generated 2026-05-06T04:08:30.292568+00:00_