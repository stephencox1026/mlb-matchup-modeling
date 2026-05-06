## Section 0: Bucket Health (drift monitor)

> **CRITICAL drift** in: HR/Very Low (drift 10.7% > 10%), HR/Medium (drift 13.1% > 10%), HR/High (High bucket lift 0.76x < 1.0x floor), Hit/Medium (drift 29.2% > 10%), XBH/Low (drift 12.2% > 10%), XBH/Medium (drift 46.3% > 10%), XBH/High (High bucket lift 0.00x < 1.0x floor)

> **WARN drift triggered refit** in: HR/Low

**Window:** trailing 14 days (2060 filled outcomes)

**HR (Beast) buckets:** Very Low → **CRITICAL** · Low → **WARN** · Medium → **CRITICAL** · High → **CRITICAL**

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
