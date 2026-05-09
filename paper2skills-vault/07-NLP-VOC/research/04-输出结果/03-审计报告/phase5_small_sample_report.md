# Phase 5 Quality Gate — WEEK1

**Overall**: ✅ PASS

| # | Check | Threshold | Value | Pass | Note |
|---|---|---|---:|:---:|---|
| 1 | LLM Top-1 accuracy vs golden | `>= 0.85` | 1.0 | ✅ PASS | n_eval=149 |
| 2 | Per-label F1 weighted (TOP-30) | `>= 0.75` | 0.9889 | ✅ PASS |  |
| 3 | Top-3 mean Jaccard (recall proxy) | `>= 0.50` | 0.9829 | ✅ PASS |  |
| 4 | LLM sentiment Cohen κ | `>= 0.65` | 0.9887 | ✅ PASS |  |
| 5 | ABSA aspect/record in [1, 5] | `1.0 <= x <= 5.0` | 2.91 | ✅ PASS | total_aspects=1303, n_total=500 |
| 6 | ABSA empty rate | `< 0.10` | 0.0876 | ✅ PASS |  |
| 7 | Proxy NPS three-way agreement vs golden | `>= 0.85` | 0.994 | ✅ PASS | n_eval=168 |
| 8 | Tag mutex (POS+NEG co-occurrence) rate | `< 0.03` | 0.0038 | ✅ PASS | violations=19, n=5000 |
| 9 | JSON parse failure rate | `< 0.01` | 0.0 | ✅ PASS | failed=0, n=5000 |
