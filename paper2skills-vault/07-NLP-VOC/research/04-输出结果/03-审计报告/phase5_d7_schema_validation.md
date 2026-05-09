# Phase 5 Schema Validation — v3.9

**Records**: 5000  ｜  **Overall**: 🟢 PASS

| # | Check | Pass | Detail |
|---|---|:---:|---|
| S1 | Required fields present | ✅ | 0 records missing |
| S2 | labels[].tag_id ∈ dict v3.9 | ✅ | 0 invalid tags (0 distinct) |
| S3 | consensus_labels[].tag_id ∈ dict | ✅ | 0 invalid consensus tags |
| S4 | persona_tags[].tag_id matches P-L2-NN | ✅ | 0 invalid persona tags |
| S5 | overall_sentiment ∈ {pos/neu/neg/None} | ✅ | 0 invalid sentiment values |
| S6 | proxy_nps_final ∈ {prom/pas/det/None} | ✅ | 0 invalid NPS values |
| S7 | POS/NEG hard conflict-free (both sides w/o evidence) | ✅ | 0 hard conflicts (mixed-sentiment reviews with evidence on both sides allowed) |
