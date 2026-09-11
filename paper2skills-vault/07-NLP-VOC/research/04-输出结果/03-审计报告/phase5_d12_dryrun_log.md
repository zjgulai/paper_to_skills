# 月度进化运行日志

- 时间：2026-05-09T12:31:19
- 模式：dry-run
- 总判定：🟢 PASS

| # | 步骤 | 状态 | 耗时 | 详情 |
|---:|---|:---:|---:|---|
| 1 | step1_zero_label_extractor | ✅ | 4.03s | total=364,569 zero=87,098 low_conf=134,243 kept=5,000 |
| 2 | step2_closed_relabel | ⏭️ SKIP | 0.02s | [dry-run] would relabel 5,000 records with latest dict |
| 3 | step3_open_set_sampling | ⏭️ SKIP | 0.07s | in=5,000 sampled_zero_only=10 (5.0%) |
| 4 | step4_candidate_filter | ⏭️ SKIP | 0.00s | [dry-run] synthesized 1 fake candidate to exercise pipeline |
| 5 | step5_alchemist_lf | ⏭️ SKIP | 0.00s | [dry-run] generated stub LF for 1 candidate |
| 6 | step6_active_learning | ⏭️ SKIP | 0.00s | [dry-run] queued 1 synthetic uncertain candidate |
| 7 | step7_dict_update | ✅ | 0.97s | base_rows=268 new_rows=269 diff=1 |
| 8 | step8_bi_recompute | ⏭️ SKIP | 0.00s | [dry-run] 仅记录应触发：dual_coverage_calculator + maa_strategy_generator + agrs_summarizer 重跑（不实际执行） |
