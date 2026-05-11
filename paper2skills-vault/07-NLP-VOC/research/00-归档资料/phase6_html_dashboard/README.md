---
name: phase6-html-dashboard-archive-readme
description: Phase 6 D10 HTML 看板归档说明（C 路径产物）。Phase 7 D3+ 已用 Superset 替代，HTML 保留作为历史对比。
archived_date: 2026-05-11
phase: phase6-d10
---

# Phase 6 D10 HTML Dashboard 归档

## 这是什么

`dashboard-2026-W19.html`（125KB）是 Phase 6 D10 的 **C 路径**产物：
通过 [`bi_dashboard_generator.py`](../../02-脚本工具/01-标签进化/bi_dashboard_generator.py) 离线渲染的单文件 HTML BI 看板。

## 为什么归档

Phase 7 D1-D4（2026-05-08 → 2026-05-11）搭建了完整的 Superset BI 系统（B 路径）：
- 12 charts + 8 dashboards + 10 native filters
- 真实数据渲染、交互过滤、ZIP 可重建
- 详见 [phase7_d3_progress_report.md](../../04-输出结果/03-审计报告/phase7_d3_progress_report.md) 和 [phase7_d4_progress_report.md](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md)

HTML 看板失去了产线用途，但作为 C 路径参考样本保留。

## 重新生成

如需重新生成：

```bash
python3 research/02-脚本工具/01-标签进化/bi_dashboard_generator.py \
  --input <vault>/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl \
  --output <vault>/04-输出结果/bi-dashboard/dashboard-<week>.html
```
