---
name: archive-root-readme
description: 00-归档资料/ 归档层入口，按子目录索引各类历史产物。所有内容不参与日常引用，仅作历史追溯使用。
title: 归档资料目录入口
doc_type: index
module: voc-nlp
status: stable
created: 2026-04-22
updated: 2026-05-11
---

# 00-归档资料: 归档层

本目录存放历史版本、已被取代的中间产物、过期产物。**不参与日常引用**，仅作历史追溯使用。

## 内容

| 子目录 | 内容 | 归档时间 |
|---|---|---|
| `phase4_archive/` | Phase 1-4 阶段产物：6 个 audit.json + README（13 个大 jsonl 已删，回收 3.8G） | Phase 5 初 |
| `phase6_html_dashboard/` | ⭐ Phase 6 D10 HTML 看板（被 Phase 7 Superset 替代） | 2026-05-11 |
| `weekly_drafts/` | ⭐ W19 / W19-v41 早期周报草稿（最终版在 [10-周报/2026-W19-d9/](../04-输出结果/10-周报/2026-W19-d9/)） | 2026-05-11 |
| `labeling-outputs/` | 历史打标输出（v1, v3.3） | Phase 4 |
| `multi-version-excel/` | ASIN 历史版本 Excel | Phase 3 |
| `runtime-outputs/` | momcozy_integration 运行时产物 | Phase 3 |

## 关键归档说明

### phase4_archive/

- 13 个 jsonl 大文件（共 ~3.8G）已于 2026-05-11 清理（Phase 5+ 已不再回看，audit.json 保留作摘要）
- 详见 [phase4_archive/README.md](phase4_archive/README.md)

### phase6_html_dashboard/

- Phase 6 D10 的 C 路径产物 `dashboard-2026-W19.html`（125KB）
- Phase 7 D3+ Superset B 路径已取代（见 [phase7_complete_retrospective.md](../04-输出结果/03-审计报告/phase7_complete_retrospective.md)）
- 详见 [phase6_html_dashboard/README.md](phase6_html_dashboard/README.md)

### weekly_drafts/

- `2026-W19/`（4 文件，srac 命名旧版）+ `2026-W19-v41/`（4 文件，v4.1 切换重打版）
- 当前产线版本在 [04-输出结果/10-周报/2026-W19-d9/](../04-输出结果/10-周报/2026-W19-d9/)（7 部门 × AGRS+MAA = 28 文件）
- 详见 [weekly_drafts/README.md](weekly_drafts/README.md)

## 保留策略

- 大数据文件（>100MB jsonl/xlsx）：在 [.gitignore](../../.gitignore) 中排除，本地保留 30 天后可手动清理
- 小元数据（audit.json / README.md / 周报草稿等）：入仓保留作历史依据
- **不再补充新内容**：本目录是"过去"的快照，新阶段产物归位到对应 phase 目录

## 历史回溯入口

| 你要查 | 去这里 |
|---|---|
| Phase 1-4 怎么覆盖 82.58% 的 | [phase4_archive/](phase4_archive/) 下的 audit.json |
| Phase 6 D10 HTML 看板长什么样 | [phase6_html_dashboard/dashboard-2026-W19.html](phase6_html_dashboard/dashboard-2026-W19.html) |
| 早期周报 srac 命名格式 | [weekly_drafts/2026-W19/](weekly_drafts/2026-W19/) |
| v4.1 字典切换前后的周报对比 | [weekly_drafts/2026-W19-v41/](weekly_drafts/2026-W19-v41/) vs [10-周报/2026-W19-d9/](../04-输出结果/10-周报/2026-W19-d9/) |
