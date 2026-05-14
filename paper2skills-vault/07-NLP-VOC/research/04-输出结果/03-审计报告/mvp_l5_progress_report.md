---
name: mvp-l5-progress-report
description: VOC 深度分析 MVP L5 D-Diag 三专题诊断看板上线进度报告。15 chart × 3 dashboard 通过 Superset API 装配 + Playwright 端到端验证 + multimodal-looker 视觉抽查 0 console error / 15/15 chart 渲染。沿用 L4 的 4 个 dataset，0 view 改动、0 schema 改动，对存量 9 个 dashboard / 19 个 chart / 10 个 dataset 完全 0 影响。
title: MVP L5 · D-Diag 3 专题诊断看板上线
doc_type: progress-report
module: voc-nlp
phase: mvp-l5
status: completed
created: 2026-05-14
updated: 2026-05-14
owner: self
source: ai
---

# MVP L5 · D-Diag 三专题诊断看板上线 进度报告

> 落地日期：2026-05-14 · 上线分支：`feat/voc-deep-analysis-mvp` · 上一节点：[L4 D-Health 看板](mvp_l4_progress_report.md)（如已归档）。

## 一、目标

把 L5.1 已设计的「**产品力 / 服务力 / 内容品牌力**」3 个 D-Diag 专题，
通过 **15 个 Superset chart + 3 个 dashboard**，
在生产 BI 上一次性发布，沿用 L4 已注册的 4 个深度分析 dataset。

## 二、上线对象一览

### Dashboards（3 个 · id 10-12）

| id | 主题 | slug | URL |
|----|------|------|-----|
| 10 | 产品力诊断 D-Diag-Product | `voc-deep-diag-product` | https://voc.lute-tlz-dddd.top/superset/dashboard/10/ |
| 11 | 服务力诊断 D-Diag-Service | `voc-deep-diag-service` | https://voc.lute-tlz-dddd.top/superset/dashboard/11/ |
| 12 | 内容品牌力诊断 D-Diag-Brand | `voc-deep-diag-brand` | https://voc.lute-tlz-dddd.top/superset/dashboard/12/ |

### Charts（15 个 · id 20-34）

| id | code | dashboard | viz | name |
|----|------|-----------|------|------|
| 20 | P-1 | Product | table | 产品中心 SAT 失血 Top 10 |
| 21 | P-2 | Product | heatmap | 产品中心 × 国家 负向率热力 |
| 22 | P-3 | Product | table | 产品中心 × 国家 体量+负向率 |
| 23 | P-4 | Product | dist_bar | L1 使用层 SAT 全量表现 |
| 24 | P-5 | Product | dist_bar | 产品中心 国家×负向率（单批快照） |
| 25 | S-1 | Service | heatmap | 服务三部门 × 国家 负向率热力 |
| 26 | S-2 | Service | table | L3 服务层 SAT 失血 Top 10 |
| 27 | S-3 | Service | heatmap | 数据源 × 国家 NPS 校准矩阵 |
| 28 | S-4 | Service | dist_bar | 仓储物流 SAT 明细排行 |
| 29 | S-5 | Service | pie | 核心质量 SAT 体量分布 |
| 30 | B-1 | Brand | table | 品牌市场中心 SAT Top 10 |
| 31 | B-2 | Brand | dist_bar | L4 品牌层 SAT × 国家 |
| 32 | B-3 | Brand | dist_bar | SAT_L4_01 品牌声量 by 国家（单批快照） |
| 33 | B-4 | Brand | dist_bar | SAT_L3_17 内容/社群 × 国家 |
| 34 | B-5 | Brand | dist_bar | 节点体量 vs 节点得分 双轴 |

## 三、验收（端到端）

### 3.1 Playwright 自动化

3 dashboard 全部 200 加载 + 0 `pageerror` + 0 `console.error`。结果文件：[playwright_l5_results.json](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L5-20260514-163344/playwright_l5_results.json)。

### 3.2 视觉抽查（multimodal-looker）

| Dashboard | 5/5 chart 渲染 | 错误覆盖层 | 关键信号 |
|---|:---:|:---:|---|
| 10 Product | ✅ | 0 | OTHER 44.69% / UK 35.30% / SAT_L1_01 30K hits |
| 11 Service | ✅ | 0 | 品质 78.71% 极端 / S-3 NPS 矩阵成形 |
| 12 Brand | ✅ | 0 | SAT_L4_01 36K hits / SAT_L3_17 112K hits |

截图固化：

- [Product 全页 PNG](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L5-20260514-163344/voc-d-diag-product-10.png)
- [Service 全页 PNG](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L5-20260514-163344/voc-d-diag-service-11.png)
- [Brand 全页 PNG](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L5-20260514-163344/voc-d-diag-brand-12.png)

## 四、已知非阻塞限制（loaded_at proxy）

L1 阶段 `ts_inferred = voc_review.loaded_at`（ETL 入库时间，非真实写评时间），当前所有 364,569 条 review 都共享 `month=2026-05-01`（单批 ETL）。

直接影响：line/趋势 chart 在当前数据下退化为单点 → 不可见。

L5.4 处置：

- **P-5**（产品中心月度负向率趋势）→ 改为 `dist_bar` 展现「单批快照 by 国家」语义
- **B-3**（SAT_L4_01 品牌声量趋势 by source）→ 改为 `dist_bar` 展现「pos/neg/neu by 国家」

upgrade path：未来 [01-mvp-design.md §九](../../01-设计文档/10-VOC深度分析MVP/01-mvp-design.md) 的 "真实 timestamp 入库" 完成后，**仅修改这 2 chart 的 `viz_type` + `granularity_sqla`**，无需重建。

## 五、对存量资产的影响（0 影响）

| 类型 | L5 前 | L5 后 | 影响 |
|---|---:|---:|---|
| Dataset | 10 | 10 | **0 新增 / 0 删除**（沿用 L4 的 4 个：v_atomic_indicator_score / v_aipl_node_score / v_country_dept_health / v_proxy_nps_calibrated） |
| Chart | 19 | 34 | +15（id 20-34，与 id 1-19 完全独立） |
| Dashboard | 9 | 12 | +3（id 10-12） |
| Postgres view | 10 | 10 | **0 新增 / 0 删除** |
| voc_review schema | — | — | **0 改动** |
| dim_tag schema | — | — | **0 改动** |

## 六、回滚预案

一键脚本：[mvp_l5_rollback.py](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L5-20260514-163344/mvp_l5_rollback.py)

```python
DELETE /api/v1/dashboard/{10,11,12}
DELETE /api/v1/chart/{20..34}
# 0 dataset / 0 view 改动 → L4 D-Health (id 9) 完全不受影响
```

## 七、备份索引

`~/.secrets/backups/voc-deep-analysis-mvp/L5-20260514-163344/`

| 文件 | 用途 |
|---|---|
| `MANIFEST.md` | L5 baseline 完整说明 |
| `mvp_l5_create_charts.py` | 15 chart 装配脚本（参数完整可重放） |
| `mvp_l5_create_dashboards.py` | 3 dashboard 装配 + chart binding |
| `mvp_l5_rollback.py` | 一键回滚 |
| `mvp_l5_chart_ids.json` | 创建结果（含 viz_type） |
| `mvp_l5_dashboard_ids.json` | 创建结果（含 chart_ids） |
| `pre_l5_snapshot.json` | 上线前 Superset 状态 |
| `post_l5_snapshot.json` | 上线后 Superset 状态 |
| `l5_full_definitions.json` | 15 chart + 3 dashboard 的 `params` / `position_json` 完整定义（GitOps 重放源） |
| `playwright_l5_results.json` | 端到端验证结果 |
| `voc-d-diag-{product,service,brand}-{10,11,12}.png` | 全页截图 |

## 八、MVP 总进度

| 阶段 | 状态 | 工日 |
|---|:---:|---:|
| L0 准备 | ✅ | done |
| L1 voc_review 扩展 | ✅ | done |
| L2 50 SAT 映射 | ✅ | done |
| L3 4 深度视图 | ✅ | done |
| L4 D-Health 看板 | ✅ | done |
| **L5 D-Diag 3 专题诊断** | **✅** | **done** |
| L6 D-Action 部门行动 | ⏳ | 2 工日 |
| L7 部署 + 文档 + 培训 | ⏳ | 1 工日 |

下一步 **L6 D-Action**（7 部门行动分发表 · 合并到现有 7 部门 dashboard 末尾），需要等用户 GO L6。

## 九、文件清单（提交到仓库）

| 文件 | 说明 |
|---|---|
| `research/02-脚本工具/01-标签进化/scripts/mvp_l5_create_charts.py` | 15 chart 装配 |
| `research/02-脚本工具/01-标签进化/scripts/mvp_l5_create_dashboards.py` | 3 dashboard 装配 |
| `research/02-脚本工具/01-标签进化/scripts/mvp_l5_rollback.py` | 一键回滚 |
| `research/04-输出结果/03-审计报告/mvp_l5_progress_report.md` | 本进度报告 |
