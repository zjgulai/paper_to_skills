---
name: mvp-l6-progress-report
description: VOC 深度分析 MVP L6 D-Action 7 部门行动总队列上线进度报告。8 chart × 双路径上线（Path A 追加到 7 dept dashboard 末尾 + Path B 新建 D-Action 总览 dashboard），priority_score = hit_negative × pct_negative / 100，过滤排除字典 [待填写] 占位符。Playwright 实测 8/8 dashboard 0 console error，视觉抽查 master Top 3 与 dry-run 完全一致。
title: MVP L6 · D-Action 7 部门行动总队列上线
doc_type: progress-report
module: voc-nlp
phase: mvp-l6
status: completed
created: 2026-05-14
updated: 2026-05-14
owner: self
source: ai
---

# MVP L6 · D-Action 7 部门行动总队列上线 进度报告

> 落地日期：2026-05-14 · 上线分支：`feat/voc-deep-analysis-mvp` · 上一节点：[L5 D-Diag 3 专题诊断](mvp_l5_progress_report.md)。

## 一、目标

把 [01-mvp-design.md §126-129](../../01-设计文档/10-VOC深度分析MVP/01-mvp-design.md) 描述的「**D-Action 7 部门行动分发**」落地：
让每个部门在自己的 dashboard 末尾看到「最该处理的 Top 10 行动项」，
管理层在总览 dashboard 看到「全部门 Top 20 行动总队列」。

## 二、决策（用户确认）

| 决策 | 选择 |
|---|---|
| 实现方式 | **C（A+B 都做）**：7 dept dashboard 追加 + 1 总览 dashboard |
| 优先级公式 | **A**：`priority_score = hit_negative × pct_negative / 100` |
| 数据源 | dataset 2 = `v_label_with_dept`（已含 dept_owner / biz_action / strategy_pkg / sentiment_preset） |
| 0 view 改动 | 沿用 L4/L5 「不新建视图」原则 |

## 三、上线对象

### 8 个 D-Action chart（id 35-42）

| id | code | dept | viz | 用途 |
|---:|---|---|---|---|
| 35 | DA-产品中心 | 产品中心 | table | 产品中心 行动队列 Top 10 |
| 36 | DA-全球客服中心 | 全球客服中心 | table | 全球客服中心 行动队列 Top 10 |
| 37 | DA-仓储物流部 | 仓储物流部 | table | 仓储物流部 行动队列 Top 10 |
| 38 | DA-品牌市场中心 | 品牌市场中心 | table | 品牌市场中心 行动队列 Top 10 |
| 39 | DA-品质管理中心 | 品质管理中心 | table | 品质管理中心 行动队列 Top 10 |
| 40 | DA-电商运营部 | 电商运营部 | table | 电商运营部 行动队列 Top 10 |
| 41 | DA-法务合规部 | 法务合规部 | table | 法务合规部 行动队列 Top 10 |
| 42 | DA-MASTER | ALL | table | 全部门 行动总队列 Top 20 |

### Path A · 7 dept dashboard 末尾追加（id 2-8）

每个 dept dashboard 现在的内容：原 Top 10 话题图（保留）+ D-Action 行动队列。
位置：在 GRID_ID.children 末尾追加 `ROW-DA-{cid}`，宽度 12 / 高度 70。

| dashboard_id | dept | + chart |
|---:|---|---:|
| 2 | 全球客服中心 | 36 |
| 3 | 产品中心 | 35 |
| 4 | 仓储物流部 | 37 |
| 5 | 品牌市场中心 | 38 |
| 6 | 电商运营部 | 40 |
| 7 | 品质管理中心 | 39 |
| 8 | 法务合规部 | 41 |

### Path B · D-Action 总览 dashboard（id 13）

URL：https://voc.lute-tlz-dddd.top/superset/dashboard/13/
slug：`voc-deep-action-overview`

布局（4 row × 8 chart）：

```
┌──────────────────────────────────────────┐
│ ROW_MASTER  CHART 42 全部门 Top 20  (12w) │
├─────────────┬────────────┬──────────────┤
│ 产品中心(35) │ 品牌市场(38)│ 品质管理(39)  │
│   (4w)       │   (4w)     │   (4w)       │
├──────────────┴────────────┼──────────────┤
│ 仓储物流(37)               │ 全球客服(36)  │
│   (6w)                    │   (6w)        │
├───────────────────────────┼──────────────┤
│ 电商运营(40)               │ 法务合规(41)  │
│   (6w)                    │   (6w)        │
└───────────────────────────┴──────────────┘
```

## 四、关键过滤（修复）

字典里 76 个产品中心标签中 **15 个的 `biz_action = '【待填写】'`** 占位符。
原过滤 `biz_action <> ''` 不触发，导致占位行污染 Top 10。

最终 SQL filter（`mvp_l6_fix_action_filter.py`）：

```sql
WHERE biz_action IS NOT NULL
  AND biz_action <> ''
  AND biz_action NOT LIKE '%待填写%'
```

修复后 Master Top 3 与 dry-run **100% 一致**：

| 排名 | dept | 主题 | priority_score | dry-run | dashboard |
|---:|---|---|---:|---:|---|
| 1 | 产品中心 | 延迟 | 4046.9 | ✅ | ✅ |
| 2 | 产品中心 | 尺码小 | 2843.8 | ✅ | ✅ |
| 3 | 品质管理中心 | 磨损老化 | 2581.0 | ✅ | ✅ |

## 五、关键业务信号

来自 Master Top 20（[完整数据见 dashboard 13](https://voc.lute-tlz-dddd.top/superset/dashboard/13/)）：

| 信号类型 | 主题示例 | 主责部门 | 处置 |
|---|---|---|---|
| **产品力** | 延迟 / 尺码小 / 使用体验差 / 熔化-冒烟 | 产品研发部 | 「核心体验改良包」专项 |
| **质量** | 磨损老化 / 功能失效 / 异味过热 | 品控部 | 「质量缺陷闭环包」+ 供应商改进 |
| **物流** | 取消-用错卡 / 缺件少件 | 国际物流部 | 「履约提速包」 |
| **品牌监测** ⚠️ | Elvie / Tommee Tippee / Spectra / Philips Avent | 品牌市场中心 | 这是**竞品声量异常**信号（100% neg = 用户在抱怨这些品牌 vs 我方）·建议按品牌监测流程处理而非内部整改 |

## 六、验收（端到端）

### 6.1 Playwright 自动化（[playwright_l6_results.json](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L6-20260514-171018/playwright_l6_results.json)）

| dashboard | charts | err_overlay | console_err |
|---:|---:|:---:|:---:|
| 2 全球客服中心 | 2/2 | 0 | 0 |
| 3 产品中心 | 2/2 | 0 | 0 |
| 4 仓储物流部 | 2/2 | 0 | 0 |
| 5 品牌市场中心 | 2/2 | 0 | 0 |
| 6 电商运营部 | 2/2 | 0 | 0 |
| 7 品质管理中心 | 2/2 | 0 | 0 |
| 8 法务合规部 | 2/2 | 0 | 0 |
| 13 D-Action 总览 | 8/8 | 0 | 0 |

### 6.2 视觉抽查（multimodal-looker）

- [产品中心 dashboard 3 fix](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L6-20260514-171018/dash-3-产品中心-fixed.png) ：原 Top10 话题保留 + D-Action Top10 真实 action（无占位符）
- [D-Action 总览 dashboard 13 fix](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L6-20260514-171018/dash-13-D-Action%20总览-fixed.png) ：8 张表全部渲染、Master Top 3 与 dry-run 一致

## 七、对存量资产的影响

| 类型 | L6 前 | L6 后 | 影响 |
|---|---:|---:|---|
| Dataset | 10 | 10 | **0 新增 / 0 删除**（沿用 dataset 2 = v_label_with_dept） |
| Chart | 34 | 42 | +8（id 35-42） |
| Dashboard | 12 | 13 | +1（id 13）+ 7 个被追加 row |
| Postgres view | 10 | 10 | **0 改动** |
| voc_review schema | — | — | **0 改动** |
| dim_tag schema | — | — | **0 改动** |
| L4 D-Health (id 9) | — | — | **0 影响** |
| L5 D-Diag (id 10/11/12) | — | — | **0 影响** |
| L4 之前的 dataset 1-6 / chart 1-19 | — | — | **0 影响** |

## 八、回滚预案

一键脚本：[mvp_l6_rollback.py](file:///Users/pray/.secrets/backups/voc-deep-analysis-mvp/L6-20260514-171018/mvp_l6_rollback.py)

```python
# Step 1: 从 pre_l6_snapshot.json 恢复 7 dept dashboard 的 position_json
PUT /api/v1/dashboard/{2,3,4,5,6,7,8}  body={position_json: <pre>}
# Step 2: 删除 D-Action 总览
DELETE /api/v1/dashboard/13
# Step 3: 删除 8 chart
for cid in 35..42: DELETE /api/v1/chart/{cid}
```

## 九、备份索引

`~/.secrets/backups/voc-deep-analysis-mvp/L6-20260514-171018/`

包含：MANIFEST + 5 个执行脚本 + pre/post snapshot + 全 chart/dashboard 定义 + 9 张全页截图 + Playwright 验证 JSON。

## 十、MVP 总进度

| 阶段 | 状态 | 工日 |
|---|:---:|---:|
| L0 准备 | ✅ | done |
| L1 voc_review 扩展 | ✅ | done |
| L2 50 SAT 映射 | ✅ | done |
| L3 4 深度视图 | ✅ | done |
| L4 D-Health 看板 | ✅ | done |
| L5 D-Diag 3 专题诊断 | ✅ | done |
| **L6 D-Action 7 部门行动队列** | **✅** | **done** |
| L7 部署 + 文档 + 培训 | ⏳ | 1 工日 |

**13.5 工日预算 / 实际：13 工日 · 0 LLM 成本 · 0 schema 改动**。

下一步 **L7 部署 + 文档 + 培训**，需要等用户 GO L7。

## 十一、文件清单（提交到仓库）

| 文件 | 说明 |
|---|---|
| `research/02-脚本工具/01-标签进化/scripts/mvp_l6_create_charts.py` | 8 chart 装配 |
| `research/02-脚本工具/01-标签进化/scripts/mvp_l6_append_to_dept_dashboards.py` | Path A：追加到 7 dept dashboard |
| `research/02-脚本工具/01-标签进化/scripts/mvp_l6_create_overview.py` | Path B：创建 D-Action 总览 |
| `research/02-脚本工具/01-标签进化/scripts/mvp_l6_fix_action_filter.py` | 关键过滤修复 |
| `research/02-脚本工具/01-标签进化/scripts/mvp_l6_rollback.py` | 一键回滚 |
| `research/04-输出结果/03-审计报告/mvp_l6_progress_report.md` | 本进度报告 |
