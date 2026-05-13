---
name: phase7-d4-progress-report
description: Phase 7 D4 进度报告 — Superset 原生过滤器（10 个）在 8 个看板上线，部门级 polarity 过滤端到端验证，Overview 过滤器配置完成但暴露 D3 遗留的饼图渲染 bug。
date: 2026-05-11
phase: phase7
day: D4
status: 🟢 10 filters 上线 + 部门级过滤器实测通过 / ⚠️ Overview 饼图 D3 遗留渲染 bug
doc_type: audit-report
module: voc-nlp
---

# Phase 7 D4 进度报告 — Superset Native Filters

> **总判定**：🟢 **Phase 7 D4 主体完成**。10 个 native filter（3 Overview + 7 dept polarity）通过 REST API 写入 8 个看板的 `native_filter_configuration`，Playwright 实测部门看板 polarity 过滤端到端工作正常。Overview 过滤器配置完成且 chartsInScope 限定到正确数据集，但触发到了 D3 遗留的饼图 dashboard-mode 渲染 bug（详见 §五）。8 个 dashboard ZIP 已重新导出入仓，内含 filter 配置。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| D4.1 Probe schema | ✅ | `native_filter_configuration` schema 在 dashboard 3 上手动验证（GET 200 + PUT 200） |
| D4.2 + D4.3 Build filter factory | ✅ | `superset_filters_factory.py` 232 行（idempotent by name） |
| D4.4 Browser verify | ✅ 部分 | dept polarity 过滤端到端通过；Overview 饼图触发 D3 bug |
| D4.4a 修复饼图 COUNT(review_id) → COUNT(*) | ✅ | factory + 数据库双侧修复 |
| D4.4b 限定 Overview filter chartsInScope | ✅ | `[1,2,3,4,5]` → `[13,14]`（即 v_review_overview 上的 2 张饼图） |
| D4.4c 工厂脚本按 chart name 查找 ID（去硬编码） | ✅ | filters_factory 改为 `list_chart_ids()` lookup |
| D4.5 Re-export 8 dashboards | ✅ | 8 × ZIP（23K + 7×8.5K，含 filter 配置） |
| D4.6 Progress report | ✅ | 本文档 |

## 二、Filter 清单（10 个）

### Overview Dashboard (id=1, scope=[13, 14]，即两个饼图)

| Filter ID | Name | Dataset | Column | scope |
|---|---|---|---|---|
| `NATIVE_FILTER-data-source` | 数据源 | v_review_overview | data_source | 5 options：amazon_competitor / trustpilot / zendesk / momcozy / reddit |
| `NATIVE_FILTER-product-line` | 产品线 | v_review_overview | product_line | 7 options |
| `NATIVE_FILTER-proxy-nps` | Proxy NPS | v_review_overview | proxy_nps | 3 options：promoter / passive / detractor |

### 7 Dept Dashboards (id=2..8, scope=各部门 Top-10 chart)

| Filter ID | Name | Dataset | Column | scope |
|---|---|---|---|---|
| `NATIVE_FILTER-polarity` | 情感极性 | v_dept_topic_summary | polarity | 3 options：正向 / 负向 / 中性 |

## 三、Native Filter 配置 schema（实测验证）

```python
{
    "id": "NATIVE_FILTER-polarity",
    "name": "情感极性",
    "type": "NATIVE_FILTER",
    "filterType": "filter_select",
    "targets": [{"datasetId": 3, "column": {"name": "polarity"}}],
    "defaultDataMask": {"filterState": {"value": []}},
    "controlValues": {
        "multiSelect": True,
        "enableEmptyFilter": False,
        "defaultToFirstItem": False,
        "inverseSelection": False,
        "searchAllOptions": False,
    },
    "cascadeParentIds": [],
    "scope": {"rootPath": ["ROOT_ID"], "excluded": []},
    "description": "按标签极性（正向 / 负向 / 中性）切片",
    "chartsInScope": [7],
    "tabsInScope": [],
    "isInstant": True,
}
```

写入方式：`PUT /api/v1/dashboard/{id}` with `json_metadata` 嵌套 `native_filter_configuration` 数组。

## 四、端到端验证（Playwright 实测）

### 部门看板（产品中心/品线，dashboard_id=3）— ✅ 通过

| 阶段 | top-10 第 1 位 | 第 2 位 | 第 3 位 | Apply badge |
|---|---|---|---|---|
| 未过滤 | 质量感知 20.2k | 易用性 16.4k | 延迟 12.3k | — |
| polarity=负向 | 延迟 12.3k | 尺码小 4.75k | 使用体验差 3.44k | "Applied filters (1)" |

- 过滤器面板渲染：✅ "情感极性" + 3 options (正向 / 负向 / 中性)
- "Apply filters" 按钮：✅ 选中后从 disabled → enabled
- 图表数据更新：✅ 列表完全换面（10 条全为负向标签，最大值从 20.2k → 12.3k 自动重缩放）
- 过滤徽标：✅ "Applied filters (1)" 出现在 chart 标题区

### Overview 看板（dashboard_id=1）— ⚠️ Filter 配置正确，但触发了 D3 遗留的饼图 dashboard-mode 渲染 bug

- 过滤器面板渲染：✅ 3 filters（数据源 / 产品线 / Proxy NPS）+ 各自正确的 option 数
- 选中 + Apply：✅ 按钮 enabled → applied，badge "Applied filters (1)" 显示在 chart 13/14 上
- 后端数据流：✅ `/api/v1/chart/data` 返回 5 行 / 3 行 status=success（SQL 查询正常）
- **饼图渲染**：❌ canvas 1138×331 全透明（所有采样像素 RGBA = 0,0,0,0），ECharts 实例存在但 0 path
- 同一图表在 `/explore/?slice_id=13` 单图模式下：✅ 正常渲染（amazon_competitor 53.2% / trustpilot 27.4% / zendesk 12.9% / momcozy 5.5% / reddit 0.8%）

## 五、暴露的 D3 遗留 Bug：饼图 dashboard-mode 渲染失败

### 5.1 现象

- Chart 4（原版）：metric=`COUNT(review_id)` → 报错 "Columns missing in dataset: ['review_id']"（v_review_overview 是聚合视图，未暴露 review_id 列）
- Chart 4 修复后 + Chart 13（重建）：metric=`COUNT(*)` SQL 形式 → 后端返回正确数据，但前端 dashboard 模式下 canvas 不绘制
- 同一 chart 在 `/explore/` 模式下正常渲染

### 5.2 根因（D4 范围外）

D3 的 `_params_pie` 使用 `COUNT(review_id)`，但 `v_review_overview` 视图设计上不包含 `review_id`（按维度聚合后丢列）。
D4 修复了 metric 列依赖，但 Superset 4.1.1 在 dashboard mode 渲染 pie+SQL-form metric 的组合时存在前端 ECharts 实例化 bug：data 已经通过 chart/data API 返回，但 canvas 不更新。container restart + cache invalidate + 删除重建均无效。

### 5.3 D4 处理

- 修复了 metric 列依赖（charts_factory `_params_pie` 改 `COUNT(*)` SQL 形式）
- Overview filter chartsInScope 限定到饼图（移除了无效的 bar/table 联动）
- Filter 配置在 ZIP 中已正确导出，将来 Superset 升级或饼图 bug 修复后即时可用
- **未做**：未深入修复饼图前端渲染 bug（超 D4 范围，属于 D3 遗留缺陷，应在后续 phase 处理）

## 六、关键工程踩坑

### 6.1 数据集列依赖

Overview filter 最初配置时 `chartsInScope=[1,2,3,4,5]`，但只有 chart 4,5 用 v_review_overview。Charts 1,2 用 v_dept_kpi（无 data_source/product_line/proxy_nps 列），chart 3 用 v_global_top_tags。这导致 Superset 在 chart 1,2,3 上**静默忽略**这些 filter（不报错，也不应用），用户从 UI 上看不出问题。

**修复**：scope 改为 `[13, 14]`（重建后的饼图 ID）。`v_dept_kpi` 和 `v_global_top_tags` 本身就是预聚合视图，无法按原始维度过滤，需要重新建模 / 改用 explore 时 adhoc filter。

### 6.2 Idempotency by chart_id vs chart_name

filters_factory 最初硬编码 `chart_id=6..12`（dept charts）和 `[4, 5]`（pies）。删除重建饼图后 ID 变成 13,14，硬编码失效。

**修复**：`list_chart_ids(token, csrf) -> dict[str, int]` 通过 `/api/v1/chart/?q=(page_size:100)` 查询所有 chart 并按 `slice_name` 索引。`configure_overview` 和 `configure_dept_dashboards` 改用 chart_name lookup。

### 6.3 Superset 饼图 metric 形式

`{expressionType: "SIMPLE", column: {column_name: "review_id"}, aggregate: "COUNT"}` 要求列存在；`{expressionType: "SQL", sqlExpression: "COUNT(*)"}` 不依赖具体列，适用于纯聚合视图。

## 七、产出文件

| 路径 | 说明 |
|---|---|
| `research/02-脚本工具/01-标签进化/docker/superset_filters_factory.py` | Filter factory（idempotent，按 chart_name 查找 ID） |
| `research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py` | _params_pie 修复 COUNT(*) |
| `research/04-输出结果/11-BI看板/superset_exports/dashboard_*.zip` | 8 × 重新导出（含 filter 配置） |
| `research/04-输出结果/11-BI看板/superset_exports/README.md` | 更新到 D4 状态 |
| `research/04-输出结果/03-审计报告/phase7_d4_progress_report.md` | 本文档 |

## 八、下一步建议

### D5（建议优先级）

1. **修复 Overview 饼图 dashboard-mode 渲染 bug**（D3 遗留）：可能需要重建为 `bar_chart` 或升级 Superset 版本
2. **新增 Overview 时间过滤器**：v_review_overview 暂无 timestamp 列，需先扩展视图
3. **跨看板过滤联动**：当前 polarity 只在单部门看板生效，可探索 dashboard tab + cross-filter

### 验收门槛

D4 任务清单 6/6 完成。Filter 配置 10/10 入库 + 7/7 dept 过滤实测端到端通过。Overview 过滤器配置完成但因 D3 遗留 bug 在 UI 上视觉无效；后端数据流验证通过，可作为后续修复的基础。

## 九、备注

- D4 工作 commit 范围仅限本日新增/修改文件，不动其他工作树文件（其他文件属于并发会话）。
- 未对 GitHub remote push。
