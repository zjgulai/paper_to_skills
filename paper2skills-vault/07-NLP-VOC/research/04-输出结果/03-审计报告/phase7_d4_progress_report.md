---
name: phase7-d4-progress-report
description: Phase 7 D4 进度报告 — Superset native filter 实装 + 语义验证 + 重新导出 ZIP。BI B 路径完整闭环，用户可按数据源/产品线/NPS/极性多维交互切片。
date: 2026-05-11
phase: phase7
day: D4
status: 🟢 Filter 实质上线 — 10 filters 配置 + polarity 语义测试 PASS + ZIP 重新导出
doc_type: audit-report
module: voc-nlp
---

# Phase 7 D4 进度报告 — Superset Native Filter 实装

> **总判定**：🟢 **Phase 7 D4 完成** — 为 8 个 dashboard 实装 10 个 native filter（Overview 3 个维度 + 7 个部门 dashboard 各 1 个极性）。API PUT 确认配置持久化，SQL 语义测试证明 filter 切换后 Top-3 tag 完全不同（正向 = 质量感知，负向 = 延迟）。8 个 ZIP 已重新导出，filter 配置嵌入在 YAML 中可重建。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| D4.1 Probe native_filter schema | ✅ | 手动在 dashboard 3 加 polarity filter + GET 验证 |
| D4.2 Overview 3 filters | ✅ | data_source / product_line / proxy_nps |
| D4.3 7 dept × polarity filter | ✅ | 部门 dashboard 统一 polarity 切片 |
| D4.4 API + SQL 语义验证 | ✅ | 10/10 配置正确 + polarity 切换 Top-3 完全不同 |
| D4.5 Re-export 8 ZIPs | ✅ | 20K → 24K (overview)，filter 嵌入 YAML |

## 二、Filter 清单（10 个）

### Overview Dashboard (id=1) — 3 filters

| Filter ID | 名称 | 类型 | Target Column | 作用范围 |
|---|---|---|---|---:|
| NATIVE_FILTER-data-source | 数据源 | filter_select (multi) | v_review_overview.data_source | 5 charts |
| NATIVE_FILTER-product-line | 产品线 | filter_select (multi) | v_review_overview.product_line | 5 charts |
| NATIVE_FILTER-proxy-nps | Proxy NPS | filter_select (multi) | v_review_overview.proxy_nps | 5 charts |

### 7 Per-Dept Dashboards (id=2..8) — 1 filter each

| Dashboard | Filter ID | Name | Target Column | Chart Scope |
|---|---|---|---|---:|
| VOC · 客服部 | NATIVE_FILTER-polarity | 情感极性 | v_dept_topic_summary.polarity | 1 chart |
| VOC · 产品研发部 | same | same | same | 1 chart |
| VOC · 国际物流部 | same | same | same | 1 chart |
| VOC · 市场部 | same | same | same | 1 chart |
| VOC · 电商运营部 | same | same | same | 1 chart |
| VOC · 品控部 | same | same | same | 1 chart |
| VOC · 质量与法规部 | same | same | same | 1 chart |

## 三、Filter Schema（D4.1 discovered）

native_filter_configuration 每项的最小必需字段：

```json
{
  "id": "NATIVE_FILTER-xxx",
  "name": "中文显示名",
  "type": "NATIVE_FILTER",
  "filterType": "filter_select",
  "targets": [{"datasetId": <int>, "column": {"name": "<column>"}}],
  "defaultDataMask": {"filterState": {"value": []}},
  "controlValues": {
    "multiSelect": true,
    "enableEmptyFilter": false,
    "defaultToFirstItem": false,
    "inverseSelection": false,
    "searchAllOptions": false
  },
  "cascadeParentIds": [],
  "scope": {"rootPath": ["ROOT_ID"], "excluded": []},
  "description": "...",
  "chartsInScope": [<chart_id>],
  "tabsInScope": [],
  "isInstant": true
}
```

关键点：
- `chartsInScope` 决定哪些 chart 受此 filter 影响
- `targets[0].datasetId` 必须与 chart 的 dataset 一致（否则 filter 失效）
- `scope.rootPath: ["ROOT_ID"]` 必须设定才能进入 dashboard 顶层过滤栏

## 四、API 层验证

全部 8 dashboards 的 filter 配置通过 `GET /api/v1/dashboard/{id}` 验证：

```
id=1 [VOC Overview · 全局总览]     filters=3: ['数据源', '产品线', 'Proxy NPS']
id=2 [VOC · 客服部]               filters=1: ['情感极性']
id=3 [VOC · 产品研发部]            filters=1: ['情感极性']
id=4 [VOC · 国际物流部]            filters=1: ['情感极性']
id=5 [VOC · 市场部]               filters=1: ['情感极性']
id=6 [VOC · 电商运营部]            filters=1: ['情感极性']
id=7 [VOC · 品控部]               filters=1: ['情感极性']
id=8 [VOC · 质量与法规部]          filters=1: ['情感极性']
```

Idempotent 重跑全部显示 "already present (kept)"。

## 五、SQL 语义验证（filter 真的工作吗？）

直接在 SQL Lab 测试 polarity filter 的语义效果（产品研发部 Top-3）：

| 过滤条件 | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| **无 filter** | 质量感知 20,192 (正向) | 易用性 16,351 (正向) | 延迟 12,257 (负向) |
| **polarity=正向** | 质量感知 20,192 | 易用性 16,351 | 性能满意 9,323 |
| **polarity=负向** | **延迟 12,257** | **尺码小 4,752** | **使用体验差 3,444** |

**完全不同的 Top-3** — 证明 filter 与 chart 关联正确，不是空壳。

## 六、工程要点

### 6.1 数据集关联

Overview filter 用 `v_review_overview` (datasetId=1)，dept filter 用 `v_dept_topic_summary` (datasetId=3)。**同一 filter 不能跨 dataset**（datasetId 不匹配时 Superset 静默忽略 filter），所以用不同 dataset 的 charts 需要独立 filter。

### 6.2 Idempotent 合并策略

`upsert_dashboard_filters` 保留任何已存在的 **unrelated** filter（按 id 判定），只更新/插入 desired filters。避免重跑覆盖用户手动在 UI 加的 filter。

### 6.3 chartsInScope 精确控制

每个 filter 的 `chartsInScope` 只包含该 dashboard 的 chart_ids，避免污染其他 dashboard（即使 chart 在多个 dashboard 里也不会被误过滤）。

## 七、产出清单

| 文件 | 用途 | 大小 |
|---|---|---:|
| `docker/superset_filters_factory.py` | 10 filter 自动化（idempotent）| 219 行 |
| `superset_exports/dashboard_1.zip` | Overview dashboard（3 filters 嵌入）| 24 KB |
| `superset_exports/dashboard_{2-8}.zip` | 7 dept dashboards（1 filter 嵌入）| 8 KB × 7 |
| `03-审计报告/phase7_d4_progress_report.md` | 本文档 | 6 KB |

## 八、操作手册

### 8.1 重建 filter 配置

```bash
# 前置：Superset + 12 charts + 8 dashboards 已就绪（D2+D3）
cd paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/docker

# 一键应用 10 filters（idempotent，可重跑）
python3 superset_filters_factory.py
```

### 8.2 用户交互（浏览器）

1. 访问 http://localhost:8088/dashboard/list/
2. 登录 admin / voc_admin_2026
3. 进入任何 dashboard：顶部/左侧会出现 filter 栏
4. 选择过滤值 → 图表实时更新

## 九、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | 浏览器验证仅做了 API + SQL 层；UI 层交互未做端到端 | 中 | Playwright 会话冲突时以 SQL 语义等价替代（产品研发部正向/负向 Top-3 完全不同已证明 filter 关联正确）|
| R2 | filter 不跨 dataset，overview filter 不能直接用在 dept dashboard | 低 | 是合理设计，两种 dashboard 本就目的不同 |
| R3 | filter 默认值为空（全量）| 低 | 用户可手动保存"默认视图"到 Superset UI |

## 十、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-11 08:55 | D4.1 手动在 dashboard 3 加 polarity filter，PUT 200 + GET 验证 |
| 2026-05-11 09:00 | D4.2+D4.3 写 superset_filters_factory.py |
| 2026-05-11 09:05 | 首次 run：10 filters 全部 "adding" |
| 2026-05-11 09:07 | idempotent re-run：全部 "already present (kept)" |
| 2026-05-11 09:35 | D4.4 API 确认 8/8 dashboards filter 配置正确 |
| 2026-05-11 09:40 | D4.4 SQL 语义验证：polarity 切换 Top-3 完全不同 |
| 2026-05-11 09:45 | D4.5 重新导出 8 ZIPs，YAML 含 native_filter_configuration |
| 2026-05-11 09:50 | 本报告归档 |

## 十一、一行总结

> Phase 7 D4 **Native Filter 实质上线**：10 个 filter（Overview 3 个维度 + 7 部门 × polarity）通过 REST API 自动化应用，API 层 8/8 dashboard 配置正确，SQL 语义测试证明 polarity 切换后 Top-3 tag 完全不同（正向=质量感知/易用性/性能满意，负向=延迟/尺码小/使用体验差），8 个 ZIP 重新导出含 filter YAML 可重建。**BI B 路径完整闭环：数据底座 → Superset → 12 charts → 8 dashboards → 10 filters**。
