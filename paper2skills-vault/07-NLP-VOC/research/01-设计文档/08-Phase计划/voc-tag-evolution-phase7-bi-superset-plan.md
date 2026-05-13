---
name: voc-tag-evolution-phase7-bi-superset-plan
description: Phase 7 事后追溯计划文档（4 天，2026-05-08 → 05-11）。从 Phase 6 D10 C 路径 HTML 看板，到 Phase 7 D4 完整 Superset B 路径上线（12 charts + 8 dashboards + 10 native filters）。本文档基于 D1-D4 实际执行结果反向梳理出原本应有的计划，作为外部审计材料和 Phase 8+ 计划写作的模板。
title: VOC 标签体系 Phase 7 BI Superset B 路径计划（事后追溯版）
doc_type: plan
module: voc-nlp
topic: phase7-bi-superset-plan
status: stable
created: 2026-05-11
updated: 2026-05-11
owner: self
source: ai
phase: phase7
audience: pm-and-engineer
---

# Phase 7 BI Superset B 路径计划（事后追溯版）

> **文档定位**：Phase 7（D1-D4）当时也是边走边迭代的。这份"事后追溯计划"基于 [phase7_d{1-4}_progress_report.md](../../04-输出结果/03-审计报告/) 反向梳理出原本应有的计划。
>
> **真实执行复盘**：[phase7_complete_retrospective.md](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md)

---

## 一、Phase 6 末态 + Phase 7 触发点

### 1.1 Phase 6 D10 收口时的状态

| 维度 | 值 |
|---|---|
| 字典 | v4.1（全字段质量收敛） |
| Precision（口径 B） | 0.896 |
| Week 2 Gate | 7/7 PASS |
| BI 看板 | ✅ C 路径（HTML）+ A 路径（14 周报） |
| BI 看板 B 路径 | ❌ 未做（Superset） |
| 业务方反馈 | "HTML 看板 OK，但希望能交互过滤" |

### 1.2 Phase 7 启动的 2 个动机

| # | 动机 | 优先级 |
|---|---|---|
| 1 | 业务方希望能按部门 / 极性 / 数据源切片，而不是只看快照 | P0 |
| 2 | 数据每月更新一次，HTML 重生成成本高（手动 + 邮件分发） | P1 |

---

## 二、Phase 7 目标（4 天）

### 2.1 量化目标

| 指标 | Phase 6 末态 | Phase 7 目标 | Phase 7 末态（实际） |
|---|---|---|---|
| BI 实时交互能力 | 无 | 至少 1 个全局 + 7 部门 dashboard | ✅ 1 + 7 = 8 dashboards |
| 实时切片维度 | 无 | 至少 polarity + data_source | ✅ 10 filters（含 polarity / data_source / product_line / proxy_nps） |
| 数据迁移可重建性 | 无 | < 30 分钟 from scratch | ✅ ~10 分钟（docker compose + 3 工厂脚本） |
| LLM 增量成本 | / | $0（纯工程） | ✅ $0 |
| 累计开发时间 | / | < 16 小时 | ✅ ~7 小时 |

### 2.2 范围划定

**做**：
- voc_bi 数据库 + ETL（替代 jsonl 直读）
- Superset Docker 部署 + REST API 工厂模式
- 12 charts + 8 dashboards + 10 native filters
- 8 个 dashboard ZIP 入仓（迁移可重建）

**不做**（推迟到 Phase 8+）：
- Superset 多用户 RBAC + SSO
- HTTPS + 公司域名部署
- 飞书 / 钉钉嵌入
- 时间维度过滤（v_review_overview 暂无 timestamp 列）

### 2.3 技术选型决策

| 选项 | 选择 | 理由 |
|---|---|---|
| BI 工具 | **Apache Superset 4.1.1** | 开源 + REST API 完备 + ECharts 渲染 |
| 部署方式 | **Docker Compose** | 隔离 + 一键起 + 可迁移 |
| 数据库 | **Postgres** | Superset 推荐 + 性能稳定 |
| 配置管理 | **REST API 工厂模式** | 不在 UI 手改，所有配置入 git |
| 缓存 | **Redis** | Superset 推荐 + 提升查询速度 |

---

## 三、4 天日计划（事后追溯）

### Week 3 — D1-D4：BI B 路径完整闭环

| Day | 主题 | 交付物 | 累计耗时 |
|---|---|---|---|
| **D1** | 数据底座 | voc_bi 数据库 + 4 张基表 + 6 SQL 视图 + ETL（37s 导入 364K） | 1.5h |
| **D2** | Superset 部署 | Docker Compose + voc_bi 接入 + 6 datasets 自动注册 | 2h |
| **D3** | Charts + Dashboards | 12 charts + 8 dashboards REST API 自动化 + Playwright 验证 + 8 ZIP 导出 | 2h |
| **D4** | Native Filters | 10 filters（Overview 3 + Dept polarity × 7）+ 端到端验证 + D4 修订 | 1.5h |

---

## 四、关键决策点

### 4.1 D1 数据底座选型：为什么不直接读 jsonl？

**选项**：
- A：Superset 直读 jsonl（用 Pandas dataframe）
- B：ETL 到 SQLite
- C：ETL 到 Postgres ✅

**选择 C 的理由**：
- jsonl 直读需要每次查询全表扫描，性能差（5GB+）
- SQLite 单文件简单，但生产环境扩展性受限
- Postgres + 6 SQL 视图 = 标准星型 schema，下游加 chart 不需要写新代码

### 4.2 D2 Superset 部署：Docker 还是源码？

**选择**：Docker（apache/superset:4.1.1）。

**理由**：
- 官方镜像 = 零配置
- Compose 起 3 个服务（superset + redis + postgres 接入）= 5 分钟
- 升级方便（改镜像 tag 即可）

### 4.3 D3 Chart/Dashboard 创建：UI 还是 API？

**选择**：REST API 工厂模式。

**理由**：
- UI 改不入 git，迁移环境会丢
- 12 个 chart 手工建 = 易出错 + 无法 review
- factory 脚本 = 配置即代码

**关键工程踩坑（D3）**：
- 首次实现只 PUT `position_json`，dashboard 加载后显示 "没有与此组件关联的图表定义"
- **根因**：Superset 的 `dashboard.charts` 是 canonical ownership link，`position_json` 只是 layout 引用
- **修复**：2-call protocol：
  1. PUT `/api/v1/chart/{id}` with `{"dashboards": [dashboard_id]}` — 建立所有权
  2. PUT `/api/v1/dashboard/{id}` with `position_json` — 设置布局

### 4.4 D4 Filter 设计：Overview filters 应该作用在哪些 chart 上？

**初始设计**：filter scope = `[1, 2, 3, 4, 5]`（所有 Overview chart）。

**实测问题**：
- chart 1, 2 用 v_dept_kpi（无 data_source 列）
- chart 3 用 v_global_top_tags（无 data_source 列）
- 只有 chart 4, 5 用 v_review_overview，filter 才对它们生效
- Superset **静默忽略**不匹配的 chart——UI 上看不出问题但 filter 实际无效

**修复**：
- chartsInScope 改为 `[4, 5]`（仅饼图）
- 同时修复 chart 4, 5 的 metric 用 `COUNT(*)` SQL 形式（v_review_overview 不暴露 review_id）
- factory 脚本改为按 `slice_name` 查找 chart_id（避免 delete+recreate 后硬编码失效）

### 4.5 D4 暴露的 Superset 4.1.1 饼图 dashboard-mode 渲染 bug

**症状**：
- chart 13/14（修复后的饼图）在 `/explore/?slice_id=13` 单图模式下正常渲染
- 在 `/dashboard/1/` 中 canvas 是透明的（所有像素 RGBA=0,0,0,0）
- 后端 `/api/v1/chart/data` 返回正确数据

**根因**：Superset 4.1.1 前端 ECharts 实例化在 dashboard 模式 + pie + SQL-form metric 组合下的 bug。

**当前处置**：
- 不阻塞业务（部门 dashboard 全部正常）
- 详细记录在 [phase7_d4_progress_report.md §五](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md)
- 修复推到 Phase 8 候选

---

## 五、量化交付与验证

### 5.1 D1 ETL 性能

| 阶段 | 时间 | 备注 |
|---|---|---|
| 解析 jsonl | ~5s | streaming + Pydantic |
| 字典 enrich | ~3s | dim_tag in-memory lookup |
| Postgres COPY | ~25s | execute_values batch 1000 |
| **总计** | **~37s** | 364,569 reviews + 689K labels |

### 5.2 D3 Charts × Dashboards 矩阵

详见 [phase7-architecture-diagrams.md §5](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md#图-5--12-charts--8-dashboards-映射)。

| 类型 | 数量 |
|---|---|
| Charts | 12（5 Overview + 7 Dept Top-10） |
| Dashboards | 8（1 Overview + 7 Dept） |
| ZIP exports | 8（入仓） |

### 5.3 D4 Filters

| 类型 | 数量 | 配置 |
|---|---|---|
| Overview filters | 3 | data_source / product_line / proxy_nps |
| Dept polarity filter | 7 | 每个部门 dashboard 一个 |
| **总计** | **10** | |

### 5.4 端到端验证（Playwright 实测）

详见 [phase7_d4_progress_report.md §四](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md)。

**产品中心/品线 dashboard polarity=负向 测试**：
- 未过滤：质量感知 20.2k → 易用性 16.4k → 延迟 12.3k
- 过滤后：延迟 12.3k → 尺码小 4.75k → 使用体验差 3.44k
- "Applied filters (1)" badge 显示
- X 轴自动从 0-20k 缩放到 0-12k

---

## 六、Phase 7 教训与下一步

### 6.1 教训

1. **Filter 的 dataset 必须和 chart 的 dataset 匹配**：Superset 不会报错，会静默忽略
2. **Factory 脚本不能硬编码 chart_id**：用 chart_name 查找更鲁棒
3. **饼图 metric 用 SQL form 而不是 SIMPLE form**：避免列依赖问题，但要注意 Superset 4.1.1 的 dashboard-mode 渲染 bug
4. **Playwright 端到端测试是黄金标准**：API 返回 200 不代表前端渲染正确

### 6.2 Phase 8 候选（待立项）

| 优先级 | 议题 | 工作量估算 |
|---|---|---|
| P0 | 修复 Superset 饼图 dashboard-mode 渲染 bug | 2-3 天（重做为 bar 或升级 Superset） |
| P0 | v_review_overview 加时间戳列 + 时间维度过滤 | 1 天 |
| P1 | Superset 多用户 RBAC + SSO 接入 | 3-5 天 |
| P1 | Nginx + HTTPS + 公司域名 | 1-2 天 |
| P2 | 飞书 / 钉钉工作台嵌入 | 2-3 天 |
| P2 | 月度自动化 cron（ETL + Superset 缓存清理） | 1 天 |
| P3 | 字典 v5.0 启动（如月度演进累积足够） | 5-10 天 |

---

## 七、关联文档

| 类型 | 链接 |
|---|---|
| Phase 7 完整复盘 | [phase7_complete_retrospective.md](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) |
| 白话汇报 | [phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md) |
| 架构图集（Markdown） | [phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md) |
| 架构图集（HTML 高保真） | [phase7-architecture-diagrams.html](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.html) |
| 每日进度报告 | [phase7_d{1-4}_progress_report.md](../../04-输出结果/03-审计报告/) |
| Superset 运维 SOP | [Superset_BI_SOP.md](../07-操作指南/Superset_BI_SOP.md) |
| ETL SOP | [ETL_pipeline_SOP.md](../07-操作指南/ETL_pipeline_SOP.md) |
| Phase 6 计划 | [voc-tag-evolution-phase6-dictionary-quality-plan.md](voc-tag-evolution-phase6-dictionary-quality-plan.md) |
| Phase 5 计划 | [voc-tag-evolution-phase5-product-closed-loop-plan.md](voc-tag-evolution-phase5-product-closed-loop-plan.md) |

---

> **本文档定位**：事后追溯计划，补齐文档完整性。Phase 7 实际是 D1-D4 滚动开发的，没有提前写完整 PRD。
