---
name: nlp-voc-milestone-audit-map
description: NLP-VOC项目重点里程碑审计与交叉自证底稿，涵盖数据资产、标签字典、AI打标、质量修复、HTML看板、Superset BI与生产化准备。当制作管理层汇报、项目进度汇报或HTML Presentation证据链时使用。
title: NLP-VOC 重点里程碑审计与交叉自证底稿
doc_type: audit-map
module: voc-nlp
topic: milestone-audit-cross-validation
status: draft
created: 2026-05-12
updated: 2026-05-12
owner: self
source: ai
audience: executive, business, pm, engineer
---

# NLP-VOC 重点里程碑审计与交叉自证底稿

> **文档定位**：本文件是 NLP-VOC 项目管理层汇报与 HTML Presentation 的“证据底稿”。它不替代各 Phase 复盘、计划、SOP 和审计报告，而是把重点里程碑统一整理为“目标 → 交付物 → 验证结果 → 主证据 → 交叉证据 → 遗留风险”的审计链路。
>
> **核心原则**：凡是进入正式汇报的关键结论，至少需要 2 类以上证据交叉支撑；若只有单一来源，必须标记为“需要补充确认”。

---

## 一、审计方法

### 1.1 审计五要素

| 要素 | 说明 | 汇报中回答的问题 |
|---|---|---|
| 目标 | 该阶段原本要解决什么问题 | 为什么做这一阶段 |
| 交付物 | 实际产出了什么文件、数据、系统或报告 | 是否真的完成 |
| 验证结果 | 用什么指标、测试或人工审查证明有效 | 是否可信 |
| 交叉证据 | 是否存在计划、复盘、进度报告、SOP、真实输出物等多来源互证 | 是否只是单文档自说自话 |
| 遗留风险 | 当前仍未解决或需要下一阶段处理的问题 | 是否可以放心进入下一阶段 |

### 1.2 证据分级

| 证据级别 | 类型 | 可信度 | 示例 |
|---|---|---|---|
| A | 真实输出物 / 可运行系统 / 数据文件 / 浏览器验证 | 最高 | Superset dashboard、ETL 结果、周报产物 |
| B | 审计报告 / Gate 报告 / 进度报告 | 高 | Phase 6 D9 Gate、Phase 7 D4 Progress |
| C | 计划文档 / 架构文档 / SOP | 中高 | Phase 5 计划、ETL SOP、Superset SOP |
| D | 汇报素材 / executive brief | 中 | Phase 5 白话汇报、Phase 6+7 白话汇报 |

### 1.3 汇报使用原则

1. **对管理层讲结论，但用证据兜底**：主页面保留结果与价值，附录保留证据链。
2. **对业务方讲使用场景，但保留验证状态**：说明看板怎么用，同时标注哪些功能已验证、哪些仍是后续项。
3. **对技术团队讲可复现性**：必须能追溯到脚本、SOP、数据路径和重建流程。
4. **对风险保持诚实**：如 Overview 饼图 bug、生产化部署、权限、时间维度过滤等，不应弱化。

---

## 二、重点里程碑总览

| # | 里程碑 | 阶段 | 当前状态 | 可信度 | 汇报角色 |
|---:|---|---|---|---|---|
| M1 | VOC 数据资产盘点与统一口径 | Phase 0-4 | ✅ 已完成 | 高 | 证明项目有真实数据基础 |
| M2 | 标签字典重构与业务决策字段补齐 | Phase 1-6 | ✅ 已完成，v4.1 为当前生产口径 | 高 | 证明标签体系能支撑业务分析 |
| M3 | Phase 5 AI 打标管道上线 | Phase 5 | ✅ 已完成 | 高 | 证明 AI 能力不是 demo，而是分层管道 |
| M4 | Phase 6 质量风险发现与 Method C 修复 | Phase 6 | ✅ 已完成 | 高 | 证明团队主动发现并修复精度风险 |
| M5 | 静态 HTML 看板与 7 部门周报交付 | Phase 6 D10 | ✅ 已完成，已归档 | 中高 | 证明 AI 输出已转成业务可看产物 |
| M6 | Superset BI B 路径上线 | Phase 7 | ✅ 已完成，本地可用 | 高 | 证明平台产品化闭环已跑通 |
| M7 | 生产化与组织化推广准备 | Phase 8 候选 | 🟡 待立项 | 中 | 说明下一阶段需要资源与决策 |

---

## 三、里程碑审计明细

## M1：VOC 数据资产盘点与统一口径

### 目标

证明 NLP-VOC 项目不是基于假设或样例数据，而是基于真实跨境电商 VOC 数据资产展开：明确数据源、规模、文本特征、标签缺口和指标缺口。

### 交付物

| 类型 | 交付物 | 说明 |
|---|---|---|
| 数据盘点文档 | [数据资产盘点与缺口分析](../01-数据资产盘点/数据资产盘点与缺口分析.md) | 早期真实数据基础，记录 Zendesk/Trustpilot 等数据分布与缺口 |
| 研究目录索引 | [research README](../../README.md) | 说明数据资产、脚本工具、输出结果目录结构 |
| 当前项目入口 | [07-NLP-VOC README](../../../README.md) | 当前项目状态、数据规模、BI 看板和 Phase 7 状态 |
| 运行规约 | [CLAUDE.md](../../../CLAUDE.md) | 当前阶段状态快照、数据规模、关键脚本和不可动路径 |

### 验证结果

| 验证项 | 结果 | 说明 |
|---|---|---|
| 数据源识别 | ✅ | Amazon / Trustpilot / Zendesk / Momcozy / Reddit 等来源在后续主口径中被纳入 |
| 数据规模主口径 | ✅ 364,569 | 当前 README 与 CLAUDE.md 均采用 364,569 作为 Phase 5-7 主口径 |
| 早期缺口识别 | ✅ | AIPL、Proxy NPS、画像、ABSA、情感极性等缺口已在早期矩阵中明确 |
| 后续管道输入 | ✅ | `phase6_d9_filtered.jsonl` 被 Phase 7 ETL 作为 BI 主源 |

### 主证据

- [07-NLP-VOC README](../../../README.md)
- [CLAUDE.md](../../../CLAUDE.md)
- [数据资产盘点与缺口分析](../01-数据资产盘点/数据资产盘点与缺口分析.md)

### 交叉证据

- [Phase 5 架构与工作流复盘](phase5-architecture-and-workflow-retrospective.md)
- [ETL Pipeline SOP](../07-操作指南/ETL_pipeline_SOP.md)
- [Phase 7 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md)

### 遗留风险 / 需要补充确认

| 项 | 风险 | 建议 |
|---|---|---|
| 数据总量口径 | 早期 355,697 与当前 364,569 同时存在 | 汇报中明确：355,697 是早期盘点口径，364,569 是 Phase 5-7 主口径 |
| 数据脱敏 | 代表评论和 review_id 是否可对外展示未形成统一规则 | HTML Presentation 制作前确认脱敏要求 |
| 数据更新时间 | 当前是否为 W19 / 2026-05-11 截止口径 | 汇报前确认最新数据截止日期 |

### HTML Presentation 引用建议

- 主线页：数据资产全景、数据治理链路
- 附录页：数据口径说明

---

## M2：标签字典重构与业务决策字段补齐

### 目标

证明项目不是简单扩充标签，而是把原始人工/业务标签升级为可计算、可映射、可进入 BI 与策略闭环的业务决策中间层。

### 交付物

| 类型 | 交付物 | 说明 |
|---|---|---|
| 复盘文档 | [VOC 标签体系项目整体复盘](voc-tag-system-project-review-stable.md) | 说明 v3.0 到 v3.9 的语义继承、字段扩展和业务价值 |
| Phase 6 计划 | [Phase 6 字典与质量提升计划](../08-Phase计划/voc-tag-evolution-phase6-dictionary-quality-plan.md) | 说明 v3.9 → v4.1 字典质量收敛 |
| 输出索引 | [research 输出结果 README](../../04-输出结果/README.md) | 提供 v3.5/v3.6/v3.7 等历史字典版本索引 |

### 验证结果

| 验证项 | 结果 | 说明 |
|---|---|---|
| 语义继承 | ✅ | v3.9 文档称原始 465 标签语义 100% 继承，表达结构重构 |
| 字段扩展 | ✅ | 原始少量字段扩展到情感、AIPL、主责部门、策略包、原子指标等 |
| 当前生产字典 | ✅ v4.1 | Phase 6/7 文档均以 v4.1 作为后续 BI 与 ETL 主口径 |
| 下游映射 | ✅ | 标签进入主责部门、周报、BI views、Superset 看板 |

### 主证据

- [VOC 标签体系项目整体复盘](voc-tag-system-project-review-stable.md)
- [Phase 6 字典与质量提升计划](../08-Phase计划/voc-tag-evolution-phase6-dictionary-quality-plan.md)

### 交叉证据

- [Phase 5+6 完整复盘](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md)
- [Phase 7 架构图集](phase7-architecture-diagrams.md)
- [ETL Pipeline SOP](../07-操作指南/ETL_pipeline_SOP.md)

### 遗留风险 / 需要补充确认

| 项 | 风险 | 建议 |
|---|---|---|
| 字典版本口径 | v3.7/v3.9/v4.1 在不同文档中承担不同角色 | 汇报中统一称“当前 BI 生产口径为 v4.1，历史演进见附录” |
| 字典文件为 xlsx | 管理层不适合直接看 Excel | 为汇报制作“字典能力摘要页” |
| 字段完整性 | 历史文档中有 validator warning 记录 | 若对外汇报，确认 v4.1 validator 最新结果 |

### HTML Presentation 引用建议

- 主线页：标签体系从分类工具升级为决策中间层
- 附录页：字典版本演进表

---

## M3：Phase 5 AI 打标管道上线

### 目标

证明 AI 能力不是一次性 prompt 或试验脚本，而是分层、可回归、可监控、可扩展的产品级 AI 打标管道。

### 交付物

| 层级 | 交付物 | 说明 |
|---|---|---|
| L0 | 规则层 | Phase 4 保留，0 成本打底 |
| L1 | LLM 闭集打标 | DeepSeek 主 + Kimi 兜底，闭集 tag_id 输出 |
| L2 | ABSA | 抽取 aspect / sentiment / confidence |
| L3 | 画像 + NPS | 55 原子画像标签 + Proxy NPS 三法投票 |
| 统一出口 | unified labeler | 合并 L0-L3 输出，保持 Phase 4 兼容契约 |

### 验证结果

| 指标 | 结果 | 说明 |
|---|---:|---|
| 5K 子集覆盖率 | 97.22% | Phase 5 关键验证指标 |
| F1_weighted | 0.831 | LLM 三方评估指标 |
| Proxy NPS Cohen κ | 0.996 | NPS 与人工/规则高度一致 |
| ABSA aspect/记录 | 2.91 | 方面级情感抽取可用 |
| 画像标签渗透率 | 73.92% | 55 画像标签恢复后实测 |
| Week 1 Gate | 9/9 PASS | Phase 5 D7 收口 |

### 主证据

- [Phase 5 产品级闭环执行规划](../08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md)
- [Phase 5 架构与工作流复盘](phase5-architecture-and-workflow-retrospective.md)
- [Phase 5 架构图集](phase5-architecture-diagrams.md)

### 交叉证据

- [审计报告索引：Phase 5 D1-D14](../../04-输出结果/03-审计报告/00-INDEX.md)
- [CLAUDE.md 当前状态](../../../CLAUDE.md)
- [Phase 5 白话汇报](phase5-executive-brief.md)

### 遗留风险 / 需要补充确认

| 项 | 风险 | 建议 |
|---|---|---|
| 指标展示口径 | 97.22% 是 5K 子集覆盖率，不等同后续全量 Gate 指标 | 汇报中明确“Phase 5 小样本验证指标” |
| Top-1 100% | 基于人工 149 条严格金标，容易被误解为全量 100% | 建议主线慎用，放附录或加口径说明 |
| LLM 服务依赖 | DeepSeek/Kimi 是外部依赖 | 在风险页说明 API、缓存、重试和 fallback |

### HTML Presentation 引用建议

- 主线页：AI 打标管道架构、一条评论如何变成结构化洞察
- 附录页：Week 1 Gate 与 L0-L3 技术细节

---

## M4：Phase 6 质量风险发现与 Method C 修复

### 目标

证明团队没有只展示漂亮指标，而是主动用更严格口径暴露 AI 精度风险，并通过可解释的后处理策略完成修复。

### 交付物

| 阶段 | 交付物 / 事件 | 结果 |
|---|---|---|
| Phase 6 D7 | LLM 输出抽样质量评估 | 暴露 precision 0.639 风险 |
| Phase 6 D8 | strict prompt 重打 | precision 提升至 0.885，但 Gate 退至 5/7 |
| Phase 6 D9 | Method C 后处理过滤 | precision 0.896 + Week 2 Gate 7/7 |
| Phase 6 D10 | BI C+A 双路径 | 7 部门周报 + 静态 HTML 看板 |

### 验证结果

| 指标 | 修复前 | 修复后 | 说明 |
|---|---:|---:|---|
| precision | 0.639 | 0.896 | Method C 后处理过滤后 |
| Week 2 Gate | 5/7 或 7/7 波动 | 7/7 PASS | D9 最终收口 |
| 数据可发布性 | 风险较高 | 可进入部门看板 | 低置信 / 高风险标签经后处理过滤 |

### 主证据

- [Phase 6 字典与质量提升计划](../08-Phase计划/voc-tag-evolution-phase6-dictionary-quality-plan.md)
- [Phase 5+6 完整复盘](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md)

### 交叉证据

- [审计报告索引：Phase 6 D7-D9](../../04-输出结果/03-审计报告/00-INDEX.md)
- [Phase 6+7 白话汇报](phase6-7-executive-brief.md)
- [CLAUDE.md 当前状态](../../../CLAUDE.md)

### 遗留风险 / 需要补充确认

| 项 | 风险 | 建议 |
|---|---|---|
| 抽样规模 | 部分 spot check 样本规模较小 | 后续固定每周/每月抽样机制 |
| 高风险 tag 覆盖 | D9 Method C 主要针对一组高风险 tag | 后续扩展更多 tag 覆盖 |
| 全局 precision | 文档中存在 targeted/global 口径差异 | 汇报主线使用 targeted precision 0.896，并说明口径 |

### HTML Presentation 引用建议

- 主线页：质量治理：主动暴露问题而不是粉饰指标
- 附录页：Gate 演进表、Method C 原理

---

## M5：静态 HTML 看板与 7 部门周报交付

### 目标

证明 AI 输出已经转化为业务可读、可分发的交付物，为 Phase 7 实时交互 BI 奠定内容验证基础。

### 交付物

| 类型 | 交付物 | 说明 |
|---|---|---|
| 静态看板 | `dashboard-2026-W19.html` | Phase 6 D10 C 路径，125KB 单文件 |
| 部门周报 | 7 部门 × AGRS + MAA | 代表评论、摘要、行动建议 |
| 归档说明 | [Phase 6 HTML Dashboard 归档](../../00-归档资料/phase6_html_dashboard/README.md) | 说明 HTML 看板被 Phase 7 Superset 替代，但保留为参考样本 |

### 验证结果

| 验证项 | 结果 | 说明 |
|---|---|---|
| 看板生成 | ✅ | 单文件 HTML，可分发 |
| 部门覆盖 | ✅ 7 部门 | AGRS + MAA 形成部门级内容 |
| 业务样例 | ✅ | 产品中心/品线、客服中心等有真实周报输出 |
| 产品定位 | ✅ | 作为 C 路径快速交付，被 Phase 7 B 路径替代 |

### 主证据

- [Phase 6 HTML Dashboard 归档说明](../../00-归档资料/phase6_html_dashboard/README.md)
- [Phase 5+6 完整复盘](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md)

### 交叉证据

- [产品中心/品线 MAA 周报](../../04-输出结果/10-周报/2026-W19-d9/产品中心-品线_MAA.md)
- [全球客服与体验中心 AGRS 周报](../../04-输出结果/10-周报/2026-W19-d9/全球客服与体验中心_AGRS.md)
- [Phase 6+7 白话汇报](phase6-7-executive-brief.md)

### 遗留风险 / 需要补充确认

| 项 | 风险 | 建议 |
|---|---|---|
| 周报中的待填写项 | 部分 MAA 行动建议仍有“【待填写】” | 正式汇报前补齐或隐藏未完成字段 |
| 静态 HTML 已归档 | 不应作为当前主产品形态展示 | 作为“快速验证路径”展示，当前主路径为 Superset |
| 评论脱敏 | 周报含 review_id 与原文片段 | 对外汇报前确认脱敏要求 |

### HTML Presentation 引用建议

- 主线页：从 AI 输出到业务周报
- 附录页：AGRS / MAA 样例

---

## M6：Superset BI B 路径上线

### 目标

证明 NLP-VOC 已经从静态报告走向业务可交互分析平台：数据可入库、视图可复用、图表可重建、部门可筛选。

### 交付物

| 层级 | 交付物 | 说明 |
|---|---|---|
| 数据源 | `phase6_d9_filtered.jsonl` | 588MB，364,569 reviews，Method C 后处理后主源 |
| ETL | `etl_to_postgres.py` | 37s 导入 364K reviews + 约 690K labels |
| 数据库 | `voc_bi` Postgres | 4 张基表 + 6 SQL views |
| BI | Superset 4.1.1 Docker | 本地 `localhost:8088` |
| 看板 | 8 dashboards | 1 Overview + 7 部门 |
| 图表 | 12 charts | 5 Overview + 7 Dept Top-10 |
| 过滤器 | 10 native filters | Overview 3 个 + 7 部门 polarity |
| 运维 | Superset SOP / ETL SOP | 可重建、可迁移、可维护 |

### 验证结果

| 验证项 | 结果 | 说明 |
|---|---|---|
| D1 ETL | ✅ 37s | 导入 364K reviews + 约 690K labels |
| D2 Superset | ✅ | Docker + 6 datasets 注册 |
| D3 Dashboard | ✅ | 12 charts + 8 dashboards，ZIP 导出 |
| D4 Filters | ✅ | 10 native filters 写入 |
| 部门过滤 | ✅ | 产品中心/品线 polarity=负向后 Top-3 完全变化 |
| Overview 饼图 | ⚠️ | dashboard-mode 饼图渲染 bug，单图模式正常 |

### 主证据

- [Phase 7 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md)
- [Phase 7 D4 进度报告](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md)
- [Phase 7 架构图集](phase7-architecture-diagrams.md)

### 交叉证据

- [Phase 7 BI Superset 计划](../08-Phase计划/voc-tag-evolution-phase7-bi-superset-plan.md)
- [ETL Pipeline SOP](../07-操作指南/ETL_pipeline_SOP.md)
- [Superset BI SOP](../07-操作指南/Superset_BI_SOP.md)
- [07-NLP-VOC README](../../../README.md)

### 遗留风险 / 需要补充确认

| 项 | 风险 | 建议 |
|---|---|---|
| Overview 饼图 bug | 管理层总览页部分图表展示受影响 | Phase 8 P0：重建为 bar 或升级 Superset |
| 本地环境 | 当前是 localhost Docker | Phase 8：内网部署、HTTPS、域名 |
| 权限 | 尚未完成 RBAC/SSO | Phase 8：部门权限隔离与 SSO |
| 时间维度 | v_review_overview 暂无 timestamp | Phase 8：扩展 ETL 与视图，支持趋势分析 |
| UI E2E 范围 | D4 中部分以 SQL 语义验证替代 UI 全覆盖 | 后续补 Playwright 完整套件 |

### HTML Presentation 引用建议

- 主线页：Phase 7 Superset 产品化架构、BI 产品能力地图、典型用户操作路径
- 附录页：ETL / Superset 重建流程

---

## M7：生产化与组织化推广准备

### 目标

明确当前项目已经完成“本地可用 / 准生产验证”，但要进入组织级日常使用，需要管理层支持资源、权限、部署、流程和部门 owner。

### 待交付事项

| 优先级 | 事项 | 预估工作 |
|---|---|---|
| P0 | 修复 Overview 饼图 dashboard-mode 渲染 bug | 2-3 天 |
| P0 | 增加 timestamp / 时间过滤 | 1 天 |
| P1 | Superset 多用户 RBAC + SSO | 3-5 天 |
| P1 | Nginx + HTTPS + 公司域名 | 1-2 天 |
| P2 | 飞书 / 钉钉工作台嵌入 | 2-3 天 |
| P2 | 月度自动化 ETL + Superset 缓存清理 | 1 天 |
| P3 | 字典 v5.0 演进 | 5-10 天 |

### 当前证据

- [Phase 7 BI Superset 计划 §Phase 8 候选](../08-Phase计划/voc-tag-evolution-phase7-bi-superset-plan.md)
- [Phase 7 完整复盘 §未来工作与风险](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md)
- [Phase 7 D4 进度报告 §下一步建议](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md)

### 需要管理层决策

| 决策项 | 影响 | 建议 |
|---|---|---|
| 是否批准 Phase 8 | 决定平台是否进入生产化 | 建议批准小 Sprint，先完成 P0/P1 |
| 是否提供内网服务器/域名 | 决定是否可多人访问 | 建议由 IT/运维配合 |
| 是否指定 7 部门 owner | 决定洞察是否进入行动闭环 | 建议每部门指定 1 名业务 owner |
| 是否接入飞书/钉钉 | 决定平台触达效率 | 建议 Phase 8 后半段评估 |
| 是否启动第二品牌试点 | 决定平台复用价值验证 | 建议在生产化后启动 |

### HTML Presentation 引用建议

- 主线页：当前风险与限制、下一阶段路线图、需要管理层决策

---

## 四、交叉自证矩阵

| 关键结论 | 主证据 | 交叉证据 1 | 交叉证据 2 | 可信度 | 汇报处理 |
|---|---|---|---|---|---|
| 项目有真实 VOC 数据基础 | [数据资产盘点与缺口分析](../01-数据资产盘点/数据资产盘点与缺口分析.md) | [07-NLP-VOC README](../../../README.md) | [Phase 7 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) | 高 | 主线展示 |
| 当前主口径为 364,569 条 | [CLAUDE.md](../../../CLAUDE.md) | [07-NLP-VOC README](../../../README.md) | [Phase 7 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) | 高 | 主线展示，附口径说明 |
| 标签字典已从分类工具升级为决策中间层 | [VOC 标签体系项目整体复盘](voc-tag-system-project-review-stable.md) | [Phase 6 字典计划](../08-Phase计划/voc-tag-evolution-phase6-dictionary-quality-plan.md) | [ETL SOP](../07-操作指南/ETL_pipeline_SOP.md) | 高 | 主线展示 |
| Phase 5 AI 打标管道已上线 | [Phase 5 复盘](phase5-architecture-and-workflow-retrospective.md) | [Phase 5 图集](phase5-architecture-diagrams.md) | [审计报告索引](../../04-输出结果/03-审计报告/00-INDEX.md) | 高 | 主线展示 |
| Week 1 Gate 9/9 PASS | [CLAUDE.md](../../../CLAUDE.md) | [Phase 5 复盘](phase5-architecture-and-workflow-retrospective.md) | [审计报告索引](../../04-输出结果/03-审计报告/00-INDEX.md) | 高 | 主线展示 |
| Phase 6 precision 风险被主动发现并修复 | [Phase 6 计划](../08-Phase计划/voc-tag-evolution-phase6-dictionary-quality-plan.md) | [Phase 5+6 完整复盘](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) | [Phase 6+7 白话汇报](phase6-7-executive-brief.md) | 高 | 必须重点讲 |
| Method C 后处理后 precision 0.896 | [Phase 6 计划](../08-Phase计划/voc-tag-evolution-phase6-dictionary-quality-plan.md) | [CLAUDE.md](../../../CLAUDE.md) | [Phase 5+6 完整复盘](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) | 高 | 主线展示，附口径说明 |
| HTML 看板已交付但归档 | [Phase 6 HTML 归档说明](../../00-归档资料/phase6_html_dashboard/README.md) | [Phase 5+6 完整复盘](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) | [Phase 6+7 白话汇报](phase6-7-executive-brief.md) | 中高 | 作为过渡页展示 |
| 7 部门周报已有真实输出 | [产品中心/品线 MAA](../../04-输出结果/10-周报/2026-W19-d9/产品中心-品线_MAA.md) | [全球客服 AGRS](../../04-输出结果/10-周报/2026-W19-d9/全球客服与体验中心_AGRS.md) | [Phase 5+6 完整复盘](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) | 中高 | 展示样例，但注意待填写项 |
| Superset BI B 路径完整闭环 | [Phase 7 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) | [Phase 7 D4 报告](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md) | [Superset BI SOP](../07-操作指南/Superset_BI_SOP.md) | 高 | 主线展示 |
| 部门 polarity filter 端到端有效 | [Phase 7 D4 报告](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md) | [Phase 7 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) | [Phase 7 图集](phase7-architecture-diagrams.md) | 高 | 主线展示 |
| Overview 饼图仍有遗留 bug | [Phase 7 D4 报告](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md) | [Phase 7 计划](../08-Phase计划/voc-tag-evolution-phase7-bi-superset-plan.md) | [CLAUDE.md](../../../CLAUDE.md) | 高 | 风险页必须说明 |
| 生产化仍需 Phase 8 支持 | [Phase 7 计划](../08-Phase计划/voc-tag-evolution-phase7-bi-superset-plan.md) | [Phase 7 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) | [Phase 7 D4 报告](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md) | 中高 | 决策请求页展示 |

---

## 五、HTML Presentation 页面映射

| 汇报页 | 建议标题 | 对应里程碑 | 证据来源 |
|---|---|---|---|
| Page 2 | Executive Summary | M1-M6 | README / CLAUDE / Phase 7 复盘 |
| Page 3 | 为什么要做 | M1 | 数据资产盘点 / 指标-标签矩阵 |
| Page 4 | 数据资产全景 | M1 | 数据资产盘点 / Phase 5 复盘 |
| Page 5 | 数据治理链路 | M1-M2 | research README / ETL SOP |
| Page 6 | 标签体系升级 | M2 | 标签体系整体复盘 / Phase 6 计划 |
| Page 7 | AI 打标管道 | M3 | Phase 5 图集 / Phase 5 复盘 |
| Page 8 | 评论结构化样例 | M3 | Phase 5 复盘 / AGRS 输出 |
| Page 9 | 质量治理 | M4 | Phase 6 计划 / Phase 5+6 复盘 |
| Page 10 | AI 到业务周报 | M5 | AGRS / MAA 周报 |
| Page 11 | HTML 看板过渡路径 | M5 | HTML 归档说明 |
| Page 12 | Superset 产品化架构 | M6 | Phase 7 图集 / ETL SOP |
| Page 13 | BI 产品能力地图 | M6 | README / Phase 7 复盘 |
| Page 14 | 用户操作路径 | M6 | Phase 7 D4 报告 |
| Page 15 | 项目进度全景 | M1-M7 | 审计报告索引 |
| Page 16 | 当前风险与限制 | M6-M7 | Phase 7 D4 / Phase 7 计划 |
| Page 17 | 下一阶段路线图 | M7 | Phase 7 计划 / Phase 7 复盘 |
| Page 18 | 管理层决策请求 | M7 | 本文档 + Phase 7 计划 |
| Appendix | 里程碑审计与交叉自证矩阵 | M1-M7 | 本文档 |

---

## 六、正式汇报前待确认清单

| 类别 | 待确认项 | 原因 | 建议负责人 |
|---|---|---|---|
| 数据口径 | 汇报是否统一采用 364,569 条 | 避免与早期 355,697 混淆 | 项目负责人 |
| 指标口径 | precision 0.896 是否作为正式管理层指标 | 需说明 targeted 口径 | 项目负责人 / 算法负责人 |
| 金标口径 | Top-1 100% 是否放主线 | 易被误解为全量准确率 | 项目负责人 |
| BI 状态 | Overview 饼图 bug 是否仍存在 | 决定风险页表述 | BI 负责人 |
| 数据脱敏 | 代表评论与 review_id 是否可展示 | 涉及隐私/合规 | 业务负责人 / 法务或数据治理 |
| 周报内容 | MAA 中“【待填写】”是否补全 | 影响业务可信度 | 各部门 owner |
| 平台访问 | 是否已有内网或正式演示环境 | 决定是否可现场 demo | 运维/IT |
| 组织闭环 | 7 部门是否有 owner 与处理 SLA | 决定下一阶段落地 | 管理层 |

---

## 七、一句话审计结论

> NLP-VOC 项目已形成从数据资产盘点、标签字典治理、AI 打标管道、质量风险修复、业务周报生成到 Superset BI 平台上线的完整证据链。当前关键成果均可通过计划文档、复盘报告、进度审计、SOP 与真实输出物交叉自证；下一阶段重点不是证明“能不能做”，而是推进生产化部署、权限治理、时间维度、部门行动闭环和多品牌复用。
