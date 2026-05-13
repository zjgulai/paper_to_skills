---
name: voc-nlp-readme
description: 07-NLP-VOC 子项目 README — 项目快速上手入口，含目录全貌、按角色导航、Phase 7 当前状态、BI B 路径与关键链接。新人 5 分钟看完就能知道项目是什么、代码在哪、文档在哪、当前在做什么。
title: 07-NLP-VOC VOC 标签体系子项目
doc_type: readme
module: voc-nlp
status: stable
created: 2026-05-08
updated: 2026-05-11
owner: self
source: ai
---

# 07-NLP-VOC — VOC 标签体系子项目

> **paper2skills 仓库下的 VOC 标签体系子项目**。用 AI 把跨境电商 364K 条评论转成结构化标签 + NPS + 画像 + 方面情感，落到 Superset BI 看板支撑 7 个部门决策。

## 🚀 3 分钟入门

| 你是谁 | 先读这份 | 再读这份 |
|---|---|---|
| **老板/BD** | [Phase 5 白话汇报 §0+§2](research/01-设计文档/00-Phase5-汇报与复盘/phase5-executive-brief.md)（5 分钟） | [Phase 7 完整复盘](research/04-输出结果/03-审计报告/phase7_complete_retrospective.md)（BI 看板上线） |
| **跨部门同事** | 本文 + [Phase 7 看板访问指南](#superset-bi-看板) | [架构图集](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) |
| **技术接手** | [CLAUDE.md](CLAUDE.md)（系统规约） | [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md)（30 分钟） |
| **新人上手** | 本文 → [CLAUDE.md](CLAUDE.md) → [Phase 5 白话汇报](research/01-设计文档/00-Phase5-汇报与复盘/phase5-executive-brief.md) | [01-设计文档 索引](research/01-设计文档/README.md) |
| **外部审计** | [Phase 5 D7 Week 1 Gate 9/9](research/04-输出结果/03-审计报告/phase5_d7_week1_gate_final.md) | [Phase 6 D9 Week 2 Gate 7/7](research/04-输出结果/03-审计报告/phase6_d9_week2_gate.md) |

## 📊 当前状态（2026-05-11）

| 维度 | 值 |
|---|---|
| **当前阶段** | 🟢 Phase 7 D4 完成（BI B 路径实质上线） |
| Phase 5 | ✅ D14 部分收官（Momus 审阅通过 + 归档） |
| Phase 6 | ✅ D10 BI 看板实质上线（C+A 双路径，14 周报 + 125KB HTML） |
| Phase 7 | ✅ Superset 12 charts + 8 dashboards + 10 native filters |
| 全量打标进度 | 364,569 条 ✅ 全部完成 |
| Week 1 Gate | 🟢 9/9 PASS（D7 收口） |
| Week 2 Gate | 🟢 7/7 PASS（D9 Method C 后处理过滤后 precision 0.896） |
| LLM 评估 F1_weighted | **0.831** |
| Proxy NPS 相关性 | **0.996** |
| 严格金标 Top-1 准确率 | **100%**（人工 149 条） |

## 🎯 Superset BI 看板

### 访问入口

```bash
# 启动 Superset (本地)
docker compose up -d   # 在 research/02-脚本工具/01-标签进化/docker/ 下
open http://localhost:8088
# 用户/密码：admin / voc_admin_2026
```

### 已上线看板（8 个）

| Dashboard | 路径 | Filters |
|---|---|---|
| VOC Overview · 全局总览 | `/dashboard/1/` | 数据源 / 产品线 / Proxy NPS（针对饼图） |
| VOC · 全球客服中心 | `/dashboard/2/` | 情感极性 |
| VOC · 产品中心 | `/dashboard/3/` | 情感极性 |
| VOC · 仓储物流部 | `/dashboard/4/` | 情感极性 |
| VOC · 品牌市场中心 | `/dashboard/5/` | 情感极性 |
| VOC · 电商运营部 | `/dashboard/6/` | 情感极性 |
| VOC · 品质管理中心 | `/dashboard/7/` | 情感极性 |
| VOC · 法务合规部 | `/dashboard/8/` | 情感极性 |

### 重建（迁移环境时）

```bash
cd research/02-脚本工具/01-标签进化/docker/
python3 superset_bootstrap.py            # voc_bi 数据库 + 6 datasets
python3 superset_charts_factory.py       # 12 charts + 8 dashboards
python3 superset_filters_factory.py      # 10 native filters
```

或直接导入 8 个 ZIP：[research/04-输出结果/11-BI看板/superset_exports/](research/04-输出结果/11-BI看板/superset_exports/)

## 📁 目录地图

```
07-NLP-VOC/
├── README.md                       ← 本文（3 分钟入口）
├── CLAUDE.md                       ← AI 助手运行规约 + 阻塞处置
├── 00-知识库-Skill卡片/            ← 40+ 张论文转换的算法卡（含 INDEX）
├── 00-知识库-架构图谱/             ← 3 份业务级架构图
├── papers/                         ← 下载的论文 PDF
└── research/
    ├── 00-归档资料/                 ⭐ Phase 1-4 归档（含 phase6_html_dashboard / weekly_drafts）
    ├── 01-设计文档/                 ⭐ 核心设计文档（见 README 索引）
    │   ├── 00-Phase5-汇报与复盘/    ⭐ Phase 5 4 份核心产出
    │   ├── 02-工作流设计/           ← persona_tags_55.json（代码直读）
    │   └── 08-Phase计划/            ⭐ Phase 5 / 6 / 7 计划
    ├── 02-脚本工具/                 ⭐⭐ 代码主目录（不动结构）
    │   ├── 01-标签进化/             L0/L2/L3 + unified labeler + Superset factories
    │   │   └── docker/              ⭐ Phase 7 新增：Superset Docker compose + factories
    │   ├── 05-NPS管道/
    │   ├── 06-诊断工具/             schema validator / monitor / dual_coverage / spot_check
    │   └── 07-LLM引擎/              ⭐ Phase 5 新建（所有 LLM 工具）
    ├── 03-数据资产/                 中间产物（jsonl，gitignore 排除大文件）
    └── 04-输出结果/
        ├── 01-字典版本/             v3.5 → v4.1（v4.1 当前生产版本）
        ├── 03-审计报告/             ⭐ 118 个文件 = Phase 3-7 全程进度（含 INDEX）
        ├── 10-周报/                 ⭐ Phase 6 D9 周报：7 部门 × AGRS+MAA = 28 文件
        ├── 11-BI看板/               ⭐ Phase 7 新增：superset_exports/ 8 ZIP 可重建
        └── unified_labeling/        ⭐ phase6_d9_filtered.jsonl（Phase 7 ETL 主源）
```

## 🧭 核心能力（Phase 5-7 累计）

### 5 层 AI 打标流水线（Phase 5 主体）

| 层 | 做什么 | 引入阶段 |
|---|---|---|
| L0 规则层 | 关键词 + 品牌词 + alchemist 弱监督打底 | Phase 4 保留 |
| **L1 LLM 闭集层** | DeepSeek 主 + Kimi 兜底，643 标签闭集 | **Phase 5 D2 新增** |
| **L2 ABSA 层** | 抽 (aspect, sentiment, confidence) 三元组 | **Phase 5 D4 新增** |
| **L3 画像 + NPS 层** | 55 原子画像 + 三法投票 NPS | **Phase 5 D5/D6 新增** |
| 统一出口 | 按 review_id 合并 + meta | **Phase 5 D7 新增** |

### Phase 6 字典进化与质量提升

- **v4.0 → v4.1**：字段质量修复 + 多语言重打 + 离线 confidence 重赋
- **Method C 后处理过滤**：strict prompt 重打 + 后处理过滤 → precision 0.639 → 0.896
- **Week 2 Gate 7/7 PASS**

### Phase 7 BI B 路径（Superset 实时交互）

- voc_bi 数据库 + ETL + 6 SQL 视图
- 12 charts + 8 dashboards + 10 native filters
- 累计开发 ~7h，0 LLM 成本

## 🛡 质量保障

- **Week 1 Gate**：9/9 PASS（Phase 5 D7）
- **Week 2 Gate**：7/7 PASS（Phase 6 D9）
- **7-check Schema Validator**：7/7 PASS
- **双金标交叉验证**：500 自动共识 + 149 人工真值
- **Phase 7 端到端验证**：Playwright 实测 dept polarity 过滤通过

## 🔑 核心配置

| 项 | 位置 |
|---|---|
| LLM Keys | `~/.paper2skills/llm_keys.json`（chmod 600） |
| DeepSeek 并发 | 40 |
| Kimi 并发 | 1（RPM 200 限速） |
| Superset 用户 | admin / voc_admin_2026 |
| Superset 端口 | 8088（本地 Docker） |
| 依赖 | `openai` / `pydantic≥2` / `pyarrow` + `apache-superset` (Docker) |

Smoke test：
```bash
python research/02-脚本工具/07-LLM引擎/llm_client.py --smoke-test
curl -sS http://localhost:8088/health
```

## 🚨 不能动的清单

以下路径**被代码写死**，任何整理都不碰：

| 路径 | 引用方 |
|---|---|
| `research/02-脚本工具/` 子目录 | 20+ 脚本 import |
| `research/03-数据资产/*.jsonl` | 10+ 脚本读写 |
| `research/04-输出结果/03-审计报告/` | 脚本写报告（118 文件） |
| `research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl` | Phase 7 ETL 主源 |
| `research/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl` | 6 个脚本默认输入 + symlink 目标 |
| `research/01-设计文档/02-工作流设计/persona_tags_55.json` | labeler 直读 |
| `voc_bi` Postgres 数据库 + 6 视图 | Superset 全部依赖 |

详见 [CLAUDE.md 工作流约束](CLAUDE.md)。

## 📖 下一步

- **Phase 8 候选**：修复 Superset 4.1.1 dashboard-mode 饼图渲染 bug（D3 遗留）
- **Phase 8 候选**：v_review_overview 增加时间戳列以支持时间维度过滤
- **运维**：完善 Superset 多用户权限模型 + 7 部门 SSO 接入

## 🔗 重要链接

| 类型 | 入口 |
|---|---|
| AI 助手规约 | [CLAUDE.md](CLAUDE.md) |
| Phase 5 主复盘 | [research/01-设计文档/00-Phase5-汇报与复盘/](research/01-设计文档/00-Phase5-汇报与复盘/) |
| **Phase 6+7 白话汇报** ⭐ | [phase6-7-executive-brief.md](research/01-设计文档/00-Phase5-汇报与复盘/phase6-7-executive-brief.md) |
| **Phase 7 架构图集**（Mermaid） ⭐ | [phase7-architecture-diagrams.md](research/01-设计文档/00-Phase5-汇报与复盘/phase7-architecture-diagrams.md) |
| **Phase 7 架构图集**（HTML 高保真） ⭐ | [phase7-architecture-diagrams.html](research/01-设计文档/00-Phase5-汇报与复盘/phase7-architecture-diagrams.html) |
| **Superset BI 运维 SOP** ⭐ | [Superset_BI_SOP.md](research/01-设计文档/07-操作指南/Superset_BI_SOP.md) |
| **ETL Pipeline SOP** ⭐ | [ETL_pipeline_SOP.md](research/01-设计文档/07-操作指南/ETL_pipeline_SOP.md) |
| Phase 5+6 完整复盘 | [phase5_6_complete_retrospective.md](research/04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) |
| Phase 7 完整复盘 | [phase7_complete_retrospective.md](research/04-输出结果/03-审计报告/phase7_complete_retrospective.md) |
| 每日进度索引 | [research/04-输出结果/03-审计报告/00-INDEX.md](research/04-输出结果/03-审计报告/00-INDEX.md) |
| Phase 计划文档（5/6/7） | [research/01-设计文档/08-Phase计划/](research/01-设计文档/08-Phase计划/) |
| Superset 看板导出 | [research/04-输出结果/11-BI看板/superset_exports/](research/04-输出结果/11-BI看板/superset_exports/) |
| 算法卡片 | [00-知识库-Skill卡片/00-INDEX.md](00-知识库-Skill卡片/00-INDEX.md) |
| 业务架构图 | [00-知识库-架构图谱/00-INDEX.md](00-知识库-架构图谱/00-INDEX.md) |

---

> **本文档定位**：新人 3 分钟入门 + 老用户快速导航。深度内容在 [CLAUDE.md](CLAUDE.md) 和各 Phase 复盘文档。
