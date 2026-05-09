---
name: voc-nlp-readme
description: 07-NLP-VOC 子项目 README — 项目快速上手入口，含目录全貌、按角色导航、Phase 5 当前状态、关键链接。新人第一个读的文档，5 分钟看完就能知道项目是什么、代码在哪、文档在哪、当前在做什么。
title: 07-NLP-VOC VOC 标签体系子项目
doc_type: readme
module: voc-nlp
status: stable
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# 07-NLP-VOC — VOC 标签体系子项目

> **paper2skills 仓库下的 VOC 标签体系子项目**。用 AI 把 Momcozy 36 万条跨境电商评论转成结构化标签 + NPS + 画像 + 方面情感，支撑 7 个部门的 BI 决策。

## 🚀 3 分钟入门

| 你是谁 | 先读这份 | 再读这份 |
|---|---|---|
| **老板/BD** | [Phase 5 白话汇报 §0+§2](research/01-设计文档/00-Phase5-汇报与复盘/phase5-executive-brief.md) (5 分钟) | [§3 四个价值角度](research/01-设计文档/00-Phase5-汇报与复盘/phase5-executive-brief.md) |
| **跨部门同事** | [Phase 5 白话汇报 §0-§6](research/01-设计文档/00-Phase5-汇报与复盘/phase5-executive-brief.md) (15 分钟) | [架构图集](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) |
| **技术接手** | [CLAUDE.md](CLAUDE.md) (系统规约) | [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) (30 分钟) |
| **新人上手** | 本文 → [CLAUDE.md](CLAUDE.md) → [Phase 5 白话汇报](research/01-设计文档/00-Phase5-汇报与复盘/phase5-executive-brief.md) | [01-设计文档 索引](research/01-设计文档/README.md) |
| **外部审计** | [Phase 5 白话汇报 §7+§8+§10](research/01-设计文档/00-Phase5-汇报与复盘/phase5-executive-brief.md) | [Week 1 Gate 9/9 PASS](research/04-输出结果/03-审计报告/phase5_d7_week1_gate_final.md) |

## 📊 当前状态（2026-05-08）

| 维度 | 值 |
|---|---|
| Phase | **Phase 5 D8/14** 进行中 |
| Week 1 Gate | 🟢 9/9 PASS |
| 5K 子集覆盖率 | **97.22%**（Phase 4 是 82.58%） |
| LLM 评估 F1_weighted | **0.831** |
| Proxy NPS 相关性 | **0.996** |
| 严格金标 Top-1 准确率 | **100%** |
| D8 全量增打进度 | 87K 条后台 chunked 运行中 |
| 后台进程 | `cat /tmp/d8_labeler.pid` / `cat /tmp/d8_monitor.pid` |

## 📁 目录地图

```
07-NLP-VOC/
├── README.md                       ← 本文（3 分钟入口）
├── CLAUDE.md                       ← AI 助手运行规约 + 阻塞处置
├── 00-知识库-Skill卡片/            ← 40 张论文转换的算法卡（含 INDEX）
├── 00-知识库-架构图谱/             ← 3 份业务级架构图
├── papers/                         ← 下载的论文 PDF
└── research/
    ├── 00-归档资料/                 历史归档
    ├── 01-设计文档/                 ⭐ 核心设计文档（见 README 索引）
    │   ├── 00-Phase5-汇报与复盘/    ⭐ Phase 5 4 份核心产出 + 历史复盘
    │   ├── 01-数据资产盘点/
    │   ├── 02-工作流设计/           ← persona_tags_55.json（代码直读）
    │   ├── 03-自动打标调研/
    │   ├── 04-NPS校准方法/
    │   ├── 05-审计报告/
    │   ├── 06-设计草稿/
    │   ├── 07-操作指南/
    │   └── 08-Phase计划/            ⭐ Phase 4 / 5 原始计划
    ├── 02-脚本工具/                 ⭐⭐ 代码主目录（不动结构）
    │   ├── 01-标签进化/             L0/L2/L3 + unified labeler
    │   ├── 02-数据采样/
    │   ├── 03-批量打标/
    │   ├── 04-数据处理/
    │   ├── 05-NPS管道/
    │   ├── 06-诊断工具/             schema validator / monitor / persona diagnostic
    │   └── 07-LLM引擎/              ⭐ Phase 5 新建（所有 LLM 工具）
    ├── 03-数据资产/                 ⭐ 中间产物（jsonl）
    └── 04-输出结果/
        ├── 01-字典版本/             v3.5 → v3.9
        ├── 03-审计报告/             ⭐ D1-D8 每日进度（含 INDEX）
        ├── 05-运行日志/             后台运行日志
        └── unified_labeling/        phase4_labeled.jsonl + phase5_full_labeled_llm.jsonl
```

## 🧭 核心能力（Phase 5）

5 层流水线（详见 [架构图集 §图 1](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md)）：

| 层 | 做什么 | 引入阶段 |
|---|---|---|
| L0 规则层 | 关键词 + 品牌词 + alchemist 弱监督打底 | Phase 4 保留 |
| **L1 LLM 闭集层** | DeepSeek 主 + Kimi 兜底，643 标签闭集 | **Phase 5 D2 新增** |
| **L2 ABSA 层** | 抽 (aspect, sentiment, confidence) 三元组 | **Phase 5 D4 新增** |
| **L3 画像 + NPS 层** | 55 原子画像 + 三法投票 NPS | **Phase 5 D5/D6 新增** |
| 统一出口 | 按 review_id 合并 + meta | **Phase 5 D7 新增** |

## 🛡 质量保障

- **9 项 Quality Gate**（发布前自动判定 PASS/FAIL）→ Week 1 **9/9 PASS**
- **7-check Schema Validator** → 7/7 PASS
- **双金标交叉验证**（500 自动共识 + 149 人工真值）
- **实时监控**（D8 滑窗 1000 三红线）

## 🔑 核心配置

| 项 | 位置 |
|---|---|
| LLM Keys | `~/.paper2skills/llm_keys.json`（chmod 600） |
| DeepSeek 并发 | 40 |
| Kimi 并发 | 1（RPM 200 限速） |
| 依赖 | `openai` / `pydantic≥2` / `pyarrow` |

Smoke test：
```bash
python research/02-脚本工具/07-LLM引擎/llm_client.py --smoke-test
```

## 🚨 不能动的清单

以下路径**被代码写死**，任何整理都不碰：

| 路径 | 引用方 |
|---|---|
| `research/02-脚本工具/` 子目录 | 20+ 脚本 import |
| `research/03-数据资产/*.jsonl` | 10+ 脚本读写 |
| `research/04-输出结果/03-审计报告/` | 脚本写报告 |
| `research/04-输出结果/unified_labeling/` | D8 后台运行写入 |
| `research/01-设计文档/02-工作流设计/persona_tags_55.json` | labeler 直读 |

详见 [CLAUDE.md 工作流约束](CLAUDE.md)。

## 📖 下一步

- **完成 Phase 5 W2**：D9 字典进化 v4.0 → D14 Phase 5 收口
- **启动 Phase 6**：BI 看板 7 部门 spec + A→B 过渡试点（见 [Phase 5 主复盘 §七](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md)）

## 🔗 重要链接

| 类型 | 入口 |
|---|---|
| AI 助手规约 | [CLAUDE.md](CLAUDE.md) |
| 汇报素材库 | [research/01-设计文档/00-Phase5-汇报与复盘/](research/01-设计文档/00-Phase5-汇报与复盘/) |
| Phase 原始计划 | [research/01-设计文档/08-Phase计划/](research/01-设计文档/08-Phase计划/) |
| 每日进度 | [research/04-输出结果/03-审计报告/00-INDEX.md](research/04-输出结果/03-审计报告/00-INDEX.md) |
| 算法卡片 | [00-知识库-Skill卡片/00-INDEX.md](00-知识库-Skill卡片/00-INDEX.md) |
| 业务架构图 | [00-知识库-架构图谱/00-INDEX.md](00-知识库-架构图谱/00-INDEX.md) |
| 设计文档总索引 | [research/01-设计文档/README.md](research/01-设计文档/README.md) |

---

> **本文档定位**：新人 3 分钟入门 + 老用户快速导航。深度内容在 [CLAUDE.md](CLAUDE.md) 和 [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md)。
