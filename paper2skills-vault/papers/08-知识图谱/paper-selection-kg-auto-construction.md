---
title: 选题报告：LLM驱动的电商领域知识图谱自动构建
doc_type: analysis
module: 08-知识图谱
topic: knowledge-graph-auto-construction
status: draft
created: 2026-05-01
updated: 2026-05-01
owner: self
source: ai
---

# 选题报告：LLM驱动的电商领域知识图谱自动构建

## 搜索策略

### 搜索关键词
```
主关键词: LLM knowledge graph construction
补充关键词: automated entity relation extraction e-commerce
补充关键词: zero-shot relation extraction LLM
补充关键词: product knowledge graph AI agent
```

### 筛选标准
1. 有代码实现或明确可复现的方法论
2. 2024-2026年发表
3. 非纯综述类（优先实验验证）
4. 与电商/产品知识图谱场景相关

---

## 候选论文评估

### 候选 1（推荐主论文）

| 属性 | 内容 |
|------|------|
| **标题** | AI Agent-Driven Framework for Automated Product Knowledge Graph Construction in E-Commerce |
| **arXiv ID** | 2511.11017 |
| **发表时间** | 2025-11 |
| **核心贡献** | 用 AI Agent 自动化构建电商产品知识图谱，覆盖实体/关系抽取、Schema创建、实体规范化全流程 |
| **方法概述** | 多 Agent 协作：抽取 Agent → Schema Agent → 规范化 Agent → 验证 Agent |
| **业务匹配度** | ⭐⭐⭐⭐⭐ 直接面向电商产品 KG 构建 |
| **代码可得性** | 待确认（arXiv 预印本，可能有代码） |
| **创新性** | ⭐⭐⭐⭐☆ Agent 驱动是趋势方向 |
| **可萃取性** | ⭐⭐⭐⭐⭐ 方法清晰，易转化为代码模板 |

**优势**：
- 电商场景完全匹配，无需大量业务适配
- Agent 架构天然支持模块化，代码模板易拆分
- 2025年最新，代表技术前沿

**风险**：
- 预印本，未经同行评审
- 代码实现可能尚未公开

---

### 候选 2（方法支撑）

| 属性 | 内容 |
|------|------|
| **标题** | Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction |
| **arXiv ID** | 2404.03868 |
| **发表时间** | EMNLP 2024 |
| **核心贡献** | 端到端 LLM 框架：抽取实体关系 → 定义类型 → 规范化入库 |
| **方法概述** | 三阶段流水线：Extraction（LLM抽取三元组）→ Definition（自动推断Schema）→ Canonicalization（实体消歧与合并） |
| **业务匹配度** | ⭐⭐⭐⭐☆ 通用框架，电商可适配 |
| **代码可得性** | 高（EMNLP 论文通常有代码） |
| **创新性** | ⭐⭐⭐☆☆ 三阶段流水线思路清晰但非首创 |
| **可萃取性** | ⭐⭐⭐⭐⭐ 方法论结构化，易转化为 Skill |

**优势**：
- EMNLP 顶会，质量有保障
- 三阶段方法论清晰，适合作为 Skill Card 的算法原理
- 代码实现成熟

---

### 候选 3（增量构建参考）

| 属性 | 内容 |
|------|------|
| **标题** | iText2KG: Incremental Knowledge Graphs Construction using Large Language Models |
| **发表时间** | WISE 2024 |
| **核心贡献** | 增量式 KG 构建，流式文档持续更新图谱 |
| **业务匹配度** | ⭐⭐⭐☆☆ 适合持续更新的电商场景，但主框架不够聚焦 |
| **可萃取性** | ⭐⭐⭐☆☆ 增量逻辑较复杂，代码模板工作量较大 |

---

### 候选 4（多智能体富化）

| 属性 | 内容 |
|------|------|
| **标题** | KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment |
| **arXiv ID** | 2502.06472 |
| **发表时间** | 2025-02 |
| **核心贡献** | 多智能体 LLM 管道用于 KG 富化和精炼 |
| **业务匹配度** | ⭐⭐⭐☆☆ 偏向 KG 富化而非初始构建 |
| **可萃取性** | ⭐⭐⭐☆☆ 适合作为延伸技能参考 |

---

## 最终选择

### 主论文
**AI Agent-Driven Framework for Automated Product Knowledge Graph Construction in E-Commerce** (arXiv:2511.11017)

**选择理由**：
1. 电商场景完全匹配，母婴商品属性/关系抽取可直接套用
2. Agent 架构与项目已有 MAS（多智能体系统）技能形成自然延伸
3. 填补了 08-知识图谱领域"图谱构建"的核心缺口
4. 与现有 GraphRAG 技能形成上游依赖：本 Skill 负责"建图谱"，GraphRAG 负责"用图谱"

### 辅助参考
**Extract, Define, Canonicalize** (EMNLP 2024, arXiv:2404.03868)
- 用于补充方法论细节，特别是 Canonicalization（实体规范化）阶段

---

## 与现有技能的关系

```
本 Skill（KG Auto Construction）
    ├── 前置: Knowledge Graph for Skills Management（图基础概念）
    ├── 下游: GraphRAG（本 Skill 构建的 KG 是 GraphRAG 的输入）
    ├── 组合: VOC Semantic Blueprint（VOC 提取的结构化数据可作为 KG 数据源）
    └── 组合: Product Attribute Graph Parsing（属性图解析的延伸）
```

---

## 业务价值预判

| 场景 | 价值 |
|------|------|
| 自动构建母婴商品知识图谱 | 从商品描述/评论自动抽取"商品-属性-场景"三元组 |
| 客服对话结构化 | 从 Zendesk/Amazon 客服记录中抽取实体关系 |
| 图谱持续更新 | 新品上架时自动扩展图谱，无需人工维护 |

预估 ROI：10-15x（开发 2-3 周，节省大量人工标注和图谱维护成本）

---

## 下一步动作

1. [ ] 下载主论文 PDF
2. [ ] 通读并提取核心算法
3. [ ] 生成 Skill Card（5模块结构）
4. [ ] 编写可运行代码模板
5. [ ] 质量审核
6. [ ] 同步到 vault + GitHub

---

**报告日期**: 2026-05-01
**选题人**: Claude Code
**状态**: 选题完成，待进入萃取阶段
