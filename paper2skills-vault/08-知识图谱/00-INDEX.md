---
title: 08-知识图谱 技能索引
doc_type: index
module: 08-知识图谱
topic: skill-cards-index
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
---

# 08-知识图谱 技能索引

---

## 领域定位

08-知识图谱是 paper2skills 工作流中的**结构化表示层**：

```
SRL 抽取语义结构 → 实体归一化 → 异构图构建 → HGT/HGCN 学习 → 图推理 → 语义蓝图
```

本领域聚焦图神经网络、知识图谱构建、层次化表示学习。

---

## 技能分类

### A. 异构表示学习

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| ⭐ [Skill-HGT-Heterogeneous-Graph-Transformer](Skill-HGT-Heterogeneous-Graph-Transformer.md) | 已萃取 | HGT (WWW 2020) | 母婴电商异构图（用户-产品-评论）表示学习 |
| ⭐ [Skill-HGCN-Hyperbolic-Graph-Convolutional-Networks](Skill-HGCN-Hyperbolic-Graph-Convolutional-Networks.md) | 已萃取 | HGCN (NeurIPS 2019) | 品类层次树的双曲嵌入 |

### B. 知识图谱构建

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-KG-Auto-Construction-Agent-Driven](Skill-KG-Auto-Construction-Agent-Driven.md) | 已萃取 | 自定义框架 | Agent 驱动的知识图谱自动构建 |
| [Skill-Knowledge-Graph-for-Skills-Management](Skill-Knowledge-Graph-for-Skills-Management.md) | 已萃取 | 自定义框架 | 技能管理知识图谱 |

### C. 语义检索

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-Dense-Retrieval-Ecommerce-Semantic-Search](Skill-Dense-Retrieval-Ecommerce-Semantic-Search.md) | 已萃取 | Dense Retrieval | 电商语义搜索 |
| [Skill-GraphRAG-Knowledge-Enhanced-Retrieval](Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md) | 已萃取 | GraphRAG | 知识增强检索 |

---

## 工作流映射

```
[07-NLP-VOC] SRL + 事件框架抽取
    ↓ 输出: 实体 / 关系 / 事件
[08-知识图谱] ← 当前领域
    ├─ HGT: 异构图表示学习（用户-产品-评论多关系）
    ├─ HGCN: 层次结构编码（品类树双曲嵌入）
    └─ KG Construction: 知识图谱自动构建
    ↓ 输出: 节点 embedding + 图推理结果
[07-NLP-VOC] 语义蓝图编译器
    ↓ 输出: Task Blueprint
[10-MAS] Agent 执行
```

---

## 统计数据

- 总技能数: 6
- 已萃取: 6
- 待萃取: 0
- 代码模板: 4

## 与上下游领域的衔接

| 上游领域 | 衔接点 | 数据流 |
|---------|--------|--------|
| 07-NLP-VOC | InstructUIE / BERT-SRL 抽取结果 | 结构化实体、关系、事件 |
| 10-MAS | Agent 执行与推理 | 图推理结果作为 Agent 上下文 |

---

> 最后更新: 2026-05-10
