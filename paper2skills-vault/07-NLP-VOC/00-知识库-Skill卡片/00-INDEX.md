---
name: skill-cards-index
description: 07-NLP-VOC 子项目 40 张 Skill 卡片的分类索引。按算法领域分组，标记 Phase 5 实际被采用的 Skill 与关联脚本。当需要查找算法出处、为新阶段选型参考、或向外展示知识库广度时使用。
title: Skill 卡片索引
doc_type: index
module: voc-nlp
topic: skill-cards-index
status: stable
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# Skill 卡片索引（40 张）

> **说明**：这些 Skill 卡片是从 ArXiv 论文转换的**算法参考库**，不是被执行的代码。它们构成了 Phase 5 系统设计的**思想来源**——每张卡片记录一篇论文的：核心算法、数学直觉、业务适用场景、Python 代码模板、与其他 Skill 的关联、ROI 评估。
>
> **使用方式**：做技术选型、向外讲述算法依据、为下一阶段找灵感时查阅。
>
> **⭐ 标记**：Phase 5 已采用或直接启发的 Skill。

## 按算法领域分类（8 组）

### A. 标签体系 × 弱监督（Phase 5 核心思想）

| Skill | 被 Phase 5 用于 |
|---|---|
| ⭐ [AutoTag-SelfEvolving-Label-System](Skill-AutoTag-SelfEvolving-Label-System.md) | **字典自进化闭环** — v3.9 → v4.0 的月度 cron 基础 |
| ⭐ [ALCHEmist-Weak-Supervision](Skill-ALCHEmist-Weak-Supervision.md) | **L0 规则层** — alchemist_label_generator.py 直接复用 |
| [TaxoAdapt-Taxonomy-Evolution](Skill-TaxoAdapt-Taxonomy-Evolution.md) | 字典层级演化（backlog 至 Phase 6） |
| [Active-Learning-Annotation](Skill-Active-Learning-Annotation.md) | **主动学习队列** — active_learning_queue.py 思想来源 |
| [OpenWorld-Class-Incremental-Learning](Skill-OpenWorld-Class-Incremental-Learning.md) | 月度开集采样 5% 的理论依据 |
| [AdaNEN-Streaming-Classifier](Skill-AdaNEN-Streaming-Classifier.md) | 增量打标（D13 全量重打时参考） |

### B. ABSA × 方面情感（Phase 5 L2 层）

| Skill | 被 Phase 5 用于 |
|---|---|
| ⭐ [ABSA-BERT-MoE](Skill-ABSA-BERT-MoE.md) | **L2 ABSA 抽取** — absa_extractor.py 的 prompt 设计基础 |
| [Aspect-Based-Sentiment-Analysis](Skill-Aspect-Based-Sentiment-Analysis.md) | ABSA 通用方法论 |
| [BERT-MoE高效方面情感分析](Skill-BERT-MoE高效方面情感分析.md) | 高效 ABSA 的中文版本 |
| [AGRS-属性引导评论摘要](Skill-AGRS-属性引导评论摘要.md) | aspect 引导的评论摘要（BI 看板备选） |

### C. 用户画像 × 行为识别（Phase 5 L3a 层）

| Skill | 被 Phase 5 用于 |
|---|---|
| ⭐ [PERSONABOT-RAG用户画像生成](Skill-PERSONABOT-RAG用户画像生成.md) | **55 原子画像标签的理论出处** |
| ⭐ [SoMeR-多视角用户表示](Skill-SoMeR-多视角用户表示.md) | **55 画像标签的 7 维度设计来源** |
| ⭐ [GPLR-人群标签生成](Skill-GPLR-人群标签生成.md) | 画像 label function 设计原型 |
| [Behavioral-Intent-Tree-Parsing](Skill-Behavioral-Intent-Tree-Parsing.md) | 行为意图树解析 |
| [Dialogue-to-Action-Graph](Skill-Dialogue-to-Action-Graph.md) | 对话到动作图转化 |
| [Product-Attribute-Graph-Parsing](Skill-Product-Attribute-Graph-Parsing.md) | 产品属性图解析 |

### D. NPS × 用户价值（Phase 5 L3b 层）

| Skill | 被 Phase 5 用于 |
|---|---|
| ⭐ [VOC-Proxy-NPS-AIPL-统一萃取引擎](Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md) | **Proxy NPS 三法投票的理论基础** |
| ⭐ [AIPL-VOC-Lifecycle-Tags](Skill-AIPL-VOC-Lifecycle-Tags.md) | AIPL 生命周期标签体系 |
| [NPS-Driver-Analysis](Skill-NPS-Driver-Analysis.md) | NPS 驱动因素分析（BI 看板用） |
| [Kano-需求分类与优先级](Skill-Kano-需求分类与优先级.md) | Kano 模型需求优先级 |
| [iReFeed-需求优先级排序](Skill-iReFeed-需求优先级排序.md) | 基于反馈的需求排序 |

### E. 跨语言 × 迁移

| Skill | 被 Phase 5 用于 |
|---|---|
| [CrossLingual-Semantic-Alignment](Skill-CrossLingual-Semantic-Alignment.md) | 多语言评论对齐（Trustpilot 德/法/西） |
| [CrossLingual-Sentiment-Transfer](Skill-CrossLingual-Sentiment-Transfer.md) | 跨语言情感迁移 |
| [Spiral-of-Silence-沉默少数派挖掘](Skill-Spiral-of-Silence-沉默少数派挖掘.md) | 少数派意见挖掘 |

### F. LLM Agent × 自进化

| Skill | 被 Phase 5 用于 |
|---|---|
| ⭐ [Self-Improving-LLM-Agent-Pipeline](Skill-Self-Improving-LLM-Agent-Pipeline.md) | **自进化 LLM Agent 管线**（Phase 6 重要参考） |
| [LLM-Personalized-Marketing-Copy-Generation](Skill-LLM-Personalized-Marketing-Copy-Generation.md) | LLM 生成个性化营销文案 |
| [Review-Quality-Scoring](Skill-Review-Quality-Scoring.md) | 评论质量评分 |
| [REVISION-无点击意图挖掘](Skill-REVISION-无点击意图挖掘.md) | 无点击意图挖掘 |
| [StaR-观点语句排序](Skill-StaR-观点语句排序.md) | 观点语句排序 |

### G. 多 Agent × 业务场景

| Skill | 被 Phase 5 用于 |
|---|---|
| [MAS-Consumer-Behavior-Simulation](Skill-MAS-Consumer-Behavior-Simulation.md) | 多 Agent 消费者行为模拟 |
| [MAS-MARL-Dynamic-Pricing](Skill-MAS-MARL-Dynamic-Pricing.md) | 多 Agent 强化学习动态定价 |
| [MAS-Multi-Objective-Recommendation](Skill-MAS-Multi-Objective-Recommendation.md) | 多 Agent 多目标推荐 |
| [MAS-VOC-Data-Analyst](Skill-MAS-VOC-Data-Analyst.md) | 多 Agent VOC 数据分析 |
| [TSCAN-上下文感知挽回策略](Skill-TSCAN-上下文感知挽回策略.md) | 上下文感知用户挽回 |
| [OfflineRL-触达时机优化](Skill-OfflineRL-触达时机优化.md) | 离线强化学习触达优化 |
| [TJAP-跨市场品类组合定价](Skill-TJAP-跨市场品类组合定价.md) | 跨市场品类定价 |

### H. 意见抽取 × 行动建议

| Skill | 被 Phase 5 用于 |
|---|---|
| ⭐ [MAA-行动建议生成](Skill-MAA-行动建议生成.md) | **MAA 行动建议算法**（Phase 6 策略包生成 P0） |
| [CSK-Customer-Sentiment-Clustering](Skill-CSK-Customer-Sentiment-Clustering.md) | 客户情感聚类 |
| [TopicImpact-观点单元画像抽取](Skill-TopicImpact-观点单元画像抽取.md) | 主题影响力观点抽取 |
| [VOC-Semantic-Blueprint](Skill-VOC-Semantic-Blueprint.md) | VOC 语义蓝图 |

---

## 按 Phase 5 直接采用情况分类

### 🟢 D1-D8 已直接采用（11 张）

| Skill | 采用位置 |
|---|---|
| [ALCHEmist-Weak-Supervision](Skill-ALCHEmist-Weak-Supervision.md) | L0 规则层 alchemist LFs |
| [AutoTag-SelfEvolving-Label-System](Skill-AutoTag-SelfEvolving-Label-System.md) | 字典自进化闭环思想 |
| [ABSA-BERT-MoE](Skill-ABSA-BERT-MoE.md) | L2 ABSA 抽取设计 |
| [PERSONABOT-RAG用户画像生成](Skill-PERSONABOT-RAG用户画像生成.md) | 55 画像标签出处 |
| [SoMeR-多视角用户表示](Skill-SoMeR-多视角用户表示.md) | 画像 7 维度设计 |
| [GPLR-人群标签生成](Skill-GPLR-人群标签生成.md) | 画像 LF 原型 |
| [VOC-Proxy-NPS-AIPL-统一萃取引擎](Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md) | Proxy NPS 三法投票 |
| [AIPL-VOC-Lifecycle-Tags](Skill-AIPL-VOC-Lifecycle-Tags.md) | AIPL 体系 |
| [Active-Learning-Annotation](Skill-Active-Learning-Annotation.md) | 主动学习队列 |
| [OpenWorld-Class-Incremental-Learning](Skill-OpenWorld-Class-Incremental-Learning.md) | 月度开集采样 |
| [Self-Improving-LLM-Agent-Pipeline](Skill-Self-Improving-LLM-Agent-Pipeline.md) | 自进化管线设计 |

### 🟡 Phase 6 规划采用（3 张）

| Skill | 规划采用 |
|---|---|
| [MAA-行动建议生成](Skill-MAA-行动建议生成.md) | BI 看板策略包生成 P0 |
| [TaxoAdapt-Taxonomy-Evolution](Skill-TaxoAdapt-Taxonomy-Evolution.md) | 字典层级演化 |
| [AdaNEN-Streaming-Classifier](Skill-AdaNEN-Streaming-Classifier.md) | 增量流式打标 |

### 🔵 参考价值 / 未采用（26 张）

其余卡片保持参考价值，作为 Phase 7+ 可能的技术方向或对外展示团队技术广度。

---

## 统计

- **总数**：40 张
- **Phase 5 D1-D8 已采用**：11 张 = 27.5%
- **Phase 6 规划采用**：3 张 = 7.5%
- **待激活**：26 张 = 65%
- **总代码模板**：40 个对应 Python 模块（在 `paper2skills-code/nlp_voc/` 下）

---

> **查找其他领域算法**：除 NLP-VOC 外，项目还有因果推断（01-）、A/B 实验（02-）、时间序列（03-）、供应链（04-）、推荐（05-）、增长（06-）等 6 个并列域的 Skill 库，位于同级 `paper2skills-vault/` 下。本索引只覆盖 NLP-VOC 域。
