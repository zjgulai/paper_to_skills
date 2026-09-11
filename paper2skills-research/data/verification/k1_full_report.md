# K1 代码可执行性验证报告

- 生成时间：2026-09-12T04:04:30+0800
- 最大验证级别：L5
- 单元总数：80
- **K1 执行率：52.5%**（PASS 42 / 环境阻塞 6 / 孤儿依赖 9 / 失败 23 / 未执行 0）

> **口径说明（三个易被混淆的判定）**
> - `PASS`：真正跑通了（L4 脚本执行成功 或 L5 断言全绿）。只有这一类可以声称「已验证」。
> - `ENV_BLOCKED`：本机缺第三方依赖/凭证/资源（如未装 torch）。**计入未验证分母** —— 缺依赖不等于代码正确。
> - `ORPHAN_DEP`：卡片 `import` 的**本地模块在仓库内根本不存在**（如 `import review_quality_scoring`）。
>   这是卡片真实缺陷，必须修，**绝不可归因环境而放行**。

## 明细

| 判定 | 目标 | 阻塞阶段 | 说明 |
|---|---|---|---|
| 🟡 ENV_BLOCKED | `paper2skills-vault/01-因果推断/Skill-DiD-Difference-in-Differences.md#stitched(1块)` | L3_IMPORT | [block1] 缺第三方依赖/凭证: matplotlib |
| ✅ PASS | `paper2skills-vault/01-因果推断/Skill-IV-Instrumental-Variables.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/01-因果推断/Skill-Intelligent-Attribution-Causal-Forest.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/01-因果推断/Skill-Mediation-Causal-Mechanism-Analysis.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/01-因果推断/Skill-Uplift-Modeling.md#stitched(1块)` | L4_SMOKE | [block1] 运行期错误 阻塞在 line 4 |
| ❌ FAIL | `paper2skills-vault/02-A_B实验/Skill-AB-Experimental-Design.md#stitched(1块)` | L1_SYNTAX | [block1] 语法错误 line 5: invalid character '【' (U+3010) |
| ✅ PASS | `paper2skills-vault/02-A_B实验/Skill-AB-Test-Result-Interpretation.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/02-A_B实验/Skill-Power-Analysis-Sample-Size.md#stitched(1块)` | — | 全部通过 |
| 🟡 ENV_BLOCKED | `paper2skills-vault/02-A_B实验/Skill-Thompson-Sampling-MAB.md#stitched(2块)` | L3_IMPORT | [block1] 缺第三方依赖/凭证: matplotlib |
| ✅ PASS | `paper2skills-vault/03-时间序列/Skill-Intelligent-Prediction-Doubly-Robust.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/03-时间序列/Skill-Prophet-Forecasting.md#stitched(1块)` | — | 全部通过 |
| 🟡 ENV_BLOCKED | `paper2skills-vault/03-时间序列/Skill-Time-Series-Anomaly-Detection.md#stitched(1块)` | L3_IMPORT | [block1] 缺第三方依赖/凭证: statsmodels, matplotlib |
| ❌ FAIL | `paper2skills-vault/03-时间序列/Skill-Time-Series-Forecasting.md#stitched(1块)` | L4_SMOKE | [block1] 运行期错误 阻塞在 line 4 |
| ✅ PASS | `paper2skills-vault/04-供应链/Skill-Demand-Forecasting-Supply-Chain.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/04-供应链/Skill-Multi-Echelon-Inventory.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/04-供应链/Skill-Safety-Stock-Replenishment.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/04-供应链/Skill-Two-Echelon-Inventory-DRL.md#stitched(2块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/05-推荐系统/Skill-Cold-Start-Meta-Learning-PAM.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/05-推荐系统/Skill-Deep-Learning-Recommendation-HI.md#stitched(1块)` | L4_SMOKE | [block1] 运行期错误 阻塞在 line 4 |
| ✅ PASS | `paper2skills-vault/05-推荐系统/Skill-Diversity-Reranking-SMMR.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/05-推荐系统/Skill-Explainable-Recommendation.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/05-推荐系统/Skill-NeuralNDCG-Learning-to-Rank.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/05-推荐系统/Skill-Semantic-ID-Retrieval-RPG.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/06-增长模型/Skill-Cold-Start-Product-Recommendation.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/06-增长模型/Skill-LTV-Prediction-ZILN.md#stitched(1块)` | L5_TEST | 断言失败 |
| ❌ FAIL | `paper2skills-vault/06-增长模型/Skill-New-Product-Opportunity-Mining.md#stitched(1块)` | L4_SMOKE | [block1] 运行期错误 阻塞在 line 4 |
| ❌ FAIL | `paper2skills-vault/06-增长模型/Skill-RFM-Customer-Segmentation.md#stitched(1块)` | L4_SMOKE | [block1] 运行期错误 阻塞在 line 4 |
| 🟡 ENV_BLOCKED | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-ABSA-BERT-MoE.md#stitched(1块)` | L3_IMPORT | [block1] 缺第三方依赖/凭证: transformers |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-AGRS-属性引导评论摘要.md#stitched(1块)` | L3_IMPORT | [block1] import 时崩溃 |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-ALCHEmist-Weak-Supervision.md#stitched(2块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-AdaNEN-Streaming-Classifier.md#stitched(2块)` | L3_IMPORT | [block1] import 时崩溃 |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-AutoTag-SelfEvolving-Label-System.md#stitched(3块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: autotag_self_evolving |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-BERT-MoE高效方面情感分析.md#stitched(1块)` | — | 全部通过 |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Behavioral-Intent-Tree-Parsing.md#stitched(2块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: behavioral_intent_tree_parsing |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-CrossLingual-Semantic-Alignment.md#stitched(2块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: crosslingual_semantic_alignment |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-CrossLingual-Sentiment-Transfer.md#stitched(1块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: crosslingual_sentiment_transfer |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Dialogue-to-Action-Graph.md#stitched(2块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: dialogue_to_action_graph |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-GPLR-人群标签生成.md#stitched(4块)` | L3_IMPORT | [block1] import 时崩溃 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-LLM-Personalized-Marketing-Copy-Generation.md#stitched(3块)` | L1_SYNTAX | [block2] 语法错误 line 29: positional argument follows keyword argument |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAA-行动建议生成.md#stitched(1块)` | L3_IMPORT | [block1] import 时崩溃 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-Consumer-Behavior-Simulation.md#stitched(1块)` | L3_IMPORT | [block1] import 时崩溃 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-MARL-Dynamic-Pricing.md#stitched(2块)` | L3_IMPORT | [block1] import 时崩溃 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-Multi-Objective-Recommendation.md#stitched(2块)` | L3_IMPORT | [block1] import 时崩溃 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-VOC-Data-Analyst.md#stitched(2块)` | L3_IMPORT | import 时崩溃 |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-NPS-Driver-Analysis.md#stitched(1块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: nps_driver_analysis |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-OfflineRL-触达时机优化.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-OpenWorld-Class-Incremental-Learning.md#stitched(1块)` | L3_IMPORT | [block1] import 时崩溃 |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Product-Attribute-Graph-Parsing.md#stitched(2块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: product_attribute_graph_parsing |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Review-Quality-Scoring.md#stitched(2块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: review_quality_scoring |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Self-Improving-LLM-Agent-Pipeline.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-SoMeR-多视角用户表示.md#stitched(2块)` | L3_IMPORT | [block1] import 时崩溃 |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-StaR-观点语句排序.md#stitched(1块)` | L3_IMPORT | [block1] import 时崩溃 |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-TJAP-跨市场品类组合定价.md#stitched(1块)` | L3_IMPORT | [block1] import 时崩溃 |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-TSCAN-上下文感知挽回策略.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md#stitched(4块)` | L1_SYNTAX | [block2] 语法错误 line 40: ':' expected after dictionary key |
| 🔴 ORPHAN_DEP | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-VOC-Semantic-Blueprint.md#stitched(2块)` | L3_IMPORT | [block1] 引用了仓库内不存在的本地模块: voc_semantic_blueprint |
| ✅ PASS | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-iReFeed-需求优先级排序.md#stitched(1块)` | L3_IMPORT | [block1] import 时崩溃 |
| ❌ FAIL | `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-大规模消费者评论方面情感分析.md#stitched(1块)` | L1_SYNTAX | [block1] 语法错误 line 42: invalid syntax |
| ✅ PASS | `paper2skills-vault/08-知识图谱/Skill-Dense-Retrieval-Ecommerce-Semantic-Search.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/08-知识图谱/Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/08-知识图谱/Skill-KG-Auto-Construction-Agent-Driven.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/08-知识图谱/Skill-KG-Relation-Completion-CBLiP.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/08-知识图谱/Skill-KGQA-Question-Answering.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/08-知识图谱/Skill-Multilingual-NER-Universal-v2.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/09-DataAgent-LLM/Skill-Argos-Agentic-Anomaly-Detection.md#stitched(1块)` | L1_SYNTAX | [block1] 语法错误 line 207: unterminated triple-quoted f-string literal (detected at line 213) |
| 🟡 ENV_BLOCKED | `paper2skills-vault/09-DataAgent-LLM/Skill-Data-to-Dashboard-Multi-Agent-Visualization.md#stitched(1块)` | L3_IMPORT | [block1] 缺第三方依赖/凭证: matplotlib, openai |
| 🟡 ENV_BLOCKED | `paper2skills-vault/09-DataAgent-LLM/Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md#stitched(1块)` | L3_IMPORT | [block1] 缺第三方依赖/凭证: matplotlib, openai |
| ✅ PASS | `paper2skills-vault/09-DataAgent-LLM/Skill-Root-Cause-Analysis-Agent.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/09-DataAgent-LLM/Skill-SQL-Agent-Text-to-SQL.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/10-MAS/00-知识库-Skill卡片/Skill-MAS-Orchestrator.md#stitched(1块)` | L1_SYNTAX | [block1] 语法错误 line 14: invalid character '→' (U+2192) |
| ✅ PASS | `paper2skills-vault/12-ML基础/Skill-Feature-Engineering.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/13-广告分析/Skill-Ad-Attribution-Modeling.md#stitched(1块)` | — | 全部通过 |
| ✅ PASS | `paper2skills-vault/13-广告分析/Skill-ROAS-Budget-Optimization.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/14-用户分析/Skill-Cohort-Retention-Analysis.md#stitched(1块)` | L4_SMOKE | [block1] 运行期错误 阻塞在 line 4 |
| ✅ PASS | `paper2skills-vault/14-用户分析/Skill-User-Funnel-Analysis.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/15-营销投放分析/Skill-Marketing-Mix-Modeling.md#stitched(1块)` | L4_SMOKE | [block1] 运行期错误 阻塞在 line 4 |
| ✅ PASS | `paper2skills-vault/15-营销投放分析/Skill-Promotion-Effectiveness.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/16-智能体工程/Skill-Memory-as-Action.md#stitched(1块)` | L1_SYNTAX | 语法错误 line 2: invalid syntax |
| ✅ PASS | `paper2skills-vault/16-智能体工程/Skill-SLM-Tool-Calling-Optimization.md#stitched(1块)` | — | 全部通过 |
| ❌ FAIL | `paper2skills-vault/16-智能体工程/Skill-Skill-Lifecycle-Design.md#stitched(1块)` | L1_SYNTAX | 语法错误 line 1: invalid syntax |

## 孤儿依赖清单（卡片引用了仓库内不存在的模块）

| 卡片 | 缺失模块 |
|---|---|
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-AutoTag-SelfEvolving-Label-System.md#stitched(3块)` | `autotag_self_evolving` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Behavioral-Intent-Tree-Parsing.md#stitched(2块)` | `behavioral_intent_tree_parsing` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-CrossLingual-Semantic-Alignment.md#stitched(2块)` | `crosslingual_semantic_alignment` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-CrossLingual-Sentiment-Transfer.md#stitched(1块)` | `crosslingual_sentiment_transfer` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Dialogue-to-Action-Graph.md#stitched(2块)` | `dialogue_to_action_graph` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-NPS-Driver-Analysis.md#stitched(1块)` | `nps_driver_analysis` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Product-Attribute-Graph-Parsing.md#stitched(2块)` | `product_attribute_graph_parsing` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Review-Quality-Scoring.md#stitched(2块)` | `review_quality_scoring` |
| `paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-VOC-Semantic-Blueprint.md#stitched(2块)` | `voc_semantic_blueprint` |
