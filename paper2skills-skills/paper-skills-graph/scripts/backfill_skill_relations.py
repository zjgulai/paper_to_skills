#!/usr/bin/env python3
"""
backfill_skill_relations.py — 一次性批量为孤立 Skill 卡片补充"技能关联"模块。

输入: 工单 isolated-skills-backfill-checklist-20260517.md（硬编码在 RECOMMENDATIONS 字典中）
输出: 直接编辑 paper2skills-vault/ 下的 Skill-*.md 文件
策略:
  - 若卡片已有 ④ 技能关联章节, 替换该章节
  - 若卡片缺 ④ 技能关联章节, 在 ⑤ 商业价值之前插入; 若也缺 ⑤, 追加到文件末尾

操作前自动备份每张卡到 .bak 副本以便回滚。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional


BASE = Path(os.environ.get("PAPER2SKILLS_ROOT") or Path(__file__).resolve().parents[3])
VAULT = BASE / "paper2skills-vault"


RECOMMENDATIONS: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "05-推荐系统/Skill-Matrix-Factorization.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "矩阵分解前需要对评分稀疏矩阵做基础特征预处理（缺失值、归一化）"}],
        "延伸": [
            {"name": "Skill-NeuralNDCG-Learning-to-Rank", "domain": "05-推荐系统", "reason": "在 MF 召回后做精排，提升 Top-N 推荐质量"},
            {"name": "Skill-Cold-Start-Meta-Learning-PAM", "domain": "05-推荐系统", "reason": "解决 MF 对新用户/新商品的冷启动短板"},
        ],
        "可组合": [{"name": "Skill-Deep-Learning-Recommendation-HI", "domain": "05-推荐系统", "reason": "MF 隐因子可作为深度推荐网络的初始 embedding"}],
    },
    "05-推荐系统/Skill-Cold-Start-Meta-Learning-PAM.md": {
        "前置": [{"name": "Skill-Matrix-Factorization", "domain": "05-推荐系统", "reason": "理解隐因子学习是元学习初始化的基础"}],
        "延伸": [{"name": "Skill-Explainable-Recommendation", "domain": "05-推荐系统", "reason": "冷启动后向用户说明『为什么推荐』提升信任"}],
        "可组合": [{"name": "Skill-Cold-Start-Product-Recommendation", "domain": "05-推荐系统", "reason": "新品上架的端到端冷启动管线"}],
    },
    "05-推荐系统/Skill-Cold-Start-Product-Recommendation.md": {
        "前置": [{"name": "Skill-Cold-Start-Meta-Learning-PAM", "domain": "05-推荐系统", "reason": "元学习是冷启动 SKU 推荐的核心方法学"}],
        "延伸": [{"name": "Skill-New-Product-Opportunity-Mining", "domain": "06-增长模型", "reason": "冷启动推荐效果反哺新品机会挖掘"}],
        "可组合": [{"name": "Skill-Matrix-Factorization", "domain": "05-推荐系统", "reason": "热启动 SKU 用 MF，冷启动 SKU 用元学习，二者互补"}],
    },
    "05-推荐系统/Skill-NeuralNDCG-Learning-to-Rank.md": {
        "前置": [{"name": "Skill-Matrix-Factorization", "domain": "05-推荐系统", "reason": "L2R 的输入特征通常包含 MF 隐因子"}],
        "延伸": [{"name": "Skill-Diversity-Reranking-SMMR", "domain": "05-推荐系统", "reason": "NDCG 排序后做多样性重排"}],
        "可组合": [{"name": "Skill-Session-Based-Recommendation-SR-GNN", "domain": "05-推荐系统", "reason": "Session-based 召回 + L2R 精排"}],
    },
    "05-推荐系统/Skill-Diversity-Reranking-SMMR.md": {
        "前置": [{"name": "Skill-NeuralNDCG-Learning-to-Rank", "domain": "05-推荐系统", "reason": "重排建立在已有排序结果之上"}],
        "延伸": [{"name": "Skill-Explainable-Recommendation", "domain": "05-推荐系统", "reason": "解释为何选取多样性最大化的子集"}],
        "可组合": [{"name": "Skill-Semantic-ID-Retrieval-RPG", "domain": "05-推荐系统", "reason": "语义 ID 召回 + SMMR 多样性重排"}],
    },
    "05-推荐系统/Skill-Explainable-Recommendation.md": {
        "前置": [
            {"name": "Skill-NeuralNDCG-Learning-to-Rank", "domain": "05-推荐系统", "reason": "排序模型是解释性推荐的基础对象"},
            {"name": "Skill-Matrix-Factorization", "domain": "05-推荐系统", "reason": "隐因子是解释性归因的常用维度"},
        ],
        "延伸": [],
        "可组合": [{"name": "Skill-Knowledge-Graph-for-Skills-Management", "domain": "08-知识图谱", "reason": "KG 路径提供天然的解释性推理链"}],
    },
    "05-推荐系统/Skill-Semantic-ID-Retrieval-RPG.md": {
        "前置": [{"name": "Skill-Matrix-Factorization", "domain": "05-推荐系统", "reason": "理解 embedding 的语义检索基础"}],
        "延伸": [{"name": "Skill-Diversity-Reranking-SMMR", "domain": "05-推荐系统", "reason": "语义召回后做多样性重排"}],
        "可组合": [
            {"name": "Skill-Session-Based-Recommendation-SR-GNN", "domain": "05-推荐系统", "reason": "session 序列与语义 ID 联合召回"},
            {"name": "Skill-Dense-Retrieval-Ecommerce-Semantic-Search", "domain": "08-知识图谱", "reason": "搜索-推荐共享语义索引"},
        ],
    },
    "05-推荐系统/Skill-Session-Based-Recommendation-SR-GNN.md": {
        "前置": [
            {"name": "Skill-Matrix-Factorization", "domain": "05-推荐系统", "reason": "GNN 推荐的隐因子初始化常用 MF 结果"},
            {"name": "Skill-HGT-Heterogeneous-Graph-Transformer", "domain": "08-知识图谱", "reason": "异构图结构是 SR-GNN 的方法学基础"},
        ],
        "延伸": [{"name": "Skill-NeuralNDCG-Learning-to-Rank", "domain": "05-推荐系统", "reason": "session 召回后用 L2R 精排"}],
        "可组合": [{"name": "Skill-Semantic-ID-Retrieval-RPG", "domain": "05-推荐系统", "reason": "session 序列与语义 ID 双路召回"}],
    },
    "02-A_B实验/Skill-AB-Experimental-Design.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "实验前需要明确指标定义与维度切分（特征化）"}],
        "延伸": [
            {"name": "Skill-AB-Test-Result-Interpretation", "domain": "02-A_B实验", "reason": "实验设计后接结果解读"},
            {"name": "Skill-Power-Analysis-Sample-Size", "domain": "02-A_B实验", "reason": "实验设计的核心步骤之一是样本量估算"},
        ],
        "可组合": [
            {"name": "Skill-Multi-Armed-Bandit", "domain": "02-A_B实验", "reason": "传统 A/B 与 MAB 是流量分配的两种范式"},
            {"name": "Skill-Thompson-Sampling-MAB", "domain": "02-A_B实验", "reason": "贝叶斯化 A/B 与 Thompson 采样互补"},
        ],
    },
    "02-A_B实验/Skill-AB-Test-Result-Interpretation.md": {
        "前置": [
            {"name": "Skill-AB-Experimental-Design", "domain": "02-A_B实验", "reason": "解读建立在严谨的实验设计之上"},
            {"name": "Skill-Power-Analysis-Sample-Size", "domain": "02-A_B实验", "reason": "样本量决定结果置信度"},
        ],
        "延伸": [{"name": "Skill-Intelligent-Attribution-Causal-Forest", "domain": "01-因果推断", "reason": "结果解读后做异质性归因（CATE）"}],
        "可组合": [{"name": "Skill-Uplift-Modeling", "domain": "01-因果推断", "reason": "Uplift 与 A/B 实验的因果效应估计互补"}],
    },
    "01-因果推断/Skill-Causal-Discovery-PC-Algorithm.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "PC 算法的输入是变量集合，需要先做特征筛选"}],
        "延伸": [
            {"name": "Skill-Intelligent-Attribution-Causal-Forest", "domain": "01-因果推断", "reason": "因果发现后估计因果效应"},
            {"name": "Skill-Mediation-Causal-Mechanism-Analysis", "domain": "01-因果推断", "reason": "因果发现后做中介机制分析"},
        ],
        "可组合": [{"name": "Skill-Uplift-Modeling", "domain": "01-因果推断", "reason": "因果图指导 Uplift 特征选择"}],
    },
    "06-增长模型/Skill-Deep-Learning-Churn-Prediction.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "深度流失预测高度依赖特征工程"}],
        "延伸": [{"name": "Skill-Uplift-Churn-Prediction", "domain": "06-增长模型", "reason": "从预测谁会流失升级到预测干预谁能挽留"}],
        "可组合": [{"name": "Skill-DQN-Purchase-Prediction", "domain": "06-增长模型", "reason": "流失预测 + 购买预测形成完整生命周期决策"}],
    },
    "06-增长模型/Skill-RFM-Customer-Segmentation.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "RFM 三维度本身是特征工程产物"}],
        "延伸": [{"name": "Skill-LTV-Prediction-ZILN", "domain": "06-增长模型", "reason": "RFM 分群是 LTV 预测的常用先验"}],
        "可组合": [
            {"name": "Skill-Cohort-Retention-Analysis", "domain": "14-用户分析", "reason": "RFM 分群后看每群的留存曲线"},
            {"name": "Skill-User-Funnel-Analysis", "domain": "14-用户分析", "reason": "RFM 分群后对比各群的漏斗转化"},
        ],
    },
    "06-增长模型/Skill-New-Product-Opportunity-Mining.md": {
        "前置": [{"name": "Skill-RFM-Customer-Segmentation", "domain": "06-增长模型", "reason": "新品机会挖掘需要先识别目标客群"}],
        "延伸": [{"name": "Skill-Cold-Start-Product-Recommendation", "domain": "05-推荐系统", "reason": "新品上线后接冷启动推荐"}],
        "可组合": [{"name": "Skill-Knowledge-Graph-for-Skills-Management", "domain": "08-知识图谱", "reason": "用 KG 在已有品类上发现机会缺口"}],
    },
    "14-用户分析/Skill-User-Funnel-Analysis.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "漏斗节点定义本质是事件特征构造"}],
        "延伸": [
            {"name": "Skill-Cohort-Retention-Analysis", "domain": "14-用户分析", "reason": "漏斗看转化，留存看持续，互补维度"},
            {"name": "Skill-RFM-Customer-Segmentation", "domain": "06-增长模型", "reason": "漏斗各步可结合 RFM 切群分析"},
        ],
        "可组合": [{"name": "Skill-Customer-Churn-Prediction", "domain": "06-增长模型", "reason": "漏斗流失节点是流失预测的关键输入"}],
    },
    "14-用户分析/Skill-Cohort-Retention-Analysis.md": {
        "前置": [{"name": "Skill-User-Funnel-Analysis", "domain": "14-用户分析", "reason": "漏斗分析是留存分析的姊妹方法"}],
        "延伸": [
            {"name": "Skill-RFM-Customer-Segmentation", "domain": "06-增长模型", "reason": "对各 cohort 进一步做 RFM 分群"},
            {"name": "Skill-LTV-Prediction-ZILN", "domain": "06-增长模型", "reason": "cohort 留存曲线是 LTV 模型核心输入"},
        ],
        "可组合": [{"name": "Skill-Customer-Churn-Prediction", "domain": "06-增长模型", "reason": "Cohort 留存指标定义流失阈值"}],
    },
    "15-营销投放分析/Skill-Marketing-Mix-Modeling.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "MMM 输入需要 adstock/saturation 特征转换"}],
        "延伸": [
            {"name": "Skill-Promotion-Effectiveness", "domain": "15-营销投放分析", "reason": "MMM 渠道层结果驱动促销因果验证"},
            {"name": "Skill-ROAS-Budget-Optimization", "domain": "13-广告分析", "reason": "MMM 估计的渠道弹性指导预算优化"},
        ],
        "可组合": [{"name": "Skill-Ad-Attribution-Modeling", "domain": "13-广告分析", "reason": "MMM 长周期 + MTA 单次归因互补"}],
    },
    "15-营销投放分析/Skill-Promotion-Effectiveness.md": {
        "前置": [
            {"name": "Skill-Marketing-Mix-Modeling", "domain": "15-营销投放分析", "reason": "MMM 提供渠道基线，促销在此基线上叠加"},
            {"name": "Skill-Intelligent-Prediction-Doubly-Robust", "domain": "03-时间序列", "reason": "DR 估计是促销因果效应的核心方法"},
        ],
        "延伸": [{"name": "Skill-Monodense-单品价格弹性估计", "domain": "04-供应链", "reason": "促销下沉到 SKU 级别价格弹性"}],
        "可组合": [{"name": "Skill-Ad-Attribution-Modeling", "domain": "13-广告分析", "reason": "归因 + 促销因果联合给出 ROI 全景"}],
    },
    "13-广告分析/Skill-Ad-Attribution-Modeling.md": {
        "前置": [{"name": "Skill-Intelligent-Attribution-Causal-Forest", "domain": "01-因果推断", "reason": "因果森林为归因提供反事实基础"}],
        "延伸": [{"name": "Skill-ROAS-Budget-Optimization", "domain": "13-广告分析", "reason": "归因结果驱动预算分配优化"}],
        "可组合": [{"name": "Skill-Marketing-Mix-Modeling", "domain": "15-营销投放分析", "reason": "渠道归因 + MMM 形成短长期视角统一"}],
    },
    "13-广告分析/Skill-ROAS-Budget-Optimization.md": {
        "前置": [{"name": "Skill-Ad-Attribution-Modeling", "domain": "13-广告分析", "reason": "归因结果是预算优化的输入"}],
        "延伸": [{"name": "Skill-Promotion-Effectiveness", "domain": "15-营销投放分析", "reason": "预算优化后看促销因果增量"}],
        "可组合": [{"name": "Skill-Marketing-Mix-Modeling", "domain": "15-营销投放分析", "reason": "MMM 弹性曲线为优化提供约束"}],
    },
    "08-知识图谱/Skill-Multilingual-NER-Universal-v2.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "多语 NER 训练前需要语料预处理"}],
        "延伸": [
            {"name": "Skill-KG-Auto-Construction-Agent-Driven", "domain": "08-知识图谱", "reason": "NER 是 KG 自动构建的实体抽取入口"},
            {"name": "Skill-KG-Relation-Completion-CBLiP", "domain": "08-知识图谱", "reason": "NER 抽实体后做关系补全"},
        ],
        "可组合": [{"name": "Skill-GraphRAG-Knowledge-Enhanced-Retrieval", "domain": "08-知识图谱", "reason": "NER 实体作为 GraphRAG 检索锚点"}],
    },
    "08-知识图谱/Skill-KG-Relation-Completion-CBLiP.md": {
        "前置": [
            {"name": "Skill-Multilingual-NER-Universal-v2", "domain": "08-知识图谱", "reason": "实体识别是关系补全的前置"},
            {"name": "Skill-Knowledge-Graph-for-Skills-Management", "domain": "08-知识图谱", "reason": "理解 KG schema 是关系建模的基础"},
        ],
        "延伸": [{"name": "Skill-KGQA-Question-Answering", "domain": "08-知识图谱", "reason": "完整 KG 是 KGQA 的查询底座"}],
        "可组合": [{"name": "Skill-GraphRAG-Knowledge-Enhanced-Retrieval", "domain": "08-知识图谱", "reason": "补全后的 KG 提升 GraphRAG 检索质量"}],
    },
    "08-知识图谱/Skill-KGQA-Question-Answering.md": {
        "前置": [
            {"name": "Skill-Knowledge-Graph-for-Skills-Management", "domain": "08-知识图谱", "reason": "KG schema 是 KGQA 的查询语义基础"},
            {"name": "Skill-Dense-Retrieval-Ecommerce-Semantic-Search", "domain": "08-知识图谱", "reason": "稠密检索定位相关子图"},
        ],
        "延伸": [{"name": "Skill-GraphRAG-Knowledge-Enhanced-Retrieval", "domain": "08-知识图谱", "reason": "KGQA + RAG 形成知识增强问答"}],
        "可组合": [{"name": "Skill-SQL-Agent-Text-to-SQL", "domain": "09-DataAgent-LLM", "reason": "KGQA 与 Text-to-SQL 共同覆盖结构化问答"}],
    },
    "09-DataAgent-LLM/Skill-SQL-Agent-Text-to-SQL.md": {
        "前置": [{"name": "Skill-ReAct-Reasoning-Acting", "domain": "10-MAS", "reason": "SQL Agent 的推理-执行循环建立在 ReAct 范式上"}],
        "延伸": [
            {"name": "Skill-Data-to-Dashboard-Multi-Agent-Visualization", "domain": "09-DataAgent-LLM", "reason": "SQL 结果直接驱动可视化"},
            {"name": "Skill-Root-Cause-Analysis-Agent", "domain": "09-DataAgent-LLM", "reason": "SQL Agent 是 RCA 的查询底层"},
        ],
        "可组合": [{"name": "Skill-DeepAnalyze-Autonomous-Data-Science-Agent", "domain": "09-DataAgent-LLM", "reason": "SQL Agent 是自治数据科学 Agent 的工具节点"}],
    },
    "09-DataAgent-LLM/Skill-Argos-Agentic-Anomaly-Detection.md": {
        "前置": [{"name": "Skill-Time-Series-Anomaly-Detection", "domain": "03-时间序列", "reason": "传统异常检测算法是 Agent 推理的方法基础"}],
        "延伸": [{"name": "Skill-Root-Cause-Analysis-Agent", "domain": "09-DataAgent-LLM", "reason": "异常检测后接根因分析"}],
        "可组合": [{"name": "Skill-DeepAnalyze-Autonomous-Data-Science-Agent", "domain": "09-DataAgent-LLM", "reason": "异常检测嵌入数据科学 Agent 工作流"}],
    },
    "09-DataAgent-LLM/Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md": {
        "前置": [
            {"name": "Skill-SQL-Agent-Text-to-SQL", "domain": "09-DataAgent-LLM", "reason": "SQL Agent 是数据查询底层工具"},
            {"name": "Skill-ReAct-Reasoning-Acting", "domain": "10-MAS", "reason": "Agent 推理范式基础"},
        ],
        "延伸": [{"name": "Skill-Root-Cause-Analysis-Agent", "domain": "09-DataAgent-LLM", "reason": "数据科学 Agent 升级为 RCA Agent"}],
        "可组合": [{"name": "Skill-Data-to-Dashboard-Multi-Agent-Visualization", "domain": "09-DataAgent-LLM", "reason": "分析结果交由可视化 Agent 呈现"}],
    },
    "09-DataAgent-LLM/Skill-Root-Cause-Analysis-Agent.md": {
        "前置": [
            {"name": "Skill-Argos-Agentic-Anomaly-Detection", "domain": "09-DataAgent-LLM", "reason": "RCA 由异常检测触发"},
            {"name": "Skill-SQL-Agent-Text-to-SQL", "domain": "09-DataAgent-LLM", "reason": "RCA 需要查询多维数据切片"},
        ],
        "延伸": [{"name": "Skill-DeepAnalyze-Autonomous-Data-Science-Agent", "domain": "09-DataAgent-LLM", "reason": "RCA 是自治数据科学的核心场景"}],
        "可组合": [{"name": "Skill-Multi-Agent-Debate", "domain": "10-MAS", "reason": "多 Agent 辩论提升 RCA 假设质量"}],
    },
    "09-DataAgent-LLM/Skill-Data-to-Dashboard-Multi-Agent-Visualization.md": {
        "前置": [{"name": "Skill-SQL-Agent-Text-to-SQL", "domain": "09-DataAgent-LLM", "reason": "可视化的数据来源依赖 SQL Agent"}],
        "延伸": [{"name": "Skill-DeepAnalyze-Autonomous-Data-Science-Agent", "domain": "09-DataAgent-LLM", "reason": "可视化嵌入自治数据分析 Agent"}],
        "可组合": [{"name": "Skill-MAS-Orchestrator", "domain": "10-MAS", "reason": "可视化作为多 Agent 编排的最终输出环节"}],
    },
    "03-时间序列/Skill-Prophet-Forecasting.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "Prophet 输入需要节假日/季节特征"}],
        "延伸": [
            {"name": "Skill-Time-Series-Forecasting", "domain": "03-时间序列", "reason": "Prophet 是时间序列预测的入门方法"},
            {"name": "Skill-Temporal-Fusion-Transformer", "domain": "03-时间序列", "reason": "TFT 是 Prophet 的深度学习升级"},
        ],
        "可组合": [{"name": "Skill-Demand-Forecasting-Supply-Chain", "domain": "04-供应链", "reason": "Prophet 预测结果直接驱动备货"}],
    },
    "03-时间序列/Skill-Temporal-Fusion-Transformer.md": {
        "前置": [
            {"name": "Skill-Prophet-Forecasting", "domain": "03-时间序列", "reason": "理解经典时序模型再进入深度方法"},
            {"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "TFT 多源特征需要前置工程化"},
        ],
        "延伸": [{"name": "Skill-Time-Series-Anomaly-Detection", "domain": "03-时间序列", "reason": "TFT 残差可用于异常检测"}],
        "可组合": [{"name": "Skill-Demand-Forecasting-Supply-Chain", "domain": "04-供应链", "reason": "TFT 高精度预测落地到库存计划"}],
    },
    "04-供应链/Skill-Demand-Forecasting-Supply-Chain.md": {
        "前置": [
            {"name": "Skill-Prophet-Forecasting", "domain": "03-时间序列", "reason": "Prophet 是供应链需求预测的常用基线"},
            {"name": "Skill-Temporal-Fusion-Transformer", "domain": "03-时间序列", "reason": "TFT 提供更精细的需求曲线"},
        ],
        "延伸": [
            {"name": "Skill-Safety-Stock-Replenishment", "domain": "04-供应链", "reason": "需求预测下游接安全库存计算"},
            {"name": "Skill-Two-Echelon-Inventory-DRL", "domain": "04-供应链", "reason": "需求预测驱动多级库存优化"},
        ],
        "可组合": [{"name": "Skill-Monodense-单品价格弹性估计", "domain": "04-供应链", "reason": "弹性 + 需求预测联合定价决策"}],
    },
    "04-供应链/Skill-Safety-Stock-Replenishment.md": {
        "前置": [{"name": "Skill-Demand-Forecasting-Supply-Chain", "domain": "04-供应链", "reason": "安全库存计算依赖需求预测分布"}],
        "延伸": [{"name": "Skill-Two-Echelon-Inventory-DRL", "domain": "04-供应链", "reason": "单层安全库存升级为多层 DRL 决策"}],
        "可组合": [{"name": "Skill-Monodense-单品价格弹性估计", "domain": "04-供应链", "reason": "弹性影响最优库存水位"}],
    },
    "04-供应链/Skill-Monodense-单品价格弹性估计.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "弹性估计需要价格、销量、促销等特征工程"}],
        "延伸": [
            {"name": "Skill-ROAS-Budget-Optimization", "domain": "13-广告分析", "reason": "弹性 + 广告 ROI 形成定价-投放联合优化"},
            {"name": "Skill-Promotion-Effectiveness", "domain": "15-营销投放分析", "reason": "弹性是促销效果分析的微观基础"},
        ],
        "可组合": [{"name": "Skill-Marketing-Mix-Modeling", "domain": "15-营销投放分析", "reason": "弹性是 MMM 的关键输入参数"}],
    },
    "12-ML基础/Skill-Feature-Engineering.md": {
        "前置": [],
        "延伸": [
            {"name": "Skill-Causal-Discovery-PC-Algorithm", "domain": "01-因果推断", "reason": "特征工程后用因果发现做变量筛选"},
            {"name": "Skill-Uplift-Modeling", "domain": "01-因果推断", "reason": "特征工程是 Uplift 建模的核心环节"},
        ],
        "可组合": [
            {"name": "Skill-Matrix-Factorization", "domain": "05-推荐系统", "reason": "推荐系统的隐因子也是特征工程的延伸"},
            {"name": "Skill-Customer-Churn-Prediction", "domain": "06-增长模型", "reason": "流失模型严重依赖特征工程"},
        ],
    },
    "11-AI人文/Skill-AI-Humanities-Healing-Cards.md": {
        "前置": [{"name": "Skill-Feature-Engineering", "domain": "12-ML基础", "reason": "理解 ML 基础有助于消化技术与人文的类比"}],
        "延伸": [],
        "可组合": [
            {"name": "Skill-LTV-Prediction-ZILN", "domain": "06-增长模型", "reason": "LTV 长尾分布与人生长尾价值的金句类比"},
            {"name": "Skill-Uplift-Modeling", "domain": "01-因果推断", "reason": "因果反事实与人生决策反思的金句类比"},
        ],
    },
}


def render_relations_section(rel: Dict[str, List[Dict[str, str]]]) -> str:
    lines: List[str] = ["", "## ④ 技能关联", ""]

    def render_block(label: str, items: List[Dict[str, str]], placeholder: str) -> None:
        lines.append(f"### {label}")
        if not items:
            lines.append(f"- {placeholder}")
        else:
            for it in items:
                domain = it.get("domain", "")
                if domain and not it["name"].startswith("Skill-"):
                    pass
                target = f"../{domain}/{it['name']}.md" if domain else f"./{it['name']}.md"
                lines.append(f"- [{it['name']}]({target}) — {it['reason']}")
        lines.append("")

    render_block("前置技能", rel.get("前置", []), "无（本 Skill 是基础入口卡）")
    render_block("延伸技能", rel.get("延伸", []), "无（本 Skill 是终端/聚合卡）")
    render_block("可组合", rel.get("可组合", []), "无")

    return "\n".join(lines)


SECTION_HEADER_RELATIONS_RE = re.compile(
    r"(?m)^##\s*(?:④|4\u20e3|④\.|4\.\s*)?\s*技能关联.*?$"
)
SECTION_HEADER_VALUE_RE = re.compile(
    r"(?m)^##\s*(?:⑤|5\u20e3|⑤\.|5\.\s*)?\s*商业价值.*?$"
)


def patch_skill_card(file_path: Path, rel: Dict[str, List[Dict[str, str]]]) -> str:
    if not file_path.exists():
        return f"❌ {file_path.name}: 文件不存在"

    content = file_path.read_text(encoding="utf-8")
    backup = file_path.with_suffix(file_path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(file_path, backup)

    new_section = render_relations_section(rel).rstrip() + "\n"

    rel_match = SECTION_HEADER_RELATIONS_RE.search(content)
    val_match = SECTION_HEADER_VALUE_RE.search(content)

    if rel_match:
        end_pos = val_match.start() if val_match and val_match.start() > rel_match.start() else len(content)
        new_content = content[: rel_match.start()] + new_section + "\n" + content[end_pos:]
        action = "替换原 ④ 章节"
    elif val_match:
        new_content = content[: val_match.start()] + new_section + "\n" + content[val_match.start():]
        action = "在 ⑤ 商业价值前插入"
    else:
        if not content.endswith("\n"):
            content += "\n"
        new_content = content + "\n" + new_section
        action = "追加到文件末尾"

    file_path.write_text(new_content, encoding="utf-8")
    return f"✅ {file_path.name}: {action}"


def main() -> int:
    results: List[str] = []
    for rel_path, rel in RECOMMENDATIONS.items():
        full = VAULT / rel_path
        results.append(patch_skill_card(full, rel))

    for line in results:
        print(line)
    print(f"\n总计: {len(results)} 张卡片处理完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
