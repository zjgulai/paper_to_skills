---
title: DeepAnalyze 论文萃取
doc_type: analysis
module: data-agent-llm
topic: autonomous-data-science-agent
status: stable
created: 2026-04-26
updated: 2026-04-26
owner: self
source: ai
---

# DeepAnalyze: Agentic Large Language Models for Autonomous Data Science

## 基本信息

- **arXiv ID**: 2510.16872
- **发表时间**: 2025-10
- **作者**: Shaolei Zhang, Ju Fan, Meihao Fan, Guoliang Li, Xiaoyong Du (Renmin University of China)
- **开源代码**: https://github.com/ruc-datalab/DeepAnalyze
- **开源模型/数据**: DeepAnalyze-8B, DataScience-Instruct-500K
- **关键词**: autonomous data science, agentic LLM, curriculum learning, trajectory synthesis

## 核心问题

现有LLM数据科学agent依赖预定义工作流，无法自主编排和自适应优化，导致在复杂端到端数据科学任务上表现受限。

## 核心贡献

### 1. DeepAnalyze-8B：首个端到端自主数据科学Agentic LLM

仅8B参数，支持完整数据科学pipeline：数据准备 → 分析 → 建模 → 可视化 → 洞察 → 报告生成，以及开放式数据研究。

### 2. 课程式Agent训练 (Curriculum-based Agentic Training)

模拟人类数据科学家学习路径：
- **Stage 1 (单能力微调)**：强化基础能力 — 推理、结构化数据理解、代码生成
- **Stage 2 (多能力Agent训练)**：使用GRPO训练自主编排多动作组合的能力

### 3. 数据Grounded轨迹合成 (Data-grounded Trajectory Synthesis)

解决高质量训练数据稀缺问题：
- **Reasoning Trajectory Synthesis**：从现有结构化数据指令数据集（TableQA等）提取推理轨迹，通过distillation + keyword-guided refinement增强
- **Interaction Trajectory Synthesis**：多Agent系统（Questioner→Solver→Inspector）从真实数据源合成多轮交互轨迹

### 4. 五动作自主编排架构

- **Analyze**: 文本分析、规划、推理、反思
- **Understand**: 理解数据源内容（数据库、表格、文档）
- **Code**: 生成与环境交互的代码（Python）
- **Execute**: 执行代码并收集环境反馈
- **Answer**: 生成最终输出

推理时通过特殊token自动切换动作，无需人工定义工作流。

### 5. 混合奖励建模

- 规则奖励：检查输出格式是否符合架构
- LLM-as-judge奖励：评估报告质量（usefulness, richness, soundness, interpretability, readability）
- 交互质量评分

## 实验结果

| Benchmark | Metric | DeepAnalyze-8B | GPT-4o | Claude-3.5-Sonnet |
|-----------|--------|----------------|--------|-------------------|
| DataSciBench | Score | 61.11 | 54.18 | 30.78 |
| DSBench | Success Rate | 90.63% | 77.76% | - |
| DABStep-Research | Score | 38.88 | 27.33 | - |
| TableQA (AVG) | Accuracy | 64.47 | 58.96 | - |

- 在12个数据科学benchmark上超越最先进工作流agent和专有LLM
- 端到端pipeline能力：无需依赖外部编排框架
- 数据建模任务：成功率59.91%，完成率66.24%

## 训练细节

- **基础模型**: DeepSeek-R1-0528-Qwen3-8B
- **训练框架**: ms-swift
- **RL算法**: GRPO (Group Relative Policy Optimization)
- **训练数据规模**: 
  - Stage 1: ~470K samples (single-ability)
  - Stage 2: 32K samples (multi-ability cold-start) + 15K (RL phase)
  - 总计: DataScience-Instruct-500K
- **硬件**: NVIDIA A800 GPUs

## 母婴出海业务映射

### 场景1: 自动销售数据分析报告
输入：Amazon/Shopify/SHEIN等多平台销售数据（CSV/Excel）
输出：自动完成数据清洗 → 趋势分析 → 竞品对比 → 可视化 → 洞察报告

### 场景2: 用户评论深度研究
输入：Trustpilot/Zendesk/Amazon评论数据
输出：情感趋势分析 → 关键话题提取 → 时间维度演变 → 可执行建议报告

## 技能关系

- **前置**: Python数据分析基础, Pandas/NumPy, LLM基础
- **关联**: 
  - Skill-AutoTag-SelfEvolving-Label-System（自动化标签生成）
  - Skill-Aspect-Based-Sentiment-Analysis（情感分析Agent数据源）
  - Skill-OpenWorld-Class-Incremental-Learning（开放域分类与Agent结合）
- **扩展**: 
  - 与Argos结合 → 异常检测+自动报告
  - 与Data-to-Dashboard结合 → 自动可视化仪表板

## 代码可得性

- **模型**: HuggingFace `ruc-datalab/DeepAnalyze-8B`
- **训练数据**: HuggingFace `ruc-datalab/DataScience-Instruct-500K`
- **GitHub**: https://github.com/ruc-datalab/DeepAnalyze
- **Demo**: https://deepanalyze.github.io

## 限制与风险

1. 8B模型能力有限，极复杂统计建模可能仍需更大模型
2. 需要与环境（Python运行时、文件系统）交互，存在代码执行安全风险
3. 中文支持依赖基础模型（Qwen3-8B有中文能力，但论文实验以英文为主）
4. 训练成本高，复现需要A800级GPU
