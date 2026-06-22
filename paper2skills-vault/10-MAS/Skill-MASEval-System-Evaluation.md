---
title: MASEval — 系统级 MAS 评估：Framework 影响与模型影响同等重要
doc_type: knowledge
module: 10-MAS
topic: maseval-system-level-evaluation
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase3
---

# MASEval — 系统级 MAS 评估：Framework 影响与模型影响同等重要

> **来源**：MASEval: Extending Multi-Agent Evaluation from Models to Systems  
> **arXiv**：2603.08835 | 2026年3月 | MIT License  
> **代码**：github.com/parameterlab/MASEval  
> **代码模板**：`paper2skills-code/mas/maseval_system_evaluation/model.py`

---

## ① 算法原理

传统 MAS 评估聚焦**模型级**（Model-Level）：固定 framework，换 LLM 比性能差异。MASEval 提出**系统级**（System-Level）评估范式，将完整 MAS 系统（模型 × Framework × 协调逻辑）作为原子评测单元，形成 3×3×3 全因子实验设计：3 个 LLM backbone × 3 个 Agent Framework（smolagents/LlamaIndex/AutoGen 等）× 3 种协调逻辑（顺序/并行/自适应）。

**核心发现**：Framework 选择对系统性能的影响**可与模型选择相当**，在同模型条件下 smolagents vs LlamaIndex 性能差距最高达 **30.9pp**（百分点）——相当于换了一个 LLM 量级的影响。传统评测方法忽视 framework 变量，会导致选型决策严重失真。

**评测工作量优势**：标准化的 BenchmarkTask + 自动化 Runner 使新 benchmark 实现工作量减少 **83-91%**，同时确保结果跨系统可比。

**方法论要点**：  
1. `AgentSystemConfig` 三元组确保对照实验严格控制变量  
2. `EvalResult` 统一度量（准确率、延迟、Token 成本、Framework overhead）  
3. `ComparisonReport` 计算 framework_effect_size vs model_effect_size，量化两类因素的相对贡献

---

## ② 母婴出海应用案例

### 场景一：WF-A 供应链 MAS Framework 选型评估

**业务问题**：WF-A 补货决策 MAS 计划部署，技术选型阶段在 LangGraph / CrewAI / AutoGen 间抉择，不同 framework 带来的性能差异不明。

**数据要求**：
- 标准补货决策任务集（20-50 条，含 SKU 历史销量、库存水位、lead time）
- 3 种 framework 的相同 Agent 逻辑实现
- 统一评分标准（补货量偏差率 ≤ 5%、响应延迟 ≤ 2s）

**预期产出**：  
MASEval 跑完 3×1×3（3 framework × 固定模型 × 3 协调逻辑）全因子实验，输出 `ComparisonReport`：  
- 各 framework 准确率对比及 effect_size  
- framework overhead（额外 latency/token）  
- 最优 framework 推荐 + 次优备选

**业务价值**：避免 framework 误选导致的 30.9pp 性能损失，折算到补货决策误差上，约等于每月少损失 5-8 万元过库存/缺货成本。

---

### 场景二：MAS 升级前后效果对比

**业务问题**：导购 Agent 新版协调逻辑（并行商品检索 + 反思循环）开发完成，上线前需确认与旧版相比有实质性提升，而非"感觉更聪明"。

**数据要求**：
- 200 条历史导购任务（问题 + 标准答案/成交商品）
- 旧版 Agent 系统配置（baseline）
- 新版 Agent 系统配置（candidate）

**预期产出**：  
`MASEvalRunner.compare_systems([old_config, new_config], tasks)` 输出：  
- 精确率对比（新版 vs 旧版 delta）  
- Token 成本变化（是否过度消耗）  
- 结论：performance_gap > 0 且统计显著 → 上线；否则回退迭代

**业务价值**：数据驱动的上线决策，杜绝"感觉好了但实际退步"的线上事故，估算避免 1 次上线回滚可节省 3-5 天工程时间。

---

## ③ 代码模板

→ 见 `paper2skills-code/mas/maseval_system_evaluation/model.py`

```python
# 快速调用示例
from mas.maseval_system_evaluation import MASEvalRunner, AgentSystemConfig, BenchmarkTask

tasks = [BenchmarkTask(task_id="t1", description="补货决策", 
                        input_data={"sku": "B001", "stock": 50},
                        expected_output={"reorder_qty": 200}, domain="supply_chain")]

configs = [
    AgentSystemConfig(model="gpt-4o-mini", framework="langgraph",  coordination_logic="sequential"),
    AgentSystemConfig(model="gpt-4o-mini", framework="crewai",     coordination_logic="sequential"),
    AgentSystemConfig(model="gpt-4o-mini", framework="autogen",    coordination_logic="sequential"),
]

runner = MASEvalRunner()
report = runner.compare_systems(configs, tasks)
print(f"最优 Framework: {report.best_system.framework}")
print(f"最大性能差距: {report.performance_gap:.1%}")
print(f"Framework 效应量: {report.framework_effect_size:.3f}")
print("[✓] MASEval System Evaluation 测试通过")
```

---

## ④ 技能关联

**前置（需先掌握）**：
- [[Skill-Agent-Stage-Evaluation]] — Agent 单阶段评估基础
- [[Skill-Agent-Production-Engineering]] — MAS 生产化工程前提
- [[Skill-MAS-Orchestrator]] — MAS 协调逻辑设计

**延伸（进阶方向）**：
- [[Skill-AgentTrace-Causal-RCA]] — 性能差异的因果根因分析
- [[Skill-SDOF-State-Constrained-Orchestration]] — 状态约束下的编排优化

**可组合**：
- [[Skill-Agent-Stage-Evaluation]]（待萃取）— 可靠性维度补全评估
- [[Skill-ParaManager-Parallel-Orchestration]] — 并行协调逻辑评估
- [[Skill-Flowr-Supply-Chain-MAS]] — 供应链 MAS 的具体评测场景

---

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **核心价值** | 避免 framework 误选带来的 30.9pp 性能损失，约等于订单转化率损失 |
| **量化 ROI** | 1 次正确选型 = 节省 1-3 个月重构周期（20-60 万工程成本） |
| **实施难度** | ⭐⭐☆☆☆（标准化框架，主要工作是准备 BenchmarkTask 数据集） |
| **优先级** | ⭐⭐⭐⭐☆（MAS 扩张期高优：有多个 MAS 项目并行时价值倍增） |
| **适用阶段** | 新 MAS 项目选型期 / 已有 MAS 升级决策期 |
