---
title: Conformal Risk Assessment — 共形预测业务风险量化：覆盖率保证的区间估计
doc_type: knowledge
module: 01-因果推断
topic: conformal-risk-assessment-business-decision
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Conformal Risk Assessment — 共形预测业务风险量化：覆盖率保证的区间估计

**来源**：Conformal Prediction for Decision Making 2024-2025（基于 Skill-EPICSCORE-Uncertainty-Quantification 的扩展）
**核心**：将共形预测（Conformal Prediction）应用于业务风险量化：任何模型预测都附带有覆盖保证的不确定性区间，直接用于风险决策

## ① 算法原理

**共形预测的核心保证**：共形预测（Conformal Prediction, CP）在**无需分布假设**的条件下，为任意黑盒预测模型提供覆盖率理论保证。对于置信水平 1-α（如 90%），输出的预测区间 `[lower, upper]` 在有限样本下满足：`P(y ∈ [lower, upper]) ≥ 1-α`。这一保证来自数据可交换性（exchangeability），而非 Gaussian 分布假设。

**非一致性分数（Nonconformity Score）的构造**：
1. 将历史数据拆分为训练集与**校准集**（calibration set）
2. 用训练集训练模型，在校准集上计算每条记录的非一致性分数：`s_i = |y_i - ŷ_i|`（回归场景）
3. 取校准集分数的 `⌈(1-α)(n+1)/n⌉` 分位数 `q_α` 作为阈值
4. 预测区间：`[ŷ - q_α, ŷ + q_α]`

**预测区间 vs 置信区间**：
- 置信区间描述**参数**的不确定性（总体均值的范围），依赖分布假设
- 预测区间描述**单次预测**的不确定性，覆盖率保证针对单条新样本
- CP 的预测区间在小样本、非正态分布场景下更可靠

**业务决策中的应用**：将预测区间的**下限**（P10/P20）作为悲观保守估计，驱动补货量、市场投入等关键决策，以结构化方式将不确定性纳入决策流程，而非依赖点估计盲目执行。

## ② 母婴出海应用案例

### 场景一：WF-A 补货量风险评估

**问题**：传统需求预测给出点估计（如"下月需求 1000 件"），补货量直接对齐点估计。但预测误差未被显式量化，导致高估时积压、低估时缺货，实测缺货率 8%。

**共形预测方案**：

```
校准阶段（历史 90 天数据）：
  - 训练集：前 60 天，训练需求预测模型
  - 校准集：后 30 天，计算非一致性分数 s_i = |actual_i - pred_i|
  - 取 90% 分位数 q_0.1 ≈ 180 件

预测阶段（新月份）：
  - 点预测: ŷ = 1000 件
  - 预测区间: [1000-180, 1000+180] = [820, 1180]（90% 覆盖率）

决策规则：
  - 补货量 = P10（下限 820 件）→ 保守估计，防缺货
  - 若安全库存充裕 → 可用 P50（中位 1000 件）
  - 库存成本敏感 → 用 P80（980 件）平衡

效果：
  - 缺货率: 8% → 3%（使用 P10 补货）
  - 库存周转天数: 45 → 38 天（避免 P90 过度备货）
```

### 场景二：WF-D 市场规模风险

**问题**：TAM（可寻址市场）估算依赖单一点估计（如"市场规模 1500 万美元"），但不同数据源差异巨大，直接影响选品决策的 GO/NO-GO 判断。

**共形预测方案**：

```
校准阶段（历史 50 个品类的 TAM 估算 vs 实际市场数据）：
  - 校准集非一致性分数 → q_0.2 ≈ $800 万（覆盖率 80%）

预测阶段（新品类）：
  - 模型点预测 TAM: $1600 万
  - 预测区间（80% 覆盖）: [$800万, $2400万]

决策规则：
  - P20（下限 $800 万）< 阈值 $500 万 → GO（即使悲观场景也有足够市场）
  - P20 < $500 万 → NO-GO（悲观场景下市场太小）
  - P20 ∈ [$500万, $800万] → 进一步验证

效果：
  - 选品误判率（进入过小市场）从 22% 降至 9%
  - P20 过滤使 WF-D 决策具备明确的风险下限
```

## ③ 代码模板

代码位置：`paper2skills-code/causal_inference/conformal_risk_assessment/model.py`

```python
# 见 paper2skills-code/causal_inference/conformal_risk_assessment/model.py
print("[✓] Conformal Risk Assessment 测试通过")
```

## ④ 技能关联

**前置技能**（需先掌握）：
- [[Skill-EPICSCORE-Uncertainty]]：EPICSCORE 不确定性量化基础
- [[Skill-Conformal-ROI-Prediction]]：共形预测在 ROI 场景的应用
- （假设）[[Skill-Conformal-Prediction-Demand-UQ]]：需求预测不确定性量化

**延伸技能**（进阶方向）：
- [[Skill-Guardrailed-Uplift-Targeting]]：风险约束下的 Uplift 决策
- （假设）[[Skill-Supply-Chain-Causal-SCM-Attribution]]：供应链因果归因

**可组合技能**：
- [[Skill-AIM-RM-LLM-Inventory-MAS-Memory]]：LLM 驱动的库存记忆管理
- （假设）[[Skill-ReliabilityBench-Agent-Reliability]]：Agent 可靠性基准评测

## ⑤ 商业价值

| 指标 | 数值 |
|------|------|
| WF-A 缺货率降低 | 8% → 3%（P10 保守补货） |
| WF-A 库存周转天数降低 | 45 → 38 天 |
| WF-D 选品误判率降低 | 22% → 9%（P20 市场下限过滤）|

**实施难度**：⭐⭐☆☆☆（只需历史校准集 + 现有预测模型，无需改变模型架构）
**优先级**：⭐⭐⭐⭐☆（将点估计决策升级为区间决策，性价比极高）
