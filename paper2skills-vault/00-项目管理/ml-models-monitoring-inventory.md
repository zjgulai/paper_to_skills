---
title: 生产 ML 模型监控现状调研
created: 2026-05-25
purpose: A0 前置调研，为 Skill-Data-Drift-Detection 和 Skill-Model-Performance-Monitor 萃取提供真实业务上下文
---

# 生产 ML 模型监控现状调研

> **用途**：确保 A1/A2 Skill 卡片的业务场景章节基于真实生产模型，而非通用 ML 教材。

---

## 已上线生产模型清单

### WF-A · 需求预测

| 模型 | 代码路径 | 服务入口 | 当前监控状态 |
|---|---|---|---|
| Demand Forecasting (Prophet/TFT) | `paper2skills-code/growth_model/ltv_prediction/` + `time_series/` | `services/forecast_service.py` | ❌ 无监控 — 无 AUC/MAPE 衰减告警，无 feature drift 检测，无自动重训触发 |
| Hierarchical Demand Forecasting | `paper2skills-code/time_series/hierarchical_demand_forecasting/` | 无独立服务 | ❌ 无监控 |
| Safety Stock Replenishment | `paper2skills-code/supply_chain/safety_stock_replenishment/` | 无独立服务 | ❌ 无监控 |

**风险**：需求预测模型在大促（618/双11）前后会出现显著 demand spike，当前无机制区分「季节性波动」vs「真实 feature drift」，模型可能在大促后长期使用被污染的基线。

---

### WF-B · 广告素材 MAB

| 模型 | 代码路径 | 服务入口 | 当前监控状态 |
|---|---|---|---|
| Multi-Armed Bandit (Thompson Sampling) | `paper2skills-code/ab_testing/multi_armed_bandit/` + `ab_testing/thompson_sampling/` | `services/mab_service.py` | ⚠️ 部分监控 — 有 reward 日志，但无 regret 累计告警，无臂分布漂移检测 |

**风险**：MAB 的 reward 分布（CTR/CVR）会随广告素材生命周期衰减（Creative Fatigue），当前无机制检测臂的期望收益是否已发生结构性漂移，导致 exploration 比例失当。

---

### WF-F · 用户增长模型

| 模型 | 代码路径 | 服务入口 | 当前监控状态 |
|---|---|---|---|
| LTV Prediction (ZILN) | `paper2skills-code/growth_model/ltv_prediction/model.py` + `model_sklearn.py` | 无独立服务 | ❌ 无监控 |
| Churn Prediction | `paper2skills-code/growth_model/churn_prediction/model_dnn.py` | 无独立服务 | ❌ 无监控 |
| Uplift Modeling | `paper2skills-code/causal_inference/uplift_modeling/` | 无独立服务 | ❌ 无监控 |
| Uplift Churn Prediction | `paper2skills-code/growth_model/uplift_churn_prediction/model.py` | 无独立服务 | ❌ 无监控 |
| NBA Guardrailed CATE | `paper2skills-code/06-增长模型/nba_guardrailed_2025/model.py` | 无独立服务 | ❌ 无监控 |

**风险**：LTV/Churn 模型的特征分布（购买频次、客单价、品类偏好）会随 SKU 扩张和用户结构变化而漂移。当前所有模型训练后即静态部署，无 PSI 监控，无重训触发机制。

---

## 监控空白汇总

| 监控能力 | 现状 | 影响模型数 |
|---|---|---|
| Feature drift 检测 (PSI/KS) | ❌ 完全缺失 | 8/8 |
| Label drift 检测 | ❌ 完全缺失 | 8/8 |
| 模型性能衰减告警 (AUC/MAPE 滑动窗口) | ❌ 完全缺失 | 8/8 |
| 自动重训触发 | ❌ 完全缺失 | 8/8 |
| Shadow mode / Canary 灰度 | ❌ 完全缺失 | 8/8 |
| 大促期间基线隔离 | ❌ 完全缺失 | 8/8 |
| Reward 分布漂移 (MAB) | ⚠️ 日志存在但无告警 | 1/8 |

**结论**：当前 8 个生产 ML 模型全部裸跑，无任何系统性监控。这是高优先级风险，任何模型的无声失效都不会被及时发现。

---

## 对 A1/A2 Skill 卡片的启示

### A1 Skill-Data-Drift-Detection 应重点覆盖：
1. **PSI 计算**：适用于 LTV/Churn 的用户特征漂移检测（特征：购买频次/客单价/品类分布）
2. **ADWIN 滑动窗口**：适用于 MAB reward 分布的实时漂移检测
3. **大促季节性隔离**：如何区分节假日/大促造成的短期 spike vs 真实 concept drift
4. **母婴场景特有漂移**：新生儿用户生命周期（0-3岁）导致的自然用户群体迁移

### A2 Skill-Model-Performance-Monitor 应重点覆盖：
1. **轻量级监控框架**：8 个模型无独立服务，需要低侵入性的监控方案（不要求重构服务架构）
2. **离线批量评估**：针对无实时服务的模型（LTV/Uplift），设计离线定期评估流程
3. **`forecast_service.py` + `mab_service.py` 的监控插桩**：两个已有服务的最小改动监控方案
4. **重训触发规则**：PSI > 0.2 触发警告 / PSI > 0.25 触发自动重训的阈值设定依据

---

## 调研结论

**A0 完成**。调研发现监控空白比预期更严重（8/8 无监控），强化了 A1/A2 的 P0 优先级判断。

A1 Skill 萃取时应以「母婴 DTC 电商模型的 feature drift 检测」为主线业务场景，而非通用 ML drift 理论综述。
