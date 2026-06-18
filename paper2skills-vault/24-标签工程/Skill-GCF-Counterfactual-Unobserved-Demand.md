---
title: 图因果预测GCF — 时空GNN+Synthetic Control估计Listing删除的隐性需求
doc_type: knowledge
module: 24-标签工程
topic: gcf-counterfactual-unobserved-demand-estimation
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 图因果预测GCF — 隐性需求估计

> **来源**：AAAI 2025（GCF: Estimating Unobserved Demand Using Graph Causal Forecasting，Basu/Kumar/Kaveri）+ Synthetic Control Method（Abadie et al.）
> **桥梁**：标签工程 ↔ 因果推断 ↔ 供应链需求预测 | **类型**：因果预测+图神经网络

## ① 算法原理

**核心问题**：跨境电商存在大量"观测不到的需求"——Listing 被封号、商品搜索被屏蔽、竞品打压导致流量消失——此时记录到的销量是 0，但真实需求并非 0。传统预测模型只能学习"已观测的销量"，会系统性低估备货需求。

**GCF 两大创新**（AAAI 2025）：

**1. 时空图神经网络（Spatio-Temporal GNN）**：

```
图结构：节点=SKU，边=相似度（品类/价格/季节模式相似）
时序特征：每个SKU过去T期的销量序列
消息传播：相邻SKU的销量模式互相借鉴（"邻居还在卖，证明需求存在"）
输出：每个SKU的需求分布（含被屏蔽期间的反事实需求）
```

**2. Synthetic Control（合成控制）反事实估计**：

```python
# 核心思想：构建"反事实SKU"
# 反事实SKU = 加权平均其他"未被干扰"的相似SKU
# 权重优化：使干预前合成SKU与目标SKU销量最接近

# 数学形式：
# Y_counterfactual(t) = Σ_j w_j · Y_j(t)   for t > T_intervention
# 其中 w_j 是通过最优化 min ||Y_target(pre) - Σ w_j Y_j(pre)||² 求得

# 隐性需求 = Y_counterfactual - Y_observed（被屏蔽期间的销量差）
```

**MAPE 提升原理**：传统预测在断货/屏蔽期数据为0，训练后会记忆"这段时间销量=0"，造成系统性低估。GCF 用合成控制填充缺失需求，使模型看到"真实需求"，最终 MAPE↓75.3%，推荐数量准确率↑61.2%。

## ② 母婴出海应用案例

**场景A：FBA 断货期间的真实需求估计**

吸奶器旗舰 ASIN 因入库超限导致 FBA 断货 3 周，期间销量记录为 0。备货决策时系统预测"日均 5 件"（被历史0值污染），但真实日均需求约 25 件。

GCF 通过同类产品合成控制，估计断货期真实需求为日均 23 件，备货量提升 4.6 倍，避免断货后恢复期的 GMV 损失。

**数据要求**：目标 SKU 历史销量（含断货期）、同品类对照 SKU 销量、品类相似度特征
**预期产出**：断货期反事实需求曲线 + 修正后的预测区间 + 建议备货量
**业务价值**：备货量准确率提升 61.2%，减少因历史0值污染导致的系统性低估，年化防损 5-15 万元

**场景B：竞品搜索打压后的需求恢复估计**

自身 Listing 搜索排名被竞品刷单打压，流量骤降 70%，销量从日均 30 降至 9。决策者需要判断：是真实需求萎缩（可能是市场季节性），还是竞争干扰（需要打广告反击）？

GCF 合成控制隔离竞争干扰效应，发现真实需求仍为日均 28，流量损失才是主因 → 建议增加广告预算而非降低备货。

**数据要求**：自身 ASIN 销量、竞品 BSR 变化、广告流量数据、对照 ASIN 销量
**预期产出**：竞争干扰效应量化（-21件/天）+ 干预建议（打广告 vs 降价）
**业务价值**：区分"市场萎缩"和"竞争干扰"，避免错误降价损失毛利 3-8 万元/年

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class GCFConfig:
    """GCF 模型配置"""
    n_control_units: int = 10       # 对照 SKU 数量
    pre_period_weight: float = 0.7  # 拟合前期权重
    similarity_threshold: float = 0.3  # 最低相似度阈值
    min_obs: int = 30               # 最少历史观测期数

class SyntheticControlSC:
    """
    供应链 Synthetic Control 反事实需求估计
    
    适用场景：
    - FBA 断货期需求估计
    - Listing 屏蔽期隐性需求
    - 竞品干扰导致的流量损失量化
    """
    
    def __init__(self, config: GCFConfig = None):
        self.config = config or GCFConfig()
        self.weights_: Optional[np.ndarray] = None
        self.control_units_: Optional[List[str]] = None
        self.pre_period_end_: Optional[int] = None
    
    def _compute_sku_similarity(self, target: pd.Series,
                                donors: pd.DataFrame,
                                pre_period_end: int) -> np.ndarray:
        """计算目标SKU与对照SKU的相似度（基于预干预期销量模式）"""
        target_pre = target.values[:pre_period_end]
        similarities = []
        for col in donors.columns:
            donor_pre = donors[col].values[:pre_period_end]
            # 归一化后的 cosine 相似度
            if np.std(target_pre) > 0 and np.std(donor_pre) > 0:
                corr = np.corrcoef(target_pre, donor_pre)[0, 1]
                sim = (corr + 1) / 2  # 映射到 [0, 1]
            else:
                sim = 0.0
            similarities.append(max(0, sim))
        return np.array(similarities)
    
    def fit(self, target: pd.Series, donors: pd.DataFrame,
            intervention_start: int) -> 'SyntheticControlSC':
        """
        拟合合成控制模型
        
        Args:
            target: 目标 SKU 销量时序（含干预期）
            donors: 对照 SKU 矩阵（列=SKU，行=时期）
            intervention_start: 干预开始的时期索引（如断货第一天）
        """
        self.pre_period_end_ = intervention_start
        T_pre = intervention_start
        
        # 按相似度筛选对照 SKU
        sims = self._compute_sku_similarity(target, donors, T_pre)
        valid_donors = donors.columns[sims >= self.config.similarity_threshold]
        
        if len(valid_donors) == 0:
            # 降低阈值重试
            valid_donors = donors.columns[sims >= 0.1]
        
        # 取最相似的 N 个对照 SKU
        sorted_idx = np.argsort(sims)[::-1][:self.config.n_control_units]
        self.control_units_ = list(donors.columns[sorted_idx])
        donor_matrix = donors[self.control_units_].values  # shape: (T, n_donors)
        
        target_pre = target.values[:T_pre]
        donor_pre = donor_matrix[:T_pre, :]
        
        # 优化权重：最小化预干预期拟合误差
        n_donors = len(self.control_units_)
        
        def objective(w):
            synthetic_pre = donor_pre @ w
            return np.mean((target_pre - synthetic_pre) ** 2)
        
        # 约束：权重非负且和为1（凸组合）
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n_donors
        w0 = np.ones(n_donors) / n_donors
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            self.weights_ = result.x
        else:
            # 回退到等权重
            self.weights_ = np.ones(n_donors) / n_donors
        
        return self
    
    def predict_counterfactual(self, donors: pd.DataFrame,
                                intervention_start: int,
                                intervention_end: int) -> pd.Series:
        """
        预测反事实需求（如果没有干预会是什么）
        
        Returns:
            counterfactual demand series for intervention period
        """
        assert self.weights_ is not None, "请先调用 fit()"
        
        donor_matrix = donors[self.control_units_].values
        synthetic_all = donor_matrix @ self.weights_
        
        counterfactual = synthetic_all[intervention_start:intervention_end]
        return pd.Series(counterfactual, 
                        index=range(intervention_start, intervention_end),
                        name="counterfactual_demand")
    
    def compute_hidden_demand(self, target: pd.Series,
                               donors: pd.DataFrame,
                               intervention_start: int,
                               intervention_end: int) -> Dict:
        """
        计算隐性需求（反事实 - 实际观测）
        
        Returns:
            dict: 完整的隐性需求分析结果
        """
        self.fit(target, donors, intervention_start)
        counterfactual = self.predict_counterfactual(donors, intervention_start, intervention_end)
        
        actual_during = target.values[intervention_start:intervention_end]
        hidden = np.maximum(0, counterfactual.values - actual_during)
        
        # 预干预期拟合质量
        synthetic_pre = donors[self.control_units_].values[:intervention_start] @ self.weights_
        target_pre = target.values[:intervention_start]
        pre_mape = np.mean(np.abs(target_pre - synthetic_pre) / np.maximum(target_pre, 1)) * 100
        
        total_intervention_days = intervention_end - intervention_start
        
        return {
            "intervention_period_days": total_intervention_days,
            "observed_demand_total": int(np.sum(actual_during)),
            "counterfactual_demand_total": int(np.sum(counterfactual.values)),
            "hidden_demand_total": int(np.sum(hidden)),
            "hidden_demand_daily_avg": round(float(np.mean(hidden)), 1),
            "demand_suppression_pct": round(
                np.sum(hidden) / max(np.sum(counterfactual.values), 1) * 100, 1
            ),
            "pre_period_fit_mape": round(pre_mape, 2),
            "control_units_used": self.control_units_,
            "optimal_weights": {k: round(float(v), 3) 
                               for k, v in zip(self.control_units_, self.weights_)},
            "recommended_safety_stock_uplift": int(np.sum(hidden) / total_intervention_days * 
                                                    target.values[:intervention_start].std() * 1.5)
        }


# ===== 测试用例 =====
def run_test():
    np.random.seed(42)
    T = 120  # 总时期数
    intervention_start = 80   # 第80天开始断货
    intervention_end = 100    # 第100天恢复
    
    # 生成合成数据：目标SKU有真实需求但断货期销量=0
    true_demand = 25 + 5 * np.sin(np.linspace(0, 4*np.pi, T)) + np.random.normal(0, 2, T)
    
    # 目标SKU：断货期销量归零
    target_sales = true_demand.copy()
    target_sales[intervention_start:intervention_end] = 0  # 断货
    target = pd.Series(target_sales, name="TARGET_SKU")
    
    # 对照 SKU：同品类未受影响的产品（有相关性）
    donors_data = {}
    for i in range(8):
        corr_factor = 0.7 + 0.05 * i
        noise_scale = 3 + i
        donor = (corr_factor * true_demand + 
                 (1 - corr_factor) * 20 + 
                 np.random.normal(0, noise_scale, T))
        donors_data[f"DONOR_SKU_{i:02d}"] = np.maximum(0, donor)
    donors = pd.DataFrame(donors_data)
    
    # 执行隐性需求估计
    model = SyntheticControlSC(GCFConfig(n_control_units=5, similarity_threshold=0.2))
    result = model.compute_hidden_demand(target, donors, intervention_start, intervention_end)
    
    # 验证
    assert result["hidden_demand_total"] > 0, "断货期应有隐性需求"
    assert result["demand_suppression_pct"] > 30, f"需求抑制应>30%，实际{result['demand_suppression_pct']}%"
    assert result["pre_period_fit_mape"] < 30, f"预拟合 MAPE 应<30%，实际{result['pre_period_fit_mape']}%"
    assert len(result["control_units_used"]) > 0, "应使用至少1个对照SKU"
    
    print(f"  干预期: {result['intervention_period_days']} 天")
    print(f"  观测销量: {result['observed_demand_total']} 件 (记录值)")
    print(f"  反事实需求: {result['counterfactual_demand_total']} 件 (真实估计)")
    print(f"  隐性需求: {result['hidden_demand_total']} 件 ({result['demand_suppression_pct']}% 被抑制)")
    print(f"  日均隐性需求: {result['hidden_demand_daily_avg']} 件/天")
    print(f"  预期拟合 MAPE: {result['pre_period_fit_mape']}%")
    
    print("\n[✓] GCF-Counterfactual-Demand 测试通过 — 合成控制+隐性需求估计就绪")

run_test()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]] — 传统预测是本Skill的对比基础
- **前置（prerequisite）**：[[Skill-Causal-Decision-Graph-SC-Inference]] — 因果图提供干预判断的理论框架
- **延伸（extends）**：[[Skill-SC-Causal-DAG-E2E-Attribution]] — GCF估计隐性需求，DAG归因解释为什么需求被抑制
- **延伸（extends）**：[[Skill-Forecast-Bias-Adjustment-Detection]] — 隐性需求修正是预测偏差调整的特殊情况
- **可组合（combinable）**：[[Skill-LLM-SC-MultiAgent-Consensus-Replenishment]] — GCF提供修正需求预测，多智能体用于补货决策
- **可组合（combinable）**：[[Skill-Flash-Sale-Realtime-Sellthrough-Forecast]] — 促销期真实弹性估计需要排除库存约束的反事实

## ⑤ 商业价值评估

- **ROI 预估**：MAPE 降低 75.3%（AAAI 2025 验证），备货量推荐准确率↑61.2%，年化减少因历史0值污染导致的系统性低估损失 5-15 万元
- **实施难度**：⭐⭐⭐☆☆（主要是数据处理 + scipy 优化，无复杂 DL 依赖）
- **优先级**：⭐⭐⭐⭐☆（Listing 断货是跨境电商常态，此方法论独特价值高）
- **企业AI知识库依赖**：中 — 需要同品类对照 SKU 历史数据 + 干预事件记录
