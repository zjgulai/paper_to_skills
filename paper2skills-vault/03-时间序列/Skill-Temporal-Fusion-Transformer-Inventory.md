---
title: Temporal Fusion Transformer Inventory — TFT 多变量时序库存补货决策
doc_type: knowledge
module: 03-时间序列
topic: temporal-fusion-transformer-inventory
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: arxiv:1912.09363
roadmap_phase: phase1
tags:
  - time-series
  - forecasting
  - inventory-management
  - transformer
  - multi-variate
keywords:
  - Temporal Fusion Transformer
  - TFT
  - 库存补货
  - 分位数预测
  - 可解释性
difficulty: intermediate
estimated_time: 45
---

# Skill Card: Skill-Temporal-Fusion-Transformer-Inventory

## ① 算法原理（≤300字）

> **论文**：Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting | **年份**：2019

**核心问题**：库存补货决策需要融合多种异质信号——过去销量、广告投放计划、节假日、竞品价格、评论数等。传统 ARIMA/Prophet 只建模单变量；LSTM 虽可多变量但无可解释性，难以调试。TFT（Temporal Fusion Transformer）同时解决多变量融合和可解释性。

**TFT 架构三核**：

1. **变量选择网络（VSN）**：门控机制自动筛选对预测有用的输入变量，输出每个变量的贡献权重（可解释）
2. **时序编码层**：LSTM 编码历史序列 + Transformer 自注意力捕获长程依赖
3. **分位数输出**：同时预测 P10/P50/P90 分位数，输出区间而非点预测，直接对应「乐观/基准/保守」三种备货策略

**关键公式（注意力权重）**：
$$\alpha_{t,\tau} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

注意力权重 $\alpha_{t,\tau}$ 揭示模型在预测 $t$ 时刻时，重点参考了哪些历史时间步。

**与 Prophet 的对比**：TFT 在变量 > 5、数据量 > 500 条时显著优于 Prophet；小数据场景 Prophet 更稳定。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：母婴卖家旗下吸奶器主力 SKU，需要融合广告计划（Sponsored Products 预算）、竞品价格变动、Prime 会员日安排，提前 8 周预测周维度销量用于供应链锁单。

**数据要求**：过去 52 周销量，广告花费，BSR 排名，竞品价格，促销标记，节假日标签。

**TFT 应用**：VSN 显示广告花费和 BSR 排名贡献权重各占 35%/28%，模型识别到 Prime Day 前 3 周广告加速 → 销量提升的滞后效应。P90 分位数用于安全库存设定。

**量化产出**：8 周预测 MAPE 从朴素方法 28% 降至 12%，备货过剩率从 22% 降至 8%，年化降低库存成本 **35 万元**。

**三轨验证** | 成本轨：月均成本约2,800元（模型训练与维护1,500元/月，数据标注与清洗800元/月，服务器资源500元/月），人工投入约15小时/月（数据审核8小时，模型调优5小时，结果验证2小时）| 合规轨：符合《跨境电商商品质量管理规范》和《食品安全法》要求，预测结果需经食品安全员审核后方可用于补货决策，依据：GB 2760食品添加剂使用标准和进出口食品检验检疫规定 | 风险轨：模型漂移风险（概率25%，季节性变化导致MAPE超15%），数据质量风险（概率15%，缺失订单数据影响准确性），供应链中断风险（概率10%，预测准确但无法及时补货）

## ③ 代码模板

```python
import numpy as np
from collections import defaultdict

def tft_simple_quantile_forecast(
    y_hist: np.ndarray,
    exog: np.ndarray,
    horizon: int = 8,
    quantiles: list = None
) -> dict:
    """
    简化版 TFT 分位数预测（演示架构逻辑，生产建议用 pytorch-forecasting）
    
    Args:
        y_hist: 历史销量序列 (T,)
        exog: 外部变量矩阵 (T, n_features)
        horizon: 预测步数
        quantiles: 分位数列表
    
    Returns:
        dict: 包含各分位数预测和变量重要性
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
    
    T, n_feat = exog.shape
    
    # 变量重要性计算（模拟 VSN 门控）
    # 使用相关系数的绝对值作为重要性指标
    var_importance = np.zeros(n_feat)
    for i in range(n_feat):
        corr = np.corrcoef(y_hist, exog[:, i])[0, 1]
        var_importance[i] = np.abs(corr) if not np.isnan(corr) else 0.0
    
    # 归一化重要性权重
    var_importance_sum = var_importance.sum()
    if var_importance_sum > 1e-8:
        var_importance = var_importance / var_importance_sum
    else:
        var_importance = np.ones(n_feat) / n_feat
    
    # 加权特征均值作为趋势信号
    weighted_signal = exog @ var_importance
    
    # 基于最近 8 期加权平均的基准预测
    window = min(8, T)
    base = np.mean(y_hist[-window:])
    trend = (y_hist[-1] - y_hist[-window]) / window if window > 1 else 0.0
    signal_adj = (weighted_signal[-1] - np.mean(weighted_signal[-window:])) * 0.3
    
    # 计算历史波动率
    hist_std = np.std(y_hist[-window:]) if window > 1 else np.std(y_hist)
    hist_std = max(hist_std, 1e-6)  # 防止除以零
    
    # 生成分位数预测
    results = {}
    for q in quantiles:
        # 根据分位数调整噪声尺度
        noise_scale = hist_std * (0.5 + q)
        preds = []
        for h in range(1, horizon + 1):
            # 基础预测 + 趋势 + 信号调整 + 分位数偏移
            quantile_offset = np.sqrt(2) * hist_std * (q - 0.5) * 0.5
            pred = base + trend * h + signal_adj + quantile_offset
            preds.append(pred)
        results[f'q{int(q*100)}'] = np.array(preds)
    
    # 保存变量重要性
    results['var_importance'] = {
        f'feat_{i}': float(var_importance[i]) 
        for i in range(n_feat)
    }
    
    return results


def validate_tft_results(result: dict, horizon: int = 8) -> bool:
    """验证 TFT 预测结果的有效性"""
    # 检查必要的分位数存在
    if 'q10' not in result or 'q50' not in result or 'q90' not in result:
        return False
    
    # 检查预测长度
    if len(result['q50']) != horizon:
        return False
    
    # 检查分位数大小关系（P10 <= P50 <= P90）
    q10 = result['q10']
    q50 = result['q50']
    q90 = result['q90']
    
    if not np.all(q10 <= q50 + 1e-6) or not np.all(q50 <= q90 + 1e-6):
        return False
    
    # 检查变量重要性
    if 'var_importance' not in result:
        return False
    
    var_imp_sum = sum(result['var_importance'].values())
    if not (0.99 <= var_imp_sum <= 1.01):
        return False
    
    return True


# ============ 测试代码 ============
np.random.seed(42)

# 生成模拟数据：52周历史数据
T = 52
# 销量序列：基础值 + 趋势 + 随机波动
y = 100 + np.cumsum(np.random.randn(T) * 5) + np.arange(T) * 0.5

# 外部变量：广告花费、BSR排名、竞品价格
exog = np.column_stack([
    np.random.randn(T) * 1000 + 5000,  # 广告花费 (5000±1000)
    np.random.randn(T) * 50 + 200,     # BSR排名 (200±50)
    np.random.randn(T) * 2 + 30        # 竞品价格 (30±2)
])

# 运行 TFT 预测
result = tft_simple_quantile_forecast(y, exog, horizon=8, quantiles=[0.1, 0.5, 0.9])

# 验证结果
assert validate_tft_results(result, horizon=8), "TFT 结果验证失败"

# 验证分位数预测存在
assert 'q10' in result and 'q50' in result and 'q90' in result, "缺少分位数预测"

# 验证预测长度
assert len(result['q50']) == 8, "预测长度不正确"

# 验证分位数大小关系
assert np.all(result['q10'] <= result['q50']), "P10 应小于等于 P50"
assert np.all(result['q50'] <= result['q90']), "P50 应小于等于 P90"

# 验证变量重要性
assert 'var_importance' in result, "缺少变量重要性"
assert len(result['var_importance']) == 3, "变量重要性数量不正确"

# 输出结果
print(f"P10 预测（未来8周）: {result['q10'].round(1)}")
print(f"P50 预测（未来8周）: {result['q50'].round(1)}")
print(f"P90 预测（未来8周）: {result['q90'].round(1)}")
print(f"变量重要性: {result['var_importance']}")
print("[✓] Temporal-Fusion-Transformer-Inventory 测试通过")
```

## ④ 技能关联

- 前置技能：[[Skill-Temporal-Fusion-Transformer]]
- 前置技能：[[Skill-Demand-Forecasting-Supply-Chain]]
- 延伸技能：[[Skill-Multi-Echelon-Inventory]]
- 延伸技能：[[Skill-Forecast-Driven-Inventory]]
- 可组合：[[Skill-Safety-Stock-Replenishment]]
- 可组合：[[Skill-DRL-Inventory-Optimization]]

## ⑤ 商业价值评估

- **ROI量化**: 8 周预测误差降低 57%，库存成本年化节省 30-50 万元
- **实施难度**: ⭐⭐⭐（需要 pytorch-forecasting，调参成本较高）
- **优先级**: ⭐⭐⭐⭐⭐（多变量场景的最优方案）
