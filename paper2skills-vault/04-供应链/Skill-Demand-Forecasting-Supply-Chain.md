---
title: Demand Forecasting for Supply Chain
module: 04-供应链
topic: demand-forecasting
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase1
---

# Skill Card: Demand Forecasting (Supply Chain)

## ① 算法原理

**核心问题**：供应链的需求预测不同于通用时序预测——它必须考虑促销日历、竞品行动、渠道库存、季节性生命周期等商业因素。预测不准的代价是：过高→库存积压，过低→断货丢单。

**关键区别于通用时序**：

| 因素 | 通用时序 | 供应链需求预测 |
|------|---------|--------------|
| 季节性 | 自然季节性 | 促销日历驱动的"人造季节" |
| 外部变量 | 可选 | 必须（价格、促销、竞品价格） |
| 粒度 | 单SKU | SKU×仓库×渠道的多层级 |
| 更新频率 | 定期 | 实时（促销期间） |
| 评估指标 | MAE/RMSE | 偏差成本（Bias Cost） |

**分层预测（Hierarchical Forecasting）**：

需求在多个层级上存在：
- 顶层：全品类总需求
- 中层：品类/品牌维度
- 底层：SKU×仓库维度

分层预测确保各层级的预测相互一致（底层之和=中层之和=顶层）。方法：
- **Bottom-Up**：从SKU预测汇总到顶层，适合SKU差异大的场景
- **Top-Down**：从顶层按比例分解到SKU，适合SKU相似的场景
- **Middle-Out**：中层预测，向上汇总+向下分解
- **Reconciliation**：各层级独立预测后用最小二乘法协调

**促销效应建模**：

促销是母婴电商需求波动的最大来源。需要建模：
- **促销类型**：满减、折扣、买赠、捆绑
- **促销幅度**：折扣深度对销量的弹性
- **促销衰减**：促销结束后的需求回落（cannibalization + pull-forward）
- **竞争促销**：竞品同期促销对本品的交叉效应

**反直觉洞察**：
- 预测准确率不是越高越好——**预测偏差的方向更重要**。过度预测（安全）的代价是库存积压，低估预测（激进）的代价是断货。两种偏差的成本不对称。
- 新品需求预测不能用历史数据——需要用"类比法"（找相似老品的历史模式）或"Bass扩散模型"。
- 90%的预测误差来自10%的异常事件（大促、爆款、断货），而非日常波动。

---

## ② 母婴出海应用案例

### 场景1：奶粉SKU的月度需求预测

**业务问题**：Momcozy 代理某品牌奶粉在欧洲销售，涉及5个段位×3个规格×4个仓库=60个SKU-仓库组合。需要预测未来4周的周需求量，用于向供应商下采购单（ lead time 6周）。

**预测流程**：
1. **数据准备**：
   - 历史销量：过去104周（2年）的周销量
   - 促销日历：黑五、圣诞、复活节、Prime Day
   - 价格数据：自身价格 + 竞品价格
   - 外部数据：Google Trends（"baby formula"搜索指数）

2. **特征工程**：
   - 滞后销量：上周、上月同期、去年同期的销量
   - 促销特征：是否促销周、促销深度、促销类型
   - 生命周期：SKU上市周数（新品效应）
   - 季节特征：周数、是否节假日

3. **模型选择**：
   - 基线：移动平均
   - 主力：LightGBM（处理促销等非线性效应）
   - 校准：Prophet（捕捉趋势和季节性）

4. **分层协调**：
   - 先预测各仓库的总需求（Top-Down）
   - 按比例分解到各SKU
   - 用历史比例作为分解权重

**预期产出**：
- 预测准确率（WAPE）：基线 25% → 模型 15%
- 缺货率：8% → 3%
- 库存周转：4次/年 → 6次/年

### 场景2：促销期间的需求峰值预测

**业务问题**：黑五期间某爆款吸奶器预计做30% off促销，需要预测促销周的需求量（可能是平时的5-10倍），避免断货或过度备货。

**预测方法**：
1. 找历史相似促销（去年黑五、Prime Day）
2. 计算促销弹性：促销期间的销量 / 无促销基线销量
3. 考虑库存约束：如果预测需求 > 可用库存，实际销量被库存封顶
4. 考虑竞争：竞品是否同期促销

---

## ③ 代码模板

```python
"""
Demand Forecasting for Supply Chain — 供应链需求预测
支持：分层预测、促销效应建模、多SKU批量预测
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


class SupplyChainDemandForecaster:
    """供应链需求预测器"""

    def __init__(self, model=None):
        self.model = model or GradientBoostingRegressor(n_estimators=100, max_depth=4)
        self.is_fitted = False

    def build_features(self, df):
        """构建供应链需求预测特征"""
        df = df.copy()
        df = df.sort_values(['sku', 'date'])

        # 滞后特征
        for lag in [1, 2, 4, 12]:
            df[f'sales_lag_{lag}'] = df.groupby('sku')['sales'].shift(lag)

        # 滚动统计
        for window in [4, 12]:
            df[f'sales_ma_{window}'] = df.groupby('sku')['sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        # 促销特征
        if 'is_promo' in df.columns:
            df['promo_depth'] = df.get('discount_rate', 0)
            df['promo_lag_1'] = df.groupby('sku')['is_promo'].shift(1)

        # 季节特征
        df['week_of_year'] = pd.to_datetime(df['date']).dt.isocalendar().week
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['is_holiday'] = df['week_of_year'].isin([47, 48, 49, 50, 51]).astype(int)

        # 价格特征
        if 'price' in df.columns:
            df['price_lag_1'] = df.groupby('sku')['price'].shift(1)
            df['price_change'] = (df['price'] - df['price_lag_1']) / df['price_lag_1']

        return df.dropna()

    def fit(self, df, feature_cols, target_col='sales'):
        """训练模型"""
        df_train = df.dropna(subset=feature_cols + [target_col])
        X = df_train[feature_cols]
        y = df_train[target_col]
        self.model.fit(X, y)
        self.feature_cols = feature_cols
        self.is_fitted = True
        return self

    def predict(self, df):
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = df[self.feature_cols].fillna(0)
        return np.maximum(self.model.predict(X), 0)

    def evaluate(self, df, feature_cols, target_col='sales'):
        """评估：WAPE（加权绝对百分比误差）"""
        df_eval = df.dropna(subset=feature_cols + [target_col])
        preds = self.predict(df_eval)
        actuals = df_eval[target_col].values
        wape = np.sum(np.abs(preds - actuals)) / np.sum(actuals)
        mae = mean_absolute_error(actuals, preds)
        return {'wape': wape, 'mae': mae}


def hierarchical_reconcile(bottom_up_preds, proportions):
    """
    分层预测协调

    Args:
        bottom_up_preds: SKU级预测 {sku: pred}
        proportions: 各SKU占总需求的历史比例
    """
    total_pred = sum(bottom_up_preds.values())
    reconciled = {}
    for sku, pred in bottom_up_preds.items():
        # 用历史比例校准
        reconciled[sku] = total_pred * proportions.get(sku, pred / total_pred)
    return reconciled


def generate_supply_chain_data(n_skus=10, n_weeks=104, random_state=42):
    """生成供应链需求预测模拟数据"""
    np.random.seed(random_state)
    dates = pd.date_range(end='2026-05-15', periods=n_weeks, freq='W')

    data = []
    for sku_id in range(n_skus):
        base_demand = np.random.uniform(50, 200)
        trend = np.linspace(0, 20, n_weeks)
        seasonal = 30 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

        # 促销（第45、47、49周——黑五期间）
        promo = np.zeros(n_weeks)
        promo[[44, 46, 48]] = [100, 150, 80]

        noise = np.random.normal(0, 15, n_weeks)
        sales = base_demand + trend + seasonal + promo + noise
        sales = np.maximum(sales, 0)

        for i, date in enumerate(dates):
            data.append({
                'sku': f'SKU_{sku_id:03d}',
                'date': date,
                'sales': sales[i],
                'is_promo': 1 if promo[i] > 0 else 0,
                'discount_rate': 0.3 if promo[i] > 0 else 0,
                'price': 100 if promo[i] == 0 else 70
            })

    return pd.DataFrame(data)


if __name__ == '__main__':
    df = generate_supply_chain_data()
    forecaster = SupplyChainDemandForecaster()
    df_features = forecaster.build_features(df)

    feature_cols = [c for c in df_features.columns if c.startswith(('sales_lag', 'sales_ma', 'promo', 'week', 'price'))]
    train = df_features[df_features['date'] < '2026-01-01']
    test = df_features[df_features['date'] >= '2026-01-01']

    forecaster.fit(train, feature_cols)
    metrics = forecaster.evaluate(test, feature_cols)
    print(f"WAPE: {metrics['wape']:.2%}, MAE: {metrics['mae']:.1f}")
print("[✓] Demand Forecasting Supply 测试通过")
```

---


## ④ 技能关联

### 前置技能
- [Skill-Prophet-Forecasting](../03-时间序列/[[Skill-Prophet-Forecasting]].md) — Prophet 是供应链需求预测的常用基线
- [Skill-Temporal-Fusion-Transformer](../03-时间序列/[[Skill-Temporal-Fusion-Transformer]].md) — TFT 提供更精细的需求曲线

### 延伸技能
- [Skill-Safety-Stock-Replenishment](../04-供应链/[[Skill-Safety-Stock-Replenishment]].md) — 需求预测下游接安全库存计算
- [Skill-Two-Echelon-Inventory-DRL](../04-供应链/[[Skill-Two-Echelon-Inventory-DRL]].md) — 需求预测驱动多级库存优化

### 可组合
- [Skill-Monodense-单品价格弹性估计](../04-供应链/Skill-Monodense-单品价格弹性估计.md) — 弹性 + 需求预测联合定价决策


- **可组合（延伸）**：[[Skill-Bass-Diffusion-New-Product-Forecasting]] / [[Skill-Cross-Border-Cold-Start-Forecast]] / [[Skill-Category-Trend-Forecasting]] / [[Skill-Switchback-Experiment-Design]]
- 可组合：[[Skill-Predictive-Tag-Engine-Supply-Chain]]
- 可组合：[[Skill-Dynamic-ABC-Stratification-Adaptive-Policy]]

## ⑤ 商业价值评估

- **ROI**：缺货率降低50%，库存周转提升50%，年节省库存成本30万+
- **难度**：⭐⭐⭐☆☆（3/5）
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 供应链决策的起点，没有预测就没有优化
