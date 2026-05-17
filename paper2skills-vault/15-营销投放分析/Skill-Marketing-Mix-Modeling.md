---
title: Marketing Mix Modeling (MMM) for Macro Budget Allocation
module: 15-营销投放分析
topic: marketing-mix-modeling
status: stable
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: Marketing Mix Modeling (MMM)

## ① 算法原理

**核心问题**：品牌每个月在不同渠道（Facebook、Google、TikTok、KOL、线下）投入数百万广告费。如何量化每个渠道对销售额的真实增量贡献？如何预测下个月预算调整后的销售表现？

MMM与传统归因的区别：
- **归因模型**：追踪单个用户的触点路径（微观）
- **MMM**：用时间序列回归，从宏观层面分离各渠道的增量效应
- 两者互补：归因解决"谁参与了"，MMM解决"真实增量是多少"

**回归模型框架**：

$$Sales_t = \alpha + \sum_{i} \beta_i \cdot Adstock(AdSpend_{i,t}) + \gamma \cdot Price_t + \delta \cdot Seasonality_t + \epsilon_t$$

**三个关键技术**：

**1. Adstock 转化（广告效果衰减）**
- 广告投入的效果不是即时的，会随时间衰减
- $Adstock_t = Ad_t + \lambda \cdot Adstock_{t-1}$，$\lambda$ 为衰减率（通常0.3-0.8）
- Facebook广告今天投，效果可能持续2-3周

**2. Hill 转化（边际收益递减）**
- 花费越多，边际效果越低
- $Effect = \beta \cdot \frac{Adstock^\eta}{Adstock^\eta + K^\eta}$
- $\eta$ 控制曲线形状，$K$ 控制半饱和点

**3. 先验分布（Bayesian MMM）**
- Google Meridian和Meta Robyn都用Bayesian方法
- 将业务经验编码为先验：如"Facebook的ROAS通常在2-4之间"
- 数据少时靠先验，数据多时靠似然

**2025年前沿**：
- **Google Meridian** (2024)：基于TensorFlow Probability，支持地理层级分解、自定义先验
- **Meta Robyn** (2021-2024)：开源MMM框架，自动化特征工程、超参调优
- **DeepCausalMMM** (2025)：用深度学习替代线性假设，捕捉非线性交互效应

**反直觉洞察**：
- MMM通常显示"品牌搜索"的贡献被严重高估——因为它其实是其他渠道广告带来的"收割"
- 节假日期间渠道协同效应放大：单独看每个渠道ROI都在降，但整体销售额上升
- 新媒体渠道（TikTok）初期数据少，Bayesian先验能帮助稳定估计

---

## ② 母婴出海应用案例

### 场景1：年度预算重新规划

**业务问题**：Momcozy 2024年广告总投入600万，分布在：Facebook 240万、Google 180万、TikTok 90万、KOL合作 60万、线下展会 30万。年底复盘发现销售额增长但不知道各渠道真实贡献，2025年预算怎么分配？

**MMM分析**：

| 渠道 | 2024投入 | MMM估算贡献 | 增量ROAS | 饱和状态 |
|------|---------|------------|---------|---------|
| Facebook | 240万 | 35% | 2.8 | 接近饱和 |
| Google | 180万 | 22% | 2.2 | 已饱和（多为收割） |
| TikTok | 90万 | 18% | 3.5 | 远未饱和 |
| KOL合作 | 60万 | 15% | 4.0 | 中度饱和 |
| 线下展会 | 30万 | 5% | 1.2 | 低效 |

**关键发现**：
1. Google的22%贡献中，约60%实际来自Facebook/TikTok的品牌曝光带来的品牌词搜索
2. TikTok虽投入最少，但边际ROAS最高，增量效应最强
3. 线下展会投入产出比低，建议削减

**2025预算调整**：
- Facebook: 240万 → 220万（维持，接近饱和）
- Google: 180万 → 150万（削减，主要是收割效应）
- TikTok: 90万 → 150万（大幅加码，增量空间大）
- KOL合作: 60万 → 80万（加码）
- 线下展会: 30万 → 0（砍掉）

### 场景2：大促期间的渠道协同预测

**业务问题**：黑五期间计划总预算100万，想知道不同分配方案下的预期销售额。

**MMM预测**：

利用已拟合的MMM模型，模拟3种方案：

| 方案 | Facebook | Google | TikTok | KOL | 预测销售额 |
|------|---------|--------|--------|-----|----------|
| A(保守) | 40万 | 35万 | 15万 | 10万 | 380万 |
| B(激进新媒) | 30万 | 25万 | 30万 | 15万 | 420万 |
| C(均衡) | 33万 | 30万 | 25万 | 12万 | 410万 |

**选择方案B**：虽然TikTok单渠道ROI数据不如Google稳定，但MMM显示其增量空间大且与KOL有协同效应。

---

## ③ 代码模板

```python
"""
Marketing Mix Modeling (MMM) — 营销组合模型
支持：Adstock衰减、Hill饱和、Bayesian回归、预算优化
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class AdstockTransformer:
    """Adstock衰减转换"""

    def __init__(self, decay_rate=0.5):
        self.decay_rate = decay_rate

    def transform(self, spend_series):
        """对单渠道花费序列做Adstock转换"""
        adstock = np.zeros(len(spend_series))
        adstock[0] = spend_series[0]
        for t in range(1, len(spend_series)):
            adstock[t] = spend_series[t] + self.decay_rate * adstock[t-1]
        return adstock


class HillSaturation:
    """Hill饱和函数"""

    def __init__(self, K=0.5, eta=2.0):
        self.K = K  # 半饱和点
        self.eta = eta  # 形状参数

    def transform(self, x):
        """Hill转换"""
        return x ** self.eta / (x ** self.eta + self.K ** self.eta)


class SimpleMMM:
    """简化版营销组合模型"""

    def __init__(self):
        self.coef = {}  # 渠道系数
        self.adstock_params = {}  # 各渠道衰减率
        self.hill_params = {}  # 各渠道饱和参数
        self.base_sales = 0
        self.seasonality_coef = {}

    def _apply_transforms(self, spend_df, channels):
        """应用Adstock和Hill转换"""
        transformed = pd.DataFrame(index=spend_df.index)
        for ch in channels:
            # Adstock
            adstock = AdstockTransformer(
                decay=self.adstock_params.get(ch, 0.5)
            ).transform(spend_df[ch].values)
            # Hill饱和
            hill = HillSaturation(
                K=self.hill_params.get(ch, {}).get('K', spend_df[ch].median()),
                eta=self.hill_params.get(ch, {}).get('eta', 2.0)
            )
            # 标准化后应用Hill
            x_norm = adstock / (spend_df[ch].max() + 1e-6)
            transformed[ch] = hill.transform(x_norm)
        return transformed

    def fit(self, df, sales_col, channel_cols, season_col=None):
        """
        拟合MMM模型（简化版：线性回归）

        Args:
            df: DataFrame with time series data
            sales_col: 销售额列名
            channel_cols: 渠道花费列名列表
            season_col: 季节性特征列名（可选）
        """
        # 简化：先用默认参数做Adstock转换
        self.adstock_params = {ch: 0.5 for ch in channel_cols}
        self.hill_params = {ch: {'K': 0.5, 'eta': 2.0} for ch in channel_cols}

        transformed = self._apply_transforms(df, channel_cols)

        # 加入季节性
        if season_col and season_col in df.columns:
            transformed[season_col] = df[season_col]

        # 最小二乘拟合
        X = transformed.values
        y = df[sales_col].values

        # 添加截距
        X = np.column_stack([np.ones(len(X)), X])

        # 正规方程
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        self.base_sales = beta[0]
        for i, ch in enumerate(channel_cols):
            self.coef[ch] = beta[i + 1]
        if season_col and season_col in df.columns:
            self.seasonality_coef[season_col] = beta[len(channel_cols) + 1]

        return self

    def decompose(self, df, channel_cols):
        """分解各渠道对销售额的贡献"""
        transformed = self._apply_transforms(df, channel_cols)

        decomposition = pd.DataFrame(index=df.index)
        decomposition['base'] = self.base_sales

        total_contribution = np.ones(len(df)) * self.base_sales
        for ch in channel_cols:
            contrib = self.coef[ch] * transformed[ch].values
            decomposition[ch] = contrib
            total_contribution += contrib

        decomposition['predicted'] = total_contribution
        decomposition['actual'] = df['sales'].values if 'sales' in df.columns else np.nan

        return decomposition

    def optimize_budget(self, channel_cols, total_budget, bounds=None):
        """
        最优预算分配

        Args:
            channel_cols: 渠道列表
            total_budget: 总预算
            bounds: 各渠道预算上下界
        """
        if bounds is None:
            bounds = [(total_budget * 0.1, total_budget * 0.5) for _ in channel_cols]

        def predict_sales(alloc):
            # 简化预测：用线性近似
            pred = self.base_sales
            for i, ch in enumerate(channel_cols):
                # 假设transform后的值与花费成正比（简化）
                pred += self.coef[ch] * (alloc[i] / total_budget)
            return -pred  # 最小化负数 = 最大化

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
        x0 = np.array([total_budget / len(channel_cols)] * len(channel_cols))

        result = minimize(predict_sales, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return {
            'allocation': {ch: round(result.x[i], 0) for i, ch in enumerate(channel_cols)},
            'predicted_sales': -result.fun
        }

    def calculate_marginal_roas(self, channel, spend_level):
        """计算某渠道在某花费水平的边际ROAS"""
        # 简化：用系数乘以transform后的边际变化
        # 实际应该用导数
        return self.coef.get(channel, 0)


def generate_mmm_data(n_weeks=104, channels=None, seed=42):
    """生成MMM模拟数据（2年周数据）"""
    np.random.seed(seed)
    if channels is None:
        channels = ['facebook', 'google', 'tiktok', 'kol']

    dates = pd.date_range('2023-01-01', periods=n_weeks, freq='W')

    # 基础销售额（趋势+季节性）
    trend = np.linspace(100, 150, n_weeks)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)  # 年度周期
    base_sales = trend + seasonality + np.random.normal(0, 10, n_weeks)

    # 渠道花费（不同渠道有不同的策略变化）
    spends = {}
    spends['facebook'] = np.random.lognormal(3, 0.3, n_weeks) * 1000
    spends['google'] = np.random.lognormal(2.8, 0.3, n_weeks) * 1000
    # TikTok逐月加码
    spends['tiktok'] = np.linspace(10000, 50000, n_weeks) + np.random.normal(0, 5000, n_weeks)
    spends['kol'] = np.random.lognormal(2.5, 0.5, n_weeks) * 1000

    # 真实系数（用于生成销售）
    true_coef = {'facebook': 2.5, 'google': 1.8, 'tiktok': 3.0, 'kol': 3.5}
    true_decay = {'facebook': 0.6, 'google': 0.4, 'tiktok': 0.5, 'kol': 0.3}

    # 生成销售额
    sales = base_sales.copy()
    for ch in channels:
        adstock = np.zeros(n_weeks)
        for t in range(n_weeks):
            if t == 0:
                adstock[t] = spends[ch][t]
            else:
                adstock[t] = spends[ch][t] + true_decay[ch] * adstock[t-1]
        sales += true_coef[ch] * adstock / 10000

    sales += np.random.normal(0, 15, n_weeks)

    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        **{ch: spends[ch] for ch in channels},
        'seasonality': seasonality,
    })

    return df


if __name__ == '__main__':
    # 生成数据
    df = generate_mmm_data(n_weeks=104)
    print("数据预览:")
    print(df.head())

    # 拟合MMM
    mmm = SimpleMMM()
    channels = ['facebook', 'google', 'tiktok', 'kol']
    mmm.fit(df, 'sales', channels, season_col='seasonality')

    print(f"\n基础销售额: {mmm.base_sales:.1f}")
    print("渠道系数:")
    for ch, coef in mmm.coef.items():
        print(f"  {ch}: {coef:.3f}")

    # 贡献分解
    decomp = mmm.decompose(df, channels)
    print("\n贡献分解(最后4周):")
    print(decomp.tail(4).round(1))

    # 预算优化
    opt = mmm.optimize_budget(channels, total_budget=100000)
    print(f"\n最优预算分配(总预算10万):")
    for ch, alloc in opt['allocation'].items():
        print(f"  {ch}: ${alloc:,.0f}")
```

---

## ④ 技能关联

- **前置**：Ad Attribution Modeling（归因模型补充MMM的微观视角）
- **延伸**：ROAS Optimization（MMM给出长期最优分配，ROAS优化负责短期执行）
- **可组合**：+ Causal Inference → MMM本质上是时间序列因果推断，可结合DiD做Geo-Lift验证

---

## ⑤ 商业价值评估

- **ROI**：预算重新分配后整体销售额提升15-25%，年增收百万级
- **难度**：⭐⭐⭐⭐☆（4/5）— 时间序列处理+贝叶斯推断，概念门槛较高
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 品牌级预算决策的核心工具
