---
title: Promotion Effectiveness Evaluation with Causal ML
module: 15-营销投放分析
topic: promotion-effectiveness
created: 2026-05-15
updated: 2026-05-15
status: stable
---

# Skill Card: Promotion Effectiveness Evaluation

## ① 算法原理

**核心问题**：促销活动期间销售额暴涨30%，这是促销的真实效果，还是"本来就会买的用户"恰好遇到了促销？如果不做促销，销售额会是多少？促销是否侵蚀了利润？

**关键概念：反事实（Counterfactual）**
- 问题本质是一个因果推断问题
- 我们需要知道：同一个用户在"有促销"和"无促销"两种情况下的购买概率差异
- 但现实中只能观察到一种情况（fundamental problem of causal inference）

**因果ML框架（基于DoorDash KDD 2025方法）**：

**1. 双重机器学习（DML）估计平均处理效应**
- 将促销视为Treatment（T=1有促销，T=0无促销）
- 结果变量Y：购买金额/利润
- 协变量X：用户特征、历史行为、时间特征
- 模型：
  $$Y = \theta(X) \cdot T + g(X) + \epsilon$$
  $$T = m(X) + \eta$$
- $\theta(X)$ 就是条件平均处理效应（CATE）——促销对每个用户群体的增量效果
- DML用交叉拟合（cross-fitting）避免过拟合偏差

**2. Uplift Modeling识别"促销敏感"用户**
- 促销效果在不同用户间异质性很大
- "肯定会买"的用户：促销只是给了折扣（利润损失）
- "促销敏感"用户：本来不买，因为促销才买（增量收入）
- "促销反感"用户：促销反而降低购买意愿（少见但存在）

**3. 利润视角的优化**
- 增量收入 ≠ 增量利润
- 促销成本 = 折扣金额 + 运营成本
- 有效促销：Uplift收入 × 毛利率 > 促销成本

**反直觉洞察**：
- 促销的"增量"通常只有表面增长的30-50%——其余是用户的时间转移（把下个月的购买提前了）
- 最响应促销的用户往往不是最有价值的用户——高价值用户本来就会买
- "全站促销"的ROI通常远低于"定向促销"——给不需要的人发折扣是浪费
- 黑五/双十一的数据不能直接用来评估促销效果——因为同期竞品也在促销，需要对照组

---

## ② 母婴出海应用案例

### 场景1：新用户首单折扣的真实效果

**业务问题**：Momcozy对新注册用户发20%首单优惠券，使用率为35%，使用后平均订单金额$80。团队认为"首单折扣很成功"。真实增量是多少？

**因果分析**：

** naive分析**（错误）：
- 使用优惠券的用户平均消费$80
- 假设不用券会买$0 → 增量=$80
- 1000人使用 → 增量收入=$80,000

**DML因果分析**（正确）：
1. 找到"类似但未收到券"的用户作为对照组（Propensity Score Matching）
2. 控制变量：用户来源渠道、注册时间、浏览行为、 demographics
3. DML估计结果：

| 用户群 | 表面收入 | 反事实收入 | 增量收入 | 增量率 |
|--------|---------|----------|---------|--------|
| 整体 | $80 | $55 | $25 | 31% |
| 自然流量 | $85 | $70 | $15 | 18% |
| 广告流量 | $78 | $45 | $33 | 42% |
| 老用户推荐 | $82 | $65 | $17 | 21% |

**利润计算**：
- 折扣成本 = $80 × 20% = $16
- 增量毛利（按40%毛利率）= $25 × 40% = $10
- 净效果 = $10 - $16 = **-$6（亏损）**

**决策反转**：
- 表面看"优惠券带来了收入"，实际看"每笔使用券的订单亏损$6"
- 优化方向：只对"广告流量"用户发券（他们的增量率42%，净效果为正）
- 取消对自然流量用户的折扣（他们本来就会买）

### 场景2：黑五促销的边际效应递减

**业务问题**：黑五期间测试了3个折扣力度：10% off、20% off、30% off。哪个ROI最高？

**Uplift分析**：

| 折扣力度 | 转化率 | 订单量 | 平均客单价 | 增量订单 | 增量利润 |
|---------|--------|--------|----------|---------|---------|
| 无折扣 | 3% | 1000 | $100 | — | — |
| 10% off | 4.2% | 1400 | $95 | +280 | +$8,400 |
| 20% off | 5.5% | 1833 | $90 | +420 | +$8,820 |
| 30% off | 6.5% | 2167 | $85 | +400 | +$2,000 |

**分析**：
- 20% off的增量利润最高
- 30% off虽然转化率最高，但额外折扣侵蚀了利润，边际效应递减
- 最佳策略：主推20% off，对高Uplift用户定向发30% off

---

## ③ 代码模板

```python
"""
Promotion Effectiveness Evaluation — 促销效果评估
支持：DML估计CATE、Uplift Modeling、利润分析
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold


class PromotionEffectDML:
    """基于DML的促销效果评估"""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.model_y = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model_t = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.theta = None  # CATE估计
        self.residual_y_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.residual_t_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    def fit(self, X, T, Y):
        """
        拟合DML模型估计CATE

        Args:
            X: 特征矩阵 (n, d)
            T: 处理变量 (n,) — 1=有促销, 0=无促销
            Y: 结果变量 (n,) — 购买金额/利润
        """
        n = len(Y)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # 第一阶段：交叉拟合预测
        Y_hat = np.zeros(n)
        T_hat = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]

            # 拟合E[Y|X]
            self.model_y.fit(X_train, Y_train)
            Y_hat[test_idx] = self.model_y.predict(X_test)

            # 拟合E[T|X]
            self.model_t.fit(X_train, T_train)
            T_hat[test_idx] = self.model_t.predict_proba(X_test)[:, 1]

        # 计算残差
        Y_tilde = Y - Y_hat
        T_tilde = T - T_hat

        # 第二阶段：用残差估计CATE
        # theta(X) = E[Y_tilde / T_tilde | X] ≈ 用X预测Y_tilde/T_tilde
        # 简化：对每个样本直接计算局部效应，再用模型平滑
        local_effects = np.divide(Y_tilde, T_tilde,
                                  out=np.zeros_like(Y_tilde),
                                  where=T_tilde != 0)

        self.residual_y_model.fit(X, Y_tilde)
        self.residual_t_model.fit(X, T_tilde)

        # CATE估计：在点上求预测残差比
        self.X_train = X.copy()
        self.Y_tilde_train = Y_tilde.copy()
        self.T_tilde_train = T_tilde.copy()

        return self

    def predict_cate(self, X):
        """预测条件平均处理效应"""
        y_tilde_pred = self.residual_y_model.predict(X)
        t_tilde_pred = self.residual_t_model.predict(X)

        # 避免除0
        t_tilde_pred = np.where(np.abs(t_tilde_pred) < 0.01, 0.01, t_tilde_pred)
        cate = y_tilde_pred / t_tilde_pred

        return cate

    def estimate_ate(self, X):
        """估计平均处理效应"""
        cate = self.predict_cate(X)
        return np.mean(cate), np.std(cate) / np.sqrt(len(cate))


class PromotionUpliftAnalyzer:
    """促销Uplift分析器"""

    def __init__(self, profit_margin=0.4):
        self.profit_margin = profit_margin

    def analyze_segments(self, df, cate_col, segment_cols, promo_cost_col):
        """
        分群分析促销效果

        Args:
            df: DataFrame with CATE estimates
            cate_col: CATE列名
            segment_cols: 分群维度列
            promo_cost_col: 促销成本列
        """
        results = []

        for col in segment_cols:
            for segment, group in df.groupby(col):
                avg_cate = group[cate_col].mean()
                avg_cost = group[promo_cost_col].mean()
                incremental_profit = avg_cate * self.profit_margin - avg_cost

                results.append({
                    'segment_by': col,
                    'segment': segment,
                    'n_users': len(group),
                    'avg_cate': avg_cate,
                    'avg_promo_cost': avg_cost,
                    'incremental_profit': incremental_profit,
                    'roi': incremental_profit / avg_cost if avg_cost > 0 else 0,
                    'recommendation': '继续' if incremental_profit > 0 else '停止'
                })

        return pd.DataFrame(results).sort_values('incremental_profit', ascending=False)

    def find_optimal_discount(self, discounts, conversion_rates, baseline_rate,
                              aov, margin, fixed_cost=0):
        """
        找到最优折扣力度

        Args:
            discounts: 折扣率列表 [0.1, 0.2, 0.3]
            conversion_rates: 对应转化率
            baseline_rate: 无折扣转化率
            aov: 平均订单金额
            margin: 毛利率
            fixed_cost: 固定促销成本
        """
        results = []
        for d, conv in zip(discounts, conversion_rates):
            # 假设1000个用户
            n = 1000
            orders = n * conv
            baseline_orders = n * baseline_rate
            incremental_orders = orders - baseline_orders

            revenue = orders * aov * (1 - d)
            cost = orders * aov * d + fixed_cost
            incremental_revenue = incremental_orders * aov * (1 - d)
            incremental_profit = incremental_orders * aov * margin - incremental_orders * aov * d - fixed_cost

            results.append({
                'discount': d,
                'conversion_rate': conv,
                'orders': orders,
                'incremental_orders': incremental_orders,
                'revenue': revenue,
                'cost': cost,
                'incremental_profit': incremental_profit,
                'incremental_roi': incremental_profit / cost if cost > 0 else 0
            })

        return pd.DataFrame(results)


def generate_promotion_data(n=10000, seed=42):
    """生成促销效果模拟数据"""
    np.random.seed(seed)

    # 用户特征
    age = np.random.randint(18, 50, n)
    income = np.random.lognormal(10.5, 0.5, n)  # 年收入
    past_orders = np.random.poisson(3, n)
    days_since_last = np.random.exponential(30, n)
    channel = np.random.choice(['organic', 'facebook', 'google', 'tiktok'], n)
    is_new = (past_orders == 0).astype(int)

    # 促销分配（非随机，新用户和特定渠道更容易收到促销）
    promo_prob = 0.2 + 0.3 * is_new + 0.1 * (channel == 'facebook') + 0.1 * (channel == 'tiktok')
    promo = (np.random.random(n) < promo_prob).astype(int)

    # 反事实购买金额（无促销时）
    base_spend = 50 + 5 * past_orders + 0.001 * income + np.random.normal(0, 20, n)
    base_spend = np.maximum(base_spend, 0)

    # 促销效应（异质性）
    # 新用户效应强，老用户效应弱（甚至为负——时间转移）
    promo_effect = 30 * is_new + 10 * (channel == 'facebook') + 5 * (channel == 'tiktok') - 5 * (past_orders > 5)
    promo_effect = np.maximum(promo_effect, -20)  # 最低-20

    # 观察到的购买金额
    spend = base_spend + promo * promo_effect + np.random.normal(0, 15, n)
    spend = np.maximum(spend, 0)

    # 促销成本（折扣金额）
    promo_cost = promo * spend * 0.2  # 20%折扣

    df = pd.DataFrame({
        'user_id': range(n),
        'age': age,
        'income': income,
        'past_orders': past_orders,
        'days_since_last': days_since_last,
        'channel': channel,
        'is_new': is_new,
        'promo': promo,
        'spend': spend,
        'base_spend': base_spend,  # 隐藏的反事实
        'promo_effect': promo_effect,  # 隐藏的真实效应
        'promo_cost': promo_cost,
    })

    return df


if __name__ == '__main__':
    # 生成数据
    df = generate_promotion_data(n=5000)

    print("数据预览:")
    print(df[['channel', 'is_new', 'promo', 'spend', 'promo_cost']].head(10))

    # 简单对比（naive分析）
    print("\n=== Naive分析（有促销 vs 无促销）===")
    naive = df.groupby('promo')['spend'].agg(['mean', 'std', 'count'])
    print(naive)
    print(f"表面增量: ${naive.loc[1, 'mean'] - naive.loc[0, 'mean']:.2f}")

    # DML因果分析
    print("\n=== DML因果分析 ===")
    # 特征编码
    X = pd.get_dummies(df[['age', 'income', 'past_orders', 'days_since_last', 'is_new', 'channel']],
                       columns=['channel'], drop_first=True).values
    T = df['promo'].values
    Y = df['spend'].values

    dml = PromotionEffectDML(n_splits=3)
    dml.fit(X, T, Y)

    cate = dml.predict_cate(X)
    df['cate'] = cate

    ate, ate_se = dml.estimate_ate(X)
    print(f"平均处理效应(ATE): ${ate:.2f} ± ${ate_se:.2f}")
    print(f"真实平均效应: ${df['promo_effect'].mean():.2f}")

    # 分群Uplift分析
    analyzer = PromotionUpliftAnalyzer(profit_margin=0.4)
    segment_results = analyzer.analyze_segments(
        df, 'cate', ['channel', 'is_new'], 'promo_cost'
    )
    print("\n=== 分群促销效果 ===")
    print(segment_results[['segment_by', 'segment', 'n_users', 'avg_cate',
                           'incremental_profit', 'recommendation']].to_string(index=False))

    # 折扣力度优化
    print("\n=== 折扣力度优化 ===")
    discounts = [0.1, 0.2, 0.3]
    conv_rates = [0.042, 0.055, 0.065]  # 对应转化率
    optimal = analyzer.find_optimal_discount(
        discounts, conv_rates, baseline_rate=0.03,
        aov=100, margin=0.4, fixed_cost=500
    )
    print(optimal[['discount', 'incremental_orders', 'incremental_profit', 'incremental_roi']].to_string(index=False))
```

---


## ④ 技能关联

### 前置技能
- [Skill-Marketing-Mix-Modeling](../15-营销投放分析/Skill-Marketing-Mix-Modeling.md) — MMM 提供渠道基线，促销在此基线上叠加
- [Skill-Intelligent-Prediction-Doubly-Robust](../03-时间序列/Skill-Intelligent-Prediction-Doubly-Robust.md) — DR 估计是促销因果效应的核心方法

### 延伸技能
- [Skill-Monodense-单品价格弹性估计](../04-供应链/Skill-Monodense-单品价格弹性估计.md) — 促销下沉到 SKU 级别价格弹性

### 可组合
- [Skill-Ad-Attribution-Modeling](../13-广告分析/Skill-Ad-Attribution-Modeling.md) — 归因 + 促销因果联合给出 ROI 全景

## ⑤ 商业价值评估

- **ROI**：识别无效促销支出后，可削减20-40%的促销浪费，年节省数十万
- **难度**：⭐⭐⭐⭐☆（4/5）— 因果推断概念门槛高，DML实现复杂
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 避免"促销幻觉"，确保每一分折扣都产生真实增量
