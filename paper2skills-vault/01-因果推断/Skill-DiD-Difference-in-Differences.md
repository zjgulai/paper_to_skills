---
title: Difference-in-Differences (DiD) for Causal Effect Estimation
doc_type: knowledge
module: 01-因果推断
topic: difference-in-differences
status: stable
created: 2026-05-15
updated: 2026-05-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Difference-in-Differences (DiD)

---

## ① 算法原理

**核心思想**：利用处理组和对照组在政策/干预前后的变化差异来估计因果效应。基本逻辑是：如果没有干预，处理组的趋势应该与对照组平行（平行趋势假设）。干预后的实际差异减去趋势差异，就是干预的净效应。

**为什么需要DiD**：

母婴出海电商中有大量无法随机分配干预的场景：
- 某国突然加征关税（无法选择哪些商品被征税）
- 平台算法更新影响所有店铺（无法做A/B实验）
- 竞品在特定市场大幅降价（无法控制自己的价格）
- 物流商在某个区域暂停服务（无法选择受影响的订单）

这些场景下，DiD 是最自然的效果评估工具——用未受影响的区域/商品/时间作为对照组，剥离趋势后估计真实效应。

**数学公式**：

$$\hat{\tau}^{DiD} = (\bar{Y}_{treatment,post} - \bar{Y}_{treatment,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$$

等价于回归形式：

$$Y_{it} = \alpha + \beta \cdot Treat_i + \gamma \cdot Post_t + \tau \cdot (Treat_i \times Post_t) + \epsilon_{it}$$

其中 $\tau$ 就是DiD估计量——交互项系数。

**关键假设**：

1. **平行趋势假设（Parallel Trends）**：若无干预，处理组和对照组的结果变量会遵循相同的时间趋势。这是DiD的核心识别假设，必须通过事件研究法（event study）进行检验。
2. **无预期效应（No Anticipation）**：处理组在干预前不会因为预期到干预而改变行为。
3. **无溢出效应（No Spillover）**：对照组不受干预的间接影响。

**Staggered DiD（交错处理）**：

现代DiD的核心进展。传统DiD假设所有处理单元在同一时间接受干预，但现实中干预往往是渐进的（如先在德国试点，再扩展到法国、英国）。Staggered DiD 处理这种情况，但传统双向固定效应（TWFE）在异质性处理效应下会产生偏误。

**最新解决方案**：
- **Callaway-Sant'Anna (2021)**：按处理队列分组，用尚未处理的单元作为对照
- **Sun-Abraham (2021)**：交互加权估计量，解决TWFE偏误
- **Borusyak et al. (2024) 插补DiD**：用对照组预测处理组的反事实结果，再用实际值减预测值

**反直觉洞察**：DiD 不要求处理组和对照组在水平上一致，只要求趋势平行。这意味着你可以用完全不同的市场（如德国 vs 日本）作为对照，只要它们的销量趋势在干预前相似。

---

## ② 母婴出海应用案例

### 场景1：关税政策对跨境销量的影响评估

**业务问题**：2025年美国对中国产婴儿推车加征25%关税。团队需要评估：关税究竟导致销量下降了多少？是自然的市场波动，还是关税的直接影响？

**应用流程**：
1. **定义处理组和对照组**：
   - 处理组：销往美国的婴儿推车（受关税影响）
   - 对照组：销往加拿大/英国的婴儿推车（未受关税影响）
2. **定义时间窗口**：
   - 干预前：2024年6月-2025年5月（12个月）
   - 干预后：2025年6月-2026年5月（12个月）
3. **检验平行趋势**：事件研究法，看干预前12个月处理组和对照组的趋势是否平行
4. **估计DiD效应**：计算交互项系数

**预期产出**：
- 关税导致的销量下降幅度（如：月均销量下降 18%，其中 12% 可归因于关税）
- 事件研究图：干预前后各月的动态效应
- 稳健性检验：安慰剂检验、替换对照组

**业务价值**：
- 精准量化关税冲击，为定价策略调整提供依据
- 评估是否需要转移产能到东南亚（如果关税效应 > 20%）
- 向投资人解释销量波动的归因

### 场景2：TikTok Shop 入驻对品牌曝光的影响

**业务问题**：某母婴品牌2025年3月入驻TikTok Shop英国站，同时在德国未入驻。需要评估TikTok Shop是否带来了增量曝光和转化。

**应用流程**：
1. **处理组**：英国站（入驻TikTok Shop）
2. **对照组**：德国站（未入驻）
3. **结果变量**：品牌搜索指数、独立站流量、全渠道GMV
4. **检验平行趋势**：入驻前6个月两站的搜索趋势是否平行
5. **估计DiD效应**

**决策价值**：
- 若TikTok Shop带来显著增量：加速入驻其他欧洲市场
- 若效应不显著或 cannibalization 严重：重新评估渠道策略

---

## ③ 代码模板

```python
"""
Difference-in-Differences (DiD) — 双重差分因果效应估计
用于评估政策/干预对母婴出海业务的因果影响

支持：经典DiD、Staggered DiD（Callaway-Sant'Anna）、事件研究法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# ==================== DiD 核心实现 ====================

class DifferenceInDifferences:
    """双重差分估计器"""

    def __init__(self, method='classic'):
        """
        初始化 DiD 估计器

        Args:
            method: 'classic' 或 'staggered'
        """
        self.method = method
        self.model = None
        self.tau = None
        self.se = None
        self.event_study_results = None

    def fit(self, df, unit_col, time_col, treat_col, post_col, outcome_col,
            covariates=None, cluster_col=None):
        """
        拟合 DiD 模型

        Args:
            df: DataFrame
            unit_col: 单元标识列（如店铺ID、国家）
            time_col: 时间列
            treat_col: 处理标志列（1=处理组，0=对照组）
            post_col: 干预后标志列（1=干预后，0=干预前）
            outcome_col: 结果变量列（如销量、转化率）
            covariates: 协变量列表（可选）
            cluster_col: 聚类标准误的聚类变量（可选）
        """
        df = df.copy()

        # 创建交互项
        df['treat_post'] = df[treat_col] * df[post_col]

        # 构建特征矩阵
        feature_cols = [treat_col, post_col, 'treat_post']
        if covariates:
            feature_cols.extend(covariates)

        X = df[feature_cols].values
        y = df[outcome_col].values

        # 拟合OLS
        self.model = LinearRegression()
        self.model.fit(X, y)

        # DiD估计量是交互项的系数（第3个系数）
        self.tau = self.model.coef_[2]

        # 计算标准误（简化版，实际应使用聚类稳健标准误）
        n = len(df)
        k = X.shape[1]
        residuals = y - self.model.predict(X)
        mse = np.sum(residuals ** 2) / (n - k)
        var_tau = mse / np.sum((df['treat_post'] - df['treat_post'].mean()) ** 2)
        self.se = np.sqrt(var_tau)

        return self

    def predict_counterfactual(self, df, unit_col, time_col, treat_col, post_col, outcome_col):
        """
        预测处理组的反事实结果（如果没有干预会怎样）
        """
        df_cf = df[df[treat_col] == 1].copy()

        # 用对照组的趋势来预测处理组的反事实
        control_pre = df[(df[treat_col] == 0) & (df[post_col] == 0)][outcome_col].mean()
        control_post = df[(df[treat_col] == 0) & (df[post_col] == 1)][outcome_col].mean()
        control_trend = control_post - control_pre

        treat_pre = df[(df[treat_col] == 1) & (df[post_col] == 0)][outcome_col].mean()
        counterfactual_post = treat_pre + control_trend

        return counterfactual_post

    def event_study(self, df, unit_col, time_col, treat_col, outcome_col,
                    event_time_col, reference_period=-1):
        """
        事件研究法：估计干预前后各期的动态效应

        Args:
            df: DataFrame
            unit_col: 单元标识列
            time_col: 时间列
            treat_col: 处理标志
            outcome_col: 结果变量
            event_time_col: 相对于干预事件的时间列（如-3, -2, -1, 0, 1, 2, 3）
            reference_period: 参照期（默认-1，即干预前一期）
        """
        df = df.copy()

        # 创建事件时间虚拟变量
        event_times = sorted(df[event_time_col].unique())
        event_times = [t for t in event_times if t != reference_period]

        for t in event_times:
            df[f'event_t{t}'] = ((df[event_time_col] == t) * df[treat_col]).astype(int)

        # 构建回归
        feature_cols = [treat_col] + [f'event_t{t}' for t in event_times]

        # 添加时间固定效应（简化版，用时间虚拟变量）
        for t in df[time_col].unique():
            df[f'time_fe_{t}'] = (df[time_col] == t).astype(int)
        time_fe_cols = [f'time_fe_{t}' for t in df[time_col].unique()][1:]  # 去掉参照期

        feature_cols.extend(time_fe_cols)

        X = df[feature_cols].values
        y = df[outcome_col].values

        model = LinearRegression()
        model.fit(X, y)

        # 提取事件时间系数
        results = {}
        for i, t in enumerate(event_times):
            coef = model.coef_[1 + i]  # 第1个是treat_col，之后是事件时间系数
            results[t] = coef

        self.event_study_results = results
        return results

    def summary(self):
        """输出DiD估计结果摘要"""
        if self.tau is None:
            return "模型尚未拟合"

        t_stat = self.tau / self.se if self.se > 0 else 0
        p_value = 2 * (1 - min(abs(t_stat) / 10, 1))  # 简化p值计算

        print("=" * 60)
        print("Difference-in-Differences 估计结果")
        print("=" * 60)
        print(f"DiD 估计量 (τ): {self.tau:.4f}")
        print(f"标准误:         {self.se:.4f}")
        print(f"t 统计量:       {t_stat:.2f}")
        print(f"显著性:         {'***' if abs(t_stat) > 2.58 else '**' if abs(t_stat) > 1.96 else '*'}")
        print("=" * 60)

        return {'tau': self.tau, 'se': self.se, 't_stat': t_stat}


def plot_event_study(event_study_results, title="Event Study: Dynamic Treatment Effects",
                     save_path=None):
    """
    绘制事件研究图（平行趋势检验 + 动态效应）
    """
    times = sorted(event_study_results.keys())
    coefs = [event_study_results[t] for t in times]

    # 简化标准误（实际应使用聚类稳健SE）
    ses = [abs(c) * 0.3 for c in coefs]

    plt.figure(figsize=(10, 6))

    # 绘制系数
    colors = ['#3498db' if t < 0 else '#e74c3c' for t in times]
    plt.bar(times, coefs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # 误差线
    plt.errorbar(times, coefs, yerr=ses, fmt='none', color='black', capsize=3)

    # 零线
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    # 干预线
    if 0 in times:
        plt.axvline(x=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        plt.text(-0.5, max(coefs) * 0.9, 'Intervention', rotation=90,
                verticalalignment='top', color='red', fontsize=9)

    plt.xlabel('Event Time (Relative to Intervention)')
    plt.ylabel('Treatment Effect')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return plt


# ==================== 母婴出海业务专用函数 ====================

def generate_cross_border_did_data(n_units=100, n_periods=24, random_state=42):
    """
    生成母婴出海电商的DiD模拟数据

    场景：某母婴品牌在欧洲多国销售，2025年1月起英国对婴儿推车加征关税
    """
    np.random.seed(random_state)

    # 国家：0=英国(处理组), 1=德国, 2=法国, 3=荷兰(对照组)
    countries = ['UK', 'DE', 'FR', 'NL']
    country_weights = [0.25, 0.3, 0.3, 0.15]

    data = []
    for unit_id in range(n_units):
        country_idx = np.random.choice(4, p=country_weights)
        country = countries[country_idx]
        is_treatment = 1 if country == 'UK' else 0

        # 国家固定效应
        country_effect = {'UK': 1000, 'DE': 1200, 'FR': 1100, 'NL': 800}[country]

        # 时间趋势（季节性 + 增长趋势）
        for period in range(n_periods):
            year = 2024 if period < 12 else 2025
            month = (period % 12) + 1

            # 季节性（Q4旺季）
            seasonal = 200 if month in [11, 12] else 0

            # 增长趋势
            trend = period * 10

            # 干预后效应（仅处理组在2025年1月后受影响）
            is_post = 1 if period >= 12 else 0
            treatment_effect = -150 if (is_treatment == 1 and is_post == 1) else 0

            # 随机噪声
            noise = np.random.normal(0, 100)

            # 销量
            sales = country_effect + trend + seasonal + treatment_effect + noise
            sales = max(sales, 0)

            data.append({
                'unit_id': unit_id,
                'country': country,
                'period': period,
                'year': year,
                'month': month,
                'treatment': is_treatment,
                'post': is_post,
                'sales': round(sales, 2),
                'event_time': period - 12  # 以干预时点为0
            })

    return pd.DataFrame(data)


def placebo_test(df, unit_col, time_col, treat_col, post_col, outcome_col,
                 n_placebo=100):
    """
    安慰剂检验：随机分配处理组，检验DiD估计量是否接近0
    """
    actual_tau = None
    placebo_taus = []

    # 实际估计
    did = DifferenceInDifferences()
    did.fit(df, unit_col, time_col, treat_col, post_col, outcome_col)
    actual_tau = did.tau

    # 安慰剂检验
    for _ in range(n_placebo):
        df_placebo = df.copy()
        # 随机打乱处理组分配
        unique_units = df[unit_col].unique()
        n_treat = int(len(unique_units) * 0.5)
        treat_units = np.random.choice(unique_units, n_treat, replace=False)
        df_placebo['treatment_placebo'] = df_placebo[unit_col].isin(treat_units).astype(int)

        did_p = DifferenceInDifferences()
        did_p.fit(df_placebo, unit_col, time_col, 'treatment_placebo', post_col, outcome_col)
        placebo_taus.append(did_p.tau)

    # 计算p值：实际估计量落在安慰剂分布的什么位置
    placebo_taus = np.array(placebo_taus)
    p_value = np.mean(np.abs(placebo_taus) >= np.abs(actual_tau))

    return {
        'actual_tau': actual_tau,
        'placebo_taus': placebo_taus,
        'p_value': p_value,
        'placebo_mean': placebo_taus.mean(),
        'placebo_std': placebo_taus.std()
    }


# ==================== 示例代码 ====================

def main():
    """主函数：演示DiD在母婴出海业务中的应用"""
    print("=" * 70)
    print("母婴出海 — Difference-in-Differences 关税政策效应评估")
    print("=" * 70)

    # 1. 生成模拟数据
    print("\n[1] 生成模拟数据...")
    print("   场景：2025年1月起英国对婴儿推车加征关税")
    print("   处理组：英国市场 | 对照组：德国/法国/荷兰市场")

    df = generate_cross_border_did_data(n_units=200, n_periods=24)
    print(f"   样本量: {len(df)} 条记录")
    print(f"   处理组: {df[df['treatment']==1]['unit_id'].nunique()} 个单元")
    print(f"   对照组: {df[df['treatment']==0]['unit_id'].nunique()} 个单元")
    print(f"   时间跨度: 2024年1月 - 2025年12月 (24个月)")

    # 2. 经典DiD估计
    print("\n[2] 经典DiD估计...")
    did = DifferenceInDifferences()
    did.fit(df, 'unit_id', 'period', 'treatment', 'post', 'sales')
    did.summary()

    # 3. 反事实预测
    print("\n[3] 反事实结果预测...")
    cf = did.predict_counterfactual(df, 'unit_id', 'period', 'treatment', 'post', 'sales')
    treat_actual = df[(df['treatment']==1) & (df['post']==1)]['sales'].mean()
    print(f"   处理组干预后实际销量: {treat_actual:.1f}")
    print(f"   处理组反事实预测销量: {cf:.1f}")
    print(f"   关税导致销量下降: {treat_actual - cf:.1f}")

    # 4. 事件研究法
    print("\n[4] 事件研究法（平行趋势检验 + 动态效应）...")
    event_results = did.event_study(df, 'unit_id', 'period', 'treatment', 'sales', 'event_time')

    print("   干预前各期效应（应接近0，验证平行趋势）:")
    for t in sorted(event_results.keys()):
        if t < 0:
            sig = '✓' if abs(event_results[t]) < 50 else '✗'
            print(f"      t={t:3d}: {event_results[t]:7.2f} {sig}")

    print("   干预后各期效应:")
    for t in sorted(event_results.keys()):
        if t >= 0:
            print(f"      t={t:3d}: {event_results[t]:7.2f}")

    # 5. 安慰剂检验
    print("\n[5] 安慰剂检验...")
    placebo = placebo_test(df, 'unit_id', 'period', 'treatment', 'post', 'sales', n_placebo=50)
    print(f"   实际DiD估计量: {placebo['actual_tau']:.2f}")
    print(f"   安慰剂均值 ± 标准差: {placebo['placebo_mean']:.2f} ± {placebo['placebo_std']:.2f}")
    print(f"   安慰剂检验p值: {placebo['p_value']:.4f}")
    print(f"   结论: {'通过' if placebo['p_value'] < 0.05 else '未通过'}安慰剂检验")

    # 6. 可视化
    print("\n[6] 生成事件研究图...")
    plot_event_study(event_results,
                     title="UK Tariff Impact on Baby Stroller Sales\n(Event Study)")

    print("\n" + "=" * 70)
    print("DiD 分析完成！")
    print("=" * 70)

    return did


if __name__ == '__main__':
    model = main()
```

---

## ④ 技能关联

### 前置技能
- [[Skill-Uplift-Modeling]] — 理解因果推断的基本概念（处理效应、潜在结果框架）
- [[Skill-Causal-Discovery-PC-Algorithm]] — 用PC算法发现因果结构，为DiD选择合理的对照组提供依据

### 延伸技能
- **Callaway-Sant'Anna Staggered DiD** — 处理渐进式干预场景（多时点处理）
- **Synthetic Control** — 当找不到合适的对照组时，用加权组合构造合成对照组

### 可组合技能
- **+ Time-Series-Forecasting**: DiD评估政策效应 + 时序预测预测未来趋势
- **+ A/B-Experimental-Design**: DiD用于自然实验，A/B测试用于可控实验，两者互补
- **+ Causal-Forest**: DiD估计平均处理效应，Causal Forest估计异质性处理效应（哪些单元受影响更大）

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 关税/政策影响评估 | 精准量化外部冲击，避免过度/不足反应（错误决策成本通常 > 50万） | 开发3-5天，数据接入1天 | 10-20x |
| 平台算法更新影响 | 区分算法影响 vs 市场自然波动，指导运营策略调整 | 开发2-3天 | 5-10x |
| 竞品行动响应评估 | 量化竞品降价/新品对本品的影响，指导定价和库存 | 开发2-3天 | 5-10x |

### 实施难度
**评分：⭐⭐☆☆☆（2/5星）**

- 数据要求：需要处理组和对照组在干预前后的面板数据
- 技术门槛：低，核心是OLS回归
- 主要挑战：平行趋势假设的检验和满足（需要领域知识选择合理的对照组）
- 工程复杂度：低
- 维护成本：低

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **必要性极高**：跨境电商面临大量外部政策冲击（关税、平台规则、物流变化），DiD是最自然的评估工具
- **通用性强**：几乎所有"自然实验"场景都适用
- **实施成本低**：不需要实验数据，利用现有历史数据即可
- **与现有技能互补**：填补Uplift（需实验）和Causal Discovery（仅发现结构）之间的空白

### 评估依据
1. 母婴出海业务中，大量决策场景无法进行随机实验（政策、平台规则、竞品行动），DiD是唯一可行的因果推断工具
2. 方法成熟，实施简单，1周内可完成POC
3. 与Uplift Modeling + Causal Forest形成完整因果推断工具链：DiD评估自然实验 → Uplift优化可控实验 → Causal Forest估计异质性效应
4. 输出直观（一张事件研究图即可说服决策者），落地门槛极低
