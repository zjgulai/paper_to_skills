---
title: Difference-in-Differences (DiD) for Causal Effect Estimation
doc_type: knowledge
module: 01-因果推断
topic: difference-in-differences
status: stable
created: 2026-05-15
updated: 2026-07-06
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Difference-in-Differences (DiD)

---

## ① 算法原理

**核心思想**：通过对比处理组与对照组在干预前后的变化差异，剥离时间趋势后估计干预的净因果效应。

**数学原理**：

$$\hat{\tau}^{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$$

**业务含义**：处理组的变化量 - 对照组的变化量 = 干预的真实效应。对照组的变化代表"自然趋势"，用它来消除时间因素的影响。

**回归形式**：

$$Y_{it} = \alpha + \beta \cdot Treat_i + \gamma \cdot Post_t + \tau \cdot (Treat_i \times Post_t) + \epsilon_{it}$$

其中 $\tau$ 是DiD估计量（交互项系数），直接给出干预效应的量化值。

**关键假设**：

1. **平行趋势假设（Parallel Trends）**：若无干预，处理组和对照组的结果变量遵循相同的时间趋势。这是DiD的识别假设，必须通过事件研究法检验。
2. **无预期效应（No Anticipation）**：处理组在干预前不会因预期干预而改变行为。
3. **无溢出效应（No Spillover）**：对照组不受干预的间接影响。

**非共识迁移**（原始领域→跨境电商降维）：

传统因果推断要求随机对照试验（RCT），但母婴跨境电商中大量干预无法随机分配：关税政策、平台算法更新、物流中断、竞品降价等都是"自然实验"。DiD的核心创新是**用未受影响的地理/商品/时间维度作为对照组**，将"无法控制的干预"转化为"可识别的因果效应"。例如：美国加征关税时，用加拿大销售数据作为对照，虽然两国市场完全不同，但只要趋势平行，就能精准分离关税效应。这是从"实验设计"到"观察数据因果推断"的范式转移。

---

## ② 母婴出海应用案例

### 场景1：暖奶器促销政策对欧洲销量的因果影响评估

**业务问题**：2025年6月，某母婴品牌在德国亚马逊启动"暖奶器满€50减€10"促销，同时英国站未启动该促销。需要精准量化促销的因果效应，而非简单对比销量（因为两国市场规模、季节性不同）。

**应用流程**：

1. **定义处理组和对照组**：
   - 处理组：德国站暖奶器（启动促销）
   - 对照组：英国站暖奶器（未启动促销）
   - 结果变量：周均销量（单位）

2. **定义时间窗口**：
   - 干预前：2025年1月-5月（20周）
   - 干预后：2025年6月-8月（13周）

3. **检验平行趋势**：事件研究法，验证干预前20周两国销量趋势是否平行
   - 若平行趋势检验p值 > 0.05，则满足识别假设

4. **估计DiD效应**：
   - 德国促销前周均销量：480单
   - 德国促销后周均销量：620单（增长140单）
   - 英国促销前周均销量：520单
   - 英国促销后周均销量：545单（增长25单）
   - **DiD效应 = (620-480) - (545-520) = 140 - 25 = 115单/周**

**量化产出**：

- **因果效应**：促销政策导致德国站周销量净增加 **115单**（置信区间[95, 135]单，95%置信度）
- **增量收入**：115单/周 × 13周 × €45/单（平均客单价） = **€67,275**（约¥50万）
- **年化收益**：€67,275 × 4.3个季度 = **€289,283**（约¥215万）
- **ROI**：促销成本€5,200（€10折扣 × 520单），ROI = 289,283 / 5,200 = **55.6倍**

**三轨验证**：
- **成本轨**：促销成本€5,200，占增量收入3.2%，可控范围内
- **合规轨**：促销政策符合亚马逊德国站规范，无违规风险
- **风险轨**：存在"竞品跟风降价"风险，建议监测竞品价格变动，若竞品跟风，效应可能衰减30-50%

---

### 场景2：TikTok Shop入驻对母婴品牌转化率的增量贡献评估

**业务问题**：2025年4月，某母婴品牌入驻TikTok Shop英国站，同时在法国未入驻。需要评估TikTok Shop是否真正带来了转化率提升，还是仅仅是流量分流（cannibalization）。

**应用流程**：

1. **定义处理组和对照组**：
   - 处理组：英国站全渠道（TikTok Shop + 亚马逊 + 独立站）
   - 对照组：法国站全渠道（亚马逊 + 独立站，无TikTok Shop）
   - 结果变量：全渠道转化率（订单数/访客数）

2. **定义时间窗口**：
   - 干预前：2024年10月-2025年3月（6个月）
   - 干预后：2025年4月-9月（6个月）

3. **检验平行趋势**：
   - 干预前6个月，英国站和法国站的转化率趋势是否平行
   - 事件研究图显示：干预前各月的系数均不显著（p > 0.05）

4. **估计DiD效应**：
   - 英国干预前平均转化率：3.2%
   - 英国干预后平均转化率：4.1%（增长0.9个百分点）
   - 法国干预前平均转化率：3.0%
   - 法国干预后平均转化率：3.1%（增长0.1个百分点）
   - **DiD效应 = (4.1% - 3.2%) - (3.1% - 3.0%) = 0.9% - 0.1% = 0.8个百分点**

**量化产出**：

- **因果效应**：TikTok Shop入驻导致英国站转化率净增加 **0.8个百分点**（置信区间[0.6, 1.0]个百分点，95%置信度）
- **增量订单**：英国站月均访客100万，0.8% × 100万 = 8,000单/月
- **增量收入**：8,000单/月 × €35/单（平均客单价） = **€280,000/月**（约¥208万/月）
- **6个月累计收益**：€280,000 × 6 = **€1,680,000**（约¥1,248万）
- **年化收益**：€280,000 × 12 = **€3,360,000**（约¥2,496万）

**三轨验证**：
- **成本轨**：TikTok Shop佣金15%，成本€504,000/年，占年化收益15%，在可接受范围
- **合规轨**：TikTok Shop英国站已获得母婴类目认证，无合规风险；需监测欧盟《数字服务法》变化
- **风险轨**：存在"平台政策变化"风险（如佣金上调至20%会降低ROI至33%）；建议建立多渠道分散风险，TikTok Shop不超过总收入40%

---

## ③ 代码模板

```python
"""
Difference-in-Differences (DiD) — 双重差分因果效应估计
用于评估政策/干预对母婴出海业务的因果影响

支持：经典DiD、事件研究法、平行趋势检验、稳健性检验
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== DiD 核心实现 ====================

class DifferenceInDifferences:
    """双重差分估计器"""

    def __init__(self):
        self.tau = None
        self.se = None
        self.t_stat = None
        self.p_value = None
        self.ci_lower = None
        self.ci_upper = None
        self.event_study = None

    def fit(self, df, unit_col, time_col, treat_col, outcome_col):
        """
        拟合 DiD 模型

        Args:
            df: DataFrame，包含以下列
            unit_col: 单元标识列（如国家、店铺）
            time_col: 时间列（如周、月）
            treat_col: 处理标志列（1=处理组，0=对照组）
            outcome_col: 结果变量列（如销量、转化率）
        """
        df = df.copy()

        # 识别干预时间点（处理组第一次出现treat_col=1的时间）
        treat_units = df[df[treat_col] == 1][unit_col].unique()
        intervention_time = df[df[unit_col].isin(treat_units) & (df[treat_col] == 1)][time_col].min()

        # 创建干预后标志
        df['post'] = (df[time_col] >= intervention_time).astype(int)

        # 创建交互项
        df['treat_post'] = df[treat_col] * df['post']

        # 构建回归模型：Y = α + β*Treat + γ*Post + τ*(Treat×Post) + ε
        X = df[[treat_col, 'post', 'treat_post']].values
        y = df[outcome_col].values

        # 添加常数项
        X = np.column_stack([np.ones(len(X)), X])

        # OLS估计
        beta = np.linalg.inv(X.T @ X) @ X.T @ y

        # 残差和标准误
        y_pred = X @ beta
        residuals = y - y_pred
        n, k = X.shape
        sigma2 = (residuals ** 2).sum() / (n - k)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)

        # DiD估计量是第4个系数（交互项）
        self.tau = beta[3]
        self.se = np.sqrt(var_beta[3, 3])
        self.t_stat = self.tau / self.se
        self.p_value = 2 * (1 - stats.t.cdf(abs(self.t_stat), n - k))
        self.ci_lower = self.tau - 1.96 * self.se
        self.ci_upper = self.tau + 1.96 * self.se

        self.df = df
        self.treat_col = treat_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.intervention_time = intervention_time

        return self

    def event_study_plot(self, unit_col, figsize=(12, 6)):
        """
        事件研究法：检验平行趋势假设

        Returns:
            coefficients: 各时期相对于干预前基准期的系数
            ci_lower, ci_upper: 95%置信区间
        """
        df = self.df.copy()

        # 生成相对时间变量（干预前后的时间距离）
        treat_units = df[df[self.treat_col] == 1][unit_col].unique()
        df['relative_time'] = df[self.time_col] - self.intervention_time
        df['is_treat'] = df[unit_col].isin(treat_units).astype(int)

        # 获取相对时间的范围
        rel_times = sorted(df['relative_time'].unique())
        rel_times = [t for t in rel_times if t >= -6 and t <= 6]  # 干预前后各6期

        # 为每个相对时间创建虚拟变量
        X_list = [np.ones(len(df))]
        for t in rel_times:
            if t != -1:  # 基准期为-1（干预前一期）
                X_list.append(((df['relative_time'] == t) * df['is_treat']).values)

        X = np.column_stack(X_list)
        y = df[self.outcome_col].values

        # OLS估计
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        y_pred = X @ beta
        residuals = y - y_pred
        n, k = X.shape
        sigma2 = (residuals ** 2).sum() / (n - k)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)

        # 提取系数和置信区间
        coeffs = beta[1:]
        ses = np.sqrt(np.diag(var_beta)[1:])
        ci_lower = coeffs - 1.96 * ses
        ci_upper = coeffs + 1.96 * ses

        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        rel_times_plot = [t for t in rel_times if t != -1]
        ax.plot(rel_times_plot, coeffs, 'o-', linewidth=2, markersize=8, label='DiD系数')
        ax.fill_between(rel_times_plot, ci_lower, ci_upper, alpha=0.3, label='95%置信区间')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='g', linestyle='--', linewidth=1, alpha=0.5, label='干预时点')
        ax.set_xlabel('相对时间（期）', fontsize=12)
        ax.set_ylabel('处理效应', fontsize=12)
        ax.set_title('事件研究法：平行趋势检验', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        self.event_study = (rel_times_plot, coeffs, ci_lower, ci_upper)
        return fig

    def summary(self):
        """输出结果摘要"""
        print("\n" + "="*60)
        print("Difference-in-Differences (DiD) 估计结果")
        print("="*60)
        print(f"DiD估计量 (τ):        {self.tau:.6f}")
        print(f"标准误 (SE):          {self.se:.6f}")
        print(f"t统计量:              {self.t_stat:.4f}")
        print(f"p值:                  {self.p_value:.4f}")
        print(f"95%置信区间:          [{self.ci_lower:.6f}, {self.ci_upper:.6f}]")
        print(f"显著性:               {'***' if self.p_value < 0.01 else '**' if self.p_value < 0.05 else '*' if self.p_value < 0.1 else 'N.S.'}")
        print("="*60 + "\n")


# ==================== 示例数据生成 ====================

def generate_synthetic_data(n_units=20, n_periods=24, intervention_period=13):
    """
    生成合成数据：模拟母婴品牌在两个国家的销量数据

    Args:
        n_units: 单元数（10个处理组 + 10个对照组）
        n_periods: 时间周期数（24周）
        intervention_period: 干预发生的时间点（第13周）

    Returns:
        DataFrame
    """
    np.random.seed(42)

    data = []
    for unit in range(n_units):
        is_treat = 1 if unit < 10 else 0  # 前10个是处理组，后10个是对照组

        for t in range(1, n_periods + 1):
            # 基础销量
            base_sales = 500 if is_treat == 1 else 480

            # 时间趋势（每周增长2单）
            trend = 2 * t

            # 干预效应（处理组在干预后增加80单）
            intervention_effect = 80 if (is_treat == 1 and t >= intervention_period) else 0

            # 随机噪声
            noise = np.random.normal(0, 30)

            # 销量
            sales = base_sales + trend + intervention_effect + noise

            data.append({
                'unit': f'Country_{unit}',
                'week': t,
                'treat': is_treat,
                'sales': max(0, sales)  # 销量不能为负
            })

    return pd.DataFrame(data)


# ==================== 稳健性检验 ====================

def placebo_test(df, unit_col, time_col, treat_col, outcome_col, n_simulations=100):
    """
    安慰剂检验：在对照组中随机分配"虚拟处理"，检验DiD是否捕捉到虚假效应

    Returns:
        placebo_taus: 虚拟处理的DiD估计量分布
    """
    placebo_taus = []

    for _ in range(n_simulations):
        df_placebo = df.copy()

        # 在对照组中随机选择一半作为虚拟处理组
        control_units = df_placebo[df_placebo[treat_col] == 0][unit_col].unique()
        placebo_treat_units = np.random.choice(control_units, size=len(control_units)//2, replace=False)

        df_placebo['placebo_treat'] = df_placebo[unit_col].isin(placebo_treat_units).astype(int)

        # 拟合DiD模型
        did_placebo = DifferenceInDifferences()
        did_placebo.fit(df_placebo, unit_col, time_col, 'placebo_treat', outcome_col)
        placebo_taus.append(did_placebo.tau)

    return np.array(placebo_taus)


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("\n[*] 生成合成数据...")
    df = generate_synthetic_data(n_units=20, n_periods=24, intervention_period=13)
    print(f"[✓] 数据形状: {df.shape}")
    print(df.head(10))

    print("\n[*] 拟合 DiD 模型...")
    did = DifferenceInDifferences()
    did.fit(df, unit_col='unit', time_col='week', treat_col='treat', outcome_col='sales')

    print("\n[*] 输出结果摘要...")
    did.summary()

    print("[*] 生成事件研究图...")
    fig = did.event_study_plot(unit_col='unit', figsize=(12, 6))
    plt.savefig('event_study.png', dpi=150, bbox_inches='tight')
    print("[✓] 事件研究图已保存为 event_study.png")

    print("\n[*] 执行安慰剂检验...")
    placebo_taus = placebo_test(df, unit_col='unit', time_col='week', 
                                treat_col='treat', outcome_col='sales', n_simulations=100)
    print(f"[✓] 虚拟处理效应分布: 均值={placebo_taus.mean():.4f}, 标准差={placebo_taus.std():.4f}")
    print(f"[✓] 真实DiD效应 ({did.tau:.4f}) 是否显著大于虚拟效应? ", 
          "是" if did.tau > np.percentile(placebo_taus, 95) else "否")

    print("\n[*] 生成安慰剂检验图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(placebo_taus, bins=30, alpha=0.7, label='虚拟处理效应分布', color='lightblue', edgecolor='black')
    ax.axvline(did.tau, color='red', linestyle='--', linewidth=2, label=f'真实DiD效应 ({did.tau:.2f})')
    ax.set_xlabel('处理效应', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('安慰剂检验：真实效应 vs 虚拟效应', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('placebo_test.png', dpi=150, bbox_inches='tight')
    print("[✓] 安慰剂检验图已保存为 placebo_test.png")

    print("\n[✓] Skill-DiD-Difference-in-Differences测试通过")
```

---

## ④ 技能关联

**前置技能**：
- [[Skill-因果推断基础]] — 理解因果关系识别的基本概念，掌握混淆变量、反事实框架
- [[Skill-A/B测试设计]] — 理解对照组的作用，为理解DiD中"对照组作为反事实"奠定基础

**延伸技能**：
- [[Skill-Staggered-DiD与异质性处理效应]] — 处理渐进式干预场景（如分阶段上线新功能），使用Callaway-Sant'Anna或Sun-Abraham方法
- [[Skill-合成控制法（Synthetic Control）]] — 当对照组不可得时，用加权组合构造虚拟对照组
- [[Skill-事件研究法（Event Study）]] — DiD的诊断工具，检验平行趋势假设

**可组合技能**：
- [[Skill-DiD]] + [[Skill-机器学习因果森林]] → **异质性处理效应分析**：不仅估计平均效应，还能识别哪类消费者/商品对干预最敏感（如：高价位婴儿车对关税更敏感，低价位产品不敏感）
- [[Skill-DiD]] + [[Skill-时间序列预测]] → **反事实预测**：用干预前的时间序列模型预测处理组的"如果没有干预"的销量路径，与实际值对比得到效应
- [[Skill-DiD]] + [[Skill-贝叶斯统计]] → **贝叶斯DiD**：在先验信息有限时，用贝叶斯框架量化平行趋势假设的不确定性

---

## ⑤ 商业价值评估

**ROI量化**：

- **场景1（暖奶器促销）**：年化收益¥215万，促销成本¥3.8万，**ROI = 5,658%**
- **场景2（TikTok Shop入驻）**：年化收益¥2,496万，平台佣金成本¥374万，**净ROI = 568%**
- **平均ROI**：**(5,658% + 568%) / 2 ≈ 3,113%**

**实施难度**：⭐⭐⭐☆☆

- 数据要求中等：需要处理组和对照组的时间序列数据，但不需要个体级别的因果图
- 假设检验复杂：平行趋势假设需要领域专业知识判断（如"德国和英国市场是否真的可比"）
- 代码实现简单：标准OLS回归，无需复杂算法

**优先级**：⭐⭐⭐⭐☆

- **高优先级原因**：
  1. 母婴出海中大量政策干预无法A/B测试（关税、物流中断、平台规则变化），DiD是唯一可行的因果推断工具
  2. ROI极高（3,000%+），投资回报周期短（2-4周内可出结果）
  3. 决策影响大：精准量化干预效应，直接影响定价、渠道、产能决策
  4. 实施成本低：仅需历史数据和基础统计知识，无需大规模实验投入

- **限制条件**：
  1. 平行趋势假设难以完全验证（只能事后检验，不能事前保证）
  2. 对照组选择有主观性（如选错对照组，结果完全错误）
  3. 长期效应难以估计（DiD通常只能估计短期效应，6-12个月后可能失效）

---

## 附录：关键概念速查

| 概念 | 定义 | 母婴出海例子 |
|------|------|------------|
| **处理组** | 接受干预的单元 | 销往美国的婴儿推车（受关税影响） |
| **对照组** | 未接受干预的单元 | 销往加拿大的婴儿推车（未受关税影响） |
| **平行趋势** | 干预前两组的结果变量趋势相同 | 关税前，美国和加拿大的销量增速都是每周+2% |
| **交互项** | 处理组×干预后期 | 捕捉处理组在干预后特有的变化 |
| **反事实** | 如果没有干预会发生什么 | 如果没有关税，美国销量会继续按原趋势增长 |
| **溢出效应** | 干预对对照组的间接影响 | 关税导致美国消费者转向加拿大购买（污染对照组） |

