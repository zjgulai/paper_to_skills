---
title: Causal Mediation Analysis — Decomposing "Why It Works"
doc_type: knowledge
module: 01-因果推断
topic: mediation-analysis
status: stable
created: 2026-05-15
updated: 2026-05-15
owner: self
source: human+ai
---

# Skill Card: Causal Mediation Analysis

---

## ① 算法原理

**核心思想**：中介分析回答"为什么"——一个干预（如推荐算法更新）通过什么机制影响了结果（如转化率）。它将总效应分解为：
- **直接效应**：干预直接影响结果（不经过中介变量）
- **间接效应**：干预通过改变中介变量，再由中介变量影响结果

**为什么需要中介分析**：

母婴出海电商中，只知道"什么有效"不够，必须知道"为什么有效"才能复制和放大：
- 新推荐算法提升了转化率——是通过提升点击率实现的，还是通过提升客单价实现的？
- 客服响应速度加快降低了退货率——是因为用户更满意了，还是因为问题更快解决了？
- KOL合作带来了品牌曝光——是通过搜索量增长实现的，还是通过社交裂变实现的？

没有机制分解，就只能盲目复制表面动作，无法优化核心杠杆。

**核心公式（自然效应框架，Pearl 2001）**：

设：
- $T$ = 干预（如新旧推荐算法）
- $M$ = 中介变量（如点击率、浏览深度）
- $Y$ = 结果（如转化率、GMV）

**总效应（Total Effect, TE）**：
$$TE = E[Y(1, M(1))] - E[Y(0, M(0))]$$

**自然直接效应（Natural Direct Effect, NDE）**：
$$NDE = E[Y(1, M(0))] - E[Y(0, M(0))]$$
即：干预改变，但中介保持在未干预时的水平。

**自然间接效应（Natural Indirect Effect, NIE）**：
$$NIE = E[Y(1, M(1))] - E[Y(1, M(0))]$$
即：干预固定在接受状态，但中介从"未干预时的水平"变为"干预后的水平"。

**关键恒等式**：$TE = NDE + NIE$

**识别假设**：
1. **无未观测混杂**：干预→结果、干预→中介、中介→结果三条路径都没有未观测混杂
2. **无干预-中介交互混杂**：不存在同时影响中介和结果的变量受干预影响（这是最难满足的假设）
3. **序贯可忽略性**：给定协变量后，干预分配与潜在结果独立

**Surrogate Index（替代指标指数）**：

Chetty & Imai (2025) 提出的前沿方法：用短期可观测的中介变量（如点击率、加购率）构建"替代指标指数"，预测长期结果（如LTV、年度留存）。解决了长期效应评估的数据滞后问题。

**反直觉洞察**：直接效应和间接效应的符号可能相反。例如：促销活动（T）通过提升点击率（M）提升了转化率（Y），但同时促销活动也让用户产生"低价预期"心理，直接降低了转化率。此时 $NIE > 0$（点击率路径正向），但 $NDE < 0$（心理预期路径负向），总效应可能不显著——不看机制分解就会得出"促销无效"的错误结论。

---

## ② 母婴出海应用案例

### 场景1：推荐算法更新的机制分解

**业务问题**：产品团队上线了新的推荐算法（基于用户行为的协同过滤 → 基于内容的混合推荐），全站转化率提升了2%。但产品经理想知道：这2%是通过什么路径实现的？是点击率提升了？还是用户浏览深度增加了？还是客单价变了？

**中介变量选择**：
- $M_1$ = 点击率（CTR）
- $M_2$ = 平均浏览页面数
- $M_3$ = 平均客单价

**应用流程**：
1. **总效应估计**：对比算法上线前后，处理组（被推荐商品）vs 对照组（未被推荐商品）的转化率变化
2. **逐个中介分析**：对每个中介变量M，估计NDE和NIE
3. **机制排序**：比较各中介的间接效应大小，找出主要驱动机制

**预期产出**：
- 总效应：+2.0% 转化率提升
- 机制分解：
  - 点击率路径（NIE₁）：+1.5%（主要驱动）
  - 浏览深度路径（NIE₂）：+0.4%
  - 客单价路径（NIE₃）：-0.2%（负向，被低价商品推荐稀释）
  - 直接效应（NDE）：+0.3%

**业务价值**：
- 明确核心杠杆是"点击率"，而非"浏览深度"或"客单价"
- 后续优化方向：继续优化CTR（如改进商品主图、标题），而非投入资源优化浏览深度
- 警惕负向机制：客单价下降需要关注，避免推荐系统过度推荐低价商品

### 场景2：客服响应提速的退货率机制分析

**业务问题**：客服团队将平均响应时间从4小时缩短到30分钟，退货率从8%降到6%。想知道：响应提速是通过"用户满意度提升"还是"问题更快解决"降低了退货率？

**中介变量**：
- $M_1$ = 用户满意度评分（CSAT）
- $M_2$ = 问题解决时长

**决策价值**：
- 如果主要通过 $M_1$（满意度）：说明情感安抚是关键，后续应投入情感智能客服培训
- 如果主要通过 $M_2$（解决速度）：说明流程效率是关键，后续应优化SOP和知识库

---

## ③ 代码模板

```python
"""
Causal Mediation Analysis — 因果中介效应分析
用于分解干预的总效应为直接效应和间接效应

支持：经典中介分析、多重中介分析、Surrogate Index
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# ==================== 中介分析核心实现 ====================

class MediationAnalysis:
    """因果中介分析器"""

    def __init__(self, method='parametric'):
        """
        初始化中介分析器

        Args:
            method: 'parametric' (基于回归) 或 'bootstrap'
        """
        self.method = method
        self.a_coef = None  # T → M 的效应
        self.b_coef = None  # M → Y 的效应（控制T）
        self.c_prime = None  # T → Y 的直接效应（控制M）
        self.results = {}

    def fit(self, T, M, Y, X_controls=None):
        """
        拟合中介分析模型

        Args:
            T: 干预变量 (n_samples,)
            M: 中介变量 (n_samples,) — 可扩展为多重中介
            Y: 结果变量 (n_samples,)
            X_controls: 控制变量 (n_samples, n_controls)
        """
        T = np.array(T).ravel()
        M = np.array(M).ravel()
        Y = np.array(Y).ravel()
        n = len(T)

        # 构建特征矩阵
        if X_controls is not None:
            X_controls = np.array(X_controls)
            if X_controls.ndim == 1:
                X_controls = X_controls.reshape(-1, 1)

        # ========== 模型1: T → M (路径a) ==========
        if X_controls is not None:
            X_a = np.column_stack([T, X_controls])
        else:
            X_a = T.reshape(-1, 1)

        model_a = LinearRegression()
        model_a.fit(X_a, M)
        self.a_coef = model_a.coef_[0]  # T对M的效应

        # ========== 模型2: T + M → Y (路径b和c') ==========
        if X_controls is not None:
            X_b = np.column_stack([T, M, X_controls])
        else:
            X_b = np.column_stack([T, M])

        model_b = LinearRegression()
        model_b.fit(X_b, Y)
        self.c_prime = model_b.coef_[0]  # 直接效应 c'
        self.b_coef = model_b.coef_[1]   # M对Y的效应 b

        # ========== 模型3: T → Y (总效应c) ==========
        if X_controls is not None:
            X_c = np.column_stack([T, X_controls])
        else:
            X_c = T.reshape(-1, 1)

        model_c = LinearRegression()
        model_c.fit(X_c, Y)
        c_coef = model_c.coef_[0]  # 总效应

        # ========== 效应分解 ==========
        # 间接效应 = a * b
        indirect_effect = self.a_coef * self.b_coef

        # 直接效应 = c'
        direct_effect = self.c_prime

        # 总效应 = c (应与 direct + indirect 近似)
        total_effect = c_coef

        # 中介比例
        mediation_ratio = indirect_effect / total_effect if total_effect != 0 else 0

        self.results = {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'mediation_ratio': mediation_ratio,
            'a_path': self.a_coef,
            'b_path': self.b_coef,
            'c_prime': self.c_prime,
            'c_total': c_coef
        }

        return self

    def summary(self):
        """输出中介分析结果"""
        if not self.results:
            return "模型尚未拟合"

        print("=" * 60)
        print("因果中介分析结果")
        print("=" * 60)
        print(f"\n路径系数:")
        print(f"  a (T → M):     {self.results['a_path']:.4f}")
        print(f"  b (M → Y):     {self.results['b_path']:.4f}")
        print(f"  c' (T → Y| M): {self.results['c_prime']:.4f}")
        print(f"\n效应分解:")
        print(f"  总效应 (c):    {self.results['total_effect']:.4f}")
        print(f"  直接效应:      {self.results['direct_effect']:.4f}")
        print(f"  间接效应:      {self.results['indirect_effect']:.4f}")
        print(f"  中介比例:      {self.results['mediation_ratio']*100:.1f}%")
        print(f"\n验证: 直接 + 间接 = {self.results['direct_effect'] + self.results['indirect_effect']:.4f}")
        print(f"      总效应     = {self.results['total_effect']:.4f}")
        print("=" * 60)

        return self.results


class MultipleMediationAnalysis:
    """多重中介分析器"""

    def __init__(self):
        self.mediation_results = {}

    def fit(self, T, M_dict, Y, X_controls=None):
        """
        多重中介分析

        Args:
            T: 干预变量
            M_dict: 中介变量字典 {name: array}
            Y: 结果变量
            X_controls: 控制变量
        """
        T = np.array(T).ravel()
        Y = np.array(Y).ravel()

        # 总效应
        if X_controls is not None:
            X_c = np.column_stack([T, X_controls])
        else:
            X_c = T.reshape(-1, 1)
        model_total = LinearRegression()
        model_total.fit(X_c, Y)
        total_effect = model_total.coef_[0]

        # 逐个中介分析
        all_mediations = []
        for m_name, M in M_dict.items():
            ma = MediationAnalysis()
            ma.fit(T, M, Y, X_controls)
            all_mediations.append({
                'mediator': m_name,
                **ma.results
            })

        self.mediation_results = {
            'total_effect': total_effect,
            'mediations': all_mediations
        }

        return self

    def summary(self):
        """输出多重中介分析结果"""
        print("=" * 70)
        print("多重因果中介分析结果")
        print("=" * 70)
        print(f"\n总效应: {self.mediation_results['total_effect']:.4f}")
        print(f"\n机制分解:")
        print("-" * 70)
        print(f"{'中介变量':<15} {'间接效应':>10} {'占比':>8} {'直接效应':>10}")
        print("-" * 70)

        for m in self.mediation_results['mediations']:
            print(f"{m['mediator']:<15} {m['indirect_effect']:>10.4f} "
                  f"{m['mediation_ratio']*100:>7.1f}% {m['direct_effect']:>10.4f}")

        # 计算各中介效应之和
        total_indirect = sum(m['indirect_effect'] for m in self.mediation_results['mediations'])
        direct = self.mediation_results['mediations'][0]['direct_effect']

        print("-" * 70)
        print(f"{'间接效应合计':<15} {total_indirect:>10.4f}")
        print(f"{'直接效应':<15} {direct:>10.4f}")
        print(f"{'合计':<15} {total_indirect + direct:>10.4f}")
        print("=" * 70)

        return self.mediation_results


# ==================== 母婴出海业务专用函数 ====================

def generate_recommendation_mediation_data(n_samples=5000, random_state=42):
    """
    生成推荐算法更新的中介分析模拟数据

    场景：新旧推荐算法对比，中介变量为点击率、浏览深度、客单价
    """
    np.random.seed(random_state)

    # 干预：0=旧算法, 1=新算法
    T = np.random.binomial(1, 0.5, n_samples)

    # 中介变量1：点击率（新算法显著提升）
    # M1 = 0.05 + 0.03*T + noise
    M1 = 0.05 + 0.03 * T + np.random.normal(0, 0.01, n_samples)
    M1 = np.clip(M1, 0, 0.5)

    # 中介变量2：浏览深度（新算法轻微提升）
    # M2 = 3 + 0.5*T + noise
    M2 = 3 + 0.5 * T + np.random.normal(0, 1, n_samples)
    M2 = np.maximum(M2, 1)

    # 中介变量3：客单价（新算法轻微下降——推荐低价商品更多）
    # M3 = 80 - 5*T + noise
    M3 = 80 - 5 * T + np.random.normal(0, 15, n_samples)
    M3 = np.maximum(M3, 20)

    # 结果：转化率
    # Y = 0.02 + 0.5*M1 + 0.002*M2 - 0.0001*M3 + 0.005*T + noise
    Y_prob = (
        0.02 +                    # 基础转化率
        0.5 * M1 +                # 点击率对转化率的正向影响（主路径）
        0.002 * M2 +              # 浏览深度的轻微正向影响
        -0.0001 * M3 +            # 客单价的轻微负向影响
        0.005 * T +               # 直接效应（算法本身的其他改进）
        np.random.normal(0, 0.005, n_samples)
    )
    Y = (Y_prob > np.percentile(Y_prob, 85)).astype(int)  # 二值化

    df = pd.DataFrame({
        'treatment': T,
        'ctr': M1,
        'browse_depth': M2,
        'aov': M3,
        'conversion': Y
    })

    return df


# ==================== 示例代码 ====================

def main():
    """主函数：演示中介分析在推荐算法评估中的应用"""
    print("=" * 70)
    print("母婴出海 — 因果中介分析：推荐算法更新机制分解")
    print("=" * 70)

    # 1. 生成模拟数据
    print("\n[1] 生成模拟数据...")
    print("   场景：新旧推荐算法对比")
    print("   中介变量：点击率、浏览深度、客单价")

    df = generate_recommendation_mediation_data(n_samples=5000)
    print(f"   样本量: {len(df)}")
    print(f"   新算法组: {df['treatment'].sum()} ({df['treatment'].mean()*100:.0f}%)")
    print(f"   旧算法组: {len(df) - df['treatment'].sum()}")

    # 2. 单中介分析（点击率）
    print("\n[2] 单中介分析：点击率路径...")
    ma_ctr = MediationAnalysis()
    ma_ctr.fit(df['treatment'], df['ctr'], df['conversion'])
    ma_ctr.summary()

    # 3. 多重中介分析
    print("\n[3] 多重中介分析（完整机制分解）...")
    m_dict = {
        '点击率': df['ctr'],
        '浏览深度': df['browse_depth'],
        '客单价': df['aov']
    }

    mma = MultipleMediationAnalysis()
    mma.fit(df['treatment'], m_dict, df['conversion'])
    mma.summary()

    # 4. 业务解读
    print("\n[4] 业务解读...")
    total = mma.mediation_results['total_effect']
    mediations = mma.mediation_results['mediations']

    # 找出最大贡献的中介
    main_mediator = max(mediations, key=lambda x: abs(x['indirect_effect']))
    print(f"   核心驱动机制: {main_mediator['mediator']}")
    print(f"   贡献占比: {main_mediator['mediation_ratio']*100:.1f}%")

    # 检查负向机制
    negative_meds = [m for m in mediations if m['indirect_effect'] < 0]
    if negative_meds:
        print(f"   ⚠️ 负向机制: {', '.join(m['mediator'] for m in negative_meds)}")
        print(f"      建议: 关注并抑制负向路径")

    print("\n" + "=" * 70)
    print("中介分析完成！")
    print("=" * 70)

    return mma


if __name__ == '__main__':
    result = main()
```

---

## ④ 技能关联

### 前置技能
- **Skill-Uplift-Modeling** — 理解因果效应估计的基本概念
- **Skill-DiD-Difference-in-Differences** — 中介分析常与DiD结合使用（DiD估计总效应，中介分析分解机制）

### 延伸技能
- **Surrogate Index (Chetty & Imai, 2025)** — 用短期中介变量预测长期结果，解决LTV评估的滞后问题
- **Causal Forest Heterogeneous Effects** — 在总效应和中介效应的基础上，进一步分析"对谁有效"

### 可组合技能
- **+ DiD**: DiD估计政策/算法更新的总效应 → 中介分析分解"为什么有效"
- **+ A/B-Testing**: A/B测试随机分配干预 → 中介分析识别最优机制路径
- **+ VOC-Analysis**: VOC挖掘发现用户反馈主题 → 中介分析验证哪些主题确实是转化驱动因素

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 算法更新机制分解 | 明确优化方向，避免无效投入（年节省10-30万） | 开发2-3天 | 15-30x |
| 促销策略机制分析 | 识别负向机制，保护品牌价值 | 开发2-3天 | 10-20x |
| 客服流程优化 | 精准定位核心杠杆，避免全面改造 | 开发2-3天 | 10-15x |

### 实施难度
**评分：⭐⭐☆☆☆（2/5星）**

- 数据要求：需要同时观测干预、中介和结果三个变量
- 技术门槛：低，核心是OLS回归的组合
- 主要挑战：选择合理的中介变量 + 论证识别假设（无未观测混杂）
- 工程复杂度：低
- 维护成本：低

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **从"知道有效"到"知道为什么有效"**：这是从执行层到策略层的跃迁
- **方法极简单**：两个OLS回归即可，但洞察极深刻
- **通用性最强**：任何A/B测试、政策评估、算法更新后都应该做机制分解
- **与现有技能完美互补**：DiD/Uplift告诉你"效应有多大"，Mediation告诉你"为什么"

### 评估依据
1. 母婴出海电商的决策链条长（广告→点击→浏览→加购→支付→复购），每个环节都是潜在中介。不看机制就只能盲目优化。
2. 与DiD + IV形成完整因果推断工具链：DiD（时间维度）+ IV（截面维度）+ Mediation（机制维度）= 三维因果分析
3. 实施成本极低（两个回归），但业务价值极高（避免错误优化方向）
4. 是A/B测试后必做的分析——没有机制分解的A/B测试只完成了一半
