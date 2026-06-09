---
title: Power Analysis and Sample Size Calculation for A/B Testing
doc_type: knowledge
module: 02-A_B实验
topic: power-analysis-sample-size
status: stable
created: 2026-05-15
updated: 2026-05-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Power Analysis and Sample Size Calculation

---

## ① 算法原理

**核心问题**：A/B测试需要多少样本才能检测出真实的效应？样本太少——检验力不足，假阴性率高（漏掉真实有效的改动）；样本太多——浪费流量和时间，拖慢迭代速度。

**四个核心参数**：

| 参数 | 符号 | 含义 | 常用取值 |
|------|------|------|---------|
| 显著性水平 | $\alpha$ | 假阳性率（I类错误） | 0.05 |
| 统计功效 | $1 - \beta$ | 正确拒绝原假设的概率 | 0.80 |
| 最小可检测效应 | MDE | 业务上值得检测的最小差异 | 依场景而定 |
| 标准差 | $\sigma$ | 结果变量的波动程度 | 历史数据估计 |

**样本量公式（连续型指标）**：

$$n = \frac{2\sigma^2(Z_{1-\alpha/2} + Z_{1-\beta})^2}{MDE^2}$$

其中：
- $Z_{1-\alpha/2} = 1.96$（双侧检验，α=0.05）
- $Z_{1-\beta} = 0.84$（功效80%）

**比例型指标（如转化率）**：

$$n = \frac{(Z_{1-\alpha/2}\sqrt{2p(1-p)} + Z_{1-\beta}\sqrt{p(1-p) + (p+MDE)(1-p-MDE)})^2}{MDE^2}$$

其中 $p$ 是基准转化率。

**反直觉洞察**：
- 样本量与 MDE 的平方成反比——想把检测精度提高一倍，需要四倍样本
- 提升功效从80%到90%，样本量增加约30%；从90%到95%，再增加约60%
- 90%的测试失败不是因为改动无效，而是因为样本量不够检测出真实的小幅改进

---

## ② 母婴出海应用案例

### 场景1：首页改版实验的样本量计算

**业务问题**：产品团队想测试新版首页布局对转化率的影响。预期转化率从2.0%提升到2.2%（相对提升10%），需要多少样本？实验要跑多久？

**计算流程**：
1. **确定参数**：
   - 基准转化率 $p = 0.02$
   - MDE = 0.002（绝对提升0.2个百分点）
   - $\alpha = 0.05$（双侧）
   - 功效 = 0.80
2. **计算样本量**：
   - 每组需要约 39,000 用户
   - 两组共 78,000 用户
3. **计算实验时长**：
   - 日均活跃用户（DAU）= 50,000
   - 实验分流50% → 每天进入实验 25,000 人
   - 所需天数 = 78,000 / 25,000 ≈ 3.1 天

**决策输出**：
- 实验设计：50/50分流，跑4天（含1天缓冲）
- 若4天后结果不显著，不要急于下结论"无效"——可能是MDE设得太小

### 场景2：多实验并行时的流量分配

**业务问题**：同时有3个A/B测试在进行（首页改版、详情页优化、购物车流程），每个都需要一定样本量。如何在有限流量下合理分配？

**策略**：
1. 计算每个实验所需样本量
2. 按业务优先级排序
3. 高优先级实验全流量，低优先级实验降流量延长实验周期
4. 避免实验间相互污染（正交分层）

---

## ③ 代码模板

```python
"""
Power Analysis and Sample Size Calculation for A/B Testing
用于实验前的样本量规划和功效分析
"""

import numpy as np
from scipy import stats


def sample_size_continuous(mde, sigma, alpha=0.05, power=0.8, ratio=1.0):
    """
    连续型指标的样本量计算

    Args:
        mde: 最小可检测效应（绝对值）
        sigma: 标准差
        alpha: 显著性水平
        power: 统计功效
        ratio: 实验组/对照组样本量比
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = ((1 + 1 / ratio) * sigma ** 2 * (z_alpha + z_beta) ** 2) / mde ** 2
    return int(np.ceil(n))


def sample_size_proportion(p, mde, alpha=0.05, power=0.8, ratio=1.0):
    """
    比例型指标（转化率）的样本量计算

    Args:
        p: 基准转化率
        mde: 最小可检测效应（绝对值）
        alpha: 显著性水平
        power: 统计功效
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = p
    p2 = p + mde
    p_avg = (p1 + p2) / 2

    n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta) ** 2) / mde ** 2
    return int(np.ceil(n))


def calculate_mde(n, sigma, alpha=0.05, power=0.8):
    """
    给定样本量，计算可检测的最小效应

    Args:
        n: 每组的样本量
        sigma: 标准差
        alpha: 显著性水平
        power: 统计功效
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    mde = np.sqrt(2 * sigma ** 2 / n) * (z_alpha + z_beta)
    return mde


def experiment_duration(total_sample, daily_traffic, split_ratio=0.5):
    """
    计算实验所需天数

    Args:
        total_sample: 总样本量（两组之和）
        daily_traffic: 日流量
        split_ratio: 进入实验的流量比例
    """
    daily_experiment_traffic = daily_traffic * split_ratio
    days = total_sample / daily_experiment_traffic
    return int(np.ceil(days))


def power_analysis_table(p, mde_range, alpha=0.05, power=0.8):
    """
    生成功效分析表格：不同MDE对应的样本量

    Args:
        p: 基准转化率
        mde_range: MDE范围列表
    """
    print(f"\n基准转化率: {p*100:.1f}%, α={alpha}, 功效={power}")
    print("-" * 50)
    print(f"{'MDE (绝对)':>12} {'相对提升':>10} {'每组样本量':>12} {'总样本量':>12}")
    print("-" * 50)

    for mde in mde_range:
        n = sample_size_proportion(p, mde, alpha, power)
        rel_lift = mde / p * 100
        print(f"{mde:>12.4f} {rel_lift:>9.1f}% {n:>12,} {n*2:>12,}")


# ==================== 母婴出海业务专用函数 ====================

def plan_momcozy_experiment(metric_type, baseline, target_lift, daily_uv, split=0.5):
    """
    Momcozy 实验规划工具

    Args:
        metric_type: 'conversion' 或 'revenue'
        baseline: 基准值
        target_lift: 目标相对提升（如0.1表示10%）
        daily_uv: 日均UV
        split: 实验流量比例
    """
    mde = baseline * target_lift

    if metric_type == 'conversion':
        n_per_group = sample_size_proportion(baseline, mde)
        sigma = None
    else:  # revenue
        # 假设收入标准差约为均值的1.5倍
        sigma = baseline * 1.5
        n_per_group = sample_size_continuous(mde, sigma)

    total_n = n_per_group * 2
    days = experiment_duration(total_n, daily_uv, split)

    print("=" * 60)
    print("Momcozy A/B 实验规划")
    print("=" * 60)
    print(f"指标类型: {metric_type}")
    print(f"基准值: {baseline}")
    print(f"目标提升: {target_lift*100:.1f}%")
    print(f"MDE (绝对): {mde:.4f}")
    if sigma:
        print(f"估计标准差: {sigma:.2f}")
    print(f"每组样本量: {n_per_group:,}")
    print(f"总样本量: {total_n:,}")
    print(f"日均UV: {daily_uv:,}")
    print(f"实验流量比例: {split*100:.0f}%")
    print(f"预计实验天数: {days} 天")
    print("=" * 60)

    return {
        'n_per_group': n_per_group,
        'total_n': total_n,
        'days': days,
        'mde': mde
    }


# ==================== 示例代码 ====================

def main():
    """主函数：演示功效分析"""
    print("=" * 70)
    print("A/B 测试功效分析与样本量计算")
    print("=" * 70)

    # 1. 转化率实验
    print("\n[1] 转化率实验样本量计算")
    print("   场景：吸奶器详情页CTA按钮优化")
    print("   基准转化率：2.0%，目标检测相对提升：10%")
    n = sample_size_proportion(p=0.02, mde=0.002, alpha=0.05, power=0.8)
    print(f"   每组所需样本量: {n:,}")
    print(f"   总样本量: {n*2:,}")

    # 2. 收入实验
    print("\n[2] 收入实验样本量计算")
    print("   场景：满减促销策略")
    print("   基准客单价：$120，目标检测提升：$6（5%）")
    print("   假设标准差：$180")
    n = sample_size_continuous(mde=6, sigma=180, alpha=0.05, power=0.8)
    print(f"   每组所需样本量: {n:,}")
    print(f"   总样本量: {n*2:,}")

    # 3. 功效分析表格
    print("\n[3] 不同MDE下的样本量对比")
    power_analysis_table(p=0.02, mde_range=[0.001, 0.002, 0.003, 0.005])

    # 4. Momcozy实验规划
    print("\n[4] Momcozy 实验规划示例")
    plan_momcozy_experiment(
        metric_type='conversion',
        baseline=0.025,
        target_lift=0.08,
        daily_uv=50000,
        split=0.5
    )

    # 5. 给定样本量反推MDE
    print("\n[5] 给定样本量反推可检测最小效应")
    mde = calculate_mde(n=10000, sigma=50, alpha=0.05, power=0.8)
    print(f"   每组10,000样本，标准差50")
    print(f"   可检测最小效应: {mde:.2f}")

    print("\n" + "=" * 70)
    print("功效分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
```

---

## ④ 技能关联

### 前置技能
- [[Skill-AB-Experimental-Design]] — 理解实验设计的基本概念（对照组、随机化、盲法）
- **基础统计推断** — 理解假设检验、P值、置信区间

### 延伸技能
- **Sequential Testing** — 序贯检验，支持提前停止实验
- **Multi-Armed Bandit** — 在探索和利用之间动态分配流量

### 可组合技能
- **+ AB-Experimental-Design**: 实验设计确定"测什么"，功效分析确定"测多少"
- **+ Uplift-Modeling**: 功效分析规划A/B测试，Uplift Modeling分析实验后的增量效果
- **+ Time-Series-Forecasting**: 用预测模型估计实验期间的外部波动，调整MDE预期

---

- **可组合**：[[Skill-Uplift-Modeling]] / [[Skill-Multi-Armed-Bandit]]

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 避免假阴性 | 避免因样本不足而丢弃有效的改动（每次有效改动的价值通常 > 10万） | 开发1天 | 极高 |
| 优化流量分配 | 合理分配实验流量，加速迭代周期 | 开发1天 | 高 |
| 实验规划标准化 | 建立实验前的标准检查清单，提升团队效率 | 开发2天 | 高 |

### 实施难度
**评分：⭐☆☆☆☆（1/5星）**

- 数据要求：只需要历史数据的基准值和标准差
- 技术门槛：极低，公式直接计算
- 工程复杂度：极低
- 主要挑战：与业务方沟通MDE的设定（需要平衡统计严谨性和业务实际）

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **所有A/B测试的前提**：没有功效分析的A/B测试是不完整的
- **实施成本极低**：公式计算，1天完成
- **避免巨大浪费**：90%的实验失败源于样本量不足，而非改动无效
- **团队基础设施**：是每个数据科学家和产品经理应该掌握的基础能力

### 评估依据
1. 样本量计算是A/B测试的标准前置步骤，没有它就无法设计有效的实验
2. 与现有的AB Experimental Design、MAB形成完整的实验方法论体系
3. 母婴出海业务流量有限（vs国内大厂），更需要精打细算地使用每一份流量
4. 实施成本极低，但价值极高——属于"高杠杆"技能
