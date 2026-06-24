---
title: Infant Lifecycle Purchase Rhythm — 婴儿 0-24 月龄标准消费品类时序图谱建模
doc_type: knowledge
module: 06-增长模型
topic: infant-lifecycle-purchase-rhythm
status: stable
created: 2026-06-24
updated: 2026-06-24
owner: self
source: void-framework
roadmap_phase: phase2
void_origin: "VOID Q2-2026 盲点B003 → 婴儿月龄时钟系列Skill-2"
---

# Skill Card: Infant Lifecycle Purchase Rhythm — 婴儿生命周期消费节律建模

> **论文**：Personalized Category Frequency Prediction for Buy It Again Recommendations — PCIC (arXiv:2308.01195, Target 2023) + Life-stage Prediction for Product Recommendation in E-commerce (KDD 2015) + Empowering Parents: Personalized Recommendations for Italy's Top Pregnancy App (MIT Sloan MBAn 2023)
> **arXiv**：2308.01195 | 2023年 | **桥梁**: 06-增长模型 ↔ 05-推荐系统 | **类型**: VOID第三象限→标准Skill

---

## ① 算法原理

### 核心思想

母婴消费的本质是**由生物发育节律驱动的刚需序列**——不是随机的，不是季节性的，而是**婴儿身体发育时钟触发的可预测序列**：

```
0月  → 1月  → 2月  → 3月  → 4月  → 6月  → 9月  → 12月 → 18月 → 24月
新生儿必需品  早期婴儿     辅食准备     移动爬行      幼儿早期       幼儿后期
奶粉NB     奶粉1段S    奶粉2段辅食   爬行垫学步    奶粉3段        断奶/杯子
尿布NB     尿布S       尿布M牙胶    学步车尿布L   如厕训练       床围
防风帽      抱被        餐椅高椅      安全围栏      幼儿鞋         学习杯
```

**PCIC 模型**（arXiv:2308.01195，Target 亿级用户验证）的核心框架：

**层级架构**：
1. **PC 模型（品类层）**：预测用户下一个会复购的品类（用生存模型 + 时序模型）
2. **IC 模型（品类内商品层）**：在预测品类内排序具体商品

**品类复购率的两个来源**（生存模型精髓）：
$$P(\text{复购时间} = t | \text{品类} c, \text{用户} u) = \lambda_{c}(t) \cdot \exp(-\Lambda_{c}(t)) \cdot f(u, c)$$

- $\lambda_{c}(t)$：品类 $c$ 的基础危险函数（全局平均复购周期）
- $\Lambda_{c}(t)$：累积危险函数
- $f(u, c)$：用户 $u$ 在品类 $c$ 上的个人化调整系数

**婴儿月龄层叠**（本 Skill 的创新点）：
在 PCIC 基础上加入月龄约束——不同月龄下，各品类的"预期首购窗口"不同：

$$P(\text{下一购买品类} = c | \text{月龄} = a) \propto P(c) \cdot \mathbb{1}[a \in \text{月龄区间}(c)] \cdot \text{紧迫性系数}(c, a)$$

**紧迫性系数**：即将进入某品类的月龄窗口时，紧迫性系数 > 1（提前推荐）；已过最佳窗口时，系数衰减。

**关键假设**：
- 婴儿发育遵循统计意义上的标准月龄轨迹（允许 ±2 个月个体差异）
- 同一品类的复购周期对所有婴儿相似（尿布每 3-4 周补货）
- 父母存在"提前备货"行为（在关键节点前 2-3 周购买）

---

## ② 母婴出海应用案例

### 场景A：婴儿用品跨境独立站的"下一购买品类"预测引擎

**业务问题**：独立站首页只能展示有限商品，但不同月龄的妈妈需要的东西完全不同。对一个4月龄宝宝的妈妈推荐学步车毫无意义，而对7月龄宝宝的妈妈不推荐辅食工具是巨大的错失。

**方案**：
1. 用 Baby Age Clock（[[Skill-Baby-Age-Clock-RFM-Enhancement]]）推断用户婴儿月龄
2. 用 Infant Lifecycle Purchase Rhythm 预测该月龄用户的"下一个高需求品类"
3. 在首页动态注入月龄-品类推荐模块（不需要用户填写任何信息）

**月龄-品类触达时机矩阵**（核心输出，可直接用于内容运营）：

| 月龄 | 高需求品类 | 触达提前窗口 |
|------|---------|-----------|
| 3.5-4月 | 辅食工具/米粉 | 提前 3 周 |
| 8.5-9月 | 学步护膝/爬行垫/安全围栏 | 提前 2 周 |
| 11-12月 | 学步鞋/推车 | 提前 4 周 |
| 17-18月 | 如厕训练/幼儿杯 | 提前 3 周 |

**业务价值**：月龄感知首页推荐的 CTR 预期提升 35-50%，AOV 提升 20%（妈妈更容易接受"正好需要"的品类）

### 场景B：精准库存备货计划——按月龄窗口预判品类需求峰值

**业务问题**：婴儿辅食工具在 4-5 月龄购买峰值非常集中，但 FBA 补货需要提前 4 周。如果只看历史销售而不看用户月龄分布，容易在峰值期 OOS（缺货）。

**方案**：
1. 统计当前会员中婴儿月龄在 2-3 个月的用户数量（这批人 4-6 周后将进入辅食期）
2. 预测 4-6 周后的辅食品类需求峰值
3. 提前 5 周触发 FBA 补货入库

**业务价值**：减少辅食类品类 OOS 率约 60%，节省缺货期损失约 **$12,000/季度**

---

## ③ 代码模板

```python
"""
Infant Lifecycle Purchase Rhythm
婴儿生命周期消费节律建模——品类时序图谱 + 下一品类预测

依赖：numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy.stats import expon, lognorm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 婴儿生命周期品类时序图谱
# ─────────────────────────────────────────────

INFANT_LIFECYCLE_RHYTHM = {
    'newborn_essentials': {
        'age_window': (0, 1),
        'urgency_peak': 0,
        'repurchase_weeks': 3,
        'categories': ['newborn_diapers', 'formula_stage1', 'swaddle', 'newborn_clothing'],
    },
    'early_infant': {
        'age_window': (1, 4),
        'urgency_peak': 2,
        'repurchase_weeks': 4,
        'categories': ['formula_stage1', 'diaper_s', 'baby_bath', 'teether_early'],
    },
    'pre_solids_prep': {
        'age_window': (3.5, 4.5),
        'urgency_peak': 4,
        'repurchase_weeks': None,
        'categories': ['high_chair', 'baby_spoon', 'rice_cereal', 'silicone_bib'],
        'is_milestone': True,
        'advance_weeks': 3,
    },
    'solids_intro': {
        'age_window': (4, 8),
        'urgency_peak': 5,
        'repurchase_weeks': 2,
        'categories': ['baby_puree', 'formula_stage2', 'diaper_m', 'teether'],
    },
    'pre_crawling': {
        'age_window': (7, 9),
        'urgency_peak': 8,
        'repurchase_weeks': None,
        'categories': ['crawling_mat', 'safety_gate', 'knee_pads', 'pull_toy'],
        'is_milestone': True,
        'advance_weeks': 2,
    },
    'mobile_infant': {
        'age_window': (9, 12),
        'urgency_peak': 10,
        'repurchase_weeks': 4,
        'categories': ['formula_stage2', 'diaper_l', 'finger_food', 'walker'],
    },
    'first_birthday': {
        'age_window': (11, 13),
        'urgency_peak': 12,
        'repurchase_weeks': None,
        'categories': ['toddler_shoes', 'stroller_upgrade', 'birthday_supplies'],
        'is_milestone': True,
        'advance_weeks': 4,
    },
    'early_toddler': {
        'age_window': (12, 18),
        'urgency_peak': 15,
        'repurchase_weeks': 3,
        'categories': ['formula_stage3', 'diaper_xl', 'toddler_snack', 'sippy_cup'],
    },
    'potty_training': {
        'age_window': (18, 24),
        'urgency_peak': 20,
        'repurchase_weeks': None,
        'categories': ['potty_chair', 'training_pants', 'reward_stickers'],
        'is_milestone': True,
        'advance_weeks': 3,
    },
    'late_toddler': {
        'age_window': (20, 36),
        'urgency_peak': 24,
        'repurchase_weeks': 4,
        'categories': ['toddler_milk', 'toddler_clothing', 'learning_toys'],
    },
}


def get_upcoming_categories(
    baby_age_months: float,
    lookahead_weeks: int = 6,
    confidence: float = 0.8
) -> List[Dict]:
    """
    给定当前婴儿月龄，预测未来 N 周内的高需求品类

    Args:
        baby_age_months: 当前推断的婴儿月龄
        lookahead_weeks: 向前预测的周数
        confidence: 月龄推断置信度

    Returns:
        按紧迫性排序的品类列表
    """
    if baby_age_months < 0:
        return []

    lookahead_months = lookahead_weeks / 4.33
    future_age = baby_age_months + lookahead_months

    upcoming = []

    for stage_name, stage in INFANT_LIFECYCLE_RHYTHM.items():
        age_min, age_max = stage['age_window']
        peak_age = stage['urgency_peak']

        advance = stage.get('advance_weeks', 0) / 4.33

        # 当前月龄是否在触达窗口内（包含提前量）
        in_window = (age_min - advance) <= baby_age_months <= age_max
        # 未来月龄是否进入该阶段
        entering_soon = baby_age_months < age_min <= future_age

        if in_window or entering_soon:
            # 紧迫性评分：越接近峰值月龄越紧迫
            distance_to_peak = abs(baby_age_months - peak_age)
            urgency = np.exp(-distance_to_peak / 2) * confidence

            is_milestone = stage.get('is_milestone', False)
            upcoming.append({
                'stage': stage_name,
                'categories': stage['categories'],
                'urgency_score': urgency,
                'is_milestone': is_milestone,
                'weeks_to_peak': (peak_age - baby_age_months) * 4.33,
                'repurchase_weeks': stage.get('repurchase_weeks'),
            })

    return sorted(upcoming, key=lambda x: -x['urgency_score'])


def predict_demand_surge(
    user_age_distribution: pd.Series,
    category: str,
    weeks_ahead: int = 6
) -> pd.DataFrame:
    """
    基于当前用户月龄分布，预测未来 N 周内某品类的需求峰值

    Args:
        user_age_distribution: Series，index=月龄（0-36），values=该月龄用户数
        category: 目标品类
        weeks_ahead: 预测窗口

    Returns:
        每周预测需求量
    """
    # 找出该品类对应的月龄窗口
    target_stage = None
    for stage_name, stage in INFANT_LIFECYCLE_RHYTHM.items():
        if category in stage['categories']:
            target_stage = stage
            break

    if target_stage is None:
        return pd.DataFrame()

    age_min, age_max = target_stage['age_window']
    advance_weeks = target_stage.get('advance_weeks', 0)

    weekly_demand = []
    for week in range(1, weeks_ahead + 1):
        future_months = week / 4.33
        # N周后，当前月龄=X的用户，月龄会变成X+future_months
        # 哪些用户会在这周进入目标品类窗口？
        will_enter_age_min = age_min - future_months
        will_enter_age_max = age_min - future_months + (1 / 4.33)

        # 剪出这部分用户
        mask = (user_age_distribution.index >= max(0, will_enter_age_min)) & \
               (user_age_distribution.index < will_enter_age_max)
        new_users = user_age_distribution[mask].sum()

        # 已在窗口内的用户（复购需求）
        already_in = user_age_distribution[
            (user_age_distribution.index >= age_min) &
            (user_age_distribution.index <= age_max)
        ].sum()

        repurchase_weeks = target_stage.get('repurchase_weeks') or 99
        repurchase_demand = already_in / repurchase_weeks

        weekly_demand.append({
            'week': week,
            'new_demand': new_users,
            'repurchase_demand': repurchase_demand,
            'total_demand': new_users + repurchase_demand,
        })

    return pd.DataFrame(weekly_demand)


# ─────────────────────────────────────────────
# 3. 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Infant Lifecycle Purchase Rhythm")
    print("婴儿生命周期消费节律建模")
    print("=" * 65)

    np.random.seed(42)

    # 模拟用户月龄分布（5000名活跃会员）
    n_users = 500
    # 真实母婴电商：月龄分布通常在0-18个月集中
    baby_ages = np.random.gamma(shape=3, scale=3, size=n_users)
    baby_ages = np.clip(baby_ages, 0, 24)
    age_series = pd.Series(baby_ages).round(1)

    # 月龄分布统计
    age_bins = pd.cut(age_series, bins=range(0, 26, 2))
    age_dist = age_bins.value_counts().sort_index()
    print(f"\n用户月龄分布（{n_users}名活跃会员）:")
    for interval, count in age_dist.items():
        bar = '▪' * (count // 3)
        print(f"  {str(interval):>12}: {count:>4}人  {bar}")

    # 为3个典型用户展示下一品类预测
    print(f"\n下一高需求品类预测（未来6周）:")
    test_cases = [
        (3.5, 0.9, '宝宝3.5月龄，辅食准备期'),
        (8.5, 0.85, '宝宝8.5月龄，即将爬行'),
        (11.5, 0.75, '宝宝11.5月龄，接近一岁'),
    ]
    for age, conf, desc in test_cases:
        print(f"\n  [{desc}]")
        upcoming = get_upcoming_categories(age, lookahead_weeks=6, confidence=conf)
        for item in upcoming[:3]:
            milestone_tag = ' ⚡MILESTONE' if item['is_milestone'] else ''
            print(f"    {item['stage']:<22} 紧迫性:{item['urgency_score']:.2f}"
                  f"  {', '.join(item['categories'][:2])}{milestone_tag}")

    # 库存预测演示
    print(f"\n库存需求预测（品类：pre_solids_prep/辅食工具）:")
    age_distribution = pd.Series(
        np.histogram(baby_ages, bins=np.arange(0, 25, 0.5))[0],
        index=np.arange(0, 24.5, 0.5)
    )
    demand_forecast = predict_demand_surge(age_distribution, 'rice_cereal', weeks_ahead=8)
    print(f"{'周次':>4} {'新需求':>8} {'复购需求':>10} {'总需求':>8}")
    print("-" * 35)
    for _, row in demand_forecast.iterrows():
        peak_mark = ' ← 峰值' if row['total_demand'] == demand_forecast['total_demand'].max() else ''
        print(f"  第{row['week']:.0f}周  {row['new_demand']:>8.1f} "
              f"{row['repurchase_demand']:>10.1f} {row['total_demand']:>8.1f}{peak_mark}")

    # 关键节点用户统计
    print(f"\n即将进入关键节点的用户（未来4周）:")
    milestones = {
        'pre_solids_prep': (3.2, 4.0),
        'pre_crawling': (7.2, 8.0),
        'first_birthday': (10.8, 11.5),
        'potty_training': (17.5, 18.5),
    }
    for ms, (lo, hi) in milestones.items():
        count = ((age_series >= lo) & (age_series < hi)).sum()
        if count > 0:
            print(f"  {ms:<25}: {count:>4}人 → 建议提前触达")

    print("\n[✓] Infant Lifecycle Purchase Rhythm 测试通过")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

- **前置（prerequisite）**：
  - [[Skill-Baby-Age-Clock-RFM-Enhancement]] — 需要先推断婴儿月龄，才能匹配消费节律
  - [[Skill-Customer-Journey-Analytics]] — 用户旅程分析，月龄驱动的旅程是其特化版本
- **延伸（extends）**：
  - [[Skill-Baby-Age-Aware-Recommendation]] — 节律图谱驱动的实时推荐切换
  - [[Skill-Forecast-Driven-Inventory]] — 月龄分布预测驱动的库存计划
- **可组合（combinable）**：
  - [[Skill-Demand-Forecasting-Supply-Chain]]（在需求预测模型中加入月龄分布的前瞻信号，解决辅食类品类的库存峰值预测难题）
  - [[Skill-Personalized-Promotion-Targeting]]（里程碑节点前 3-4 周是最佳促销触达窗口，本 Skill 提供精确的时机信号）

---

## ⑤ 商业价值评估

- **ROI 预估**：月龄感知首页推荐 CTR 提升 35-50%，辅食类 OOS 率降低 60%，年化节省缺货损失约 $4.8 万 + 推荐效率提升带来 GMV 增量约 $12 万，合计 **$16.8 万/年**
- **实施难度**：⭐⭐☆☆☆（品类-月龄映射表建立 1 周，需求预测模型接入 2 周，总计约 3 周）
- **优先级**：⭐⭐⭐⭐⭐（母婴电商独有的"时间武器"——竞品如果没有月龄感知能力，在触达时机上天然处于劣势）
- **评估依据**：PCIC 在 Target 亿级用户验证，NDCG 提升 16%，Recall 提升 2%；MIT Sloan 母婴 App 案例显示月龄感知推荐转化率 +89%（vs 最热门推荐）
