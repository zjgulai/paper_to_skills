---
title: Cohort Retention Analysis for User Lifecycle
module: 14-用户分析
topic: cohort-retention
status: stable
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: Cohort Retention Analysis

## ① 算法原理

**核心问题**：新用户来了之后，第7天还剩多少？第30天呢？第90天呢？不同月份来的用户，留存曲线一样吗？Cohort分析把用户按"首次活跃时间"分组，追踪每组的留存轨迹。

**Cohort定义**：
- **时间Cohort**：按首次购买/注册月份分组（如"2025年1月Cohort"）
- **行为Cohort**：按首次行为特征分组（如"首单买奶粉的用户"vs"首单买纸尿裤的用户"）
- **渠道Cohort**：按获客渠道分组（Facebook/Google/TikTok）

**留存曲线（Retention Curve）**：

$$Retention(d) = \frac{\text{首次活跃后第d天仍活跃的用户数}}{\text{该Cohort总用户数}}$$

**关键指标**：

| 指标 | 定义 | 业务含义 |
|------|------|---------|
| **D1/D7/D30留存** | 第1/7/30天留存率 | 短期/中期/长期粘性 |
| **半衰期** | 留存率降到50%的天数 | 用户生命周期长度 |
| **曲线曲率** | 前7天下降速度 |  onboarding 质量 |
| **长期平台值** | 留存曲线渐近线 | 核心用户占比 |

**预测留存的方法**：

**1. 幂律模型（Power Law）**
$$Retention(d) = a \cdot d^{-b}$$
- $a$ ≈ D1留存
- $b$ 决定下降速度
- 拟合历史数据预测未来留存

**2. BG/NBD模型**
- 概率模型，假设用户的购买服从泊松过程，流失服从几何分布
- 可预测：未来某段时间内的购买次数、活跃用户数量
- 适用于非契约型场景（如电商）

**反直觉洞察**：
- D1留存提升5%，LTV可能提升20%——因为留存是复利效应
- 不同渠道的用户留存差异巨大：Facebook广告用户D30留存可能只有5%，而自然搜索用户可能30%
- " cohort 退化"是常态——每月新增用户的留存曲线会逐渐变差，因为好摘的果子先摘完了

---

## ② 母婴出海应用案例

### 场景1：新客留存诊断

**业务问题**：Momcozy 2025年1月新注册用户10,000人，D7留存15%，D30留存5%。行业标杆D7=25%，D30=12%。差距在哪？

**Cohort分析**：

| Cohort | D1 | D7 | D30 | 诊断 |
|--------|-----|-----|-----|------|
| 2024-10 | 35% | 22% | 10% | 基准 |
| 2024-11 | 33% | 20% | 9% | 下降 |
| 2024-12 | 30% | 18% | 8% | 继续下降 |
| 2025-01 | 28% | 15% | 5% | 恶化明显 |

**根因分析**：
- 渠道构成变化：1月新客中TikTok占比从20%提升到50%，TikTok用户质量较低
- Onboarding流程：1月更新了注册流程，但新用户引导缺失
- 首单体验：1月物流延迟增加，影响复购意愿

**优化策略**：
- TikTok用户单独设计onboarding流程
- 注册后24小时内发送"首单引导"邮件
- 物流延迟用户自动发放补偿优惠券

### 场景2：不同品类的留存差异

**业务问题**：首单买奶粉的用户 vs 首单买吸奶器的用户，谁的长期留存更好？

**Cohort对比**：

| 首单品类 | D1 | D7 | D30 | D90 | LTV(12月) |
|---------|-----|-----|-----|-----|----------|
| 奶粉 | 40% | 28% | 18% | 12% | $450 |
| 吸奶器 | 25% | 15% | 8% | 5% | $280 |
| 纸尿裤 | 35% | 22% | 14% | 10% | $380 |

**洞察**：奶粉用户留存最高（消耗品+定期复购），吸奶器用户留存最低（耐用品+一次性购买）。

**策略**：
- 吸奶器用户首单后强推配件（储奶袋、奶嘴）提升复购
- 纸尿裤用户推套装订阅（按月配送）锁定长期留存

---

## ③ 代码模板

```python
"""
Cohort Retention Analysis — Cohort留存分析
支持：留存矩阵计算、留存曲线、幂律拟合、BG/NBD简化版
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class CohortAnalyzer:
    """Cohort分析器"""

    def __init__(self):
        pass

    def create_cohort_table(self, df, user_col, date_col, period='M'):
        """
        创建Cohort留存表

        Args:
            df: DataFrame with user_id and activity_date
            user_col: 用户ID列
            date_col: 活跃日期列
            period: 'D'=日, 'W'=周, 'M'=月
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # 获取每个用户的首次活跃日期
        user_first = df.groupby(user_col)[date_col].min().reset_index()
        user_first.columns = [user_col, 'first_date']

        # 合并
        df = df.merge(user_first, on=user_col)

        # 计算period
        if period == 'M':
            df['first_period'] = df['first_date'].dt.to_period('M')
            df['activity_period'] = df[date_col].dt.to_period('M')
        elif period == 'W':
            df['first_period'] = df['first_date'].dt.to_period('W')
            df['activity_period'] = df[date_col].dt.to_period('W')
        else:
            df['first_period'] = df['first_date'].dt.date
            df['activity_period'] = df[date_col].dt.date

        # 计算period_diff
        df['period_diff'] = (df['activity_period'] - df['first_period']).apply(
            lambda x: x.n if hasattr(x, 'n') else (x.days if hasattr(x, 'days') else 0)
        )

        # Cohort大小
        cohort_sizes = user_first.groupby(
            user_first['first_date'].dt.to_period('M') if period == 'M' else user_first['first_date'].dt.date
        )[user_col].nunique()

        # 留存矩阵
        cohort_data = df.groupby(['first_period', 'period_diff'])[user_col].nunique().reset_index()
        cohort_table = cohort_data.pivot(index='first_period', columns='period_diff', values=user_col)

        # 计算留存率
        retention_table = cohort_table.divide(cohort_sizes, axis=0)

        return retention_table, cohort_sizes

    def fit_power_law(self, retention_series):
        """
        拟合留存幂律模型: Retention(d) = a * d^(-b)
        """
        days = np.array(retention_series.index)
        rates = np.array(retention_series.values)

        # 过滤掉0值和NaN
        mask = (rates > 0) & (~np.isnan(rates))
        days = days[mask]
        rates = rates[mask]

        if len(days) < 2:
            return None, None

        # 对数线性回归
        log_d = np.log(days)
        log_r = np.log(rates)

        b, log_a = np.polyfit(log_d, log_r, 1)
        a = np.exp(log_a)

        return a, -b  # 注意：polyfit返回斜率，我们模型中是 -b

    def predict_ltv(self, a, b, arpu, max_days=365):
        """
        基于幂律留存预测LTV

        LTV ≈ ARPU * ∑(Retention(d)) ≈ ARPU * a * ∑(d^(-b))
        """
        days = np.arange(1, max_days + 1)
        retentions = a * (days ** (-b))
        total_active_days = np.sum(retentions)
        return arpu * total_active_days

    def compare_cohorts(self, retention_tables, cohort_names):
        """对比多个Cohort的留存曲线"""
        results = []
        for table, name in zip(retention_tables, cohort_names):
            avg_retention = table.mean()
            results.append({
                'cohort': name,
                'd1': avg_retention.get(1, np.nan),
                'd7': avg_retention.get(7, np.nan),
                'd30': avg_retention.get(30, np.nan),
                'd90': avg_retention.get(90, np.nan),
            })
        return pd.DataFrame(results)


def generate_cohort_data(n_users=5000, start_date='2024-10-01', periods=6):
    """生成Cohort分析模拟数据"""
    np.random.seed(42)
    start = pd.to_datetime(start_date)

    records = []
    for month in range(periods):
        cohort_date = start + pd.DateOffset(months=month)
        n = int(1000 * (1 - month * 0.05))  # 逐月略有下降

        for _ in range(n):
            user_id = f"user_{month}_{np.random.randint(100000)}"

            # D1留存率逐月下降（模拟质量恶化）
            d1_rate = 0.40 - month * 0.03

            if np.random.random() < d1_rate:
                records.append({'user_id': user_id, 'date': cohort_date + timedelta(days=1)})

                # D7留存
                d7_rate = d1_rate * 0.70
                if np.random.random() < d7_rate:
                    records.append({'user_id': user_id, 'date': cohort_date + timedelta(days=7)})

                    # D30留存
                    d30_rate = d7_rate * 0.50
                    if np.random.random() < d30_rate:
                        records.append({'user_id': user_id, 'date': cohort_date + timedelta(days=30)})

    return pd.DataFrame(records)


if __name__ == '__main__':
    df = generate_cohort_data()
    analyzer = CohortAnalyzer()
    retention, sizes = analyzer.create_cohort_table(df, 'user_id', 'date', period='M')

    print("Cohort留存矩阵:")
    print(retention.round(3))

    print("\n平均留存曲线:")
    avg = retention.mean()
    print(avg.round(3))

    a, b = analyzer.fit_power_law(avg.dropna())
    if a and b:
        print(f"\n幂律模型: Retention(d) = {a:.3f} * d^(-{b:.3f})")
        ltv = analyzer.predict_ltv(a, b, arpu=50, max_days=365)
        print(f"预测LTV (ARPU=$50): ${ltv:.2f}")
```

---


## ④ 技能关联

### 前置技能
- [Skill-User-Funnel-Analysis](../14-用户分析/[[Skill-User-Funnel-Analysis]].md) — 漏斗分析是留存分析的姊妹方法

### 延伸技能
- [Skill-RFM-Customer-Segmentation](../06-增长模型/[[Skill-RFM-Customer-Segmentation]].md) — 对各 cohort 进一步做 RFM 分群
- [Skill-LTV-Prediction-ZILN](../06-增长模型/[[Skill-LTV-Prediction-ZILN]].md) — cohort 留存曲线是 LTV 模型核心输入

### 可组合
- [Skill-Customer-Churn-Prediction](../06-增长模型/[[Skill-Customer-Churn-Prediction]].md) — Cohort 留存指标定义流失阈值

## ⑤ 商业价值评估

- **ROI**：D1留存提升5% → LTV提升20%，年增收百万级
- **难度**：⭐⭐☆☆☆（2/5）— 主要是数据透视和可视化
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 衡量产品健康度的核心指标
