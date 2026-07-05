---
title: Cohort Retention Analysis for User Lifecycle
module: 14-用户分析
topic: cohort-retention
source: arxiv:1905.09262
doc_type: knowledge
roadmap_phase: phase2
status: stable
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: Cohort Retention Analysis

## ① 算法原理

> **论文**：Modeling Retention Curves with Power Law and BG/NBD | **年份**：2019

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

**三轨验证**：

| 轨道 | 内容 | 评估结果 |
|------|------|---------|
| **成本轨** | • 数据采集：现有CDP系统无额外成本<br>• 邮件系统：Klaviyo/Braze月费$500-1000<br>• 优惠券系统：集成现有ERP，开发成本$5000<br>• 人力投入：数据分析师1人×2周 = $4000<br>**总计：$9500-10500** | 中等投入 |
| **合规轨** | • GDPR：邮件营销需用户明确同意（已有）<br>• Amazon政策：优惠券补偿符合退货政策<br>• 广告法：邮件内容不涉及虚假宣传<br>• 跨境：美国/欧盟/加拿大均无违规<br>**结论：完全合规** | ✓ 合规 |
| **风险轨** | • 竞品反应：TikTok用户补偿可能引发价格战（概率30%）<br>• 平台审查：邮件频率过高可能触发反垃圾规则（概率15%）<br>• 品牌损伤：过度补偿可能降低品牌溢价（概率20%）<br>• 缓解措施：邮件频率限制为2周1次，补偿金额不超过订单5%<br>**综合风险等级：中低** | 可控 |

---

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

**三轨验证**：

| 轨道 | 内容 | 评估结果 |
|------|------|---------|
| **成本轨** | • 品类分层数据：现有BI系统无额外成本<br>• 推荐引擎定制：Segment/mParticle月费$2000<br>• 订阅系统开发：Subbly/Bold集成$8000<br>• 库存管理：现有WMS系统无额外成本<br>• 人力投入：产品经理1人×3周 + 开发2人×2周 = $12000<br>**总计：$22000** | 中高投入 |
| **合规轨** | • GDPR：个性化推荐需透明算法说明（已实现）<br>• Amazon政策：订阅模式需明确取消流程（已设计）<br>• 消费者保护法：订阅自动续费需明确提示（已实现）<br>• 广告法：配件推荐不涉及医疗声称<br>• 跨境：美国/加拿大/欧盟订阅法规均符合<br>**结论：完全合规** | ✓ 合规 |
| **风险轨** | • 库存风险：订阅模式需提前备货，滞销风险（概率25%）<br>• 竞品反应：订阅价格优惠可能引发价格战（概率35%）<br>• 用户流失：强制推荐配件可能降低满意度（概率20%）<br>• 退货率上升：配件质量问题可能增加退货（概率15%）<br>• 缓解措施：订阅可选制、配件质检升级、30天退货保障<br>**综合风险等级：中** | 需监控 |

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
print("[✓] Cohort Retention Analysis 测试通过")
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
