---
title: RFM Customer Segmentation for Targeted Marketing
module: 06-增长模型
topic: customer-segmentation
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase2
---

# Skill Card: RFM Customer Segmentation

## ① 算法原理

**核心思想**：用三个维度刻画用户价值——
- **R (Recency)**：最近一次购买距今多少天。越近越可能再次购买。
- **F (Frequency)**：购买次数。越高越忠诚。
- **M (Monetary)**：累计消费金额。越高价值越大。

**为什么RFM有效**：

母婴电商的特殊性使RFM尤其适用：
- **生命周期明确**：从孕期到 toddler（0-3岁），每个阶段需求完全不同
- **复购驱动力强**：奶粉、纸尿裤是消耗品，必须定期复购
- **高价值用户集中**：少量高消费用户贡献大部分GMV

**分群方法**：

1. **等频分箱（Quantile-based）**：将R/F/M各自分为1-5分（5分最高）
2. **组合标签**：如 R=5, F=4, M=5 → "冠军用户"

**典型用户群**：

| 用户群 | R | F | M | 特征 | 策略 |
|--------|---|---|---|------|------|
| 冠军用户 | 高 | 高 | 高 | 核心高价值 | VIP专属服务、新品优先体验 |
| 忠诚用户 | 中 | 高 | 中 | 经常买但金额一般 | 升级客单价（套装推荐） |
| 潜力用户 | 高 | 低 | 高 | 新用户但消费高 | 快速建立复购习惯 |
| 沉睡用户 | 低 | 中 | 中 | 很久没买 | 召回活动、优惠券 |
| 流失风险 | 低 | 低 | 高 | 曾经高价值但很久没来 | 大额优惠券、电话回访 |
| 新用户 | 高 | 低 | 低 | 刚注册/首单 | 引导二次购买 |
| 低价值 | 低 | 低 | 低 | 偶尔买且金额低 | 降低营销成本，自然留存 |

**反直觉洞察**：
- RFM不是静态的——用户每个月都在不同群之间迁移
- "冠军用户"的流失是最痛的损失——维护1个老用户的成本是获取新用户的1/5
- 母婴用户的RFM变化有强规律性：怀孕期（高F）→ 新生儿期（极高M）→ toddler期（F下降）→ 二胎（重新高F）

---

## ② 母婴出海应用案例

### 场景1：精准营销推送

**业务问题**：Momcozy 有10万注册用户，营销团队想给不同用户发不同的邮件/推送。但一刀切的消息打开率<2%，转化率<0.1%。

**RFM应用**：
1. 计算每个用户的R、F、M得分
2. 分群并制定差异化策略：
   - 冠军用户（~5%）：新品预售邀请、VIP专属折扣
   - 沉睡用户（~20%）："我们想念你" + 15% off coupon
   - 新用户（~15%）：首单复购引导（买吸奶器→推荐储奶袋）
   - 潜力用户（~10%）：快速升级（满$150送配件套装）

**预期产出**：
- 邮件打开率：2% → 8%
- 点击率：0.3% → 1.5%
- 转化率：0.1% → 0.5%

### 场景2：用户生命周期预警

**业务问题**：识别即将从"忠诚用户"滑向"沉睡用户"的人群，在流失前干预。

**RFM迁移分析**：
1. 每月计算用户RFM并记录历史标签
2. 检测迁移模式：
   - 冠军 → 忠诚 → 沉睡 → 流失（危险路径）
   - 忠诚 → 冠军（升级路径）
3. 对处于"忠诚→沉睡"迁移中的用户触发召回

**预期产出**：
- 预警准确率：70%+
- 召回成功率：20-30%（vs 无预警召回的5%）

---

## ③ 代码模板

```python
"""
RFM Customer Segmentation — 用户价值分群
用于精准营销和用户生命周期管理
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class RFMSegmentation:
    """RFM用户分群"""

    def __init__(self, recency_bins=5, frequency_bins=5, monetary_bins=5):
        self.recency_bins = recency_bins
        self.frequency_bins = frequency_bins
        self.monetary_bins = monetary_bins

    def calculate_rfm(self, df, customer_col, date_col, amount_col):
        """
        计算RFM指标

        Args:
            df: DataFrame，每行一个订单
            customer_col: 用户ID列名
            date_col: 订单日期列名
            amount_col: 订单金额列名
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # 计算基准日期（数据最后一天+1）
        snapshot_date = df[date_col].max() + timedelta(days=1)

        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
            customer_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

        # R：越小越好（最近购买），所以反转分数
        rfm['r_score'] = pd.qcut(rfm['recency'], self.recency_bins,
                                  labels=range(self.recency_bins, 0, -1)).astype(int)
        # F：越大越好
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), self.frequency_bins,
                                  labels=range(1, self.frequency_bins + 1)).astype(int)
        # M：越大越好
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), self.monetary_bins,
                                  labels=range(1, self.monetary_bins + 1)).astype(int)

        return rfm

    def segment(self, rfm):
        """根据RFM得分分群"""
        def classify(row):
            r, f, m = row['r_score'], row['f_score'], row['m_score']

            if r >= 4 and f >= 4 and m >= 4:
                return '冠军用户'
            elif r >= 3 and f >= 3 and m >= 3:
                return '忠诚用户'
            elif r >= 4 and f <= 2 and m >= 4:
                return '潜力用户'
            elif r <= 2 and f >= 3:
                return '沉睡用户'
            elif r <= 2 and f <= 2 and m >= 4:
                return '流失风险'
            elif r >= 4 and f <= 2 and m <= 2:
                return '新用户'
            else:
                return '低价值'

        rfm['segment'] = rfm.apply(classify, axis=1)
        return rfm

    def get_segment_strategy(self, segment):
        """获取分群策略建议"""
        strategies = {
            '冠军用户': {'action': 'VIP专属服务', 'channel': '专属客服+新品预售', 'discount': '0%'},
            '忠诚用户': {'action': '升级客单价', 'channel': '邮件推送', 'discount': '5%'},
            '潜力用户': {'action': '快速建立复购', 'channel': 'APP推送', 'discount': '10%'},
            '沉睡用户': {'action': '召回激活', 'channel': '邮件+短信', 'discount': '15%'},
            '流失风险': {'action': '大额挽回', 'channel': '电话+邮件', 'discount': '20%'},
            '新用户': {'action': '引导复购', 'channel': 'APP推送', 'discount': '10%'},
            '低价值': {'action': '低成本维护', 'channel': '自然留存', 'discount': '0%'},
        }
        return strategies.get(segment, {})


def generate_momcozy_orders(n_customers=5000, n_orders=15000, random_state=42):
    """
    生成Momcozy模拟订单数据
    """
    np.random.seed(random_state)

    customer_ids = np.random.choice(range(10000, 20000), n_customers, replace=False)

    orders = []
    start_date = datetime(2025, 1, 1)

    for _ in range(n_orders):
        cid = np.random.choice(customer_ids)
        # 模拟购买时间（近30天概率更高）
        days_ago = int(np.random.exponential(60))
        days_ago = min(days_ago, 365)
        order_date = start_date + timedelta(days=365-days_ago)

        # 模拟金额（母婴客单价 $50-$200）
        amount = np.random.lognormal(4.5, 0.5)
        amount = np.clip(amount, 20, 300)

        orders.append({
            'customer_id': cid,
            'order_date': order_date,
            'amount': round(amount, 2)
        })

    return pd.DataFrame(orders)


# 示例
if __name__ == '__main__':
    df = generate_momcozy_orders()
    rfm_seg = RFMSegmentation()
    rfm = rfm_seg.calculate_rfm(df, 'customer_id', 'order_date', 'amount')
    rfm = rfm_seg.segment(rfm)

    print("RFM分群结果:")
    print(rfm['segment'].value_counts())
    print("\n冠军用户示例:")
    print(rfm[rfm['segment'] == '冠军用户'].head(3)[['customer_id', 'recency', 'frequency', 'monetary']])
print("[✓] RFM Customer Segmentation 测试通过")
```

---


## ④ 技能关联

### 前置技能
- [Skill-Feature-Engineering](../12-ML基础/[[Skill-Feature-Engineering]].md) — RFM 三维度本身是特征工程产物

### 延伸技能
- [Skill-LTV-Prediction-ZILN](../06-增长模型/[[Skill-LTV-Prediction-ZILN]].md) — RFM 分群是 LTV 预测的常用先验

### 可组合
- [Skill-Cohort-Retention-Analysis](../14-用户分析/[[Skill-Cohort-Retention-Analysis]].md) — RFM 分群后看每群的留存曲线
- [Skill-User-Funnel-Analysis](../14-用户分析/[[Skill-User-Funnel-Analysis]].md) — RFM 分群后对比各群的漏斗转化

- **可组合（combinable）**：[[Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎]]（VOC标签增强RFM分群解释性）
## ⑤ 商业价值评估

- **ROI**：营销转化率提升3-5x，营销成本降低30-50%
- **难度**：⭐☆☆☆☆（1/5）— 最简单实用的分群方法
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 任何营销团队的必备基础能力
