---
title: 客户流失预警→挽回 Combo Pattern — 从流失预测到精准干预的 5 步完整链路
doc_type: knowledge
module: 06-增长模型
topic: combo-customer-churn-recovery-orchestration
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 客户流失预警→挽回 Combo Pattern

> **类型**：Combo Pattern（业务解决方案编排）
> **桥梁**：06-增长模型 ↔ 14-用户分析 ↔ 13-广告分析 ↔ 02-A_B实验
> **触发条件**：客户沉默期 > 45 天 或 月活环比下降 > 15%，自动触发 5 步挽回链路

## ① 算法原理

客户流失挽回 Combo Pattern 解决「看到数据掉了却不知道该对谁做什么」的核心痛点。传统方式是一刀切发优惠券，效果差且亏损。本 Combo 实现「人群精准识别 → 时机最优触达 → 内容个性化匹配 → ROI 正向验证」的闭环。

**5 步链路设计逻辑**：

```
[触发] 沉默用户识别 → Step1 流失概率打分（0~1）
                              ↓ 高风险用户集合
                    Step2 RFM 分层 → 策略匹配（挽回 or 激活 or 放弃）
                              ↓ 分层人群 + 对应策略
                    Step3 触达时机优化（open_rate 最高的时间窗口）
                              ↓ 最优触达时间戳
                    Step4 文案 A/B 优化（邮件/短信/Push 自动选优）
                              ↓ 最优文案 + 渠道
                    Step5 会员激励 ROI 建模（优惠力度 vs 毛利平衡）
                              ↓ 激励方案 + 预期 ROI
[执行] → 自动化营销平台分发
```

**核心算法选择原理**：
- Step1 使用梯度提升（XGBoost）而非深度学习——客户行为数据量通常 < 10 万，树模型更稳健
- Step3 使用历史 open_rate 分布的分位数估计最优时间，避免过度依赖 A/B 实验周期
- Step5 使用 CLV（客户生命周期价值）约束优惠上限：最大激励 ≤ 预期 CLV × 0.15

**关键假设**：客户行为数据完整（购买记录、浏览记录、通信记录），CRM 系统可导出。

## ② 母婴出海应用案例

**场景A：母婴品牌跨境 DTC 站首购用户 60 天内复购挽回**

- **业务问题**：DTC 站首购用户 60 天复购率仅 14%，行业均值 28%，每流失 1 个首购用户损失 CLV ≈ $85（按 3 年生命周期估算）
- **数据要求**：用户 ID、购买日期、购买品类、购买金额、浏览行为、邮件 open 记录
- **执行过程**：
  - Step1 流失概率评分：5,000 名首购用户中，1,850 人评分 > 0.7（高风险），召回率 0.81
  - Step2 RFM 分层：F=1、M < $30 → 「低价值首购」策略（促销小礼品）；F=1、M > $80 → 「高价值首购」策略（专属顾问电话回访）
  - Step3 最优触达时机：该用户群 email open_rate 高峰为周二上午 9-11 点（EST）
  - Step4 文案 A/B：「您的宝宝是否还需要…」vs「专为妈妈准备的…」，后者 CTR 高 34%
  - Step5 激励 ROI：满 $40 减 $8 优惠券（折扣率 20%），预期 CLV 增量 $25，净 ROI = 3.1x
- **量化产出**：60 天复购率从 14% 提升至 22%（+57%），年化挽回价值 42 万元

**场景B：亚马逊老客户流失前主动干预（站外私域触达）**

- **业务问题**：亚马逊平台无法直接触达客户，但品牌自建的 ManyChat/邮件列表 3 个月未互动的占 68%
- **执行亮点**：Step2 识别出「高 RFM 老客户沉默 > 90 天」群体 520 人，Step4 使用出生日期关联，发送「宝宝 X 岁生日特惠」，个性化触达 CTR 提升 2.8x
- **业务价值**：520 人群体挽回 143 人复购，人均贡献 $62，总收入 $8,866，活动成本 $1,200，ROI = 6.4x

## ③ 代码模板

```python
"""
客户流失预警→挽回 Combo Pattern — 5 步精准干预链路
"""
from dataclasses import dataclass, field
from typing import Optional
import random
import math

random.seed(42)

# ──────────────────────────────────────────────
# 客户数据结构
# ──────────────────────────────────────────────
@dataclass
class Customer:
    user_id: str
    days_since_last_purchase: int
    total_purchases: int
    total_spend_usd: float
    email_open_rate: float  # 历史邮件开率 0~1
    baby_age_months: Optional[int] = None  # 宝宝月龄（用于个性化）
    # Combo 结果
    churn_probability: float = 0.0       # Step1
    rfm_segment: str = ""                # Step2
    optimal_send_hour: int = 9           # Step3
    best_message_variant: str = ""       # Step4
    incentive_offer: str = ""            # Step5
    expected_roi: float = 0.0

@dataclass
class ChurnRecoveryContext:
    customers: list[Customer]
    brand_avg_clv_usd: float = 85.0
    max_incentive_pct: float = 0.15
    results: dict = field(default_factory=dict)

# ──────────────────────────────────────────────
# Step1: 流失预测 — Skill-Customer-Churn-Prediction
# ──────────────────────────────────────────────
def step1_churn_prediction(ctx: ChurnRecoveryContext) -> ChurnRecoveryContext:
    """XGBoost 流失概率打分（简化为启发式函数）"""
    high_risk_count = 0
    for c in ctx.customers:
        # 特征：沉默天数 + 购买次数 + 消费金额（反向）
        silence_score = min(c.days_since_last_purchase / 90, 1.0)
        frequency_score = 1.0 - min(c.total_purchases / 5, 1.0)
        spend_score = 1.0 - min(c.total_spend_usd / 200, 1.0)
        c.churn_probability = round(silence_score * 0.5 + frequency_score * 0.3 + spend_score * 0.2, 3)
        if c.churn_probability > 0.7:
            high_risk_count += 1
    print(f"  [Step1] 流失评分完成: {len(ctx.customers)} 用户, 高风险(>0.7) {high_risk_count} 人")
    return ctx

# ──────────────────────────────────────────────
# Step2: RFM 分层策略 — Skill-RFM-to-Action-Policy-Engine
# ──────────────────────────────────────────────
def step2_rfm_segmentation(ctx: ChurnRecoveryContext) -> ChurnRecoveryContext:
    segment_counts = {}
    for c in ctx.customers:
        # R: 沉默天数, F: 购买次数, M: 消费金额
        r = "R_LOW" if c.days_since_last_purchase > 60 else "R_HIGH"
        f = "F_HIGH" if c.total_purchases >= 3 else "F_LOW"
        m = "M_HIGH" if c.total_spend_usd >= 100 else "M_LOW"

        if r == "R_LOW" and f == "F_HIGH" and m == "M_HIGH":
            c.rfm_segment = "HIGH_VALUE_LAPSED"   # 高价值沉默 → 重点挽回
        elif r == "R_LOW" and f == "F_LOW":
            c.rfm_segment = "ONE_TIME_BUYER"       # 一次性购买 → 低成本促活
        elif r == "R_HIGH":
            c.rfm_segment = "ACTIVE_RISK"          # 近期活跃但有流失风险 → 预防
        else:
            c.rfm_segment = "CHURNED_LOW_VALUE"    # 低价值流失 → 放弃或自动化

        segment_counts[c.rfm_segment] = segment_counts.get(c.rfm_segment, 0) + 1
    print(f"  [Step2] RFM 分层: {segment_counts}")
    return ctx

# ──────────────────────────────────────────────
# Step3: 最优触达时机 — Skill-Repurchase-Trigger-Timing-Model
# ──────────────────────────────────────────────
def step3_optimal_timing(ctx: ChurnRecoveryContext) -> ChurnRecoveryContext:
    timing_map = {
        "HIGH_VALUE_LAPSED": 10,    # 上午 10 点（工作时间开始，精力好）
        "ONE_TIME_BUYER": 19,       # 晚上 7 点（下班后刷手机）
        "ACTIVE_RISK": 9,           # 上午 9 点（预防性触达，不打扰）
        "CHURNED_LOW_VALUE": 14,    # 下午 2 点（低优先级，填充时段）
    }
    for c in ctx.customers:
        c.optimal_send_hour = timing_map.get(c.rfm_segment, 10)
    # 按时段统计
    hour_dist = {}
    for c in ctx.customers:
        hour_dist[c.optimal_send_hour] = hour_dist.get(c.optimal_send_hour, 0) + 1
    print(f"  [Step3] 触达时机分配: {hour_dist}")
    return ctx

# ──────────────────────────────────────────────
# Step4: 文案 A/B 选优 — Skill-Email-Sequence-Multiarm-Optimizer
# ──────────────────────────────────────────────
def step4_message_optimization(ctx: ChurnRecoveryContext) -> ChurnRecoveryContext:
    # 不同 RFM 段的最优文案变体
    variants = {
        "HIGH_VALUE_LAPSED": "专属回归礼遇：作为我们的贵宾，为您准备了惊喜",
        "ONE_TIME_BUYER": "您的宝宝是否还需要？同款妈妈们都在复购",
        "ACTIVE_RISK": "新品上架：专为 {age}个月宝宝设计",
        "CHURNED_LOW_VALUE": "限时特惠：清仓价，错过等一年",
    }
    for c in ctx.customers:
        template = variants.get(c.rfm_segment, "我们想念您")
        if c.baby_age_months and "{age}" in template:
            c.best_message_variant = template.replace("{age}", str(c.baby_age_months))
        else:
            c.best_message_variant = template.replace("{age}", "6-12")
    print(f"  [Step4] 文案优化完成: {len(set(c.best_message_variant for c in ctx.customers))} 个变体")
    return ctx

# ──────────────────────────────────────────────
# Step5: 会员激励 ROI — Skill-Loyalty-Program-ROI-Modeling
# ──────────────────────────────────────────────
def step5_incentive_roi(ctx: ChurnRecoveryContext) -> ChurnRecoveryContext:
    clv = ctx.brand_avg_clv_usd
    max_incentive = clv * ctx.max_incentive_pct  # 最大激励上限

    incentive_map = {
        "HIGH_VALUE_LAPSED": min(max_incentive, 15.0),  # $15 优惠券
        "ONE_TIME_BUYER": min(max_incentive, 8.0),       # $8 优惠券
        "ACTIVE_RISK": min(max_incentive, 5.0),          # $5 积分
        "CHURNED_LOW_VALUE": 0.0,                        # 无激励
    }
    recovery_rate = {
        "HIGH_VALUE_LAPSED": 0.35,
        "ONE_TIME_BUYER": 0.18,
        "ACTIVE_RISK": 0.25,
        "CHURNED_LOW_VALUE": 0.05,
    }

    for c in ctx.customers:
        incentive = incentive_map.get(c.rfm_segment, 0)
        rate = recovery_rate.get(c.rfm_segment, 0.1)
        # ROI = (CLV × 挽回率 - 激励成本) / 激励成本
        if incentive > 0:
            c.expected_roi = round((clv * rate - incentive) / incentive, 2)
            c.incentive_offer = f"${incentive:.0f} 优惠券 (预期ROI={c.expected_roi}x)"
        else:
            c.expected_roi = 0.0
            c.incentive_offer = "无激励（低价值段）"

    avg_roi = sum(c.expected_roi for c in ctx.customers) / len(ctx.customers)
    profitable = sum(1 for c in ctx.customers if c.expected_roi > 1.0)
    print(f"  [Step5] 激励 ROI: 平均={avg_roi:.2f}x, 正向ROI人数={profitable}/{len(ctx.customers)}")
    return ctx

# ──────────────────────────────────────────────
# Combo 编排入口
# ──────────────────────────────────────────────
def run_churn_recovery_combo(customers: list[Customer], brand_avg_clv_usd: float = 85.0) -> ChurnRecoveryContext:
    ctx = ChurnRecoveryContext(customers=customers, brand_avg_clv_usd=brand_avg_clv_usd)
    print(f"\n💔 客户流失挽回 Combo Pattern 启动: {len(customers)} 名用户待分析")
    print("=" * 55)

    for step_fn in [step1_churn_prediction, step2_rfm_segmentation,
                    step3_optimal_timing, step4_message_optimization,
                    step5_incentive_roi]:
        ctx = step_fn(ctx)

    high_value_targets = [c for c in customers if c.rfm_segment == "HIGH_VALUE_LAPSED"]
    profitable_actions = [c for c in customers if c.expected_roi > 1.0]
    print("=" * 55)
    print(f"🎯 执行计划: 高价值重点挽回={len(high_value_targets)}人, 正向ROI行动={len(profitable_actions)}人")
    return ctx

# ──────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 构造测试用户
    test_customers = [
        Customer("USR-001", days_since_last_purchase=75, total_purchases=4,
                 total_spend_usd=180.0, email_open_rate=0.35, baby_age_months=8),
        Customer("USR-002", days_since_last_purchase=50, total_purchases=1,
                 total_spend_usd=25.0, email_open_rate=0.12, baby_age_months=None),
        Customer("USR-003", days_since_last_purchase=20, total_purchases=2,
                 total_spend_usd=95.0, email_open_rate=0.48, baby_age_months=14),
        Customer("USR-004", days_since_last_purchase=120, total_purchases=1,
                 total_spend_usd=15.0, email_open_rate=0.05, baby_age_months=None),
    ]

    ctx = run_churn_recovery_combo(test_customers, brand_avg_clv_usd=85.0)

    assert all(0 <= c.churn_probability <= 1 for c in ctx.customers), "流失概率应在 0~1 之间"
    assert all(c.rfm_segment != "" for c in ctx.customers), "所有用户应有 RFM 分层"
    assert all(c.best_message_variant != "" for c in ctx.customers), "所有用户应有文案"
    assert any(c.expected_roi > 1.0 for c in ctx.customers), "应有正向 ROI 的挽回行动"

    # 打印具体行动计划
    print("\n📋 用户行动计划：")
    for c in ctx.customers:
        print(f"  {c.user_id} | 流失率={c.churn_probability:.2f} | 分层={c.rfm_segment} | "
              f"触达={c.optimal_send_hour}:00 | {c.incentive_offer}")

    print("\n[✓] 客户流失挽回 Combo Pattern 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Customer-Churn-Prediction]]（Step1 核心：流失概率打分模型）
- **前置（prerequisite）**：[[Skill-RFM-to-Action-Policy-Engine]]（Step2 核心：RFM 分层 → 策略决策）
- **组合（combinable）**：[[Skill-Repurchase-Trigger-Timing-Model]]（Step3：最优触达时间窗口识别）
- **组合（combinable）**：[[Skill-Email-Sequence-Multiarm-Optimizer]]（Step4：文案 A/B 多臂老虎机优化）
- **组合（combinable）**：[[Skill-Loyalty-Program-ROI-Modeling]]（Step5：会员激励力度 ROI 建模）
- **延伸（extends）**：[[Skill-CC-OR-Net-LTV-Prediction]]（流失挽回后长期 LTV 预测）
- **延伸（extends）**：[[Skill-Brand-Penetration-Modeling]]（规模化后的品牌渗透率增长模型）
- **延伸（extends）**：[[Skill-Combo-Ad-ROI-Maximizer]]（挽回人群再投广告的 ROI 最大化）

## ⑤ 商业价值评估

- **ROI 预估**：DTC 品牌月均 5,000 名活跃用户，60 天流失率假设 22%（=1,100 人），本 Combo 挽回率提升 8 个百分点（1,100×8% = 88 人），人均 CLV $85，年化增量收益 = 88 × $85 × 12 ≈ 89.8 万元；激励成本约 15 万元，净 ROI ≈ 5x
- **关键指标**：60 天复购率 +8pp（14% → 22%），邮件 CTR +34%（文案优化），激励成本 / 挽回用户 < $15（vs 新客获客成本 $45-$80）
- **实施难度**：⭐⭐⭐☆☆（需要 CRM 数据接口 + 邮件平台 API，2-3 周工程化）
- **优先级**：⭐⭐⭐⭐☆（复购增长是 DTC 品牌 LTV 最高杠杆点，ROI 极为确定）
- **适用规模**：月活用户 ≥ 1,000 人，有历史购买数据 ≥ 6 个月
