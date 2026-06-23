---
title: PPC Bid Manipulation Defense — 识别竞品恶意点击耗费广告费
doc_type: knowledge
module: 19-风控反欺诈
topic: ppc-bid-manipulation-defense
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-PPC-Bid-Manipulation-Defense

## ① 算法原理（≤300字）

**核心问题**：竞品通过频繁点击你的 PPC 广告耗尽你的每日预算，导致广告在下午就停止展示，你的 Listing 在黄金时段失去曝光。这种「点击欺诈」（Click Fraud）在 Amazon 平台上较难直接确认，但可以通过流量异常模式检测。

**检测特征**：

恶意点击 vs 真实点击的统计差异：

| 特征 | 真实点击 | 恶意点击 |
|------|---------|---------|
| 点击→购买转化率 | 8-15% | < 1% |
| 时间分布 | 全天分散 | 短时集中爆发 |
| 点击间隔 | > 30 分钟 | < 2 分钟 |
| 预算耗尽时间 | 下午 4-8 点 | 上午 9-11 点 |

**统计检测模型**：

1. **转化率骤降检测**：$\text{CVR}_t < \mu_{\text{CVR}} - 2.5\sigma_{\text{CVR}}$ 且 $\text{Clicks}_t > \mu_{\text{Clicks}} + 2\sigma_{\text{Clicks}}$
2. **预算耗尽时间分布**：正常右偏（下午耗尽），攻击期间左偏（上午即耗尽）
3. **点击速率异常**：Poisson 过程假设下，短时间内点击数超过 Poisson 上界

**防御响应策略**：检测到攻击 → 降低目标关键词出价 → 设置每小时预算分配 → 向 Amazon 申请无效点击退款。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某母婴卖家广告预算 $300/天，竞争对手在 9-11 点集中点击，使预算在 11 点耗尽。黄金时段 12-8pm 完全失去广告曝光，销售损失明显。

**数据要求**：Amazon 广告报告（小时级点击量、花费、转化），过去 60 天历史。

**检测应用**：识别连续 5 天上午 9-11 点 CVR 仅 0.3%（正常 10%），点击集中爆发。向 Amazon 申请无效点击退款 $450，并设置小时预算分配（上午限额 30%）。

**量化产出**：黄金时段广告恢复展示，转化率回升，月 GMV 提升 **8-15 万元**；追讨无效点击退款 **2-5 万元/年**。

## ③ 代码模板

```python
import numpy as np
from scipy import stats

def detect_click_fraud(
    clicks: np.ndarray,      # 每小时点击量（过去 N 天 × 24 小时）
    conversions: np.ndarray, # 对应转化量
    budget_exhaust_hours: list,  # 每天预算耗尽时间（小时，如 [18, 17, 11, 10, 11]）
    cvr_baseline: float = 0.10  # 历史正常 CVR
) -> dict:
    """
    PPC 点击欺诈检测
    """
    # 1. CVR 异常检测
    total_clicks = clicks.sum(axis=1)  # 每天总点击
    total_conv = conversions.sum(axis=1)
    daily_cvr = total_conv / (total_clicks + 1e-8)

    cvr_mean = np.mean(daily_cvr)
    cvr_std = np.std(daily_cvr) + 1e-8
    cvr_z_scores = (daily_cvr - cvr_mean) / cvr_std

    # 2. 预算耗尽时间异常（是否早于正常）
    if budget_exhaust_hours:
        exhaust_mean = np.mean(budget_exhaust_hours)
        early_exhaust_days = sum(1 for h in budget_exhaust_hours if h < 12)
        exhaust_anomaly = early_exhaust_days / len(budget_exhaust_hours)
    else:
        exhaust_mean = 18
        exhaust_anomaly = 0

    # 3. 早上点击集中度（9-11点vs全天比例）
    if clicks.shape[1] >= 12:
        morning_clicks = clicks[:, 9:12].sum(axis=1)
        daily_total = clicks.sum(axis=1)
        morning_ratio = morning_clicks / (daily_total + 1e-8)
        morning_ratio_mean = np.mean(morning_ratio)
    else:
        morning_ratio_mean = 0

    # 4. 综合评分
    attack_signals = {
        'low_cvr': (cvr_mean < cvr_baseline * 0.5),
        'early_budget_exhaust': (exhaust_anomaly > 0.4),
        'morning_click_spike': (morning_ratio_mean > 0.35)
    }
    attack_count = sum(attack_signals.values())

    return {
        'attack_detected': attack_count >= 2,
        'attack_signals': attack_signals,
        'avg_cvr': cvr_mean,
        'cvr_drop_pct': (cvr_baseline - cvr_mean) / cvr_baseline * 100,
        'early_exhaust_rate': exhaust_anomaly,
        'morning_click_ratio': morning_ratio_mean,
        'daily_cvr': daily_cvr,
        'cvr_z_scores': cvr_z_scores
    }

# 测试：模拟 PPC 攻击场景
np.random.seed(42)
n_days = 10
n_hours = 24

# 正常场景（前5天）
clicks_normal = np.random.poisson(15, (5, n_hours))
conv_normal = (clicks_normal * 0.10).astype(int)

# 攻击场景（后5天）：9-11点集中点击，CVR 骤降
clicks_attack = np.random.poisson(8, (5, n_hours))
clicks_attack[:, 9:12] = np.random.poisson(45, (5, 3))  # 早上集中
conv_attack = np.zeros((5, n_hours), dtype=int)
conv_attack[:, 9:12] = np.random.poisson(1, (5, 3))  # CVR 极低

clicks = np.vstack([clicks_normal, clicks_attack])
convs = np.vstack([conv_normal, conv_attack])
exhaust_hours = [18, 17, 19, 18, 17, 10, 11, 10, 11, 10]

result = detect_click_fraud(clicks, convs, exhaust_hours, cvr_baseline=0.10)
print(f"平均 CVR: {result['avg_cvr']:.1%}（正常: 10.0%）")
print(f"CVR 下降: {result['cvr_drop_pct']:.1f}%")
print(f"早晨点击集中度: {result['morning_click_ratio']:.1%}")
print(f"攻击信号: {result['attack_signals']}")
print(f"状态: {'⚠️ 检测到点击欺诈' if result['attack_detected'] else '✅ 正常'}")
assert result['attack_detected'], "应检测到 PPC 攻击"
print("[✓] PPC-Bid-Manipulation-Defense 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-Ad-Fraud-IVT-Detection]]（无效流量检测基础）
> 延伸: [[Skill-Listing-Suppression-Detection]]（Listing 可见性联合监控）
> 可组合: [[Skill-Competitor-Negative-Campaign-Detection]]（竞品攻击综合防御）

## ⑤ 商业价值评估

- **ROI量化**: 月 GMV 提升 8-15 万元 + 追讨无效退款 2-5 万元/年
- **实施难度**: ⭐⭐（广告报告数据直接可用，无需外部数据）
- **优先级**: ⭐⭐⭐⭐（广告预算 > $200/天的卖家必备防御）
