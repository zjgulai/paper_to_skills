---
title: Listing Suppression Detection — Listing 被平台隐藏/降权检测（非账号问题）
doc_type: knowledge
module: 19-风控反欺诈
topic: listing-suppression-detection
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Listing-Suppression-Detection

## ① 算法原理（≤300字）

**核心问题**：Amazon Listing 被隐藏/降权（Suppressed）有时不触发账号警告，而是静默失效——搜索流量骤降、转化率正常但曝光消失。卖家往往 2-3 天后才发现，每天损失销售额数千美元。

**检测信号体系**：

三类核心信号的异常组合可识别 Listing 被压制：

1. **Session 骤降**（Seller Central 数据）：正常日 Session 下降 > 40%，且非广告问题（ACoS 正常）
2. **自然流量与广告流量解耦**：Session 总量下降但 PPC Session 正常 → 自然搜索被压制
3. **BSR 不变但 Session 归零**：排名稳定但搜索流量消失 → 关键词层面被隐藏

**统计检测**：用 CUSUM（累积和控制图）对 Organic Session 序列做实时异常检测：
$$C_t = \max(0, C_{t-1} + (x_t - \mu_0 - k))$$
当 $C_t > h$（控制限）时触发告警。$k$ 为允许漂移量，$h$ 为检测敏感度。

**根因分类器**：Session 骤降后，自动检查：图片合规性、标题违禁词、Listing Quality 分数、Flat file 错误标记，快速定位抑制原因。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某吸奶器 Listing 因图片中含有「医疗器械」相关词汇被 Amazon 算法静默压制，Organic Session 在 2 天内归零，但 BSR 和广告 Session 暂时未变。

**数据要求**：每日 Organic Session、PPC Session、BSR、Page Views（Seller Central Business Report）。

**CUSUM 检测**：Session 异常在第 1.5 天触发预警（传统阈值法需要 3 天），运营立即排查发现图片合规问题，修复图片并重新提交后 24 小时恢复。

**量化产出**：检测提前 1.5 天，避免缺货式流量损失，日 GMV 损失从 3 天压缩至 1 天，年化减少 Listing 压制损失约 **15-30 万元**。

## ③ 代码模板

```python
import numpy as np

def cusum_listing_monitor(
    organic_sessions: np.ndarray,
    ppc_sessions: np.ndarray,
    k: float = 0.5,  # 允许漂移量（标准差单位）
    h: float = 4.0,  # 控制限
    warmup: int = 14  # 历史基线期（天）
) -> dict:
    """
    Listing 压制 CUSUM 检测器
    organic_sessions: 自然 Session 序列
    ppc_sessions: PPC Session 序列
    """
    n = len(organic_sessions)
    assert n > warmup, "数据不足"

    # 基线统计（warmup 期）
    mu_0 = np.mean(organic_sessions[:warmup])
    sigma_0 = np.std(organic_sessions[:warmup]) + 1e-8

    # 标准化
    x_norm = (organic_sessions - mu_0) / sigma_0

    # 下行 CUSUM（检测急剧下降）
    C_neg = np.zeros(n)
    alerts = np.zeros(n, dtype=bool)

    for t in range(1, n):
        C_neg[t] = max(0, C_neg[t - 1] - x_norm[t] - k)
        alerts[t] = C_neg[t] > h

    # 流量解耦检测：Organic 骤降但 PPC 正常
    organic_pct_change = np.zeros(n)
    ppc_pct_change = np.zeros(n)
    for t in range(1, n):
        if organic_sessions[t - 1] > 0:
            organic_pct_change[t] = (organic_sessions[t] - organic_sessions[t - 1]) / organic_sessions[t - 1]
        if ppc_sessions[t - 1] > 0:
            ppc_pct_change[t] = (ppc_sessions[t] - ppc_sessions[t - 1]) / ppc_sessions[t - 1]

    # 解耦信号：Organic 下降 > 30% 且 PPC 变化 < 15%
    decoupling = (organic_pct_change < -0.30) & (np.abs(ppc_pct_change) < 0.15)

    alert_days = np.where(alerts)[0].tolist()
    return {
        'cusum_values': C_neg,
        'alert_days': alert_days,
        'decoupling_days': np.where(decoupling)[0].tolist(),
        'suppressed': len(alert_days) > 0,
        'baseline_daily_sessions': mu_0
    }

# 测试：模拟 Listing 压制场景
np.random.seed(42)
n = 30
organic = np.random.poisson(500, n).astype(float)
ppc = np.random.poisson(200, n).astype(float)

# 第 17 天起 Organic Session 归零（Listing 被压制）
organic[17:] = np.random.poisson(20, n - 17)  # 骤降至 4%
# PPC 暂时正常
ppc[17:20] = np.random.poisson(190, 3)

result = cusum_listing_monitor(organic, ppc)
assert result['suppressed'], "应检测到 Listing 压制"
assert any(d >= 17 for d in result['alert_days']), "告警应在压制发生后触发"
assert len(result['decoupling_days']) > 0

print(f"基线日 Session: {result['baseline_daily_sessions']:.0f}")
print(f"CUSUM 告警天数: {result['alert_days']}")
print(f"流量解耦天数: {result['decoupling_days']}")
print(f"Listing 状态: {'⚠️ 被压制' if result['suppressed'] else '✅ 正常'}")
print("[✓] Listing-Suppression-Detection 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-Account-Health-Early-Warning-System]]（账号健康监控）
> 延伸: [[Skill-Competitor-Negative-Campaign-Detection]]（竞品恶意投诉检测）
> 可组合: [[Skill-Seller-Rating-Attack-Pattern]]（多维 Listing 攻击溯源）

## ⑤ 商业价值评估

- **ROI量化**: 检测提前 1.5 天，年化减少 Listing 压制损失 15-30 万元
- **实施难度**: ⭐⭐（Seller Central API 数据获取，算法简单）
- **优先级**: ⭐⭐⭐⭐⭐（每个卖家 Listing 监控必备）
