---
title: Skill-Profitability-Waterfall-By-ASIN — 单品盈利瀑布分析
doc_type: knowledge
module: 23-运营财务
topic: profitability-waterfall-by-asin
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Profitability-Waterfall-By-ASIN

## ① 算法原理（≤300字）

单品盈利瀑布（Profitability Waterfall）是将 GMV 逐步扣除各类成本，直到净利润的可视化分析框架。每个"瀑布段"代表一类成本对利润的侵蚀，让卖家一眼看出哪个环节是最大利润泄漏点。

**标准 ASIN 级瀑布层次**：
1. **GMV**（起点）→ 扣除退款/退货 = **Net Revenue**
2. 扣除 Amazon 佣金（Referral Fee）
3. 扣除 FBA 配送费（Fulfillment Fee）
4. 扣除 FBA 仓储费（Storage Fee + LTSF）
5. 扣除商品成本（COGS / Landed Cost）
6. 扣除广告花费（Ad Spend）
7. 扣除头程/物流（Freight）
8. = **净利润（Net Profit）**

**关键分析视角**：
- **纵向**：单 ASIN 各层成本绝对值和占比
- **横向**：跨 ASIN 的利润结构对比，识别优劣品
- **时序**：同一 ASIN 每月瀑布变化，追踪利润漂移

**自动预警规则**：当某层成本占 GMV 超过阈值（如仓储费 > 5%、广告 > 20%），触发告警提示卖家审查。

## ② 母婴出海应用案例

**场景**：母婴品牌有 15 个在售 ASIN，运营团队主观认为主力 SKU（月销最高）即为最赚钱 SKU。

通过单品盈利瀑布分析发现：
- **ASIN-001**（月 GMV 8 万元）：净利率仅 3%，因广告花费占 22% 且仓储费超期
- **ASIN-007**（月 GMV 3 万元）：净利率 28%，广告极低，自然排名稳定
- **ASIN-003**（月 GMV 5 万元）：亏损 SKU，COGS 过高（工厂报价未更新）

**决策**：停止对 ASIN-001 的广告扩张，将预算集中到 ASIN-007 和新品；对 ASIN-003 重新议价或停售。**月净利润从 1.2 万元提升至 4.8 万元**，年化增量利润 **43 万元**。

## ③ 代码模板

```python
import numpy as np
import pandas as pd

# 单品盈利瀑布分析

THRESHOLDS = {
    'referral_pct': 0.15,    # 佣金警戒线 15%
    'fulfillment_pct': 0.12, # 配送费警戒线 12%
    'storage_pct': 0.05,     # 仓储费警戒线 5%
    'ad_pct': 0.20,          # 广告警戒线 20%
    'min_net_margin': 0.05,  # 最低净利润率 5%
}


def compute_asin_waterfall(asin_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个ASIN的盈利瀑布

    asin_data列: asin, gmv, returns, referral_fee, fulfillment_fee,
                 storage_fee, cogs, ad_spend, freight
    """
    df = asin_data.copy()

    df['net_revenue'] = df['gmv'] - df['returns']
    df['after_referral'] = df['net_revenue'] - df['referral_fee']
    df['after_fulfillment'] = df['after_referral'] - df['fulfillment_fee']
    df['after_storage'] = df['after_fulfillment'] - df['storage_fee']
    df['after_cogs'] = df['after_storage'] - df['cogs']
    df['after_ads'] = df['after_cogs'] - df['ad_spend']
    df['net_profit'] = df['after_ads'] - df['freight']
    df['net_margin'] = df['net_profit'] / df['gmv']

    # 各层占GMV比例
    for col in ['referral_fee', 'fulfillment_fee', 'storage_fee', 'cogs', 'ad_spend', 'freight']:
        df[f'{col}_pct'] = df[col] / df['gmv']

    return df


def generate_alerts(waterfall_df: pd.DataFrame) -> pd.DataFrame:
    """生成超阈值预警"""
    alerts = []
    for _, row in waterfall_df.iterrows():
        row_alerts = []
        if row.get('referral_fee_pct', 0) > THRESHOLDS['referral_pct']:
            row_alerts.append(f"⚠️ 佣金率 {row['referral_fee_pct']:.1%} > {THRESHOLDS['referral_pct']:.0%}")
        if row.get('fulfillment_fee_pct', 0) > THRESHOLDS['fulfillment_pct']:
            row_alerts.append(f"⚠️ 配送费率 {row['fulfillment_fee_pct']:.1%} > 12%")
        if row.get('storage_fee_pct', 0) > THRESHOLDS['storage_pct']:
            row_alerts.append(f"🔴 仓储费率 {row['storage_fee_pct']:.1%} > 5%（积压！）")
        if row.get('ad_spend_pct', 0) > THRESHOLDS['ad_pct']:
            row_alerts.append(f"⚠️ 广告占比 {row['ad_spend_pct']:.1%} > 20%")
        if row.get('net_margin', 0) < THRESHOLDS['min_net_margin']:
            row_alerts.append(f"🔴 净利率 {row['net_margin']:.1%} < 5%（危险！）")

        alerts.append({'asin': row['asin'], '预警': ' | '.join(row_alerts) if row_alerts else '✅ 正常'})

    return pd.DataFrame(alerts)


def rank_asins(waterfall_df: pd.DataFrame) -> pd.DataFrame:
    """按净利润排名，区分现金牛 vs 问题品"""
    df = waterfall_df[['asin', 'gmv', 'net_profit', 'net_margin']].copy()
    df = df.sort_values('net_profit', ascending=False)
    df['排名'] = range(1, len(df) + 1)
    df['类型'] = df['net_margin'].apply(
        lambda x: '💰 现金牛' if x >= 0.15 else ('⚠️ 关注' if x >= 0.05 else '🔴 亏损品')
    )
    return df


# ── 测试 ──
if __name__ == '__main__':
    np.random.seed(42)
    n = 10
    asin_data = pd.DataFrame({
        'asin': [f'ASIN-{i:03d}' for i in range(1, n+1)],
        'gmv': np.random.uniform(20000, 80000, n),
        'returns': np.random.uniform(500, 3000, n),
        'referral_fee': np.random.uniform(2000, 8000, n),
        'fulfillment_fee': np.random.uniform(1500, 6000, n),
        'storage_fee': np.random.uniform(200, 3000, n),
        'cogs': np.random.uniform(8000, 35000, n),
        'ad_spend': np.random.uniform(1000, 12000, n),
        'freight': np.random.uniform(300, 2000, n),
    })

    wf = compute_asin_waterfall(asin_data)

    print("=== ASIN盈利瀑布（前5行）===")
    print(wf[['asin', 'gmv', 'net_revenue', 'after_cogs', 'net_profit', 'net_margin']].head().to_string(index=False))

    print("\n=== ASIN排名（现金牛 vs 问题品）===")
    print(rank_asins(wf).to_string(index=False))

    print("\n=== 预警报告 ===")
    alerts = generate_alerts(wf)
    print(alerts.to_string(index=False))
    print(f"\n[✓] 单品盈利瀑布分析测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-FBA-Cost-Forecast-Adjustment]]
- 前置技能：[[Skill-PL-Attribution-Analysis]]
- 延伸技能：[[Skill-SKU-Level-PL-Dashboard]]
- 延伸技能：[[Skill-FBA-Fee-Waterfall-Attribution]]
- 可组合：[[Skill-Advertising-TACOS-PnL-Integration]]
- 可组合：[[Skill-Logistics-Cost-PL-Attribution]]

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| ROI | 识别并止损亏损 ASIN，聚焦现金牛，年化利润提升 30-100 万元 |
| 实施难度 | ⭐⭐（Amazon 账单 + 成本台账即可） |
| 优先级 | ⭐⭐⭐⭐⭐（SKU 超过 10 个时立即启用） |
| 数据要求 | Amazon 账单 + 商品成本 + 广告花费（ASIN 级） |
| 典型收益 | 发现 20% 的 ASIN 贡献负利润，止损后利润率提升 5-15 个点 |
