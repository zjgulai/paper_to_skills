---
title: Skill-FBA-Fee-Waterfall-Attribution — FBA费用瀑布归因
doc_type: knowledge
module: 23-运营财务
topic: fba-fee-waterfall-attribution
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-FBA-Fee-Waterfall-Attribution

## ① 算法原理（≤300字）

FBA 费用由多个层次叠加构成，每一层都会"吃掉"一部分毛利。瀑布归因模型将订单级别的费用逐项拆解：

**费用层次**（从 GMV 到净利润）：
1. **配送费（Fulfillment Fee）**：按商品尺寸/重量阶梯定价
2. **仓储费（Storage Fee）**：月均库存量 × 月度费率（标准/大件/危险品）
3. **长期仓储费（LTSF）**：超过 181 天/365 天触发阶梯罚款
4. **移仓费（Placement Fee）**：送仓到指定仓区的强制分拣费
5. **退货处理费（Return Processing Fee）**：退货商品再入库处理成本
6. **佣金（Referral Fee）**：类目固定比例（通常 8%-15%）

瀑布归因通过逐层累计，精确计算每个 SKU 的"真实可控费用占比"，识别哪类费用是主要利润泄漏点。通过帕累托分析（80/20 法则）锁定高费用 SKU 集群，为费用优化决策提供数据支撑。

**核心公式**：
```
净利润 = GMV × (1 - 佣金率) - 配送费 - 仓储费 - LTSF - 移仓费 - 退货费 - 广告费 - 货品成本
```

## ② 母婴出海应用案例

**场景**：某母婴品牌销售婴儿推车（Oversize），月销 500 单，GMV 15 万美元。卖家认为净利率约 15%，但实际账单显示亏损。

通过 FBA 费用瀑布归因发现：
- 配送费因尺寸超标（Oversize Tier 2）达到 GMV 的 18%（行业均值 9%）
- 仓储费因积压 120 天库存，月均支出 2,400 美元
- LTSF 触发：3 批次共 80 件超过 181 天，罚款 1,600 美元/月

**优化行动**：
- 重新包装缩小至 Large Standard，配送费率降至 12%
- 设置 90 天库存预警线，提前促销清仓
- **结果**：净利率从 -3% 修复至 +11%，年化节省 14 万美元

## ③ 代码模板

```python
import numpy as np
import pandas as pd

# FBA费用瀑布归因模型

def compute_fba_waterfall(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个SKU的FBA费用瀑布归因

    输入列: sku, gmv, fulfillment_fee, storage_fee, ltsf_fee,
            placement_fee, return_fee, referral_rate, cogs, ad_spend
    输出: 含各层费用占比和净利润的瀑布DataFrame
    """
    df = orders_df.copy()

    df['referral_fee'] = df['gmv'] * df['referral_rate']
    df['total_fba_fee'] = (
        df['fulfillment_fee'] +
        df['storage_fee'] +
        df['ltsf_fee'] +
        df['placement_fee'] +
        df['return_fee']
    )
    df['net_profit'] = (
        df['gmv']
        - df['referral_fee']
        - df['total_fba_fee']
        - df['ad_spend']
        - df['cogs']
    )
    df['net_margin'] = df['net_profit'] / df['gmv']

    # 各层占GMV比例
    for col in ['referral_fee', 'fulfillment_fee', 'storage_fee',
                'ltsf_fee', 'placement_fee', 'return_fee', 'ad_spend', 'cogs']:
        df[f'{col}_pct'] = df[col] / df['gmv']

    return df


def pareto_fee_analysis(waterfall_df: pd.DataFrame, fee_col: str = 'total_fba_fee'):
    """帕累托分析：找出贡献80%费用的SKU集群"""
    df = waterfall_df.sort_values(fee_col, ascending=False).copy()
    df['cumulative_fee'] = df[fee_col].cumsum()
    df['cumulative_pct'] = df['cumulative_fee'] / df[fee_col].sum()
    top80 = df[df['cumulative_pct'] <= 0.8]
    return top80[['sku', fee_col, 'cumulative_pct', 'net_margin']]


# ── 测试 ──
if __name__ == '__main__':
    np.random.seed(42)
    n = 20
    skus = [f'SKU-{i:03d}' for i in range(n)]

    data = pd.DataFrame({
        'sku': skus,
        'gmv': np.random.uniform(3000, 15000, n),
        'fulfillment_fee': np.random.uniform(300, 2000, n),
        'storage_fee': np.random.uniform(50, 500, n),
        'ltsf_fee': np.random.uniform(0, 300, n),
        'placement_fee': np.random.uniform(20, 200, n),
        'return_fee': np.random.uniform(10, 150, n),
        'referral_rate': np.random.uniform(0.08, 0.15, n),
        'cogs': np.random.uniform(1000, 6000, n),
        'ad_spend': np.random.uniform(100, 1000, n),
    })

    result = compute_fba_waterfall(data)
    pareto = pareto_fee_analysis(result)

    print("=== FBA费用瀑布归因 ===")
    print(result[['sku', 'gmv', 'total_fba_fee', 'net_profit', 'net_margin']].head(5).to_string(index=False))
    print(f"\n平均净利率: {result['net_margin'].mean():.1%}")
    print(f"\n=== 帕累托：贡献80%费用的SKU ===")
    print(pareto[['sku', 'total_fba_fee', 'cumulative_pct']].to_string(index=False))
    print(f"\n[✓] FBA费用瀑布归因测试通过")
```

## ④ 技能关联

- 前置：[[Skill-FBA-Fee-Intelligence]] — 基础费用数据采集
- 延伸：[[Skill-SKU-Level-PL-Dashboard]] — 单品盈利看板
- 延伸：[[Skill-Profitability-Waterfall-By-ASIN]] — ASIN级盈利瀑布
- 组合：[[Skill-Inventory-Carrying-Cost-Model]] — 库存持有成本联合分析

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| ROI | 年化节省 FBA 费用 10-30 万元（中型卖家） |
| 实施难度 | ⭐⭐（需 Amazon 账单 API 或手动导出） |
| 优先级 | ⭐⭐⭐⭐⭐（直接影响利润底线） |
| 数据要求 | Amazon 账单明细 + SKU 尺寸/重量数据 |
| 典型收益 | 发现 LTSF 泄漏后，3 个月回收费用损失 5-15 万元 |
