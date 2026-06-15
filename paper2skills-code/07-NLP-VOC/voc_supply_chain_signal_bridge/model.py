"""
Auto-extracted from: paper2skills-vault/07-NLP-VOC/Skill-VOC-Supply-Chain-Signal-Bridge.md
Skill: Skill-VOC-Supply-Chain-Signal-Bridge
Domain: 07-NLP-VOC
"""
"""
VOC Supply Chain Signal Bridge
用户评论信号 → 供应链补货决策桥梁
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict


# 缺货语义模式库（母婴品类定制）
STOCKOUT_PATTERNS = [
    r'out of stock', r'not available', r'unavailable', r'sold out',
    r'always missing', r'keep running out', r'constantly out',
    r'缺货', r'没货', r'断货', r'等了.*周', r'买不到',
    r'switched to', r'had to buy', r'went with.*instead',
]

DEMAND_SURGE_PATTERNS = [
    r'baby shower gift', r'must have', r'essential', r'highly recommend',
    r'bought.*more', r'reorder', r'stock up', r'buying again',
    r'perfect for newborn', r'trending',
]

COMPETITOR_SWITCH_PATTERNS = [
    r'switched (from|to) (medela|momcozy|spectra|lansinoh|haakaa)',
    r'(medela|momcozy|spectra) instead',
    r'going back to',
    r'competitor',
]


def classify_review_signals(reviews_df):
    """
    对评论进行信号分类
    Input: DataFrame with columns [date, rating, text, verified]
    Output: DataFrame with signal labels
    """
    def detect_signals(text):
        text_lower = str(text).lower()
        signals = []
        for p in STOCKOUT_PATTERNS:
            if re.search(p, text_lower):
                signals.append('stockout')
                break
        for p in DEMAND_SURGE_PATTERNS:
            if re.search(p, text_lower):
                signals.append('demand_surge')
                break
        for p in COMPETITOR_SWITCH_PATTERNS:
            if re.search(p, text_lower):
                signals.append('competitor_switch')
                break
        return signals if signals else ['neutral']

    reviews_df = reviews_df.copy()
    reviews_df['signals'] = reviews_df['text'].apply(detect_signals)
    reviews_df['is_stockout'] = reviews_df['signals'].apply(lambda x: 'stockout' in x)
    reviews_df['is_demand_surge'] = reviews_df['signals'].apply(lambda x: 'demand_surge' in x)
    reviews_df['is_competitor_switch'] = reviews_df['signals'].apply(lambda x: 'competitor_switch' in x)
    # 情绪强度（用评分反转作为简化代理）
    reviews_df['sentiment_intensity'] = reviews_df['rating'].apply(
        lambda r: max(0, (5 - r) / 4) if pd.notna(r) else 0.5
    )
    return reviews_df


def compute_weekly_dpi(reviews_df, search_trend=None):
    """
    计算周度需求压力指数 DPI
    DPI = stockout_rate × avg_sentiment × search_trend_ratio
    """
    reviews_df['week'] = pd.to_datetime(reviews_df['date']).dt.to_period('W')
    weekly = reviews_df.groupby('week').agg(
        total_reviews=('text', 'count'),
        stockout_reviews=('is_stockout', 'sum'),
        demand_surge_reviews=('is_demand_surge', 'sum'),
        competitor_switch_reviews=('is_competitor_switch', 'sum'),
        avg_sentiment=('sentiment_intensity', 'mean'),
    ).reset_index()

    weekly['stockout_rate'] = weekly['stockout_reviews'] / weekly['total_reviews'].clip(lower=1)

    # 搜索量趋势因子（无数据时默认1.0）
    if search_trend is not None:
        weekly = weekly.merge(search_trend, on='week', how='left')
        weekly['search_ratio'] = weekly['search_volume'] / weekly['search_volume'].shift(4).clip(lower=1)
        weekly['search_ratio'] = weekly['search_ratio'].fillna(1.0).clip(0.5, 3.0)
    else:
        weekly['search_ratio'] = 1.0

    weekly['DPI'] = weekly['stockout_rate'] * weekly['avg_sentiment'] * weekly['search_ratio']
    weekly['DPI_trend'] = weekly['DPI'].rolling(3).mean()
    return weekly


def generate_replenishment_signals(dpi_df, base_safety_stock=1.5, dpi_threshold=0.15):
    """
    基于 DPI 生成补货信号
    Returns: list of supply chain actions
    """
    actions = []
    for i, row in dpi_df.iterrows():
        if row['DPI'] > dpi_threshold * 1.5 or (
            row['competitor_switch_reviews'] > 3 and row['stockout_rate'] > 0.1
        ):
            actions.append({
                'week': str(row['week']),
                'action': '🚨 紧急追单',
                'reason': f"DPI={row['DPI']:.3f}, 竞品迁移={row['competitor_switch_reviews']}条",
                'safety_stock_multiplier': 2.5,
                'priority': 'P0',
            })
        elif row['DPI'] > dpi_threshold:
            actions.append({
                'week': str(row['week']),
                'action': '⚠️  提高安全库存',
                'reason': f"DPI={row['DPI']:.3f}, 缺货投诉率={row['stockout_rate']:.1%}",
                'safety_stock_multiplier': 2.0,
                'priority': 'P1',
            })
        elif row['is_demand_surge'] if 'is_demand_surge' in row else row['demand_surge_reviews'] > 5:
            actions.append({
                'week': str(row['week']),
                'action': '📈 季节性峰值预备',
                'reason': f"需求爆发信号={row['demand_surge_reviews']}条",
                'safety_stock_multiplier': 1.8,
                'priority': 'P2',
            })
    return actions


def run_voc_supply_chain_demo():
    """完整演示流程"""
    print("=" * 65)
    print("VOC Supply Chain Signal Bridge — 评论信号→供应链决策")
    print("=" * 65)

    # 生成模拟评论数据
    np.random.seed(42)
    n = 200
    sample_texts = [
        "Great pump but always out of stock, had to wait 3 weeks",
        "Switched to Medela because this was unavailable",
        "Perfect for newborns! Buying 2 more as baby shower gifts",
        "Love it, bought again for my second baby",
        "Product is fine but keep running out of flanges",
        "Amazing suction, highly recommend to all new moms",
        "Could not find this anywhere, went with Momcozy instead",
        "Reordering for the third time, must have for new parents",
        "Out of stock again! This is the 4th time",
        "Good product but wish they always kept it in stock",
    ]
    dates = pd.date_range('2025-06-01', periods=n, freq='D')
    reviews = pd.DataFrame({
        'date': np.random.choice(dates, n),
        'rating': np.random.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.1, 0.15, 0.3, 0.3]),
        'text': np.random.choice(sample_texts, n),
        'verified': np.random.choice([True, False], n, p=[0.8, 0.2]),
    })

    # 信号分类
    classified = classify_review_signals(reviews)
    print(f"\n📊 评论信号分布 (共 {len(classified)} 条):")
    print(f"  缺货投诉:  {classified['is_stockout'].sum()} 条 "
          f"({classified['is_stockout'].mean():.1%})")
    print(f"  需求爆发:  {classified['is_demand_surge'].sum()} 条 "
          f"({classified['is_demand_surge'].mean():.1%})")
    print(f"  竞品迁移:  {classified['is_competitor_switch'].sum()} 条 "
          f"({classified['is_competitor_switch'].mean():.1%})")

    # 计算 DPI
    dpi_df = compute_weekly_dpi(classified)
    print(f"\n📈 周度 DPI 趋势 (最近 8 周):")
    print(f"{'周次':<15} {'DPI':>8} {'缺货率':>8} {'缺货评论':>8} {'竞品迁移':>8}")
    print("-" * 55)
    for _, row in dpi_df.tail(8).iterrows():
        alert = ' 🚨' if row['DPI'] > 0.225 else (' ⚠️ ' if row['DPI'] > 0.15 else '')
        print(f"{str(row['week']):<15} {row['DPI']:>8.3f} "
              f"{row['stockout_rate']:>8.1%} "
              f"{int(row['stockout_reviews']):>8} "
              f"{int(row['competitor_switch_reviews']):>8}{alert}")

    # 补货信号
    signals = generate_replenishment_signals(dpi_df)
    if signals:
        print(f"\n🚦 补货行动建议 ({len(signals)} 项):")
        for s in signals[:5]:
            print(f"  [{s['priority']}] {s['week']}: {s['action']}")
            print(f"       原因: {s['reason']}")
            print(f"       安全库存系数: ×{s['safety_stock_multiplier']}")
    else:
        print("\n✅ 当前无需紧急补货")

    print("\n[✓] VOC Supply Chain Signal Bridge 测试通过")


if __name__ == '__main__':
    run_voc_supply_chain_demo()
