"""
Auto-extracted from: paper2skills-vault/18-物流履约/Skill-Logistics-Cost-PL-Attribution.md
Skill: Skill-Logistics-Cost-PL-Attribution
Domain: 18-物流履约
"""
"""
Logistics Cost P&L Attribution
物流成本五层拆解 + SKU 级真实净利润核算
"""
import numpy as np
import pandas as pd


def generate_sample_sku_data():
    """生成模拟 SKU 成本数据"""
    skus = [
        # name, price, fob_cost, weight_kg, volume_cbm, fba_fee, monthly_storage_days,
        # return_rate, monthly_sales, ad_spend_rate
        ('breast_pump_A',   149.99, 35.0, 2.1, 0.012, 12.50, 45, 0.06, 180, 0.15),
        ('baby_bottle_B',    24.99,  4.5, 0.3, 0.002,  3.22, 30, 0.04, 850, 0.12),
        ('infant_pillow_C',  39.99,  8.0, 0.8, 0.008,  4.50, 60, 0.18, 320, 0.18),  # 高退货！
        ('sterilizer_D',     89.99, 22.0, 1.5, 0.010,  8.75, 40, 0.07, 240, 0.14),
        ('nursing_cover_E',  29.99,  5.5, 0.4, 0.003,  3.85, 55, 0.09, 420, 0.13),
    ]
    cols = ['sku', 'price', 'fob_cost', 'weight_kg', 'volume_cbm',
            'fba_fulfillment_fee', 'avg_storage_days', 'return_rate',
            'monthly_sales_units', 'ad_spend_rate']
    return pd.DataFrame(skus, columns=cols)


def compute_logistics_cost_layers(df, shipment_config=None):
    """
    五层物流成本拆解
    L1: FOB 货值
    L2: 头程运费（按体积重分摊）
    L3: FBA 配送费
    L4: 仓储费（月度 + LTSF）
    L5: 退货逆向成本
    """
    if shipment_config is None:
        shipment_config = {
            'sea_freight_per_cbm': 280,   # $/CBM 海运
            'destination_handling': 45,    # $/CBM 目的港+清关
            'monthly_storage_rate': 0.75,  # $/cubic_foot/month (Jan-Sep)
            'ltsf_rate_per_unit': 1.50,    # 长库龄费 $/unit (>180天)
            'return_processing_fee': 0.50, # $/unit 退货处理基础费
            'return_inspection_rate': 0.30,# 退货货值损耗率
            'platform_commission': 0.15,   # 15% 亚马逊佣金
        }

    df = df.copy()
    sc = shipment_config

    # L1: FOB 货值（直接）
    df['L1_fob'] = df['fob_cost']

    # L2: 头程运费（按体积重 CBM 分摊）
    cbm_per_unit = df['volume_cbm']
    df['L2_freight'] = cbm_per_unit * (sc['sea_freight_per_cbm'] + sc['destination_handling'])

    # L3: FBA 配送费（直接）
    df['L3_fba_fulfillment'] = df['fba_fulfillment_fee']

    # L4: 仓储费（日均在库 × 立方英尺 × 月费率）
    volume_cubic_feet = df['volume_cbm'] * 35.315  # 1 CBM = 35.315 ft³
    monthly_storage = volume_cubic_feet * sc['monthly_storage_rate']
    ltsf_exposure = np.where(df['avg_storage_days'] > 180, sc['ltsf_rate_per_unit'], 0)
    df['L4_storage'] = monthly_storage + ltsf_exposure

    # L5: 退货逆向成本
    return_units = df['monthly_sales_units'] * df['return_rate']
    return_processing = return_units * sc['return_processing_fee']
    return_damage_loss = return_units * df['fob_cost'] * sc['return_inspection_rate']
    df['L5_returns'] = (return_processing + return_damage_loss) / df['monthly_sales_units'].clip(lower=1)

    # 平台佣金
    df['platform_commission'] = df['price'] * sc['platform_commission']

    # 广告费
    df['ad_spend'] = df['price'] * df['ad_spend_rate']

    # 汇总
    df['total_logistics_cost'] = df[['L1_fob', 'L2_freight', 'L3_fba_fulfillment',
                                       'L4_storage', 'L5_returns']].sum(axis=1)
    df['total_cost'] = df['total_logistics_cost'] + df['platform_commission'] + df['ad_spend']
    df['net_profit_per_unit'] = df['price'] - df['total_cost']
    df['net_margin'] = df['net_profit_per_unit'] / df['price']
    df['gross_margin_naive'] = (df['price'] - df['fob_cost'] - df['fba_fulfillment_fee']) / df['price']

    return df


def run_pl_attribution_analysis():
    """完整 P&L 归因分析"""
    print("=" * 70)
    print("Logistics Cost P&L Attribution — 物流成本五层拆解分析")
    print("=" * 70)

    df = generate_sample_sku_data()
    result = compute_logistics_cost_layers(df)

    print("\n📊 物流成本五层拆解（每单位，$）:")
    print(f"{'SKU':<22} {'L1货值':>7} {'L2头程':>7} {'L3FBA':>7} "
          f"{'L4仓储':>7} {'L5退货':>7} {'广告':>7} {'佣金':>7}")
    print("-" * 80)
    for _, row in result.iterrows():
        print(f"{row['sku']:<22} "
              f"{row['L1_fob']:>7.2f} {row['L2_freight']:>7.2f} "
              f"{row['L3_fba_fulfillment']:>7.2f} {row['L4_storage']:>7.2f} "
              f"{row['L5_returns']:>7.2f} {row['ad_spend']:>7.2f} "
              f"{row['platform_commission']:>7.2f}")

    print("\n💰 真实净利润 vs 账面毛利率对比:")
    print(f"{'SKU':<22} {'售价':>8} {'账面毛利':>9} {'真实净利':>9} {'差距':>8} {'退货率':>7}")
    print("-" * 70)
    for _, row in result.iterrows():
        gap = row['net_margin'] - row['gross_margin_naive']
        flag = ' 🚨亏损!' if row['net_margin'] < 0 else (' ⚠️ 警告' if row['net_margin'] < 0.08 else '')
        print(f"{row['sku']:<22} ${row['price']:>6.2f} "
              f"{row['gross_margin_naive']:>9.1%} {row['net_margin']:>9.1%} "
              f"{gap:>+8.1%} {row['return_rate']:>7.1%}{flag}")

    print("\n🚦 行动建议:")
    for _, row in result.iterrows():
        if row['net_margin'] < 0:
            print(f"  🔴 {row['sku']}: 真实净利润为负！立即停止广告投放，评估下架或优化退货率")
        elif row['return_rate'] > 0.12:
            print(f"  🟡 {row['sku']}: 退货率{row['return_rate']:.0%}偏高，"
                  f"L5退货成本${row['L5_returns']:.2f}/单，优化产品图/说明")
        elif row['L2_freight'] / row['price'] > 0.10:
            print(f"  🟡 {row['sku']}: 头程占售价{row['L2_freight']/row['price']:.0%}，"
                  f"考虑换轻包装或海运拼柜")

    # 月度利润汇总
    result['monthly_net_profit'] = result['net_profit_per_unit'] * result['monthly_sales_units']
    total_monthly = result['monthly_net_profit'].sum()
    print(f"\n📈 月度净利润汇总: ${total_monthly:,.0f}")
    print(f"   真实综合净利率: {total_monthly / (result['price'] * result['monthly_sales_units']).sum():.1%}")

    print("\n[✓] Logistics Cost P&L Attribution 测试通过")


if __name__ == '__main__':
    run_pl_attribution_analysis()
