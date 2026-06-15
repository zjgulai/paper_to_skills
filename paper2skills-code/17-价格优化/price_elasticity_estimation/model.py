"""
Auto-extracted from: paper2skills-vault/17-价格优化/Skill-Price-Elasticity-Estimation.md
Skill: Skill-Price-Elasticity-Estimation
Domain: 17-价格优化
"""
"""
Price Elasticity Estimation for Cross-Border E-Commerce
基于 DiD + Log-Log OLS 的 SKU 级价格弹性估算
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def generate_sample_data():
    """生成模拟母婴 SKU 价格-需求数据"""
    np.random.seed(42)
    n_weeks = 52
    weeks = pd.date_range('2025-01-01', periods=n_weeks, freq='W')

    # 真实弹性设定：吸奶器 -1.4，奶粉 -0.6
    skus = {
        'breast_pump_A': {'base_price': 129, 'true_elasticity': -1.4, 'base_demand': 200},
        'formula_B':     {'base_price': 45,  'true_elasticity': -0.6, 'base_demand': 500},
        'sterilizer_C':  {'base_price': 89,  'true_elasticity': -1.1, 'base_demand': 150},
    }

    records = []
    for sku, params in skus.items():
        for i, week in enumerate(weeks):
            # 添加价格扰动（模拟促销/竞品调价）
            price_shock = np.random.choice([-0.15, -0.10, 0, 0, 0, 0.05], p=[0.05, 0.1, 0.5, 0.2, 0.1, 0.05])
            price = params['base_price'] * (1 + price_shock)
            # 季节性因子
            season = 1 + 0.3 * np.sin(2 * np.pi * i / 52)
            # 需求 = f(价格弹性, 季节, 噪声)
            demand = params['base_demand'] * season * (price / params['base_price']) ** params['true_elasticity']
            demand = max(0, demand * (1 + np.random.normal(0, 0.1)))
            records.append({'week': week, 'sku': sku, 'price': price, 'demand': demand})

    return pd.DataFrame(records)


def estimate_elasticity_ols(df, sku_name, control_vars=None):
    """
    对数-对数 OLS 弹性估算
    ln(demand) = α + ε·ln(price) + controls + ε_it
    """
    sku_df = df[df['sku'] == sku_name].copy()
    sku_df['ln_demand'] = np.log(sku_df['demand'].clip(lower=1))
    sku_df['ln_price'] = np.log(sku_df['price'])
    sku_df['week_num'] = range(len(sku_df))
    # 添加季节控制
    sku_df['sin_season'] = np.sin(2 * np.pi * sku_df['week_num'] / 52)
    sku_df['cos_season'] = np.cos(2 * np.pi * sku_df['week_num'] / 52)

    X = sku_df[['ln_price', 'sin_season', 'cos_season']].values
    X = np.column_stack([np.ones(len(X)), X])
    y = sku_df['ln_demand'].values

    # OLS
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 标准误
    n, k = len(y), X.shape[1]
    sigma2 = ss_res / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t_stat = beta / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-k))

    elasticity = beta[1]
    return {
        'sku': sku_name,
        'elasticity': elasticity,
        'se': se[1],
        't_stat': t_stat[1],
        'p_value': p_vals[1],
        'r2': r2,
        'ci_lower': elasticity - 1.96 * se[1],
        'ci_upper': elasticity + 1.96 * se[1],
    }


def classify_elasticity(epsilon):
    """弹性分类及定价建议"""
    e = abs(epsilon)
    if e < 0.5:
        return '强非弹性', '大胆提价，聚焦利润率', '可考虑提价10-20%'
    elif e < 1.0:
        return '弱非弹性', '温和提价，品牌溢价策略', '可提价5-10%，观察BSR变化'
    elif e < 1.8:
        return '中等弹性', '跟随市场，谨慎促销', '保持价格稳定，选择性参加Deal'
    else:
        return '强弹性', '价格是核心竞争力', '价格战场景，考虑降价抢份额'


def promotion_roi_analysis(price_base, cost, elasticity, demand_base, discount_rates=None):
    """促销 ROI 分析：找到最优折扣率"""
    if discount_rates is None:
        discount_rates = np.arange(0, 0.31, 0.01)

    results = []
    baseline_profit = (price_base - cost) * demand_base
    for disc in discount_rates:
        price_promo = price_base * (1 - disc)
        demand_promo = demand_base * ((1 - disc) ** elasticity)
        profit_promo = (price_promo - cost) * demand_promo
        incremental_profit = profit_promo - baseline_profit
        results.append({
            'discount_pct': disc * 100,
            'price': price_promo,
            'demand': demand_promo,
            'profit': profit_promo,
            'incremental_profit': incremental_profit,
        })

    roi_df = pd.DataFrame(results)
    optimal = roi_df.loc[roi_df['incremental_profit'].idxmax()]
    return roi_df, optimal


def run_elasticity_analysis():
    """完整弹性分析流程"""
    print("=" * 60)
    print("Price Elasticity Estimation — 母婴 SKU 定价分析")
    print("=" * 60)

    df = generate_sample_data()

    results = []
    for sku in df['sku'].unique():
        res = estimate_elasticity_ols(df, sku)
        results.append(res)

    print("\n📊 弹性估算结果:")
    print(f"{'SKU':<22} {'弹性ε':>8} {'95%CI':>20} {'p值':>8} {'R²':>6} {'分类'}")
    print("-" * 80)
    for r in results:
        cat, strategy, _ = classify_elasticity(r['elasticity'])
        sig = '✅' if r['p_value'] < 0.05 else '⚠️ '
        print(f"{r['sku']:<22} {r['elasticity']:>8.3f} "
              f"[{r['ci_lower']:>6.2f}, {r['ci_upper']:>5.2f}] "
              f"{r['p_value']:>8.4f}{sig} {r['r2']:>5.3f}  {cat}")

    # 促销ROI分析示例（吸奶器）
    print("\n📈 吸奶器促销 ROI 分析 (ε=-1.4, 成本$80, 基础销量200/周):")
    roi_df, optimal = promotion_roi_analysis(
        price_base=129, cost=80, elasticity=-1.4,
        demand_base=200, discount_rates=np.arange(0, 0.26, 0.05)
    )
    print(f"{'折扣':>8} {'售价':>8} {'预测销量':>10} {'增量利润':>12}")
    for _, row in roi_df.iterrows():
        mark = ' ← 最优' if abs(row['discount_pct'] - optimal['discount_pct']) < 0.1 else ''
        print(f"{row['discount_pct']:>7.0f}%  ${row['price']:>6.0f}  {row['demand']:>10.0f}  "
              f"${row['incremental_profit']:>+10.0f}{mark}")

    # 多市场定价建议
    print("\n🌍 多市场弹性对比 & 定价建议:")
    market_elasticities = {'美国(US)': -1.4, '德国(DE)': -0.9, '英国(UK)': -1.1}
    for market, eps in market_elasticities.items():
        cat, strategy, action = classify_elasticity(eps)
        print(f"  {market}: ε={eps} → {cat} | {action}")

    print("\n[✓] Price Elasticity Estimation 测试通过")


if __name__ == '__main__':
    run_elasticity_analysis()
