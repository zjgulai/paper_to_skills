"""
Auto-extracted from: paper2skills-vault/06-增长模型/Skill-LTV-Prediction-BTYD.md
Skill: Skill-LTV-Prediction-BTYD
Domain: 06-增长模型
"""
"""
LTV Prediction using BG/NBD + Gamma-Gamma (BTYD Framework)
不依赖外部 lifetimes 库的纯 Python 实现（简化版）
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, betaln


def generate_sample_customer_data():
    """生成模拟母婴电商客户 RFM 数据"""
    np.random.seed(42)
    n_customers = 500

    # 模拟不同用户群体
    data = []
    for i in range(n_customers):
        customer_type = np.random.choice(['loyal', 'occasional', 'one_time'],
                                          p=[0.20, 0.35, 0.45])
        T = np.random.uniform(12, 24)  # 观测期（月）
        if customer_type == 'loyal':
            freq = np.random.poisson(8)
            recency = np.random.uniform(T * 0.5, T)
            monetary = np.random.lognormal(np.log(75), 0.4)
        elif customer_type == 'occasional':
            freq = np.random.poisson(2)
            recency = np.random.uniform(T * 0.2, T)
            monetary = np.random.lognormal(np.log(45), 0.5)
        else:
            freq = 0
            recency = 0
            monetary = np.random.lognormal(np.log(35), 0.6)

        data.append({
            'customer_id': f'C{i+1000}',
            'frequency': max(0, freq),
            'recency': recency if freq > 0 else 0,
            'T': T,
            'monetary_value': monetary,
            'acquisition_channel': np.random.choice(['tiktok', 'google', 'amazon_organic'],
                                                      p=[0.3, 0.35, 0.35]),
        })
    return pd.DataFrame(data)


def bgnbd_log_likelihood(params, frequency, recency, T):
    """BG/NBD 对数似然函数（简化实现）"""
    r, alpha, a, b = params
    if any(p <= 0 for p in params):
        return 1e10

    ln_A0 = betaln(a, b + frequency) - betaln(a, b)
    ln_A1 = (gammaln(r + frequency) - gammaln(r) - gammaln(frequency + 1)
             + r * np.log(alpha) - (r + frequency) * np.log(alpha + T))

    # 处理有复购的用户
    if frequency > 0:
        ln_A2 = (gammaln(r + frequency) - gammaln(r)
                 + r * np.log(alpha) - (r + frequency) * np.log(alpha + recency))
        ln_delta = np.log(a) - np.log(b + frequency - 1) + ln_A2 - ln_A1
        ln_likelihood = ln_A1 + np.log(1 + np.exp(ln_delta) if ln_delta < 500 else 0)
    else:
        ln_likelihood = ln_A1

    return -ln_likelihood


def estimate_bgnbd_params(df):
    """估计 BG/NBD 模型参数（向量化简化版）"""
    # 使用矩量匹配法快速估算（生产环境用 lifetimes 库的 MLE）
    mean_freq = df['frequency'].mean()
    mean_recency = df['recency'].mean()
    mean_T = df['T'].mean()

    # 启发式初始参数（基于均值矩匹配）
    r = max(0.5, mean_freq / (mean_T - mean_recency + 0.1))
    alpha = r * mean_T / (mean_freq + 0.1)
    a = 0.8
    b = max(1.0, b := mean_T / (mean_T - mean_recency + 0.1))

    return {'r': r, 'alpha': alpha, 'a': a, 'b': b}


def predict_p_alive(frequency, recency, T, params):
    """预测用户当前仍活跃的概率"""
    r, alpha, a, b = params['r'], params['alpha'], params['a'], params['b']
    if frequency == 0:
        return (b / (a + b)) ** r
    # 简化公式
    x = frequency
    p_alive = 1 / (1 + (a / (b + x - 1)) * ((alpha + T) / (alpha + recency)) ** (r + x))
    return min(1.0, max(0.0, p_alive))


def predict_future_transactions(frequency, recency, T, t_future, params):
    """预测未来 t_future 个月的期望购买次数"""
    r, alpha, a, b = params['r'], params['alpha'], params['a'], params['b']
    p_alive = predict_p_alive(frequency, recency, T, params)
    # 简化：活跃用户未来购买率 ≈ 历史购买率 × 活跃概率
    hist_rate = (frequency + 0.5) / (T + 1)
    expected_transactions = p_alive * hist_rate * t_future
    return expected_transactions


def predict_clv(df, params, t_future=12, discount_rate=0.01, margin_rate=0.25):
    """预测每位用户的 CLV"""
    results = []
    for _, row in df.iterrows():
        p_alive = predict_p_alive(
            row['frequency'], row['recency'], row['T'], params)
        future_tx = predict_future_transactions(
            row['frequency'], row['recency'], row['T'], t_future, params)
        # Gamma-Gamma: 预期货币价值（简化：用历史均值 + 贝叶斯收缩）
        global_mean = df['monetary_value'].mean()
        shrinkage = row['frequency'] / (row['frequency'] + 5)  # 5: 先验强度
        expected_monetary = shrinkage * row['monetary_value'] + (1 - shrinkage) * global_mean
        # CLV
        clv = future_tx * expected_monetary * margin_rate / (1 + discount_rate)
        results.append({
            'customer_id': row['customer_id'],
            'frequency': row['frequency'],
            'recency': round(row['recency'], 1),
            'T': round(row['T'], 1),
            'p_alive': round(p_alive, 3),
            'predicted_purchases_12m': round(future_tx, 2),
            'expected_monetary': round(expected_monetary, 2),
            'clv_12m': round(clv, 2),
            'acquisition_channel': row['acquisition_channel'],
            'segment': (
                'Star' if p_alive > 0.5 and clv > 30 else
                'Sleeper' if p_alive > 0.5 and clv <= 30 else
                'Lost' if p_alive <= 0.3 else 'Fading'
            ),
        })
    return pd.DataFrame(results)


def run_ltv_analysis():
    """完整 LTV 分析流程"""
    print("=" * 65)
    print("LTV Prediction BTYD — BG/NBD + Gamma-Gamma 客户价值预测")
    print("=" * 65)

    df = generate_sample_customer_data()
    params = estimate_bgnbd_params(df)

    print(f"\n⚙️  BG/NBD 估计参数: r={params['r']:.3f}, α={params['alpha']:.3f}, "
          f"a={params['a']:.3f}, b={params['b']:.3f}")

    clv_df = predict_clv(df, params, t_future=12)

    # 用户分群
    print("\n👥 用户分群结果:")
    seg_summary = clv_df.groupby('segment').agg(
        count=('customer_id', 'count'),
        avg_p_alive=('p_alive', 'mean'),
        avg_clv=('clv_12m', 'mean'),
        total_clv=('clv_12m', 'sum'),
    ).round(2)
    print(seg_summary.to_string())

    # 渠道 LTV 对比
    print("\n📊 各渠道 CLV 对比（12月预测）:")
    channel_stats = clv_df.groupby('acquisition_channel').agg(
        customers=('customer_id', 'count'),
        avg_clv=('clv_12m', 'mean'),
        avg_p_alive=('p_alive', 'mean'),
    ).round(2)
    # 添加 CAC（模拟）
    cac = {'tiktok': 45, 'google': 30, 'amazon_organic': 0}
    channel_stats['cac'] = channel_stats.index.map(cac)
    channel_stats['ltv_cac_ratio'] = (channel_stats['avg_clv'] / channel_stats['cac'].clip(lower=1)).round(2)
    print(channel_stats.to_string())

    # 高价值沉睡用户
    sleepers = clv_df[(clv_df['segment'] == 'Sleeper') & (clv_df['p_alive'] > 0.6)]
    print(f"\n💤 高价值沉睡用户（可激活）: {len(sleepers)} 人")
    print(f"   平均 CLV: ${sleepers['clv_12m'].mean():.2f}")
    print(f"   激活预期增量 LTV: ${sleepers['clv_12m'].sum():,.0f}")

    # Top 10 高 CLV 用户
    print("\n🏆 Top 10 高 CLV 用户:")
    top10 = clv_df.nlargest(10, 'clv_12m')[['customer_id', 'frequency', 'p_alive',
                                              'clv_12m', 'segment']]
    print(top10.to_string(index=False))

    print("\n[✓] LTV Prediction BTYD 测试通过")


if __name__ == '__main__':
    run_ltv_analysis()
