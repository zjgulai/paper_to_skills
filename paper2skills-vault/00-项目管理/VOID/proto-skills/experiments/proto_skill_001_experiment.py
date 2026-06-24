"""
proto-skill-001: 无平台数据下的内容效力痕迹推断
最小验证实验 — 回溯验证

核心假设：当 TikTok 等平台数据断供时，
痕迹信号（搜索指数变化、客服咨询词频、站内搜索词、出货节律）
能否区分"有效内容发布时段"和"无效内容发布时段"？

成功标准：AUC-ROC > 0.65，且至少2个痕迹信号特征重要性 > 0.15
失败标准：AUC-ROC < 0.55 → 明确放弃假设1

---
数据说明：
  真实实验需要：历史 TikTok 后台数据 + 同期痕迹信号（6个月）
  本验证使用：结构化模拟数据（保留真实业务分布特征）
  模拟依据：
    - 内容类型分布参考真实母婴品牌创作比例（教育30%/促销20%/情感35%/测评15%）
    - 痕迹信号的延迟窗口参考"记忆痕迹效应"（24-168小时）
    - 信号强度分布参考 TikTok 内容效力文献的 effect size 范围
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


CONTENT_TYPES = ['educational', 'promotional', 'emotional', 'review']
CONTENT_TYPE_PROBS = [0.30, 0.20, 0.35, 0.15]

TRACE_SIGNALS = [
    'delta_search_brand',        # 品牌词搜索指数变化（归一化）
    'delta_search_product',      # 产品词搜索指数变化
    'delta_cs_new_keywords',     # 客服咨询中新关键词出现率
    'delta_cs_volume',           # 客服咨询量变化
    'delta_listing_search',      # 站内搜索词中提及视频关键词的频次
    'delta_shipping_volume',     # 出货量变化（vs 同期基线）
]

DELAY_WINDOWS = [24, 48, 72, 168]


def generate_content_publish_data(
    n_days: int = 180,
    posts_per_week: float = 4.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成内容发布日志（模拟6个月的母婴品牌发布记录）

    真实场景：从 TikTok 创作者后台导出发布日志
    模拟依据：母婴 DTC 品牌平均发布频次 4-5条/周，内容类型分布已知
    """
    np.random.seed(seed)
    dates = pd.date_range('2025-12-01', periods=n_days, freq='D')

    posts = []
    for date in dates:
        n_posts = np.random.poisson(posts_per_week / 7)
        for _ in range(n_posts):
            content_type = np.random.choice(CONTENT_TYPES, p=CONTENT_TYPE_PROBS)
            posts.append({'date': date, 'content_type': content_type})

    df = pd.DataFrame(posts)
    print(f"[数据] 生成内容发布日志: {len(df)} 条，时间跨度 {n_days} 天")
    print(f"       类型分布: {dict(df['content_type'].value_counts())}")
    return df


def generate_platform_ground_truth(
    content_df: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成 TikTok 平台侧 ground truth（模拟真实转化数据）

    真实场景：从 TikTok 商家后台获取每条内容的转化数
    模拟依据：
      - 内容效力存在真实差异（教育类转化相对稳定，促销类方差大，情感类尾部长）
      - 发布时机（工作日 vs 周末，早晚高峰）对效力有影响
      - 加入随机噪声模拟平台算法的不确定性
    """
    np.random.seed(seed)

    base_conversion = {
        'educational': 85,
        'promotional': 120,
        'emotional': 95,
        'review': 75,
    }
    variance = {
        'educational': 25,
        'promotional': 60,
        'emotional': 45,
        'review': 30,
    }

    df = content_df.copy()

    conversions = []
    for _, row in df.iterrows():
        ct = row['content_type']
        base = base_conversion[ct]
        std = variance[ct]

        weekend_boost = 1.2 if row['date'].weekday() >= 5 else 1.0
        evening_boost = 1.15 if row['date'].month in [11, 12, 1] else 1.0

        conv = max(0, np.random.normal(base * weekend_boost * evening_boost, std))
        conversions.append(conv)

    df['platform_conversions'] = conversions
    median_conv = df['platform_conversions'].median()
    df['effective'] = (df['platform_conversions'] > median_conv).astype(int)

    print(f"[数据] 平台 Ground Truth: 有效内容 {df['effective'].sum()}/{len(df)} "
          f"({df['effective'].mean():.1%})")
    return df


def generate_trace_signals(
    content_df: pd.DataFrame,
    signal_strength: float = 0.6,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成痕迹信号特征（模拟从各渠道采集的周边信号）

    真实场景：
      - delta_search_brand：百度指数/Google Trends API
      - delta_cs_new_keywords：客服系统每日关键词统计
      - delta_listing_search：Amazon/独立站搜索词报告
      - delta_shipping_volume：WMS 系统出货日报

    模拟关键假设：
      signal_strength 控制痕迹信号与内容效力的相关程度
      0.0 = 完全随机（纯噪声），1.0 = 完美预测
      真实世界预期 signal_strength ≈ 0.3-0.6（如果假设1成立）

    实验设计：
      - 用 signal_strength=0.6 测试"信号存在"的极端乐观假设
      - 用 signal_strength=0.2 测试接近真实世界的保守假设
      - 两种条件下模型的 AUC 变化范围 = 假设1的可信区间
    """
    np.random.seed(seed)
    df = content_df.copy()

    effective_signal = df['effective'].values.astype(float)

    for signal in TRACE_SIGNALS:
        noise = np.random.randn(len(df))

        lag = np.zeros(len(df))
        lag[1:] = effective_signal[:-1]
        for i in range(2, min(8, len(df))):
            lag[i:] += effective_signal[:-i] * (0.5 ** (i - 1))

        raw = signal_strength * lag + (1 - signal_strength) * noise

        if signal in ['delta_search_brand', 'delta_search_product']:
            raw = np.abs(raw) * 1.2
        elif 'cs' in signal:
            raw = raw * 0.8

        df[signal] = raw

    for window in DELAY_WINDOWS:
        days = window // 24
        for signal in TRACE_SIGNALS[:3]:
            col = f"{signal}_lag{window}h"
            df[col] = df[signal].shift(days).fillna(0)

    print(f"[数据] 痕迹信号生成完成: {len(TRACE_SIGNALS)} 个基础信号 + "
          f"{len(TRACE_SIGNALS[:3]) * len(DELAY_WINDOWS)} 个时滞特征")
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """构建模型输入特征矩阵"""
    lag_cols = [c for c in df.columns if any(s in c for s in TRACE_SIGNALS)
                and c not in ['effective', 'platform_conversions', 'content_type', 'date']]

    content_dummies = pd.get_dummies(df['content_type'], prefix='type')

    feature_cols = lag_cols + list(content_dummies.columns)
    X = pd.concat([df[lag_cols], content_dummies], axis=1).fillna(0).values
    y = df['effective'].values

    return X, y, feature_cols


def run_timeseries_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
    n_splits: int = 4
) -> Dict:
    """
    时序交叉验证：防止数据泄露

    用前 N 个月训练，验证第 N+1 个月
    不使用随机分割（时序数据必须保持时间顺序）
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    models = {
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.05, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'LR': LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {name: [] for name in models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in models.items():
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, proba)
            results[name].append(auc)

    return results, models, scaler, X, y, feature_cols


def analyze_feature_importance(
    model: GradientBoostingClassifier,
    feature_cols: list,
    top_n: int = 12
) -> pd.DataFrame:
    """提取并排序特征重要性"""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_cols[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    fi_df['signal_group'] = fi_df['feature'].apply(lambda x:
        next((s for s in TRACE_SIGNALS if s in x), 'content_type'))

    return fi_df.head(top_n)


def evaluate_verdict(cv_results: Dict, feature_importance_df: pd.DataFrame) -> Dict:
    """
    按成功/失败标准判定假设1

    成功标准：
      1. 最佳模型 AUC-ROC 均值 > 0.65
      2. 至少 2 个不同痕迹信号的特征重要性 > 0.15

    失败标准：AUC < 0.55（比随机无显著提升）
    """
    aucs = {name: np.mean(aucs) if aucs else 0
            for name, aucs in cv_results.items()}
    best_model = max(aucs, key=aucs.get)
    best_auc = aucs[best_model]

    high_importance_signals = (
        feature_importance_df[feature_importance_df['importance'] > 0.15]
        ['signal_group'].unique().tolist()
    )
    content_type_signals = [s for s in high_importance_signals if s != 'content_type']

    if best_auc > 0.65 and len(content_type_signals) >= 2:
        verdict = "✅ 假设1成立：痕迹信号对内容效力有预测力"
        recommendation = "升级为标准Skill，启动真实数据验证"
    elif best_auc > 0.55:
        verdict = "⚠️ 信号微弱：AUC 在阈值区间，需要真实数据进一步验证"
        recommendation = "启动真实数据采集，勿直接升级"
    else:
        verdict = "❌ 假设1证伪：痕迹信号无法有效代理平台数据"
        recommendation = "存档失败案例，降级至 Level 2"

    return {
        'best_model': best_model,
        'best_auc': best_auc,
        'all_aucs': aucs,
        'high_importance_signals': content_type_signals,
        'verdict': verdict,
        'recommendation': recommendation,
    }


def run_sensitivity_analysis(df_with_signals_func, n_trials: int = 3) -> pd.DataFrame:
    """
    敏感性分析：在不同 signal_strength 假设下，AUC 如何变化？

    这回答了：如果真实世界的信号强度是 X，模型能有多好？
    帮助设定真实数据采集的预期。
    """
    results = []
    for strength in [0.2, 0.4, 0.6]:
        aucs_at_strength = []
        for trial in range(n_trials):
            content_df = generate_content_publish_data(seed=trial * 100)
            gt_df = generate_platform_ground_truth(content_df, seed=trial * 100)
            trace_df = generate_trace_signals(gt_df, signal_strength=strength,
                                              seed=trial * 100)
            X, y, feature_cols = build_feature_matrix(trace_df)
            cv_results, _, _, _, _, _ = run_timeseries_cv(X, y, feature_cols)
            best_auc = max(np.mean(v) for v in cv_results.values() if v)
            aucs_at_strength.append(best_auc)

        results.append({
            'signal_strength': strength,
            'mean_auc': np.mean(aucs_at_strength),
            'std_auc': np.std(aucs_at_strength),
        })
    return pd.DataFrame(results)


def main():
    print("=" * 65)
    print("proto-skill-001 验证实验")
    print("无平台数据下的内容效力痕迹推断")
    print("=" * 65)

    print("\n【Phase 1】数据生成（模拟6个月母婴品牌运营数据）")
    print("-" * 45)
    content_df = generate_content_publish_data(n_days=180)
    gt_df = generate_platform_ground_truth(content_df)
    trace_df = generate_trace_signals(gt_df, signal_strength=0.5)

    print("\n【Phase 2】特征工程")
    print("-" * 45)
    X, y, feature_cols = build_feature_matrix(trace_df)
    print(f"特征矩阵: {X.shape[0]} 样本 × {X.shape[1]} 特征")
    print(f"标签分布: 有效={y.sum()}, 无效={len(y)-y.sum()}, 平衡比={y.mean():.2f}")

    print("\n【Phase 3】时序交叉验证（4折，防数据泄露）")
    print("-" * 45)
    cv_results, models, scaler, X, y, feature_cols = run_timeseries_cv(
        X, y, feature_cols, n_splits=4)

    print("各模型 AUC-ROC 均值：")
    for name, aucs in cv_results.items():
        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            bar = "█" * int(mean_auc * 20)
            print(f"  {name:<8}: {mean_auc:.4f} ± {std_auc:.4f}  {bar}")

    print("\n【Phase 4】特征重要性分析")
    print("-" * 45)
    gbm_model = models['GBM']
    X_all_s = StandardScaler().fit_transform(X)
    gbm_model.fit(X_all_s, y)

    fi_df = analyze_feature_importance(gbm_model, feature_cols)
    print(f"{'特征名':<35} {'重要性':>8} {'信号组'}")
    print("-" * 60)
    for _, row in fi_df.iterrows():
        bar = "▪" * int(row['importance'] * 100)
        print(f"  {row['feature']:<33} {row['importance']:>8.4f}  "
              f"{row['signal_group']:<25} {bar}")

    signal_diversity = fi_df[fi_df['importance'] > 0.15]['signal_group'].nunique()
    print(f"\n重要性 > 0.15 的信号类型数: {signal_diversity}")

    print("\n【Phase 5】敏感性分析（信号强度梯度测试）")
    print("-" * 45)
    print("测试不同真实世界信号强度下的预测能力...")
    sensitivity_df = run_sensitivity_analysis(None, n_trials=3)
    print(f"\n{'信号强度':>8} {'均值AUC':>10} {'标准差':>8}  {'评级'}")
    print("-" * 45)
    for _, row in sensitivity_df.iterrows():
        rating = "✅优秀" if row['mean_auc'] > 0.65 else ("⚠️可用" if row['mean_auc'] > 0.55 else "❌差")
        print(f"  {row['signal_strength']:>6.1f}    {row['mean_auc']:>10.4f} "
              f"{row['std_auc']:>8.4f}  {rating}")

    print("\n【Phase 6】假设判定")
    print("=" * 65)
    verdict = evaluate_verdict(cv_results, fi_df)

    print(f"\n最佳模型: {verdict['best_model']}")
    print(f"最佳 AUC: {verdict['best_auc']:.4f}")
    print(f"高重要性痕迹信号 (>0.15): {verdict['high_importance_signals']}")
    print(f"\n判定结果: {verdict['verdict']}")
    print(f"建议操作: {verdict['recommendation']}")

    print("\n" + "=" * 65)
    print("实验摘要（用于更新 proto-skill-001 进展日志）")
    print("=" * 65)
    best_auc = verdict['best_auc']
    print(f"""
阶段    : 最小验证实验（模拟数据回溯验证）
数据规模: 180天 × {X.shape[0]}条发布记录
特征数量: {X.shape[1]}个（{len(TRACE_SIGNALS)}信号×{len(DELAY_WINDOWS)}时滞+内容类型）
最佳AUC : {best_auc:.4f} （判定阈值：>0.65成功，<0.55失败）
高重要性: {', '.join(verdict['high_importance_signals']) if verdict['high_importance_signals'] else '无'}
结论    : {verdict['verdict']}
下一步  : {verdict['recommendation']}
""")

    print(f"[✓] proto-skill-001 验证实验完成")
    return verdict


if __name__ == "__main__":
    result = main()
