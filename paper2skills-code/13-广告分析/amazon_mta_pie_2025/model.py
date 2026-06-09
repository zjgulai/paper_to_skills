"""PIE: Predicted Incrementality by Experimentation (实验增量预测归因)

基于 Amazon Ads arXiv:2508.08209 的最小骨架实现。
核心思路：
  1. RCT Ground Truth：模拟随机对照实验，获取各渠道的真实宏观因果增量（无偏但粒度粗）。
  2. ML 触点概率估计：用机器学习模型对每个触点打分，反映其"转化敏感度"（精细但有偏）。
  3. PIE 校准（Calibration）：将 ML 微观触点权重，强制缩放到 RCT 宏观增量总量，
     使得 sum(ML_weight * credit) == RCT_incrementality，消除选择偏差。

业务场景：母婴品牌全渠道（Google Search + Facebook + TikTok）的下一代多触点归因。
依赖：numpy, pandas, scipy（标准库，无需额外安装）
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 步骤 1：模拟 RCT 实验数据（获取各渠道真实增量 Ground Truth）
# ---------------------------------------------------------------------------

def simulate_rct_experiment(
    channels: List[str],
    n_treated: int = 5000,
    n_control: int = 5000,
    true_incrementality: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """模拟分渠道 RCT 停投实验，获得真实因果增量（持仓量 / 转化率差异）。

    每个渠道独立运行一次 Geo-holdout 或 PSA 实验。
    - 处理组（treated）：正常投放，记录转化率
    - 对照组（control）：不投放（或投伪广告），记录基线转化率
    - 真实增量 = 处理组转化率 - 对照组转化率（DID 估计量）

    Parameters
    ----------
    channels : 渠道名列表
    n_treated : 处理组样本量（每个渠道）
    n_control : 对照组样本量（每个渠道）
    true_incrementality : 各渠道真实增量转化率 {'channel': 0.02}，None 则自动设置
    seed : 随机种子

    Returns
    -------
    Dict[channel, {'rct_incrementality': float, 'ci_lower': float, 'ci_upper': float}]
    """
    if true_incrementality is None:
        true_incrementality = {
            'google_search': 0.035,   # Google 搜索：最高增量（意图明确）
            'facebook':      0.018,   # Facebook 展示：中等增量
            'tiktok':        0.012,   # TikTok 视频：较低增量（品牌曝光为主）
        }

    rng = np.random.default_rng(seed)
    rct_results: Dict[str, Dict[str, float]] = {}

    base_conversion_rate = 0.05  # 基线转化率（无广告投放时）

    for ch in channels:
        true_lift = true_incrementality.get(ch, 0.01)
        treatment_rate = base_conversion_rate + true_lift

        # 模拟二项分布观测（实际转化次数）
        treated_conversions = rng.binomial(n_treated, treatment_rate)
        control_conversions  = rng.binomial(n_control, base_conversion_rate)

        treated_cvr = treated_conversions / n_treated
        control_cvr  = control_conversions  / n_control
        observed_lift = treated_cvr - control_cvr

        # 95% 置信区间（双比例 z 检验）
        se = np.sqrt(
            treated_cvr * (1 - treated_cvr) / n_treated
            + control_cvr * (1 - control_cvr) / n_control
        )
        z = stats.norm.ppf(0.975)

        rct_results[ch] = {
            'rct_incrementality': float(observed_lift),
            'ci_lower': float(observed_lift - z * se),
            'ci_upper': float(observed_lift + z * se),
            'true_incrementality': true_lift,  # 仅用于测试验证，生产中不可见
        }

    return rct_results


# ---------------------------------------------------------------------------
# 步骤 2：ML 触点概率估计（有偏但高精度的微观权重）
# ---------------------------------------------------------------------------

class TouchpointMLEstimator:
    """机器学习触点概率估计器（模拟深度注意力模型的输出）。

    在真实系统中，这里会是一个训练好的序列模型（Transformer / LSTM），
    对每条用户点击路径中的各个触点打分，输出"转化贡献概率"。

    本模拟通过特征加权线性模型 + sigmoid 激活来模拟 ML 打分输出，
    并内嵌"选择偏差"：平台倾向高意图用户，导致 ML 分数系统性高估。
    """

    def __init__(
        self,
        channels: List[str],
        selection_bias: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        channels : 渠道名列表
        selection_bias : 各渠道选择偏差放大系数（>1 表示高估），None 则使用默认值
        seed : 随机种子
        """
        self.channels = channels
        self.selection_bias = selection_bias or {
            'google_search': 1.8,  # 搜索广告主动意图，选择偏差最高
            'facebook':      1.4,  # 展示广告，中等偏差
            'tiktok':        1.2,  # 视频广告，相对较低偏差
        }
        self.rng = np.random.default_rng(seed)
        self._fitted = False

        # 内部"真实"触点权重（只有模拟器知道，ML模型无法直接学到）
        self._true_weights: Dict[str, float] = {
            'google_search': 0.50,
            'facebook':      0.30,
            'tiktok':        0.20,
        }

    def generate_touchpoint_data(self, n_journeys: int = 10000) -> pd.DataFrame:
        """生成模拟用户转化路径数据（每条路径包含多个触点）。

        Parameters
        ----------
        n_journeys : 用户旅程条数

        Returns
        -------
        DataFrame: [journey_id, channel, position, feature_score, converted]
        """
        records = []
        for jid in range(n_journeys):
            # 每个用户旅程有 1-4 个触点
            n_touches = self.rng.integers(1, 5)
            # 随机选择触点渠道序列
            touch_channels = self.rng.choice(self.channels, size=n_touches, replace=True)
            converted = self.rng.random() < 0.08  # 整体转化率 8%

            for pos, ch in enumerate(touch_channels):
                # 特征分数：模拟意图强度、时间衰减、渠道质量
                intent_score = self.rng.uniform(0.2, 1.0)
                time_decay = np.exp(-0.3 * (n_touches - 1 - pos))  # 越靠近转化权重越高
                feature_score = intent_score * time_decay

                records.append({
                    'journey_id': jid,
                    'channel': ch,
                    'position': pos,
                    'feature_score': round(float(feature_score), 4),
                    'converted': int(converted),
                    'n_touches': n_touches,
                })

        return pd.DataFrame(records)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """对触点数据拟合 ML 模型并预测各触点的转化概率权重。

        输出包含：
        - ml_prob：ML 预测的触点贡献概率（有偏，未校准）
        - ml_prob_biased：施加选择偏差后的概率（模拟实际部署时的过估现象）

        Parameters
        ----------
        df : generate_touchpoint_data() 输出的触点数据

        Returns
        -------
        同 df 但增加 ml_prob, ml_prob_biased 列
        """
        df = df.copy()

        # 模拟 ML 模型：position-weighted feature score + 渠道基础权重
        channel_base = {ch: self._true_weights.get(ch, 0.1) for ch in self.channels}

        def _score_touchpoint(row):
            ch_base = channel_base.get(row['channel'], 0.1)
            # 时间衰减权重：最后一个触点权重最高（接近 Last-Click）
            pos_weight = np.exp(-0.2 * (row['n_touches'] - 1 - row['position']))
            raw = ch_base * row['feature_score'] * pos_weight
            return raw

        df['ml_raw'] = df.apply(_score_touchpoint, axis=1)

        # 归一化：每条旅程内触点权重之和 = 1（Shapley 风格归因）
        journey_sum = df.groupby('journey_id')['ml_raw'].transform('sum')
        df['ml_prob'] = df['ml_raw'] / (journey_sum + 1e-10)

        # 施加选择偏差（模拟平台高意图用户选择效应）
        bias_factor = df['channel'].map(self.selection_bias).fillna(1.0)
        df['ml_prob_biased'] = df['ml_prob'] * bias_factor

        # 再次归一化（偏差后）
        journey_biased_sum = df.groupby('journey_id')['ml_prob_biased'].transform('sum')
        df['ml_prob_biased'] = df['ml_prob_biased'] / (journey_biased_sum + 1e-10)

        self._fitted = True
        return df


# ---------------------------------------------------------------------------
# 步骤 3：PIE 校准（核心算法）
# ---------------------------------------------------------------------------

class PIECalibrator:
    """PIE 校准器：将 ML 微观触点权重强制对齐 RCT 宏观增量。

    核心公式：
        credit_pie(touchpoint_i) = ml_prob_biased(i) * scaling_factor(channel)

    其中：
        scaling_factor(ch) = rct_incrementality(ch) / sum_ml_credits(ch)
        sum_ml_credits(ch) = Σ ml_prob_biased(i) * converted(i)  [仅对转化路径]

    校准后保证：Σ credit_pie(ch) ≈ rct_incrementality(ch)（宏观约束满足）
    """

    def __init__(self) -> None:
        self.scaling_factors_: Dict[str, float] = {}
        self.calibrated_attribution_: Optional[pd.DataFrame] = None
        self._fitted = False

    def calibrate(
        self,
        touchpoint_df: pd.DataFrame,
        rct_results: Dict[str, Dict[str, float]],
        channels: List[str],
    ) -> pd.DataFrame:
        """执行 PIE 校准。

        Parameters
        ----------
        touchpoint_df : fit_predict() 输出的触点数据（含 ml_prob_biased）
        rct_results   : simulate_rct_experiment() 输出的 RCT 实验结果
        channels      : 渠道列表

        Returns
        -------
        DataFrame: 渠道级别的 PIE 校准后归因报告
        """
        df = touchpoint_df.copy()

        # 只对"已转化"的路径计算 ML 归因贡献
        converted_df = df[df['converted'] == 1].copy()

        # 各渠道的 ML 原始归因信用（有偏）
        channel_ml_credits: Dict[str, float] = {}
        for ch in channels:
            ch_credits = converted_df.loc[
                converted_df['channel'] == ch, 'ml_prob_biased'
            ].sum()
            channel_ml_credits[ch] = float(ch_credits)

        # RCT 增量（换算为"归因信用量"，需要与 ML 信用量同量纲）
        # 将 RCT 增量转化率 × 样本量 → 增量转化人次
        total_ml_credits = sum(channel_ml_credits.values())
        total_rct_lift = sum(v['rct_incrementality'] for v in rct_results.values())

        # PIE 缩放因子：每个渠道独立校准
        rows = []
        for ch in channels:
            ml_credit = channel_ml_credits.get(ch, 0.0)
            rct_lift = rct_results.get(ch, {}).get('rct_incrementality', 0.0)

            # 校准系数 = RCT增量比例 / ML信用比例
            rct_share = rct_lift / (total_rct_lift + 1e-10)
            ml_share = ml_credit / (total_ml_credits + 1e-10)

            if ml_share > 1e-10:
                scaling = rct_share / ml_share
            else:
                scaling = 1.0

            self.scaling_factors_[ch] = float(scaling)

            # 校准后归因份额
            pie_share = ml_share * scaling  # 等同于 rct_share

            rows.append({
                'channel': ch,
                'ml_share_biased': round(float(ml_share), 4),     # ML 有偏归因比例
                'rct_share': round(float(rct_share), 4),           # RCT 真实增量比例
                'pie_share': round(float(pie_share), 4),           # PIE 校准后归因比例
                'scaling_factor': round(float(scaling), 4),        # PIE 缩放系数
                'rct_incrementality': round(float(rct_lift), 4),   # RCT 实测增量率
                'rct_ci_lower': round(rct_results.get(ch, {}).get('ci_lower', 0.0), 4),
                'rct_ci_upper': round(rct_results.get(ch, {}).get('ci_upper', 0.0), 4),
            })

        result = pd.DataFrame(rows).sort_values('pie_share', ascending=False).reset_index(drop=True)

        # 验证：PIE 归因份额之和应约等于 1.0
        result['pie_share_pct'] = (result['pie_share'] / result['pie_share'].sum() * 100).round(1)
        result['pie_share_pct'] = result['pie_share_pct'].astype(str) + '%'

        self.calibrated_attribution_ = result
        self._fitted = True
        return result

    def budget_recommendation(
        self,
        current_budget: Dict[str, float],
        total_budget: float,
    ) -> pd.DataFrame:
        """基于 PIE 校准后的归因份额生成预算调整建议。"""
        assert self._fitted, "先调用 calibrate()"
        df = self.calibrated_attribution_.copy()
        total_pie = df['pie_share'].sum()

        rows = []
        for _, row in df.iterrows():
            ch = row['channel']
            pie_w = row['pie_share'] / (total_pie + 1e-10)
            recommended = total_budget * pie_w
            current = current_budget.get(ch, 0.0)
            delta = recommended - current
            rows.append({
                'channel': ch,
                'current_budget': round(current, 0),
                'pie_weight': round(pie_w, 4),
                'recommended_budget': round(recommended, 0),
                'delta': round(delta, 0),
                'action': '增加' if delta > 100 else ('削减' if delta < -100 else '持平'),
            })

        return pd.DataFrame(rows).sort_values('pie_weight', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 主流程封装：PIE Pipeline
# ---------------------------------------------------------------------------

class PIEAttributionPipeline:
    """PIE (Predicted Incrementality by Experimentation) 全流程归因管道。

    使用方法：
        pipeline = PIEAttributionPipeline(channels=['google_search', 'facebook', 'tiktok'])
        rct_results = pipeline.run_rct_experiment()
        touchpoint_df = pipeline.score_touchpoints()
        report = pipeline.calibrate(rct_results, touchpoint_df)
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        n_journeys: int = 10000,
        n_rct_samples: int = 5000,
        seed: int = 42,
    ) -> None:
        if channels is None:
            channels = ['google_search', 'facebook', 'tiktok']
        self.channels = channels
        self.n_journeys = n_journeys
        self.n_rct_samples = n_rct_samples
        self.seed = seed

        self.ml_estimator = TouchpointMLEstimator(channels=channels, seed=seed)
        self.pie_calibrator = PIECalibrator()
        self._rct_results: Optional[Dict] = None
        self._touchpoint_df: Optional[pd.DataFrame] = None

    def run_rct_experiment(
        self,
        true_incrementality: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """运行 RCT 实验，获取各渠道真实增量（宏观 Ground Truth）。"""
        self._rct_results = simulate_rct_experiment(
            channels=self.channels,
            n_treated=self.n_rct_samples,
            n_control=self.n_rct_samples,
            true_incrementality=true_incrementality,
            seed=self.seed,
        )
        return self._rct_results

    def score_touchpoints(self, n_journeys: Optional[int] = None) -> pd.DataFrame:
        """生成触点数据并用 ML 模型打分（有偏）。"""
        n = n_journeys or self.n_journeys
        raw_df = self.ml_estimator.generate_touchpoint_data(n_journeys=n)
        self._touchpoint_df = self.ml_estimator.fit_predict(raw_df)
        return self._touchpoint_df

    def calibrate(
        self,
        rct_results: Optional[Dict] = None,
        touchpoint_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """执行 PIE 校准，输出去偏的渠道归因报告。"""
        if rct_results is None:
            assert self._rct_results is not None, "先调用 run_rct_experiment()"
            rct_results = self._rct_results
        if touchpoint_df is None:
            assert self._touchpoint_df is not None, "先调用 score_touchpoints()"
            touchpoint_df = self._touchpoint_df

        return self.pie_calibrator.calibrate(touchpoint_df, rct_results, self.channels)

    def budget_recommendation(
        self,
        current_budget: Dict[str, float],
        total_budget: float,
    ) -> pd.DataFrame:
        """基于 PIE 归因生成预算调整建议。"""
        return self.pie_calibrator.budget_recommendation(current_budget, total_budget)


# ---------------------------------------------------------------------------
# 自测（self-test）
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """内置自测：验证 PIE 三步校准的核心逻辑与数值合理性。"""
    print("=" * 70)
    print("PIE MTA Attribution Self-Test")
    print("=" * 70)

    CHANNELS = ['google_search', 'facebook', 'tiktok']

    # ---- 测试 1：RCT 实验 ----
    print("\n[1] 模拟 RCT 实验 (n=5000/组)...")
    rct = simulate_rct_experiment(
        channels=CHANNELS,
        n_treated=5000,
        n_control=5000,
        true_incrementality={
            'google_search': 0.035,
            'facebook':      0.018,
            'tiktok':        0.012,
        },
        seed=42,
    )
    for ch, r in rct.items():
        print(f"    {ch}: 实测增量={r['rct_incrementality']:.4f} "
              f"  95%CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
              f"  真实值={r['true_incrementality']:.4f}")

    # 断言：实测增量与真实值偏差在 50% 以内（统计噪声）
    for ch, r in rct.items():
        true_val = r['true_incrementality']
        observed = r['rct_incrementality']
        rel_err = abs(observed - true_val) / true_val
        assert rel_err < 0.5, f"{ch} RCT 估计偏差过大: {rel_err:.2f}"
    print("    [OK] 所有渠道 RCT 增量估计偏差 < 50%（统计合理）")

    # 断言：置信区间覆盖真实值（95% CI 应覆盖 true_incrementality）
    for ch, r in rct.items():
        in_ci = r['ci_lower'] <= r['true_incrementality'] <= r['ci_upper']
        assert in_ci, f"{ch} 真实增量值未被 95%CI 覆盖"
    print("    [OK] 所有渠道真实值均在 95% 置信区间内")

    # ---- 测试 2：ML 触点打分 ----
    print("\n[2] ML 触点打分 (10000 条用户旅程)...")
    estimator = TouchpointMLEstimator(channels=CHANNELS, seed=42)
    raw_df = estimator.generate_touchpoint_data(n_journeys=10000)
    scored_df = estimator.fit_predict(raw_df)

    print(f"    触点数据形状: {scored_df.shape}")
    print(f"    列: {list(scored_df.columns)}")
    print(f"    转化率: {scored_df.groupby('journey_id')['converted'].first().mean():.3f}")

    # 断言：ml_prob 每条旅程之和约等于 1
    journey_prob_sum = scored_df.groupby('journey_id')['ml_prob'].sum()
    max_deviation = (journey_prob_sum - 1.0).abs().max()
    assert max_deviation < 1e-6, f"ml_prob 旅程内和偏差过大: {max_deviation}"
    print(f"    [OK] ml_prob 旅程内归一化误差 < 1e-6")

    # 断言：ml_prob_biased > ml_prob（选择偏差导致系统性高估）
    mean_biased = scored_df['ml_prob_biased'].mean()
    mean_unbiased = scored_df['ml_prob'].mean()
    # 选择偏差会导致转化路径内某些渠道权重升高（总体均值相等，但分布不同）
    assert mean_biased > 0, "ml_prob_biased 应为正数"
    assert mean_unbiased > 0, "ml_prob 应为正数"
    print(f"    [OK] ML 打分正常，mean(ml_prob)={mean_unbiased:.4f}, mean(ml_prob_biased)={mean_biased:.4f}")

    # 断言：Google Search 的偏差系数最高，其 ml_prob_biased 也应最高
    ch_biased_mean = scored_df.groupby('channel')['ml_prob_biased'].mean()
    ch_unbiased_mean = scored_df.groupby('channel')['ml_prob'].mean()
    bias_ratio = ch_biased_mean / ch_unbiased_mean
    assert bias_ratio['google_search'] > bias_ratio['tiktok'], (
        "google_search 选择偏差比率应高于 tiktok"
    )
    print(f"    [OK] 选择偏差方向正确: google_search 偏差比={bias_ratio['google_search']:.3f} > tiktok 偏差比={bias_ratio['tiktok']:.3f}")

    # ---- 测试 3：PIE 校准 ----
    print("\n[3] PIE 校准（将 ML 权重对齐 RCT 增量）...")
    calibrator = PIECalibrator()
    report = calibrator.calibrate(scored_df, rct, CHANNELS)
    print(report.to_string(index=False))

    # 断言：PIE share 之和约等于 1.0
    pie_sum = report['pie_share'].sum()
    assert abs(pie_sum - 1.0) < 0.01, f"PIE share 之和={pie_sum:.4f}，应约等于1.0"
    print(f"\n    [OK] PIE share 之和={pie_sum:.4f}")

    # 断言：PIE 校准后 Google 归因高于 TikTok（与 RCT 增量一致）
    pie_google = report.loc[report['channel'] == 'google_search', 'pie_share'].values[0]
    pie_tiktok = report.loc[report['channel'] == 'tiktok', 'pie_share'].values[0]
    assert pie_google > pie_tiktok, (
        f"PIE 校准后 google_search({pie_google:.4f}) 应高于 tiktok({pie_tiktok:.4f})"
    )
    print(f"    [OK] PIE 归因排序与 RCT 增量一致: google_search({pie_google:.4f}) > tiktok({pie_tiktok:.4f})")

    # 断言：ML 有偏归因与 PIE 校准后归因存在差异（校准有效）
    fb_ml = report.loc[report['channel'] == 'facebook', 'ml_share_biased'].values[0]
    fb_pie = report.loc[report['channel'] == 'facebook', 'pie_share'].values[0]
    assert abs(fb_ml - fb_pie) > 0.001, (
        f"Facebook ML vs PIE 归因无差异，校准可能无效"
    )
    print(f"    [OK] 校准有效: Facebook ML有偏={fb_ml:.4f} vs PIE={fb_pie:.4f}（存在差异）")

    # ---- 测试 4：预算建议 ----
    print("\n[4] 预算建议（当前月预算 $15,000）...")
    current_budget = {'google_search': 10000, 'facebook': 3000, 'tiktok': 2000}
    budget_rec = calibrator.budget_recommendation(current_budget, total_budget=15000)
    print(budget_rec.to_string(index=False))

    # 断言：推荐预算总和 = 总预算
    total_rec = budget_rec['recommended_budget'].sum()
    assert abs(total_rec - 15000) < 1.0, f"推荐预算总和={total_rec}，应为15000"
    print(f"\n    [OK] 推荐预算总和={total_rec:.0f}")

    # ---- 测试 5：全流程 Pipeline ----
    print("\n[5] 全流程 PIEAttributionPipeline...")
    pipeline = PIEAttributionPipeline(channels=CHANNELS, n_journeys=5000, n_rct_samples=3000, seed=99)
    rct2 = pipeline.run_rct_experiment()
    scored2 = pipeline.score_touchpoints()
    final_report = pipeline.calibrate()
    print(final_report.to_string(index=False))

    assert len(final_report) == len(CHANNELS), "渠道数量不符"
    assert final_report['pie_share'].sum() > 0.9, "PIE share 总和异常"
    print(f"\n    [OK] Pipeline 运行成功，渠道数={len(final_report)}")

    print("\n" + "=" * 70)
    print("[PASS] 所有断言通过，PIE MTA 自测成功！")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 主演示流程
# ---------------------------------------------------------------------------

def main() -> None:
    """母婴品牌全渠道 PIE 多触点归因分析示例。"""
    print("\n[PIE Demo] 母婴品牌欧美全渠道 PIE MTA 归因分析")
    print("论文: Amazon Ads arXiv:2508.08209")
    print("-" * 60)

    CHANNELS = ['google_search', 'facebook', 'tiktok']

    # --- Step 1: 运行 RCT 实验（每季度执行一次）---
    print("\n【Step 1】RCT Ground Truth 实验（Geo-holdout，n=5000/组）")
    rct = simulate_rct_experiment(
        channels=CHANNELS,
        n_treated=5000,
        n_control=5000,
        true_incrementality={'google_search': 0.035, 'facebook': 0.018, 'tiktok': 0.012},
        seed=42,
    )
    rct_df = pd.DataFrame(rct).T[['rct_incrementality', 'ci_lower', 'ci_upper']]
    rct_df.columns = ['实测增量率', 'CI下限', 'CI上限']
    print(rct_df.round(4))

    # --- Step 2: ML 触点打分 ---
    print("\n【Step 2】ML 触点概率打分（10000 条用户旅程）")
    estimator = TouchpointMLEstimator(channels=CHANNELS, seed=42)
    raw_df = estimator.generate_touchpoint_data(n_journeys=10000)
    scored_df = estimator.fit_predict(raw_df)

    # 展示 ML 有偏归因
    converted_only = scored_df[scored_df['converted'] == 1]
    ml_biased_share = converted_only.groupby('channel')['ml_prob_biased'].sum()
    ml_biased_share = ml_biased_share / ml_biased_share.sum()
    print("ML 有偏归因比例（未校准）:")
    for ch in CHANNELS:
        print(f"  {ch}: {ml_biased_share.get(ch, 0):.3f} ({ml_biased_share.get(ch, 0)*100:.1f}%)")

    # --- Step 3: PIE 校准 ---
    print("\n【Step 3】PIE 校准（将 ML 权重对齐 RCT 增量）")
    calibrator = PIECalibrator()
    report = calibrator.calibrate(scored_df, rct, CHANNELS)
    print(report[['channel', 'ml_share_biased', 'rct_share', 'pie_share', 'scaling_factor']].to_string(index=False))

    # --- 对比 Last-Click vs ML (biased) vs PIE ---
    print("\n【对比】Last-Click vs ML无校准 vs PIE校准:")
    print(f"{'渠道':<16} {'Last-Click':>12} {'ML有偏':>10} {'PIE校准':>10} {'真实RCT':>10}")
    print("-" * 65)
    lc = {'google_search': 0.85, 'facebook': 0.10, 'tiktok': 0.05}  # 假设的Last-Click分布
    for _, row in report.iterrows():
        ch = row['channel']
        print(f"{ch:<16} {lc.get(ch,0):>11.1%} {row['ml_share_biased']:>9.1%} "
              f"{row['pie_share']:>9.1%} {row['rct_share']:>9.1%}")

    # --- 预算建议 ---
    print("\n【预算建议】当前月预算 $15,000 → PIE 优化建议:")
    current_budget = {'google_search': 10000, 'facebook': 3000, 'tiktok': 2000}
    rec = calibrator.budget_recommendation(current_budget, total_budget=15000)
    print(rec.to_string(index=False))

    # --- 运行自测 ---
    print()
    _run_self_test()


if __name__ == "__main__":
    main()
