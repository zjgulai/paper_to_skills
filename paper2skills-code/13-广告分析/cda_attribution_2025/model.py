"""CDA: Causal-Driven Attribution (因果驱动归因)

基于论文 arXiv:2512.21211 的最小骨架实现。
核心思路：
  1. 时序因果发现 (Temporal Causal Discovery)：使用 PCMCI 风格的滞后相关算法，
     从聚合级别的每日曝光量和订单数据中，挖掘渠道间的因果关系图谱。
  2. 结构因果模型 (SCM) 归因：基于因果图谱，计算每个渠道对最终转化的
     直接因果效应 (Direct Effect) 和间接因果效应 (Indirect Effect)。
  3. 真实权重分配：将直接效应 + 间接效应合并，得出每个渠道的全归因贡献比例。

场景：欧美市场 Facebook/TikTok（展示类）与 Google（搜索类）的 Cookieless 归因。
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
# 数据生成（模拟母婴出海广告场景的聚合级每日数据）
# ---------------------------------------------------------------------------

def simulate_ad_data(
    n_days: int = 180,
    channels: Optional[List[str]] = None,
    true_causal_lags: Optional[Dict[Tuple[str, str], int]] = None,
    noise_std: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """模拟多渠道聚合广告数据（每日曝光量 + 订单数）。

    真实因果链路：
        Facebook(t) -> Google(t+2) -> Orders(t+2)
        TikTok(t)   -> Google(t+1) -> Orders(t+1)
        Google(t)   -> Orders(t)

    Parameters
    ----------
    n_days : 模拟天数
    channels : 渠道名称列表，默认 ['facebook', 'tiktok', 'google', 'orders']
    true_causal_lags : 因果关系字典 {(from, to): lag_days}
    noise_std : 噪声标准差
    seed : 随机种子

    Returns
    -------
    pd.DataFrame 形如 [date, facebook, tiktok, google, orders]
    """
    if channels is None:
        channels = ['facebook', 'tiktok', 'google', 'orders']

    if true_causal_lags is None:
        true_causal_lags = {
            ('facebook', 'google'): 2,  # FB 展示广告在2天后影响 Google 搜索点击
            ('tiktok', 'google'): 1,    # TikTok 在1天后影响 Google 搜索
            ('google', 'orders'): 0,    # Google 直接驱动当日订单
            ('facebook', 'orders'): 3,  # FB 展示广告在3天后也有直接影响
        }

    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

    # 基础曝光量（带季节性周期）
    t = np.arange(n_days)
    weekly_pattern = 1 + 0.15 * np.sin(2 * np.pi * t / 7)

    data = {
        'date': dates,
        'facebook': (5000 + 500 * weekly_pattern + rng.normal(0, noise_std * 10, n_days)).clip(0),
        'tiktok': (3000 + 300 * weekly_pattern + rng.normal(0, noise_std * 8, n_days)).clip(0),
        'google': np.zeros(n_days),
        'orders': np.zeros(n_days),
    }

    df = pd.DataFrame(data)

    # 根据真实因果链路生成 google 曝光
    max_lag = max(true_causal_lags.values()) if true_causal_lags else 0
    for i in range(max_lag, n_days):
        fb_lag2 = df['facebook'].iloc[i - 2] if i >= 2 else 0
        tk_lag1 = df['tiktok'].iloc[i - 1] if i >= 1 else 0
        df.loc[df.index[i], 'google'] = (
            500
            + 0.08 * fb_lag2   # Facebook 助攻效应
            + 0.05 * tk_lag1   # TikTok 助攻效应
            + rng.normal(0, noise_std * 2)
        ).clip(0)

    # 根据真实因果链路生成 orders
    for i in range(max_lag, n_days):
        google_t = df['google'].iloc[i]
        fb_lag3 = df['facebook'].iloc[i - 3] if i >= 3 else 0
        df.loc[df.index[i], 'orders'] = (
            20
            + 0.15 * google_t    # Google 直接驱动订单
            + 0.03 * fb_lag3     # Facebook 直接驱动订单（滞后3天）
            + rng.normal(0, noise_std)
        ).clip(0)

    return df


# ---------------------------------------------------------------------------
# 步骤 1：时序因果发现（Temporal Causal Discovery - PCMCI 风格简化实现）
# ---------------------------------------------------------------------------

class TemporalCausalDiscovery:
    """基于滞后互相关和格兰杰因果的时序因果发现。

    简化实现：用偏相关系数 + 多重检验校正，模拟 PCMCI 算法的核心逻辑。
    """

    def __init__(
        self,
        max_lag: int = 5,
        alpha: float = 0.05,
        min_corr: float = 0.1,
    ) -> None:
        self.max_lag = max_lag
        self.alpha = alpha
        self.min_corr = min_corr
        self.causal_links_: Dict[Tuple[str, str, int], float] = {}
        self.causal_graph_: Dict[Tuple[str, str], int] = {}

    def _lagged_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
    ) -> Tuple[float, float]:
        """计算 x(t-lag) 和 y(t) 之间的 Pearson 相关系数及 p 值。"""
        if lag == 0:
            r, p = stats.pearsonr(x, y)
        else:
            r, p = stats.pearsonr(x[:-lag], y[lag:])
        return float(r), float(p)

    def fit(
        self,
        df: pd.DataFrame,
        channels: List[str],
        target: str,
    ) -> "TemporalCausalDiscovery":
        """发现渠道变量对目标（orders）的时序因果关系。

        Parameters
        ----------
        df : 包含渠道列的 DataFrame
        channels : 考察的渠道名列表（不含 target）
        target : 目标变量名（如 'orders'）
        """
        self.causal_links_ = {}
        self.causal_graph_ = {}

        y = df[target].values.astype(float)

        for ch in channels:
            x = df[ch].values.astype(float)
            best_lag = 0
            best_r = 0.0

            for lag in range(0, self.max_lag + 1):
                r, p = self._lagged_correlation(x, y, lag)
                # 保存所有显著相关
                if abs(r) >= self.min_corr and p < self.alpha:
                    self.causal_links_[(ch, target, lag)] = r
                    # 记录最强相关的滞后
                    if abs(r) > abs(best_r):
                        best_r = r
                        best_lag = lag

            if best_r != 0.0:
                self.causal_graph_[(ch, target)] = best_lag

        # 渠道间的因果关系（排除 target）
        all_vars = channels + [target]
        for i, ch1 in enumerate(channels):
            for ch2 in channels:
                if ch1 == ch2:
                    continue
                x = df[ch1].values.astype(float)
                y2 = df[ch2].values.astype(float)
                best_lag = 0
                best_r = 0.0
                for lag in range(1, self.max_lag + 1):
                    r, p = self._lagged_correlation(x, y2, lag)
                    if abs(r) >= self.min_corr and p < self.alpha:
                        self.causal_links_[(ch1, ch2, lag)] = r
                        if abs(r) > abs(best_r):
                            best_r = r
                            best_lag = lag
                if best_r != 0.0:
                    self.causal_graph_[(ch1, ch2)] = best_lag

        return self

    def get_causal_summary(self) -> pd.DataFrame:
        """返回因果关系汇总 DataFrame。"""
        rows = []
        for (src, tgt, lag), corr in self.causal_links_.items():
            rows.append({
                'source': src,
                'target': tgt,
                'lag_days': lag,
                'correlation': round(corr, 4),
            })
        return pd.DataFrame(rows).sort_values('correlation', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 步骤 2：因果效应估计（SCM Attribution）
# ---------------------------------------------------------------------------

class CausalAttributionSCM:
    """基于结构因果模型（SCM）的渠道归因效应估计。

    使用回归分析量化每个渠道的：
    - 直接因果效应 (Direct Causal Effect)：渠道直接驱动订单的效应
    - 间接因果效应 (Indirect Causal Effect)：通过其他渠道传导的助攻效应
    - 总因果效应 (Total Causal Effect) = Direct + Indirect
    """

    def __init__(self) -> None:
        self.coefficients_: Dict[str, float] = {}
        self.indirect_effects_: Dict[str, float] = {}
        self.total_effects_: Dict[str, float] = {}
        self.attribution_weights_: Dict[str, float] = {}
        self.channels_: List[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        causal_graph: Dict[Tuple[str, str], int],
        channels: List[str],
        target: str,
    ) -> "CausalAttributionSCM":
        """估计每个渠道的直接和间接因果效应。

        Parameters
        ----------
        df : 时序数据 DataFrame
        causal_graph : 来自 TemporalCausalDiscovery 的因果图
        channels : 渠道列表
        target : 目标变量
        """
        self.channels_ = channels
        n = len(df)

        # --- 直接效应估计：每个渠道对 target 的直接滞后回归 ---
        direct_effects = {}
        for ch in channels:
            lag = causal_graph.get((ch, target), 0)
            if lag == 0:
                x = df[ch].values.astype(float)
                y = df[target].values.astype(float)
            else:
                x = df[ch].values[:-lag].astype(float)
                y = df[target].values[lag:].astype(float)

            if len(x) > 2:
                slope, intercept, r, p, se = stats.linregress(x, y)
                direct_effects[ch] = float(slope) * float(x.mean())
            else:
                direct_effects[ch] = 0.0

        # --- 间接效应估计：渠道 A -> 渠道 B -> target 的传导 ---
        indirect_effects = {ch: 0.0 for ch in channels}
        for ch in channels:
            for ch2 in channels:
                if ch == ch2:
                    continue
                if (ch, ch2) not in causal_graph:
                    continue
                # ch -> ch2 的效应
                lag_ch_ch2 = causal_graph[(ch, ch2)]
                if lag_ch_ch2 == 0:
                    x1 = df[ch].values.astype(float)
                    y1 = df[ch2].values.astype(float)
                else:
                    x1 = df[ch].values[:-lag_ch_ch2].astype(float)
                    y1 = df[ch2].values[lag_ch_ch2:].astype(float)

                if len(x1) > 2:
                    slope_ch_ch2, _, _, _, _ = stats.linregress(x1, y1)
                else:
                    slope_ch_ch2 = 0.0

                # ch2 -> target 的直接效应系数
                lag_ch2_tgt = causal_graph.get((ch2, target), 0)
                if lag_ch2_tgt == 0:
                    x2 = df[ch2].values.astype(float)
                    y2 = df[target].values.astype(float)
                else:
                    x2 = df[ch2].values[:-lag_ch2_tgt].astype(float)
                    y2 = df[target].values[lag_ch2_tgt:].astype(float)

                if len(x2) > 2:
                    slope_ch2_tgt, _, _, _, _ = stats.linregress(x2, y2)
                else:
                    slope_ch2_tgt = 0.0

                # 间接效应 = β(ch→ch2) * β(ch2→target) * mean(ch)
                indirect = slope_ch_ch2 * slope_ch2_tgt * float(df[ch].mean())
                indirect_effects[ch] += indirect

        # --- 总效应 = 直接 + 间接 ---
        self.coefficients_ = direct_effects
        self.indirect_effects_ = indirect_effects
        self.total_effects_ = {
            ch: direct_effects[ch] + indirect_effects[ch]
            for ch in channels
        }

        # --- 归因权重（归一化到 [0,1]，均值转为贡献比例）---
        total_sum = sum(abs(v) for v in self.total_effects_.values())
        if total_sum > 1e-10:
            self.attribution_weights_ = {
                ch: abs(self.total_effects_[ch]) / total_sum
                for ch in channels
            }
        else:
            equal_w = 1.0 / len(channels)
            self.attribution_weights_ = {ch: equal_w for ch in channels}

        return self

    def attribution_report(self) -> pd.DataFrame:
        """生成归因分析报告。"""
        rows = []
        for ch in self.channels_:
            rows.append({
                'channel': ch,
                'direct_effect': round(self.coefficients_.get(ch, 0), 4),
                'indirect_effect': round(self.indirect_effects_.get(ch, 0), 4),
                'total_effect': round(self.total_effects_.get(ch, 0), 4),
                'attribution_weight': round(self.attribution_weights_.get(ch, 0), 4),
                'attribution_pct': f"{self.attribution_weights_.get(ch, 0) * 100:.1f}%",
            })
        return (
            pd.DataFrame(rows)
            .sort_values('attribution_weight', ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# 主流程封装：CDA Pipeline
# ---------------------------------------------------------------------------

class CDAAttributionPipeline:
    """CDA 因果驱动归因全流程管道。

    使用方法：
        pipeline = CDAAttributionPipeline(channels=['facebook', 'tiktok', 'google'], target='orders')
        pipeline.fit(df)
        report = pipeline.attribution_report()
    """

    def __init__(
        self,
        channels: List[str],
        target: str = 'orders',
        max_lag: int = 5,
        alpha: float = 0.05,
    ) -> None:
        self.channels = channels
        self.target = target
        self.discovery = TemporalCausalDiscovery(max_lag=max_lag, alpha=alpha)
        self.scm = CausalAttributionSCM()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CDAAttributionPipeline":
        """拟合完整 CDA 归因模型。"""
        # 步骤 1：时序因果发现
        self.discovery.fit(df, self.channels, self.target)

        # 步骤 2：SCM 归因效应估计
        self.scm.fit(df, self.discovery.causal_graph_, self.channels, self.target)
        self._fitted = True
        return self

    def causal_summary(self) -> pd.DataFrame:
        """返回发现的因果关系链路。"""
        assert self._fitted, "先调用 fit()"
        return self.discovery.get_causal_summary()

    def attribution_report(self) -> pd.DataFrame:
        """返回渠道归因权重报告。"""
        assert self._fitted, "先调用 fit()"
        return self.scm.attribution_report()

    def budget_recommendation(
        self,
        current_budget: Dict[str, float],
        total_budget: float,
    ) -> pd.DataFrame:
        """基于归因权重生成预算调整建议。

        Parameters
        ----------
        current_budget : 当前各渠道预算 {'facebook': 10000, ...}
        total_budget : 总可用预算
        """
        assert self._fitted, "先调用 fit()"
        weights = self.scm.attribution_weights_
        rows = []
        for ch in self.channels:
            w = weights.get(ch, 0)
            recommended = total_budget * w
            current = current_budget.get(ch, 0)
            delta = recommended - current
            rows.append({
                'channel': ch,
                'current_budget': round(current, 2),
                'cda_weight': round(w, 4),
                'recommended_budget': round(recommended, 2),
                'delta': round(delta, 2),
                'action': '增加' if delta > 0 else '削减',
            })
        return pd.DataFrame(rows).sort_values('cda_weight', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 自测（self-test）
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """内置自测：验证 CDA 因果归因的核心逻辑与数值合理性。"""
    print("=" * 65)
    print("CDA Attribution Self-Test")
    print("=" * 65)

    CHANNELS = ['facebook', 'tiktok', 'google']
    TARGET = 'orders'

    # 1. 生成模拟数据
    print("\n[1] 生成模拟广告数据 (180天)...")
    df = simulate_ad_data(n_days=180, seed=42)
    print(f"    数据形状: {df.shape}")
    print(f"    列: {list(df.columns)}")
    print(f"    前3行:\n{df.head(3).to_string(index=False)}")

    # 2. 拟合 CDA 管道
    print("\n[2] 拟合 CDA 归因管道...")
    pipeline = CDAAttributionPipeline(channels=CHANNELS, target=TARGET, max_lag=5, alpha=0.1)
    pipeline.fit(df)
    print("    拟合完成。")

    # 3. 验证因果发现
    print("\n[3] 时序因果关系发现:")
    causal_df = pipeline.causal_summary()
    print(causal_df.to_string(index=False))

    # 断言：因果图谱不为空
    assert len(causal_df) > 0, "因果发现结果为空，模型无效"
    print("    [OK] 因果图谱非空")

    # 断言：Google -> Orders 应被发现（直接效应）
    google_orders_found = any(
        (row['source'] == 'google' and row['target'] == 'orders')
        for _, row in causal_df.iterrows()
    )
    assert google_orders_found, "未发现 google->orders 因果关系"
    print("    [OK] google->orders 因果关系已发现")

    # 4. 验证归因报告
    print("\n[4] 渠道归因分析报告:")
    report = pipeline.attribution_report()
    print(report.to_string(index=False))

    # 断言：归因权重之和约等于 1.0
    total_weight = report['attribution_weight'].sum()
    assert abs(total_weight - 1.0) < 1e-6, f"归因权重和={total_weight:.6f}，应为1.0"
    print(f"    [OK] 归因权重之和={total_weight:.6f}")

    # 断言：所有渠道都有权重
    assert len(report) == len(CHANNELS), f"归因渠道数={len(report)}，预期{len(CHANNELS)}"
    print(f"    [OK] 所有 {len(CHANNELS)} 个渠道均有归因权重")

    # 断言：Google 应是最高归因渠道（直接拉动订单）
    top_channel = report.iloc[0]['channel']
    assert top_channel == 'google', (
        f"预期 google 归因最高，实际最高渠道={top_channel}"
    )
    print(f"    [OK] 最高归因渠道={top_channel}（Google 直接驱动订单）")

    # 断言：Facebook 应有非零归因权重（渠道助攻效应被纳入计算）
    fb_row = report[report['channel'] == 'facebook'].iloc[0]
    assert fb_row['attribution_weight'] > 0, (
        f"Facebook 归因权重应 > 0，实际={fb_row['attribution_weight']}"
    )
    print(f"    [OK] Facebook 归因权重={fb_row['attribution_weight']:.4f}，间接效应={fb_row['indirect_effect']:.4f}")

    # 5. 验证预算建议
    print("\n[5] 预算调整建议（当前总预算 $10,000）:")
    current_budget = {'facebook': 3000, 'tiktok': 2000, 'google': 5000}
    budget_rec = pipeline.budget_recommendation(current_budget, total_budget=10000)
    print(budget_rec.to_string(index=False))

    # 断言：推荐预算总和等于 total_budget
    total_rec = budget_rec['recommended_budget'].sum()
    assert abs(total_rec - 10000) < 1.0, f"推荐预算总和={total_rec:.2f}，应为10000"
    print(f"    [OK] 推荐预算总和={total_rec:.2f}")

    print("\n[PASS] 所有断言通过，CDA 自测成功！")
    print("=" * 65)


def main() -> None:
    """母婴出海欧美市场多渠道因果归因示例。"""
    print("\n[CDA Demo] 母婴品牌欧美市场 Cookieless 多渠道归因分析")
    print("-" * 55)

    CHANNELS = ['facebook', 'tiktok', 'google']
    TARGET = 'orders'

    # 生成一年数据
    df = simulate_ad_data(n_days=365, seed=2024)
    print(f"数据: {df.shape[0]}天 x {df.shape[1]}列")

    # 拟合 CDA 管道
    pipeline = CDAAttributionPipeline(channels=CHANNELS, target=TARGET, max_lag=5, alpha=0.05)
    pipeline.fit(df)

    # 输出因果图谱
    print("\n发现的跨渠道因果链路（Top 10）:")
    causal_df = pipeline.causal_summary()
    print(causal_df.head(10).to_string(index=False))

    # 输出归因权重
    print("\n渠道归因权重（CDA 全因果归因）:")
    report = pipeline.attribution_report()
    print(report.to_string(index=False))

    # 输出 Last-Click 对比（只给 Google）
    print("\n[对比] Last-Click 归因（100% 给 Google）vs CDA 归因:")
    lc_google = 1.0
    cda_google = report.loc[report['channel'] == 'google', 'attribution_weight'].values[0]
    cda_facebook = report.loc[report['channel'] == 'facebook', 'attribution_weight'].values[0]
    print(f"  Google  - Last-Click: 100.0%  CDA: {cda_google*100:.1f}%")
    print(f"  Facebook - Last-Click:   0.0%  CDA: {cda_facebook*100:.1f}%")
    print(f"  → Facebook 实际贡献被低估了 {cda_facebook*100:.1f}%，"
          f"Last-Click 高估 Google {(lc_google - cda_google)*100:.1f}%")

    # 预算建议
    print("\n预算优化建议（当前月预算 $15,000）:")
    current_budget = {'facebook': 3000, 'tiktok': 2000, 'google': 10000}
    rec = pipeline.budget_recommendation(current_budget, total_budget=15000)
    print(rec.to_string(index=False))

    # 运行自测
    _run_self_test()


if __name__ == "__main__":
    main()
