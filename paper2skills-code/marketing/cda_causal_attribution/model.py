"""
CDA — 因果驱动归因（隐私保护多渠道归因）
论文：Causal-driven attribution (CDA): Estimating channel influence without user-level data
arXiv：2512.21211 | 2024年12月
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import statistics
from collections import defaultdict


# ──────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────

@dataclass
class ChannelTimeSeries:
    """单渠道时序数据（汇总级，无用户ID）"""
    channel_name: str
    daily_impressions: list[float]  # 每日曝光量
    daily_conversions: list[float]  # 每日转化量（汇总，非用户级）

    def __post_init__(self):
        assert len(self.daily_impressions) == len(self.daily_conversions), \
            "曝光量和转化量时序长度必须一致"

    @property
    def n_days(self) -> int:
        return len(self.daily_impressions)

    def impression_rate(self) -> list[float]:
        """归一化曝光率（0-1）"""
        max_imp = max(self.daily_impressions) or 1.0
        return [x / max_imp for x in self.daily_impressions]


@dataclass
class CausalDAG:
    """因果有向无环图"""
    channels: list[str]
    edges: list[tuple[str, str, int]]  # (from_channel, to_channel, lag_days)
    edge_weights: dict[tuple[str, str], float] = field(default_factory=dict)

    def get_parents(self, channel: str) -> list[tuple[str, int]]:
        """获取某渠道的所有父节点（直接因果来源）"""
        return [(src, lag) for src, dst, lag in self.edges if dst == channel]

    def get_children(self, channel: str) -> list[tuple[str, int]]:
        """获取某渠道的所有子节点（直接因果影响）"""
        return [(dst, lag) for src, dst, lag in self.edges if src == channel]


# ──────────────────────────────────────────────
# 简化版 PCMCI 因果发现
# ──────────────────────────────────────────────

class PCMCICausalDiscovery:
    """
    简化版 PCMCI：格兰杰因果 + 条件独立性检验
    完整版使用 tigramite 库的 PCMCI 类（pip install tigramite）
    """

    def __init__(self, max_lag: int = 3, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha = alpha  # 显著性阈值

    @staticmethod
    def _correlation(x: list[float], y: list[float]) -> float:
        """皮尔逊相关系数"""
        n = len(x)
        if n < 3:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        if den_x == 0 or den_y == 0:
            return 0.0
        return num / (den_x * den_y)

    def _granger_test(
        self,
        cause_series: list[float],
        effect_series: list[float],
        lag: int,
    ) -> tuple[float, bool]:
        """
        简化格兰杰因果检验
        返回 (相关系数, 是否存在显著因果关系)
        完整版使用 statsmodels.tsa.stattools.grangercausalitytests
        """
        if len(cause_series) <= lag:
            return 0.0, False

        x_lagged = cause_series[:-lag] if lag > 0 else cause_series
        y_target = effect_series[lag:] if lag > 0 else effect_series

        min_len = min(len(x_lagged), len(y_target))
        if min_len < 5:
            return 0.0, False

        corr = self._correlation(x_lagged[:min_len], y_target[:min_len])
        # 简化显著性：|r| > 0.3 视为显著（完整版用 Fisher z-test, p < alpha）
        return corr, abs(corr) > 0.3

    def discover_dag(self, channel_data: list[ChannelTimeSeries]) -> CausalDAG:
        """从多渠道时序数据中发现因果 DAG"""
        channels = [ch.channel_name for ch in channel_data]
        edges: list[tuple[str, str, int]] = []
        edge_weights: dict[tuple[str, str], float] = {}

        for i, ch_cause in enumerate(channel_data):
            for j, ch_effect in enumerate(channel_data):
                if i == j:
                    continue
                best_corr, best_lag = 0.0, 0
                for lag in range(1, self.max_lag + 1):
                    corr, significant = self._granger_test(
                        ch_cause.impression_rate(),
                        ch_effect.impression_rate(),
                        lag,
                    )
                    if significant and abs(corr) > abs(best_corr):
                        best_corr, best_lag = corr, lag

                if best_lag > 0:
                    edges.append((ch_cause.channel_name, ch_effect.channel_name, best_lag))
                    edge_weights[(ch_cause.channel_name, ch_effect.channel_name)] = best_corr

        return CausalDAG(channels=channels, edges=edges, edge_weights=edge_weights)


# ──────────────────────────────────────────────
# CDA 归因器（SCM 层）
# ──────────────────────────────────────────────

class CDAAttributor:
    """
    因果驱动归因：基于结构因果模型（SCM）估计各渠道的直接/间接贡献
    核心公式：ATE_i = E[Y | do(X_i = x_i + δ)] - E[Y | do(X_i = x_i)]
    """

    def __init__(self):
        self._channel_data: list[ChannelTimeSeries] = []
        self._dag: Optional[CausalDAG] = None
        self._channel_weights: dict[str, float] = {}

    def fit(self, channel_data: list[ChannelTimeSeries], dag: CausalDAG) -> None:
        """拟合归因模型"""
        self._channel_data = channel_data
        self._dag = dag
        self._channel_weights = self._estimate_weights()

    def _estimate_weights(self) -> dict[str, float]:
        """
        SCM 权重估计：直接效应 + 间接效应
        直接效应：渠道曝光与总转化的 Pearson 相关
        间接效应：通过 DAG 中介节点的贡献（折扣系数 0.3）
        """
        weights: dict[str, float] = {}

        for ch in self._channel_data:
            # 计算当日总转化（跨渠道汇总）
            total_conv = [
                sum(c.daily_conversions[d] for c in self._channel_data if d < c.n_days)
                for d in range(ch.n_days)
            ]

            corr = self._pearson(ch.impression_rate(), total_conv)
            direct_effect = max(0.0, corr)

            # 间接效应：通过子节点路径
            indirect_effect = 0.0
            if self._dag:
                for child_name, _ in self._dag.get_children(ch.channel_name):
                    w = self._dag.edge_weights.get((ch.channel_name, child_name), 0.0)
                    indirect_effect += abs(w) * 0.3

            weights[ch.channel_name] = direct_effect + indirect_effect

        # 归一化（权重之和=1）
        total = sum(weights.values()) or 1.0
        return {ch: w / total for ch, w in weights.items()}

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        n = min(len(x), len(y))
        if n < 3:
            return 0.0
        x, y = x[:n], y[:n]
        mean_x, mean_y = sum(x) / n, sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = math.sqrt(
            sum((xi - mean_x) ** 2 for xi in x) *
            sum((yi - mean_y) ** 2 for yi in y)
        )
        return num / den if den > 0 else 0.0

    def attribute_conversions(self) -> dict[str, float]:
        """返回各渠道归因权重（之和=1.0）"""
        if not self._channel_weights:
            raise RuntimeError("请先调用 fit() 方法")
        return dict(self._channel_weights)

    def intervention_analysis(self, channel: str, pct_change: float) -> float:
        """
        CATE 干预估计：渠道预算变化 pct_change% 对每日转化的影响
        pct_change: 0.1 表示 +10%
        返回：每日转化变化量
        """
        if channel not in self._channel_weights:
            raise ValueError(f"未知渠道: {channel}")

        avg_daily_conv = sum(
            sum(ch.daily_conversions) / ch.n_days for ch in self._channel_data
        )
        channel_daily_conv = avg_daily_conv * self._channel_weights[channel]
        return round(channel_daily_conv * pct_change, 2)

    def channel_ranking(self) -> list[tuple[str, float]]:
        """按归因权重降序排列渠道"""
        return sorted(self.attribute_conversions().items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────
# 测试：3 渠道 30 天归因
# ──────────────────────────────────────────────

def test_cda_attribution() -> None:
    """
    测试：Google + Meta + TikTok 三渠道，30天数据
    验证：权重之和=1，渠道排名合理（Google 曝光最大应排名第一）
    """
    import random
    random.seed(42)

    n_days = 30

    google_imp = [50000 + random.randint(-2000, 2000) for _ in range(n_days)]
    meta_imp = [30000 + random.randint(-1500, 1500) for _ in range(n_days)]
    tiktok_imp = [20000 + random.randint(-1000, 1000) for _ in range(n_days)]

    # 转化量由各渠道共同驱动（Google 权重最高）
    total_conv = [
        int(g * 0.002 + m * 0.001 + t * 0.0008 + random.randint(0, 5))
        for g, m, t in zip(google_imp, meta_imp, tiktok_imp)
    ]
    channel_conv = [c / 3 for c in total_conv]

    channels = [
        ChannelTimeSeries("Google", google_imp, channel_conv),
        ChannelTimeSeries("Meta", meta_imp, channel_conv),
        ChannelTimeSeries("TikTok", tiktok_imp, channel_conv),
    ]

    print("=== 步骤1：PCMCI 因果发现 ===")
    discoverer = PCMCICausalDiscovery(max_lag=3)
    dag = discoverer.discover_dag(channels)
    print(f"发现因果边：{len(dag.edges)} 条")
    for src, dst, lag in dag.edges:
        w = dag.edge_weights.get((src, dst), 0.0)
        print(f"  {src} → {dst} (lag={lag}天, r={w:.3f})")

    print("\n=== 步骤2：CDA 归因 ===")
    attributor = CDAAttributor()
    attributor.fit(channels, dag)
    weights = attributor.attribute_conversions()

    for ch, w in attributor.channel_ranking():
        print(f"  {ch}: {w*100:.1f}%")

    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6, f"权重之和应为1，实际={total_weight}"
    print(f"\n✅ 权重之和 = {total_weight:.6f}")

    print("\n=== 步骤3：CATE 干预分析 ===")
    for ch_name in ["Google", "Meta", "TikTok"]:
        delta = attributor.intervention_analysis(ch_name, 0.1)
        print(f"  {ch_name} 预算 +10% → 每日转化 +{delta:.2f}")

    ranking = attributor.channel_ranking()
    assert ranking[0][0] == "Google", f"Google 应排名第一，实际={ranking[0][0]}"
    print(f"\n渠道排名：{[ch for ch, _ in ranking]}")
    print("✅ CDA 归因全部测试通过")


if __name__ == "__main__":
    test_cda_attribution()
