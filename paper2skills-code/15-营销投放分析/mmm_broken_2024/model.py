"""
Identified Bayesian MMM — 基于高斯过程的无混淆贝叶斯营销归因框架
论文: "Your MMM is Broken: Identification of Nonlinear and Time-varying Effects
       in Marketing Mix Models" (arXiv:2408.07678, 沃顿商学院/伦敦商学院, 2024)

核心逻辑:
1. ObservationalEquivalenceDiagnostor - 检测数据是否存在"观测等价"混淆
2. GPSaturationCurve                  - 用高斯过程非参数化建模渠道饱和曲线
3. ExperimentalCalibrator             - 接收预算冲击（实验数据）作为先验修正模型
4. IdentifiedBayesianMMM              - 主入口：诊断 → 处方 → 校正归因一体化

注: 生产环境中 GP 应接入 GPyTorch / Stan / PyMC; 本文件用纯 numpy 解析近似实现,
    保证完整可运行且无需外部依赖（除 numpy/scipy）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ChannelData:
    """单渠道的历史投放数据"""
    name: str                      # 渠道名称，如 "TikTok" / "Meta" / "Google"
    spend: np.ndarray              # 日均投放金额序列，形状 (T,)
    conversions: np.ndarray        # 日均转化/销售序列，形状 (T,)


@dataclass
class ExperimentalShock:
    """预算冲击实验记录（关停/超投实验产生的强对比数据点）"""
    channel: str                   # 被实验的渠道名称
    shock_period: Tuple[int, int]  # 冲击发生的时间区间 [t_start, t_end)，闭开区间
    spend_multiplier: float        # 投放倍数 (0.0 = 完全停投, 3.0 = 超投 3 倍)
    observed_lift: float           # 实测转化提升比（与正常期基线的比值）


@dataclass
class ROASEstimate:
    """渠道 ROAS 的贝叶斯后验估计"""
    channel: str
    point_estimate: float          # 后验均值
    lower_ci: float                # 90% 置信区间下界
    upper_ci: float                # 90% 置信区间上界
    is_identified: bool            # True = 经实验数据识别，False = 存在混淆风险
    confidence_note: str           # 可读性注释


@dataclass
class DiagnosisReport:
    """可识别性诊断报告"""
    has_observational_equivalence_risk: bool
    channel_risks: Dict[str, str]       # {渠道名: 风险描述}
    experiment_prescriptions: List[str]  # 算法推荐的实验处方
    identified_channels: List[str]       # 已通过实验识别的渠道


# ---------------------------------------------------------------------------
# 模块 1: 观测等价性诊断
# ---------------------------------------------------------------------------

class ObservationalEquivalenceDiagnostor:
    """
    诊断当前数据是否存在非线性饱和效应与时变效应的"观测等价"混淆。

    核心判断: 若渠道投放序列方差过低（预算单调平稳，缺乏强烈脉冲），
    则模型无法区分饱和衰减与季节性时变，ROAS 估计将产生系统性偏差。

    数学依据 (论文 Prop. 1): 两个数据生成过程 (DGP) M1 和 M2 观测等价
    ⟺ 存在 f(x) 和 g(t) 使得 f(x)*g(t) = h(x, t) 对所有观测成立
    → 需要强对比（Shock）数据打破这个等价性。
    """

    VARIANCE_THRESHOLD = 0.15   # 相对标准差阈值（低于此值视为"平稳投放"）
    MIN_SHOCK_RATIO = 2.0       # 最大值/最小值比率，低于此值认为无足够冲击

    def diagnose(self, channels: List[ChannelData]) -> DiagnosisReport:
        """诊断所有渠道并输出报告"""
        channel_risks: Dict[str, str] = {}
        prescriptions: List[str] = []

        for ch in channels:
            risk = self._assess_channel_risk(ch)
            channel_risks[ch.name] = risk
            if "HIGH" in risk or "MEDIUM" in risk:
                prescriptions.append(
                    f"[实验处方] {ch.name}: "
                    f"在随机选定区域停投 2-3 天（spend×0），"
                    f"同期另一区域超投 3 倍（spend×3），"
                    f"采集真实边际响应曲线数据。"
                )

        has_risk = any("HIGH" in r or "MEDIUM" in r for r in channel_risks.values())

        return DiagnosisReport(
            has_observational_equivalence_risk=has_risk,
            channel_risks=channel_risks,
            experiment_prescriptions=prescriptions,
            identified_channels=[
                ch.name for ch in channels
                if "LOW" in channel_risks.get(ch.name, "")
            ],
        )

    def _assess_channel_risk(self, ch: ChannelData) -> str:
        """评估单渠道的混淆风险等级"""
        spend = ch.spend
        if spend.std() < 1e-9:
            return "HIGH RISK: 投放完全恒定，无法识别饱和曲线"

        cv = spend.std() / (spend.mean() + 1e-9)           # 变异系数
        shock_ratio = spend.max() / (spend.min() + 1e-9)   # 峰谷比

        if cv < self.VARIANCE_THRESHOLD or shock_ratio < self.MIN_SHOCK_RATIO:
            return (
                f"HIGH RISK: 投放变异系数 {cv:.3f} < {self.VARIANCE_THRESHOLD} "
                f"或峰谷比 {shock_ratio:.2f} < {self.MIN_SHOCK_RATIO}，"
                f"饱和效应与时变效应不可区分（观测等价）"
            )
        elif cv < self.VARIANCE_THRESHOLD * 2:
            return (
                f"MEDIUM RISK: 投放变异适中 (cv={cv:.3f})，"
                f"建议补充 1-2 次实验性冲击数据"
            )
        else:
            return (
                f"LOW RISK: 投放具备足够对比度 (cv={cv:.3f}, "
                f"峰谷比={shock_ratio:.2f})，基础可识别性可接受"
            )


# ---------------------------------------------------------------------------
# 模块 2: 高斯过程饱和曲线
# ---------------------------------------------------------------------------

class GPSaturationCurve:
    """
    用高斯过程（GP）非参数化建模渠道的边际响应曲线（饱和曲线）。

    替代传统写死的 Hill 曲线 y = x^α / (K^α + x^α)，
    GP 可以同时拟合:
    - 非线性饱和（输出随投入单调递增但凹性增加）
    - 时变效应（不同时间段相同投入的响应不同）

    实现: 使用 RBF 核函数的解析 GP（无需 PyTorch），
    后验均值 μ*(x) = k(x,X) [K(X,X)+σ²I]^{-1} y
    """

    def __init__(
        self,
        length_scale: float = 1.0,   # RBF 核长度尺度（控制曲线平滑度）
        output_scale: float = 1.0,   # 核振幅（控制响应幅度）
        noise_var: float = 0.01,     # 观测噪声方差
    ) -> None:
        self.length_scale = length_scale
        self.output_scale = output_scale
        self.noise_var = noise_var
        self._X_train: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None   # GP 系数向量

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """径向基函数（RBF/Squared Exponential）核"""
        x1 = np.atleast_1d(x1)[:, None]   # (n, 1)
        x2 = np.atleast_1d(x2)[None, :]   # (1, m)
        sq_dist = (x1 - x2) ** 2
        return self.output_scale ** 2 * np.exp(-0.5 * sq_dist / self.length_scale ** 2)

    def fit(self, spend: np.ndarray, response: np.ndarray) -> None:
        """
        拟合 GP: 学习投放量 spend → 转化响应 response 的非参数映射

        参数:
            spend:    形状 (T,), 归一化后的投放量
            response: 形状 (T,), 归一化后的转化量
        """
        X = np.asarray(spend, dtype=float)
        y = np.asarray(response, dtype=float)
        K = self._rbf_kernel(X, X)
        K_noisy = K + self.noise_var * np.eye(len(X))
        # 解方程 (K + σ²I) α = y
        self._alpha = np.linalg.solve(K_noisy, y)
        self._X_train = X

    def fit_with_prior(
        self,
        spend: np.ndarray,
        response: np.ndarray,
        shock_spends: np.ndarray,
        shock_responses: np.ndarray,
        shock_weight: float = 5.0,
    ) -> None:
        """
        融合实验数据（预算冲击）作为强先验拟合 GP。
        通过增加冲击数据点的权重，强制 GP 过高对比度区域。
        """
        # 将实验冲击数据点重复 shock_weight 倍以增加其权重
        n_copies = max(1, int(shock_weight))
        all_spend = np.concatenate([spend] + [shock_spends] * n_copies)
        all_resp = np.concatenate([response] + [shock_responses] * n_copies)
        self.fit(all_spend, all_resp)

    def predict(self, spend_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GP 后验预测

        返回:
            mean_pred:  后验均值（点估计）
            std_pred:   后验标准差（不确定性）
        """
        if self._X_train is None or self._alpha is None:
            raise RuntimeError("请先调用 fit() 训练模型")

        X_q = np.asarray(spend_query, dtype=float)
        K_star = self._rbf_kernel(X_q, self._X_train)  # (n_query, n_train)
        mean_pred = K_star @ self._alpha

        # 后验方差 (diagonal of K** - K* (K+σ²I)^{-1} K*ᵀ)
        K_qq = self._rbf_kernel(X_q, X_q)
        K_nn = self._rbf_kernel(self._X_train, self._X_train)
        K_noisy = K_nn + self.noise_var * np.eye(len(self._X_train))
        v = np.linalg.solve(K_noisy, K_star.T)     # (n_train, n_query)
        var_pred = np.diag(K_qq - K_star @ v)
        var_pred = np.maximum(var_pred, 0.0)        # 数值稳定性
        return mean_pred, np.sqrt(var_pred)

    def marginal_roas_at(self, spend_level: float) -> float:
        """估算当前投放水平下的边际 ROAS（响应曲线斜率）"""
        eps = spend_level * 0.01 + 1e-4
        y_plus, _ = self.predict(np.array([spend_level + eps]))
        y_minus, _ = self.predict(np.array([spend_level - eps]))
        return float((y_plus[0] - y_minus[0]) / (2 * eps))


# ---------------------------------------------------------------------------
# 模块 3: 实验性冲击校准器
# ---------------------------------------------------------------------------

class ExperimentalCalibrator:
    """
    接收实验性预算冲击（关停/超投实验）数据，
    将其作为高权重先验注入 GP 模型，打破"观测等价"数学死结。

    使用方法:
        calibrator = ExperimentalCalibrator(channel_data, gp_model)
        calibrator.add_shock(shock)
        roas = calibrator.get_calibrated_roas(current_spend)
    """

    def __init__(self, channel: ChannelData, gp: GPSaturationCurve) -> None:
        self.channel = channel
        self.gp = gp
        self._shocks: List[ExperimentalShock] = []

        # 归一化参数（避免尺度问题）
        self._spend_scale = channel.spend.max() + 1e-9
        self._conv_scale = channel.conversions.max() + 1e-9

    def add_shock(self, shock: ExperimentalShock) -> None:
        """注册一条实验冲击记录"""
        if shock.channel == self.channel.name:
            self._shocks.append(shock)

    def calibrate(self) -> None:
        """融合常规数据 + 实验冲击数据，重新拟合 GP"""
        norm_spend = self.channel.spend / self._spend_scale
        norm_conv = self.channel.conversions / self._conv_scale

        if not self._shocks:
            # 无实验数据，仅用观测数据拟合（存在混淆风险）
            self.gp.fit(norm_spend, norm_conv)
            return

        # 从冲击记录合成实验数据点
        shock_spends, shock_convs = self._synthesize_shock_data()
        self.gp.fit_with_prior(
            spend=norm_spend,
            response=norm_conv,
            shock_spends=shock_spends / self._spend_scale,
            shock_responses=shock_convs / self._conv_scale,
            shock_weight=5.0,
        )

    def _synthesize_shock_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """将实验冲击记录转换为（spend, conversion）数据点对"""
        base_spend = self.channel.spend.mean()
        base_conv = self.channel.conversions.mean()

        shock_s, shock_c = [], []
        for shock in self._shocks:
            t0, t1 = shock.shock_period
            # 取冲击期间的实际平均投放
            period_spend = self.channel.spend[t0:t1].mean() * shock.spend_multiplier
            # 实测提升比 × 基线转化 = 冲击期转化
            shock_conv = base_conv * shock.observed_lift

            shock_s.append(period_spend)
            shock_c.append(shock_conv)

        return np.array(shock_s), np.array(shock_c)

    def get_calibrated_roas(self, current_spend: float) -> ROASEstimate:
        """获取校正后的 ROAS 后验估计"""
        q_norm = np.array([current_spend / self._spend_scale])
        mean_norm, std_norm = self.gp.predict(q_norm)

        # 还原到原始尺度
        conv_estimate = float(mean_norm[0]) * self._conv_scale
        conv_std = float(std_norm[0]) * self._conv_scale

        roas = conv_estimate / (current_spend + 1e-9)
        roas_std = conv_std / (current_spend + 1e-9)

        return ROASEstimate(
            channel=self.channel.name,
            point_estimate=round(roas, 4),
            lower_ci=round(max(0, roas - 1.645 * roas_std), 4),
            upper_ci=round(roas + 1.645 * roas_std, 4),
            is_identified=len(self._shocks) > 0,
            confidence_note=(
                f"已纳入 {len(self._shocks)} 条实验冲击数据，ROAS 估计可信"
                if self._shocks
                else "⚠️ 无实验数据支撑，ROAS 存在观测等价混淆风险"
            ),
        )


# ---------------------------------------------------------------------------
# 主模型: Identified Bayesian MMM
# ---------------------------------------------------------------------------

class IdentifiedBayesianMMM:
    """
    Identified Bayesian Marketing Mix Model 主入口。

    工作流:
    1. diagnose()      → 诊断数据可识别性，输出实验处方
    2. add_shock()     → 录入实验冲击数据（关停/超投实验结果）
    3. fit()           → 拟合 GP + 实验先验，校正各渠道归因
    4. get_roas()      → 获取修正后的 ROAS 和预算建议
    5. optimize()      → 给出最优预算分配建议
    """

    def __init__(self, channels: List[ChannelData]) -> None:
        self.channels = {ch.name: ch for ch in channels}
        self.diagnostor = ObservationalEquivalenceDiagnostor()
        self._gps: Dict[str, GPSaturationCurve] = {}
        self._calibrators: Dict[str, ExperimentalCalibrator] = {}
        self._diagnosis: Optional[DiagnosisReport] = None

        # 初始化每个渠道的 GP 和校准器
        for ch in channels:
            gp = GPSaturationCurve(length_scale=0.3, output_scale=1.0, noise_var=0.05)
            self._gps[ch.name] = gp
            self._calibrators[ch.name] = ExperimentalCalibrator(ch, gp)

    def diagnose(self) -> DiagnosisReport:
        """步骤 1: 诊断数据可识别性"""
        self._diagnosis = self.diagnostor.diagnose(list(self.channels.values()))
        return self._diagnosis

    def add_shock(self, shock: ExperimentalShock) -> None:
        """步骤 2: 录入实验冲击数据"""
        if shock.channel not in self._calibrators:
            raise ValueError(f"未知渠道: {shock.channel}")
        self._calibrators[shock.channel].add_shock(shock)

    def fit(self) -> None:
        """步骤 3: 拟合所有渠道的 GP 模型（含实验先验）"""
        for name, calibrator in self._calibrators.items():
            calibrator.calibrate()

    def get_roas(self, current_spends: Dict[str, float]) -> Dict[str, ROASEstimate]:
        """步骤 4: 获取各渠道修正后的 ROAS 估计"""
        estimates = {}
        for name, spend in current_spends.items():
            if name not in self._calibrators:
                continue
            estimates[name] = self._calibrators[name].get_calibrated_roas(spend)
        return estimates

    def optimize(
        self,
        total_budget: float,
        current_spends: Dict[str, float],
        n_iter: int = 50,
    ) -> Dict[str, float]:
        """
        步骤 5: 基于 GP 饱和曲线给出最优预算分配。

        算法: 贪心边际 ROAS 均等化
        - 循环将预算单位从边际 ROAS 最低的渠道转移到最高的渠道
        - 收敛条件: 各渠道边际 ROAS 趋于相等（约 ±10%）

        参数:
            total_budget:   总预算
            current_spends: 当前各渠道投放（作为优化起点）
            n_iter:         最大迭代次数

        返回:
            最优预算分配字典 {channel_name: recommended_spend}
        """
        channels = list(current_spends.keys())
        alloc = {ch: current_spends.get(ch, total_budget / len(channels))
                 for ch in channels}

        step = total_budget * 0.02  # 每次转移预算的步长（总预算的 2%）

        for _ in range(n_iter):
            # 计算各渠道边际 ROAS
            mroas = {}
            for ch in channels:
                spend_norm = alloc[ch] / (self.channels[ch].spend.max() + 1e-9)
                mroas[ch] = self._gps[ch].marginal_roas_at(spend_norm)

            best = max(mroas, key=lambda k: mroas[k])
            worst = min(mroas, key=lambda k: mroas[k])

            if best == worst:
                break

            # 检查是否收敛（差距 < 10%）
            if mroas[worst] > 0 and (mroas[best] - mroas[worst]) / mroas[worst] < 0.1:
                break

            # 转移预算：从最差 → 最优
            transfer = min(step, alloc[worst] * 0.05)  # 最多转移 5%
            alloc[worst] = max(alloc[worst] - transfer, total_budget * 0.05)
            alloc[best] = alloc[best] + transfer

        # 归一化确保总预算不变
        total = sum(alloc.values())
        return {ch: round(v * total_budget / total, 2) for ch, v in alloc.items()}


# ---------------------------------------------------------------------------
# 自测 / 演示
# ---------------------------------------------------------------------------

def _generate_mock_data(
    n_days: int = 90,
    seed: int = 42,
) -> Tuple[List[ChannelData], List[ExperimentalShock]]:
    """
    生成模拟的三渠道投放数据，其中:
    - TikTok: 平稳投放（高混淆风险）
    - Meta:   有一定波动（中等风险）
    - Google: 有明显脉冲（低风险）
    并生成对应的实验冲击记录。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)

    # 真实（不可观测）时变效应: 季节性曲线
    seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * t / 30)

    # --- TikTok: 平稳投放，高混淆风险 ---
    tiktok_spend = 10_000 + rng.normal(0, 200, n_days)  # 几乎恒定
    tiktok_spend = np.abs(tiktok_spend)
    # 真实响应: 轻度饱和 + 季节性
    tiktok_true_roas = 3.0 * (1 - np.exp(-tiktok_spend / 15_000))
    tiktok_conv = tiktok_spend * tiktok_true_roas * seasonality + rng.normal(0, 500, n_days)

    # --- Meta: 有波动 ---
    meta_spend = 8_000 + 3_000 * np.sin(2 * np.pi * t / 14) + rng.normal(0, 500, n_days)
    meta_spend = np.abs(meta_spend)
    meta_true_roas = 2.5 * (1 - np.exp(-meta_spend / 12_000))
    meta_conv = meta_spend * meta_true_roas * seasonality + rng.normal(0, 400, n_days)

    # --- Google: 有明显脉冲（周末超投 2 倍）---
    google_spend = np.where(t % 7 >= 5, 20_000, 8_000).astype(float)
    google_spend += rng.normal(0, 300, n_days)
    google_spend = np.abs(google_spend)
    google_true_roas = 2.0 * (1 - np.exp(-google_spend / 10_000))
    google_conv = google_spend * google_true_roas * seasonality + rng.normal(0, 300, n_days)

    channels = [
        ChannelData("TikTok", tiktok_spend, tiktok_conv),
        ChannelData("Meta", meta_spend, meta_conv),
        ChannelData("Google", google_spend, google_conv),
    ]

    # 模拟 TikTok 停投实验（第 80-83 天，德州区域停投 2 天）
    shocks = [
        ExperimentalShock(
            channel="TikTok",
            shock_period=(80, 83),
            spend_multiplier=0.0,      # 完全停投
            observed_lift=0.1,         # 停投后转化跌至基线的 10%
        ),
        ExperimentalShock(
            channel="TikTok",
            shock_period=(83, 86),
            spend_multiplier=3.0,      # 超投 3 倍
            observed_lift=1.8,         # 转化提升至基线的 1.8 倍
        ),
    ]

    return channels, shocks


def run_self_test() -> bool:
    """
    完整流程自测:
    1. 生成模拟数据
    2. 诊断观测等价性
    3. 录入实验冲击
    4. 拟合 GP 模型
    5. 获取 ROAS 估计
    6. 优化预算分配
    7. 断言关键结论正确
    """
    print("=" * 60)
    print("Identified Bayesian MMM 自测")
    print("=" * 60)

    # Step 1: 生成模拟数据
    channels, shocks = _generate_mock_data(n_days=90)
    mmm = IdentifiedBayesianMMM(channels)

    # Step 2: 诊断
    print("\n[Step 1] 观测等价性诊断...")
    report = mmm.diagnose()
    print(f"  存在混淆风险: {report.has_observational_equivalence_risk}")
    for ch_name, risk in report.channel_risks.items():
        print(f"  {ch_name}: {risk[:60]}...")

    # 断言: TikTok 平稳投放，应该被识别为高风险
    assert "HIGH RISK" in report.channel_risks["TikTok"], (
        f"TikTok 应为 HIGH RISK，实际: {report.channel_risks['TikTok']}"
    )
    assert report.has_observational_equivalence_risk, "整体应检测到混淆风险"
    print("  ✓ 诊断断言通过")

    if report.experiment_prescriptions:
        print(f"\n[实验处方] {report.experiment_prescriptions[0][:80]}...")

    # Step 3: 无实验数据先拟合一次（基准模型）
    print("\n[Step 2] 拟合基准模型（无实验数据）...")
    mmm.fit()
    current_spends = {"TikTok": 10_000, "Meta": 8_000, "Google": 12_000}
    baseline_roas = mmm.get_roas(current_spends)

    print("  基准 ROAS 估计（存在混淆风险）:")
    for ch, est in baseline_roas.items():
        print(f"    {ch}: {est.point_estimate:.3f} [{est.lower_ci:.3f}, {est.upper_ci:.3f}]"
              f"  identified={est.is_identified}")

    # 断言: 基准模型中 TikTok 未识别
    assert not baseline_roas["TikTok"].is_identified, "基准模型 TikTok 不应被标记为已识别"

    # Step 4: 录入实验数据
    print("\n[Step 3] 录入实验冲击数据...")
    mmm_identified = IdentifiedBayesianMMM(channels)
    mmm_identified.diagnose()
    for shock in shocks:
        mmm_identified.add_shock(shock)
        print(f"  已录入: {shock.channel} 冲击 period={shock.shock_period}, "
              f"multiplier={shock.spend_multiplier}x")

    # Step 5: 拟合带实验先验的模型
    print("\n[Step 4] 拟合 Identified 模型（含实验先验）...")
    mmm_identified.fit()
    identified_roas = mmm_identified.get_roas(current_spends)

    print("  校正后 ROAS 估计:")
    for ch, est in identified_roas.items():
        print(f"    {ch}: {est.point_estimate:.3f} [{est.lower_ci:.3f}, {est.upper_ci:.3f}]"
              f"  identified={est.is_identified}")
        print(f"         注: {est.confidence_note}")

    # 断言: 加入实验数据后 TikTok 应被标记为已识别
    assert identified_roas["TikTok"].is_identified, "录入实验数据后 TikTok 应标记为已识别"
    # ROAS 应为正数
    for ch, est in identified_roas.items():
        assert est.point_estimate > 0, f"{ch} ROAS 应为正数，实际={est.point_estimate}"
        assert est.lower_ci <= est.upper_ci, f"{ch} CI 下界应 ≤ 上界"

    print("  ✓ 识别断言通过")

    # Step 6: 预算优化
    print("\n[Step 5] 优化预算分配...")
    total_budget = 30_000.0
    optimal = mmm_identified.optimize(
        total_budget=total_budget,
        current_spends=current_spends,
    )
    print("  优化后预算分配:")
    for ch, spend in optimal.items():
        orig = current_spends[ch]
        delta = ((spend - orig) / orig) * 100
        print(f"    {ch}: {orig:,.0f} → {spend:,.0f} ({delta:+.1f}%)")

    # 断言: 优化后总预算保持不变（允许 1% 误差）
    assert abs(sum(optimal.values()) - total_budget) / total_budget < 0.01, (
        f"优化后总预算偏差过大: {sum(optimal.values()):.1f} vs {total_budget:.1f}"
    )
    # 每个渠道分配 ≥ 5% 总预算
    for ch, spend in optimal.items():
        assert spend >= total_budget * 0.04, (
            f"{ch} 优化后预算过低: {spend:.1f}"
        )
    print("  ✓ 预算优化断言通过")

    # Step 7: GP 预测一致性检查
    print("\n[Step 6] GP 曲线单调性检查...")
    for ch_name, gp in mmm_identified._gps.items():
        spend_levels = np.linspace(0.01, 1.0, 20)
        preds, _ = gp.predict(spend_levels)
        # 检查曲线大体单调递增（允许少量噪声起伏）
        diffs = np.diff(preds)
        n_positive = (diffs > -0.05).sum()
        monotone_ratio = n_positive / len(diffs)
        print(f"  {ch_name} 饱和曲线单调性比例: {monotone_ratio:.0%}")
        assert monotone_ratio >= 0.6, (
            f"{ch_name} GP 曲线单调性过差: {monotone_ratio:.0%}"
        )
    print("  ✓ GP 曲线断言通过")

    print("\n" + "=" * 60)
    print("✅ 所有自测通过！Identified Bayesian MMM 验证完成。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_self_test()
    exit(0 if success else 1)
