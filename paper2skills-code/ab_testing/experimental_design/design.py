"""
A/B Experimental Design Core Toolkit
A/B 实验设计核心工具包

功能:
- 样本量计算 (连续/二分类/相对提升)
- 统计功效 (Power) 分析
- 最小可检测效应 (MDE) 计算
- 分层随机分配 (Stratified Allocation)
- CUPED 方差缩减
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Sample Size Calculations
# ---------------------------------------------------------------------------

def sample_size_continuous(
    baseline_mean: float,
    mde: float,
    std: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> int:
    """
    连续型指标的样本量计算 (两样本 t 检验)。

    公式来源: Zhou et al. (2023) 公式 (1)
    n = (Z_{1-α/2} + Z_{1-β})^2 * σ^2 * (1 + 1/ratio) / δ^2
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    variance_factor = std**2 * (1 + 1 / ratio)
    n = (z_alpha + z_beta) ** 2 * variance_factor / (mde**2)
    return int(np.ceil(n))


def sample_size_binary(
    p_control: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> int:
    """
    二分类指标(转化率)的样本量计算。

    公式来源: 合并比例的两样本比例检验
    n = (Z_{1-α/2} + Z_{1-β})^2 * (p_pool*(1-p_pool)) * (1+1/ratio) / δ^2
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_treatment = p_control + mde_absolute
    p_pool = (p_control + p_treatment) / 2
    variance_factor = p_pool * (1 - p_pool) * (1 + 1 / ratio)
    n = (z_alpha + z_beta) ** 2 * variance_factor / (mde_absolute**2)
    return int(np.ceil(n))


def sample_size_relative_lift(
    p_control: float,
    relative_lift: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> int:
    """
    相对提升(Relative Lift)的样本量计算。

    公式来源: Zhou et al. (2023) 公式 (10-11)
    对于二分类指标, 相对提升 δ_rel = (p_t - p_c) / p_c
    所需样本量需考虑控制组均值在分母带来的额外方差。
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_treatment = p_control * (1 + relative_lift)
    delta = p_treatment - p_control
    if delta == 0:
        raise ValueError("relative_lift cannot be zero")

    # 使用 Delta method 调整后的方差因子
    # n_rel ≈ (1/p_c^2 + p_t^2/(p_c^4)) * p_pool(1-p_pool) * (Zα+Zβ)^2 / δ_rel^2
    p_pool = (p_control + p_treatment) / 2
    var_pool = p_pool * (1 - p_pool)
    variance_factor = var_pool * (1 / p_control**2 + p_treatment**2 / p_control**4)
    n = (z_alpha + z_beta) ** 2 * variance_factor / (relative_lift**2)
    return int(np.ceil(n))


# ---------------------------------------------------------------------------
# 2. Power & MDE
# ---------------------------------------------------------------------------

def power_analysis(
    n_per_group: int,
    effect_size: float,
    std: float = 1.0,
    alpha: float = 0.05,
    ratio: float = 1.0,
) -> float:
    """
    计算给定样本量下的统计功效 (Power)。

    Power = 1 - Φ( Z_{1-α/2} - δ√n_t / (σ√(1+1/ratio)) )
    其中 n_t 为治疗组样本量。
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    sigma_eff = std * np.sqrt(1 + 1 / ratio)
    z = z_alpha - effect_size * np.sqrt(n_per_group) / sigma_eff
    power = 1 - stats.norm.cdf(z)
    return float(power)


def mde_calculator(
    n_per_group: int,
    std: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> float:
    """
    计算给定样本量下可检测的最小效应 (MDE)。

    MDE = (Z_{1-α/2} + Z_{1-β}) * σ * √(1+1/ratio) / √n_t
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    sigma_eff = std * np.sqrt(1 + 1 / ratio)
    mde = (z_alpha + z_beta) * sigma_eff / np.sqrt(n_per_group)
    return float(mde)


# ---------------------------------------------------------------------------
# 3. Stratified Allocation
# ---------------------------------------------------------------------------

def stratified_allocation(
    df: pd.DataFrame,
    strata_cols: List[str],
    treatment_prob: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    分层随机分配: 确保每个 stratum 内治疗/控制比例恒定。

    参数:
        df: 用户数据框
        strata_cols: 分层维度, 如 ["country", "device_type"]
        treatment_prob: 治疗组分配概率
        seed: 随机种子

    返回:
        增加 assignment 列的 DataFrame ("T" / "C")
    """
    rng = np.random.RandomState(seed)
    result = df.copy()
    result["assignment"] = ""
    for _, group in result.groupby(strata_cols):
        n = len(group)
        treat = rng.random(n) < treatment_prob
        result.loc[group.index, "assignment"] = ["T" if t else "C" for t in treat]
    return result


# ---------------------------------------------------------------------------
# 4. CUPED Variance Reduction
# ---------------------------------------------------------------------------

def cuped_adjustment(
    y: np.ndarray,
    x: np.ndarray,
    assignment: np.ndarray,
) -> Tuple[float, float, float]:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) 方差缩减。

    公式:
        θ = Cov(y, x) / Var(x)
        y_cuped = y - θ * (x - mean(x))
        提升量估计使用调整后的 y_cuped

    参数:
        y: 实验中观测到的结果指标
        x: 实验前的协变量(同一指标的历史值)
        assignment: 分组标记, "T" 或 "C"

    返回:
        (原始估计提升量, CUPED调整后提升量, 方差缩减比例)
    """
    is_treat = assignment == "T"
    is_ctrl = assignment == "C"

    # 原始估计
    y_t = y[is_treat].mean()
    y_c = y[is_ctrl].mean()
    raw_lift = y_t - y_c

    # 计算最优系数 θ
    theta = np.cov(y, x)[0, 1] / np.var(x)
    y_adj = y - theta * (x - x.mean())

    adj_t = y_adj[is_treat].mean()
    adj_c = y_adj[is_ctrl].mean()
    cuped_lift = adj_t - adj_c

    var_raw = y[is_treat].var(ddof=1) / is_treat.sum() + y[is_ctrl].var(ddof=1) / is_ctrl.sum()
    var_cuped = (
        y_adj[is_treat].var(ddof=1) / is_treat.sum()
        + y_adj[is_ctrl].var(ddof=1) / is_ctrl.sum()
    )
    reduction = 1 - var_cuped / var_raw if var_raw > 0 else 0.0

    return float(raw_lift), float(cuped_lift), float(reduction)


# ---------------------------------------------------------------------------
# 5. Experiment Designer (统一封装)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    """实验设计规格"""

    metric_type: str  # "continuous" | "binary"
    baseline: float
    mde: float
    alpha: float = 0.05
    power: float = 0.8
    ratio: float = 1.0


@dataclass
class ExperimentPlan:
    """实验设计方案"""

    n_control: int
    n_treatment: int
    total_n: int
    duration_days: int
    daily_traffic: int
    actual_power: float
    mde_verify: float


class ABTestDesigner:
    """
    A/B 实验设计器: 输入业务参数, 输出完整实验方案。
    """

    def __init__(self, daily_traffic: int = 1000):
        self.daily_traffic = daily_traffic

    def design(self, spec: ExperimentSpec) -> ExperimentPlan:
        """根据指标类型计算所需样本量并给出实验计划。"""
        if spec.metric_type == "continuous":
            # 假设 std = baseline / 3 作为缺省估计
            std = spec.baseline / 3.0
            n_treatment = sample_size_continuous(
                baseline_mean=spec.baseline,
                mde=spec.mde,
                std=std,
                alpha=spec.alpha,
                power=spec.power,
                ratio=spec.ratio,
            )
        elif spec.metric_type == "binary":
            n_treatment = sample_size_binary(
                p_control=spec.baseline,
                mde_absolute=spec.mde,
                alpha=spec.alpha,
                power=spec.power,
                ratio=spec.ratio,
            )
        else:
            raise ValueError(f"Unsupported metric_type: {spec.metric_type}")

        n_control = int(np.ceil(n_treatment * spec.ratio))
        total_n = n_control + n_treatment
        duration_days = int(np.ceil(total_n / self.daily_traffic))

        # 回验实际 power 和 mde
        if spec.metric_type == "continuous":
            std = spec.baseline / 3.0
            actual_power = power_analysis(
                n_per_group=n_treatment,
                effect_size=spec.mde,
                std=std,
                alpha=spec.alpha,
                ratio=spec.ratio,
            )
            mde_verify = mde_calculator(
                n_per_group=n_treatment,
                std=std,
                alpha=spec.alpha,
                power=spec.power,
                ratio=spec.ratio,
            )
        else:
            # 对于二分类, 用正态近似回验, std 使用与 sample_size_binary 一致的 p_pool
            p_treatment = spec.baseline + spec.mde
            p_pool = (spec.baseline + p_treatment) / 2
            std_pool = np.sqrt(p_pool * (1 - p_pool))
            actual_power = power_analysis(
                n_per_group=n_treatment,
                effect_size=spec.mde,
                std=std_pool,
                alpha=spec.alpha,
                ratio=spec.ratio,
            )
            mde_verify = mde_calculator(
                n_per_group=n_treatment,
                std=std_pool,
                alpha=spec.alpha,
                power=spec.power,
                ratio=spec.ratio,
            )

        return ExperimentPlan(
            n_control=n_control,
            n_treatment=n_treatment,
            total_n=total_n,
            duration_days=duration_days,
            daily_traffic=self.daily_traffic,
            actual_power=actual_power,
            mde_verify=mde_verify,
        )


# ---------------------------------------------------------------------------
# 6. Demo: Momcozy 母婴电商场景
# ---------------------------------------------------------------------------

def generate_momcozy_users(n_users: int = 5000, seed: int = 42) -> pd.DataFrame:
    """生成 Momcozy 母婴电商合成用户数据。"""
    rng = np.random.RandomState(seed)
    countries = ["US", "UK", "DE", "CA"]
    devices = ["mobile", "desktop"]
    user_types = ["new", "returning"]

    df = pd.DataFrame(
        {
            "user_id": [f"U_{i:05d}" for i in range(n_users)],
            "country": rng.choice(countries, size=n_users, p=[0.5, 0.2, 0.2, 0.1]),
            "device_type": rng.choice(devices, size=n_users, p=[0.7, 0.3]),
            "user_type": rng.choice(user_types, size=n_users, p=[0.4, 0.6]),
        }
    )
    # 历史转化率 (pre-period) 作为 CUPED 协变量
    base_rate = 0.025
    df["pre_conversion_rate"] = rng.beta(2, 78, size=n_users) * 0.5 + base_rate * 0.5
    df["pre_conversion_rate"] = df["pre_conversion_rate"].clip(0.001, 0.5)

    # 历史平均订单价值 (AOV) 作为连续指标基线
    df["pre_aov"] = rng.lognormal(3.8, 0.4, size=n_users)
    return df


def main() -> None:
    print("=" * 60)
    print("A/B 实验设计基础 - Momcozy 母婴电商场景演示")
    print("=" * 60)

    # ---------------------------
    # 场景 1: 落地页转化率优化
    # ---------------------------
    print("\n【场景 1】落地页转化率优化实验设计")
    baseline_cr = 0.025  # 2.5%
    target_relative_lift = 0.10  # 期望提升 10% (到 2.75%)
    mde_abs = baseline_cr * target_relative_lift  # 0.0025

    n_binary = sample_size_binary(
        p_control=baseline_cr, mde_absolute=mde_abs, alpha=0.05, power=0.8
    )
    n_rel = sample_size_relative_lift(
        p_control=baseline_cr, relative_lift=target_relative_lift, alpha=0.05, power=0.8
    )

    print(f"  基线转化率: {baseline_cr:.2%}")
    print(f"  目标相对提升: {target_relative_lift:.0%}")
    print(f"  绝对提升 MDE: {mde_abs:.4f}")
    print(f"  按绝对提升计算样本量: 每组 {n_binary:,} 人")
    print(f"  按相对提升计算样本量: 每组 {n_rel:,} 人")
    print(f"  -> 相对提升设计更保守, 建议采用每组 {max(n_binary, n_rel):,} 人")

    # 实验设计器输出完整方案
    designer = ABTestDesigner(daily_traffic=2000)
    plan = designer.design(
        ExperimentSpec(
            metric_type="binary",
            baseline=baseline_cr,
            mde=mde_abs,
            alpha=0.05,
            power=0.8,
        )
    )
    print(f"\n  实验计划:")
    print(f"    控制组: {plan.n_control:,} 人")
    print(f"    治疗组: {plan.n_treatment:,} 人")
    print(f"    总计:   {plan.total_n:,} 人")
    print(f"    日流量: {plan.daily_traffic:,} 人/天")
    print(f"    预计时长: {plan.duration_days} 天")
    print(f"    回验 Power: {plan.actual_power:.2%}")

    # ---------------------------
    # 场景 2: 分层随机分配
    # ---------------------------
    print("\n【场景 2】分层随机分配 (按国家 + 设备类型 + 新老用户)")
    users = generate_momcozy_users(n_users=20000, seed=42)
    assigned = stratified_allocation(
        users, strata_cols=["country", "device_type", "user_type"], treatment_prob=0.5, seed=42
    )
    balance = (
        assigned.groupby(["country", "device_type", "user_type"])["assignment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(3)
    )
    print("  各分层内分配比例 (前 6 组):")
    print(balance.head(6).to_string())

    # ---------------------------
    # 场景 3: CUPED 方差缩减
    # ---------------------------
    print("\n【场景 3】CUPED 方差缩减演示")
    rng = np.random.RandomState(42)
    n = 10000
    # 模拟真实转化行为: 当前转化率与历史转化率高度相关
    pre = rng.beta(2, 78, size=n) * 0.5 + 0.0125
    theta = 0.6  # 历史对未来的预测系数
    noise = rng.normal(0, 0.008, size=n)
    y = 0.025 + theta * (pre - pre.mean()) + noise
    y = np.clip(y, 0, 1)
    assignment = np.array(["T"] * (n // 2) + ["C"] * (n // 2))
    # 给治疗组加一点真实提升
    y[: n // 2] += 0.002

    raw_lift, cuped_lift, reduction = cuped_adjustment(y, pre, assignment)
    print(f"  原始估计提升量: {raw_lift:.4f}")
    print(f"  CUPED 调整后提升量: {cuped_lift:.4f}")
    print(f"  方差缩减比例: {reduction:.1%}")
    print(f"  -> 等效于样本量缩减至原来的 {1/(1-reduction):.1f} 倍")

    # ---------------------------
    # 场景 4: Power 与 MDE 快速查询
    # ---------------------------
    print("\n【场景 4】Power / MDE 快速查询表")
    print("  假设基线转化率 2.5%, 每组 50,000 用户:")
    for rel_lift in [0.05, 0.08, 0.10, 0.15]:
        mde = baseline_cr * rel_lift
        p_t = baseline_cr + mde
        p_pool = (baseline_cr + p_t) / 2
        std = np.sqrt(p_pool * (1 - p_pool))
        pwr = power_analysis(n_per_group=50000, effect_size=mde, std=std, alpha=0.05)
        print(f"    相对提升 {rel_lift:>4.0%} -> Power = {pwr:.1%}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
