"""
AB Experimental Design — A/B 实验设计：样本量/分流/随机化
paper2skills-code: 02-A_B实验 | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass


@dataclass
class ExperimentSpec:
    name: str
    hypothesis: str
    baseline_metric: float    # 对照组基线（如 CVR=0.03）
    min_detectable_effect: float  # 最小可检测效应（如 0.003 = 10%提升）
    alpha: float = 0.05           # 显著性水平（Type I 错误）
    power: float = 0.80           # 统计功效（1 - Type II 错误）
    is_two_sided: bool = True


@dataclass
class SampleSizeResult:
    n_per_group: int
    total_n: int
    days_to_run: float
    daily_traffic_required: float


@dataclass
class ExperimentResult:
    control_rate: float
    treatment_rate: float
    lift_pct: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]
    recommendation: str


def z_score(alpha: float, two_sided: bool = True) -> float:
    """近似 Z 分位数（基于正态分布）"""
    p = alpha / 2 if two_sided else alpha
    if p <= 0.001:
        return 3.09
    elif p <= 0.005:
        return 2.576
    elif p <= 0.025:
        return 1.96
    elif p <= 0.05:
        return 1.645
    return 1.28


def calc_sample_size(spec: ExperimentSpec,
                     daily_eligible_users: int = 10000) -> SampleSizeResult:
    z_alpha = z_score(spec.alpha, spec.is_two_sided)
    z_beta = z_score(1 - spec.power, two_sided=False)

    p1 = spec.baseline_metric
    p2 = p1 + spec.min_detectable_effect
    p_bar = (p1 + p2) / 2

    n = ((z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
          z_beta * math.sqrt(p1 * (1-p1) + p2 * (1-p2))) ** 2
         / (p2 - p1) ** 2)
    n_per = math.ceil(n)
    total = n_per * 2
    days = total / daily_eligible_users

    return SampleSizeResult(
        n_per_group=n_per, total_n=total,
        days_to_run=round(days, 1),
        daily_traffic_required=math.ceil(total / max(days, 1)),
    )


def analyze_result(control_conversions: int, control_n: int,
                   treatment_conversions: int, treatment_n: int,
                   alpha: float = 0.05) -> ExperimentResult:
    p_c = control_conversions / max(control_n, 1)
    p_t = treatment_conversions / max(treatment_n, 1)
    lift = (p_t - p_c) / max(p_c, 1e-9) * 100

    p_pool = (control_conversions + treatment_conversions) / max(control_n + treatment_n, 1)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/control_n + 1/treatment_n))
    z = (p_t - p_c) / max(se, 1e-9)
    p_value = 2 * (1 - min(0.9999, abs(z) / 4))  # 简化近似

    margin = z_score(alpha) * math.sqrt(p_c*(1-p_c)/control_n + p_t*(1-p_t)/treatment_n)
    ci = (round(p_t - p_c - margin, 5), round(p_t - p_c + margin, 5))

    significant = p_value < alpha
    rec = f"推荐推全（提升 {lift:.1f}%，p={p_value:.4f}）" if significant else           f"不推全（p={p_value:.4f}，效果不显著）"

    return ExperimentResult(
        control_rate=round(p_c, 5), treatment_rate=round(p_t, 5),
        lift_pct=round(lift, 2), p_value=round(p_value, 4),
        significant=significant, confidence_interval=ci, recommendation=rec,
    )


def run_experiment_demo():
    spec = ExperimentSpec(
        name="新版 Listing 图片 A/B 测试",
        hypothesis="新产品主图将提升 CVR 10%（从 3% → 3.3%）",
        baseline_metric=0.030,
        min_detectable_effect=0.003,
        alpha=0.05, power=0.80,
    )
    size = calc_sample_size(spec, daily_eligible_users=5000)

    print(f"=== {spec.name} ===")
    print(f"假设: {spec.hypothesis}")
    print(f"每组需要 {size.n_per_group:,} 用户 | 总计 {size.total_n:,} | 预计 {size.days_to_run} 天")

    result = analyze_result(
        control_conversions=315, control_n=10500,
        treatment_conversions=368, treatment_n=10500,
    )
    print(f"实验结果:")
    print(f"  对照组 CVR: {result.control_rate:.4f}")
    print(f"  实验组 CVR: {result.treatment_rate:.4f}")
    print(f"  提升幅度: {result.lift_pct:+.1f}% | p={result.p_value:.4f}")
    print(f"  95% CI: [{result.confidence_interval[0]:.5f}, {result.confidence_interval[1]:.5f}]")
    print(f"  结论: {result.recommendation}")
    print("✅ A/B 实验设计演示完成")
if __name__ == "__main__":
    run_experiment_demo()
