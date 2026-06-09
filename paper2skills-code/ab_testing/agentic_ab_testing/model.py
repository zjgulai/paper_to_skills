"""
Agentic AB Testing — AI Agent 驱动 A/B 实验全自动化
paper2skills-code: 02-A_B实验 | 母婴出海跨境电商

核心设计：
  - Hypothesis：假设数据类
  - HypothesisGenerator：从历史数据自动生成假设
  - ExperimentDesigner：样本量计算 + 流量分配
  - ResultInterpreter：统计显著性 + 业务含义自然语言解读
  - AgenticABTestRunner：全流程编排
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrafficStrategy(str, Enum):
    EQUAL_SPLIT = "equal_split"   # 等比例分流
    MAB_THOMPSON = "mab_thompson"  # Thompson Sampling 自适应


@dataclass
class Hypothesis:
    """A/B 实验假设"""
    metric: str                    # 主指标（如 ctr / cvr / revenue_per_session）
    description: str               # 假设描述（自然语言）
    expected_lift_pct: float       # 预期提升百分比（如 12.0 表示 12%）
    baseline_value: float          # 当前基线指标值
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confidence_source: str = ""    # 假设来源（历史数据/领域知识）


@dataclass
class ExperimentDesign:
    """实验设计结果"""
    hypothesis: Hypothesis
    n_per_group: int                # 每组所需样本量
    total_n: int                    # 总样本量
    estimated_days: float           # 预计运行天数
    daily_traffic_required: float   # 每日所需流量
    alpha: float = 0.05
    power: float = 0.80
    traffic_strategy: TrafficStrategy = TrafficStrategy.EQUAL_SPLIT
    n_variants: int = 2             # 方案数量（包含对照组）


@dataclass
class ExperimentObservation:
    """实验观测数据（单组）"""
    variant_name: str
    n_samples: int
    metric_sum: float              # 总指标值（如总点击数）
    metric_mean: float             # 平均指标值（如 CTR）


@dataclass
class InterpretedResult:
    """结果解读"""
    significant: bool
    p_value: float
    lift_pct: float                # 实际提升百分比
    confidence_pct: float          # 置信度（1 - p_value） * 100
    recommendation: str            # 自然语言决策建议
    winner: str                    # 获胜方案名（或 "none"）
    bonferroni_corrected: bool = False
    n_metrics_tested: int = 1


# ─────────────────────────────────────────────
# 假设生成器
# ─────────────────────────────────────────────

@dataclass
class HistoricalExperiment:
    """历史实验记录"""
    experiment_type: str   # 如 "image_type" / "price" / "title"
    lift_pct: float        # 历史实际提升
    metric: str
    succeeded: bool


class HypothesisGenerator:
    """从历史数据模式自动生成实验假设"""

    def generate(
        self,
        metric: str,
        baseline_value: float,
        historical_experiments: list[HistoricalExperiment],
        context: dict[str, Any] | None = None,
    ) -> list[Hypothesis]:
        """
        基于历史实验记录和当前基线，生成可测试假设列表。
        返回按预期提升降序排列的假设列表。
        """
        hypotheses: list[Hypothesis] = []
        context = context or {}

        relevant = [e for e in historical_experiments if e.metric == metric]
        type_lifts: dict[str, list[float]] = {}
        for exp in relevant:
            if exp.succeeded:
                type_lifts.setdefault(exp.experiment_type, []).append(exp.lift_pct)

        for exp_type, lifts in type_lifts.items():
            avg_lift = sum(lifts) / len(lifts)
            if avg_lift > 2.0:
                risk = RiskLevel.LOW if avg_lift > 10 else RiskLevel.MEDIUM
                desc = self._describe_hypothesis(exp_type, metric, avg_lift, context)
                hypotheses.append(Hypothesis(
                    metric=metric,
                    description=desc,
                    expected_lift_pct=avg_lift,
                    baseline_value=baseline_value,
                    risk_level=risk,
                    confidence_source=f"historical_avg_n={len(lifts)}",
                ))

        hypotheses.sort(key=lambda h: h.expected_lift_pct, reverse=True)
        return hypotheses

    def _describe_hypothesis(
        self,
        exp_type: str,
        metric: str,
        avg_lift: float,
        context: dict[str, Any],
    ) -> str:
        templates = {
            "image_type": f"将主图改为婴儿实际使用场景图，预计提升 {metric} {avg_lift:.1f}%",
            "price": f"调整定价策略，预计提升 {metric} {avg_lift:.1f}%",
            "title": f"优化 Listing 标题关键词，预计提升 {metric} {avg_lift:.1f}%",
            "bullet_points": f"重写 Bullet Points 强调核心卖点，预计提升 {metric} {avg_lift:.1f}%",
        }
        return templates.get(exp_type, f"{exp_type} 优化，预计提升 {metric} {avg_lift:.1f}%")


# ─────────────────────────────────────────────
# 实验设计器
# ─────────────────────────────────────────────

class ExperimentDesigner:
    """样本量计算 + 流量分配策略"""

    def design(
        self,
        hypothesis: Hypothesis,
        daily_traffic: float,
        alpha: float = 0.05,
        power: float = 0.80,
        n_variants: int = 2,
        strategy: TrafficStrategy = TrafficStrategy.EQUAL_SPLIT,
    ) -> ExperimentDesign:
        """计算实验所需样本量并输出设计方案"""
        p = hypothesis.baseline_value
        mde = p * hypothesis.expected_lift_pct / 100.0

        z_alpha = self._z_score(alpha / 2)
        z_beta = self._z_score(1 - power, one_tail=True)
        n_per_group = int(math.ceil(
            2 * (z_alpha + z_beta) ** 2 * p * (1 - p) / max(mde ** 2, 1e-10)
        ))

        total_n = n_per_group * n_variants
        daily_per_variant = daily_traffic / n_variants
        estimated_days = n_per_group / max(daily_per_variant, 1)

        return ExperimentDesign(
            hypothesis=hypothesis,
            n_per_group=n_per_group,
            total_n=total_n,
            estimated_days=estimated_days,
            daily_traffic_required=total_n / max(estimated_days, 1),
            alpha=alpha,
            power=power,
            traffic_strategy=strategy,
            n_variants=n_variants,
        )

    def _z_score(self, p: float, one_tail: bool = False) -> float:
        """近似 Z 分位数（标准正态分布）"""
        lookup = {0.005: 2.576, 0.025: 1.96, 0.05: 1.645, 0.10: 1.282, 0.20: 0.842}
        for threshold, z in sorted(lookup.items()):
            if p <= threshold:
                return z
        return 0.0


# ─────────────────────────────────────────────
# 结果解读器
# ─────────────────────────────────────────────

class ResultInterpreter:
    """统计显著性检验 + 业务含义自然语言解读"""

    def interpret(
        self,
        control: ExperimentObservation,
        treatment: ExperimentObservation,
        design: ExperimentDesign,
        n_metrics_tested: int = 1,
    ) -> InterpretedResult:
        """
        双比例 Z 检验 + Bonferroni 校正。
        使用正态近似（大样本量条件下有效）。
        """
        p1 = control.metric_mean
        p2 = treatment.metric_mean
        n1, n2 = control.n_samples, treatment.n_samples

        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

        if se < 1e-10:
            z_stat = 0.0
        else:
            z_stat = (p2 - p1) / se

        p_value = self._two_tailed_p(abs(z_stat))
        alpha = design.alpha / n_metrics_tested  # Bonferroni 校正
        significant = p_value < alpha
        lift_pct = (p2 - p1) / max(p1, 1e-10) * 100

        winner = treatment.variant_name if (significant and lift_pct > 0) else (
            control.variant_name if (significant and lift_pct < 0) else "none"
        )
        recommendation = self._generate_recommendation(
            significant, lift_pct, p_value, treatment, design, n_metrics_tested
        )

        return InterpretedResult(
            significant=significant,
            p_value=p_value,
            lift_pct=lift_pct,
            confidence_pct=(1 - p_value) * 100,
            recommendation=recommendation,
            winner=winner,
            bonferroni_corrected=n_metrics_tested > 1,
            n_metrics_tested=n_metrics_tested,
        )

    def _two_tailed_p(self, z: float) -> float:
        """近似双尾 p 值（标准正态分布，分位数查表插值）"""
        # 查表：(z_threshold, p_value) 降序排列，找第一个 z >= threshold 的区间
        table = [
            (3.29, 0.001), (3.09, 0.002), (2.807, 0.005),
            (2.576, 0.010), (2.326, 0.020), (2.054, 0.040),
            (1.960, 0.050), (1.751, 0.080), (1.645, 0.100),
            (1.440, 0.150), (1.282, 0.200), (1.036, 0.300),
        ]
        for threshold, p in table:
            if z >= threshold:
                return p
        return 0.500

    def _generate_recommendation(
        self,
        significant: bool,
        lift_pct: float,
        p_value: float,
        treatment: ExperimentObservation,
        design: ExperimentDesign,
        n_metrics: int,
    ) -> str:
        if not significant:
            if abs(lift_pct) < design.hypothesis.expected_lift_pct * 0.3:
                return (
                    f"❌ 未达显著性（p={p_value:.3f}）。效应量远小于预期，"
                    f"建议放弃该假设或重新设计实验。"
                )
            return (
                f"⏳ 未达显著性（p={p_value:.3f}，当前提升 {lift_pct:.1f}%）。"
                f"建议继续实验，预计还需 {design.estimated_days:.0f} 天。"
            )

        direction = "提升" if lift_pct > 0 else "下降"
        bonf_note = f"（已 Bonferroni 校正，n_metrics={n_metrics}）" if n_metrics > 1 else ""
        return (
            f"✅ 推荐 {treatment.variant_name} 上线：{design.hypothesis.metric} "
            f"{direction} {abs(lift_pct):.1f}%，置信度 {(1-p_value)*100:.1f}%"
            f"{bonf_note}。"
        )


# ─────────────────────────────────────────────
# Thompson Sampling MAB（多方案自适应分流）
# ─────────────────────────────────────────────

@dataclass
class MABArm:
    """MAB 方案臂（Beta 分布参数）"""
    name: str
    alpha: float = 1.0   # Beta 分布 alpha（成功次数 + 1）
    beta: float = 1.0    # Beta 分布 beta（失败次数 + 1）

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """从 Beta 后验分布采样"""
        a, b = self.alpha, self.beta
        x = random.betavariate(a, b)
        return x

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1
        else:
            self.beta += 1


class ThompsonSamplingMAB:
    """Thompson Sampling 多臂 Bandit，用于多方案定价/版本自适应测试"""

    def __init__(self, arm_names: list[str]) -> None:
        self.arms = {name: MABArm(name=name) for name in arm_names}

    def select_arm(self) -> str:
        """采样选择本次流量分配给哪个方案"""
        samples = {name: arm.sample() for name, arm in self.arms.items()}
        return max(samples, key=lambda k: samples[k])

    def update(self, arm_name: str, reward: float, threshold: float = 0.5) -> None:
        """更新方案的 Beta 后验（reward > threshold 视为成功）"""
        if arm_name in self.arms:
            self.arms[arm_name].update(success=reward > threshold)

    def get_allocation_pct(self) -> dict[str, float]:
        """返回各方案当前后验均值（近似流量分配比例）"""
        total = sum(arm.posterior_mean for arm in self.arms.values())
        return {
            name: arm.posterior_mean / max(total, 1e-10) * 100
            for name, arm in self.arms.items()
        }


# ─────────────────────────────────────────────
# 全流程编排
# ─────────────────────────────────────────────

class AgenticABTestRunner:
    """A/B 实验全流程 Agent 编排器：假设→设计→执行→解读"""

    def __init__(self) -> None:
        self._generator = HypothesisGenerator()
        self._designer = ExperimentDesigner()
        self._interpreter = ResultInterpreter()

    def generate_hypothesis(
        self,
        metric: str,
        baseline_value: float,
        historical_experiments: list[HistoricalExperiment],
        context: dict[str, Any] | None = None,
    ) -> Hypothesis:
        hypotheses = self._generator.generate(
            metric, baseline_value, historical_experiments, context
        )
        if not hypotheses:
            return Hypothesis(
                metric=metric,
                description=f"测试 {metric} 指标的默认优化方案",
                expected_lift_pct=5.0,
                baseline_value=baseline_value,
            )
        return hypotheses[0]

    def design_experiment(
        self,
        hypothesis: Hypothesis,
        daily_traffic: float,
        n_variants: int = 2,
    ) -> ExperimentDesign:
        return self._designer.design(hypothesis, daily_traffic, n_variants=n_variants)

    def interpret_result(
        self,
        control: ExperimentObservation,
        treatment: ExperimentObservation,
        design: ExperimentDesign,
        n_metrics_tested: int = 1,
    ) -> InterpretedResult:
        return self._interpreter.interpret(control, treatment, design, n_metrics_tested)


# ─────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────

def _test_agentic_ab() -> None:
    """Listing 图片实验：自动设计 + 模拟结果 + 解读"""

    runner = AgenticABTestRunner()

    historical = [
        HistoricalExperiment("image_type", 15.2, "ctr", True),
        HistoricalExperiment("image_type", 12.8, "ctr", True),
        HistoricalExperiment("image_type", 9.1, "ctr", True),
        HistoricalExperiment("price", 8.3, "ctr", False),
        HistoricalExperiment("title", 6.5, "ctr", True),
    ]

    hypothesis = runner.generate_hypothesis(
        metric="ctr",
        baseline_value=0.023,
        historical_experiments=historical,
    )
    assert "image_type" in hypothesis.description or "主图" in hypothesis.description
    assert hypothesis.expected_lift_pct > 10.0
    print(f"[假设生成] {hypothesis.description}")
    print(f"  预期提升: {hypothesis.expected_lift_pct:.1f}%, 风险: {hypothesis.risk_level}")

    design = runner.design_experiment(hypothesis, daily_traffic=3000, n_variants=2)
    assert design.n_per_group > 0
    assert design.estimated_days > 0
    print(f"\n[实验设计] 每组样本量: {design.n_per_group:,}")
    print(f"  预计运行: {design.estimated_days:.1f} 天")
    print(f"  每日所需流量: {design.daily_traffic_required:.0f}")

    control = ExperimentObservation("Control", n_samples=27000, metric_sum=621, metric_mean=0.023)
    treatment = ExperimentObservation("Variant_B", n_samples=27000, metric_sum=705, metric_mean=0.0261)
    result = runner.interpret_result(control, treatment, design, n_metrics_tested=1)

    assert result.significant, f"p={result.p_value:.4f}，应该显著"
    assert result.lift_pct > 10.0, f"提升 {result.lift_pct:.1f}%，应该 >10%"
    assert result.winner == "Variant_B"
    print(f"\n[结果解读] {result.recommendation}")
    print(f"  p-value: {result.p_value:.4f}, 提升: {result.lift_pct:.1f}%")

    mab = ThompsonSamplingMAB(["$42", "$45", "$48"])
    random.seed(42)
    for _ in range(100):
        arm = mab.select_arm()
        revenue = {"$42": 0.40, "$45": 0.55, "$48": 0.65}[arm]
        mab.update(arm, revenue)

    alloc = mab.get_allocation_pct()
    assert alloc["$48"] > alloc["$42"], f"$48 方案应获得更多流量，实际分配: {alloc}"
    print(f"\n[MAB 定价] 流量分配: {', '.join(f'{k}={v:.1f}%' for k, v in sorted(alloc.items()))}")

    not_sig_result = runner.interpret_result(
        ExperimentObservation("Control", 1000, 23, 0.023),
        ExperimentObservation("Variant_C", 1000, 24, 0.024),
        design,
    )
    assert not not_sig_result.significant
    print(f"\n[不显著案例] {not_sig_result.recommendation}")

    print("\n✅ 所有测试通过：假设生成、实验设计、结果解读、MAB 分配均正常")


if __name__ == "__main__":
    _test_agentic_ab()
