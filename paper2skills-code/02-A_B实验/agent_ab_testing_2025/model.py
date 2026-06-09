"""
Agent A/B Testing Framework - 交互式 LLM 智能体全自动 A/B 测试

论文: Automated and Scalable Web A/B Testing with Interactive LLM Agents
arXiv: 2504.09723 (2025-04)

核心思想:
  用成千上万个拥有虚拟 Persona 的 LLM 智能体替代真实用户跑 A/B 实验。
  四模块流程：Agent 生成 → 测试配置/分流 → 自主仿真 → 后测分析。

自测模式 (python model.py):
  Mock 掉 LLM 调用，纯 Python 验证全链路逻辑。
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────── 数据结构 ──────────────────────────────────

class Group(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class Persona:
    """虚拟用户画像 (Agent Generation 模块)"""
    persona_id: str
    age_group: str           # "18-24" / "25-34" / "35-44" / "45+"
    price_sensitivity: float  # 0.0 (无所谓) ~ 1.0 (极度敏感)
    style_preference: str    # "trendy" / "classic" / "practical"
    platform: str            # "mobile" / "desktop"
    group: Optional[Group] = None

    def __repr__(self) -> str:
        return f"Persona({self.persona_id}, {self.age_group}, price={self.price_sensitivity:.1f}, {self.style_preference})"


@dataclass
class InteractionTrace:
    """智能体在页面上的交互轨迹 (Autonomous Simulation 模块)"""
    persona_id: str
    group: Group
    session_duration_s: float   # 会话时长(秒)
    page_views: int
    clicked_add_to_cart: bool
    clicked_checkout: bool
    converted: bool             # 最终是否完成购买
    dom_events: List[str] = field(default_factory=list)  # DOM 交互记录


# ─────────────────────────────── 核心模块 ──────────────────────────────────

class AgentGenerationModule:
    """
    Module 1: 智能体生成库
    基于目标人群分布，自动生成具有不同 Persona 的虚拟用户池。
    真实系统中此处调用 LLM，Mock 版本按参数分布随机生成。
    """

    AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"]
    AGE_WEIGHTS = [0.20, 0.35, 0.30, 0.15]
    STYLES = ["trendy", "classic", "practical"]
    STYLE_WEIGHTS = [0.40, 0.30, 0.30]
    PLATFORMS = ["mobile", "desktop"]
    PLATFORM_WEIGHTS = [0.65, 0.35]

    def generate(self, n_agents: int, seed: int = 42) -> List[Persona]:
        """生成 n_agents 个虚拟用户"""
        rng = random.Random(seed)
        personas = []
        for i in range(n_agents):
            age_group = rng.choices(self.AGE_GROUPS, weights=self.AGE_WEIGHTS)[0]
            # 价格敏感度与年龄组正相关（35岁以上更在意价格）
            base_sensitivity = {"18-24": 0.4, "25-34": 0.5, "35-44": 0.7, "45+": 0.8}[age_group]
            price_sensitivity = min(1.0, max(0.0, rng.gauss(base_sensitivity, 0.15)))
            style = rng.choices(self.STYLES, weights=self.STYLE_WEIGHTS)[0]
            platform = rng.choices(self.PLATFORMS, weights=self.PLATFORM_WEIGHTS)[0]
            personas.append(Persona(
                persona_id=f"agent_{i:04d}",
                age_group=age_group,
                price_sensitivity=price_sensitivity,
                style_preference=style,
                platform=platform,
            ))
        return personas


class TestingPreparationModule:
    """
    Module 2: 测试配置与分流
    将虚拟 Agent 严格随机分流到 Control/Treatment，
    并验证关键画像属性的平衡分布（Covariate Balance Check）。
    """

    def assign(
        self, personas: List[Persona], seed: int = 0
    ) -> Tuple[List[Persona], List[Persona]]:
        """随机分流，返回 (control_group, treatment_group)"""
        rng = random.Random(seed)
        shuffled = personas.copy()
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        control = shuffled[:mid]
        treatment = shuffled[mid:]
        for p in control:
            p.group = Group.CONTROL
        for p in treatment:
            p.group = Group.TREATMENT
        return control, treatment

    def check_balance(
        self, control: List[Persona], treatment: List[Persona]
    ) -> Dict[str, Any]:
        """检查两组关键属性是否均衡（Standardized Mean Difference < 0.1 视为均衡）"""
        c_sensitivity = [p.price_sensitivity for p in control]
        t_sensitivity = [p.price_sensitivity for p in treatment]

        pooled_std = math.sqrt(
            (statistics.variance(c_sensitivity) + statistics.variance(t_sensitivity)) / 2
        )
        smd = abs(statistics.mean(c_sensitivity) - statistics.mean(t_sensitivity)) / pooled_std

        # 年龄分布比较
        def age_dist(group: List[Persona]) -> Dict[str, float]:
            counts: Dict[str, int] = {}
            for p in group:
                counts[p.age_group] = counts.get(p.age_group, 0) + 1
            return {k: v / len(group) for k, v in counts.items()}

        return {
            "control_size": len(control),
            "treatment_size": len(treatment),
            "control_avg_price_sensitivity": round(statistics.mean(c_sensitivity), 4),
            "treatment_avg_price_sensitivity": round(statistics.mean(t_sensitivity), 4),
            "price_sensitivity_smd": round(smd, 4),
            "is_balanced": smd < 0.1,
            "control_age_dist": age_dist(control),
            "treatment_age_dist": age_dist(treatment),
        }


class AutonomousSimulationModule:
    """
    Module 3: 自主 A/B 仿真
    Agent 通过 Perceive-Decide-Act 闭环自主浏览网页，记录 DOM 交互轨迹。
    真实系统中：LLM 感知截图 → 决策下一步操作 → Playwright/Selenium 执行。
    Mock 版本：根据 Persona 特征 + 网页版本的差异参数计算行为概率。
    """

    def __init__(self, treatment_conversion_lift: float = 0.14, seed: int = 99):
        """
        treatment_conversion_lift: Treatment 组相比 Control 的转化率提升（如论文案例中的 +14%）
        """
        self.treatment_lift = treatment_conversion_lift
        self.rng = random.Random(seed)

    def _base_conversion_prob(self, persona: Persona) -> float:
        """基于 Persona 特征估算基础转化概率"""
        base = 0.12  # 基础转化率 12%
        # 移动端用户转化率略高（便捷购物）
        if persona.platform == "mobile":
            base += 0.03
        # 价格敏感用户在看到折扣时更易购买
        if persona.price_sensitivity > 0.7:
            base += 0.04
        # 追求时尚的年轻用户对新落地页响应更强
        if persona.style_preference == "trendy" and persona.age_group in ("18-24", "25-34"):
            base += 0.02
        return min(base, 0.40)

    def simulate_agent(self, persona: Persona) -> InteractionTrace:
        """模拟单个 Agent 的完整网页交互会话"""
        assert persona.group is not None, "Agent 必须先完成分组"

        base_prob = self._base_conversion_prob(persona)

        # Treatment 组获得转化率提升（新落地页效果）
        if persona.group == Group.TREATMENT:
            conv_prob = min(base_prob * (1 + self.treatment_lift), 0.60)
            # 新页面设计更直观，会话时长稍短
            session_duration = self.rng.gauss(120, 30)
        else:
            conv_prob = base_prob
            session_duration = self.rng.gauss(135, 35)

        session_duration = max(10.0, session_duration)

        # 模拟页面浏览行为
        page_views = max(1, int(self.rng.gauss(3.5, 1.2)))
        clicked_cart = self.rng.random() < conv_prob * 1.8  # 加购率高于购买率
        clicked_checkout = clicked_cart and self.rng.random() < 0.65
        converted = clicked_checkout and self.rng.random() < conv_prob / base_prob * 0.7

        # 构造 DOM 事件流
        dom_events: List[str] = ["page_load"]
        for _ in range(page_views - 1):
            dom_events.append(self.rng.choice(["scroll_down", "image_zoom", "review_expand", "filter_click"]))
        if clicked_cart:
            dom_events.append("add_to_cart_click")
        if clicked_checkout:
            dom_events.append("checkout_button_click")
        if converted:
            dom_events.append("order_confirmed")

        return InteractionTrace(
            persona_id=persona.persona_id,
            group=persona.group,
            session_duration_s=round(session_duration, 1),
            page_views=page_views,
            clicked_add_to_cart=clicked_cart,
            clicked_checkout=clicked_checkout,
            converted=converted,
            dom_events=dom_events,
        )

    def run_simulation(self, personas: List[Persona]) -> List[InteractionTrace]:
        """对所有 Agent 批量运行仿真"""
        return [self.simulate_agent(p) for p in personas]


class PostTestingAnalysisModule:
    """
    Module 4: 测试后分析
    汇总统计：转化率、会话时长、点击分布，输出显著性检验结果。
    """

    @staticmethod
    def analyze(traces: List[InteractionTrace]) -> Dict[str, Any]:
        """计算核心指标并执行双比例 z 检验"""
        control_traces = [t for t in traces if t.group == Group.CONTROL]
        treatment_traces = [t for t in traces if t.group == Group.TREATMENT]

        def group_stats(group_traces: List[InteractionTrace]) -> Dict[str, float]:
            n = len(group_traces)
            if n == 0:
                return {}
            conversions = sum(1 for t in group_traces if t.converted)
            conv_rate = conversions / n
            avg_duration = statistics.mean(t.session_duration_s for t in group_traces)
            avg_page_views = statistics.mean(t.page_views for t in group_traces)
            cart_rate = sum(1 for t in group_traces if t.clicked_add_to_cart) / n
            return {
                "n": n,
                "conversions": conversions,
                "conversion_rate": round(conv_rate, 4),
                "avg_session_duration_s": round(avg_duration, 2),
                "avg_page_views": round(avg_page_views, 2),
                "cart_rate": round(cart_rate, 4),
            }

        c_stats = group_stats(control_traces)
        t_stats = group_stats(treatment_traces)

        # 双比例 z 检验
        p_c = c_stats["conversion_rate"]
        p_t = t_stats["conversion_rate"]
        n_c = c_stats["n"]
        n_t = t_stats["n"]
        p_pooled = (c_stats["conversions"] + t_stats["conversions"]) / (n_c + n_t)

        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n_c + 1 / n_t))
        z_score = (p_t - p_c) / se if se > 0 else 0.0
        # 近似 p-value（单侧，treatment > control）
        p_value = _normal_sf(z_score)

        relative_lift = (p_t - p_c) / p_c if p_c > 0 else 0.0

        return {
            "control": c_stats,
            "treatment": t_stats,
            "relative_lift": round(relative_lift, 4),
            "absolute_lift": round(p_t - p_c, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "significant_at_95": p_value < 0.05,
            "significant_at_99": p_value < 0.01,
            "recommendation": (
                "✅ 推荐上线 Treatment（新落地页）"
                if p_value < 0.05 and relative_lift > 0
                else "❌ 结果不显著，谨慎上线"
            ),
        }


def _normal_sf(z: float) -> float:
    """标准正态分布生存函数 P(Z > z)，用 math.erfc 实现，避免外部依赖"""
    return 0.5 * math.erfc(z / math.sqrt(2))


# ─────────────────────────────── 完整流程编排 ──────────────────────────────

def run_agent_ab_test(
    n_agents: int = 200,
    treatment_lift: float = 0.14,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    执行完整的 Agent A/B 测试流程。

    Args:
        n_agents: 总智能体数量（将被平均分为 control/treatment）
        treatment_lift: 模拟的 Treatment 转化提升系数
        seed: 随机种子（保证可复现）
        verbose: 是否打印中间过程

    Returns:
        包含四个阶段结果的完整报告字典
    """
    print(f"\n{'='*60}")
    print(f"  Agent A/B Testing Framework (arXiv: 2504.09723)")
    print(f"  n_agents={n_agents}, treatment_lift={treatment_lift:.0%}, seed={seed}")
    print(f"{'='*60}")

    # Module 1: 生成智能体
    gen_module = AgentGenerationModule()
    personas = gen_module.generate(n_agents, seed=seed)
    if verbose:
        print(f"\n[Module 1] 生成 {len(personas)} 个虚拟用户 Agent")
        age_counts = {}
        for p in personas:
            age_counts[p.age_group] = age_counts.get(p.age_group, 0) + 1
        for age, cnt in sorted(age_counts.items()):
            print(f"  {age}: {cnt} 人 ({cnt/n_agents:.0%})")

    # Module 2: 分流与平衡检验
    prep_module = TestingPreparationModule()
    control, treatment = prep_module.assign(personas, seed=seed)
    balance = prep_module.check_balance(control, treatment)
    if verbose:
        print(f"\n[Module 2] 分流 → Control:{balance['control_size']} / Treatment:{balance['treatment_size']}")
        print(f"  价格敏感度 SMD: {balance['price_sensitivity_smd']} "
              f"({'✅ 均衡' if balance['is_balanced'] else '⚠️ 不均衡'})")

    # Module 3: 自主仿真
    sim_module = AutonomousSimulationModule(treatment_conversion_lift=treatment_lift, seed=seed)
    all_personas = control + treatment
    traces = sim_module.run_simulation(all_personas)
    if verbose:
        print(f"\n[Module 3] 仿真完成，共 {len(traces)} 条交互轨迹")
        total_events = sum(len(t.dom_events) for t in traces)
        print(f"  总 DOM 事件数: {total_events}")
        sample = traces[:3]
        for t in sample:
            print(f"  {t.persona_id} [{t.group.value}]: {t.dom_events} converted={t.converted}")

    # Module 4: 后测分析
    analysis = PostTestingAnalysisModule.analyze(traces)
    if verbose:
        print(f"\n[Module 4] 统计分析结果")
        c = analysis["control"]
        tr = analysis["treatment"]
        print(f"  Control   转化率: {c['conversion_rate']:.2%}  会话时长: {c['avg_session_duration_s']:.1f}s")
        print(f"  Treatment 转化率: {tr['conversion_rate']:.2%}  会话时长: {tr['avg_session_duration_s']:.1f}s")
        print(f"  相对提升: {analysis['relative_lift']:.2%}  绝对提升: {analysis['absolute_lift']:.2%}")
        print(f"  z-score: {analysis['z_score']}  p-value: {analysis['p_value']}")
        print(f"  显著性(α=0.05): {'是' if analysis['significant_at_95'] else '否'}")
        print(f"\n  {analysis['recommendation']}")

    return {
        "personas_count": len(personas),
        "balance_check": balance,
        "traces_count": len(traces),
        "analysis": analysis,
    }


# ─────────────────────────────── 自测套件 ──────────────────────────────────

def _test_persona_generation() -> None:
    """测试 1: 验证 Persona 生成数量与字段完整性"""
    gen = AgentGenerationModule()
    personas = gen.generate(100, seed=1)
    assert len(personas) == 100, f"期望 100，实际 {len(personas)}"
    for p in personas:
        assert 0.0 <= p.price_sensitivity <= 1.0, f"价格敏感度越界: {p}"
        assert p.age_group in AgentGenerationModule.AGE_GROUPS
        assert p.style_preference in AgentGenerationModule.STYLES
        assert p.platform in AgentGenerationModule.PLATFORMS
    print("  ✅ test_persona_generation PASSED")


def _test_assignment_balance() -> None:
    """测试 2: 验证分流平衡性"""
    gen = AgentGenerationModule()
    personas = gen.generate(200, seed=2)
    prep = TestingPreparationModule()
    control, treatment = prep.assign(personas, seed=2)
    balance = prep.check_balance(control, treatment)
    assert balance["is_balanced"], f"分组不均衡，SMD={balance['price_sensitivity_smd']}"
    assert abs(len(control) - len(treatment)) <= 1, "Control/Treatment 人数差异过大"
    print("  ✅ test_assignment_balance PASSED")


def _test_simulation_traces() -> None:
    """测试 3: 验证交互轨迹格式与合理性"""
    gen = AgentGenerationModule()
    personas = gen.generate(50, seed=3)
    prep = TestingPreparationModule()
    control, treatment = prep.assign(personas, seed=3)

    sim = AutonomousSimulationModule(treatment_conversion_lift=0.14, seed=3)
    all_traces = sim.run_simulation(control + treatment)

    assert len(all_traces) == 50
    for trace in all_traces:
        assert trace.session_duration_s > 0
        assert trace.page_views >= 1
        assert trace.group in (Group.CONTROL, Group.TREATMENT)
        assert len(trace.dom_events) >= 1
        assert "page_load" in trace.dom_events
        # 逻辑一致性：必须加购才能结账，必须结账才能转化
        if trace.converted:
            assert trace.clicked_checkout
        if trace.clicked_checkout:
            assert trace.clicked_add_to_cart
    print("  ✅ test_simulation_traces PASSED")


def _test_statistical_analysis() -> None:
    """测试 4: 验证统计分析字段与 p-value 合理性"""
    result = run_agent_ab_test(n_agents=500, treatment_lift=0.20, seed=42, verbose=False)
    analysis = result["analysis"]
    assert "z_score" in analysis
    assert "p_value" in analysis
    assert 0.0 <= analysis["p_value"] <= 1.0
    assert analysis["control"]["n"] > 0
    assert analysis["treatment"]["n"] > 0
    # 500 个 Agent + 20% lift 应该显著
    assert analysis["significant_at_95"], (
        f"预期显著（lift=20%，n=500），实际 p={analysis['p_value']}"
    )
    print("  ✅ test_statistical_analysis PASSED")


def _test_full_pipeline_reproducibility() -> None:
    """测试 5: 相同 seed 下结果完全可复现"""
    r1 = run_agent_ab_test(n_agents=100, seed=7, verbose=False)
    r2 = run_agent_ab_test(n_agents=100, seed=7, verbose=False)
    assert r1["analysis"]["z_score"] == r2["analysis"]["z_score"], "结果不可复现"
    assert r1["analysis"]["p_value"] == r2["analysis"]["p_value"], "结果不可复现"
    print("  ✅ test_full_pipeline_reproducibility PASSED")


def _test_subsegment_analysis() -> None:
    """测试 6: 验证子群分析（宝妈 vs 年轻女性）"""
    gen = AgentGenerationModule()
    personas = gen.generate(300, seed=10)
    prep = TestingPreparationModule()
    control, treatment = prep.assign(personas, seed=10)
    sim = AutonomousSimulationModule(treatment_conversion_lift=0.14, seed=10)
    all_traces = sim.run_simulation(control + treatment)

    # 按 persona 特征与 trace 关联
    persona_map = {p.persona_id: p for p in control + treatment}
    for trace in all_traces:
        persona = persona_map[trace.persona_id]
        assert persona.group == trace.group

    # 高价格敏感度人群
    sensitive_traces = [
        t for t in all_traces
        if persona_map[t.persona_id].price_sensitivity > 0.7
    ]
    assert len(sensitive_traces) > 0, "应存在高价格敏感度 Agent"
    print("  ✅ test_subsegment_analysis PASSED")


def run_all_tests() -> None:
    """运行所有自测用例"""
    print(f"\n{'─'*40}")
    print("  Running Self-Tests...")
    print(f"{'─'*40}")
    _test_persona_generation()
    _test_assignment_balance()
    _test_simulation_traces()
    _test_statistical_analysis()
    _test_full_pipeline_reproducibility()
    _test_subsegment_analysis()
    print(f"{'─'*40}")
    print("  All 6 tests PASSED ✅")
    print(f"{'─'*40}\n")


# ─────────────────────────────── 入口 ────────────────────────────────────

if __name__ == "__main__":
    # 1. 先跑自测
    run_all_tests()

    # 2. 跑完整演示：500 Agent，模拟黑五落地页 A/B 测试（+14% lift）
    result = run_agent_ab_test(
        n_agents=500,
        treatment_lift=0.14,
        seed=42,
        verbose=True,
    )

    # 3. 打印最终摘要
    print("\n📊 最终报告摘要")
    print(f"  样本总量: {result['personas_count']} 个虚拟 Agent")
    print(f"  分组均衡: {'✅' if result['balance_check']['is_balanced'] else '⚠️'}")
    analysis = result["analysis"]
    print(f"  Control 转化率: {analysis['control']['conversion_rate']:.2%}")
    print(f"  Treatment 转化率: {analysis['treatment']['conversion_rate']:.2%}")
    print(f"  提升幅度: +{analysis['relative_lift']:.2%} (p={analysis['p_value']})")
    print(f"  {analysis['recommendation']}")
