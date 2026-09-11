"""SkillForge: 领域特定自演化 Agent Skill 萃取与优化框架.

参考论文:Liu et al. (2026) SkillForge: Forging Domain-Specific, Self-Evolving
Agent Skills in Cloud Technical Support. SIGIR 2026 Industry Track.
arxiv: 2604.08618

本实现是简化版,用规则替代真实 LLM 调用,但保留论文的完整工作流与数据结构。
生产环境替换 `_llm_call` 为真实 LLM API 即可。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


Severity = str  # "high" | "medium" | "low" | "none"
DIM_PRIORITY = ("knowledge", "tool", "clarification", "style")


@dataclass
class TicketRecord:
    ticket_id: str
    category: str
    dialogue: list[str]
    tools_used: list[str]
    resolution: str


@dataclass
class WorkflowPattern:
    name: str
    steps: list[str]
    frequency: int


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, str]
    usage_frequency: int
    scenarios: list[str] = field(default_factory=list)


@dataclass
class KnowledgeItem:
    topic: str
    content: str
    source: str  # "kb" 或 "ticket_cited"


@dataclass
class Skill:
    skill_md: str
    tools: dict[str, ToolSchema] = field(default_factory=dict)
    references: dict[str, KnowledgeItem] = field(default_factory=dict)
    version: int = 0

    def render(self) -> str:
        tools_block = "\n".join(f"- {name}: {t.description}" for name, t in self.tools.items())
        refs_block = "\n".join(f"- {name}: {item.topic}" for name, item in self.references.items())
        return (
            f"{self.skill_md}\n\n## Tools\n{tools_block}\n\n## References\n{refs_block}"
        )


@dataclass
class FailureRecord:
    case_id: str
    knowledge: Severity = "none"
    tool: Severity = "none"
    clarification: Severity = "none"
    style: Severity = "none"

    @property
    def primary_category(self) -> str:
        levels = {"high": 3, "medium": 2, "low": 1, "none": 0}
        scored = [(dim, levels[getattr(self, dim)]) for dim in DIM_PRIORITY]
        scored.sort(key=lambda x: (-x[1], DIM_PRIORITY.index(x[0])))
        return scored[0][0] if scored[0][1] > 0 else "none"


# Mining stages -------------------------------------------------------------


class WorkflowMiner:
    """从历史工单挖工作流模式. 简化版用启发式规则,实际用 LLM."""

    PHASE_KEYWORDS = {
        "clarify": ["请问", "could you", "确认", "screenshot", "what is"],
        "diagnose": ["检查", "diagnose", "排查", "error code", "log"],
        "resolve": ["建议", "please try", "解决", "fix", "已为您"],
    }

    def mine(self, tickets: Iterable[TicketRecord]) -> list[WorkflowPattern]:
        pattern_counter: Counter[tuple[str, ...]] = Counter()
        for ticket in tickets:
            steps = self._extract_phases(ticket.dialogue)
            if steps:
                pattern_counter[tuple(steps)] += 1
        return [
            WorkflowPattern(name=" → ".join(steps), steps=list(steps), frequency=freq)
            for steps, freq in pattern_counter.most_common()
        ]

    def _extract_phases(self, dialogue: list[str]) -> list[str]:
        phases: list[str] = []
        for turn in dialogue:
            for phase, kws in self.PHASE_KEYWORDS.items():
                if any(kw in turn.lower() for kw in kws) and (not phases or phases[-1] != phase):
                    phases.append(phase)
        return phases


class ToolMiner:
    def mine(
        self,
        tickets: Iterable[TicketRecord],
        registry: dict[str, ToolSchema],
        frequency_threshold: int = 2,
    ) -> dict[str, ToolSchema]:
        usage: Counter[str] = Counter()
        scenario_map: dict[str, set[str]] = defaultdict(set)
        for ticket in tickets:
            for tool in ticket.tools_used:
                usage[tool] += 1
                scenario_map[tool].add(ticket.category)
        selected: dict[str, ToolSchema] = {}
        for tool_name, freq in usage.items():
            if freq < frequency_threshold or tool_name not in registry:
                continue
            schema = registry[tool_name]
            schema.usage_frequency = freq
            schema.scenarios = sorted(scenario_map[tool_name])
            selected[tool_name] = schema
        return selected


class KnowledgeExtractor:
    def extract(
        self,
        tickets: Iterable[TicketRecord],
        kb_docs: dict[str, str],
    ) -> dict[str, KnowledgeItem]:
        items: dict[str, KnowledgeItem] = {}
        for topic, content in kb_docs.items():
            items[topic] = KnowledgeItem(topic=topic, content=content, source="kb")
        cited = self._extract_cited_refs(tickets)
        for url, summary in cited.items():
            key = f"cited_{re.sub(r'[^a-z0-9]+', '_', url.lower())[:40]}"
            items[key] = KnowledgeItem(topic=url, content=summary, source="ticket_cited")
        return items

    @staticmethod
    def _extract_cited_refs(tickets: Iterable[TicketRecord]) -> dict[str, str]:
        urls: dict[str, str] = {}
        url_pat = re.compile(r"https?://\S+")
        for ticket in tickets:
            for turn in ticket.dialogue:
                for url in url_pat.findall(turn):
                    urls.setdefault(url, f"Referenced in {ticket.category}")
        return urls


class SkillSynthesizer:
    """把挖矿结果填入论文 Appendix A 定义的 SKILL.md 模板."""

    def synthesize(
        self,
        workflows: list[WorkflowPattern],
        tools: dict[str, ToolSchema],
        knowledge: dict[str, KnowledgeItem],
        scenario_name: str,
    ) -> Skill:
        top_workflow = workflows[0] if workflows else None
        sections = [
            f"# SKILL: {scenario_name}",
            "",
            "## 1. Background Knowledge",
            *[f"- {item.topic}: {item.content[:120]}" for item in list(knowledge.values())[:5]],
            "",
            "## 2. Scenario Triage",
            "Route incoming request by category keywords; fallback to FAQ §4.",
            "",
            "## 3. Per-Scenario Handling",
            f"Default workflow: {top_workflow.name if top_workflow else 'N/A'}",
            *[f"- step: {step}" for step in (top_workflow.steps if top_workflow else [])],
            "",
            "## 4. FAQ",
            "(long-tail issues; populated after deployment)",
            "",
            "## 5. Reference Index",
            *[f"- tool::{name}" for name in tools],
            *[f"- ref::{name}" for name in knowledge],
        ]
        return Skill(skill_md="\n".join(sections), tools=tools, references=knowledge)


# Self-evolution pipeline ----------------------------------------------------


@dataclass
class BadCase:
    case_id: str
    user_query: str
    agent_response: str
    reference_response: str


class FailureAnalyzer:
    """4 维并行分析. 实际生产把每个 `_analyze_*` 替换为 LLM 调用."""

    def analyze(self, case: BadCase) -> FailureRecord:
        return FailureRecord(
            case_id=case.case_id,
            knowledge=self._severity_from_overlap(case.agent_response, case.reference_response, "knowledge"),
            tool=self._severity_from_overlap(case.agent_response, case.reference_response, "tool"),
            clarification=self._severity_from_overlap(case.agent_response, case.reference_response, "clarification"),
            style=self._severity_from_overlap(case.agent_response, case.reference_response, "style"),
        )

    @staticmethod
    def _severity_from_overlap(actual: str, ref: str, dim: str) -> Severity:
        # 简化:用 token 集合 Jaccard 作占位评分.生产替换为 4 个专用 LLM prompt.
        a_tokens = set(actual.lower().split())
        r_tokens = set(ref.lower().split())
        if not r_tokens:
            return "none"
        jaccard = len(a_tokens & r_tokens) / len(r_tokens)
        markers = {
            "knowledge": ["policy", "rule", "spec", "政策", "规格"],
            "tool": ["http", "api", "url", "调用"],
            "clarification": ["?", "请", "could", "what", "where"],
            "style": ["sorry", "thank", "感谢", "抱歉"],
        }[dim]
        ref_has_marker = any(m in ref.lower() for m in markers)
        actual_has_marker = any(m in actual.lower() for m in markers)
        if ref_has_marker and not actual_has_marker:
            return "high"
        if jaccard < 0.2:
            return "medium"
        if jaccard < 0.5:
            return "low"
        return "none"


class SkillDiagnostician:
    """把聚合后的失败映射到 SKILL.md 具体段落,产出 OptimizationPlan."""

    def diagnose(self, records: list[FailureRecord], skill: Skill) -> dict[str, list[str]]:
        del skill  # 简化版按类别计数,生产版用 ReAct 读 skill.skill_md 段落.
        plan: dict[str, list[str]] = defaultdict(list)
        category_count = Counter(r.primary_category for r in records if r.primary_category != "none")
        for category, count in category_count.most_common():
            if count == 0:
                continue
            section, action = {
                "knowledge": ("## 1. Background Knowledge", f"Add or update {count} missing/incorrect items"),
                "tool": ("## 5. Reference Index", f"Add or fix tool schemas for {count} cases"),
                "clarification": ("## 3. Per-Scenario Handling", f"Insert clarification step for {count} cases"),
                "style": ("## 3. Per-Scenario Handling", f"Soften tone in {count} response templates"),
            }[category]
            plan[section].append(action)
        return dict(plan)


class SkillOptimizer:
    """按 minimal-modification 改写 SKILL.md;返回新版本 Skill."""

    def optimize(self, skill: Skill, plan: dict[str, list[str]]) -> Skill:
        new_md = skill.skill_md
        for section, actions in plan.items():
            marker = section
            if marker not in new_md:
                new_md += f"\n\n{marker}\n"
            insertion = "\n".join(f"- AUTO-FIX (v{skill.version + 1}): {a}" for a in actions)
            new_md = new_md.replace(marker, f"{marker}\n{insertion}", 1)
        return Skill(skill_md=new_md, tools=dict(skill.tools), references=dict(skill.references), version=skill.version + 1)


class VirtualFS:
    """版本化 SKILL 快照,支持回滚."""

    def __init__(self) -> None:
        self.snapshots: list[Skill] = []

    def commit(self, skill: Skill) -> int:
        self.snapshots.append(skill)
        return len(self.snapshots) - 1

    def rollback(self, version: int) -> Skill:
        return self.snapshots[version]


# Orchestrator ---------------------------------------------------------------


@dataclass
class SkillForge:
    workflow_miner: WorkflowMiner = field(default_factory=WorkflowMiner)
    tool_miner: ToolMiner = field(default_factory=ToolMiner)
    knowledge_extractor: KnowledgeExtractor = field(default_factory=KnowledgeExtractor)
    synthesizer: SkillSynthesizer = field(default_factory=SkillSynthesizer)
    failure_analyzer: FailureAnalyzer = field(default_factory=FailureAnalyzer)
    diagnostician: SkillDiagnostician = field(default_factory=SkillDiagnostician)
    optimizer: SkillOptimizer = field(default_factory=SkillOptimizer)
    vfs: VirtualFS = field(default_factory=VirtualFS)

    def create_initial_skill(
        self,
        tickets: list[TicketRecord],
        kb_docs: dict[str, str],
        tool_registry: dict[str, ToolSchema],
        scenario_name: str,
    ) -> Skill:
        workflows = self.workflow_miner.mine(tickets)
        tools = self.tool_miner.mine(tickets, tool_registry)
        knowledge = self.knowledge_extractor.extract(tickets, kb_docs)
        skill = self.synthesizer.synthesize(workflows, tools, knowledge, scenario_name)
        self.vfs.commit(skill)
        return skill

    def evolve(self, skill: Skill, bad_cases: list[BadCase], iterations: int = 3) -> Skill:
        current = skill
        for _ in range(iterations):
            records = [self.failure_analyzer.analyze(c) for c in bad_cases]
            plan = self.diagnostician.diagnose(records, current)
            if not plan:
                break
            current = self.optimizer.optimize(current, plan)
            self.vfs.commit(current)
        return current


# Demo ----------------------------------------------------------------------


def _demo_tickets() -> list[TicketRecord]:
    return [
        TicketRecord(
            ticket_id="T001",
            category="mb_diaper_size",
            dialogue=[
                "Customer: 我家宝宝6kg,纸尿裤选M还是L?",
                "Agent: 请问宝宝月龄?",
                "Agent: 6kg属S/M过渡,建议M.参考 https://kb.example/diaper-sizing",
            ],
            tools_used=["product_spec_lookup", "weight_chart_calculator"],
            resolution="recommend M size based on 6kg + 3mo",
        ),
        TicketRecord(
            ticket_id="T002",
            category="mb_diaper_size",
            dialogue=[
                "Customer: 7.5kg应该穿M还是L?",
                "Agent: 请确认月龄和品牌",
                "Agent: 建议L,过渡更舒适. https://kb.example/diaper-sizing",
            ],
            tools_used=["product_spec_lookup"],
            resolution="recommend L size",
        ),
        TicketRecord(
            ticket_id="T003",
            category="mb_baby_formula",
            dialogue=[
                "Customer: 这款奶粉过敏可以退么?",
                "Agent: 请提供过敏症状和购买批次号",
                "Agent: 已为您处理退款.参考 https://kb.example/return-policy",
            ],
            tools_used=["batch_query", "return_initiator"],
            resolution="refund initiated",
        ),
    ]


def _demo_kb() -> dict[str, str]:
    return {
        "diaper_sizing_chart": "S:0-5kg, M:6-8kg, L:9-12kg, XL:13kg+",
        "formula_return_policy": "Allergy returns within 30d with batch number and photo.",
    }


def _demo_tool_registry() -> dict[str, ToolSchema]:
    return {
        "product_spec_lookup": ToolSchema(
            name="product_spec_lookup",
            description="Lookup product SKU specifications",
            parameters={"sku": "string"},
            usage_frequency=0,
        ),
        "weight_chart_calculator": ToolSchema(
            name="weight_chart_calculator",
            description="Suggest size based on baby weight + age",
            parameters={"weight_kg": "float", "age_months": "int"},
            usage_frequency=0,
        ),
        "batch_query": ToolSchema(
            name="batch_query",
            description="Query product batch for recalls or allergens",
            parameters={"batch_id": "string"},
            usage_frequency=0,
        ),
        "return_initiator": ToolSchema(
            name="return_initiator",
            description="Initiate cross-border return process",
            parameters={"order_id": "string", "reason_code": "string"},
            usage_frequency=0,
        ),
    }


def _demo_bad_cases() -> list[BadCase]:
    return [
        BadCase(
            case_id="B01",
            user_query="过敏退款流程要多久?",
            agent_response="我们会尽快处理.",
            reference_response="过敏退款需要提供批次号和症状照片,审核后 3-5 工作日.参考 https://kb.example/return-policy",
        ),
        BadCase(
            case_id="B02",
            user_query="7kg选什么尺码?",
            agent_response="请联系人工客服.",
            reference_response="建议M码.可参考体重表,7kg对应M.",
        ),
    ]


def main() -> None:
    forge = SkillForge()
    tickets = _demo_tickets()
    kb = _demo_kb()
    registry = _demo_tool_registry()

    print("=== Stage 1: Create initial skill from historical tickets ===")
    skill_v0 = forge.create_initial_skill(tickets, kb, registry, scenario_name="mb_customer_service")
    print(skill_v0.render())

    print("\n=== Stage 2: Evolve from bad cases ===")
    bad_cases = _demo_bad_cases()
    skill_v3 = forge.evolve(skill_v0, bad_cases, iterations=3)
    print(f"Final version: v{skill_v3.version}")
    print(skill_v3.skill_md[-500:])

    print("\n=== VFS history ===")
    for i, snapshot in enumerate(forge.vfs.snapshots):
        print(f"v{i}: skill_md {len(snapshot.skill_md)} chars, tools={list(snapshot.tools)}")


def test_pipeline() -> None:
    forge = SkillForge()
    tickets = _demo_tickets()
    skill = forge.create_initial_skill(tickets, _demo_kb(), _demo_tool_registry(), scenario_name="test")
    assert skill.version == 0
    assert len(skill.tools) >= 1, "expected at least one tool above threshold"
    assert "Background Knowledge" in skill.skill_md

    evolved = forge.evolve(skill, _demo_bad_cases(), iterations=2)
    assert evolved.version > skill.version, "evolution must produce a new version"
    assert "AUTO-FIX" in evolved.skill_md, "optimizer must insert fix markers"
    assert len(forge.vfs.snapshots) == evolved.version + 1, "VFS history mismatch"

    analyzer = FailureAnalyzer()
    rec = analyzer.analyze(_demo_bad_cases()[0])
    assert rec.primary_category in DIM_PRIORITY, "primary_category must be one of known dims"
    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()
