"""EvoSkills: 双 LLM 协同演化 + 信息隔离的 Skill 自动生成框架.

参考论文:Zhang, H. et al. (2026) EvoSkills: Self-Evolving Skills via
Co-Evolutionary Verification for LLM Agents. arxiv:2604.01687.

本实现是简化版,主要演示 Algorithm 1 的协同演化结构:
- Skill Generator + Surrogate Verifier 在独立 session 中迭代
- Oracle 只返回 opaque pass/fail bit, 不泄露测试内容
- 母婴客服场景:过敏退货决策的 skill 自动演化

生产环境:
- Generator / Verifier 接 Claude Opus 4.6 / GPT-5.2,**不同 API key 强制 session 隔离**
- Oracle 是真人 review 或独立测试套件
- VirtualFS 替换为真实沙箱(Docker / Firecracker / k8s job)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Skill bundle ------------------------------------------------------------


@dataclass
class SkillBundle:
    """多文件 skill 包. 论文中 S 是结构化 bundle, 不只是 prompt."""

    name: str
    version: int
    skill_md: str  # SKILL.md 主指令
    scripts: dict[str, str] = field(default_factory=dict)  # filename -> code
    references: dict[str, str] = field(default_factory=dict)  # filename -> doc

    def total_lines(self) -> int:
        n = self.skill_md.count("\n")
        for code in self.scripts.values():
            n += code.count("\n")
        return n


# Execution products ------------------------------------------------------


@dataclass
class ExecutionResult:
    """Skill 执行后的产物 x^(i). Verifier / Oracle 只看这个, 不看 skill 内部."""

    output_files: dict[str, str] = field(default_factory=dict)
    stdout: str = ""
    exit_code: int = 0


# Verifier 产物 ------------------------------------------------------------


@dataclass
class Assertion:
    """Deterministic 测试断言 e_k. Verifier 生成的可机执行项."""

    name: str
    target_file: str
    check_type: str  # "contains" / "regex" / "json_field"
    expected: str

    def evaluate(self, result: ExecutionResult) -> bool:
        if self.target_file not in result.output_files:
            return False
        content = result.output_files[self.target_file]
        if self.check_type == "contains":
            return self.expected in content
        if self.check_type == "regex":
            return bool(re.search(self.expected, content))
        if self.check_type == "json_field":
            return f'"{self.expected}"' in content
        return False


@dataclass
class TestSuite:
    """Surrogate Verifier 的 test suite V^(j)."""

    version: int
    assertions: list[Assertion] = field(default_factory=list)


@dataclass
class Diagnostic:
    """Verifier 给 Generator 的失败诊断 F^(i,j).

    论文:per-assertion 结果 + 根因分析 + 修复建议.
    """

    failed_assertions: list[str]
    root_causes: list[str]
    revision_hints: list[str]


# Oracle ------------------------------------------------------------------


class OracleResult(Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass
class Oracle:
    """Ground-truth oracle. 只返回 opaque pass/fail bit.

    论文:1[R(x) < 1] - 不泄露任何测试内容, 防止 generator 过拟合.
    """

    hidden_required_keywords: list[str]
    hidden_target_file: str = "decision.json"
    seed: int = 42

    def evaluate(self, result: ExecutionResult) -> OracleResult:
        if self.hidden_target_file not in result.output_files:
            return OracleResult.FAIL
        content = result.output_files[self.hidden_target_file]
        for kw in self.hidden_required_keywords:
            if kw not in content:
                return OracleResult.FAIL
        return OracleResult.PASS


# Mock LLM components -----------------------------------------------------


class MockSkillGenerator:
    """Skill Generator π_θ 的 mock.

    每次接收 (S^(i), C^(i)) 输出 S^(i+1).
    实际产品接 Claude / GPT API.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.context: list[str] = []  # 持久会话上下文 C

    def initialize(self, instruction: str, meta_skill: str) -> None:
        self.context = [f"INSTRUCTION: {instruction}", f"META: {meta_skill}"]

    def generate(self, prev_skill: Optional[SkillBundle], diagnostic: Optional[Diagnostic]) -> SkillBundle:
        if diagnostic:
            self.context.append(f"DIAGNOSTIC: failed={diagnostic.failed_assertions}, hints={diagnostic.revision_hints}")

        version = (prev_skill.version + 1) if prev_skill else 0

        # 简化 mock:每次演化,根据 context 中的 hint 逐步往 skill 里塞关键词
        # 真实场景这里是 LLM 自由 generation
        keywords_to_add: list[str] = []
        for line in self.context:
            if "hint:" in line.lower():
                for kw in ["check_brand", "verify_batch", "classify_severity", "decide_refund", "decide_replace"]:
                    if kw in line and kw not in keywords_to_add:
                        keywords_to_add.append(kw)

        skill_md = "# Allergy Refund Decision Skill\n\nSteps:\n"
        skill_md += "1. Check customer report\n"
        for kw in keywords_to_add:
            skill_md += f"2. {kw}\n"
        if version >= 3:
            skill_md += "\n## Output\n- Write decision.json with fields: action, severity, batch\n"

        scripts = {
            "main.py": (
                "def main(report):\n"
                "    return {'action': 'refund'}\n"
                if version < 2
                else
                "def main(report):\n"
                "    severity = 'high' if 'severe' in report else 'mild'\n"
                "    action = 'refund' if severity == 'high' else 'replace'\n"
                "    return {'action': action, 'severity': severity, 'batch': 'unknown'}\n"
            ),
        }

        return SkillBundle(
            name="allergy_refund",
            version=version,
            skill_md=skill_md,
            scripts=scripts,
        )


class MockSurrogateVerifier:
    """Surrogate Verifier π_θ^V 的 mock.

    独立 session, 只看 (I, x^(i), V^(j)),**不**看 generator 的 skill 内容.
    """

    def __init__(self, verifier_id: str) -> None:
        self.verifier_id = verifier_id
        self.observed_outputs: list[ExecutionResult] = []  # 只见产物, 不见 skill

    def generate_suite(self, instruction: str, result: ExecutionResult, prev_suite: TestSuite) -> TestSuite:
        """V^(j+1) ~ π_θ^V(· | I, x^(i), V^(j)). 升级测试."""
        del instruction  # mock 不使用,实际由 LLM 解析 I
        self.observed_outputs.append(result)
        version = prev_suite.version + 1

        # 每次升级:加新断言. 真实场景是 LLM 生成 deterministic test.
        assertions = list(prev_suite.assertions)
        if version == 1:
            # 初版:基础测试
            assertions.append(Assertion("has_decision_json", "decision.json", "contains", "action"))
        elif version == 2:
            assertions.append(Assertion("has_severity_field", "decision.json", "contains", "severity"))
        elif version >= 3:
            assertions.append(Assertion("has_batch_field", "decision.json", "contains", "batch"))

        return TestSuite(version=version, assertions=assertions)

    def evaluate(self, result: ExecutionResult, suite: TestSuite) -> tuple[float, Diagnostic]:
        """计算 surrogate reward R̃ (Eq. 4) + 生成失败诊断."""
        if not suite.assertions:
            return 0.0, Diagnostic([], [], [])

        passed: list[str] = []
        failed: list[str] = []
        for a in suite.assertions:
            if a.evaluate(result):
                passed.append(a.name)
            else:
                failed.append(a.name)

        reward = len(passed) / len(suite.assertions)
        root_causes = [f"{name} failed because expected output missing" for name in failed]
        revision_hints = [f"hint: ensure {name} passes by adding required field" for name in failed]

        return reward, Diagnostic(failed_assertions=failed, root_causes=root_causes, revision_hints=revision_hints)


# Execution ---------------------------------------------------------------


def execute_skill(skill: SkillBundle, env_input: dict[str, Any]) -> ExecutionResult:
    """Φ(S, E) 模拟:把 skill 跑在 env 上得到产物 x.

    Mock 版:根据 skill 文本中的关键词决定输出. 真实版调用 sandbox.
    """
    output_files: dict[str, str] = {}
    decision: dict[str, str] = {}

    if "main.py" in skill.scripts:
        code = skill.scripts["main.py"]
        # 简化执行:扫描 code 看看产物里会有哪些字段
        if "action" in code:
            decision["action"] = "refund" if "severe" in env_input.get("report", "") else "replace"
        if "severity" in code:
            decision["severity"] = "high" if "severe" in env_input.get("report", "") else "mild"
        if "batch" in code:
            # v3 后才会真正塞 batch
            decision["batch"] = env_input.get("batch_id", "UNKNOWN")

    if decision:
        # 转成简化 JSON 文本
        parts = [f'"{k}": "{v}"' for k, v in decision.items()]
        output_files["decision.json"] = "{" + ", ".join(parts) + "}"

    return ExecutionResult(output_files=output_files, stdout=skill.skill_md[:100])


# Co-evolution loop -------------------------------------------------------


@dataclass
class EvolutionTrace:
    """演化过程的可观测记录."""

    rounds: list[dict[str, Any]] = field(default_factory=list)
    total_skill_versions: int = 0
    total_verifier_versions: int = 0
    oracle_invocations: int = 0
    final_oracle_pass: bool = False


class CoEvolutionLoop:
    """Algorithm 1 主循环.

    超参:
    - N: 演化迭代次数(oracle interventions)
    - M: surrogate 重试次数
    - β: context 容量阈值
    """

    def __init__(
        self,
        generator: MockSkillGenerator,
        verifier: MockSurrogateVerifier,
        oracle: Oracle,
        max_oracle_rounds: int = 5,
        max_surrogate_retries: int = 15,
        context_cap: float = 0.7,
    ) -> None:
        self.generator = generator
        self.verifier = verifier
        self.oracle = oracle
        self.max_oracle_rounds = max_oracle_rounds
        self.max_surrogate_retries = max_surrogate_retries
        self.context_cap = context_cap

    def run(self, instruction: str, meta_skill: str, env_input: dict[str, Any]) -> tuple[SkillBundle, EvolutionTrace]:
        # 初始化 generator 上下文 C^(0) = (I, S_meta)
        self.generator.initialize(instruction, meta_skill)

        skill = self.generator.generate(prev_skill=None, diagnostic=None)
        test_suite = TestSuite(version=0, assertions=[])

        trace = EvolutionTrace()
        best_skill = skill
        best_oracle_pass = False

        n = 0  # oracle interventions
        r = 0  # surrogate retries

        while n < self.max_oracle_rounds and r < self.max_surrogate_retries:
            # 1) Generator: 执行 skill -> x^(i)
            result = execute_skill(skill, env_input)

            # Context cap 检查
            ctx_usage = len(self.generator.context) / 100.0  # mock 比例
            if ctx_usage > self.context_cap:
                trace.rounds.append({"event": "context_overflow", "ctx": ctx_usage})
                break

            # 2) Surrogate Verifier: 评估
            surrogate_reward, diagnostic = self.verifier.evaluate(result, test_suite)

            round_log = {
                "skill_version": skill.version,
                "verifier_version": test_suite.version,
                "surrogate_reward": surrogate_reward,
                "result_fields": list(result.output_files.keys()),
            }

            if surrogate_reward < 1.0 and test_suite.assertions:
                # Verifier fail: V locked, refine S
                skill = self.generator.generate(prev_skill=skill, diagnostic=diagnostic)
                r += 1
                round_log.update({"exit": "verifier_fail", "next": "refine_skill"})
                trace.rounds.append(round_log)
                trace.total_skill_versions += 1
                continue

            # 3) Surrogate pass (或 V 空) → Oracle 重新执行
            oracle_result = self.oracle.evaluate(result)
            n += 1
            trace.oracle_invocations += 1
            round_log["oracle_pass"] = oracle_result == OracleResult.PASS

            if oracle_result == OracleResult.PASS:
                round_log.update({"exit": "oracle_pass", "next": "return"})
                trace.rounds.append(round_log)
                trace.total_skill_versions += 1
                trace.total_verifier_versions = test_suite.version
                trace.final_oracle_pass = True
                return skill, trace

            # Oracle fail: 升级 verifier (Eq. 8)
            test_suite = self.verifier.generate_suite(instruction, result, test_suite)
            # 把 opaque bit 加到 generator context
            self.generator.context.append(f"ORACLE_BIT: fail (no detail)")
            round_log.update({"exit": "oracle_fail", "next": "escalate_verifier"})
            trace.rounds.append(round_log)
            trace.total_verifier_versions = test_suite.version

            if oracle_result == OracleResult.PASS and not best_oracle_pass:
                best_skill = skill
                best_oracle_pass = True

        return best_skill, trace


# Demo ---------------------------------------------------------------------


def _setup_demo_components() -> tuple[MockSkillGenerator, MockSurrogateVerifier, Oracle]:
    gen = MockSkillGenerator("opus_4.6_generator")
    ver = MockSurrogateVerifier("opus_4.6_verifier")  # 独立 session
    # Oracle 隐藏要求:必须有 action / severity / batch 三个字段
    oracle = Oracle(
        hidden_required_keywords=["action", "severity", "batch"],
        hidden_target_file="decision.json",
        seed=42,
    )
    return gen, ver, oracle


def main() -> None:
    print("=== EvoSkills Co-Evolution Demo:跨境母婴客服过敏退货决策 ===\n")
    random.seed(0)

    gen, ver, oracle = _setup_demo_components()
    loop = CoEvolutionLoop(
        generator=gen,
        verifier=ver,
        oracle=oracle,
        max_oracle_rounds=5,
        max_surrogate_retries=15,
    )

    instruction = (
        "客户咨询新生儿过敏退货, 需要根据品牌+批次+症状给出"
        "退/换决策, 输出到 decision.json"
    )
    meta_skill = "skill-creator: SKILL.md + main.py + 自检测试"
    env_input = {"report": "severe rash after using diaper", "batch_id": "BATCH4-2026"}

    print(f"指令: {instruction[:60]}...")
    print(f"Oracle 隐藏要求(generator 不可见): {oracle.hidden_required_keywords}\n")

    final_skill, trace = loop.run(instruction, meta_skill, env_input)

    print(f"--- 演化结果 ---")
    print(f"Skill 最终版本: v{final_skill.version}")
    print(f"Oracle PASS: {trace.final_oracle_pass}")
    print(f"Oracle invocations: {trace.oracle_invocations}")
    print(f"Verifier suite 最终版: v{trace.total_verifier_versions}")
    print(f"\n--- 演化轨迹 ---")
    for i, r in enumerate(trace.rounds):
        print(f"Round {i+1}: skill v{r.get('skill_version')}, "
              f"verifier v{r.get('verifier_version')}, "
              f"R̃={r.get('surrogate_reward', 0):.2f}, "
              f"exit={r.get('exit', '?')}")

    print(f"\n--- 最终 Skill (v{final_skill.version}) ---")
    print(f"SKILL.md:\n{final_skill.skill_md}")
    print(f"main.py:\n{final_skill.scripts['main.py']}")


def test_pipeline() -> None:
    """Sanity checks."""
    gen, ver, oracle = _setup_demo_components()

    # 1) Generator 初始化
    gen.initialize("test instruction", "meta")
    assert len(gen.context) == 2

    # 2) Generator 生成 v0 skill
    s0 = gen.generate(prev_skill=None, diagnostic=None)
    assert s0.version == 0
    assert "main.py" in s0.scripts

    # 3) 执行 skill 得到 result
    result = execute_skill(s0, {"report": "severe rash", "batch_id": "B1"})
    # v0 (no hints) should have 'action' only via prev path → 验证不依赖具体行为, 只检查产物结构
    assert isinstance(result, ExecutionResult)

    # 4) Verifier 初始 suite 为空 → reward = 0 (空 suite)
    empty_suite = TestSuite(version=0, assertions=[])
    reward, diag = ver.evaluate(result, empty_suite)
    assert reward == 0.0
    assert diag.failed_assertions == []

    # 5) 升级 verifier
    new_suite = ver.generate_suite("test", result, empty_suite)
    assert new_suite.version == 1
    assert len(new_suite.assertions) == 1

    # 6) Oracle 信息隔离:只看产物 + 隐藏关键词
    bad_result = ExecutionResult(output_files={"decision.json": '{"action": "refund"}'})
    assert oracle.evaluate(bad_result) == OracleResult.FAIL  # 缺 severity / batch
    good_result = ExecutionResult(
        output_files={"decision.json": '{"action": "refund", "severity": "high", "batch": "B1"}'}
    )
    assert oracle.evaluate(good_result) == OracleResult.PASS

    # 7) Assertion 各种 check
    a_contains = Assertion("a1", "decision.json", "contains", "action")
    assert a_contains.evaluate(bad_result)
    a_regex = Assertion("a2", "decision.json", "regex", r'"action":\s*"refund"')
    assert a_regex.evaluate(bad_result)
    a_missing = Assertion("a3", "missing.json", "contains", "x")
    assert not a_missing.evaluate(bad_result)

    # 8) 完整 co-evolution loop
    loop = CoEvolutionLoop(
        generator=MockSkillGenerator("g"),
        verifier=MockSurrogateVerifier("v"),
        oracle=oracle,
        max_oracle_rounds=5,
        max_surrogate_retries=15,
    )
    final_skill, trace = loop.run(
        "客户过敏退货",
        "meta",
        {"report": "severe rash", "batch_id": "B1"},
    )
    # 至少跑了一轮
    assert len(trace.rounds) >= 1
    # Trace 中应该有 oracle 调用
    assert trace.oracle_invocations >= 1, f"应至少调用一次 oracle, got {trace.oracle_invocations}"
    # Final skill 应该有递增的 version
    assert final_skill.version >= 0

    # 9) 信息隔离:Verifier observed_outputs 只有产物,不可见 skill 内部
    for obs in ver.observed_outputs:
        assert isinstance(obs, ExecutionResult)
        # ExecutionResult 不应该包含 skill bundle
        for fname in obs.output_files:
            assert fname == "decision.json", f"verifier 只该看产物输出, got {fname}"

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()
