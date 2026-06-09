"""ACON: Agent Context Optimization with failure-driven NL guideline tuning.

参考论文:Kang, M. et al. (2025) ACON: Optimizing Context Compression for
Long-horizon LLM Agents. arxiv:2510.00615 (KAIST + Microsoft).

本实现是简化版:
- LLM 调用用 mock (rule-based) 替代;生产替换为 GPT-4.1 / Claude / Qwen3-Max
- 失败驱动准则优化用 contrastive feedback 思路 + 简化的 keyword retention rule
- 蒸馏接口预留,实际训练用 LoRA + cross-entropy loss

核心抽象:
- HistoryCompressor / ObservationCompressor (两类压缩器)
- GuidelineOptimizer (UT + CO 两阶段)
- TrajectoryCollector (收集 success/failure pair)
- Acon (统一 orchestrator)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# Data structures -----------------------------------------------------------


@dataclass
class InteractionTurn:
    step: int
    role: str  # "user" / "agent" / "tool"
    content: str

    def token_count(self) -> int:
        return _approx_tokens(self.content)


@dataclass
class Trajectory:
    task_id: str
    turns: list[InteractionTurn]
    success: bool
    final_response: str = ""

    def total_tokens(self) -> int:
        return sum(t.token_count() for t in self.turns)


@dataclass
class CompressionGuideline:
    """ACON 的核心可优化对象:自然语言压缩准则."""

    instruction: str
    must_preserve_patterns: list[str] = field(default_factory=list)  # 必保留的关键词
    can_summarize_patterns: list[str] = field(default_factory=list)  # 可摘要的内容
    version: int = 0


# Token estimator (simplified BPE 估算) ---------------------------------------


def _approx_tokens(text: str) -> int:
    """简化估算:英文按词,中文按字,总和 × 0.75 系数估算 BPE token."""
    words = re.findall(r"[A-Za-z0-9]+|[一-鿿]", text)
    return int(len(words) * 0.75) + 1


# Compressors ---------------------------------------------------------------


class HistoryCompressor:
    """阈值触发的历史压缩. 简化版用 rule-based 摘要替代 LLM."""

    def __init__(self, guideline: CompressionGuideline, threshold_tokens: int = 4096) -> None:
        self.guideline = guideline
        self.threshold = threshold_tokens

    def compress(self, history: list[InteractionTurn]) -> list[InteractionTurn]:
        total = sum(t.token_count() for t in history)
        if total <= self.threshold:
            return history
        return self._apply_guideline(history)

    def _apply_guideline(self, history: list[InteractionTurn]) -> list[InteractionTurn]:
        # 论文 §3.2:压缩历史时,保留 must_preserve 模式;summarize can_summarize 模式
        compressed: list[InteractionTurn] = []
        for turn in history:
            if self._contains_must_preserve(turn.content):
                compressed.append(turn)
            elif self._matches_can_summarize(turn.content):
                summary = f"[SUMMARY] {turn.content[:40]}"
                # 只在摘要确实更短时才替换,避免短 turn 被加 prefix 后变长
                if len(summary) < len(turn.content):
                    compressed.append(InteractionTurn(step=turn.step, role=turn.role, content=summary))
                else:
                    compressed.append(turn)
            else:
                compressed.append(turn)
        return compressed

    def _contains_must_preserve(self, text: str) -> bool:
        return any(p.lower() in text.lower() for p in self.guideline.must_preserve_patterns)

    def _matches_can_summarize(self, text: str) -> bool:
        return any(p.lower() in text.lower() for p in self.guideline.can_summarize_patterns)


class ObservationCompressor:
    """阈值触发的观察压缩. 输入压缩比历史压缩更省 cost (论文 §4.5)."""

    def __init__(self, guideline: CompressionGuideline, threshold_tokens: int = 1024) -> None:
        self.guideline = guideline
        self.threshold = threshold_tokens

    def compress(self, observation: str) -> str:
        if _approx_tokens(observation) <= self.threshold:
            return observation
        return self._apply_guideline(observation)

    def _apply_guideline(self, observation: str) -> str:
        preserved_lines = []
        for line in observation.split("\n"):
            if any(p.lower() in line.lower() for p in self.guideline.must_preserve_patterns):
                preserved_lines.append(line)
        if preserved_lines:
            return "\n".join(preserved_lines) + f"\n[truncated by ACON, original {_approx_tokens(observation)} tokens]"
        # fallback: 截断到 threshold 等量字符,保证压缩后 size 真的更小
        target_chars = self.threshold * 4  # ~1 token ≈ 4 chars
        return observation[:target_chars] + f"\n[truncated by ACON, original {_approx_tokens(observation)} tokens]"


# Guideline optimization (UT + CO 两阶段) -----------------------------------


@dataclass
class TrajectoryPair:
    """contrastive pair: full-context 成功 vs compressed-context 失败."""

    task_id: str
    full_trajectory: Trajectory
    compressed_trajectory: Trajectory


class GuidelineOptimizer:
    """失败驱动的 NL guideline 优化器. 论文 §3.3."""

    def __init__(self, llm_feedback_fn: Optional[Callable[[TrajectoryPair, CompressionGuideline], str]] = None) -> None:
        self.llm_feedback_fn = llm_feedback_fn or self._mock_feedback

    def utility_step(
        self,
        current: CompressionGuideline,
        contrastive_pairs: list[TrajectoryPair],
    ) -> CompressionGuideline:
        """UT 步骤:从失败 trajectory 中学习,加保留模式."""
        new_preserve = list(current.must_preserve_patterns)
        for pair in contrastive_pairs:
            feedback = self.llm_feedback_fn(pair, current)
            extracted = self._extract_patterns_from_feedback(feedback)
            for pattern in extracted:
                if pattern not in new_preserve:
                    new_preserve.append(pattern)
        return CompressionGuideline(
            instruction=current.instruction + " [updated via UT contrastive feedback]",
            must_preserve_patterns=new_preserve,
            can_summarize_patterns=current.can_summarize_patterns,
            version=current.version + 1,
        )

    def compression_step(
        self,
        current: CompressionGuideline,
        successful_compressed: list[Trajectory],
    ) -> CompressionGuideline:
        """CO 步骤:从 success trajectory 中分析哪些信息实际被用,标记可摘要."""
        new_can_summarize = list(current.can_summarize_patterns)
        for traj in successful_compressed:
            extracted = self._extract_unused_patterns(traj, current)
            for pattern in extracted:
                if pattern not in new_can_summarize and pattern not in current.must_preserve_patterns:
                    new_can_summarize.append(pattern)
        return CompressionGuideline(
            instruction=current.instruction + " [updated via CO success analysis]",
            must_preserve_patterns=current.must_preserve_patterns,
            can_summarize_patterns=new_can_summarize,
            version=current.version + 1,
        )

    @staticmethod
    def _mock_feedback(pair: TrajectoryPair, guideline: CompressionGuideline) -> str:
        # 简化版:对比成功与失败,找出 full 中有但 compressed 中没的关键 token
        full_text = " ".join(t.content for t in pair.full_trajectory.turns)
        compressed_text = " ".join(t.content for t in pair.compressed_trajectory.turns)
        full_tokens = set(re.findall(r"[A-Za-z0-9]+|[一-鿿]+", full_text.lower()))
        compressed_tokens = set(re.findall(r"[A-Za-z0-9]+|[一-鿿]+", compressed_text.lower()))
        lost = full_tokens - compressed_tokens
        # 启发式:在 full 中频繁出现但在 compressed 中丢失的就是关键
        full_text_lower = full_text.lower()
        important = sorted(
            (t for t in lost if full_text_lower.count(t) >= 2 and len(t) >= 3),
            key=lambda t: -full_text_lower.count(t),
        )[:5]
        return f"Lost important: {important}"

    @staticmethod
    def _extract_patterns_from_feedback(feedback: str) -> list[str]:
        # 从 mock feedback 中抽出 [a, b, c] 形式
        match = re.search(r"\[(.*?)\]", feedback)
        if not match:
            return []
        items = [s.strip().strip("'\"") for s in match.group(1).split(",")]
        return [s for s in items if s]

    @staticmethod
    def _extract_unused_patterns(traj: Trajectory, guideline: CompressionGuideline) -> list[str]:
        # 简化:统计 traj 文本中频次低的 token 标记为"可摘要"
        text = " ".join(t.content for t in traj.turns)
        text_lower = text.lower()
        tokens = re.findall(r"[A-Za-z0-9]+|[一-鿿]+", text_lower)
        from collections import Counter
        counter = Counter(tokens)
        # 频次 1 且 ≥ 4 chars 的视为"可摘要"
        low_freq = [t for t, c in counter.items() if c == 1 and len(t) >= 4]
        return low_freq[:5]


# Trajectory collector -----------------------------------------------------


class TrajectoryCollector:
    """收集 success/failure trajectory pair 用于 UT 训练."""

    def __init__(self) -> None:
        self.pairs: list[TrajectoryPair] = []

    def add(self, full_traj: Trajectory, compressed_traj: Trajectory) -> None:
        # 论文 §3.3:contrastive pair = full 成功 + compressed 失败
        if full_traj.success and not compressed_traj.success:
            self.pairs.append(TrajectoryPair(
                task_id=full_traj.task_id,
                full_trajectory=full_traj,
                compressed_trajectory=compressed_traj,
            ))


# Distiller (接口预留, 实际训练用 LoRA + cross-entropy) -----------------------


@dataclass
class CompressorDistiller:
    """蒸馏接口. 实际生产用 transformers + LoRA 训练."""

    teacher_guideline: CompressionGuideline

    def collect_training_pairs(
        self,
        success_trajectories: list[Trajectory],
        threshold_tokens: int = 50,
    ) -> list[tuple[str, str]]:
        # 论文 §3.4:训练对 = (raw_input, teacher_compressed_output).
        # threshold 强制低,以确保所有训练样本都被压缩(否则学不到任何东西).
        pairs = []
        teacher_hist = HistoryCompressor(self.teacher_guideline, threshold_tokens=threshold_tokens)
        for traj in success_trajectories:
            if not traj.success:
                continue
            compressed = teacher_hist.compress(traj.turns)
            raw = "\n".join(f"{t.role}: {t.content}" for t in traj.turns)
            compressed_text = "\n".join(f"{t.role}: {t.content}" for t in compressed)
            pairs.append((raw, compressed_text))
        return pairs


# Orchestrator -------------------------------------------------------------


@dataclass
class Acon:
    """ACON 统一编排器. 论文 Figure 3."""

    guideline: CompressionGuideline
    history_threshold: int = 4096
    obs_threshold: int = 1024

    def history_compressor(self) -> HistoryCompressor:
        return HistoryCompressor(self.guideline, self.history_threshold)

    def observation_compressor(self) -> ObservationCompressor:
        return ObservationCompressor(self.guideline, self.obs_threshold)

    def optimize_guideline(
        self,
        contrastive_pairs: list[TrajectoryPair],
        successful_compressed: list[Trajectory],
    ) -> CompressionGuideline:
        optimizer = GuidelineOptimizer()
        # Stage 1: UT
        self.guideline = optimizer.utility_step(self.guideline, contrastive_pairs)
        # Stage 2: CO
        self.guideline = optimizer.compression_step(self.guideline, successful_compressed)
        return self.guideline


# Demo: 母婴客服长对话压缩 ----------------------------------------------------


def _demo_long_conversation() -> list[InteractionTurn]:
    """模拟 30 turn 跨境母婴客服对话,含订单 + 物流 + 过敏咨询."""
    turns = []
    for i in range(15):
        if i % 3 == 0:  # 1/3 含订单/批次(must_preserve)
            turns.extend([
                InteractionTurn(2 * i, "user", f"轮 {i}: 订单号 ORD{1000+i}, 批次 BATCH{i}, 宝宝喝奶粉过敏"),
                InteractionTurn(2 * i + 1, "agent", f"轮 {i}: 已查询订单 ORD{1000+i} 批次 BATCH{i} 无召回记录."),
            ])
        else:  # 2/3 是寒暄/重复(can_summarize 候选)
            turns.extend([
                InteractionTurn(2 * i, "user", f"轮 {i}: 你好 thank you 请问能再确认一下 greeting"),
                InteractionTurn(2 * i + 1, "agent", f"轮 {i}: greeting 没问题, thank you for your patience 我们会尽快回复"),
            ])
    # 一个长 observation
    turns.append(InteractionTurn(
        30, "tool",
        "API_RESPONSE: " + "{order_status: shipped, tracking: DHL12345, weight: 500g, allergen_test: passed, "
        "shipping_address: anonymized}\n" * 50  # 模拟长 JSON 响应
    ))
    return turns


def _demo_full_traj() -> Trajectory:
    return Trajectory(
        task_id="T1",
        turns=_demo_long_conversation(),
        success=True,
        final_response="建议提供过敏症状照片,我们将启动退款流程.参考 https://kb.example/return-policy",
    )


def _demo_compressed_failed_traj() -> Trajectory:
    """模拟一个 compressed context 下失败的 trajectory(关键信息丢失)."""
    turns = [
        InteractionTurn(0, "user", "宝宝过敏"),
        InteractionTurn(1, "agent", "请联系人工"),  # 错误回复,因为压缩丢了订单号 / 批次号
    ]
    return Trajectory(task_id="T1", turns=turns, success=False, final_response="请联系人工")


def main() -> None:
    print("=== ACON: 母婴跨境客服长对话压缩 demo ===\n")

    # 初始 guideline (人工 prompt)
    initial_guideline = CompressionGuideline(
        instruction="压缩客服对话历史,保留关键事件",
        must_preserve_patterns=["订单", "批次"],
        can_summarize_patterns=["问候", "thank"],
    )

    acon = Acon(guideline=initial_guideline, history_threshold=200)  # 低阈值便于 demo

    # 原始 trajectory
    full_traj = _demo_full_traj()
    print(f"原始 trajectory: {len(full_traj.turns)} turns, {full_traj.total_tokens()} tokens")

    # 应用初始压缩
    compressor = acon.history_compressor()
    compressed_turns = compressor.compress(full_traj.turns)
    compressed_total = sum(t.token_count() for t in compressed_turns)
    print(f"初始压缩后: {len(compressed_turns)} turns, {compressed_total} tokens")
    print(f"压缩率: {(1 - compressed_total / full_traj.total_tokens()) * 100:.1f}%")

    # 收集 contrastive pair (full 成功, compressed 失败)
    collector = TrajectoryCollector()
    collector.add(full_traj, _demo_compressed_failed_traj())
    print(f"\n收集到 {len(collector.pairs)} 个 contrastive pair (UT 训练数据)")

    # UT + CO 优化 guideline
    print("\n=== 执行 UT + CO 准则优化 ===")
    optimized = acon.optimize_guideline(
        contrastive_pairs=collector.pairs,
        successful_compressed=[full_traj],
    )
    print(f"优化后 guideline 版本: v{optimized.version}")
    print(f"must_preserve 模式: {optimized.must_preserve_patterns}")
    print(f"can_summarize 模式: {optimized.can_summarize_patterns}")

    # 用优化后 guideline 重新压缩
    new_compressor = acon.history_compressor()
    new_compressed = new_compressor.compress(full_traj.turns)
    new_total = sum(t.token_count() for t in new_compressed)
    print(f"\n优化后压缩: {len(new_compressed)} turns, {new_total} tokens")
    print(f"压缩率: {(1 - new_total / full_traj.total_tokens()) * 100:.1f}%")

    # Observation compression demo
    print("\n=== Observation Compression demo ===")
    obs_compressor = acon.observation_compressor()
    long_obs = full_traj.turns[-1].content  # API 长响应
    print(f"原始 observation: {_approx_tokens(long_obs)} tokens")
    compressed_obs = obs_compressor.compress(long_obs)
    print(f"压缩后 observation: {_approx_tokens(compressed_obs)} tokens")
    print(f"压缩率: {(1 - _approx_tokens(compressed_obs) / _approx_tokens(long_obs)) * 100:.1f}%")

    # Distillation training pair collection demo
    print("\n=== Distillation 训练对收集 demo ===")
    distiller = CompressorDistiller(teacher_guideline=optimized)
    pairs = distiller.collect_training_pairs([full_traj])
    print(f"收集到 {len(pairs)} 个训练对 (raw → compressed)")
    for raw, comp in pairs[:1]:
        print(f"\n  raw 长度: {len(raw)} chars")
        print(f"  compressed 长度: {len(comp)} chars")


def test_pipeline() -> None:
    guideline = CompressionGuideline(
        instruction="test",
        must_preserve_patterns=["订单", "批次"],
        can_summarize_patterns=["greeting"],
    )

    # HistoryCompressor: 阈值未触发
    short_history = [InteractionTurn(0, "user", "hi"), InteractionTurn(1, "agent", "hello")]
    hc = HistoryCompressor(guideline, threshold_tokens=10000)
    assert hc.compress(short_history) == short_history, "below threshold should not compress"

    # HistoryCompressor: 阈值触发,must_preserve 保留
    long_history = _demo_long_conversation()
    hc_low = HistoryCompressor(guideline, threshold_tokens=50)
    compressed = hc_low.compress(long_history)
    assert any("订单" in t.content for t in compressed), "must_preserve should keep 订单 turns"

    # ObservationCompressor
    long_obs = "\n".join(["noise line " + str(i) for i in range(5000)]) + "\n订单 ORD1234 关键信息"
    oc = ObservationCompressor(guideline, threshold_tokens=100)
    compressed_obs = oc.compress(long_obs)
    assert "订单" in compressed_obs, "must_preserve key info should be retained in observation"
    assert len(compressed_obs) < len(long_obs), "compressed observation must be smaller"

    # GuidelineOptimizer UT
    collector = TrajectoryCollector()
    collector.add(_demo_full_traj(), _demo_compressed_failed_traj())
    assert len(collector.pairs) == 1

    optimizer = GuidelineOptimizer()
    new_guideline = optimizer.utility_step(guideline, collector.pairs)
    assert new_guideline.version == guideline.version + 1
    assert len(new_guideline.must_preserve_patterns) >= len(guideline.must_preserve_patterns)

    # GuidelineOptimizer CO
    co_guideline = optimizer.compression_step(new_guideline, [_demo_full_traj()])
    assert co_guideline.version == new_guideline.version + 1

    # Acon orchestrator
    acon = Acon(guideline=guideline, history_threshold=50)
    optimized = acon.optimize_guideline(collector.pairs, [_demo_full_traj()])
    assert optimized.version >= 2, "should go through both UT and CO"

    # Distiller
    distiller = CompressorDistiller(teacher_guideline=optimized)
    train_pairs = distiller.collect_training_pairs([_demo_full_traj()], threshold_tokens=20)
    assert len(train_pairs) == 1
    assert len(train_pairs[0][1]) <= len(train_pairs[0][0]), "compressed should not be longer than raw"

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()
