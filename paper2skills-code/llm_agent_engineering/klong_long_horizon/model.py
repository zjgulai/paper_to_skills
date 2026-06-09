"""
KLong — 超长时域 Agent 训练：轨迹分割 SFT + 渐进 RL
论文: KLong: Training LLM Agents for Extreme Long-Horizon Tasks
arXiv: 2602.17547 | 2026-02 (v2 2026-04)

核心组件:
- AgentTrajectory: 完整任务轨迹数据结构
- TrajectorySplitter: 带重叠窗口的轨迹分割器（解决 context 限制）
- ProgressiveRLScheduler: 渐进式 RL 课程调度器（+6.67%/stage）
- ResearchFactory: 训练数据自动化生成管线
- KLongTrainer: 完整训练流程（SFT + Progressive RL）

运行方式:
    python model.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ──────────────────────────────────────────
# 1. 轨迹数据结构
# ──────────────────────────────────────────

@dataclass
class Step:
    """单步 Agent 交互"""
    step_id: int
    role: str               # "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    token_count: int = 0


@dataclass
class AgentTrajectory:
    """完整任务轨迹"""
    trajectory_id: str
    task_type: str
    steps: list[Step]
    total_tokens: int
    context_limit: int
    success: bool = False
    reward: float = 0.0

    @property
    def assistant_turns(self) -> int:
        return sum(1 for s in self.steps if s.role == "assistant")

    def exceeds_context(self) -> bool:
        return self.total_tokens > self.context_limit


@dataclass
class SubTrajectory:
    """分割后的子轨迹（重叠窗口机制产物）"""
    parent_id: str
    sub_id: int
    steps: list[Step]
    overlap_with_prev: int      # 与上一个子轨迹的重叠 step 数（保证连贯性）
    is_first: bool
    is_last: bool

    @property
    def assistant_turns(self) -> int:
        return sum(1 for s in self.steps if s.role == "assistant")


# ──────────────────────────────────────────
# 2. 轨迹分割器
# ──────────────────────────────────────────

class TrajectorySplitter:
    """
    带重叠窗口的轨迹分割器。
    核心思路：将超出 context window 的超长轨迹切成有重叠的子轨迹训练。
    重叠机制保证跨子轨迹的上下文连贯性（避免断点处信息丢失）。
    效果：assistant turns 从 114 → 732（6.4×）。
    """

    def __init__(self, window_size: int = 20, overlap_ratio: float = 0.2) -> None:
        if not 0.0 < overlap_ratio < 1.0:
            raise ValueError("overlap_ratio 必须在 (0, 1) 之间")
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.overlap_steps = max(1, int(window_size * overlap_ratio))

    def split_with_overlap(self, trajectory: AgentTrajectory) -> list[SubTrajectory]:
        """
        将轨迹切分为重叠子轨迹。
        步长 stride = window_size - overlap_steps（滑动窗口，确保重叠覆盖）。
        """
        steps = trajectory.steps
        total = len(steps)
        stride = self.window_size - self.overlap_steps

        if total <= self.window_size:
            return [SubTrajectory(
                parent_id=trajectory.trajectory_id,
                sub_id=0,
                steps=steps,
                overlap_with_prev=0,
                is_first=True,
                is_last=True,
            )]

        sub_trajectories: list[SubTrajectory] = []
        start = 0
        sub_id = 0

        while start < total:
            end = min(start + self.window_size, total)
            overlap = self.overlap_steps if sub_id > 0 else 0
            sub_trajectories.append(SubTrajectory(
                parent_id=trajectory.trajectory_id,
                sub_id=sub_id,
                steps=steps[start:end],
                overlap_with_prev=overlap,
                is_first=sub_id == 0,
                is_last=end >= total,
            ))
            if end >= total:
                break
            start += stride
            sub_id += 1

        return sub_trajectories

    def verify_coverage(self, trajectory: AgentTrajectory, subs: list[SubTrajectory]) -> float:
        """
        验证所有 step 被至少一个子轨迹覆盖（覆盖率应为 100%）。
        这是 KLong 分割正确性的核心保证。
        """
        covered = {s.step_id for sub in subs for s in sub.steps}
        all_ids = {s.step_id for s in trajectory.steps}
        return len(covered & all_ids) / max(len(all_ids), 1)


# ──────────────────────────────────────────
# 3. 渐进式 RL 调度器
# ──────────────────────────────────────────

class TrainingStage(str, Enum):
    STAGE_1 = "stage_1"    # 简单短任务（建立基础动作策略）
    STAGE_2 = "stage_2"    # 中等任务（多工具调用）
    STAGE_3 = "stage_3"    # 复杂任务（多跳推理）
    STAGE_4 = "stage_4"    # 超长任务（完整工作流）


@dataclass
class StageConfig:
    stage: TrainingStage
    timeout_minutes: float
    max_steps: int
    difficulty_label: str
    expected_gain_pct: float    # 预期准确率提升（百分比，论文实测 6.67%）


class ProgressiveRLScheduler:
    """
    渐进式 RL 课程调度器（课程学习原理）。
    从短任务开始，逐阶段增加 timeout，Agent 长时域能力持续提升 +6.67%/stage。
    """

    STAGE_CONFIGS = [
        StageConfig(TrainingStage.STAGE_1,  30.0,  20, "简单短任务",          6.67),
        StageConfig(TrainingStage.STAGE_2,  60.0,  35, "中等任务（多工具）",   6.67),
        StageConfig(TrainingStage.STAGE_3,  90.0,  55, "复杂任务（多跳推理）", 6.67),
        StageConfig(TrainingStage.STAGE_4, 120.0,  80, "超长任务（完整工作流）", 6.67),
    ]

    def __init__(self) -> None:
        self._current_stage_idx = 0
        self._stage_accuracies: list[float] = []

    def get_current_timeout(self, training_stage: int | None = None) -> float:
        """获取指定阶段（或当前阶段）的 timeout（分钟）"""
        idx = training_stage if training_stage is not None else self._current_stage_idx
        idx = min(idx, len(self.STAGE_CONFIGS) - 1)
        return self.STAGE_CONFIGS[idx].timeout_minutes

    def advance_stage(self, achieved_accuracy: float) -> bool:
        """记录当前阶段准确率并推进到下一阶段"""
        self._stage_accuracies.append(achieved_accuracy)
        if self._current_stage_idx < len(self.STAGE_CONFIGS) - 1:
            self._current_stage_idx += 1
            return True
        return False

    def cumulative_improvement(self) -> float:
        """累计能力提升估算（百分比）"""
        return self._current_stage_idx * 6.67

    def stage_sequence(self) -> list[dict[str, Any]]:
        """返回完整训练阶段序列"""
        return [
            {
                "stage": cfg.stage.value,
                "timeout_min": cfg.timeout_minutes,
                "max_steps": cfg.max_steps,
                "difficulty": cfg.difficulty_label,
                "expected_gain": f"+{cfg.expected_gain_pct}%",
            }
            for cfg in self.STAGE_CONFIGS
        ]


# ──────────────────────────────────────────
# 4. Research-Factory 自动化数据管线
# ──────────────────────────────────────────

class ResearchFactory:
    """
    训练数据自动化生成管线（Research-Factory）。
    生产环境中应替换为真实任务执行引擎；此处为模拟实现。
    管线：任务生成 → 执行 → 结果校验 → 轨迹过滤 → 分割打包。
    """

    def __init__(self, task_type: str, context_limit: int = 32000) -> None:
        self.task_type = task_type
        self.context_limit = context_limit
        self._counter = 0

    def generate_training_data(
        self,
        num_trajectories: int = 50,
        min_steps: int = 30,
        max_steps: int = 80,
        seed: int = 42,
    ) -> list[AgentTrajectory]:
        """
        模拟批量生成并过滤训练轨迹（只保留成功轨迹）。
        token_count 为每步随机模拟，总量可能超出 context_limit 以触发分割。
        """
        rng = random.Random(seed)
        trajectories = []

        for i in range(num_trajectories):
            n_steps = rng.randint(min_steps, max_steps)
            steps = [
                Step(
                    step_id=j,
                    role="assistant" if j % 3 != 0 else "user",
                    content=f"{self.task_type}_step_{j}",
                    token_count=rng.randint(50, 300),
                )
                for j in range(n_steps)
            ]
            total_tokens = sum(s.token_count for s in steps)
            traj = AgentTrajectory(
                trajectory_id=f"{self.task_type}_{self._counter}_{i}",
                task_type=self.task_type,
                steps=steps,
                total_tokens=total_tokens,
                context_limit=self.context_limit,
                success=rng.random() > 0.3,
                reward=rng.uniform(0.5, 1.0) if rng.random() > 0.3 else 0.0,
            )
            trajectories.append(traj)

        self._counter += num_trajectories
        # 质量过滤：只保留成功轨迹（模拟 Research-Factory 的 Top 20% 过滤）
        return [t for t in trajectories if t.success]


# ──────────────────────────────────────────
# 5. KLong Trainer — 完整训练流程
# ──────────────────────────────────────────

class KLongTrainer:
    """
    KLong 训练器：Stage 1 轨迹分割 SFT + Stage 2 渐进式 RL。
    106B 参数规模，PaperBench 上超越 1T 模型 11.28%。
    """

    def __init__(
        self,
        task_type: str,
        context_limit: int = 32000,
        window_size: int = 20,
        overlap_ratio: float = 0.2,
    ) -> None:
        self.splitter = TrajectorySplitter(window_size, overlap_ratio)
        self.scheduler = ProgressiveRLScheduler()
        self.factory = ResearchFactory(task_type, context_limit)
        self.task_type = task_type

    def stage1_sft_phase(self, num_trajectories: int = 100) -> dict[str, Any]:
        """
        阶段 1：轨迹分割 SFT。
        生成轨迹 → 过滤超长轨迹 → 分割 → 统计 assistant turns 增长（论文：114→732）。
        每个子轨迹的覆盖率必须为 100%（assert 验证）。
        """
        trajectories = self.factory.generate_training_data(num_trajectories)
        long_trajectories = [t for t in trajectories if t.exceeds_context()]

        original_assistant_turns = sum(t.assistant_turns for t in long_trajectories)
        all_subs: list[SubTrajectory] = []

        for traj in long_trajectories:
            subs = self.splitter.split_with_overlap(traj)
            coverage = self.splitter.verify_coverage(traj, subs)
            assert coverage >= 1.0, f"轨迹 {traj.trajectory_id} 覆盖率不足: {coverage:.2%}"
            all_subs.extend(subs)

        new_assistant_turns = sum(s.assistant_turns for s in all_subs)

        return {
            "total_trajectories": len(trajectories),
            "long_trajectories": len(long_trajectories),
            "sub_trajectories": len(all_subs),
            "original_assistant_turns": original_assistant_turns,
            "new_assistant_turns": new_assistant_turns,
            "assistant_turns_ratio": new_assistant_turns / max(original_assistant_turns, 1),
            "coverage_verified": True,
        }

    def stage2_progressive_rl(self, simulate_stages: int = 4) -> dict[str, Any]:
        """
        阶段 2：渐进式 RL（模拟）。
        每阶段 timeout 递增，能力持续提升 +6.67%/stage。
        生产环境替换为真实 RL 训练循环（GRPO/PPO）。
        """
        results = []
        base_accuracy = 0.60

        for stage_idx in range(min(simulate_stages, len(self.scheduler.STAGE_CONFIGS))):
            cfg = self.scheduler.STAGE_CONFIGS[stage_idx]
            achieved = min(base_accuracy + stage_idx * (cfg.expected_gain_pct / 100), 0.99)
            results.append({
                "stage": cfg.stage.value,
                "timeout_min": cfg.timeout_minutes,
                "difficulty": cfg.difficulty_label,
                "achieved_accuracy": round(achieved, 4),
                "improvement": f"+{cfg.expected_gain_pct}%",
            })
            self.scheduler.advance_stage(achieved)

        return {
            "stages_completed": len(results),
            "stage_results": results,
            "cumulative_improvement": f"+{self.scheduler.cumulative_improvement():.1f}%",
            "final_accuracy": results[-1]["achieved_accuracy"] if results else 0.0,
        }


# ──────────────────────────────────────────
# 6. 验证测试：选品调研长轨迹训练
# ──────────────────────────────────────────

def demo_product_selection_training() -> None:
    """
    验证场景：母婴出海选品调研 Agent 训练
    - 验证轨迹分割覆盖率 100%
    - 验证 assistant turns 增长（SFT 效果）
    - 验证渐进 RL 阶段 timeout 单调递增（课程学习合理性）
    """
    print("=== KLong Demo: 母婴选品调研 Agent 训练 ===\n")

    trainer = KLongTrainer(
        task_type="product_selection_research",
        context_limit=8000,     # 低 context limit 触发大量分割，加速验证
        window_size=15,
        overlap_ratio=0.2,
    )

    # Stage 1: 轨迹分割 SFT
    print("【Stage 1: 轨迹分割 SFT】")
    sft = trainer.stage1_sft_phase(num_trajectories=50)
    print(f"  总轨迹数: {sft['total_trajectories']}")
    print(f"  超长轨迹数: {sft['long_trajectories']}")
    print(f"  分割后子轨迹: {sft['sub_trajectories']}")
    print(f"  Assistant turns: {sft['original_assistant_turns']}"
          f" → {sft['new_assistant_turns']}"
          f" (×{sft['assistant_turns_ratio']:.1f})")
    print(f"  覆盖率验证: {'✓ 100%' if sft['coverage_verified'] else '✗'}")

    assert sft["assistant_turns_ratio"] >= 1.0, "SFT 未增加 assistant turns"
    assert sft["coverage_verified"], "轨迹覆盖率验证失败"

    # Stage 2: 渐进式 RL
    print("\n【Stage 2: 渐进式 RL（4阶段）】")
    rl = trainer.stage2_progressive_rl(simulate_stages=4)
    for r in rl["stage_results"]:
        print(f"  {r['stage']}: timeout={r['timeout_min']}min, "
              f"准确率={r['achieved_accuracy']:.2%}, {r['improvement']}")
    print(f"\n  累计能力提升: {rl['cumulative_improvement']}")
    print(f"  最终准确率: {rl['final_accuracy']:.2%}")

    # 验证 timeout 单调递增（课程学习核心约束）
    timeouts = [r["timeout_min"] for r in rl["stage_results"]]
    assert timeouts == sorted(timeouts), "渐进 RL timeout 序列不单调递增"
    print("  ✓ timeout 序列单调递增（课程学习约束满足）")

    print("\n✅ 所有验证通过")


if __name__ == "__main__":
    demo_product_selection_training()
