"""MemAct: Memory-as-Action 框架 + DCPO 训练算法.

参考论文:Zhang, Y. et al. (2025) Memory-as-Action: Autonomous Context Curation
for Long-Horizon Agentic Tasks. arxiv:2510.12635.

本实现简化版,演示:
- 统一 policy 同时输出 task action + memory action
- WorkingMemory 状态(每条记录有唯一 id)
- prune_context tool (论文 A.1)
- Trajectory segmentation at memory action points (DCPO 核心)
- Group-normalized advantage (GRPO-style)
- 母婴客服 multi-objective demo

生产环境:
- Policy 接 Qwen2.5-7B/14B 微调模型
- DCPO 集成到 veRL / OpenRLHF / DeepSpeed-Chat
- Reward 接业务 success judge (gpt-oss 或自训判别器)
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# Working memory ---------------------------------------------------------


@dataclass
class MemoryRecord:
    """单条记忆记录, 带 unique id 便于 prune."""

    record_id: str
    role: str  # "user" / "action" / "observation" / "memory_op"
    content: str

    def estimate_tokens(self) -> int:
        return max(1, len(self.content) // 4)


@dataclass
class WorkingMemory:
    """H_t: agent 的工作记忆 (论文 Section 2.2)."""

    records: list[MemoryRecord] = field(default_factory=list)

    def append(self, role: str, content: str) -> MemoryRecord:
        rec = MemoryRecord(
            record_id=str(uuid.uuid4())[:8],
            role=role,
            content=content,
        )
        self.records.append(rec)
        return rec

    def prune(self, summary: str, ids_to_prune: list[str]) -> int:
        """prune_context 实现: 删除指定 id 的记录, 用 summary 替换.

        Returns:
            被删除的 record 数量
        """
        to_drop = set(ids_to_prune)
        kept = [r for r in self.records if r.record_id not in to_drop]
        dropped_count = len(self.records) - len(kept)
        # 插入 summary 作为新 record
        summary_rec = MemoryRecord(
            record_id=str(uuid.uuid4())[:8],
            role="memory_op",
            content=f"[PRUNE_SUMMARY] {summary}",
        )
        self.records = kept + [summary_rec]
        return dropped_count

    def total_tokens(self) -> int:
        return sum(r.estimate_tokens() for r in self.records)

    def get_ids(self) -> list[str]:
        return [r.record_id for r in self.records]

    def view(self) -> str:
        parts = []
        for r in self.records:
            parts.append(f"  [{r.record_id}] {r.role}: {r.content[:60]}")
        return "\n".join(parts)


# Actions ----------------------------------------------------------------


class ActionType(Enum):
    TASK = "task"
    MEMORY = "memory"


@dataclass
class Action:
    """统一 Action 抽象, 论文 Section 2.2: A = A_task ∪ A_mem."""

    action_type: ActionType
    name: str
    args: dict[str, Any] = field(default_factory=dict)


def task_action(name: str, **args: Any) -> Action:
    return Action(action_type=ActionType.TASK, name=name, args=args)


def memory_action(name: str, **args: Any) -> Action:
    return Action(action_type=ActionType.MEMORY, name=name, args=args)


# MemAct Agent -----------------------------------------------------------


@dataclass
class StepRecord:
    """trajectory 中一步的记录."""

    step_index: int
    action: Action
    observation: str
    memory_size_before: int
    memory_size_after: int

    @property
    def is_memory_action(self) -> bool:
        return self.action.action_type == ActionType.MEMORY


class MemActAgent:
    """统一 policy: π_θ(a_t | H_t) 同时输出 task / memory action.

    论文 Algorithm 2 的 simplified executor.
    """

    def __init__(
        self,
        agent_id: str,
        policy_fn: Callable[[WorkingMemory, list[StepRecord]], Action],
        env_fn: Callable[[Action], str],
        max_turns: int = 35,  # 论文配置
    ) -> None:
        self.agent_id = agent_id
        self.policy_fn = policy_fn  # 简化版: 用户传入 mock policy
        self.env_fn = env_fn
        self.max_turns = max_turns

    def execute(self, initial_prompt: str) -> "Trajectory":
        memory = WorkingMemory()
        memory.append("user", initial_prompt)

        steps: list[StepRecord] = []
        for t in range(self.max_turns):
            mem_size_before = memory.total_tokens()
            action = self.policy_fn(memory, steps)

            if action.action_type == ActionType.TASK:
                # H_{t+1} = H_t ⊕ (a_t, o_t)
                obs = self.env_fn(action)
                memory.append("action", f"{action.name}({action.args})")
                memory.append("observation", obs)
            else:
                # Memory action: H_{t+1} = a_t(H_t)
                if action.name == "prune_context":
                    summary = action.args.get("summary", "")
                    ids = action.args.get("ids_to_prune", [])
                    dropped = memory.prune(summary, ids)
                    obs = f"pruned {dropped} records"
                elif action.name == "finish":
                    obs = action.args.get("answer", "")
                    memory.append("action", f"finish({action.args})")
                    memory.append("observation", obs)
                    mem_size_after = memory.total_tokens()
                    steps.append(StepRecord(
                        step_index=t,
                        action=action,
                        observation=obs,
                        memory_size_before=mem_size_before,
                        memory_size_after=mem_size_after,
                    ))
                    break
                else:
                    obs = f"unknown memory action: {action.name}"

            mem_size_after = memory.total_tokens()
            steps.append(StepRecord(
                step_index=t,
                action=action,
                observation=obs,
                memory_size_before=mem_size_before,
                memory_size_after=mem_size_after,
            ))

            if action.action_type == ActionType.TASK and action.name == "finish":
                break

        return Trajectory(
            prompt=initial_prompt,
            steps=steps,
            final_memory=memory,
        )


# Trajectory -------------------------------------------------------------


@dataclass
class Trajectory:
    """完整轨迹: prompt + step records + final memory."""

    prompt: str
    steps: list[StepRecord]
    final_memory: WorkingMemory
    reward: float = 0.0

    def memory_action_indices(self) -> list[int]:
        """返回 memory action 发生的 step indices (含起点 -1 和终点 len-1)."""
        indices = [-1]
        for i, s in enumerate(self.steps):
            if s.is_memory_action:
                indices.append(i)
        indices.append(len(self.steps) - 1)
        return indices

    def total_tokens(self) -> int:
        return self.final_memory.total_tokens()

    def num_task_actions(self) -> int:
        return sum(1 for s in self.steps if not s.is_memory_action)

    def num_memory_actions(self) -> int:
        return sum(1 for s in self.steps if s.is_memory_action)


@dataclass
class Segment:
    """DCPO segment: 由 trajectory fracture 点切出的连续子段."""

    trajectory_id: str
    segment_index: int
    prefix_token_count: int  # 该 segment 的 prefix 长度
    generated_token_count: int  # 该 segment 新生成的 token 数
    start_step: int  # 起始 step (排除前一个 memory action 后)
    end_step: int  # 结束 step (含)
    advantage: float = 0.0


def segment_trajectory(traj: Trajectory, traj_id: str) -> list[Segment]:
    """实现论文 Section 2.3.2 trajectory segmentation.

    Memory action 时刻: t^mem_1, ..., t^mem_K
    Segment σ_i 是 (t^mem_i, t^mem_{i+1}] 之间的子段.
    """
    indices = traj.memory_action_indices()
    segments: list[Segment] = []
    for i in range(len(indices) - 1):
        start = indices[i] + 1  # 排除 memory action 本身
        end = indices[i + 1]
        if start > end:
            continue
        # 估算 prefix token = memory state 在 start 时的 tokens
        # 简化:用 start 之前所有 step 的 memory_size_before
        prefix_tokens = traj.steps[start].memory_size_before if start < len(traj.steps) else 0
        # 生成 token = 该段所有 step 的 observation 长度
        gen_tokens = sum(
            max(1, len(traj.steps[j].observation) // 4)
            for j in range(start, end + 1)
            if j < len(traj.steps)
        )
        segments.append(Segment(
            trajectory_id=traj_id,
            segment_index=i,
            prefix_token_count=prefix_tokens,
            generated_token_count=gen_tokens,
            start_step=start,
            end_step=end,
        ))
    return segments


# DCPO Trainer -----------------------------------------------------------


def compute_advantages(trajectories: list[Trajectory]) -> list[float]:
    """GRPO-style group-normalized advantage (论文 Eq. 3).

    A(τ) = (R(τ) - μ_u) / σ_u
    """
    returns = [t.reward for t in trajectories]
    if not returns:
        return []
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / max(1, len(returns) - 1)
    sigma = max(math.sqrt(var), 1e-8)
    return [(r - mu) / sigma for r in returns]


@dataclass
class DCPOTrainer:
    """DCPO 训练算法 (论文 Algorithm 1) 简化版.

    简化:不做真实 SGD, 只演示 segmentation + advantage 计算 + loss 估算.
    """

    n_traj: int = 8  # 论文 N_traj=8
    n_seg: int = 16  # 论文 N_seg=16
    learning_rate: float = 1e-6  # 论文 lr

    def collect_batch(
        self,
        trajectories_per_prompt: dict[str, list[Trajectory]],
    ) -> tuple[list[Segment], dict[str, float]]:
        """从一批 prompt 的 trajectories 收集 segment + advantage map.

        Returns:
            (segment list, advantage map from traj_id to advantage)
        """
        batch_segments: list[Segment] = []
        adv_map: dict[str, float] = {}

        for prompt, trajs in trajectories_per_prompt.items():
            # 1. Advantage estimation (per prompt group)
            advs = compute_advantages(trajs)
            seg_pool: list[Segment] = []

            for i, traj in enumerate(trajs):
                traj_id = f"{prompt}_{i}"
                adv_map[traj_id] = advs[i]

                # 2. Segmentation
                segs = segment_trajectory(traj, traj_id)
                for s in segs:
                    s.advantage = advs[i]  # 每个 segment 继承 trajectory 的 advantage
                seg_pool.extend(segs)

            # 3. Sample N_seg from this prompt's segment pool
            if len(seg_pool) <= self.n_seg:
                sampled = seg_pool
            else:
                # Round-robin: 每个 trajectory 先抽一条, 然后重复
                by_traj: dict[str, list[Segment]] = {}
                for s in seg_pool:
                    by_traj.setdefault(s.trajectory_id, []).append(s)
                sampled = []
                while len(sampled) < self.n_seg:
                    progressed = False
                    for tid in list(by_traj.keys()):
                        if by_traj[tid]:
                            sampled.append(by_traj[tid].pop(0))
                            progressed = True
                            if len(sampled) >= self.n_seg:
                                break
                    if not progressed:
                        break
            batch_segments.extend(sampled)

        return batch_segments, adv_map

    def compute_loss(self, segments: list[Segment]) -> float:
        """计算 policy loss (论文 Eq. 4) 的简化估算.

        实际: L = -E[advantage · log π(y|H)],
        简化版: L = -mean(advantage · gen_tokens) (代理)
        """
        if not segments:
            return 0.0
        # 简化: 用 generated_token_count 代理 log π 总和(假设单 token log prob 为 1)
        per_seg_loss = [
            -s.advantage * s.generated_token_count
            for s in segments
        ]
        return sum(per_seg_loss) / len(per_seg_loss)


# Mock policy & env for demo ----------------------------------------------


def make_mock_policy(max_tokens_before_prune: int = 200) -> Callable[[WorkingMemory, list[StepRecord]], Action]:
    """Mock policy: 模拟多目标 QA 的策略.

    策略:
    - 前 force_prune_at 步执行 task action (search / lookup)
    - context > max_tokens_before_prune 时 prune
    - 8 步后 finish
    """
    objectives = ["allergy_refund", "logistics", "tariff", "alternative_product"]
    objective_index = [0]

    def policy(memory: WorkingMemory, steps: list[StepRecord]) -> Action:
        step = len(steps)

        # 终止条件
        if step >= 20:
            return memory_action("finish", answer="aggregated_answer")

        # 检查是否需要 prune (context 太大且未刚 prune)
        if memory.total_tokens() > max_tokens_before_prune:
            # 不刚刚 prune 过
            recent_prune = (
                steps and steps[-1].is_memory_action
            )
            if not recent_prune and len(memory.records) > 4:
                # 删除前一半 record (除了 user prompt)
                ids = memory.get_ids()
                to_prune = ids[1:len(ids) // 2]
                return memory_action(
                    "prune_context",
                    summary=f"completed sub-objectives: {objectives[:objective_index[0]]}",
                    ids_to_prune=to_prune,
                )

        # 推进目标
        if objective_index[0] < len(objectives):
            obj = objectives[objective_index[0]]
            objective_index[0] += 1
            return task_action(f"answer_{obj}", target=obj)

        # 所有目标完成 → finish
        return memory_action("finish", answer=f"completed: {objectives}")

    return policy


def mock_env(action: Action) -> str:
    """Mock env: 根据 action 返回 long observation."""
    if action.action_type == ActionType.TASK:
        target = action.args.get("target", "unknown")
        return f"detailed_info_for_{target}: " + "lorem ipsum " * 20
    return ""


# Demo ---------------------------------------------------------------------


def run_one_trajectory(reward: float) -> Trajectory:
    """运行一个 trajectory 并赋 reward (mock)."""
    policy = make_mock_policy()
    agent = MemActAgent("memact_demo", policy, mock_env, max_turns=30)
    traj = agent.execute("多目标客服:客户问过敏退货+物流+关税+替代品")
    traj.reward = reward
    return traj


def main() -> None:
    print("=== MemAct + DCPO Demo:跨境客服多目标问答 ===\n")
    random.seed(42)

    # 1) 运行 1 条 trajectory 看效果
    print("--- 单条 trajectory: 不 prune vs prune 对比 ---")

    # 不允许 prune 的 baseline
    def baseline_policy(memory: WorkingMemory, steps: list[StepRecord]) -> Action:
        step = len(steps)
        del memory  # baseline 不操作 memory, 但接口需要此参数
        objectives = ["allergy_refund", "logistics", "tariff", "alternative_product"]
        if step < len(objectives):
            return task_action(f"answer_{objectives[step]}", target=objectives[step])
        return memory_action("finish", answer="done")

    baseline_agent = MemActAgent("baseline", baseline_policy, mock_env)
    baseline_traj = baseline_agent.execute("多目标问答 baseline")
    print(f"Baseline (无 memory action): {baseline_traj.num_task_actions()} task + {baseline_traj.num_memory_actions()} memory, final tokens={baseline_traj.total_tokens()}")

    # 用 MemAct policy
    memact_traj = run_one_trajectory(reward=1.0)
    print(f"MemAct (含 prune): {memact_traj.num_task_actions()} task + {memact_traj.num_memory_actions()} memory, final tokens={memact_traj.total_tokens()}")

    saved = baseline_traj.total_tokens() - memact_traj.total_tokens()
    saved_pct = saved / max(1, baseline_traj.total_tokens()) * 100
    print(f"Token 节省: {saved} ({saved_pct:.1f}%)\n")

    # 2) DCPO segmentation 演示
    print("--- DCPO Trajectory Segmentation ---")
    mem_indices = memact_traj.memory_action_indices()
    print(f"Memory action indices (含 -1, end): {mem_indices}")
    print(f"Trajectory fractures(切分点) = {len(mem_indices) - 2} 个")
    segments = segment_trajectory(memact_traj, "demo_traj_0")
    print(f"切出 {len(segments)} 个 segments:")
    for s in segments:
        print(f"  segment {s.segment_index}: steps {s.start_step}-{s.end_step}, "
              f"prefix_tok={s.prefix_token_count}, gen_tok={s.generated_token_count}")

    # 3) DCPO 训练 batch (8 trajectories)
    print("\n--- DCPO 训练 batch ---")
    trainer = DCPOTrainer(n_traj=8, n_seg=16)
    # 生成 8 条 trajectories, reward 模拟为部分成功
    rewards = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -0.1]
    trajs = [run_one_trajectory(r) for r in rewards]
    trajs_per_prompt = {"prompt_demo": trajs}

    batch_segments, adv_map = trainer.collect_batch(trajs_per_prompt)
    print(f"采样到 {len(batch_segments)} segments")
    print(f"Advantages: {[round(a, 3) for a in adv_map.values()]}")
    loss = trainer.compute_loss(batch_segments)
    print(f"Policy loss (代理): {loss:.4f}")

    # 4) 最终 memory view
    print(f"\n--- 最终 MemAct memory view ({len(memact_traj.final_memory.records)} records) ---")
    print(memact_traj.final_memory.view())


def test_pipeline() -> None:
    """Sanity checks."""

    # 1) WorkingMemory append + prune
    mem = WorkingMemory()
    _r1 = mem.append("user", "test 1")
    r2 = mem.append("observation", "long observation " * 20)
    _r3 = mem.append("observation", "another obs " * 20)
    assert _r1 and _r3  # 确认存在
    assert len(mem.records) == 3
    initial_tokens = mem.total_tokens()
    dropped = mem.prune("summary", [r2.record_id])
    assert dropped == 1
    # prune 后剩下 r1, r3, summary
    assert len(mem.records) == 3, f"应保留 r1+r3+summary, got {len(mem.records)}"
    assert any(r.role == "memory_op" for r in mem.records), "应有 summary record"
    # 总 token 应下降
    assert mem.total_tokens() < initial_tokens

    # 2) Action 创建
    a1 = task_action("search", query="test")
    assert a1.action_type == ActionType.TASK
    a2 = memory_action("prune_context", summary="x", ids_to_prune=["a"])
    assert a2.action_type == ActionType.MEMORY

    # 3) MemActAgent 执行
    policy = make_mock_policy()
    agent = MemActAgent("test", policy, mock_env, max_turns=30)
    traj = agent.execute("多目标问答测试")
    assert len(traj.steps) >= 4, f"应至少 4 步 (4 个 objective), got {len(traj.steps)}"
    assert traj.num_task_actions() >= 4

    # 4) Memory action 应被记录
    mem_count = traj.num_memory_actions()
    assert mem_count >= 1, f"应至少有 1 个 memory action (prune+finish), got {mem_count}"

    # 5) Trajectory segmentation
    mem_indices = traj.memory_action_indices()
    assert len(mem_indices) >= 3, f"应有 -1, memory_actions, end, got {mem_indices}"
    segments = segment_trajectory(traj, "test_traj")
    assert len(segments) >= 1
    # 每个 segment 应有非负 prefix/gen tokens
    for s in segments:
        assert s.prefix_token_count >= 0
        assert s.generated_token_count >= 0

    # 6) Group-normalized advantage
    t1 = run_one_trajectory(1.0)
    t2 = run_one_trajectory(1.0)
    t3 = run_one_trajectory(0.0)
    t4 = run_one_trajectory(-0.1)
    advs = compute_advantages([t1, t2, t3, t4])
    assert len(advs) == 4
    # 高 reward 应有高 advantage
    assert advs[0] > advs[2], f"reward=1 应优于 reward=0, got {advs}"
    assert advs[2] > advs[3], f"reward=0 应优于 reward=-0.1, got {advs}"
    # 均值应接近 0
    assert abs(sum(advs) / len(advs)) < 1e-6, f"normalized 应均值 0, got {sum(advs)/len(advs)}"

    # 7) DCPOTrainer batch collection
    trainer = DCPOTrainer(n_traj=8, n_seg=16)
    trajs_per_prompt = {"p1": [t1, t2, t3, t4]}
    batch, amap = trainer.collect_batch(trajs_per_prompt)
    assert len(batch) > 0
    assert len(amap) == 4
    # advantage 应正确映射
    for s in batch:
        assert s.trajectory_id in amap
        # advantage 应与 trajectory 一致 (一个 traj 的所有 segment 共享 advantage)
        assert s.advantage == amap[s.trajectory_id]

    # 8) DCPO loss 应是数值
    loss = trainer.compute_loss(batch)
    assert isinstance(loss, float)
    # 正面 reward (advantage > 0) 应贡献负 loss (我们最大化 advantage)
    pos_segs = [s for s in batch if s.advantage > 0]
    if pos_segs:
        pos_loss = trainer.compute_loss(pos_segs)
        # -advantage * gen_tokens 在 advantage>0 时为负
        assert pos_loss < 0, f"正 advantage 应给负 loss, got {pos_loss}"

    # 9) Trajectory fracture 实证: prune 前后 memory size 应下降
    found_fracture = False
    for s in traj.steps:
        if s.is_memory_action and s.action.name == "prune_context":
            assert s.memory_size_after < s.memory_size_before, (
                f"prune 后 memory 应缩小, got {s.memory_size_before} → {s.memory_size_after}"
            )
            found_fracture = True
            break
    assert found_fracture, "应检测到至少一次 trajectory fracture"

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()
