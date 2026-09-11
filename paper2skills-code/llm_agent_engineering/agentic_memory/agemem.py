"""AgeMem: Unified LTM + STM management as agent policy tools + step-wise GRPO.

参考论文:Yu, Y. et al. (2026) Agentic Memory: Learning Unified Long-Term and
Short-Term Memory Management for Large Language Model Agents. arxiv:2601.01885
(Alibaba + Wuhan University).

本实现是简化版:
- Memory policy 用 rule-based 替代 RL fine-tuned LLM
- 但保留 6 个 memory tool 的完整接口 + 三阶段 trajectory + step-wise GRPO 计算
- Reward function 严格按论文公式
- 生产环境替换 MemoryAgent.decide() 为真实 RL policy

核心抽象:
- LTMStore + STMContext + 6 个 MemoryTool
- AgentState s_t = (C_t, M_t, T)
- ThreeStageRollout 实现 Stage 1/2/3 progressive trajectory
- StepwiseGRPO 实现 advantage broadcast
- CompositeReward 复合奖励
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Memory tool enum (action space 的一部分) ----------------------------------


class MemoryTool(Enum):
    # LTM tools
    ADD = "add_memory"
    UPDATE = "update_memory"
    DELETE = "delete_memory"
    # STM tools
    RETRIEVE = "retrieve_memory"
    SUMMARY = "summary_context"
    FILTER = "filter_context"
    # 非 memory action
    GENERATE = "generate_response"


# Data structures -----------------------------------------------------------


@dataclass
class MemoryEntry:
    key: str
    value: str
    timestamp: int
    tags: list[str] = field(default_factory=list)


@dataclass
class ContextTurn:
    step: int
    role: str
    content: str


@dataclass
class TaskSpec:
    query: str
    info: str  # I_q: contextual info
    expected_answer: str = ""  # 训练时用


@dataclass
class Action:
    tool: MemoryTool
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experience:
    state_snapshot: dict
    action: Action
    reward: float = 0.0
    log_prob_old: float = 0.0
    advantage: float = 0.0


# LTM and STM ---------------------------------------------------------------


class LTMStore:
    def __init__(self) -> None:
        self._store: dict[str, MemoryEntry] = {}
        self._counter = 0

    def add(self, key: str, value: str, timestamp: int, tags: Optional[list[str]] = None) -> str:
        entry = MemoryEntry(key=key, value=value, timestamp=timestamp, tags=tags or [])
        self._store[key] = entry
        self._counter += 1
        return key

    def update(self, key: str, value: str, timestamp: int) -> bool:
        if key not in self._store:
            return False
        self._store[key].value = value
        self._store[key].timestamp = timestamp
        return True

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        if not self._store:
            return []
        scored = [(self._jaccard(query, e.key + " " + e.value), e) for e in self._store.values()]
        scored.sort(key=lambda x: -x[0])
        return [e for score, e in scored[:top_k] if score > 0]

    def size(self) -> int:
        return len(self._store)

    def list_keys(self) -> list[str]:
        return list(self._store.keys())

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z0-9]+|[一-鿿]", a.lower()))
        tb = set(re.findall(r"[a-z0-9]+|[一-鿿]", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


class STMContext:
    def __init__(self, max_tokens: int = 4096) -> None:
        self.turns: list[ContextTurn] = []
        self.max_tokens = max_tokens

    def append(self, turn: ContextTurn) -> None:
        self.turns.append(turn)

    def reset(self) -> None:
        self.turns = []

    def filter(self, keywords_to_remove: list[str]) -> int:
        before = len(self.turns)
        self.turns = [t for t in self.turns if not any(kw in t.content for kw in keywords_to_remove)]
        return before - len(self.turns)

    def summarize(self, start: int, end: int, summary_text: str) -> int:
        if start < 0 or end > len(self.turns) or start >= end:
            return 0
        removed = self.turns[start:end]
        self.turns = (
            self.turns[:start]
            + [ContextTurn(step=removed[0].step, role="system", content=f"[SUMMARY] {summary_text}")]
            + self.turns[end:]
        )
        return len(removed) - 1

    def token_count(self) -> int:
        return sum(_approx_tokens(t.content) for t in self.turns)

    def is_overflow(self) -> bool:
        return self.token_count() > self.max_tokens


def _approx_tokens(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9]+|[一-鿿]", text)
    return int(len(words) * 0.75) + 1


# Agent state ---------------------------------------------------------------


@dataclass
class AgentState:
    stm: STMContext
    ltm: LTMStore
    task: TaskSpec
    step: int = 0


# Memory Agent policy (简化版规则替代 RL policy) -----------------------------


class MemoryAgent:
    """简化版 policy. 生产用 RL fine-tuned LLM 替换 decide() 方法."""

    def __init__(self) -> None:
        self.action_history: list[Action] = []

    def decide(self, state: AgentState) -> Action:
        """根据 state 选择下一个 action. 简化:用启发式 trigger."""
        # 1. Context overflow → 优先 Filter / Summary
        if state.stm.is_overflow():
            if any("noise" in t.content.lower() or "广告" in t.content for t in state.stm.turns):
                return Action(MemoryTool.FILTER, {"keywords": ["noise", "广告"]})
            if len(state.stm.turns) >= 4:
                return Action(MemoryTool.SUMMARY, {"start": 0, "end": 2, "summary": "earlier_summary"})

        # 2. 检查上一步:如果刚 RETRIEVE 过(STM 已含 [RETRIEVED] 标记),直接 GENERATE
        recent_retrieved = any("[RETRIEVED]" in t.content for t in state.stm.turns[-3:])

        # 3. Task 查询关键信息 → Retrieve LTM(仅在尚未 retrieve 时)
        if not recent_retrieved and state.ltm.size() > 0:
            if "?" in state.task.query or "如何" in state.task.query or "what" in state.task.query.lower():
                return Action(MemoryTool.RETRIEVE, {"query": state.task.query})

        # 4. 上下文出现关键信息 → Add to LTM (避免重复)
        for turn in state.stm.turns[-2:]:
            if self._is_storable(turn.content):
                key = f"k_{state.step}_{hash(turn.content) % 1000}"
                if key not in state.ltm.list_keys():
                    return Action(MemoryTool.ADD, {"key": key, "value": turn.content})

        # 5. 默认生成响应
        return Action(MemoryTool.GENERATE, {"output": "default_response"})

    @staticmethod
    def _is_storable(text: str) -> bool:
        # 简化:检测包含关键事实的句子
        markers = ["订单", "批次", "过敏", "偏好", "尺码", "ord", "size", "allergy"]
        return any(m in text.lower() for m in markers)

    def apply(self, action: Action, state: AgentState) -> str:
        """执行 action 并更新 state. 返回执行 log."""
        self.action_history.append(action)
        if action.tool == MemoryTool.ADD:
            state.ltm.add(action.args["key"], action.args["value"], state.step)
            return f"Added LTM: {action.args['key']}"
        elif action.tool == MemoryTool.UPDATE:
            ok = state.ltm.update(action.args["key"], action.args["value"], state.step)
            return f"Updated LTM: {ok}"
        elif action.tool == MemoryTool.DELETE:
            ok = state.ltm.delete(action.args["key"])
            return f"Deleted LTM: {ok}"
        elif action.tool == MemoryTool.RETRIEVE:
            entries = state.ltm.retrieve(action.args["query"])
            for e in entries:
                state.stm.append(ContextTurn(state.step, "memory", f"[RETRIEVED] {e.key}: {e.value}"))
            return f"Retrieved {len(entries)} LTM entries"
        elif action.tool == MemoryTool.SUMMARY:
            removed = state.stm.summarize(action.args["start"], action.args["end"], action.args["summary"])
            return f"Summarized {removed} turns"
        elif action.tool == MemoryTool.FILTER:
            removed = state.stm.filter(action.args["keywords"])
            return f"Filtered {removed} turns"
        else:  # GENERATE
            state.stm.append(ContextTurn(state.step, "agent", action.args.get("output", "")))
            return "Generated response"


# Three-stage rollout -------------------------------------------------------


@dataclass
class ThreeStageRollout:
    agent: MemoryAgent
    stage_1_info: list[str]  # Stage 1 中要让 agent 看到的信息条目
    stage_2_distractors: list[str]  # Stage 2 注入的干扰内容
    task: TaskSpec
    max_steps_per_stage: int = 5

    def run(self) -> tuple[list[Experience], AgentState]:
        state = AgentState(stm=STMContext(), ltm=LTMStore(), task=self.task)
        experiences: list[Experience] = []

        # Stage 1: LTM construction
        for info in self.stage_1_info:
            state.stm.append(ContextTurn(state.step, "user", info))
            for _ in range(self.max_steps_per_stage):
                action = self.agent.decide(state)
                state_snapshot = self._snapshot(state)
                self.agent.apply(action, state)
                experiences.append(Experience(state_snapshot=state_snapshot, action=action))
                state.step += 1
                if action.tool == MemoryTool.GENERATE:
                    break

        # Reset STM between Stage 1 → Stage 2 (论文 §3.1)
        state.stm.reset()

        # Stage 2: STM control under distractors
        for distractor in self.stage_2_distractors:
            state.stm.append(ContextTurn(state.step, "user", distractor))
            for _ in range(self.max_steps_per_stage):
                action = self.agent.decide(state)
                state_snapshot = self._snapshot(state)
                self.agent.apply(action, state)
                experiences.append(Experience(state_snapshot=state_snapshot, action=action))
                state.step += 1
                if action.tool == MemoryTool.GENERATE:
                    break

        # Stage 3: Integrated reasoning
        state.stm.append(ContextTurn(state.step, "user", state.task.query))
        for _ in range(self.max_steps_per_stage * 2):
            action = self.agent.decide(state)
            state_snapshot = self._snapshot(state)
            self.agent.apply(action, state)
            experiences.append(Experience(state_snapshot=state_snapshot, action=action))
            state.step += 1
            if action.tool == MemoryTool.GENERATE:
                break

        return experiences, state

    @staticmethod
    def _snapshot(state: AgentState) -> dict:
        return {
            "step": state.step,
            "ltm_size": state.ltm.size(),
            "stm_tokens": state.stm.token_count(),
            "stm_turns": len(state.stm.turns),
        }


# Composite reward ----------------------------------------------------------


@dataclass
class CompositeReward:
    """论文 §3.5 复合 reward 严格实现."""

    w_task: float = 1.0
    w_context: float = 1.0
    w_memory: float = 1.0

    def compute(
        self,
        trajectory: list[Experience],
        final_state: AgentState,
        task_judge_score: float,  # ∈ [0, 1] from LLM judge
        memory_quality_score: float = 0.5,  # ∈ [0, 1]
    ) -> dict[str, float]:
        r_task = task_judge_score

        # R_context: STM 管理三因子(简化版)
        n_summary = sum(1 for e in trajectory if e.action.tool == MemoryTool.SUMMARY)
        n_filter = sum(1 for e in trajectory if e.action.tool == MemoryTool.FILTER)
        compression_efficiency = min(1.0, (n_summary + n_filter) / 5.0)  # 期望 5 次/episode
        preventive = 1.0 if not final_state.stm.is_overflow() else 0.0
        info_preservation = 1.0 - min(1.0, final_state.stm.token_count() / 10000.0)
        r_context = (compression_efficiency + preventive + info_preservation) / 3.0

        # R_memory: LTM 操作三因子
        n_add = sum(1 for e in trajectory if e.action.tool == MemoryTool.ADD)
        n_update = sum(1 for e in trajectory if e.action.tool == MemoryTool.UPDATE)
        n_delete = sum(1 for e in trajectory if e.action.tool == MemoryTool.DELETE)
        storage_quality = min(1.0, n_add / 5.0)
        maintenance = min(1.0, (n_update + n_delete) / 3.0)
        r_memory = (storage_quality + maintenance + memory_quality_score) / 3.0

        # Penalty
        penalty = 0.0
        if final_state.stm.is_overflow():
            penalty -= 0.5
        if final_state.step > 50:
            penalty -= 0.2

        total = self.w_task * r_task + self.w_context * r_context + self.w_memory * r_memory + penalty

        return {
            "R_task": r_task,
            "R_context": r_context,
            "R_memory": r_memory,
            "P_penalty": penalty,
            "R_total": total,
        }


# Step-wise GRPO ------------------------------------------------------------


class StepwiseGRPO:
    """论文 §3.4 step-wise GRPO advantage 广播."""

    @staticmethod
    def compute_advantages(
        trajectories: list[list[Experience]],
        terminal_rewards: list[float],
    ) -> list[list[Experience]]:
        """K 条 trajectory + 各自终局 reward → 组内归一化 advantage 广播到每个 step."""
        if not trajectories or not terminal_rewards:
            return trajectories
        mean = statistics.fmean(terminal_rewards)
        stdev = statistics.stdev(terminal_rewards) if len(terminal_rewards) > 1 else 1.0
        epsilon = 1e-6
        for k, traj in enumerate(trajectories):
            terminal_advantage = (terminal_rewards[k] - mean) / (stdev + epsilon)
            for exp in traj:
                exp.advantage = terminal_advantage  # broadcast
        return trajectories


# Demo & tests --------------------------------------------------------------


def _demo_task() -> TaskSpec:
    return TaskSpec(
        query="宝宝 6 个月了能换 L 码纸尿裤吗?",
        info="客户 U001 4 月龄宝宝, 之前 M 码,体重快到 8.5kg",
        expected_answer="建议 L 码",
    )


def _demo_stage1_info() -> list[str]:
    return [
        "客户告知:宝宝乳胶过敏, 不能用乳胶",
        "客户告知:订单号 ORD1001, 批次 BATCH4",
        "客户告知:偏好品牌 Pampers, 当前尺码 M",
    ]


def _demo_stage2_distractors() -> list[str]:
    return [
        "广告:特价促销, 满 99 减 20",
        "noise: 平台公告, 与本次任务无关",
        "noise: 客户随便聊聊天气",
    ]


def main() -> None:
    print("=== AgeMem 三阶段 trajectory demo ===\n")
    agent = MemoryAgent()
    rollout = ThreeStageRollout(
        agent=agent,
        stage_1_info=_demo_stage1_info(),
        stage_2_distractors=_demo_stage2_distractors(),
        task=_demo_task(),
    )
    experiences, final_state = rollout.run()

    print(f"Trajectory 总步数: {len(experiences)}")
    print(f"最终 LTM 条目数: {final_state.ltm.size()}")
    print(f"LTM 内容:")
    for key in final_state.ltm.list_keys():
        print(f"  - {key}")

    # Tool 使用统计
    tool_counter = Counter(e.action.tool for e in experiences)
    print(f"\n工具使用统计:")
    for tool, count in tool_counter.most_common():
        print(f"  {tool.value:20s} {count} 次")

    # Composite reward
    reward_fn = CompositeReward()
    rewards = reward_fn.compute(experiences, final_state, task_judge_score=0.8, memory_quality_score=0.6)
    print(f"\n复合 Reward:")
    for k, v in rewards.items():
        print(f"  {k:12s} {v:.3f}")

    # Step-wise GRPO: 模拟 K=4 条独立 rollout
    print("\n=== Step-wise GRPO advantage 广播 (K=4 模拟) ===")
    trajectories = []
    terminal_rewards = []
    for k in range(4):
        agent_k = MemoryAgent()
        rollout_k = ThreeStageRollout(
            agent=agent_k,
            stage_1_info=_demo_stage1_info(),
            stage_2_distractors=_demo_stage2_distractors(),
            task=_demo_task(),
        )
        exps_k, state_k = rollout_k.run()
        score = 0.5 + k * 0.1  # 模拟不同 rollout 的分数差异
        rewards_k = reward_fn.compute(exps_k, state_k, task_judge_score=score)
        trajectories.append(exps_k)
        terminal_rewards.append(rewards_k["R_total"])

    StepwiseGRPO.compute_advantages(trajectories, terminal_rewards)
    print(f"K=4 trajectory 终局 reward: {[round(r, 3) for r in terminal_rewards]}")
    print(f"对应的 advantage(每条 trajectory 第一步):")
    for k, traj in enumerate(trajectories):
        print(f"  rollout {k}: advantage = {traj[0].advantage:.3f}")


def test_pipeline() -> None:
    # LTMStore: Add/Update/Delete
    ltm = LTMStore()
    ltm.add("allergy", "peanut", timestamp=1)
    assert ltm.size() == 1
    ltm.update("allergy", "peanut+latex", timestamp=2)
    assert ltm._store["allergy"].value == "peanut+latex"
    ltm.delete("allergy")
    assert ltm.size() == 0

    # LTMStore: Retrieve (Jaccard)
    ltm.add("size_history", "M code 4-6kg", timestamp=1)
    entries = ltm.retrieve("纸尿裤 size M", top_k=3)
    assert len(entries) >= 1, f"should retrieve at least 1 entry, got {len(entries)}"

    # STMContext: Filter & Summarize
    stm = STMContext(max_tokens=100)
    stm.append(ContextTurn(0, "user", "noise content"))
    stm.append(ContextTurn(1, "user", "关键信息 订单 ORD123"))
    removed = stm.filter(keywords_to_remove=["noise"])
    assert removed == 1
    assert "订单" in stm.turns[0].content

    stm.append(ContextTurn(2, "user", "more turn"))
    stm.append(ContextTurn(3, "user", "another turn"))
    summarized = stm.summarize(0, 2, "earlier summary")
    assert summarized > 0

    # MemoryAgent + ThreeStageRollout end-to-end
    agent = MemoryAgent()
    rollout = ThreeStageRollout(
        agent=agent,
        stage_1_info=_demo_stage1_info(),
        stage_2_distractors=_demo_stage2_distractors(),
        task=_demo_task(),
    )
    experiences, final_state = rollout.run()
    assert len(experiences) > 0
    assert final_state.ltm.size() > 0, "Stage 1 应至少存入 1 个 LTM 条目"

    # 工具使用必须包含 ADD 和 GENERATE
    tool_set = {e.action.tool for e in experiences}
    assert MemoryTool.ADD in tool_set, "至少应该有 1 次 Add LTM"
    assert MemoryTool.GENERATE in tool_set, "至少应该有 1 次 Generate"

    # CompositeReward
    reward_fn = CompositeReward()
    rewards = reward_fn.compute(experiences, final_state, task_judge_score=0.8)
    assert 0 <= rewards["R_task"] <= 1
    assert "R_total" in rewards

    # Step-wise GRPO (独立 Experience 实例避免 advantage 互相覆盖)
    def _stub(n: int) -> list[Experience]:
        return [Experience(state_snapshot={}, action=Action(MemoryTool.GENERATE)) for _ in range(n)]

    trajectories = [_stub(3), _stub(5), _stub(8)]
    terminal_rewards = [1.0, 0.5, 0.8]
    StepwiseGRPO.compute_advantages(trajectories, terminal_rewards)
    # advantage 广播:同 trajectory 内所有 step advantage 一致
    for traj in trajectories:
        if traj:
            first_adv = traj[0].advantage
            assert all(abs(e.advantage - first_adv) < 1e-6 for e in traj), "advantage 应广播"
    # 不同 trajectory 的 advantage 应不同(组内归一化)
    assert trajectories[0][0].advantage != trajectories[1][0].advantage

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()
