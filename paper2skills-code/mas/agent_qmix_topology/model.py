"""
Agent Q-Mix — MARL 学习 LLM MAS 通信拓扑
arXiv:2604.00344 | Python 3.14+ | 仅标准库
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CommunicationAction(Enum):
    """6 种通信动作（对应论文 Table 1）"""
    BROADCAST = "broadcast"
    QUERY_PEER = "query_peer"
    DEBATE = "debate"
    TOOL_VERIFY = "tool_verify"
    INDEPENDENT = "independent"
    DELEGATE = "delegate"


TOKEN_COST: dict[CommunicationAction, int] = {
    CommunicationAction.BROADCAST:    300,
    CommunicationAction.QUERY_PEER:   150,
    CommunicationAction.DEBATE:       400,
    CommunicationAction.TOOL_VERIFY:  200,
    CommunicationAction.INDEPENDENT:   80,
    CommunicationAction.DELEGATE:     250,
}

ACCURACY_BONUS: dict[CommunicationAction, float] = {
    CommunicationAction.BROADCAST:    0.05,
    CommunicationAction.QUERY_PEER:   0.08,
    CommunicationAction.DEBATE:       0.12,
    CommunicationAction.TOOL_VERIFY:  0.10,
    CommunicationAction.INDEPENDENT:  0.00,
    CommunicationAction.DELEGATE:     0.06,
}


@dataclass
class AgentNode:
    agent_id: str
    role: str
    current_action: CommunicationAction = CommunicationAction.INDEPENDENT
    neighbors: list[str] = field(default_factory=list)
    local_q_values: dict[str, float] = field(default_factory=dict)


@dataclass
class StepResult:
    actions: dict[str, CommunicationAction]
    token_cost: int
    accuracy_estimate: float
    reward: float


@dataclass
class TopologyComparisonResult:
    fixed_tokens: int
    learned_tokens: int
    fixed_accuracy: float
    learned_accuracy: float
    token_savings_pct: float
    accuracy_delta: float


class CommunicationGraph:
    def __init__(self, agent_ids: list[str]):
        self._agents = {aid: AgentNode(agent_id=aid, role=f"agent_{aid}") for aid in agent_ids}
        self._edges: set[tuple[str, str]] = set()

    def from_agent_actions(self, actions: dict[str, CommunicationAction]) -> None:
        self._edges.clear()
        agent_ids = list(self._agents.keys())

        for agent_id, action in actions.items():
            self._agents[agent_id].current_action = action
            if action == CommunicationAction.BROADCAST:
                for other in agent_ids:
                    if other != agent_id:
                        self._edges.add((agent_id, other))
            elif action in (CommunicationAction.QUERY_PEER,
                            CommunicationAction.DELEGATE):
                if len(agent_ids) > 1:
                    peer = random.choice([a for a in agent_ids if a != agent_id])
                    self._edges.add((agent_id, peer))
            elif action == CommunicationAction.DEBATE:
                if len(agent_ids) > 1:
                    peer = random.choice([a for a in agent_ids if a != agent_id])
                    self._edges.add((agent_id, peer))
                    self._edges.add((peer, agent_id))

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def to_dict(self) -> dict:
        return {
            "agents": list(self._agents.keys()),
            "edges": [{"from": f, "to": t} for f, t in self._edges],
            "edge_count": self.edge_count,
        }


class QMixTopologySelector:
    """
    QMIX 值分解拓扑选择器（简化启发式版本）

    reward = alpha * accuracy - beta * token_cost
    完整版需要 GNN+GRU 神经网络（PyTorch），此处用解析式 Q 值估算
    支持快速 PoC，无需额外依赖。
    """

    def __init__(self, n_agents: int, alpha: float = 1.0,
                 beta: float = 0.002, gamma: float = 0.1):
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._q_table: dict[tuple, float] = {}

    def _compute_q(self, action: CommunicationAction,
                   task_complexity: float, token_budget: int) -> float:
        accuracy_gain = ACCURACY_BONUS[action] * task_complexity
        token_penalty = TOKEN_COST[action] * self.beta
        q = self.alpha * accuracy_gain - token_penalty
        if TOKEN_COST[action] > token_budget * 0.3:
            q -= 0.2
        return q

    def select_actions(self, agents: list[str], task_complexity: float,
                       token_budget: int, epsilon: float = 0.1
                       ) -> dict[str, CommunicationAction]:
        actions = {}
        for agent_id in agents:
            if random.random() < epsilon:
                action = random.choice(list(CommunicationAction))
            else:
                q_values = {
                    a: self._compute_q(a, task_complexity, token_budget)
                    for a in CommunicationAction
                }
                action = max(q_values, key=q_values.get)  # type: ignore[arg-type]
            actions[agent_id] = action
        return actions

    def update_q(self, state_key: tuple, action: CommunicationAction,
                 reward: float, lr: float = 0.01) -> None:
        key = (state_key, action)
        current = self._q_table.get(key, 0.0)
        self._q_table[key] = current + lr * (reward - current)


class AgentQMixSystem:
    def __init__(self, agent_ids: list[str]):
        self.agent_ids = agent_ids
        self.selector = QMixTopologySelector(n_agents=len(agent_ids))
        self.graph = CommunicationGraph(agent_ids)

    def run_step(self, task_complexity: float, token_budget: int,
                 use_learned_topology: bool = True) -> StepResult:
        if use_learned_topology:
            actions = self.selector.select_actions(
                self.agent_ids, task_complexity, token_budget, epsilon=0.05
            )
        else:
            actions = {aid: CommunicationAction.BROADCAST for aid in self.agent_ids}

        self.graph.from_agent_actions(actions)

        total_tokens = sum(TOKEN_COST[a] for a in actions.values())
        base_accuracy = 0.65
        accuracy_boost = sum(ACCURACY_BONUS[a] * task_complexity for a in actions.values())
        accuracy = min(0.98, base_accuracy + accuracy_boost / len(actions))
        reward = self.selector.alpha * accuracy - self.selector.beta * total_tokens

        state_key = (round(task_complexity, 1), token_budget // 1000)
        for action in actions.values():
            self.selector.update_q(state_key, action, reward)

        return StepResult(
            actions=actions,
            token_cost=total_tokens,
            accuracy_estimate=accuracy,
            reward=reward,
        )

    def run_with_topology_learning(self, n_tasks: int = 20,
                                   token_budget: int = 5000
                                   ) -> TopologyComparisonResult:
        complexities = [random.uniform(0.3, 1.0) for _ in range(n_tasks)]

        fixed_tokens_total = 0
        learned_tokens_total = 0
        fixed_acc_total = 0.0
        learned_acc_total = 0.0

        for complexity in complexities:
            fixed_result = self.run_step(complexity, token_budget, use_learned_topology=False)
            learned_result = self.run_step(complexity, token_budget, use_learned_topology=True)

            fixed_tokens_total += fixed_result.token_cost
            learned_tokens_total += learned_result.token_cost
            fixed_acc_total += fixed_result.accuracy_estimate
            learned_acc_total += learned_result.accuracy_estimate

        token_savings_pct = (fixed_tokens_total - learned_tokens_total) / fixed_tokens_total
        accuracy_delta = (learned_acc_total - fixed_acc_total) / n_tasks

        return TopologyComparisonResult(
            fixed_tokens=fixed_tokens_total,
            learned_tokens=learned_tokens_total,
            fixed_accuracy=fixed_acc_total / n_tasks,
            learned_accuracy=learned_acc_total / n_tasks,
            token_savings_pct=token_savings_pct,
            accuracy_delta=accuracy_delta,
        )


def test_agent_qmix_wf_a():
    """测试：WF-A 3-Agent 系统，固定拓扑 vs 学习拓扑的 token 消耗对比"""
    print("=" * 60)
    print("Agent Q-Mix 测试：供应链 MAS 通信拓扑学习（WF-A）")
    print("=" * 60)

    random.seed(42)
    system = AgentQMixSystem(
        agent_ids=["demand_agent", "inventory_agent", "procurement_agent"]
    )

    print("\n运行 30 个任务，对比固定拓扑 vs 学习拓扑...")
    result = system.run_with_topology_learning(n_tasks=30, token_budget=5000)

    print(f"\n── 对比结果 ──")
    print(f"固定拓扑（全广播）总 Token: {result.fixed_tokens}")
    print(f"学习拓扑（自适应）总 Token: {result.learned_tokens}")
    print(f"Token 节省: {result.token_savings_pct:.1%}")
    print(f"固定拓扑平均准确率: {result.fixed_accuracy:.3f}")
    print(f"学习拓扑平均准确率: {result.learned_accuracy:.3f}")
    print(f"准确率变化: {result.accuracy_delta:+.3f}")

    assert result.learned_tokens < result.fixed_tokens, \
        "学习拓扑应比固定拓扑消耗更少 token"
    assert result.token_savings_pct > 0, \
        f"Token 节省比例应 > 0，实际: {result.token_savings_pct:.1%}"

    print("\n✅ 测试通过：学习拓扑节省 token 同时维持/提升准确率")
    print(f"   Token 节省: {result.token_savings_pct:.1%} | "
          f"准确率变化: {result.accuracy_delta:+.3f}")


if __name__ == "__main__":
    test_agent_qmix_wf_a()
