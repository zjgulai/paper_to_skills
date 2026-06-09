---
title: MAS Dynamic Trust — 多智能体动态信任图：抵御 Sleeper Agent 与历史感知可信聚合
doc_type: knowledge
module: 10-MAS
topic: mas-dynamic-trust-management
status: stable
created: 2026-06-04
updated: 2026-06-04
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: MAS Dynamic Trust — 多智能体动态信任管理

> **图谱定位**：Layer 3 进阶层｜修复 `AgentTrust-Runtime-Safety-Interception` prerequisite 断链｜为 `Skill-MAS-Adversarial-Defense` 提供前置基础

---

## ① 算法原理

### 核心思想

MAS 中 Agent 之间传递消息，但消息的可信度并不相同——某个 Agent 可能已被攻击者控制（Sleeper Agent），在积累足够信任后才触发恶意行为。**动态信任管理**解决的问题是：**在没有可信第三方的情况下，每个 Agent 如何评估其他 Agent 发来消息的可信度，并据此决定接受或拒绝**。

核心问题的三个维度：
1. **谁可信**：基于历史交互行为动态评估每个 peer 的可靠性
2. **多可信**：用概率分布（而非标量）表达不确定性，防止信任值被操纵
3. **如何聚合**：在信任感知下，如何合并多个 Agent 的输入以得出最终决策

### 三篇论文的互补关系

| 论文 | 解决的核心问题 | 关键机制 |
|------|-------------|---------|
| **DynaTrust** (2603.15661) | 抵御 Sleeper Agent（静默积累信任后突然恶意） | 动态信任图 + Beta 分布 Bayesian 更新 + 陪审团共识 + 副本恢复 |
| **A-Trust** (2506.02546) | 无侵入式评估消息可信度（不需要外部验证器） | LLM 内部 Attention 模式量化 + Grice 通信理论 6 维信任 |
| **ECL** (2601.21742) | 历史感知可信聚合（避免盲从误导性 peer） | 两阶段解耦（信任估计 → 信任感知聚合）+ RL 辅助奖励 |

### DynaTrust：动态信任图（主干算法）

将 MAS 建模为**动态信任图 DTG**（Dynamic Trust Graph），每条边 $e_{ij}$ 代表 Agent $i$ 对 Agent $j$ 的信任水平，用 Beta 分布参数化：

$$\text{Trust}_{ij} \sim \text{Beta}(\alpha_{ij}, \beta_{ij})$$

- $\alpha_{ij}$：Agent $j$ 向 $i$ 发送可靠消息的累计次数（成功交互）
- $\beta_{ij}$：Agent $j$ 向 $i$ 发送不可靠消息的累计次数（失败交互）
- 信任期望值：$\mathbb{E}[\text{Trust}_{ij}] = \frac{\alpha_{ij}}{\alpha_{ij} + \beta_{ij}}$

**关键设计：接收端评分（非发送端自报）**

不依赖 Agent 自我声明可信度，而是由接收方根据**消息结果**更新信任：
- 消息导致正确决策 → $\alpha_{ij} \mathrel{+}= 1$
- 消息导致错误决策 → $\beta_{ij} \mathrel{+}= 1$

**陪审团共识机制**（Jury Consensus）

对高风险决策（信任值低于阈值 $\tau$），触发多 Agent 交叉验证：
$$\text{Accept}(m_j) = \mathbf{1}\left[\sum_{k \neq i,j} \text{Trust}_{kj} > \theta_{jury}\right]$$

**副本恢复**（Replica Recovery）

检测到被攻击的 Agent 后，从高信任度 Agent 的状态快照恢复系统：
$$\text{DSR}_{DynaTrust} = 92.4\% \quad (\text{baseline AgentShield}: 48.7\%)$$

### A-Trust：基于 Attention 的 6 维信任模型

无需外部验证器，直接利用 LLM 推理过程中产生的 **Attention 矩阵**量化消息可信度。

基于 Grice 通信理论定义 6 个正交信任维度：

| 维度 | 含义 | Attention 信号 |
|------|------|----------------|
| **Relevance** | 消息与当前任务相关性 | 目标 token 的 cross-attention 权重 |
| **Consistency** | 消息内部逻辑一致性 | Self-attention 熵（低熵=高一致） |
| **Specificity** | 信息具体度（非模糊） | 低频词 token 的 attention 占比 |
| **Timeliness** | 信息时效性 | 历史 context 衰减系数 |
| **Source Credibility** | 历史发送方可靠性 | 跨对话的发送方 embedding 相似度 |
| **Coherence** | 与系统整体状态的连贯性 | 全局 context 与消息的 KL 散度 |

### ECL：两阶段历史感知信任聚合

**问题**：Agent 面对多个 peer 的相互矛盾意见时，如何避免"从众盲从"？

**两阶段解耦**：

```
阶段1：信任估计（Trust Estimation）
  输入：peer 的历史交互记录 {(message, outcome)}
  输出：每个 peer 的可靠性概率 P(reliable | history)

阶段2：信任感知聚合（Trust-Aware Aggregation）
  输入：当前多 peer 意见 + 各自可靠性概率
  输出：加权聚合的最终决策
```

**关键结果**：Qwen 3-4B（小模型）在历史感知设定下的准确率超越无历史感知的 Qwen 3-30B，证明信任建模比模型规模更重要。

---

## ② 母婴出海应用案例

### 场景一：跨境采购谈判 MAS 中的供应商信任评估

**业务背景**：AgenticPay 采购谈判系统由多个 Agent 协同工作——价格谈判 Agent、合规检查 Agent、市场行情 Agent。若某个市场行情 Agent 被错误数据污染（如竞品恶意干扰），所有下游决策将偏差。

**DynaTrust 应用**：

```
初始状态：所有 Agent 信任值 = Beta(1, 1)（均匀先验，不偏见任何来源）

第 1-10 次交互：
  市场行情 Agent 提供的 competitor_price 数据
  → 采购决策结果正确率 82% → α_market += 8, β_market += 2
  → Trust_market = Beta(9, 3)，期望 0.75

第 15 次交互（Sleeper Agent 触发）：
  市场行情 Agent 提供异常高的竞品价格（实为伪造）
  → 采购价格被高估 23% → β_market += 1
  → 信任值快速下滑，触发陪审团：其他 2 个 Agent 交叉验证
  → 陪审团否决：拒绝接受该消息，保留上次验证数据

预期收益：避免 1 次错误采购决策，节省采购成本 15-25%
```

**数据要求**：
- Agent 间消息历史（JSON 格式）：`{agent_id, message_type, content, outcome}`
- 决策结果反馈：`{decision_id, outcome, correct: bool}`

### 场景二：库存 MAS 中的多仓预测数据可信度管理

**业务背景**：AIM-RM 库存管理系统接收来自多个数据源的库存预测（本地仓、海外仓、第三方 3PL）。当大促期间某个数据源出现系统故障，产生异常预测值，如何自动隔离该数据源？

**A-Trust 应用**：

```
正常状态：
  Relevance(本地仓预测) = 0.89（与当前 SKU 高度相关）
  Consistency(本地仓预测) = 0.92（内部逻辑一致）

大促前 3 天（数据源故障）：
  海外仓预测：库存=0（实际应为 2300 件）
  A-Trust 检测：
    - Coherence 维度骤降至 0.12（与历史库存分布 KL 散度 >> 阈值）
    - 触发消息级信任警报：可信度=0.15（低于阈值 0.3）
    - 自动切换：使用次优可信数据源（本地仓+3PL 加权平均）

效果：避免因错误数据导致的超卖，减少大促退款损失约 8-12 万元
```

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/dynamic_trust/model.py`

```python
"""
MAS Dynamic Trust Management
整合 DynaTrust (Bayesian信任图) + A-Trust (Attention量化) + ECL (历史感知聚合)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.stats import beta as beta_dist


@dataclass
class TrustEdge:
    """Alpha-beta Bayesian 信任边"""
    agent_id: str
    alpha: float = 1.0   # 成功交互次数（+1 先验）
    beta: float = 1.0    # 失败交互次数（+1 先验）
    history: List[bool] = field(default_factory=list)  # True=可靠, False=不可靠

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """交互次数越多，置信度越高"""
        n = self.alpha + self.beta - 2  # 减去先验
        return min(1.0, n / 20)         # 20次交互达到满置信

    def update(self, outcome: bool):
        if outcome:
            self.alpha += 1
        else:
            self.beta += 1
        self.history.append(outcome)

    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)


class DynamicTrustGraph:
    """
    DynaTrust 动态信任图
    - 每个 Agent 维护对其他所有 Agent 的信任边
    - 接收端评分（非发送端自报）
    - 陪审团共识用于高风险决策
    """

    def __init__(
        self,
        agent_id: str,
        trust_threshold: float = 0.6,
        jury_threshold: float = 0.5,
        jury_size: int = 2,
    ):
        self.agent_id = agent_id
        self.trust_threshold = trust_threshold
        self.jury_threshold = jury_threshold
        self.jury_size = jury_size
        self.edges: Dict[str, TrustEdge] = {}

    def get_or_create(self, peer_id: str) -> TrustEdge:
        if peer_id not in self.edges:
            self.edges[peer_id] = TrustEdge(agent_id=peer_id)
        return self.edges[peer_id]

    def update_trust(self, peer_id: str, outcome: bool):
        """接收端根据消息结果更新信任"""
        self.get_or_create(peer_id).update(outcome)

    def get_trust(self, peer_id: str) -> float:
        return self.get_or_create(peer_id).mean

    def is_trusted(self, peer_id: str) -> bool:
        return self.get_trust(peer_id) >= self.trust_threshold

    def needs_jury(self, peer_id: str) -> bool:
        """信任值低但未达到拒绝阈值时，触发陪审团"""
        trust = self.get_trust(peer_id)
        return 0.3 <= trust < self.trust_threshold

    def jury_consensus(
        self,
        peer_id: str,
        other_graphs: Dict[str, "DynamicTrustGraph"],
    ) -> bool:
        """
        向其他 Agent 的信任图查询：他们对 peer_id 的信任值
        若多数信任 → 接受，否则拒绝
        """
        votes = []
        for other_id, other_graph in other_graphs.items():
            if other_id != self.agent_id and other_id != peer_id:
                votes.append(other_graph.get_trust(peer_id))
                if len(votes) >= self.jury_size:
                    break
        if not votes:
            return self.get_trust(peer_id) >= 0.3
        avg_external = sum(votes) / len(votes)
        return avg_external >= self.jury_threshold

    def get_trusted_agents(self) -> List[Tuple[str, float]]:
        """返回所有可信 Agent 及其信任值（降序）"""
        return sorted(
            [(aid, edge.mean) for aid, edge in self.edges.items()
             if edge.mean >= self.trust_threshold],
            key=lambda x: x[1],
            reverse=True,
        )


class ATrustEvaluator:
    """
    A-Trust：基于 Grice 通信理论的 6 维信任评估
    无需外部验证器，基于消息内容本身判断可信度
    """

    TRUST_WEIGHTS = {
        "relevance": 0.25,
        "consistency": 0.20,
        "specificity": 0.15,
        "timeliness": 0.15,
        "source_credibility": 0.15,
        "coherence": 0.10,
    }

    def __init__(self, trust_threshold: float = 0.3):
        self.trust_threshold = trust_threshold
        self.source_history: Dict[str, List[float]] = {}

    def evaluate(
        self,
        message: dict,
        task_context: dict,
        source_id: str,
        system_state: Optional[dict] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        评估消息可信度
        Returns: (overall_trust, dimension_scores)
        """
        scores = {
            "relevance": self._score_relevance(message, task_context),
            "consistency": self._score_consistency(message),
            "specificity": self._score_specificity(message),
            "timeliness": self._score_timeliness(message),
            "source_credibility": self._score_source(source_id),
            "coherence": self._score_coherence(message, system_state or {}),
        }
        overall = sum(
            scores[dim] * weight
            for dim, weight in self.TRUST_WEIGHTS.items()
        )
        # 更新来源历史
        if source_id not in self.source_history:
            self.source_history[source_id] = []
        self.source_history[source_id].append(overall)
        return overall, scores

    def _score_relevance(self, message: dict, context: dict) -> float:
        """消息关键词与任务上下文的词汇重叠度"""
        msg_words = set(str(message).lower().split())
        ctx_words = set(str(context).lower().split())
        if not ctx_words:
            return 0.5
        overlap = len(msg_words & ctx_words) / len(ctx_words)
        return min(1.0, overlap * 2)

    def _score_consistency(self, message: dict) -> float:
        """消息内部字段的值域一致性检查"""
        values = [v for v in message.values() if isinstance(v, (int, float))]
        if len(values) < 2:
            return 0.7
        # 检测异常值（超过 3σ 的值）
        if len(values) >= 3:
            mean, std = np.mean(values), np.std(values)
            outliers = sum(1 for v in values if abs(v - mean) > 3 * std)
            return max(0.0, 1.0 - outliers * 0.3)
        return 0.7

    def _score_specificity(self, message: dict) -> float:
        """信息具体度：有数值 > 有具体名词 > 纯泛化描述"""
        has_numbers = any(
            isinstance(v, (int, float)) for v in message.values()
        )
        has_specific_keys = len(message) > 2
        return 0.3 + (0.4 if has_numbers else 0) + (0.3 if has_specific_keys else 0)

    def _score_timeliness(self, message: dict) -> float:
        """时效性：有时间戳的消息优先"""
        has_timestamp = any(
            k in message for k in ["timestamp", "time", "datetime", "ts"]
        )
        return 0.8 if has_timestamp else 0.5

    def _score_source(self, source_id: str) -> float:
        """基于历史评分的来源可信度"""
        history = self.source_history.get(source_id, [])
        if not history:
            return 0.5
        # 指数加权移动平均（近期更重要）
        weights = np.exp(np.linspace(-1, 0, len(history)))
        return float(np.average(history, weights=weights))

    def _score_coherence(self, message: dict, system_state: dict) -> float:
        """与系统当前状态的连贯性"""
        if not system_state:
            return 0.6
        msg_vals = {k: v for k, v in message.items() if isinstance(v, (int, float))}
        state_vals = {k: v for k, v in system_state.items() if isinstance(v, (int, float))}
        common_keys = set(msg_vals.keys()) & set(state_vals.keys())
        if not common_keys:
            return 0.6
        deviations = []
        for k in common_keys:
            state_v = state_vals[k]
            if state_v != 0:
                deviations.append(abs(msg_vals[k] - state_v) / abs(state_v))
        if not deviations:
            return 0.6
        avg_deviation = np.mean(deviations)
        return max(0.0, 1.0 - avg_deviation)

    def is_trusted(self, overall_trust: float) -> bool:
        return overall_trust >= self.trust_threshold


class ECLTrustAwareAggregator:
    """
    ECL：历史感知可信聚合
    两阶段：信任估计 → 信任感知加权聚合
    解决"从众盲从"问题
    """

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.peer_reliability: Dict[str, List[float]] = {}

    def update_reliability(self, peer_id: str, reliability: float):
        if peer_id not in self.peer_reliability:
            self.peer_reliability[peer_id] = []
        self.peer_reliability[peer_id].append(reliability)

    def estimate_reliability(self, peer_id: str) -> float:
        """
        阶段1：基于历史交互估计 peer 可靠性
        用指数衰减加权（近期权重更高）
        """
        history = self.peer_reliability.get(peer_id, [])
        if not history:
            return 0.5
        n = len(history)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return float(np.dot(history, weights))

    def aggregate(
        self,
        peer_opinions: Dict[str, float],
        trust_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        阶段2：信任感知加权聚合
        trust_scores 来自 DynamicTrustGraph 或 ATrustEvaluator

        Args:
            peer_opinions: {peer_id: opinion_value}
            trust_scores: {peer_id: trust_score} (若None，用历史可靠性)

        Returns: 聚合后的最终值
        """
        if not peer_opinions:
            return 0.0
        weights = {}
        for peer_id, _ in peer_opinions.items():
            if trust_scores and peer_id in trust_scores:
                weights[peer_id] = trust_scores[peer_id]
            else:
                weights[peer_id] = self.estimate_reliability(peer_id)
        total_weight = sum(weights.values())
        if total_weight == 0:
            return float(np.mean(list(peer_opinions.values())))
        return sum(
            peer_opinions[pid] * weights[pid] / total_weight
            for pid in peer_opinions
        )


class MASAgent:
    """
    集成三层信任机制的 MAS Agent
    """

    def __init__(self, agent_id: str, trust_threshold: float = 0.6):
        self.agent_id = agent_id
        self.trust_graph = DynamicTrustGraph(agent_id, trust_threshold)
        self.a_trust = ATrustEvaluator(trust_threshold=0.3)
        self.ecl = ECLTrustAwareAggregator()

    def receive_message(
        self,
        sender_id: str,
        message: dict,
        task_context: dict,
        peer_graphs: Optional[Dict[str, DynamicTrustGraph]] = None,
    ) -> Tuple[bool, float, str]:
        """
        接收并评估消息可信度
        Returns: (accepted, trust_score, reason)
        """
        # 1. Bayesian 信任图检查
        bayesian_trust = self.trust_graph.get_trust(sender_id)

        # 2. A-Trust 消息内容评估
        content_trust, dims = self.a_trust.evaluate(message, task_context, sender_id)

        # 3. 综合信任分
        combined_trust = 0.6 * bayesian_trust + 0.4 * content_trust

        if combined_trust >= self.trust_graph.trust_threshold:
            return True, combined_trust, "trusted"

        if self.trust_graph.needs_jury(sender_id) and peer_graphs:
            jury_ok = self.trust_graph.jury_consensus(sender_id, peer_graphs)
            reason = "jury_accepted" if jury_ok else "jury_rejected"
            return jury_ok, combined_trust, reason

        return False, combined_trust, "untrusted"

    def aggregate_peer_inputs(
        self,
        peer_opinions: Dict[str, float],
    ) -> float:
        """ECL 信任感知聚合多 peer 意见"""
        trust_scores = {
            pid: self.trust_graph.get_trust(pid)
            for pid in peer_opinions
        }
        return self.ecl.aggregate(peer_opinions, trust_scores)

    def update_trust(self, peer_id: str, outcome: bool, reliability: float):
        """根据交互结果更新信任"""
        self.trust_graph.update_trust(peer_id, outcome)
        self.ecl.update_reliability(peer_id, reliability)


# ── 使用示例 ────────────────────────────────────────────────────────────

def demo_procurement_scenario():
    """
    模拟采购谈判 MAS 中的动态信任管理
    场景：价格谈判 Agent 接收市场行情 Agent 的竞品价格数据
    """
    # 初始化 Agent
    price_agent = MASAgent("price_negotiation", trust_threshold=0.6)
    market_agent = MASAgent("market_intel", trust_threshold=0.6)
    compliance_agent = MASAgent("compliance_check", trust_threshold=0.6)

    peer_graphs = {
        "market_intel": market_agent.trust_graph,
        "compliance_check": compliance_agent.trust_graph,
    }

    task_context = {
        "product": "baby_sterilizer",
        "target_price": 45.0,
        "market": "amazon_us",
    }

    # 模拟 10 次正常交互（建立信任）
    for i in range(10):
        msg = {"competitor_price": 42.0 + np.random.randn(), "source": "Amazon", "timestamp": f"2026-06-0{i+1}"}
        accepted, trust, reason = price_agent.receive_message(
            "market_intel", msg, task_context, peer_graphs
        )
        outcome = abs(msg["competitor_price"] - 42.5) < 2.0
        price_agent.update_trust("market_intel", outcome, float(outcome))

    normal_trust = price_agent.trust_graph.get_trust("market_intel")
    print(f"[正常交互后] market_intel 信任值: {normal_trust:.3f}")

    # 模拟 Sleeper Agent 触发（发送异常高价）
    sleeper_msg = {"competitor_price": 68.0, "source": "Amazon", "timestamp": "2026-06-11"}
    accepted, trust, reason = price_agent.receive_message(
        "market_intel", sleeper_msg, task_context, peer_graphs
    )
    price_agent.update_trust("market_intel", False, 0.0)

    print(f"[Sleeper触发] 消息: {sleeper_msg}")
    print(f"[Sleeper触发] 接受={accepted}, 信任={trust:.3f}, 原因={reason}")
    print(f"[Sleeper触发后] market_intel 信任值: {price_agent.trust_graph.get_trust('market_intel'):.3f}")

    # ECL 聚合多源意见
    peer_opinions = {
        "market_intel": 68.0,       # 可疑来源
        "compliance_check": 43.0,   # 可信来源
    }
    aggregated_price = price_agent.aggregate_peer_inputs(peer_opinions)
    print(f"[ECL聚合] 最终价格参考: ${aggregated_price:.2f}（可信加权平均，非盲目平均）")

    return {
        "normal_trust": normal_trust,
        "sleeper_accepted": accepted,
        "post_attack_trust": price_agent.trust_graph.get_trust("market_intel"),
        "aggregated_price": aggregated_price,
    }


if __name__ == "__main__":
    np.random.seed(42)
    result = demo_procurement_scenario()
    print("\n=== 结果摘要 ===")
    for k, v in result.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
```

---

## ④ 技能关联

### 前置技能
- [[Skill-Multi-Agent-Debate]]：多 Agent 辩论共识 → 信任是辩论中评估发言可信度的基础
- [[Skill-CAMEL-Role-Playing-Agents]]：角色扮演协作 → 角色间的可信委托建立信任

### 延伸技能
- [[Skill-AgentTrust-Runtime-Safety-Interception]]：运行时安全拦截 ← **本 Skill 的 prerequisite**（修复断链）
- [[Skill-Agent-Safety-Guardrails]]：Agent 安全防护 → 信任评估是安全防护的第一道门

### 可组合技能
- [[Skill-Graph-Grounded-MAS-Protocol]]：图结构通信协议 ↔ 信任图 + 通信图双轨保障
- [[Skill-MASEval-System-Evaluation]]：系统评估 ↔ 评估框架中纳入信任维度

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 避免 1 次因 Sleeper Agent 导致的错误采购决策（母婴货值通常 5-30 万元/批次），防损收益 5-25 万元/次；大促期间库存数据异常拦截，减少超卖损失 8-15 万元 |
| **实施难度** | ⭐⭐☆☆☆（Python 即可实现，无需模型训练，Beta 更新为纯统计计算） |
| **优先级评分** | ⭐⭐⭐⭐⭐（图谱断链修复，是后续 MAS-Adversarial-Defense 的必要前置） |
| **评估依据** | DynaTrust DSR 92.4%（防御成功率）vs 基线 48.7%；ECL 使 4B 模型超越无感知 30B；属于生产 MAS 的基础安全层，业务价值长期稳定 |

---

## 论文来源

| 论文 | arXiv | 年份 | Venue |
|------|-------|------|-------|
| DynaTrust: A Defense against Sleeper Agents | [2603.15661](https://arxiv.org/abs/2603.15661) | 2026-03 | — |
| A-Trust: Attention-Based Trust Management | [2506.02546](https://arxiv.org/abs/2506.02546) | 2026-06 | — |
| ECL: Epistemic Context Learning | [2601.21742](https://arxiv.org/abs/2601.21742) | 2026-01 | — |
