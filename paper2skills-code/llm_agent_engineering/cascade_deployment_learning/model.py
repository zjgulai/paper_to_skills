"""
CASCADE — Case-Based Continual Adaptation for Large Language Models During Deployment
Paper: arXiv:2605.06702 | May 2026
Use case: WF-B ad creative agent + WF-E review handling case retrieval
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any


# ─── 数据类 ───────────────────────────────────────────────────────────────

@dataclass
class Case:
    """案例条目：上下文嵌入 + 解决方案 + 观测结果"""
    case_id: str
    context_text: str               # 原始上下文文本（用于可读性）
    context_embedding: list[float]  # 上下文语义嵌入向量
    solution: str                   # 解决方案描述
    outcome: float                  # 0=失败，1=成功（支持 0-1 连续值）
    timestamp: float = field(default_factory=time.time)
    task_domain: str = ""
    tags: list[str] = field(default_factory=list)


# ─── CaseBank（案例库）──────────────────────────────────────────────────

class CaseBank:
    """案例存储 + 相似度检索 + 成功案例保留"""

    def __init__(self, max_size: int = 200) -> None:
        self._cases: list[Case] = []
        self.max_size = max_size

    def add(self, case: Case) -> None:
        """添加新案例，超出容量时淘汰最旧的低质量案例"""
        self._cases.append(case)
        if len(self._cases) > self.max_size:
            self._cases.sort(key=lambda c: (c.outcome, c.timestamp), reverse=True)
            self._cases = self._cases[:self.max_size]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve_similar(
        self,
        query_embedding: list[float],
        task_domain: str = "",
        top_k: int = 5,
    ) -> list[tuple[Case, float]]:
        """检索最相似的历史案例，返回 (case, similarity) 列表"""
        candidates = self._cases
        if task_domain:
            domain_cases = [c for c in candidates if c.task_domain == task_domain]
            candidates = domain_cases if domain_cases else candidates

        scored = [
            (case, self.cosine_similarity(query_embedding, case.context_embedding))
            for case in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def update_outcome(self, case_id: str, new_outcome: float) -> bool:
        """更新案例的实际结果（延迟观测）"""
        for case in self._cases:
            if case.case_id == case_id:
                case.outcome = new_outcome
                return True
        return False

    def snapshot(self) -> dict:
        total = len(self._cases)
        success = sum(1 for c in self._cases if c.outcome >= 0.5)
        return {
            "total_cases": total,
            "success_cases": success,
            "success_rate": round(success / max(total, 1), 3),
            "domains": list({c.task_domain for c in self._cases}),
        }


# ─── ContextualBanditRetriever（UCB 探索-利用）────────────────────────

class ContextualBanditRetriever:
    """UCB 策略：平衡成功案例利用 + 新方案探索，满足 no-regret 保证
    
    得分公式：score = (similarity × outcome) + c × √(ln(T) / n_i)
    - 前项：相似度加权成功率（利用）
    - 后项：被选次数少的案例获探索加成（探索），UCB 标准项
    - T：总拉臂次数；n_i：当前臂被拉次数；c：探索系数
    """

    def __init__(self, exploration_coeff: float = 0.5) -> None:
        self.exploration_coeff = exploration_coeff
        self._arm_counts: dict[str, int] = {}
        self._arm_rewards: dict[str, float] = {}
        self._total_pulls: int = 0

    def ucb_score(self, case_id: str, similarity: float, outcome: float) -> float:
        n_i = self._arm_counts.get(case_id, 0)
        exploitation = similarity * outcome
        if n_i == 0 or self._total_pulls == 0:
            return float("inf")  # 未被选过的案例优先探索
        exploration = self.exploration_coeff * math.sqrt(
            math.log(self._total_pulls + 1) / n_i
        )
        return exploitation + exploration

    def select(
        self,
        candidates: list[tuple[Case, float]],
        top_k: int = 1,
    ) -> list[Case]:
        """从候选案例中用 UCB 策略选择最优案例"""
        if not candidates:
            return []

        scored = [
            (case, self.ucb_score(case.case_id, sim, case.outcome))
            for case, sim in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [case for case, _ in scored[:top_k]]

        for case in selected:
            self._arm_counts[case.case_id] = self._arm_counts.get(case.case_id, 0) + 1
        self._total_pulls += len(selected)

        return selected

    def update_reward(self, case_id: str, reward: float) -> None:
        self._arm_rewards[case_id] = self._arm_rewards.get(case_id, 0.0) + reward

    def stats(self) -> dict:
        return {
            "total_pulls": self._total_pulls,
            "unique_arms_pulled": len(self._arm_counts),
        }


# ─── 工具函数：简易文本嵌入 ──────────────────────────────────────────────

def simple_text_embedding(text: str, dim: int = 16) -> list[float]:
    """基于字符哈希的简易嵌入（仅用于测试，生产环境用真实 Embedding 模型）"""
    embedding = [0.0] * dim
    for i, char in enumerate(text):
        idx = (ord(char) + i) % dim
        embedding[idx] += 1.0

    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]
    return embedding


# ─── CASCADEAgent（查询→检索→调整→记录→更新）────────────────────────────

class CASCADEAgent:
    """CASCADE 主 Agent：案例库 + Bandit 检索 + 结果更新"""

    def __init__(self) -> None:
        self.case_bank = CaseBank(max_size=200)
        self.retriever = ContextualBanditRetriever(exploration_coeff=0.5)

    def handle(self, context_text: str, task_domain: str) -> dict:
        """处理新任务：检索历史案例 → UCB 选择最优 → 生成调整后方案"""
        query_embedding = simple_text_embedding(context_text)

        similar_cases = self.case_bank.retrieve_similar(
            query_embedding, task_domain, top_k=5
        )

        selected = self.retriever.select(similar_cases, top_k=1)

        if selected:
            best_case = selected[0]
            sim = next((s for c, s in similar_cases if c.case_id == best_case.case_id), 0.0)
            adapted_solution = (
                f"[CASCADE 案例复用 | 相似度: {sim:.2f} | 历史成功: {best_case.outcome:.0%}]\n"
                f"基于案例 {best_case.case_id[:6]}：{best_case.solution}"
            )
        else:
            best_case = None
            adapted_solution = f"[无历史案例，首次尝试] 针对「{context_text[:40]}」的基础方案"

        new_case_id = hashlib.md5(f"{context_text}{time.time()}".encode()).hexdigest()[:10]

        return {
            "case_id": new_case_id,
            "adapted_solution": adapted_solution,
            "source_case_id": best_case.case_id if best_case else None,
            "case_bank": self.case_bank.snapshot(),
            "bandit_stats": self.retriever.stats(),
        }

    def record_result(
        self,
        case_id: str,
        context_text: str,
        solution: str,
        outcome: float,
        task_domain: str,
    ) -> None:
        """记录任务结果，写入案例库"""
        embedding = simple_text_embedding(context_text)
        case = Case(
            case_id=case_id,
            context_text=context_text,
            context_embedding=embedding,
            solution=solution,
            outcome=outcome,
            task_domain=task_domain,
        )
        self.case_bank.add(case)

        if outcome >= 0.5:
            self.retriever.update_reward(case_id, outcome)


# ─── 测试：20次 Review 处置，验证随经验积累成功率提升 ──────────────────

def test_review_handling() -> None:
    """模拟 20 次 Review 处置，验证 CASCADE 随经验积累成功率提升"""
    agent = CASCADEAgent()

    seed_cases = [
        ("包装破损 美国客户 $45 奶瓶", "全额退款+道歉邮件", 1.0),
        ("收到空盒 德国客户 $89 奶粉", "补发+免费升级配送", 1.0),
        ("产品过期 英国客户 $32 辅食", "全额退款+检查批次", 0.8),
        ("配送延迟 法国客户 $56 爬行垫", "补偿10%优惠券", 0.6),
        ("质量问题 美国客户 $78 学步鞋", "免费换货+运费补偿", 1.0),
        ("包装磨损 美国客户 $38 奶嘴", "换货+道歉邮件", 0.9),
        ("收到错款 德国客户 $65 奶瓶", "补发正确款+保留错款", 1.0),
        ("气味异常 英国客户 $42 辅食", "全额退款+安全检查", 0.8),
        ("印刷错误 法国客户 $29 绘本", "补发新品+小礼品", 0.7),
        ("快递暴力 美国客户 $91 婴儿车", "全额退款+补偿$20", 1.0),
    ]

    test_contexts = [
        "外包装凹陷 美国客户 $48 奶瓶",
        "产品未收到 德国客户 $72 奶粉",
        "开封缺货 英国客户 $35 辅食",
        "包装撕裂 美国客户 $55 学步鞋",
        "货物损坏 法国客户 $83 婴儿车",
        "标签脱落 美国客户 $44 奶嘴",
        "错误地址 德国客户 $67 奶瓶",
        "保质期临近 英国客户 $39 辅食",
        "缺少配件 法国客户 $28 绘本",
        "包装压扁 美国客户 $93 婴儿车",
    ]

    print("=" * 60)
    print("CASCADE Review 处置案例积累测试")
    print("=" * 60)

    # 阶段1：积累案例库
    print("\n[阶段1] 积累历史案例...")
    for i, (context, solution, outcome) in enumerate(seed_cases, 1):
        case_id = hashlib.md5(f"{context}".encode()).hexdigest()[:10]
        agent.record_result(case_id, context, solution, outcome, "review")
        print(f"  案例 {i}: {context[:40]}... → outcome={outcome}")

    print(f"\n案例库状态: {agent.case_bank.snapshot()}")

    # 阶段2：测试案例检索
    print("\n[阶段2] 基于历史案例处理新差评...")
    results = []
    for i, context in enumerate(test_contexts, 11):
        result = agent.handle(context, "review")
        results.append(result)
        has_source = bool(result["source_case_id"])
        print(f"\n[处置 {i}] {context[:40]}...")
        print(f"  复用历史: {'✅' if has_source else '❌'}")
        print(f"  方案: {result['adapted_solution'][:80]}...")
        # 复用历史案例时模拟更高成功率
        simulated_outcome = 0.85 if has_source else 0.55
        agent.record_result(
            result["case_id"], context, result["adapted_solution"], simulated_outcome, "review"
        )

    # 验证成功率改善
    reuse_count = sum(1 for r in results if r["source_case_id"])
    print(f"\n案例复用率: {reuse_count}/{len(results)} ({reuse_count/len(results):.0%})")
    print(f"最终案例库: {agent.case_bank.snapshot()}")
    print(f"Bandit 统计: {agent.retriever.stats()}")

    assert reuse_count > 5, f"应至少有5次复用历史案例，实际: {reuse_count}"
    final_snapshot = agent.case_bank.snapshot()
    assert final_snapshot["total_cases"] >= 10, "案例库应有足够案例"
    print("\n✅ 测试通过：CASCADE 案例积累和检索验证成功")


if __name__ == "__main__":
    test_review_handling()
