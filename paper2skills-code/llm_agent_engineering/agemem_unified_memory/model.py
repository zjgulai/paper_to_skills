"""
AgeMem -- LTM+STM Unified Agent Memory RL Management
Paper: arXiv:2601.01885 | Jan 2026
Use case: Ad Agent cross-session keyword ROAS accumulation + selection agent category knowledge
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─── 数据类 ───────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    LTM = "LTM"
    STM = "STM"


@dataclass
class MemoryItem:
    """统一记忆条目，LTM 和 STM 共用"""
    item_id: str
    content: str
    memory_type: MemoryType
    importance: float = 0.5         # 0-1，影响 LTM 保留优先级
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    ttl_seconds: float | None = None  # None = 永久（LTM），有值 = 过期（STM）

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds


# ─── LTM 存储 ─────────────────────────────────────────────────────────────

class LTMStore:
    """
    长期记忆：持久化结构化 Store。
    生产环境：接入 vector DB（Pinecone/Qdrant）或 key-value store（Redis）。
    """

    def __init__(self):
        self._store: dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> str:
        self._store[item.item_id] = item
        return item.item_id

    def update(self, item_id: str, content: str, importance: float | None = None) -> bool:
        if item_id not in self._store:
            return False
        self._store[item_id].content = content
        self._store[item_id].timestamp = time.time()
        if importance is not None:
            self._store[item_id].importance = importance
        return True

    def delete(self, item_id: str) -> bool:
        if item_id in self._store:
            del self._store[item_id]
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """
        语义搜索（生产环境替换为向量检索）。
        当前：关键词匹配 + 重要性排序。
        """
        results = [
            item for item in self._store.values()
            if any(tag in query.lower() for tag in item.tags)
               or query.lower() in item.content.lower()
        ]
        return sorted(results, key=lambda x: x.importance, reverse=True)[:top_k]

    def __len__(self) -> int:
        return len(self._store)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {"id": k, "content": v.content, "importance": v.importance, "tags": v.tags}
            for k, v in self._store.items()
        ]


# ─── STM 缓冲 ─────────────────────────────────────────────────────────────

class STMBuffer:
    """
    短期记忆：窗口化 context 缓冲。
    超出 max_size 时自动触发摘要压缩。
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._buffer: list[MemoryItem] = []

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        relevant = [
            item for item in self._buffer
            if not item.is_expired()
               and (query.lower() in item.content.lower()
                    or any(tag in query.lower() for tag in item.tags))
        ]
        return sorted(relevant, key=lambda x: x.timestamp, reverse=True)[:top_k]

    def add(self, item: MemoryItem) -> None:
        self._buffer.append(item)
        if len(self._buffer) > self.max_size:
            self._auto_evict()

    def summary(self) -> str:
        """
        压缩当前 STM 为摘要字符串。
        生产环境：调用 LLM 生成摘要；当前：拼接高重要性条目。
        """
        active = [item for item in self._buffer if not item.is_expired()]
        top = sorted(active, key=lambda x: x.importance, reverse=True)[:3]
        if not top:
            return "（无活跃 STM 内容）"
        return "；".join(f"[{item.tags[0] if item.tags else 'general'}] {item.content[:60]}"
                         for item in top)

    def filter(self, exclude_tags: list[str]) -> int:
        before = len(self._buffer)
        self._buffer = [
            item for item in self._buffer
            if not any(tag in item.tags for tag in exclude_tags)
        ]
        return before - len(self._buffer)

    def _auto_evict(self) -> None:
        self._buffer = [item for item in self._buffer if not item.is_expired()]
        if len(self._buffer) > self.max_size:
            self._buffer.sort(key=lambda x: (x.importance, x.timestamp))
            self._buffer = self._buffer[-self.max_size:]

    def __len__(self) -> int:
        return len(self._buffer)


# ─── Agent 统一记忆管理 ───────────────────────────────────────────────────

class AgeMemAgent:
    """
    统一记忆管理：模拟 RL policy 决定调用哪个记忆操作。
    六个 tool：Add/Update/Delete（LTM）+ Retrieve/Summary/Filter（STM）。
    """

    def __init__(self, ltm: LTMStore, stm: STMBuffer):
        self.ltm = ltm
        self.stm = stm
        self._op_log: list[dict[str, Any]] = []

    def decide_and_execute(self, observation: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        核心决策：根据当前 observation 决定记忆操作。
        生产环境：此方法由 LLM policy（经 RL 训练）生成 tool_call。
        当前：基于启发式规则模拟决策。
        """
        op_result: dict[str, Any] = {"ops": [], "stm_summary": None, "retrieved": []}

        # 启发式规则 1：包含"更新"/"新增"信号 → LTM Add/Update
        if any(kw in observation for kw in ["ROAS:", "新品:", "合规更新:"]):
            item = MemoryItem(
                item_id=f"ltm_{int(time.time() * 1000)}",
                content=observation,
                memory_type=MemoryType.LTM,
                importance=self._estimate_importance(observation),
                tags=self._extract_tags(observation),
            )
            existing = self.ltm.search(observation[:20], top_k=1)
            if existing:
                self.ltm.update(existing[0].item_id, observation, item.importance)
                op_result["ops"].append({"op": "LTM.Update", "id": existing[0].item_id})
            else:
                self.ltm.add(item)
                op_result["ops"].append({"op": "LTM.Add", "id": item.item_id})

        # 启发式规则 2：context 中有"查询历史"信号 → STM Retrieve from LTM
        if context.get("need_history"):
            query = context.get("query", observation[:30])
            retrieved = self.ltm.search(query, top_k=3)
            for r in retrieved:
                self.stm.add(MemoryItem(
                    item_id=f"stm_{r.item_id}",
                    content=r.content,
                    memory_type=MemoryType.STM,
                    importance=r.importance,
                    tags=r.tags,
                    ttl_seconds=3600,  # STM 1小时过期
                ))
            op_result["ops"].append({"op": "STM.Retrieve", "count": len(retrieved)})
            op_result["retrieved"] = [r.content[:80] for r in retrieved]

        # 启发式规则 3：STM 超过阈值 → Summary 压缩
        if len(self.stm) >= self.stm.max_size * 0.8:
            summary = self.stm.summary()
            op_result["ops"].append({"op": "STM.Summary"})
            op_result["stm_summary"] = summary

        self._op_log.extend(op_result["ops"])
        return op_result

    @staticmethod
    def _estimate_importance(text: str) -> float:
        keywords = ["ROAS", "合规", "爆款", "封号", "竞品", "新品", "大促"]
        score = 0.3 + 0.1 * sum(1 for kw in keywords if kw in text)
        return min(1.0, score)

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        tag_map = {"ROAS": "广告", "合规": "合规", "竞品": "竞品",
                   "关键词": "关键词", "品类": "品类", "新品": "新品"}
        return [tag for kw, tag in tag_map.items() if kw in text]


# ─── RL 奖励计算 ─────────────────────────────────────────────────────────

class MemoryRewardCalculator:
    """
    评估记忆操作质量，用于 RL 训练信号（Step-wise GRPO）。
    论文: 复合 reward = task_reward + context_reward + memory_reward - penalty
    """

    def __init__(self, w_task: float = 0.5, w_ctx: float = 0.3, w_mem: float = 0.2):
        self.w_task = w_task
        self.w_ctx = w_ctx
        self.w_mem = w_mem

    def compute(
        self,
        task_score: float,      # 0-1，任务答案正确性
        context_precision: float,  # 0-1，STM 精准度（无干扰内容）
        memory_ops_quality: float,  # 0-1，记忆操作的合理性
        redundant_ops: int = 0,    # 多余操作次数（惩罚项）
    ) -> dict[str, float]:
        """
        Step-wise GRPO: 终局 reward 广播到所有中间步骤。
        penalty = 每次冗余操作 -0.05（避免过度频繁 Add/Delete）。
        """
        reward = (self.w_task * task_score
                  + self.w_ctx * context_precision
                  + self.w_mem * memory_ops_quality
                  - 0.05 * redundant_ops)
        return {
            "total_reward": round(reward, 4),
            "task_component": round(self.w_task * task_score, 4),
            "context_component": round(self.w_ctx * context_precision, 4),
            "memory_component": round(self.w_mem * memory_ops_quality, 4),
            "penalty": round(0.05 * redundant_ops, 4),
        }


# ─── 测试用例 ─────────────────────────────────────────────────────────────

def test_ad_agent_cross_session_memory():
    """广告 Agent 跨 5 次会话积累关键词知识，验证 LTM 持久 + STM 摘要机制"""

    ltm = LTMStore()
    stm = STMBuffer(max_size=6)
    agent = AgeMemAgent(ltm, stm)
    reward_calc = MemoryRewardCalculator()

    # 5 次会话模拟
    sessions = [
        "ROAS: 关键词'婴儿奶瓶BPA Free' ROAS=4.2，上周均值3.1，建议加价",
        "ROAS: 关键词'防摔奶瓶' ROAS=2.1，低于阈值，建议降预算",
        "竞品: BrandX 上架新款宽口奶瓶，定价89元，5星1200条",
        "合规更新: 美国市场FDA要求新增BPA测试报告，截止2026-09-01",
        "ROAS: 关键词'婴儿奶瓶BPA Free' ROAS=5.8，大促期间飙升，建议持续加码",
    ]

    print("=" * 60)
    print("广告 Agent 跨会话记忆积累测试")
    print("=" * 60)

    for i, obs in enumerate(sessions, 1):
        ctx = {"need_history": i >= 3, "query": "婴儿奶瓶"}
        result = agent.decide_and_execute(obs, ctx)
        print(f"\n[会话 {i}] {obs[:40]}...")
        print(f"  执行操作: {[op['op'] for op in result['ops']]}")
        if result["retrieved"]:
            print(f"  检索历史: {result['retrieved'][0][:50]}...")
        if result["stm_summary"]:
            print(f"  STM 摘要: {result['stm_summary'][:60]}...")

    # 验证 LTM 持久性
    print(f"\nLTM 累积条目数: {len(ltm)}")
    print("LTM 快照（前 3 条）:")
    for item in ltm.snapshot()[:3]:
        print(f"  [{item['importance']:.2f}] {item['content'][:60]}...")

    # 验证 reward 计算
    reward = reward_calc.compute(
        task_score=0.85,
        context_precision=0.80,
        memory_ops_quality=0.90,
        redundant_ops=1,
    )
    print(f"\nRL 奖励: {json.dumps(reward, ensure_ascii=False)}")

    assert len(ltm) >= 1, f"LTM 应至少有 1 条持久记忆，实际: {len(ltm)}"
    assert len(agent._op_log) >= 3, f"跨会话应产生至少 3 次操作，实际: {len(agent._op_log)}"
    assert reward["total_reward"] > 0
    print("\n✅ 测试通过")


if __name__ == "__main__":
    test_ad_agent_cross_session_memory()
