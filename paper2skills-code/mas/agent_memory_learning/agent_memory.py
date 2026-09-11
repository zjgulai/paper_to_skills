"""
MemGPT-style Agent Memory — 长期记忆与虚拟上下文管理
基于论文: Packer et al. "MemGPT: Towards LLMs as Operating Systems", 2023

核心能力:
1. Main Context — 主内存（当前活跃上下文）
2. Recall Storage — 回忆存储（近期历史）
3. Archival Memory — 档案记忆（长期向量存储）
4. 虚拟上下文管理 — LLM 主动控制记忆换入换出

母婴电商场景: 长期用户对话、VOC 知识累积与复用
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class MainContext:
    """
    主内存（类比 RAM）

    存储当前活跃的信息，容量有限但访问最快。
    """

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.entries: List[MemoryEntry] = []
        self.system_prompt: str = ""

    def add(self, content: str, metadata: Optional[Dict] = None):
        """添加条目到主内存"""
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self.entries.append(entry)
        self._evict_if_needed()

    def replace(self, key: str, content: str):
        """替换指定键的条目"""
        for i, entry in enumerate(self.entries):
            if entry.metadata.get("key") == key:
                self.entries[i] = MemoryEntry(
                    content=content,
                    metadata={"key": key, **entry.metadata}
                )
                return True
        return False

    def get_context(self) -> str:
        """获取当前上下文字符串"""
        lines = [self.system_prompt]
        for entry in self.entries:
            lines.append(entry.content)
        return "\n".join(lines)

    def utilization(self) -> float:
        """上下文利用率（简化估算）"""
        total_chars = sum(len(e.content) for e in self.entries)
        # 粗略估算：1 token ≈ 4 chars
        return min(1.0, total_chars / (self.max_tokens * 4))

    def _evict_if_needed(self):
        """当满载时淘汰最不活跃的条目"""
        while self.utilization() > 0.9 and len(self.entries) > 1:
            # 淘汰最早的条目
            self.entries.pop(0)


class RecallStorage:
    """
    回忆存储（类比磁盘缓存）

    存储近期对话历史和最近访问的记忆。
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.entries: List[MemoryEntry] = []

    def add(self, content: str, metadata: Optional[Dict] = None):
        """添加条目"""
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries.pop(0)

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """关键词搜索（简化版）"""
        query_words = set(query.lower().split())
        scored = []
        for entry in self.entries:
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                scored.append((entry, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """获取最近的 n 条记录"""
        return self.entries[-n:]


class ArchivalMemory:
    """
    档案记忆（类比磁盘/长期存储）

    存储所有历史对话、学习到的知识、用户画像。
    使用向量检索（简化版使用关键词匹配）。
    """

    def __init__(self):
        self.entries: List[MemoryEntry] = []

    def insert(self, content: str, metadata: Optional[Dict] = None):
        """插入档案"""
        entry = MemoryEntry(content=content, metadata=metadata or {})
        self.entries.append(entry)

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """语义搜索（简化版关键词匹配）"""
        query_words = set(query.lower().split())
        scored = []
        for entry in self.entries:
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                scored.append((entry, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_entries": len(self.entries),
            "total_chars": sum(len(e.content) for e in self.entries)
        }


class MemGPTAgent:
    """
    MemGPT 风格的 Agent

    具备三层记忆管理和主动记忆控制能力。
    """

    def __init__(self, agent_id: str, max_context_tokens: int = 4000):
        self.agent_id = agent_id
        self.main_context = MainContext(max_tokens=max_context_tokens)
        self.recall_storage = RecallStorage(capacity=100)
        self.archival_memory = ArchivalMemory()
        self.interaction_count = 0

        # 设置系统提示
        self.main_context.system_prompt = f"""You are Agent {agent_id}.
You have access to three memory tiers:
- Main Context: active working memory
- Recall Storage: recent history (search with recall_memory_search)
- Archival Memory: long-term storage (search with archival_memory_search)

When you need information not in Main Context, use memory functions."""

    def chat(self, user_input: str) -> str:
        """
        与用户交互，管理记忆

        简化版：自动检索相关记忆，生成回复，更新记忆。
        生产环境：LLM 通过 function calling 主动控制记忆。
        """
        self.interaction_count += 1

        # Step 1: 检查是否需要检索记忆
        retrieved_memories = self._retrieve_relevant_memories(user_input)

        # Step 2: 构建完整上下文
        context = self._build_context(user_input, retrieved_memories)

        # Step 3: 生成回复（模拟）
        response = self._generate_response(context, user_input)

        # Step 4: 更新记忆
        self._update_memories(user_input, response)

        return response

    def _retrieve_relevant_memories(self, query: str) -> List[str]:
        """检索相关记忆"""
        memories = []

        # 先查 Recall Storage
        recall_results = self.recall_storage.search(query, top_k=3)
        for entry in recall_results:
            memories.append(f"[Recall] {entry.content[:80]}...")

        # 再查 Archival Memory
        archival_results = self.archival_memory.search(query, top_k=3)
        for entry in archival_results:
            memories.append(f"[Archival] {entry.content[:80]}...")

        return memories

    def _build_context(self, user_input: str, memories: List[str]) -> str:
        """构建完整上下文"""
        # 将检索到的记忆加入主上下文
        for mem in memories:
            self.main_context.add(mem, metadata={"type": "retrieved_memory"})

        self.main_context.add(f"User: {user_input}", metadata={"type": "user_input"})
        return self.main_context.get_context()

    def _generate_response(self, context: str, user_input: str) -> str:
        """生成回复（模拟 LLM）"""
        # 根据用户输入和检索到的记忆生成回复
        if "上次" in user_input or "上次" in user_input:
            if "Spectra" in context:
                return ("您 3 个月前购买的 Spectra S1 应该还能用。"
                        "但考虑到您之前提到体积大不便携，"
                        "如果宝宝开始外出频繁，可以考虑便携版 Spectra 9 Plus（仅 0.3kg）。"
                        "需要我对比两款的具体差异吗？")

        if "价格" in user_input or "贵" in user_input:
            return ("关于价格的问题，根据历史数据，"
                    "Spectra S1 的用户中约 15% 提到价格偏高，"
                    "但 91% 的正面评价集中在静音效果上。"
                    "您是否更关注性价比还是特定功能？")

        return f"收到您的问题：'{user_input[:30]}...'。让我帮您查找相关信息。"

    def _update_memories(self, user_input: str, response: str):
        """更新记忆存储"""
        # 存入 Recall Storage
        self.recall_storage.add(
            f"User: {user_input} | Agent: {response}",
            metadata={"type": "conversation", "turn": self.interaction_count}
        )

        # 如果包含重要洞察，存入 Archival Memory
        if any(kw in user_input for kw in ["购买", "反馈", "问题", "投诉"]):
            self.archival_memory.insert(
                f"Turn {self.interaction_count}: {user_input}",
                metadata={"type": "key_insight", "category": "user_feedback"}
            )

    def core_memory_append(self, content: str, key: Optional[str] = None):
        """向主内存追加条目"""
        self.main_context.add(content, metadata={"key": key} if key else {})

    def core_memory_replace(self, key: str, content: str):
        """替换主内存中的条目"""
        return self.main_context.replace(key, content)

    def archival_memory_search(self, query: str, top_k: int = 5) -> List[str]:
        """搜索档案记忆"""
        results = self.archival_memory.search(query, top_k)
        return [r.content for r in results]

    def archival_memory_insert(self, content: str, metadata: Optional[Dict] = None):
        """插入档案记忆"""
        self.archival_memory.insert(content, metadata)

    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            "main_context": {
                "entries": len(self.main_context.entries),
                "utilization": f"{self.main_context.utilization():.1%}"
            },
            "recall_storage": {
                "entries": len(self.recall_storage.entries),
                "capacity": self.recall_storage.capacity
            },
            "archival_memory": self.archival_memory.get_stats()
        }


# ============================================
# 母婴电商场景 — 长期用户对话记忆
# ============================================

def demo_long_term_conversation():
    """演示 MemGPT 风格的长期用户对话"""
    print("=" * 70)
    print("MemGPT Agent — 长期用户对话记忆")
    print("=" * 70)

    agent = MemGPTAgent(agent_id="CustomerService-01")

    # 模拟历史对话（预填充记忆）
    print("\n[初始化] 预填充历史记忆")
    agent.archival_memory_insert("2025-12: 用户购买 Spectra S1 吸奶器，当时宝宝即将出生")
    agent.archival_memory_insert("2026-01: 用户反馈 Spectra S1 吸力够用，但体积大不便携")
    agent.archival_memory_insert("2026-01: 用户咨询配件购买，推荐储奶袋和温奶器")
    agent.archival_memory_insert("2026-02: 用户反馈夜间模式很方便，静音效果好")

    print("  档案记忆: 4 条历史记录")

    # 新对话
    print("\n[新对话]")
    user_inputs = [
        "我上次咨询的吸奶器，现在宝宝 3 个月了，需要换吗？",
        "那便携版多少钱？",
        "我之前说体积大不方便，这个便携版解决了是吧？",
    ]

    for user_input in user_inputs:
        print(f"\n  User: {user_input}")
        response = agent.chat(user_input)
        print(f"  Agent: {response}")

    # 记忆统计
    print(f"\n[记忆统计]")
    stats = agent.get_memory_stats()
    print(f"  主内存: {stats['main_context']['entries']} 条目 (利用率: {stats['main_context']['utilization']})")
    print(f"  回忆存储: {stats['recall_storage']['entries']} / {stats['recall_storage']['capacity']} 条目")
    print(f"  档案记忆: {stats['archival_memory']['total_entries']} 条目")

    print("\n" + "=" * 70)


def demo_voc_knowledge_accumulation():
    """演示 VOC 知识累积与复用"""
    print("\n" + "=" * 70)
    print("MemGPT Agent — VOC 知识累积与复用")
    print("=" * 70)

    agent = MemGPTAgent(agent_id="VOC-Analyst-01")

    # 累积历史洞察
    print("\n[历史洞察累积]")
    insights = [
        "2025-10: Spectra S1 静音提及率 34%，是核心卖点",
        "2025-11: 价格负面情感从 8% 上升至 15%",
        "2025-12: 竞品 Medela 推出静音款，构成威胁",
        "2026-01: 便携性负面提及率 12.8%，是最大单一痛点",
        "2026-02: 双边设计好评率 88%，是差异化优势",
    ]
    for insight in insights:
        agent.archival_memory_insert(insight, metadata={"type": "voc_insight"})
        print(f"  存入: {insight[:50]}...")

    # 新分析任务
    print("\n[新任务] 分析本周 Spectra S1 评论")
    print("  Agent 自动检索相关历史洞察...")

    retrieved = agent.archival_memory_search("Spectra S1 历史洞察", top_k=5)
    print(f"\n  检索到 {len(retrieved)} 条相关洞察:")
    for r in retrieved:
        print(f"    - {r[:60]}...")

    print(f"\n  Agent 分析时会重点关注:")
    print(f"    - 静音提及率是否变化？")
    print(f"    - 价格负面是否继续上升？")
    print(f"    - Medela 竞品的影响是否显现？")

    # 新洞察写入
    new_insight = "2026-03: 静音提及率降至 28%（-6pp），可能受 Medela 竞品影响"
    agent.archival_memory_insert(new_insight)
    print(f"\n  新洞察存入档案: {new_insight}")

    print("\n" + "=" * 70)


def demonstrate_memory_architecture():
    """展示三层记忆架构"""
    print("\n" + "=" * 70)
    print("MemGPT 三层记忆架构")
    print("=" * 70)

    print("""
    三层记忆架构:

    ┌──────────────────────────────────────────────┐
    │ Main Context (RAM)                           │
    │ 容量: 有限 (8K-128K tokens)                  │
    │ 速度: 最快 (直接访问)                         │
    │ 内容: 当前对话 + 活跃记忆 + 任务状态          │
    │                                              │
    │ [用户输入]                                    │
    │ [检索到的相关记忆]                            │
    │ [Agent 回复]                                  │
    └──────────────────────────────────────────────┘
              ↑↓ 换入换出 (LLM 主动控制)
    ┌──────────────────────────────────────────────┐
    │ Recall Storage (磁盘缓存)                     │
    │ 容量: 中等 (数千条)                          │
    │ 速度: 快 (内存检索)                           │
    │ 内容: 近期对话历史、最近访问的记忆             │
    └──────────────────────────────────────────────┘
              ↑↓ 归档/检索
    ┌──────────────────────────────────────────────┐
    │ Archival Memory (磁盘/向量库)                  │
    │ 容量: 无限                                    │
    │ 速度: 慢 (向量搜索)                           │
    │ 内容: 所有历史、学习到的知识、用户画像         │
    └──────────────────────────────────────────────┘

    与传统 RAG 的区别:
      RAG: 外部系统决定检索什么 → 被动
      MemGPT: LLM 自己决定存什么、取什么 → 主动

    关键创新:
      - 虚拟上下文: LLM 感觉上下文无限
      - 主动管理: LLM 通过函数调用控制记忆
      - 分层存储: 不同活跃度数据放不同层级
      - OS 类比: 页置换、中断、缓存一致
    """)


if __name__ == "__main__":
    demo_long_term_conversation()
    demo_voc_knowledge_accumulation()
    demonstrate_memory_architecture()

    print("\n生产环境建议:")
    print("  1. 使用 Pinecone/Milvus/Weaviate 作为 Archival Memory")
    print("  2. 使用 Redis 作为 Recall Storage")
    print("  3. 实现记忆去重和合并（避免重复存储相似信息）")
    print("  4. 定期归档和压缩（旧记忆生成摘要，删除细节）")
    print("  5. 与 Reflexion 集成：反思结果自动写入 Archival Memory")
    print("  6. 考虑 Letta（原 MemGPT 商业版）用于生产部署")
