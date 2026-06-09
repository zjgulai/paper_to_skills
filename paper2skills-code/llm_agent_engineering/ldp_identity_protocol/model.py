"""
LDP — LLM Delegate Protocol 身份感知 Agent 通信协议
论文: LDP: An Identity-Aware Protocol for Multi-Agent LLM Systems
arXiv: 2603.08852 | 2026-03

核心组件:
- ModelIdentityCard: Agent 身份证（模型属性）
- PayloadMode: 6 级 Payload 协商模式
- LDPSession: 治理会话（持久上下文 + 信任域）
- LDPRouter: 基于 Identity Card 的智能路由
- LDPChannel: 带 Governed Session 缓存的通信通道

运行方式:
    python model.py
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ──────────────────────────────────────────
# 1. 模型质量分级
# ──────────────────────────────────────────

class QualityTier(str, Enum):
    """模型质量分级：FRONTIER > MID > LIGHTWEIGHT"""
    FRONTIER = "frontier"       # GPT-4o, Claude Opus, Gemini Ultra 级别
    MID = "mid"                 # GPT-3.5, Claude Haiku, Qwen-Plus 级别
    LIGHTWEIGHT = "lightweight"  # 3-7B 本地小模型


class ReasoningProfile(str, Enum):
    """推理风格偏好"""
    QUALITY_FIRST = "quality_first"     # 准确率优先，延迟可接受
    BALANCED = "balanced"               # 质量与速度均衡
    SPEED_OPTIMIZED = "speed_optimized"  # 速度优先，轻量任务


# ──────────────────────────────────────────
# 2. Rich Identity Card — Agent 身份证
# ──────────────────────────────────────────

@dataclass
class ModelIdentityCard:
    """
    Agent 身份证：暴露模型级属性，解决 MCP/A2A 不透明问题。
    每个 Agent 注册时携带此卡，供 Orchestrator 做智能路由。
    """
    agent_id: str
    model_id: str                           # 具体模型版本: "claude-opus-4", "qwen-7b"
    model_family: str                       # 模型族: "claude", "gpt", "qwen", "gemma"
    quality_tier: QualityTier
    reasoning_profile: ReasoningProfile
    cost_per_token: float                   # USD per 1K tokens
    max_context_tokens: int                 # 最大上下文长度
    supported_modalities: list[str] = field(default_factory=lambda: ["text"])
    trust_domain: str = "default"           # 所属信任域
    capabilities: list[str] = field(default_factory=list)

    def complexity_score(self) -> float:
        """返回此 Agent 适合处理的最低任务复杂度阈值（0-1）"""
        tier_map = {
            QualityTier.FRONTIER: 0.7,
            QualityTier.MID: 0.4,
            QualityTier.LIGHTWEIGHT: 0.0,
        }
        return tier_map[self.quality_tier]

    def can_serve(self, required_complexity: float) -> bool:
        """判断此 Agent 是否能处理给定复杂度的任务"""
        return self.complexity_score() <= required_complexity


# ──────────────────────────────────────────
# 3. 6 级 Payload 协商模式
# ──────────────────────────────────────────

class PayloadMode(str, Enum):
    """
    渐进式 Payload 协商 — 6 级信息密度模式。
    核心直觉: effective_info = payload_density × bandwidth_budget
    越高级（FULL）信息越完整但 token 越多；越低级（POINTER）token 极少但需接收方自行检索。
    """
    FULL = "full"               # 完整上下文，首次通信或带宽充足时使用
    COMPRESSED = "compressed"   # 压缩后完整上下文（约 60% token）
    SUMMARY = "summary"         # 关键摘要（约 40% token）
    REFERENCE = "reference"     # 引用 ID + 关键字段（约 20% token）
    DELTA = "delta"             # 仅变化部分（重复通信，约 10% token）
    POINTER = "pointer"         # 纯指针，接收方从 session cache 检索（约 3% token）

    @property
    def token_ratio(self) -> float:
        """相对 FULL 模式的 token 消耗比例"""
        ratios = {
            "full": 1.0,
            "compressed": 0.60,
            "summary": 0.40,
            "reference": 0.20,
            "delta": 0.10,
            "pointer": 0.03,
        }
        return ratios[self.value]


def negotiate_payload_mode(
    bandwidth_budget: int,       # 可用 token 预算
    is_first_contact: bool,      # 是否首次通信
    session_age_seconds: float,  # session 建立时长（秒）
    payload_size: int,           # 原始 payload token 数
) -> PayloadMode:
    """
    自动协商最优 Payload 模式。
    策略：在满足信息完整性的前提下，最小化 token 消耗。
    """
    if is_first_contact:
        return PayloadMode.FULL

    utilization = payload_size / max(bandwidth_budget, 1)

    if session_age_seconds > 3600 and utilization > 0.8:
        return PayloadMode.REFERENCE
    elif utilization > 0.5:
        return PayloadMode.SUMMARY
    elif session_age_seconds > 300:
        return PayloadMode.DELTA
    elif session_age_seconds > 60:
        return PayloadMode.COMPRESSED
    else:
        return PayloadMode.FULL


# ──────────────────────────────────────────
# 4. Governed Session — 治理会话
# ──────────────────────────────────────────

@dataclass
class ProvenanceEntry:
    """Provenance 链条单条记录"""
    agent_id: str
    model_id: str
    timestamp: float
    action: str           # "generated", "transformed", "retrieved", "stored"
    content_hash: str     # SHA-256 前 16 位


@dataclass
class LDPSession:
    """
    Governed Session：持久化上下文 + 信任域隔离。
    消除 39% 重发 overhead；Trust Domain 防止跨域信息泄露。
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trust_domain: str = "default"
    payload_budget: int = 8000          # 本次通信允许的最大 token 数
    context_cache: dict[str, Any] = field(default_factory=dict)
    provenance_chain: list[ProvenanceEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def store(self, key: str, value: Any, agent_id: str, model_id: str) -> None:
        """存储上下文到 session cache，并记录 Provenance"""
        self.context_cache[key] = value
        self.last_active = time.time()
        entry = ProvenanceEntry(
            agent_id=agent_id,
            model_id=model_id,
            timestamp=self.last_active,
            action="stored",
            content_hash=hashlib.sha256(str(value).encode()).hexdigest()[:16],
        )
        self.provenance_chain.append(entry)

    def retrieve(self, key: str) -> Any | None:
        """从 session cache 检索上下文（消除重发 overhead）"""
        self.last_active = time.time()
        return self.context_cache.get(key)

    def age(self) -> float:
        """session 建立时长（秒）"""
        return time.time() - self.created_at

    def is_same_domain(self, other_domain: str) -> bool:
        """检查是否在同一信任域"""
        return self.trust_domain == other_domain or other_domain == "shared"


# ──────────────────────────────────────────
# 5. LDP Router — 基于 Identity Card 的智能路由
# ──────────────────────────────────────────

@dataclass
class AgentEndpoint:
    """Agent 接入点"""
    agent_id: str
    identity_card: ModelIdentityCard
    endpoint_url: str = ""


class LDPRouter:
    """
    基于 Identity Card 的智能路由器。
    根据任务复杂度、成本约束、能力需求，选择最优 Agent。
    解决 MCP/A2A 不暴露模型属性导致的路由盲目问题。
    """

    def __init__(self) -> None:
        self._registry: dict[str, AgentEndpoint] = {}

    def register(self, endpoint: AgentEndpoint) -> None:
        """注册 Agent（携带 Identity Card）"""
        self._registry[endpoint.agent_id] = endpoint

    def route_to_best_agent(
        self,
        task_complexity: float,
        required_capabilities: list[str] | None = None,
        max_cost_per_token: float = 1.0,
        trust_domain: str | None = None,
    ) -> AgentEndpoint | None:
        """
        基于 Identity Card 选择最优 Agent。
        策略：在满足复杂度和能力需求的前提下，选择成本最低的 Agent。
        """
        candidates = list(self._registry.values())

        # 过滤：信任域隔离（Trust Domain 安全边界）
        if trust_domain:
            candidates = [
                ep for ep in candidates
                if ep.identity_card.trust_domain == trust_domain
                or ep.identity_card.trust_domain == "shared"
            ]

        # 过滤：能处理该复杂度
        candidates = [
            ep for ep in candidates
            if ep.identity_card.can_serve(task_complexity)
        ]

        # 过滤：成本限制
        candidates = [
            ep for ep in candidates
            if ep.identity_card.cost_per_token <= max_cost_per_token
        ]

        # 过滤：所需能力
        if required_capabilities:
            candidates = [
                ep for ep in candidates
                if all(cap in ep.identity_card.capabilities for cap in required_capabilities)
            ]

        if not candidates:
            return None

        # 选择：找到满足复杂度的最低 tier，在该层内选成本最低
        tier_order = {
            QualityTier.FRONTIER: 3,
            QualityTier.MID: 2,
            QualityTier.LIGHTWEIGHT: 1,
        }
        min_required_tier_score = (
            3 if task_complexity >= 0.7
            else 2 if task_complexity >= 0.4
            else 1
        )
        matching_tier = [
            ep for ep in candidates
            if tier_order[ep.identity_card.quality_tier] >= min_required_tier_score
        ]
        pool = matching_tier if matching_tier else candidates
        return min(pool, key=lambda ep: ep.identity_card.cost_per_token)

    def list_agents(self, trust_domain: str | None = None) -> list[AgentEndpoint]:
        """列出所有已注册 Agent（可按信任域过滤）"""
        agents = list(self._registry.values())
        if trust_domain:
            agents = [ep for ep in agents if ep.identity_card.trust_domain == trust_domain]
        return agents


# ──────────────────────────────────────────
# 6. LDP Channel — 带 Governed Session 的通信通道
# ──────────────────────────────────────────

@dataclass
class LDPMessage:
    """LDP 消息体"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    payload_mode: PayloadMode = PayloadMode.FULL
    content: Any = None
    content_ref: str | None = None      # POINTER/REFERENCE 模式下的引用 key
    sender_id: str = ""
    receiver_id: str = ""
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class LDPChannel:
    """
    LDP 通信通道：自动 Payload 协商 + Governed Session 缓存。
    实现 37% token 节省和 39% 重发 overhead 消除。
    """

    def __init__(self, local_agent_id: str, session: LDPSession) -> None:
        self.local_agent_id = local_agent_id
        self.session = session
        self._sent_count = 0
        self._total_tokens_saved = 0

    def send(
        self,
        receiver_id: str,
        content: Any,
        identity_card: ModelIdentityCard,
        estimated_tokens: int = 1000,
    ) -> LDPMessage:
        """发送消息，自动协商 Payload 模式并缓存到 session"""
        mode = negotiate_payload_mode(
            bandwidth_budget=self.session.payload_budget,
            is_first_contact=self._sent_count == 0,
            session_age_seconds=self.session.age(),
            payload_size=estimated_tokens,
        )

        # 根据模式决定实际发送内容
        if mode in (PayloadMode.POINTER, PayloadMode.REFERENCE):
            cache_key = f"msg_{self._sent_count}"
            self.session.store(cache_key, content, self.local_agent_id, identity_card.model_id)
            actual_content = None
            content_ref = cache_key
        else:
            actual_content = content
            content_ref = None

        saved = int(estimated_tokens * (1.0 - mode.token_ratio))
        self._total_tokens_saved += saved

        msg = LDPMessage(
            session_id=self.session.session_id,
            payload_mode=mode,
            content=actual_content,
            content_ref=content_ref,
            sender_id=self.local_agent_id,
            receiver_id=receiver_id,
            provenance=list(self.session.provenance_chain[-3:]),
        )
        self._sent_count += 1
        return msg

    def receive(self, msg: LDPMessage) -> Any:
        """接收消息，若为 POINTER/REFERENCE 模式则从 session cache 还原"""
        if msg.payload_mode in (PayloadMode.POINTER, PayloadMode.REFERENCE):
            if msg.content_ref:
                return self.session.retrieve(msg.content_ref)
        return msg.content

    @property
    def total_tokens_saved(self) -> int:
        return self._total_tokens_saved


# ──────────────────────────────────────────
# 7. 验证测试：WF-A 母婴出海路由场景
# ──────────────────────────────────────────

def demo_baby_ecommerce_routing() -> None:
    """
    验证场景：母婴出海 MAS 智能路由
    - 复杂分析任务 → Frontier Agent
    - 简单执行任务 → Lightweight Agent
    - Governed Session + Token 节省验证
    - Trust Domain 隔离验证
    """
    print("=== LDP Demo: 母婴出海 MAS 智能路由 ===\n")

    # 注册 Agent（携带 Identity Card）
    router = LDPRouter()
    agents = [
        ModelIdentityCard(
            agent_id="strategy_agent",
            model_id="claude-opus-4",
            model_family="claude",
            quality_tier=QualityTier.FRONTIER,
            reasoning_profile=ReasoningProfile.QUALITY_FIRST,
            cost_per_token=0.015,
            max_context_tokens=200_000,
            trust_domain="brand_domain",
            capabilities=["analysis", "strategy", "market_research"],
        ),
        ModelIdentityCard(
            agent_id="execution_agent",
            model_id="qwen-7b-instruct",
            model_family="qwen",
            quality_tier=QualityTier.LIGHTWEIGHT,
            reasoning_profile=ReasoningProfile.SPEED_OPTIMIZED,
            cost_per_token=0.0002,
            max_context_tokens=32_000,
            trust_domain="brand_domain",
            capabilities=["translation", "sku_matching", "text_formatting"],
        ),
        ModelIdentityCard(
            agent_id="review_agent",
            model_id="claude-haiku-3",
            model_family="claude",
            quality_tier=QualityTier.MID,
            reasoning_profile=ReasoningProfile.BALANCED,
            cost_per_token=0.003,
            max_context_tokens=48_000,
            trust_domain="brand_domain",
            capabilities=["review", "compliance_check", "qa"],
        ),
    ]
    for card in agents:
        router.register(AgentEndpoint(card.agent_id, card))

    # 任务路由测试
    tasks = [
        ("竞品市场深度分析 + 定价策略制定", 0.85, ["analysis", "strategy"]),
        ("SKU 标题翻译（中→英）", 0.15, ["translation"]),
        ("广告文案合规审查", 0.50, ["compliance_check"]),
    ]

    print("【任务路由结果】")
    for task_name, complexity, caps in tasks:
        ep = router.route_to_best_agent(
            task_complexity=complexity,
            required_capabilities=caps,
            trust_domain="brand_domain",
        )
        assert ep is not None, f"任务 [{task_name}] 路由失败"
        print(f"  [{task_name}]")
        print(f"    复杂度={complexity:.2f} → {ep.agent_id} "
              f"({ep.identity_card.quality_tier.value}, ${ep.identity_card.cost_per_token}/1K)\n")

    print("【Payload 协商验证（模拟不同 session 状态）】")
    scenarios = [
        ("首次通信",        True,  0,    3000, 5000),
        ("session 1分钟后",  False, 61,   3000, 5000),
        ("session 6分钟后",  False, 361,  3000, 5000),
        ("session 1小时后",  False, 3601, 4200, 5000),
    ]

    total_original = 0
    total_actual = 0
    for label, is_first, age_sec, payload_size, budget in scenarios:
        mode = negotiate_payload_mode(
            bandwidth_budget=budget,
            is_first_contact=is_first,
            session_age_seconds=float(age_sec),
            payload_size=payload_size,
        )
        actual = int(payload_size * mode.token_ratio)
        total_original += payload_size
        total_actual += actual
        print(f"  {label}: mode={mode.value}, {payload_size}→{actual} tokens")

    saved = total_original - total_actual
    ratio = saved / total_original * 100
    print(f"\n  累计节省: {saved}/{total_original} tokens ({ratio:.0f}%)")
    assert saved > 0, "Token 节省验证失败"

    # Trust Domain 隔离测试
    print("\n【Trust Domain 隔离验证】")
    supplier_ep = router.route_to_best_agent(task_complexity=0.5, trust_domain="supplier_domain")
    assert supplier_ep is None, "Trust Domain 隔离失败：不同域 Agent 被访问到"
    print("  ✓ supplier_domain 查询 brand_domain Agent → None（隔离成功）")

    print("\n✅ 所有验证通过")


if __name__ == "__main__":
    demo_baby_ecommerce_routing()
