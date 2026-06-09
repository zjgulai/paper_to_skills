---
title: Cross-Org Agent Protocol — 跨组织多智能体协调协议：多委托人、联邦编排、工作区委托
doc_type: knowledge
module: 10-MAS
topic: cross-org-agent-protocol
status: stable
created: 2026-06-04
updated: 2026-06-04
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Cross-Org Agent Protocol — 跨组织 Agent 协调协议

> **图谱定位**：Layer 4 桥接层｜`G²CP` + `MCP-A2A` 的协议上层｜填补跨组织信任边界空白

---

## ① 算法原理

### 核心思想

现有 MAS 协议栈有两个标准：
- **MCP**（Model Context Protocol）：解决 Agent 与工具/数据源的连接（Host ↔ Server）
- **A2A**（Agent-to-Agent）：解决同一 orchestrator 控制下的 Agent 委托

但都不解决**跨组织场景**：当 Agent 系统需要与另一个公司的 Agent 系统协作时，谁信任谁？谁控制谁？如何处理利益冲突？

两篇论文填补这个空白：

| 论文 | 填补的空白 | 核心机制 |
|------|----------|---------|
| **MPAC** (2604.09744) | 多委托人（Multi-Principal）协调 | 21 条消息类型 + 5 层协调语义 + 仲裁机制 |
| **ACP** (2602.15055) | 联邦式去中心化 Agent 通信 | DHT 发现 + DID 身份验证 + 动态 SLA 谈判 |

同期 IETF MACP（draft-li-dmsc-macp-05）将这类协议纳入标准化轨道。

### MPAC：多委托人协调协议

**场景**：品牌 A 的采购 Agent 需要与供应商 B 的报价 Agent 协作——两个 Agent 分属不同公司（不同委托人 Principal），有各自的授权范围和利益目标。

**5 层协调语义**：

```
Layer 1: Session（会话层）
  建立跨组织安全通信信道
  消息类型：HELLO, AUTH_REQUEST, AUTH_GRANT, SESSION_END
  作用：身份认证 + 会话加密

Layer 2: Intent（意图层）
  声明本次协作的目标和约束
  消息类型：DECLARE_INTENT, ACCEPT_INTENT, REJECT_INTENT, COUNTER_INTENT
  作用：对齐目标，防止误解

Layer 3: Operation（操作层）
  具体任务执行的请求与响应
  消息类型：REQUEST, RESPONSE, DELEGATE, CALLBACK
  作用：任务执行与结果传递

Layer 4: Conflict（冲突层）
  处理利益冲突和分歧
  消息类型：DISPUTE, ARBITRATE, MEDIATE, RESOLVE
  作用：自动化争议解决（无需人工介入）

Layer 5: Governance（治理层）
  记录和审计跨组织交互
  消息类型：LOG, AUDIT_REQUEST, COMPLIANCE_CHECK, TERMINATE
  作用：可追溯性和合规性
```

**共 21 条消息类型**，覆盖跨组织协作的完整生命周期。

**仲裁机制**：当两个 Agent 出现利益冲突（如价格谈判僵局），触发 Layer 4 ARBITRATE 消息：
1. 双方提交各自的约束条件（价格区间、质量要求）
2. 中立仲裁器（预设规则引擎）找到 Pareto 最优解
3. 双方 Agent 根据仲裁结果更新行为

### ACP：联邦式 Agent 通信协议

**核心创新**：不依赖中央注册中心，用去中心化技术栈实现 Agent 的发现、身份验证和动态协议协商。

**三层技术栈**：

```
Layer 1: 联邦式发现（Federated Discovery）
  技术：DHT（分布式哈希表）+ 联盟区块链
  工作方式：
    Agent 注册时 → 向 DHT 发布能力描述
    Agent 查找协作方 → DHT 查询（无中央服务器）
  优势：无单点故障，跨组织无需共同信任的第三方

Layer 2: DID 身份验证（Decentralized Identity）
  技术：W3C DID 标准
  工作方式：
    每个 Agent 有 DID（去中心化身份标识符）
    跨组织通信时，DID 验证身份（不暴露内部信息）
    可选：零知识证明（只证明"我有权限"，不暴露权限内容）

Layer 3: 动态 SLA 谈判
  工作方式：
    A 方 Agent 提出 SLA 草案（响应时间/数据格式/重试策略）
    B 方 Agent 接受/修改
    双方签署的 SLA 自动成为通信约束条件
  效果：跨平台通信延迟降低 40%（无需人工对接 API 文档）
```

---

## ② 母婴出海应用场景

### 场景一：跨品牌供应链协同（MPAC）

**业务背景**：母婴品牌与代工工厂建立"智能协同"关系——品牌的采购 Agent 需要直接与工厂的产能规划 Agent 协作，实时获取产能数据、协商交期。但两方 Agent 系统分属不同公司，各有利益。

**MPAC 协作流程**：

```
Layer 1 Session：
  品牌采购 Agent ← AUTH_REQUEST → 工厂产能 Agent
  工厂验证品牌授权证书 → AUTH_GRANT
  建立加密会话

Layer 2 Intent：
  品牌 Agent: DECLARE_INTENT {
    "goal": "获取 Q3 产能承诺",
    "quantity": 50000,
    "deadline": "2026-07-15",
    "quality_standard": "ISO 13485"
  }
  工厂 Agent: COUNTER_INTENT {
    "goal": "同意，但数量上限 40000，交期延后 7 天"
  }

Layer 3 Operation：
  品牌接受 COUNTER_INTENT
  REQUEST: "请确认 40000 单位，交期 2026-07-22"
  RESPONSE: "确认，附产能排期文档"

Layer 4 Conflict（假设价格争议）：
  品牌 DISPUTE: "单价 $4.20 超出预算"
  工厂 DISPUTE: "材料成本上涨，$4.20 是底线"
  ARBITRATE: 系统找到折中：$4.10 + 1000 单位免检优惠
  RESOLVE: 双方接受

效果：原本 3 轮人工邮件沟通（1 周）→ Agent 自动化协商（2 小时）
```

### 场景二：跨平台广告 MAS 协同（ACP）

**业务背景**：品牌在 Amazon、TikTok、独立站三个平台各有独立 Agent 系统（不同供应商提供）。大促期间需要三个系统协同调配预算，但三方 Agent 来自不同厂商，无共同 API。

**ACP 联邦发现**：

```
初始化（一次性）：
  Amazon Agent: 注册 DID → 发布能力描述到 DHT
    capabilities: ["budget_allocation", "bid_adjustment", "reporting"]
  TikTok Agent: 同上
  独立站 Agent: 同上

大促协同（实时）：
  Amazon Agent 查询 DHT: "谁能提供 budget_allocation?"
  → 发现 TikTok Agent 和独立站 Agent

  三方 SLA 谈判（动态）：
    Amazon 提案: "每 15 分钟同步预算数据，JSON 格式"
    TikTok 接受，独立站提出: "JSON 格式 OK，但需要 30 分钟"
    SLA 最终: 15 分钟同步 Amazon↔TikTok，30 分钟同步独立站

效果：三平台预算协同延迟从 4 小时（人工操作）→ 30 分钟（Agent 自动化）
```

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/cross_org_protocol/model.py`

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import hashlib


class MPACLayer(Enum):
    SESSION = "session"
    INTENT = "intent"
    OPERATION = "operation"
    CONFLICT = "conflict"
    GOVERNANCE = "governance"


class MPACMessageType(Enum):
    HELLO = "HELLO"
    AUTH_REQUEST = "AUTH_REQUEST"
    AUTH_GRANT = "AUTH_GRANT"
    AUTH_DENY = "AUTH_DENY"
    DECLARE_INTENT = "DECLARE_INTENT"
    ACCEPT_INTENT = "ACCEPT_INTENT"
    COUNTER_INTENT = "COUNTER_INTENT"
    REJECT_INTENT = "REJECT_INTENT"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    DISPUTE = "DISPUTE"
    ARBITRATE = "ARBITRATE"
    RESOLVE = "RESOLVE"
    LOG = "LOG"
    TERMINATE = "TERMINATE"


@dataclass
class MPACMessage:
    msg_type: MPACMessageType
    sender_id: str
    receiver_id: str
    layer: MPACLayer
    payload: Dict[str, Any]
    session_id: str
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def sign(self, private_key_sim: str) -> "MPACMessage":
        content = f"{self.msg_type.value}{self.sender_id}{self.payload}{self.timestamp}"
        self.signature = hashlib.md5((content + private_key_sim).encode()).hexdigest()[:16]
        return self

    def verify(self, public_key_sim: str) -> bool:
        content = f"{self.msg_type.value}{self.sender_id}{self.payload}{self.timestamp}"
        expected = hashlib.md5((content + public_key_sim).encode()).hexdigest()[:16]
        return self.signature == expected


class MPACSession:
    def __init__(self, session_id: str, principal_a: str, principal_b: str):
        self.session_id = session_id
        self.principal_a = principal_a
        self.principal_b = principal_b
        self.state = "initializing"
        self.intent: Optional[Dict] = None
        self.messages: List[MPACMessage] = []
        self.sla: Optional[Dict] = None

    def send(self, msg: MPACMessage) -> Optional[MPACMessage]:
        self.messages.append(msg)

        if msg.msg_type == MPACMessageType.AUTH_REQUEST:
            self.state = "authenticating"
            return MPACMessage(
                msg_type=MPACMessageType.AUTH_GRANT,
                sender_id=msg.receiver_id, receiver_id=msg.sender_id,
                layer=MPACLayer.SESSION, payload={"session_token": self.session_id},
                session_id=self.session_id,
            )

        elif msg.msg_type == MPACMessageType.AUTH_GRANT:
            self.state = "authenticated"
            return None

        elif msg.msg_type == MPACMessageType.DECLARE_INTENT:
            self.intent = msg.payload
            self.state = "negotiating"
            return None

        elif msg.msg_type in (MPACMessageType.ACCEPT_INTENT, MPACMessageType.COUNTER_INTENT):
            if msg.msg_type == MPACMessageType.ACCEPT_INTENT:
                self.state = "intent_agreed"
            return None

        elif msg.msg_type == MPACMessageType.DISPUTE:
            self.state = "in_conflict"
            return self._arbitrate(msg)

        elif msg.msg_type == MPACMessageType.TERMINATE:
            self.state = "terminated"
            return None

        return None

    def _arbitrate(self, dispute: MPACMessage) -> MPACMessage:
        constraints = dispute.payload.get("constraints", {})
        resolution = {}
        for key, val in constraints.items():
            if isinstance(val, (int, float)):
                resolution[key] = val * 0.97
            else:
                resolution[key] = val

        self.state = "resolved"
        return MPACMessage(
            msg_type=MPACMessageType.ARBITRATE,
            sender_id="arbitrator", receiver_id="both",
            layer=MPACLayer.CONFLICT,
            payload={"resolution": resolution, "method": "pareto_compromise"},
            session_id=self.session_id,
        )

    def get_audit_log(self) -> List[Dict]:
        return [{"type": m.msg_type.value, "from": m.sender_id, "ts": m.timestamp}
                for m in self.messages]


@dataclass
class AgentDID:
    agent_id: str
    organization: str
    capabilities: List[str]
    did: str = ""
    public_key: str = ""

    def __post_init__(self):
        self.did = f"did:example:{hashlib.md5(self.agent_id.encode()).hexdigest()[:16]}"
        self.public_key = hashlib.md5(f"pk_{self.agent_id}".encode()).hexdigest()[:32]


class ACPRegistry:
    def __init__(self):
        self._dht: Dict[str, AgentDID] = {}

    def register(self, agent: AgentDID):
        for cap in agent.capabilities:
            self._dht.setdefault(cap, [])
            if not isinstance(self._dht[cap], list):
                self._dht[cap] = [self._dht[cap]]
            self._dht[cap].append(agent)
        self._dht[agent.did] = agent

    def discover(self, capability: str) -> List[AgentDID]:
        result = self._dht.get(capability, [])
        if isinstance(result, AgentDID):
            return [result]
        return result

    def resolve_did(self, did: str) -> Optional[AgentDID]:
        result = self._dht.get(did)
        return result if isinstance(result, AgentDID) else None


@dataclass
class SLAContract:
    party_a: str
    party_b: str
    sync_interval_minutes: int
    data_format: str
    retry_policy: Dict[str, Any]
    valid_until: float = field(default_factory=lambda: time.time() + 86400)

    def is_valid(self) -> bool:
        return time.time() < self.valid_until


class ACPNegotiator:
    def negotiate_sla(self, agent_a: AgentDID, agent_b: AgentDID,
                      proposal_a: Dict, proposal_b: Dict) -> SLAContract:
        sync_a = proposal_a.get("sync_interval_minutes", 15)
        sync_b = proposal_b.get("sync_interval_minutes", 15)
        agreed_sync = max(sync_a, sync_b)

        format_a = proposal_a.get("data_format", "json")
        format_b = proposal_b.get("data_format", "json")
        agreed_format = format_a if format_a == format_b else "json"

        return SLAContract(
            party_a=agent_a.agent_id, party_b=agent_b.agent_id,
            sync_interval_minutes=agreed_sync, data_format=agreed_format,
            retry_policy={"max_retries": 3, "backoff": "exponential"},
        )
```

---

## ④ 技能关联

### 前置技能
- [[Skill-Graph-Grounded-MAS-Protocol]]：图通信协议 → 跨组织协议的通信基础
- [[Skill-MCP-A2A-Protocol-Stack]]：MCP/A2A 协议栈（16-智能体工程）→ 跨组织是单组织协议的扩展

### 延伸技能
- [[Skill-Agent-Registry-Discovery]]：注册发现 → 跨组织扩展（ACP 的 DHT 发现）

### 可组合技能
- [[Skill-MAS-Dynamic-Trust]]：动态信任 ↔ 跨组织协作 = 跨信任边界，需要信任建立
- [[Skill-LDP-Identity-Aware-Protocol]]：身份认证 ↔ ACP 的 DID 身份验证互补

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 品牌-工厂跨组织协同：缩短产能协商周期 1 周 → 2 小时；三平台预算协同：大促期间跨平台响应延迟 4h → 30min（多次大促累计效益显著） |
| **实施难度** | ⭐⭐⭐⭐☆（需要对方系统接入协议；MPAC 有 PyPI 包可以降低实施成本；ACP 的 DHT + DID 需要基础设施支持） |
| **优先级评分** | ⭐⭐☆☆☆（短期业务主要在单组织内；当供应链数字化或多平台协同需求出现时，优先级升至 P0） |
| **评估依据** | MPAC：PyPI 包已发布（可验证）；ACP：跨平台延迟 -40%；IETF 标准化轨道（长期合规价值） |

---

## 论文来源

| 论文 | arXiv | 年份 |
|------|-------|------|
| MPAC: Multi-Principal Agent Coordination Protocol | [2604.09744](https://arxiv.org/abs/2604.09744) | 2026-04 |
| ACP: Unified Agent Communication Protocol | [2602.15055](https://arxiv.org/abs/2602.15055) | 2026-02 |
| IETF MACP draft | [draft-li-dmsc-macp-05](https://datatracker.ietf.org/doc/html/draft-li-dmsc-macp-05) | 2026 |
