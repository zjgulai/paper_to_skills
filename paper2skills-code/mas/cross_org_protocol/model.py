from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time
import hashlib


class MPACMessageType(Enum):
    AUTH_REQUEST = "AUTH_REQUEST"
    AUTH_GRANT = "AUTH_GRANT"
    AUTH_DENY = "AUTH_DENY"
    DECLARE_INTENT = "DECLARE_INTENT"
    ACCEPT_INTENT = "ACCEPT_INTENT"
    COUNTER_INTENT = "COUNTER_INTENT"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    DISPUTE = "DISPUTE"
    ARBITRATE = "ARBITRATE"
    RESOLVE = "RESOLVE"
    TERMINATE = "TERMINATE"


@dataclass
class MPACMessage:
    msg_type: MPACMessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    session_id: str
    timestamp: float = field(default_factory=time.time)


class MPACSession:
    def __init__(self, session_id: str, principal_a: str, principal_b: str):
        self.session_id = session_id
        self.principal_a = principal_a
        self.principal_b = principal_b
        self.state = "initializing"
        self.messages: List[MPACMessage] = []

    def send(self, msg: MPACMessage) -> Optional[MPACMessage]:
        self.messages.append(msg)

        if msg.msg_type == MPACMessageType.AUTH_REQUEST:
            self.state = "authenticating"
            return MPACMessage(
                msg_type=MPACMessageType.AUTH_GRANT,
                sender_id=msg.receiver_id, receiver_id=msg.sender_id,
                payload={"session_token": self.session_id},
                session_id=self.session_id,
            )
        elif msg.msg_type == MPACMessageType.AUTH_GRANT:
            self.state = "authenticated"
        elif msg.msg_type == MPACMessageType.DECLARE_INTENT:
            self.state = "negotiating"
        elif msg.msg_type == MPACMessageType.ACCEPT_INTENT:
            self.state = "intent_agreed"
        elif msg.msg_type == MPACMessageType.COUNTER_INTENT:
            self.state = "counter_proposed"
        elif msg.msg_type == MPACMessageType.DISPUTE:
            self.state = "in_conflict"
            return self._arbitrate(msg)
        elif msg.msg_type == MPACMessageType.TERMINATE:
            self.state = "terminated"
        return None

    def _arbitrate(self, dispute: MPACMessage) -> MPACMessage:
        constraints = dispute.payload.get("constraints", {})
        resolution = {
            k: round(v * 0.97, 2) if isinstance(v, (int, float)) else v
            for k, v in constraints.items()
        }
        self.state = "resolved"
        return MPACMessage(
            msg_type=MPACMessageType.ARBITRATE,
            sender_id="arbitrator", receiver_id="both",
            payload={"resolution": resolution, "method": "pareto_compromise"},
            session_id=self.session_id,
        )

    def get_audit_log(self) -> List[Dict]:
        return [{"type": m.msg_type.value, "from": m.sender_id} for m in self.messages]


@dataclass
class AgentDID:
    agent_id: str
    organization: str
    capabilities: List[str]
    did: str = ""

    def __post_init__(self):
        if not self.did:
            self.did = f"did:example:{hashlib.md5(self.agent_id.encode()).hexdigest()[:16]}"


class ACPRegistry:
    def __init__(self):
        self._capability_index: Dict[str, List[AgentDID]] = {}
        self._did_index: Dict[str, AgentDID] = {}

    def register(self, agent: AgentDID):
        for cap in agent.capabilities:
            self._capability_index.setdefault(cap, []).append(agent)
        self._did_index[agent.did] = agent

    def discover(self, capability: str) -> List[AgentDID]:
        return list(self._capability_index.get(capability, []))

    def resolve_did(self, did: str) -> Optional[AgentDID]:
        return self._did_index.get(did)


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
        format_a = proposal_a.get("data_format", "json")
        format_b = proposal_b.get("data_format", "json")
        return SLAContract(
            party_a=agent_a.agent_id, party_b=agent_b.agent_id,
            sync_interval_minutes=max(sync_a, sync_b),
            data_format=format_a if format_a == format_b else "json",
            retry_policy={"max_retries": 3, "backoff": "exponential"},
        )


def test_mpac_auth_flow():
    session = MPACSession("sess_001", "brand_agent", "factory_agent")
    auth_req = MPACMessage(MPACMessageType.AUTH_REQUEST, "brand_agent", "factory_agent",
                           {"credentials": "cert_xyz"}, "sess_001")
    response = session.send(auth_req)
    assert response is not None
    assert response.msg_type == MPACMessageType.AUTH_GRANT
    assert session.state == "authenticating"
    print(f"[PASS] mpac_auth: state={session.state}, response={response.msg_type.value}")


def test_mpac_intent_negotiation():
    session = MPACSession("sess_002", "buyer", "seller")
    session.state = "authenticated"
    intent = MPACMessage(MPACMessageType.DECLARE_INTENT, "buyer", "seller",
                         {"quantity": 50000, "price_max": 4.20}, "sess_002")
    session.send(intent)
    assert session.state == "negotiating"
    counter = MPACMessage(MPACMessageType.COUNTER_INTENT, "seller", "buyer",
                          {"quantity": 40000, "price_min": 4.10}, "sess_002")
    session.send(counter)
    assert session.state == "counter_proposed"
    print(f"[PASS] mpac_negotiation: messages={len(session.messages)}, state={session.state}")


def test_mpac_arbitration():
    session = MPACSession("sess_003", "buyer", "seller")
    dispute = MPACMessage(MPACMessageType.DISPUTE, "buyer", "seller",
                          {"constraints": {"unit_price": 4.20, "quantity": 40000}}, "sess_003")
    result = session.send(dispute)
    assert result is not None
    assert result.msg_type == MPACMessageType.ARBITRATE
    assert session.state == "resolved"
    assert result.payload["resolution"]["unit_price"] < 4.20
    print(f"[PASS] mpac_arbitration: price={result.payload['resolution']['unit_price']:.2f}")


def test_acp_discovery():
    registry = ACPRegistry()
    amazon_agent = AgentDID("amazon_ads", "Amazon", ["budget_allocation", "bid_adjustment"])
    tiktok_agent = AgentDID("tiktok_ads", "TikTok", ["budget_allocation", "creative_optimization"])
    registry.register(amazon_agent)
    registry.register(tiktok_agent)

    results = registry.discover("budget_allocation")
    assert len(results) == 2
    ids = {a.agent_id for a in results}
    assert "amazon_ads" in ids and "tiktok_ads" in ids
    print(f"[PASS] acp_discovery: found {len(results)} agents for 'budget_allocation'")


def test_acp_sla_negotiation():
    registry = ACPRegistry()
    agent_a = AgentDID("amazon_ads", "Amazon", ["budget_allocation"])
    agent_b = AgentDID("tiktok_ads", "TikTok", ["budget_allocation"])
    negotiator = ACPNegotiator()
    sla = negotiator.negotiate_sla(
        agent_a, agent_b,
        {"sync_interval_minutes": 15, "data_format": "json"},
        {"sync_interval_minutes": 30, "data_format": "json"},
    )
    assert sla.sync_interval_minutes == 30
    assert sla.data_format == "json"
    assert sla.is_valid()
    print(f"[PASS] acp_sla: sync={sla.sync_interval_minutes}min, format={sla.data_format}")


def test_acp_did_resolve():
    registry = ACPRegistry()
    agent = AgentDID("test_agent", "TestOrg", ["reporting"])
    registry.register(agent)
    resolved = registry.resolve_did(agent.did)
    assert resolved is not None
    assert resolved.agent_id == "test_agent"
    print(f"[PASS] acp_did_resolve: DID={agent.did[:30]}...")


if __name__ == "__main__":
    test_mpac_auth_flow()
    test_mpac_intent_negotiation()
    test_mpac_arbitration()
    test_acp_discovery()
    test_acp_sla_negotiation()
    test_acp_did_resolve()
    print("\n✅ All tests passed")
