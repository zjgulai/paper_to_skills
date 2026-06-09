"""
CapSeal -- Agent 秘密中介（能力封装取代直接密钥暴露）
来源：arXiv:2604.16762 | Rust 原型 Python 移植 + MCP 适配
安全注意：生产环境 Broker 内部密钥存储须使用 HSM/Secret Manager
"""
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any


@dataclass
class CapabilityHandle:
    handle_id: str
    session_id: str
    allowed_actions: List[str]
    max_amounts: Dict[str, float]   # 如 {"po_amount_usd": 50000}
    allowed_targets: List[str]      # 供应商/账户白名单
    expiry: float                   # Unix 时间戳

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expiry

    def allows_action(self, action: str) -> bool:
        return action in self.allowed_actions

    def allows_target(self, target: str) -> bool:
        return not self.allowed_targets or target in self.allowed_targets

    def within_amount_limit(self, field_name: str, amount: float) -> bool:
        limit = self.max_amounts.get(field_name)
        return limit is None or amount <= limit


class AntiReplayState:
    """防重放三重验证：sequence/nonce/timestamp"""

    TIMESTAMP_TOLERANCE_SECONDS = 30

    def __init__(self):
        self._expected_sequence: int = 0
        self._used_nonces: Set[str] = set()

    def verify(self, sequence_num: int, nonce: str, timestamp: float) -> tuple:
        """返回 (is_valid, reason)"""
        now = time.time()
        if abs(now - timestamp) > self.TIMESTAMP_TOLERANCE_SECONDS:
            return False, "timestamp out of window"
        if nonce in self._used_nonces:
            return False, f"nonce already used: {nonce}"
        if sequence_num != self._expected_sequence:
            return False, f"sequence mismatch: expected {self._expected_sequence}, got {sequence_num}"
        return True, "ok"

    def consume(self, sequence_num: int, nonce: str) -> None:
        self._used_nonces.add(nonce)
        self._expected_sequence = sequence_num + 1


class PolicyEvaluator:
    """策略评估：检查操作是否在句柄约束范围内"""

    def check_permission(
        self,
        handle: CapabilityHandle,
        action: str,
        target: Optional[str] = None,
        amount_field: Optional[str] = None,
        amount_value: Optional[float] = None,
    ) -> tuple:
        """返回 (allowed: bool, reason: str)"""
        if handle.is_expired:
            return False, f"handle {handle.handle_id} has expired"
        if not handle.allows_action(action):
            return False, f"action '{action}' not in allowed_actions {handle.allowed_actions}"
        if target and not handle.allows_target(target):
            return False, f"target '{target}' not in allowed_targets"
        if amount_field and amount_value is not None:
            if not handle.within_amount_limit(amount_field, amount_value):
                limit = handle.max_amounts.get(amount_field)
                return False, f"{amount_field}={amount_value} exceeds limit {limit}"
        return True, "permitted"


class AuditLog:
    def __init__(self):
        self._entries: List[Dict] = []

    def record(self, handle_id: str, action: str, allowed: bool, reason: str, details: Dict) -> None:
        self._entries.append({
            "timestamp": time.time(), "handle_id": handle_id,
            "action": action, "allowed": allowed, "reason": reason, "details": details,
        })

    def get_entries(self) -> List[Dict]:
        return list(self._entries)


class CapSealBroker:
    """
    中介 Broker：Agent 声明意图 -> 策略评估 -> Broker 执行 -> 结果返回
    Agent 全程不接触真实密钥
    """

    def __init__(self):
        self._handles: Dict[str, CapabilityHandle] = {}
        self._replay_states: Dict[str, AntiReplayState] = {}
        self._policy = PolicyEvaluator()
        self._audit = AuditLog()
        self._internal_secrets: Dict[str, str] = {}  # 生产使用 HSM

    def _register_secret(self, secret_name: str, secret_value: str) -> None:
        """内部方法：注册真实密钥（仅 Broker 可访问，Agent 不可见）"""
        self._internal_secrets[secret_name] = secret_value

    def issue_handle(
        self,
        session_id: str,
        allowed_actions: List[str],
        max_amounts: Optional[Dict[str, float]] = None,
        allowed_targets: Optional[List[str]] = None,
        ttl_seconds: int = 3600,
    ) -> CapabilityHandle:
        handle_id = hashlib.sha256(f"{session_id}{time.time()}".encode()).hexdigest()[:16]
        handle = CapabilityHandle(
            handle_id=handle_id,
            session_id=session_id,
            allowed_actions=allowed_actions,
            max_amounts=max_amounts or {},
            allowed_targets=allowed_targets or [],
            expiry=time.time() + ttl_seconds,
        )
        self._handles[handle_id] = handle
        self._replay_states[handle_id] = AntiReplayState()
        return handle

    def execute_action(
        self,
        handle_id: str,
        action: str,
        sequence_num: int,
        nonce: str,
        timestamp: float,
        target: Optional[str] = None,
        amount_field: Optional[str] = None,
        amount_value: Optional[float] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        handle = self._handles.get(handle_id)
        if not handle:
            return {"success": False, "reason": f"handle {handle_id} not found"}

        replay_state = self._replay_states[handle_id]
        replay_ok, replay_reason = replay_state.verify(sequence_num, nonce, timestamp)
        if not replay_ok:
            self._audit.record(handle_id, action, False, replay_reason, {})
            return {"success": False, "reason": replay_reason}

        allowed, policy_reason = self._policy.check_permission(
            handle, action, target, amount_field, amount_value
        )
        self._audit.record(handle_id, action, allowed, policy_reason, {
            "target": target, "amount_field": amount_field, "amount_value": amount_value,
        })
        if not allowed:
            return {"success": False, "reason": policy_reason}

        replay_state.consume(sequence_num, nonce)
        return {
            "success": True,
            "result": {"action": action, "target": target, "status": "executed"},
            "executed_by": "broker",
        }

    def revoke_handle(self, handle_id: str) -> bool:
        if handle_id in self._handles:
            self._handles[handle_id].expiry = 0
            return True
        return False

    def get_audit_log(self) -> List[Dict]:
        return self._audit.get_entries()


# ===== 测试：WF-A 场景，验证①正常 ②越界 ③injection ④过期 =====
def _test_capseal_wfa():
    broker = CapSealBroker()
    broker._register_secret("erp_api_key", "SUPER_SECRET_KEY")
    handle = broker.issue_handle(
        session_id="wf-a-session-001",
        allowed_actions=["create_po"],
        max_amounts={"po_amount_usd": 50000},
        allowed_targets=["supplier_001", "supplier_002"],
        ttl_seconds=3600,
    )

    # ① 正常执行
    r1 = broker.execute_action(
        handle.handle_id, "create_po", 0, "n001", time.time(),
        target="supplier_001", amount_field="po_amount_usd", amount_value=10000,
    )
    assert r1["success"], f"正常执行应成功: {r1}"

    # ② 权限越界拒绝（action 不在白名单）
    r2 = broker.execute_action(handle.handle_id, "send_api_key", 1, "n002", time.time())
    assert not r2["success"] and "not in allowed_actions" in r2["reason"]

    # ③ Prompt Injection -- transfer_funds 被拒绝
    r3 = broker.execute_action(
        handle.handle_id, "transfer_funds", 1, "n002", time.time(),
        target="attacker", amount_field="po_amount_usd", amount_value=999999,
    )
    assert not r3["success"]

    # ④ 过期自动失效（强制过期后应拒绝）
    handle.expiry = time.time() - 1
    r4 = broker.execute_action(
        handle.handle_id, "create_po", 1, "n003", time.time(),
        target="supplier_001", amount_field="po_amount_usd", amount_value=5000,
    )
    assert not r4["success"] and "expired" in r4["reason"]

    audit = broker.get_audit_log()
    print("[✓] CapSeal WF-A 场景全部通过")
    for entry in audit:
        print(f"    [{entry['action']}] allowed={entry['allowed']} | {entry['reason']}")


if __name__ == "__main__":
    _test_capseal_wfa()
