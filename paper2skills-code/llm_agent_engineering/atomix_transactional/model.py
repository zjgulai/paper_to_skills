"""
Atomix: 为 Agent 工具调用引入事务语义（epoch + frontier + 补偿）
参考: arXiv:2602.14849 — Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows
GitHub: https://github.com/mpi-dsg/atomix
"""
import uuid
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass, field


# ────────────────────────────────────────
# 1. ToolEffect 枚举
# ────────────────────────────────────────

class ToolEffect(Enum):
    BUFFERABLE = "bufferable"      # 纯内部效果，abort 时直接丢弃
    EXTERNALIZED = "externalized"  # 已产生外部副作用，abort 时需补偿


# ────────────────────────────────────────
# 2. Effect 数据类
# ────────────────────────────────────────

@dataclass
class Effect:
    tool_name: str
    effect_type: ToolEffect
    result: Any
    compensation_fn: Optional[Callable] = None
    epoch: int = 0


# ────────────────────────────────────────
# 3. Transaction
# ────────────────────────────────────────

@dataclass
class Transaction:
    tx_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    epoch: int = 0
    effects: list[Effect] = field(default_factory=list)
    status: str = "active"  # active | committed | aborted


# ────────────────────────────────────────
# 4. FrontierTracker
# ────────────────────────────────────────

class FrontierTracker:
    def __init__(self):
        self._frontiers: dict[str, int] = {}  # resource_id → max confirmed epoch

    def track(self, resource_id: str, epoch: int) -> None:
        current = self._frontiers.get(resource_id, -1)
        if epoch > current:
            self._frontiers[resource_id] = epoch

    def can_commit(self, tx: Transaction, required_resources: list[str]) -> bool:
        for res in required_resources:
            if self._frontiers.get(res, -1) < tx.epoch:
                return False
        return True


# ────────────────────────────────────────
# 5. CompensationRegistry
# ────────────────────────────────────────

class CompensationRegistry:
    def __init__(self):
        self._registry: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._registry[name] = fn

    def compensate(self, name: str, *args, **kwargs) -> Any:
        fn = self._registry.get(name)
        if fn:
            return fn(*args, **kwargs)
        raise KeyError(f"补偿函数未注册: {name}")


# ────────────────────────────────────────
# 6. AtomixRuntime
# ────────────────────────────────────────

class AtomixRuntime:
    def __init__(self):
        self.frontier = FrontierTracker()
        self.compensation = CompensationRegistry()
        self._epoch_counter = 0
        self._log: list[str] = []

    def begin_transaction(self) -> Transaction:
        self._epoch_counter += 1
        tx = Transaction(epoch=self._epoch_counter)
        self._log.append(f"[BEGIN] tx={tx.tx_id} epoch={tx.epoch}")
        return tx

    def execute_tool(
        self,
        tx: Transaction,
        tool_fn: Callable,
        effect_type: ToolEffect,
        resource_id: str,
        compensation_fn: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> Any:
        if tx.status != "active":
            raise RuntimeError(f"事务 {tx.tx_id} 已 {tx.status}，无法执行工具")

        result = tool_fn(*args, **kwargs)
        effect = Effect(
            tool_name=tool_fn.__name__,
            effect_type=effect_type,
            result=result,
            compensation_fn=compensation_fn,
            epoch=tx.epoch,
        )
        tx.effects.append(effect)
        self.frontier.track(resource_id, tx.epoch)
        self._log.append(
            f"[EXEC] tx={tx.tx_id} tool={tool_fn.__name__} effect={effect_type.value} resource={resource_id}"
        )
        return result

    def commit(self, tx: Transaction, required_resources: list[str]) -> bool:
        if not self.frontier.can_commit(tx, required_resources):
            self._log.append(f"[ABORT] tx={tx.tx_id} reason=frontier_not_satisfied")
            self._abort_with_compensation(tx)
            return False

        tx.status = "committed"
        self._log.append(f"[COMMIT] tx={tx.tx_id}")
        return True

    def abort(self, tx: Transaction, reason: str = "") -> None:
        self._log.append(f"[ABORT] tx={tx.tx_id} reason={reason}")
        self._abort_with_compensation(tx)

    def _abort_with_compensation(self, tx: Transaction) -> None:
        tx.status = "aborted"
        for effect in reversed(tx.effects):
            if effect.effect_type == ToolEffect.EXTERNALIZED and effect.compensation_fn:
                try:
                    effect.compensation_fn()
                    self._log.append(f"[COMPENSATE] tool={effect.tool_name} ✓")
                except Exception as e:
                    self._log.append(f"[COMPENSATE] tool={effect.tool_name} FAILED: {e}")

    def print_log(self) -> None:
        print("\n".join(self._log))


# ────────────────────────────────────────
# 7. 测试：WF-A 三步补货流程，步骤2失败后补偿回滚
# ────────────────────────────────────────

def test_wf_a_po_transaction():
    print("=" * 60)
    print("Atomix 事务性测试：WF-A 补货 PO 下单（步骤2失败回滚）")
    print("=" * 60)

    runtime = AtomixRuntime()
    compensated_steps: list[str] = []
    po_created: list[str] = []

    def forecast_demand() -> dict:
        print("  [步骤1] 需求预测 API 调用...成功")
        return {"sku": "SKU-001", "forecast_qty": 500}

    def calculate_safety_stock(**kwargs) -> dict:
        print("  [步骤2] 安全库存计算 API 调用...模拟失败！")
        raise ValueError("库存 API 超时")

    def create_po(qty: int = 500) -> dict:
        po_id = f"PO-{uuid.uuid4().hex[:6].upper()}"
        po_created.append(po_id)
        print(f"  [步骤3] ERP 创建 PO: {po_id}")
        return {"po_id": po_id, "qty": qty}

    def cancel_po_fn() -> None:
        if po_created:
            compensated_steps.append(f"cancel_po:{po_created[-1]}")
            print(f"  [补偿] 取消 PO: {po_created[-1]}")

    tx = runtime.begin_transaction()
    resources = ["demand_forecast", "safety_stock", "erp_po"]

    try:
        forecast = runtime.execute_tool(
            tx, forecast_demand, ToolEffect.BUFFERABLE, "demand_forecast"
        )
        _safety = runtime.execute_tool(
            tx, calculate_safety_stock, ToolEffect.BUFFERABLE, "safety_stock",
            compensation_fn=None
        )
        _po = runtime.execute_tool(
            tx, create_po, ToolEffect.EXTERNALIZED, "erp_po",
            compensation_fn=cancel_po_fn
        )
    except Exception as e:
        print(f"\n  异常捕获: {e}")
        runtime.abort(tx, reason=str(e))

    print(f"\n  事务状态: {tx.status}")
    print(f"  效果链（{len(tx.effects)} 步）:")
    for eff in tx.effects:
        print(f"    - {eff.tool_name}: {eff.effect_type.value}")

    print(f"  已执行补偿: {compensated_steps}")
    print("\n  事务执行日志:")
    runtime.print_log()

    assert tx.status == "aborted", "步骤2失败后事务应为 aborted"
    assert len(po_created) == 0, "步骤2失败时步骤3未执行，无 PO 需补偿"

    print("\n✅ WF-A 测试通过！")
    return tx


def test_wf_b_budget_atomic():
    print("\n" + "=" * 60)
    print("Atomix 事务性测试：WF-B 跨平台预算调整（全部成功 commit）")
    print("=" * 60)

    runtime = AtomixRuntime()
    rollback_calls: list[str] = []

    original_budgets = {"google": 1000, "meta": 800, "tiktok": 600}
    current_budgets = dict(original_budgets)

    def set_google_budget() -> dict:
        current_budgets["google"] = 1200
        print("  [Google] 预算调整 1000 → 1200 ✓")
        return {"platform": "google", "new_budget": 1200}

    def set_meta_budget() -> dict:
        current_budgets["meta"] = 900
        print("  [Meta] 预算调整 800 → 900 ✓")
        return {"platform": "meta", "new_budget": 900}

    def set_tiktok_budget() -> dict:
        current_budgets["tiktok"] = 700
        print("  [TikTok] 预算调整 600 → 700 ✓")
        return {"platform": "tiktok", "new_budget": 700}

    def revert_google() -> None:
        rollback_calls.append("revert_google")
        current_budgets["google"] = original_budgets["google"]
        print("  [补偿] Google 预算恢复 1200 → 1000")

    def revert_meta() -> None:
        rollback_calls.append("revert_meta")
        current_budgets["meta"] = original_budgets["meta"]
        print("  [补偿] Meta 预算恢复 900 → 800")

    def revert_tiktok() -> None:
        rollback_calls.append("revert_tiktok")
        current_budgets["tiktok"] = original_budgets["tiktok"]
        print("  [补偿] TikTok 预算恢复 700 → 600")

    tx = runtime.begin_transaction()
    resources = ["google_ads", "meta_ads", "tiktok_ads"]

    try:
        runtime.execute_tool(tx, set_google_budget, ToolEffect.EXTERNALIZED, "google_ads", revert_google)
        runtime.execute_tool(tx, set_meta_budget, ToolEffect.EXTERNALIZED, "meta_ads", revert_meta)
        runtime.execute_tool(tx, set_tiktok_budget, ToolEffect.EXTERNALIZED, "tiktok_ads", revert_tiktok)
        committed = runtime.commit(tx, resources)
        print(f"\n  commit 结果: {'成功' if committed else '失败'}")
    except Exception as e:
        runtime.abort(tx, reason=str(e))

    print(f"  事务状态: {tx.status}")
    print(f"  最终预算: {current_budgets}")
    print(f"  已触发补偿: {rollback_calls}")
    print("\n  事务执行日志:")
    runtime.print_log()

    assert tx.status == "committed", "三平台均成功，事务应 committed"
    assert len(rollback_calls) == 0, "成功提交时不应触发补偿"
    assert current_budgets == {"google": 1200, "meta": 900, "tiktok": 700}

    print("\n✅ WF-B 测试通过！")
    return tx


if __name__ == "__main__":
    test_wf_a_po_transaction()
    test_wf_b_budget_atomic()
