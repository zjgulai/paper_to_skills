"""
SDOF: State-Driven Orchestration Framework
状态机约束的多 Agent 合法性编排

论文来源: arXiv 2605.15204 | 2026年5月
应用场景: 母婴出海 WF-B 广告预算审批 / WF-A 补货订单 HITL 防护
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 工作流状态枚举（WF-B 广告预算审批场景）
# ---------------------------------------------------------------------------

class WorkflowState(str, Enum):
    """WF-B 广告预算审批的 FSM 状态"""
    ANALYZE = "ANALYZE"       # 分析阶段：读取广告数据，生成调整建议
    PROPOSE = "PROPOSE"       # 提案阶段：构造预算变更提案
    APPROVE = "APPROVE"       # 审批阶段：等待人工批复
    EXECUTE = "EXECUTE"       # 执行阶段：实施预算变更
    FAILED = "FAILED"         # 终态：任务失败
    COMPLETED = "COMPLETED"   # 终态：任务完成


# ---------------------------------------------------------------------------
# 2. FSM 迁移定义
# ---------------------------------------------------------------------------

@dataclass
class FSMTransition:
    """有限状态机迁移描述符

    Attributes:
        from_state: 源状态
        to_state: 目标状态
        trigger: 触发动作名称（对应 SkillRegistry 中的 skill_name）
        preconditions: 迁移前必须满足的条件 key 列表（在 context 中检查）
        postconditions: 迁移后自动验证的条件 key 列表
    """
    from_state: WorkflowState
    to_state: WorkflowState
    trigger: str
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"FSMTransition({self.from_state} --[{self.trigger}]--> {self.to_state})"


# ---------------------------------------------------------------------------
# 3. GoalStage FSM 管理器
# ---------------------------------------------------------------------------

class GoalStageManager:
    """GoalStage 状态机管理器

    核心功能:
    - 维护当前状态
    - check_legal_transition: 判断迁移是否合法
    - advance_state: 执行合法迁移，验证 preconditions
    """

    def __init__(self, transitions: list[FSMTransition], initial_state: WorkflowState) -> None:
        self.current_state = initial_state
        self._transitions = transitions
        # 构建迁移索引: (from_state, trigger) -> FSMTransition
        self._index: dict[tuple[WorkflowState, str], FSMTransition] = {
            (t.from_state, t.trigger): t for t in transitions
        }
        self._history: list[dict[str, Any]] = []

    def check_legal_transition(self, trigger: str) -> bool:
        """检查当前状态下 trigger 是否为合法迁移"""
        return (self.current_state, trigger) in self._index

    def advance_state(self, trigger: str, context: dict[str, Any]) -> WorkflowState:
        """执行状态迁移

        Args:
            trigger: 触发动作名称
            context: 运行时上下文，用于校验 preconditions

        Returns:
            迁移后的状态

        Raises:
            IllegalTransitionError: 迁移非法或 preconditions 不满足
        """
        key = (self.current_state, trigger)
        if key not in self._index:
            raise IllegalTransitionError(
                f"非法迁移: 当前状态={self.current_state}, trigger={trigger}"
            )

        transition = self._index[key]

        # 校验 preconditions
        for cond in transition.preconditions:
            if not context.get(cond):
                raise PreconditionError(
                    f"Precondition 未满足: {cond} (迁移: {transition})"
                )

        prev_state = self.current_state
        self.current_state = transition.to_state
        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "from": prev_state,
            "to": self.current_state,
            "trigger": trigger,
        })
        logger.info("状态迁移: %s --[%s]--> %s", prev_state, trigger, self.current_state)
        return self.current_state

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)


# ---------------------------------------------------------------------------
# 4. SkillRegistry：阶段绑定
# ---------------------------------------------------------------------------

class SkillRegistry:
    """技能注册表：每个 Skill 绑定到特定 WorkflowState

    只有当前状态允许的 Skill 才会被 Dispatcher 分发执行。
    """

    def __init__(self) -> None:
        self._registry: dict[WorkflowState, dict[str, Callable[..., Any]]] = {}

    def register_skill_for_stage(
        self,
        stage: WorkflowState,
        skill_name: str,
        skill_fn: Callable[..., Any],
    ) -> None:
        """将 skill_fn 注册到指定阶段"""
        if stage not in self._registry:
            self._registry[stage] = {}
        self._registry[stage][skill_name] = skill_fn
        logger.debug("注册 Skill: %s -> 阶段 %s", skill_name, stage)

    def get_allowed_skills(self, current_state: WorkflowState) -> dict[str, Callable[..., Any]]:
        """返回当前阶段允许使用的所有技能"""
        return dict(self._registry.get(current_state, {}))


# ---------------------------------------------------------------------------
# 5. StateAwareDispatcher：核心分发器
# ---------------------------------------------------------------------------

class StateAwareDispatcher:
    """状态感知分发器

    dispatch_with_constraint:
    1. 检查 trigger 是否为合法迁移
    2. 检查 skill_name 是否在当前阶段允许
    3. 执行状态迁移 + Skill 调用
    4. 屏蔽非法操作并写入审计日志
    """

    def __init__(
        self,
        fsm: GoalStageManager,
        registry: SkillRegistry,
    ) -> None:
        self._fsm = fsm
        self._registry = registry
        self._audit_log: list[dict[str, Any]] = []

    def dispatch_with_constraint(
        self,
        skill_name: str,
        context: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """带 FSM 约束的技能分发

        Args:
            skill_name: 要调用的技能名称（同时作为 FSM trigger）
            context: 运行时上下文（含 precondition 校验键值对）
            **kwargs: 传递给 Skill 的参数

        Returns:
            Skill 执行结果

        Raises:
            IllegalTransitionError: FSM 不允许当前迁移
            SkillNotAllowedError: 当前阶段无此 Skill
        """
        # ① FSM 合法性检查
        if not self._fsm.check_legal_transition(skill_name):
            entry = self._build_audit_entry(skill_name, "BLOCKED_ILLEGAL_TRANSITION", context)
            self._audit_log.append(entry)
            logger.warning("⛔ 非法操作被屏蔽: %s (当前状态: %s)", skill_name, self._fsm.current_state)
            raise IllegalTransitionError(
                f"FSM 拒绝: 当前状态 [{self._fsm.current_state}] 不允许触发 [{skill_name}]"
            )

        # ② 当前阶段 Skill 可用性检查
        allowed = self._registry.get_allowed_skills(self._fsm.current_state)
        if skill_name not in allowed:
            entry = self._build_audit_entry(skill_name, "BLOCKED_SKILL_NOT_ALLOWED", context)
            self._audit_log.append(entry)
            raise SkillNotAllowedError(
                f"Skill [{skill_name}] 在阶段 [{self._fsm.current_state}] 未注册"
            )

        # ③ 执行状态迁移
        self._fsm.advance_state(skill_name, context)

        # ④ 调用 Skill
        result = allowed[skill_name](**kwargs)
        entry = self._build_audit_entry(skill_name, "EXECUTED", context, result=str(result))
        self._audit_log.append(entry)
        return result

    def _build_audit_entry(
        self,
        skill_name: str,
        status: str,
        context: dict[str, Any],
        result: str | None = None,
    ) -> dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "skill": skill_name,
            "status": status,
            "state_before": self._fsm.current_state,
            "context_keys": list(context.keys()),
            "result": result,
        }

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit_log)


# ---------------------------------------------------------------------------
# 6. 自定义异常
# ---------------------------------------------------------------------------

class IllegalTransitionError(Exception):
    """FSM 非法迁移异常"""


class PreconditionError(Exception):
    """Precondition 未满足异常"""


class SkillNotAllowedError(Exception):
    """当前阶段不允许的 Skill 异常"""


# ---------------------------------------------------------------------------
# 7. 测试：WF-B 广告审批流程——验证跳过审批被屏蔽
# ---------------------------------------------------------------------------

def _make_wfb_fsm() -> GoalStageManager:
    """构建 WF-B 广告预算审批 FSM"""
    transitions = [
        FSMTransition(
            from_state=WorkflowState.ANALYZE,
            to_state=WorkflowState.PROPOSE,
            trigger="generate_budget_proposal",
        ),
        FSMTransition(
            from_state=WorkflowState.PROPOSE,
            to_state=WorkflowState.APPROVE,
            trigger="submit_for_approval",
        ),
        FSMTransition(
            from_state=WorkflowState.APPROVE,
            to_state=WorkflowState.EXECUTE,
            trigger="execute_budget_change",
            preconditions=["human_approval_received"],
        ),
    ]
    return GoalStageManager(transitions, initial_state=WorkflowState.ANALYZE)


def _make_wfb_registry(fsm: GoalStageManager) -> SkillRegistry:
    """注册 WF-B 各阶段技能"""
    registry = SkillRegistry()
    registry.register_skill_for_stage(
        WorkflowState.ANALYZE, "generate_budget_proposal",
        lambda: {"proposal": "增加 Q3 广告预算 20%", "amount": 50000}
    )
    registry.register_skill_for_stage(
        WorkflowState.PROPOSE, "submit_for_approval",
        lambda: {"submitted": True, "ticket_id": "AP-2026-001"}
    )
    registry.register_skill_for_stage(
        WorkflowState.APPROVE, "execute_budget_change",
        lambda: {"executed": True, "new_budget": 60000}
    )
    return registry


def run_wfb_test() -> None:
    """测试 WF-B：尝试从 PROPOSE 阶段直接跳到 EXECUTE（应被屏蔽）"""
    print("=" * 60)
    print("WF-B 广告预算审批 FSM 测试")
    print("=" * 60)

    fsm = _make_wfb_fsm()
    registry = _make_wfb_registry(fsm)
    dispatcher = StateAwareDispatcher(fsm, registry)
    context: dict[str, Any] = {}

    # Step 1: ANALYZE → PROPOSE（合法）
    print("\n[Step 1] generate_budget_proposal (ANALYZE → PROPOSE)")
    result = dispatcher.dispatch_with_constraint("generate_budget_proposal", context)
    print(f"  结果: {result}")
    print(f"  当前状态: {fsm.current_state}")

    # Step 2: 非法跳跃——PROPOSE → EXECUTE（应被屏蔽）
    print("\n[Step 2] 尝试非法跳跃: execute_budget_change (PROPOSE → EXECUTE)")
    try:
        dispatcher.dispatch_with_constraint(
            "execute_budget_change", context
        )
        print("  ❌ 错误：非法操作未被屏蔽！")
    except IllegalTransitionError as e:
        print(f"  ✅ 非法操作被屏蔽: {e}")

    # Step 3: 合法路径——PROPOSE → APPROVE
    print("\n[Step 3] submit_for_approval (PROPOSE → APPROVE)")
    result = dispatcher.dispatch_with_constraint("submit_for_approval", context)
    print(f"  结果: {result}")
    print(f"  当前状态: {fsm.current_state}")

    # Step 4: APPROVE → EXECUTE，但缺少 precondition
    print("\n[Step 4] execute_budget_change 但缺少 human_approval_received")
    try:
        dispatcher.dispatch_with_constraint("execute_budget_change", context)
        print("  ❌ 错误：Precondition 未被校验！")
    except PreconditionError as e:
        print(f"  ✅ Precondition 拦截: {e}")

    # Step 5: 设置 precondition 后正常执行
    print("\n[Step 5] 设置 human_approval_received=True，正式执行")
    context["human_approval_received"] = True
    result = dispatcher.dispatch_with_constraint("execute_budget_change", context)
    print(f"  结果: {result}")
    print(f"  当前状态: {fsm.current_state}")

    print(f"\n审计日志共 {len(dispatcher.audit_log)} 条:")
    for entry in dispatcher.audit_log:
        print(f"  [{entry['timestamp']}] {entry['skill']} -> {entry['status']}")

    print("\n✅ WF-B 测试通过：非法操作被完整屏蔽")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_wfb_test()
