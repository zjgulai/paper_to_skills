---
title: SDOF — 状态机约束 MAS 编排：屏蔽非法操作，任务完成率 86.5%
doc_type: knowledge
module: 10-MAS
topic: sdof-state-constrained-orchestration
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase3
---

# SDOF — 状态机约束 MAS 编排：屏蔽非法操作，任务完成率 86.5%

**来源**：SDOF: State-Driven Orchestration Framework for Multi-Agent Systems | arXiv: 2605.15204 | 2026年5月
**验证**：生产环境（招聘 SaaS，6000+ 企业）

## ① 算法原理

SDOF 将 Multi-Agent System（MAS）的执行流程建模为**有限状态机（FSM）**，通过双层防护机制确保 Agent 行为的合法性。

**GoalStage FSM 约束**：工作流被分解为若干 GoalStage（目标阶段），每个阶段由 FSM 管理状态迁移。每次 Agent 发出动作请求时，`GoalStageManager` 先检查当前状态是否允许该迁移（`check_legal_transition`），非法跳转直接屏蔽并写入审计日志，绝不执行。

**precondition / postcondition 双层防护**：`FSMTransition` 包含 `preconditions`（前置条件，触发前必须满足）和 `postconditions`（后置条件，执行后自动验证），形成执行闭环。

**SkillRegistry 阶段绑定**：每个技能（Skill）在注册时绑定到特定阶段（stage），`get_allowed_skills(current_state)` 仅返回当前阶段合法的技能列表，Agent 无法调用当前阶段以外的能力。

**Online-RLHF 意图路由器（7B 模型）**：通过 Online RLHF 持续训练轻量级意图分类器，将用户/系统意图准确路由到目标 GoalStage。在招聘 SaaS 基准上达到 80.9%（vs GPT-4o 48.9%），以极低延迟（+57ms）实现精准路由。

**核心量化**：任务完成率 86.5%，屏蔽全部 22 个非法操作（precision 100%），recall 88%。

## ② 母婴出海应用案例

### 场景一：WF-B 广告预算审批 FSM

**问题**：多 Agent 广告优化链路中，执行 Agent 可能直接触发预算变更，跳过品牌负责人审批，造成未授权的大额消耗。

**SDOF 方案**：

```
状态机：分析(ANALYZE) → 提案(PROPOSE) → 审批(APPROVE) → 执行(EXECUTE)

合法迁移：
  ANALYZE → PROPOSE（分析完成，生成预算调整提案）
  PROPOSE → APPROVE（提案提交，等待人工审批）
  APPROVE → EXECUTE（审批通过，执行预算变更）

非法迁移（被屏蔽）：
  PROPOSE → EXECUTE（跳过审批直接执行）
  ANALYZE → EXECUTE（完全绕过提案+审批）
```

**效果**：广告执行 Agent 若尝试在 PROPOSE 阶段直接调用 `execute_budget_change`，`StateAwareDispatcher` 立即拦截，记录审计日志，返回 `IllegalTransitionError`，预算变更风险归零。

### 场景二：WF-A 补货订单 FSM（大额 PO 人工确认）

**问题**：供应链补货 Agent 在需求预测后可能自动下大额采购订单（PO），缺少人工确认节点导致库存积压风险。

**SDOF 方案**：

```
状态机：预测(FORECAST) → 评估(EVALUATE) → 人工审核(HITL) → 下单(ORDER)

合法迁移：
  FORECAST → EVALUATE（生成补货需求预测）
  EVALUATE → HITL（大额 PO 触发 HITL 节点）
  HITL → ORDER（人工确认后方可下单）

非法迁移（被屏蔽）：
  EVALUATE → ORDER（跳过 HITL 直接下单）

precondition（HITL → ORDER）：
  - human_approval_received = True
  - approval_timestamp < 24h
```

**效果**：单笔 ≥ 5000 USD 的 PO 必须经过 HITL 节点，系统强制等待人工批复信号，消除自动下单的资金风险。

## ③ 代码模板

代码位置：`paper2skills-code/mas/sdof_state_orchestration/model.py`

```python
# 见 paper2skills-code/mas/sdof_state_orchestration/model.py
```

## ④ 技能关联

**前置技能**（需先掌握）：
- [[Skill-MAS-Orchestrator]]：MAS 基础编排模式
- [[Skill-Skill-Skill-Registry-Dynamic-Loading]]：技能注册与动态加载
- （假设）[[Skill-Agent-Safety-Guardrails]]：Agent 安全护栏基础

**延伸技能**（同批萃取，可直接组合）：
- [[Skill-ParaManager-Parallel-Orchestration]]：并行子任务分解（同批萃取）
- [[Skill-Flowr-Supply-Chain-MAS]]：供应链 MAS 落地实践

**可组合技能**：
- [[Skill-Agent-Fault-Tolerance]]：状态机与容错机制联合使用
- [[Skill-AgentTrace-Causal-RCA]]：审计日志 → 因果根因分析

## ⑤ 商业价值

| 指标 | 数值 |
|------|------|
| 非法操作屏蔽率 | 100%（22/22）|
| 任务完成率 | 86.5% |
| 意图路由准确率 | 80.9%（vs GPT-4o 48.9%）|
| 线上延迟增量 | +57ms |
| 核心业务价值 | 非法预算变更风险归零；大额 PO 必须人工确认 |

**实施难度**：⭐⭐⭐☆☆（需设计 FSM 状态图 + precondition 逻辑）
**优先级**：⭐⭐⭐⭐☆（高合规要求场景必备）
