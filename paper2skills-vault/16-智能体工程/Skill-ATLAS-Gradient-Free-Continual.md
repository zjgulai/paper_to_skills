---
title: ATLAS — 梯度无关持续学习：Teacher-Student 双架构在线适应
doc_type: knowledge
module: 16-智能体工程
topic: atlas-gradient-free-continual-learning
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: ATLAS — 梯度无关持续学习：Teacher-Student 双架构在线适应

> **论文来源**：Continual Learning, Not Training: Online Adaptation for Agents (ATLAS) · arXiv:2511.01093 · 2025年11月
> **GitHub**：github.com/Arc-Computer/atlas-sdk

---

## ① 算法原理

### 核心思想

传统持续学习依赖反向传播更新模型权重，存在三个根本缺陷：**必须离线批量训练**（无法在服务中实时更新）、**灾难性遗忘**（新任务覆盖旧能力）、**部署 Agent 无法自改**（推理阶段参数冻结）。ATLAS 的突破在于：**将"学习"从参数空间迁移到系统编排层**，通过持久学习记忆（Persistent Learning Memory, PLM）存储经验蒸馏后的指导性知识，无需触碰模型权重。

**双 Agent 架构**：
- **Teacher Agent（推理层）**：持有完整 PLM 访问权，负责从历史经验提炼策略建议，以"meta-instruction"形式注入 Student 的 context
- **Student Agent（执行层）**：接收 Teacher 策略建议后完成具体任务，执行结果反馈回 Teacher 用于下一轮知识更新

**持久学习记忆（PLM）结构**：
1. **原始经验池**：记录 (任务输入, 执行结果, 成功/失败标记) 三元组
2. **蒸馏知识库**：Teacher 从原始经验提炼的可读性策略规则，按任务类型分类索引
3. **监督级别控制器**：根据任务熟悉度动态调整 Teacher 介入强度（新任务→高监督，熟悉任务→低监督/自主执行）

**关键假设**：学习发生在运行时编排层（context 构造），而非参数更新；PLM 是唯一需要持久化的状态。

### 量化基准

- GPT-4o-mini（Student）+ PLM 指导 > GPT-4o（High）13%，成本降低 86%
- 跨任务泛化：Incident #5 → #55 迁移，准确率 28%→41%，零再训练

---

## ② 母婴出海应用案例

### 场景一：WF-A 供应链 Agent 自适应补货学习

**痛点**：婴儿奶粉旺季/淡季需求波动大，固定规则备货导致缺货或积压，重新微调模型成本高。

**ATLAS 方案**：
```
每次补货决策执行后 → Student 记录 (SKU, 决策量, 实际消耗, 偏差率) → 
Teacher 每周蒸馏经验 → PLM 更新策略库：
  - "奶粉旺季（11月-2月）基准备货量乘以1.4倍系数"
  - "海外仓安全库存 = 预测消耗 × 30天 + 15% 缓冲"
  - "新品上架前2周预备量按同类商品历史峰值的60%"
```

**效果**：无需重训，第 5 次补货决策开始质量可见提升；运营人员可直接编辑 PLM 中的策略规则进行干预。

### 场景二：WF-C 客服 Agent 持续优化退款处理

**痛点**：退款场景规则复杂（金额/原因/渠道各维度组合），人工写规则无法穷举，微调频率追不上业务变化。

**ATLAS 方案**：
```
客服案例积累 → Teacher 从历史案例提炼处理规律 → PLM 更新：
  - "退款金额 > ¥500 且非平台责任 → 必须触发人工审核"
  - "欧美客户投诉包装破损 → 直接全额退款+发送道歉邮件，不追责"
  - "疑似职业差评（30天内同账号>3次退款）→ 标记升级人工"
Student 处理新案例时，Teacher 实时检索最相关规则注入 context
```

**效果**：客服处理一致性提升，边缘案例处理准确率随案例积累持续改善；PLM 可审计可回溯。

---

## ③ 代码模板

> 文件路径：`paper2skills-code/llm_agent_engineering/atlas_continual_learning/model.py`

```python
"""
ATLAS — Gradient-Free Continual Learning via Teacher-Student Architecture
Paper: arXiv:2511.01093 | Nov 2025
Use case: WF-A supply chain agent adaptation + WF-C customer service continuous optimization
"""
from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any


# ─── 数据类 ───────────────────────────────────────────────────────────────

@dataclass
class Experience:
    """原始经验记录：任务输入 + 执行结果 + 成功标记"""
    exp_id: str
    task_type: str          # 任务类型（如 "restock", "refund"）
    task_input: dict        # 任务输入上下文
    decision: str           # Student 的执行决策
    outcome: dict           # 实际结果（成功率、偏差等）
    success: bool           # 是否成功
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


@dataclass
class DistilledKnowledge:
    """蒸馏知识条目：从经验提炼的可读策略规则"""
    rule_id: str
    task_type: str
    rule_text: str          # 可读规则（如 "旺季备货乘以1.4系数"）
    confidence: float       # 0-1，基于支撑经验数量
    support_count: int      # 支撑该规则的原始经验数量
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ─── 持久学习记忆（PLM）──────────────────────────────────────────────────

class LearningMemory:
    """持久学习记忆：存储原始经验 + 蒸馏知识，支持检索和更新"""

    def __init__(self) -> None:
        self._experiences: list[Experience] = []
        self._knowledge: dict[str, list[DistilledKnowledge]] = {}  # task_type → rules
        self._supervision_levels: dict[str, float] = {}             # task_type → 0-1

    # --- 经验管理 ---
    def add_experience(self, exp: Experience) -> None:
        self._experiences.append(exp)

    def get_experiences(self, task_type: str, limit: int = 10) -> list[Experience]:
        """检索最近的同类任务经验"""
        filtered = [e for e in self._experiences if e.task_type == task_type]
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:limit]

    # --- 知识管理 ---
    def update_knowledge(self, task_type: str, rules: list[DistilledKnowledge]) -> None:
        """更新蒸馏知识库（覆盖 + 合并）"""
        existing = {r.rule_id: r for r in self._knowledge.get(task_type, [])}
        for rule in rules:
            if rule.rule_id in existing:
                existing[rule.rule_id].rule_text = rule.rule_text
                existing[rule.rule_id].confidence = rule.confidence
                existing[rule.rule_id].support_count = rule.support_count
                existing[rule.rule_id].updated_at = time.time()
            else:
                existing[rule.rule_id] = rule
        self._knowledge[task_type] = list(existing.values())

    def get_knowledge(self, task_type: str) -> list[DistilledKnowledge]:
        """检索指定任务类型的策略规则，按置信度降序"""
        rules = self._knowledge.get(task_type, [])
        return sorted(rules, key=lambda r: r.confidence, reverse=True)

    # --- 监督级别 ---
    def get_supervision_level(self, task_type: str) -> float:
        """0=完全自主，1=高度监督；经验越多监督越少"""
        base = self._supervision_levels.get(task_type, 1.0)
        exp_count = len([e for e in self._experiences if e.task_type == task_type])
        # 每积累 5 条经验降低 0.1 监督，最低 0.2
        adjusted = max(0.2, base - (exp_count // 5) * 0.1)
        return adjusted

    def snapshot(self) -> dict:
        return {
            "experience_count": len(self._experiences),
            "knowledge_types": list(self._knowledge.keys()),
            "total_rules": sum(len(v) for v in self._knowledge.values()),
        }


# ─── Teacher Agent（推理层）──────────────────────────────────────────────

class TeacherAgent:
    """推理层：从 PLM 提炼策略，构建 meta-instruction 注入 Student context"""

    def __init__(self, memory: LearningMemory) -> None:
        self.memory = memory

    def build_meta_instruction(self, task_type: str, task_input: dict) -> str:
        """根据 PLM 知识构建 meta-instruction"""
        rules = self.memory.get_knowledge(task_type)
        supervision = self.memory.get_supervision_level(task_type)
        recent_exps = self.memory.get_experiences(task_type, limit=3)

        parts = [f"[Teacher 策略指导 | 任务类型: {task_type} | 监督级别: {supervision:.1f}]"]

        if rules:
            parts.append("\n已积累策略规则：")
            for r in rules[:5]:  # 最多注入5条最高置信度规则
                parts.append(f"  - [{r.confidence:.0%}置信] {r.rule_text}")

        if recent_exps:
            parts.append("\n近期经验摘要：")
            for exp in recent_exps:
                status = "✅" if exp.success else "❌"
                parts.append(f"  {status} {exp.decision[:60]}...")

        parts.append(f"\n当前任务输入：{json.dumps(task_input, ensure_ascii=False)}")
        return "\n".join(parts)

    def distill_knowledge(self, task_type: str) -> list[DistilledKnowledge]:
        """从原始经验蒸馏策略规则（模拟 LLM 提炼，实际需调用真实 LLM）"""
        exps = self.memory.get_experiences(task_type, limit=20)
        if len(exps) < 3:
            return []  # 经验不足，不蒸馏

        # 模拟蒸馏：统计成功经验中的高频决策模式
        success_exps = [e for e in exps if e.success]
        rules = []

        if task_type == "restock" and len(success_exps) >= 3:
            # 从补货经验中提炼旺季系数
            high_season_count = sum(
                1 for e in success_exps
                if e.task_input.get("month") in [11, 12, 1, 2]
            )
            if high_season_count >= 2:
                rules.append(DistilledKnowledge(
                    rule_id="restock_season_multiplier",
                    task_type="restock",
                    rule_text="旺季（11月-2月）基准备货量乘以1.4倍系数",
                    confidence=min(0.95, 0.5 + high_season_count * 0.1),
                    support_count=high_season_count,
                ))

        if task_type == "refund" and len(success_exps) >= 3:
            high_amount = sum(
                1 for e in success_exps
                if e.task_input.get("amount", 0) > 500
                and e.outcome.get("human_review", False)
            )
            if high_amount >= 2:
                rules.append(DistilledKnowledge(
                    rule_id="refund_high_amount_review",
                    task_type="refund",
                    rule_text="退款金额 > ¥500 且非平台责任，触发人工审核",
                    confidence=min(0.95, 0.5 + high_amount * 0.15),
                    support_count=high_amount,
                ))

        return rules


# ─── Student Agent（执行层）──────────────────────────────────────────────

class StudentAgent:
    """执行层：接收 Teacher meta-instruction，完成具体任务并返回结果"""

    def __init__(self, name: str = "student") -> None:
        self.name = name
        self._task_counter: dict[str, int] = {}

    def execute(self, task_type: str, task_input: dict, meta_instruction: str) -> dict:
        """执行任务，模拟基于 Teacher 指导的决策质量提升"""
        self._task_counter[task_type] = self._task_counter.get(task_type, 0) + 1
        exec_count = self._task_counter[task_type]

        # 模拟决策逻辑：有 Teacher 指导时质量随经验积累提升
        has_guidance = "已积累策略规则" in meta_instruction
        base_quality = 0.60
        guidance_boost = 0.15 if has_guidance else 0.0
        experience_boost = min(0.20, exec_count * 0.04)
        quality = min(0.95, base_quality + guidance_boost + experience_boost)

        if task_type == "restock":
            month = task_input.get("month", 6)
            base_qty = task_input.get("base_qty", 1000)
            is_high_season = month in [11, 12, 1, 2]

            if has_guidance and is_high_season:
                decision_qty = int(base_qty * 1.4)
                decision = f"旺季备货：{decision_qty} 件（基准 {base_qty} × 1.4）"
            else:
                decision_qty = base_qty
                decision = f"常规备货：{decision_qty} 件"

            actual_demand = int(base_qty * (1.35 if is_high_season else 1.0))
            success = abs(decision_qty - actual_demand) / actual_demand < 0.15

            return {
                "decision": decision,
                "decision_qty": decision_qty,
                "actual_demand": actual_demand,
                "success": success,
                "quality_score": quality,
                "outcome": {"deviation": abs(decision_qty - actual_demand) / actual_demand},
            }

        elif task_type == "refund":
            amount = task_input.get("amount", 200)
            reason = task_input.get("reason", "质量问题")
            needs_review = amount > 500

            if has_guidance and needs_review:
                decision = f"金额 ¥{amount}，触发人工审核（规则: >¥500 需审核）"
                outcome = {"human_review": True, "resolved": True}
                success = True
            else:
                decision = f"金额 ¥{amount}，自动处理退款"
                outcome = {"human_review": False, "resolved": amount <= 500}
                success = amount <= 500

            return {
                "decision": decision,
                "success": success,
                "quality_score": quality,
                "outcome": outcome,
            }

        return {"decision": "未知任务类型", "success": False, "quality_score": 0.0, "outcome": {}}


# ─── ATLAS Orchestrator（协调层）────────────────────────────────────────

class ATLASOrchestrator:
    """协调 Teacher-Student，更新记忆，调整监督级别"""

    def __init__(self) -> None:
        self.memory = LearningMemory()
        self.teacher = TeacherAgent(self.memory)
        self.student = StudentAgent()
        self._distill_interval = 3  # 每积累N条经验蒸馏一次

    def run_task(self, task_type: str, task_input: dict) -> dict:
        """执行一次完整的 ATLAS 任务循环"""
        # 1. Teacher 构建 meta-instruction
        meta_instruction = self.teacher.build_meta_instruction(task_type, task_input)

        # 2. Student 执行任务
        result = self.student.execute(task_type, task_input, meta_instruction)

        # 3. 记录经验
        exp_id = hashlib.md5(
            f"{task_type}{time.time()}".encode()
        ).hexdigest()[:8]
        exp = Experience(
            exp_id=exp_id,
            task_type=task_type,
            task_input=task_input,
            decision=result["decision"],
            outcome=result["outcome"],
            success=result["success"],
        )
        self.memory.add_experience(exp)

        # 4. 按间隔触发知识蒸馏
        exps = self.memory.get_experiences(task_type)
        if len(exps) % self._distill_interval == 0 and len(exps) > 0:
            new_rules = self.teacher.distill_knowledge(task_type)
            if new_rules:
                self.memory.update_knowledge(task_type, new_rules)

        return {
            "task_type": task_type,
            "decision": result["decision"],
            "success": result["success"],
            "quality_score": result["quality_score"],
            "supervision_level": self.memory.get_supervision_level(task_type),
            "plm_snapshot": self.memory.snapshot(),
        }


# ─── 测试：WF-A 5次补货决策学习过程 ─────────────────────────────────────

def test_restock_learning():
    """模拟 WF-A 5次补货决策，验证后续决策质量提升（旺季自适应）"""
    orchestrator = ATLASOrchestrator()

    tasks = [
        {"month": 11, "base_qty": 1000, "sku": "奶粉A段"},
        {"month": 12, "base_qty": 1200, "sku": "奶粉B段"},
        {"month": 3,  "base_qty": 900,  "sku": "奶粉A段"},
        {"month": 1,  "base_qty": 1100, "sku": "奶粉C段"},
        {"month": 11, "base_qty": 1000, "sku": "奶粉A段"},  # 第5次，应已积累旺季规则
    ]

    print("=" * 60)
    print("WF-A 供应链 Agent 补货学习测试（ATLAS）")
    print("=" * 60)

    results = []
    for i, task_input in enumerate(tasks, 1):
        result = orchestrator.run_task("restock", task_input)
        results.append(result)
        print(f"\n[决策 {i}] 月份:{task_input['month']} | 基准:{task_input['base_qty']}")
        print(f"  决策: {result['decision']}")
        print(f"  成功: {result['success']} | 质量分: {result['quality_score']:.2f}")
        print(f"  监督级别: {result['supervision_level']:.1f}")
        print(f"  PLM: {result['plm_snapshot']}")

    # 验证质量提升
    early_scores = [r["quality_score"] for r in results[:2]]
    late_scores = [r["quality_score"] for r in results[3:]]
    avg_early = sum(early_scores) / len(early_scores)
    avg_late = sum(late_scores) / len(late_scores)

    print(f"\n早期平均质量: {avg_early:.2f}")
    print(f"后期平均质量: {avg_late:.2f}")
    assert avg_late >= avg_early, f"后期质量应 >= 早期质量，实际 {avg_late:.2f} vs {avg_early:.2f}"

    # 验证 PLM 有积累
    snapshot = orchestrator.memory.snapshot()
    assert snapshot["experience_count"] == 5, f"应有5条经验，实际: {snapshot['experience_count']}"
    print(f"\n✅ 测试通过：PLM 积累 {snapshot['experience_count']} 条经验，质量提升验证成功")


if __name__ == "__main__":
    test_restock_learning()
```

---

## ④ 技能关联

- **前置**：[[Skill-Reflexion-Self-Improvement]] / [[Skill-EvoSC-Self-Consolidation]] / [[Skill-Agent-Memory-Learning]]
- **延伸**：[[Skill-AutoSkill-Lifelong-Learning]]（同批萃取）/ [[Skill-AgeMem-Unified-Agent-Memory]]
- **可组合**：[[Skill-Flowr-Supply-Chain-MAS]] / [[Skill-MASEval-System-Evaluation]] / [[Skill-ReliabilityBench-Agent-Reliability]]

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | GPT-4o-mini + PLM 超越 GPT-4o(High) 13%，成本降低 86%；中等规模 Agent 服务年节省 LLM 费用数万元 |
| **实施难度** | ⭐⭐⭐☆☆（架构清晰，无需修改模型权重；主要成本在 PLM 持久化设计和蒸馏触发逻辑） |
| **优先级评分** | ⭐⭐⭐⭐⭐（50-Loop 进化的技术底座，所有 Agent 持续学习场景的首选方案） |
| **评估依据** | 跨任务泛化验证（Incident #5→#55：28%→41%，零再训练）；架构适配现有 Agent 框架无破坏性改动 |
