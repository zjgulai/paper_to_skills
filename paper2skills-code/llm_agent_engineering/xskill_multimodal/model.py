"""
XSkill — 多模态 Agent 双流自进化框架
来源: arXiv:2603.12056 | 2026年3月
场景: 商品图片分析 Agent + 选品 Agent 的持续知识积累
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class TaskOutcome(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass
class Experience:
    """经验条目：战术层知识（具体轨迹反思）"""
    task_type: str
    context_summary: str
    action_taken: str
    outcome: TaskOutcome
    lesson: str
    visual_context_uid: Optional[str] = None   # 视觉上下文哈希 ID（多模态关联）
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    relevance_score: float = 0.5

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "context_summary": self.context_summary,
            "action_taken": self.action_taken,
            "outcome": self.outcome.value,
            "lesson": self.lesson,
            "visual_context_uid": self.visual_context_uid,
            "created_at": self.created_at,
            "relevance_score": self.relevance_score,
        }


@dataclass
class Skill:
    """技能条目：战略层知识（结构化可复用流程）"""
    skill_id: str
    task_type: str
    task_description: str
    procedure_steps: list
    success_rate: float = 0.5
    usage_count: int = 0
    version: int = 1                # 版本号，合并后递增
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "procedure_steps": self.procedure_steps,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            skill_id=d["skill_id"],
            task_type=d["task_type"],
            task_description=d["task_description"],
            procedure_steps=d["procedure_steps"],
            success_rate=d.get("success_rate", 0.5),
            usage_count=d.get("usage_count", 0),
            version=d.get("version", 1),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )

    def record_usage(self, succeeded: bool) -> None:
        """指数移动平均更新成功率：α=0.3"""
        self.usage_count += 1
        alpha = 0.3
        self.success_rate = round(
            (1 - alpha) * self.success_rate + alpha * (1.0 if succeeded else 0.0), 4
        )
        self.updated_at = datetime.now(timezone.utc).isoformat()


class _SimilarityCalculator:
    """基于词重叠的 Jaccard 相似度（无外部依赖）"""

    @staticmethod
    def jaccard(text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class ExperienceStream:
    """经验流：积累/检索/层次整合"""

    def __init__(self, max_size: int = 1000):
        self._experiences: list = []
        self._max_size = max_size
        self._sim = _SimilarityCalculator()

    def add(self, experience: Experience) -> None:
        """添加经验，超出上限时淘汰最旧的低价值条目"""
        self._experiences.append(experience)
        if len(self._experiences) > self._max_size:
            # 优先保留失败经验（防止重蹈覆辙），淘汰旧的成功经验
            self._experiences.sort(
                key=lambda e: (e.outcome == TaskOutcome.FAILURE, e.created_at)
            )
            self._experiences = self._experiences[-self._max_size:]

    def retrieve(self, query: str, task_type: Optional[str] = None, top_k: int = 3) -> list:
        """按 Jaccard 相似度检索最相关经验"""
        candidates = self._experiences
        if task_type:
            candidates = [e for e in candidates if e.task_type == task_type]

        scored = []
        for exp in candidates:
            sim = self._sim.jaccard(query, exp.context_summary + " " + exp.lesson)
            exp.relevance_score = sim
            scored.append((sim, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:top_k]]

    def hierarchical_integrate(self, task_type: str) -> str:
        """层次整合：将同类经验浓缩为宏观教训摘要"""
        relevant = [e for e in self._experiences if e.task_type == task_type]
        if not relevant:
            return ""
        success_lessons = [e.lesson for e in relevant if e.outcome == TaskOutcome.SUCCESS]
        failure_lessons = [e.lesson for e in relevant if e.outcome == TaskOutcome.FAILURE]
        return (
            f"[{task_type}] 共 {len(relevant)} 条经验。"
            f"成功模式: {'; '.join(success_lessons[-3:])}。"
            f"避免: {'; '.join(failure_lessons[-2:])}。"
        )

    def size(self) -> int:
        return len(self._experiences)


class SkillStream:
    """技能流：提炼/版本化/相似度去重"""

    MERGE_THRESHOLD = 0.6

    def __init__(self):
        self._skills: dict = {}
        self._sim = _SimilarityCalculator()

    def _generate_id(self, task_type: str, description: str) -> str:
        raw = f"{task_type}:{description}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    def add_or_merge(self, task_type: str, description: str, steps: list) -> str:
        """添加新技能，或与相似技能合并（语义去冗余）"""
        for skill_id, skill in self._skills.items():
            if skill.task_type != task_type:
                continue
            sim = self._sim.jaccard(description, skill.task_description)
            if sim >= self.MERGE_THRESHOLD:
                merged_steps = list(dict.fromkeys(skill.procedure_steps + steps))
                self._skills[skill_id] = Skill(
                    skill_id=skill_id,
                    task_type=task_type,
                    task_description=description,
                    procedure_steps=merged_steps,
                    success_rate=skill.success_rate,
                    usage_count=skill.usage_count,
                    version=skill.version + 1,
                    created_at=skill.created_at,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                return skill_id

        skill_id = self._generate_id(task_type, description)
        self._skills[skill_id] = Skill(
            skill_id=skill_id,
            task_type=task_type,
            task_description=description,
            procedure_steps=steps,
        )
        return skill_id

    def retrieve_best(self, task_type: str, query: str) -> Optional[Skill]:
        """检索同类中成功率最高的技能"""
        used = [s for s in self._skills.values() if s.task_type == task_type and s.usage_count > 0]
        if not used:
            fallback = [s for s in self._skills.values() if s.task_type == task_type]
            return fallback[0] if fallback else None
        used.sort(key=lambda s: (s.success_rate, s.usage_count), reverse=True)
        return used[0]

    def record_outcome(self, skill_id: str, succeeded: bool) -> None:
        if skill_id in self._skills:
            self._skills[skill_id].record_usage(succeeded)

    def all_skills(self) -> list:
        return list(self._skills.values())

    def size(self) -> int:
        return len(self._skills)


class XSkillAgent:
    """XSkill 双流 Agent：retrieve → adapt → record 协同循环"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.experience_stream = ExperienceStream()
        self.skill_stream = SkillStream()
        self._task_history: list = []

    def execute_task(
        self,
        task_type: str,
        task_description: str,
        context: str,
        visual_context_uid: Optional[str] = None,
    ) -> dict:
        """执行任务：检索双流知识 → 生成执行计划"""
        past_experiences = self.experience_stream.retrieve(
            query=context, task_type=task_type, top_k=3
        )
        best_skill = self.skill_stream.retrieve_best(task_type, task_description)

        plan_steps = list(best_skill.procedure_steps) if best_skill else [
            f"分析 {task_type} 任务", "执行核心决策", "输出结果"
        ]
        avoid_list = [e.lesson for e in past_experiences if e.outcome == TaskOutcome.FAILURE][:2]

        return {
            "task_type": task_type,
            "plan_steps": plan_steps,
            "avoid": avoid_list,
            "skill_used": best_skill.skill_id if best_skill else None,
            "experience_count": len(past_experiences),
        }

    def record_outcome(
        self,
        task_type: str,
        context: str,
        action_taken: str,
        outcome: TaskOutcome,
        lesson: str,
        skill_id: Optional[str],
        visual_context_uid: Optional[str] = None,
    ) -> None:
        """记录任务结果：更新双流知识"""
        exp = Experience(
            task_type=task_type,
            context_summary=context,
            action_taken=action_taken,
            outcome=outcome,
            lesson=lesson,
            visual_context_uid=visual_context_uid,
        )
        self.experience_stream.add(exp)

        if skill_id:
            self.skill_stream.record_outcome(skill_id, outcome == TaskOutcome.SUCCESS)

        self._task_history.append({
            "task_type": task_type,
            "outcome": outcome.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def distill_skill(self, task_type: str, description: str, steps: list) -> str:
        return self.skill_stream.add_or_merge(task_type, description, steps)

    def get_stats(self) -> dict:
        total = len(self._task_history)
        if total == 0:
            return {"total_tasks": 0, "success_rate": 0.0, "skills": 0, "experiences": 0}
        success = sum(1 for t in self._task_history if t["outcome"] == "success")
        return {
            "total_tasks": total,
            "success_rate": round(success / total, 3),
            "skills": self.skill_stream.size(),
            "experiences": self.experience_stream.size(),
        }


def test_xskill_self_improvement() -> None:
    """5次选品任务，验证技能积累和经验检索"""
    print("=" * 55)
    print("测试：XSkill 双流自进化 — 5次选品任务")
    print("=" * 55)

    agent = XSkillAgent("wfd_selection_agent")

    agent.distill_skill(
        task_type="product_selection",
        description="有机认证母婴产品选品",
        steps=["检查有机认证（EU/USDA）", "评估价格区间（40-90 USD）", "分析竞争密度（<20 SKU）", "计算预期 ROI"],
    )

    tasks = [
        {
            "desc": "HiPP 有机奶粉 Stage 2 选品评估",
            "context": "HiPP 有机认证 EU 级，价格 68 USD，竞品 12 个",
            "outcome": TaskOutcome.SUCCESS,
            "lesson": "EU 有机认证 + 价格 60-80 USD + 竞品 <15 是黄金组合",
        },
        {
            "desc": "无品牌国产奶粉低价选品",
            "context": "无认证，价格 22 USD，竞品 45 个",
            "outcome": TaskOutcome.FAILURE,
            "lesson": "无认证低价品类竞争过激，避免进入",
        },
        {
            "desc": "Aptamil 有机 Stage 1 选品",
            "context": "Aptamil EU 认证，价格 75 USD，竞品 10 个",
            "outcome": TaskOutcome.SUCCESS,
            "lesson": "知名品牌 + EU 认证溢价可支撑高价格",
        },
        {
            "desc": "有机辅食泥新品选品",
            "context": "Ella's Kitchen 有机，价格 24 USD，竞品 8 个",
            "outcome": TaskOutcome.SUCCESS,
            "lesson": "辅食类价格敏感度低，有机溢价接受度高",
        },
        {
            "desc": "Holle 有机 Stage 3 选品",
            "context": "Holle 生物动力农场认证，价格 82 USD，竞品 6 个",
            "outcome": TaskOutcome.SUCCESS,
            "lesson": "生物动力认证（高于 EU Organic）可支撑 80+ USD 高价",
        },
    ]

    for i, task in enumerate(tasks):
        print(f"\n[Task {i+1}] {task['desc']}")
        result = agent.execute_task(
            task_type="product_selection",
            task_description=task["desc"],
            context=task["context"],
        )
        print(f"  技能步骤: {result['plan_steps']}")
        print(f"  规避项: {result['avoid']}")

        agent.record_outcome(
            task_type="product_selection",
            context=task["context"],
            action_taken="评估并给出选品建议",
            outcome=task["outcome"],
            lesson=task["lesson"],
            skill_id=result["skill_used"],
        )

    new_skill_id = agent.distill_skill(
        task_type="product_selection",
        description="高端有机母婴产品选品优化流程",
        steps=[
            "验证有机认证等级（EU Organic / 生物动力 / USDA）",
            "价格区间过滤（40-90 USD，生物动力可放宽至 100）",
            "竞争密度检查（同 BSR 前 50 内 <20 SKU）",
            "品牌知名度评分（欧洲知名品牌额外 +0.2 分）",
            "计算 6 周预期 ROI（目标 >1.5）",
        ],
    )

    print("\n【Agent 统计】")
    stats = agent.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n【技能列表】")
    for skill in agent.skill_stream.all_skills():
        print(
            f"  [{skill.skill_id}] {skill.task_description[:40]:40s} "
            f"success={skill.success_rate:.2f} used={skill.usage_count}"
        )

    print("\n【经验检索测试】")
    relevant = agent.experience_stream.retrieve(
        query="有机认证 EU 价格 竞品少", task_type="product_selection", top_k=2
    )
    assert len(relevant) > 0, "❌ 经验检索为空"
    print(f"  检索到 {len(relevant)} 条相关经验")
    for exp in relevant:
        print(f"    - [{exp.outcome.value}] {exp.lesson}")

    print("\n【关键断言】")
    assert stats["total_tasks"] == 5, "❌ 任务数应为 5"
    assert stats["experiences"] == 5, "❌ 经验数应为 5"
    assert stats["skills"] >= 1, "❌ 技能数应≥1"
    assert stats["success_rate"] >= 0.6, f"❌ 成功率应≥0.6，实际={stats['success_rate']}"
    print(f"  ✅ 5 次任务完成，成功率={stats['success_rate']:.1%}")
    print(f"  ✅ 技能积累: {stats['skills']} 个技能")
    print(f"  ✅ 经验积累: {stats['experiences']} 条经验")

    all_skills = agent.skill_stream.all_skills()
    merged = [s for s in all_skills if s.version > 1]
    print(f"  ✅ 版本化合并: {len(merged)} 个技能已迭代升级")

    print("\n✅ 所有断言通过！")


if __name__ == "__main__":
    test_xskill_self_improvement()
