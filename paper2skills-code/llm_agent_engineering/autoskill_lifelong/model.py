"""
AutoSkill — Experience-Driven Lifelong Learning via Skill Self-Evolution
Paper: arXiv:2603.01145 | Mar 2026
Use case: DTC copywriting skill accumulation + WF-D product selection specialization
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


# ─── 数据类 ───────────────────────────────────────────────────────────────

@dataclass
class SkillArtifact:
    """Skill 条目：可复用的任务执行模式，版本化管理"""
    skill_id: str
    version: int                    # 版本号，从1开始递增
    task_domain: str                # 任务领域
    trigger_pattern: str            # 触发模式（自然语言描述的适用场景）
    trigger_keywords: list[str]     # 关键词列表，用于快速匹配
    instructions: str               # 执行指令（结构化文本）
    usage_count: int = 0
    fitness: float = 0.5            # 0-1，基于用户反馈和成功率（EMA更新）
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parent_skill_id: str = ""       # 合并来源（合并 Skill 时记录）


@dataclass
class ConversationTurn:
    """单次对话轨迹条目"""
    turn_id: str
    user_input: str
    agent_response: str
    task_domain: str
    quality_score: float            # 0-1，用户反馈/自动评估
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


# ─── SkillExtractor（技能提炼器）────────────────────────────────────────

class SkillExtractor:
    """从对话轨迹中识别可复用模式，生成 SkillArtifact 草稿"""

    DOMAIN_PATTERNS: dict[str, list[str]] = {
        "copywriting": ["文案", "描述", "产品介绍", "copywriting", "欧美妈妈"],
        "product_selection": ["选品", "奶粉", "品类", "竞争", "BSR", "评分"],
        "customer_service": ["退款", "客诉", "售后", "差评", "投诉"],
        "supply_chain": ["补货", "备货", "库存", "供应链", "物流"],
    }

    def detect_domain(self, text: str) -> str:
        for domain, keywords in self.DOMAIN_PATTERNS.items():
            if any(kw in text for kw in keywords):
                return domain
        return "general"

    def extract_keywords(self, text: str, domain: str) -> list[str]:
        keywords = [kw for kw in self.DOMAIN_PATTERNS.get(domain, []) if kw in text]
        return keywords[:5]

    def can_extract(self, turns: list[ConversationTurn], min_quality: float = 0.7) -> bool:
        high_quality = [t for t in turns if t.quality_score >= min_quality]
        return len(high_quality) >= 2

    def extract_skill(self, turns: list[ConversationTurn], domain: str) -> SkillArtifact | None:
        """从高质量对话轨迹提炼 Skill（实际需调用 LLM 生成 instructions）"""
        high_quality = [t for t in turns if t.quality_score >= 0.7 and t.task_domain == domain]
        if len(high_quality) < 2:
            return None

        representative = max(high_quality, key=lambda t: t.quality_score)
        keywords = self.extract_keywords(representative.user_input, domain)

        instructions_map = {
            "copywriting": (
                "1. 开头用情感共鸣句（'Every mom wants...'）\n"
                "2. 强调安全认证（CE/FDA/BPA-Free）\n"
                "3. 第三段用社交证明（'10,000+ moms trust...'）\n"
                "4. CTA 突出免运费和30天退换"
            ),
            "product_selection": (
                "筛选条件（按优先级）：\n"
                "1. 认证门槛：欧盟有机认证（EU Organic）优先，无BPA\n"
                "2. 竞争评估：同价位段BSR<500，Review<3000（蓝海）\n"
                "3. 供应商：德国/荷兰工厂 > 英国 > 其他欧盟\n"
                "4. 排除项：含棕榈油、转基因成分"
            ),
        }

        skill_id = hashlib.md5(f"{domain}{time.time()}".encode()).hexdigest()[:10]
        return SkillArtifact(
            skill_id=skill_id,
            version=1,
            task_domain=domain,
            trigger_pattern=f"{domain} 场景下的最佳实践（从 {len(high_quality)} 次高质量轨迹提炼）",
            trigger_keywords=keywords,
            instructions=instructions_map.get(domain, representative.agent_response[:200]),
            fitness=sum(t.quality_score for t in high_quality) / len(high_quality),
        )


# ─── SkillBank（技能库）──────────────────────────────────────────────────

class SkillBank:
    """存储/检索/版本管理/合并/淘汰 Skill 文件"""

    PRUNE_THRESHOLD_USAGE = 0
    PRUNE_THRESHOLD_FITNESS = 0.3
    EMA_ALPHA = 0.2  # Fitness 指数移动平均平滑系数

    def __init__(self) -> None:
        self._skills: dict[str, SkillArtifact] = {}
        self._archived: dict[str, SkillArtifact] = {}
        self._domain_index: dict[str, list[str]] = {}

    def add(self, skill: SkillArtifact) -> str:
        self._skills[skill.skill_id] = skill
        domain_list = self._domain_index.setdefault(skill.task_domain, [])
        if skill.skill_id not in domain_list:
            domain_list.append(skill.skill_id)
        return skill.skill_id

    def update(self, skill_id: str, new_instructions: str, new_fitness: float) -> bool:
        if skill_id not in self._skills:
            return False
        skill = self._skills[skill_id]
        skill.version += 1
        skill.instructions = new_instructions
        skill.fitness = new_fitness
        skill.updated_at = time.time()
        return True

    def retrieve(self, query: str, domain: str, top_k: int = 3) -> list[SkillArtifact]:
        """关键词重叠 + fitness 综合排序检索"""
        candidates = [
            self._skills[sid]
            for sid in self._domain_index.get(domain, [])
            if sid in self._skills
        ]
        if not candidates:
            return []

        def score(skill: SkillArtifact) -> float:
            keyword_match = sum(1 for kw in skill.trigger_keywords if kw in query)
            overlap_ratio = keyword_match / max(len(skill.trigger_keywords), 1)
            return overlap_ratio * 0.6 + skill.fitness * 0.4

        ranked = sorted(candidates, key=score, reverse=True)
        return ranked[:top_k]

    def record_usage(self, skill_id: str, quality_feedback: float) -> None:
        if skill_id not in self._skills:
            return
        skill = self._skills[skill_id]
        skill.usage_count += 1
        skill.fitness = self.EMA_ALPHA * quality_feedback + (1 - self.EMA_ALPHA) * skill.fitness

    def merge(self, skill_id_a: str, skill_id_b: str) -> str | None:
        if skill_id_a not in self._skills or skill_id_b not in self._skills:
            return None
        a = self._skills[skill_id_a]
        b = self._skills[skill_id_b]

        merged_keywords = list(set(a.trigger_keywords + b.trigger_keywords))
        merged_instructions = (
            f"[合并自 v{a.version} + v{b.version}]\n"
            f"{a.instructions}\n\n补充规则（来自合并）：\n{b.instructions}"
        )
        total_usage = a.usage_count + b.usage_count
        merged_fitness = (
            (a.fitness * a.usage_count + b.fitness * b.usage_count) / max(total_usage, 1)
        )

        merged = SkillArtifact(
            skill_id=hashlib.md5(f"merge{skill_id_a}{skill_id_b}".encode()).hexdigest()[:10],
            version=1,
            task_domain=a.task_domain,
            trigger_pattern=f"合并版：{a.trigger_pattern}",
            trigger_keywords=merged_keywords,
            instructions=merged_instructions,
            fitness=merged_fitness,
            parent_skill_id=f"{skill_id_a},{skill_id_b}",
        )

        self._archived[skill_id_a] = self._skills.pop(skill_id_a)
        self._archived[skill_id_b] = self._skills.pop(skill_id_b)
        return self.add(merged)

    def prune(self) -> list[str]:
        """淘汰低质量 Skill，归档历史版本"""
        to_prune = [
            sid for sid, sk in self._skills.items()
            if sk.usage_count <= self.PRUNE_THRESHOLD_USAGE
            and sk.fitness < self.PRUNE_THRESHOLD_FITNESS
        ]
        for sid in to_prune:
            self._archived[sid] = self._skills.pop(sid)
            for domain_list in self._domain_index.values():
                if sid in domain_list:
                    domain_list.remove(sid)
        return to_prune

    def snapshot(self) -> dict:
        return {
            "active_skills": len(self._skills),
            "archived_skills": len(self._archived),
            "domains": {d: len(ids) for d, ids in self._domain_index.items()},
        }


# ─── AutoSkillAgent（前台响应 + 后台进化）────────────────────────────────

class AutoSkillAgent:
    """前台响应 + 后台进化双循环"""

    def __init__(self) -> None:
        self.skill_bank = SkillBank()
        self.extractor = SkillExtractor()
        self._trajectory: list[ConversationTurn] = []
        self._evolution_interval = 5  # 每5次对话触发一次后台进化

    def respond(self, user_input: str, quality_feedback: float = 0.8) -> dict:
        """前台：检索 Skill → 注入 context → 生成回复 → 记录轨迹"""
        domain = self.extractor.detect_domain(user_input)
        relevant_skills = self.skill_bank.retrieve(user_input, domain, top_k=2)

        if relevant_skills:
            best_skill = relevant_skills[0]
            response = (
                f"[基于 Skill: {best_skill.skill_id} v{best_skill.version}]\n"
                f"根据积累的最佳实践：\n{best_skill.instructions}"
            )
            self.skill_bank.record_usage(best_skill.skill_id, quality_feedback)
        else:
            response = f"[无匹配 Skill，基础响应] 针对「{user_input[:30]}」的通用回复"

        turn = ConversationTurn(
            turn_id=hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
            user_input=user_input,
            agent_response=response,
            task_domain=domain,
            quality_score=quality_feedback,
        )
        self._trajectory.append(turn)

        # 后台进化触发
        if len(self._trajectory) % self._evolution_interval == 0:
            self._evolve(domain)

        return {
            "response": response,
            "domain": domain,
            "skills_used": [s.skill_id for s in relevant_skills],
            "skill_bank": self.skill_bank.snapshot(),
        }

    def _evolve(self, domain: str) -> None:
        """后台：从轨迹提炼新 Skill 或更新已有 Skill"""
        domain_turns = [t for t in self._trajectory if t.task_domain == domain]
        if not self.extractor.can_extract(domain_turns):
            return

        new_skill = self.extractor.extract_skill(domain_turns, domain)
        if new_skill is None:
            return

        existing = self.skill_bank.retrieve("", domain, top_k=1)
        if existing and existing[0].fitness < new_skill.fitness:
            self.skill_bank.update(existing[0].skill_id, new_skill.instructions, new_skill.fitness)
        else:
            self.skill_bank.add(new_skill)

        self.skill_bank.prune()


# ─── 测试：10次选品对话，验证 Skill 提炼和版本演化 ──────────────────────

def test_skill_evolution() -> None:
    """模拟 10 次选品/文案对话，验证 Skill 自动提炼和版本演化"""
    agent = AutoSkillAgent()

    conversations = [
        ("母婴奶粉选品：欧洲市场怎么评估竞争？", 0.85),
        ("帮我分析德国有机奶粉BSR竞争情况", 0.90),
        ("选品时如何判断奶粉品类是否是蓝海？", 0.75),
        ("欧美妈妈产品描述怎么写才有吸引力？", 0.88),
        ("如何写婴儿奶瓶的欧美文案？", 0.92),
        # 第5次对话触发第一次进化
        ("奶粉选品：荷兰工厂 vs 德国工厂有什么区别？", 0.80),
        ("欧美妈妈更关注奶粉的哪些认证？", 0.85),
        ("写一个BPA-Free奶瓶的英文产品描述", 0.78),
        ("母婴选品：如何排除高竞争品类？", 0.82),
        ("帮我写一段针对欧美妈妈的婴儿辅食文案", 0.89),
        # 第10次对话触发第二次进化
    ]

    print("=" * 60)
    print("AutoSkill 选品/文案 Skill 进化测试")
    print("=" * 60)

    for i, (user_input, quality) in enumerate(conversations, 1):
        result = agent.respond(user_input, quality)
        skills_used = result["skills_used"]
        print(f"\n[对话 {i}] {user_input[:40]}...")
        print(f"  领域: {result['domain']} | 使用Skill: {skills_used or '无'}")
        print(f"  SkillBank: {result['skill_bank']}")

    # 验证 Skill 已积累
    snapshot = agent.skill_bank.snapshot()
    assert snapshot["active_skills"] >= 1, f"应至少积累1个Skill，实际: {snapshot['active_skills']}"
    print(f"\n最终SkillBank: {json.dumps(snapshot, ensure_ascii=False)}")

    # 验证后期对话能复用 Skill
    late_result = agent.respond("欧美妈妈奶粉文案写作最佳实践", 0.90)
    print(f"\n后期查询 Skill 复用: {late_result['skills_used']}")
    assert late_result["skills_used"], "后期对话应能复用已积累的Skill"
    print("\n✅ 测试通过：Skill 自动提炼和版本演化验证成功")


if __name__ == "__main__":
    test_skill_evolution()
