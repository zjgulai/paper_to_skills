"""
Skill Registry — 技能注册表与动态发现
自定义框架（基于 Microservices Service Discovery + DAG Dependency Resolution）

核心能力:
1. 技能注册 — 声明式注册技能元数据
2. 技能发现 — 根据 Task Blueprint 动态匹配技能
3. 依赖解析 — 拓扑排序确定执行顺序
4. 版本管理 — 多版本共存与回滚

母婴电商场景: Task Blueprint 动态匹配 VOC 分析所需技能
"""

from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class SkillStatus(Enum):
    """技能状态"""
    ACTIVE = "active"           # 活跃可用
    DEPRECATED = "deprecated"   # 已弃用
    BETA = "beta"               # 测试中


@dataclass
class SkillSchema:
    """技能输入/输出 Schema"""
    input_type: str
    input_format: str
    output_type: str
    output_format: str
    required_fields: List[str] = field(default_factory=list)


@dataclass
class SkillMetrics:
    """技能性能指标"""
    f1_score: float = 0.0
    latency_ms: int = 0
    cost_per_call: float = 0.0
    success_rate: float = 1.0
    last_evaluated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SkillMetadata:
    """技能元数据"""
    name: str
    version: str
    description: str
    skill_type: str           # entity_extraction, sentiment_analysis, etc.
    schema: SkillSchema
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他技能
    status: SkillStatus = SkillStatus.ACTIVE
    metrics: SkillMetrics = field(default_factory=SkillMetrics)
    tags: List[str] = field(default_factory=list)
    quality_threshold: float = 0.8


class SkillRegistry:
    """
    技能注册表

    管理所有可用技能的元数据、依赖关系和运行时状态。
    是 MAS 工作流的"技能目录"。
    """

    def __init__(self):
        self._skills: Dict[str, List[SkillMetadata]] = {}  # name -> versions
        self._by_type: Dict[str, List[str]] = {}           # skill_type -> skill_names

    def register(self, skill: SkillMetadata) -> bool:
        """注册技能"""
        name = skill.name

        if name not in self._skills:
            self._skills[name] = []
        self._skills[name].append(skill)

        # 按类型索引
        if skill.skill_type not in self._by_type:
            self._by_type[skill.skill_type] = []
        if name not in self._by_type[skill.skill_type]:
            self._by_type[skill.skill_type].append(name)

        # 按版本排序（最新在前）
        self._skills[name].sort(key=lambda s: s.version, reverse=True)
        return True

    def discover(self, skill_type: str, min_f1: float = 0.0,
                 max_latency_ms: int = 10000,
                 status: Optional[SkillStatus] = None) -> List[SkillMetadata]:
        """
        根据类型和约束发现技能

        Returns:
            按 F1 分数排序的候选技能列表
        """
        candidates = []
        for name in self._by_type.get(skill_type, []):
            for skill in self._skills[name]:
                # 过滤条件
                if status and skill.status != status:
                    continue
                if skill.metrics.f1_score < min_f1:
                    continue
                if skill.metrics.latency_ms > max_latency_ms:
                    continue
                candidates.append(skill)

        # 按 F1 排序
        candidates.sort(key=lambda s: s.metrics.f1_score, reverse=True)
        return candidates

    def resolve_dependencies(self, skill_names: List[str]) -> List[str]:
        """
        解析技能依赖，返回拓扑排序后的执行顺序

        Returns:
            按依赖顺序排列的技能名称列表
        """
        # 构建依赖图
        graph: Dict[str, Set[str]] = {}
        all_skills = set()

        for name in skill_names:
            all_skills.add(name)
            skill = self._get_latest(name)
            if skill:
                graph[name] = set(skill.dependencies)
                all_skills.update(skill.dependencies)

        # Kahn 算法拓扑排序
        in_degree = {name: 0 for name in all_skills}
        for name, deps in graph.items():
            for dep in deps:
                in_degree[name] = in_degree.get(name, 0) + 1

        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for name, deps in graph.items():
                if node in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # 过滤只保留原始请求的技能（依赖项作为前置）
        ordered = [n for n in result if n in skill_names]
        return ordered

    def check_compatibility(self, upstream: str, downstream: str) -> bool:
        """检查两个技能的 Schema 兼容性"""
        up_skill = self._get_latest(upstream)
        down_skill = self._get_latest(downstream)

        if not up_skill or not down_skill:
            return False

        # 简化检查：输出格式与输入格式是否匹配
        return (up_skill.schema.output_type == down_skill.schema.input_type and
                up_skill.schema.output_format == down_skill.schema.input_format)

    def get_execution_plan(self, required_skills: List[str]) -> Dict:
        """
        生成完整执行计划

        Returns:
            包含技能列表、执行顺序、兼容性检查、预估质量
        """
        # 1. 获取技能元数据
        skill_metas = []
        for name in required_skills:
            skill = self._get_latest(name)
            if skill:
                skill_metas.append(skill)

        # 2. 解析依赖
        execution_order = self.resolve_dependencies(required_skills)

        # 3. 检查兼容性
        compatibility_issues = []
        for i in range(len(execution_order) - 1):
            up = execution_order[i]
            down = execution_order[i + 1]
            if not self.check_compatibility(up, down):
                compatibility_issues.append(f"{up} -> {down}: Schema 不兼容")

        # 4. 预估质量
        estimated_quality = min(s.metrics.f1_score for s in skill_metas) if skill_metas else 0.0

        return {
            "execution_order": execution_order,
            "skills": [
                {
                    "name": s.name,
                    "version": s.version,
                    "f1": s.metrics.f1_score,
                    "latency_ms": s.metrics.latency_ms,
                }
                for s in skill_metas
            ],
            "compatibility_issues": compatibility_issues,
            "estimated_quality": estimated_quality,
            "can_execute": len(compatibility_issues) == 0,
        }

    def _get_latest(self, name: str) -> Optional[SkillMetadata]:
        """获取技能的最新版本"""
        versions = self._skills.get(name, [])
        if versions:
            # 返回第一个 ACTIVE 版本，如果没有则返回最新
            for v in versions:
                if v.status == SkillStatus.ACTIVE:
                    return v
            return versions[0]
        return None

    def list_skills(self) -> Dict[str, List[Dict]]:
        """列出所有注册的技能"""
        return {
            name: [
                {
                    "version": s.version,
                    "type": s.skill_type,
                    "f1": s.metrics.f1_score,
                    "status": s.status.value,
                }
                for s in versions
            ]
            for name, versions in self._skills.items()
        }


class VersionManager:
    """技能版本管理"""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def canary_release(self, skill_name: str, traffic_ratio: float = 0.1):
        """金丝雀发布：将部分流量切换到新版本"""
        skill = self.registry._get_latest(skill_name)
        if skill:
            print(f"  金丝雀发布: {skill_name} v{skill.version}, 流量比例 {traffic_ratio:.0%}")
            return True
        return False

    def rollback(self, skill_name: str, target_version: str) -> bool:
        """回滚到指定版本"""
        versions = self.registry._skills.get(skill_name, [])
        for v in versions:
            if v.version == target_version:
                print(f"  回滚: {skill_name} -> v{target_version}")
                return True
        return False


# ============================================
# 母婴电商场景 — Skill Registry 演示
# ============================================

def demo_skill_registry():
    """演示 Skill Registry 的核心功能"""
    print("=" * 70)
    print("Skill Registry — 技能注册表")
    print("=" * 70)

    registry = SkillRegistry()

    # 注册技能
    print("\n[1] 注册技能")
    skills = [
        SkillMetadata(
            name="InstructUIE",
            version="2.1",
            description="统一信息抽取",
            skill_type="entity_extraction",
            schema=SkillSchema(
                input_type="raw_text", input_format="string",
                output_type="structured", output_format="json",
                required_fields=["entities", "relations", "events"]
            ),
            metrics=SkillMetrics(f1_score=0.91, latency_ms=120, cost_per_call=0.002),
            tags=["NLP", "extraction", "multilingual"]
        ),
        SkillMetadata(
            name="ABSA-BERT-MoE",
            version="1.5",
            description="方面级情感分析",
            skill_type="sentiment_analysis",
            schema=SkillSchema(
                input_type="structured", input_format="json",
                output_type="structured", output_format="json",
                required_fields=["aspect", "sentiment", "confidence"]
            ),
            metrics=SkillMetrics(f1_score=0.89, latency_ms=80, cost_per_call=0.001),
            tags=["NLP", "sentiment", "ABSA"]
        ),
        SkillMetadata(
            name="HGT-Ecommerce",
            version="1.0",
            description="电商异构图表示学习",
            skill_type="graph_reasoning",
            schema=SkillSchema(
                input_type="structured", input_format="json",
                output_type="embedding", output_format="tensor",
                required_fields=["nodes", "edges", "node_types"]
            ),
            dependencies=["InstructUIE"],
            metrics=SkillMetrics(f1_score=0.85, latency_ms=300, cost_per_call=0.005),
            tags=["GNN", "HGT", "reasoning"]
        ),
        SkillMetadata(
            name="TextBlob",
            version="0.17",
            description="简单文档级情感分析",
            skill_type="sentiment_analysis",
            schema=SkillSchema(
                input_type="raw_text", input_format="string",
                output_type="structured", output_format="json"
            ),
            metrics=SkillMetrics(f1_score=0.72, latency_ms=20, cost_per_call=0.0),
            status=SkillStatus.DEPRECATED,
            tags=["baseline", "fast"]
        ),
    ]

    for skill in skills:
        registry.register(skill)
        print(f"  注册: {skill.name} v{skill.version} ({skill.skill_type}, F1={skill.metrics.f1_score})")

    # 技能发现
    print("\n[2] 技能发现 — entity_extraction")
    candidates = registry.discover("entity_extraction", min_f1=0.8)
    for c in candidates:
        print(f"  候选: {c.name} v{c.version} (F1={c.metrics.f1_score}, latency={c.metrics.latency_ms}ms)")

    print("\n[3] 技能发现 — sentiment_analysis")
    candidates = registry.discover("sentiment_analysis", min_f1=0.8)
    for c in candidates:
        print(f"  候选: {c.name} v{c.version} (F1={c.metrics.f1_score})")

    # 生成执行计划
    print("\n[4] 生成执行计划")
    plan = registry.get_execution_plan(["InstructUIE", "ABSA-BERT-MoE", "HGT-Ecommerce"])

    print(f"  执行顺序: {' -> '.join(plan['execution_order'])}")
    print(f"  预估质量: {plan['estimated_quality']:.3f}")
    print(f"  兼容性检查: {'通过' if plan['can_execute'] else '失败'}")
    if plan['compatibility_issues']:
        for issue in plan['compatibility_issues']:
            print(f"    问题: {issue}")

    # Schema 兼容性检查
    print("\n[5] Schema 兼容性检查")
    compat1 = registry.check_compatibility("InstructUIE", "ABSA-BERT-MoE")
    compat2 = registry.check_compatibility("ABSA-BERT-MoE", "HGT-Ecommerce")
    print(f"  InstructUIE -> ABSA: {'兼容' if compat1 else '不兼容'}")
    print(f"  ABSA -> HGT: {'兼容' if compat2 else '不兼容'}")

    # 版本管理
    print("\n[6] 版本管理")
    vm = VersionManager(registry)
    vm.canary_release("InstructUIE", traffic_ratio=0.1)
    vm.rollback("InstructUIE", "2.0")

    print("\n" + "=" * 70)
    return registry


def demo_task_blueprint_matching():
    """演示 Task Blueprint 动态匹配"""
    print("\n" + "=" * 70)
    print("Task Blueprint 动态技能匹配")
    print("=" * 70)

    registry = SkillRegistry()

    # 注册更多技能
    registry.register(SkillMetadata(
        name="InstructUIE", version="2.1",
        description="统一信息抽取", skill_type="entity_extraction",
        schema=SkillSchema("raw_text", "string", "structured", "json"),
        metrics=SkillMetrics(f1_score=0.91, latency_ms=120)
    ))
    registry.register(SkillMetadata(
        name="ABSA-BERT-MoE", version="1.5",
        description="方面级情感分析", skill_type="sentiment_analysis",
        schema=SkillSchema("structured", "json", "structured", "json"),
        metrics=SkillMetrics(f1_score=0.89, latency_ms=80)
    ))
    registry.register(SkillMetadata(
        name="BERT-CRF", version="1.0",
        description="传统实体抽取", skill_type="entity_extraction",
        schema=SkillSchema("raw_text", "string", "structured", "json"),
        metrics=SkillMetrics(f1_score=0.87, latency_ms=50)
    ))

    # 模拟 Task Blueprint
    task_blueprint = {
        "task_type": "EXTRACT",
        "description": "抽取本周吸奶器评论中的实体和情感",
        "required_skill_types": ["entity_extraction", "sentiment_analysis"],
        "quality_threshold": 0.85,
    }

    print(f"\n[输入 Task Blueprint]")
    print(f"  任务: {task_blueprint['description']}")
    print(f"  所需技能类型: {task_blueprint['required_skill_types']}")
    print(f"  质量阈值: {task_blueprint['quality_threshold']}")

    print(f"\n[匹配结果]")
    matched_skills = []
    for skill_type in task_blueprint["required_skill_types"]:
        candidates = registry.discover(skill_type, min_f1=task_blueprint["quality_threshold"])
        if candidates:
            best = candidates[0]
            matched_skills.append(best.name)
            print(f"  {skill_type}: {best.name} v{best.version} (F1={best.metrics.f1_score}) ← 最优匹配")
            for c in candidates[1:]:
                print(f"    备选: {c.name} v{c.version} (F1={c.metrics.f1_score})")

    plan = registry.get_execution_plan(matched_skills)
    print(f"\n[执行计划]")
    print(f"  执行顺序: {' -> '.join(plan['execution_order'])}")
    print(f"  预估质量: {plan['estimated_quality']:.3f} (阈值: {task_blueprint['quality_threshold']})")
    print(f"  可执行: {'是' if plan['can_execute'] else '否'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_skill_registry()
    demo_task_blueprint_matching()

    print("\n生产环境建议:")
    print("  1. 使用 PostgreSQL/MongoDB 持久化技能元数据")
    print("  2. 实现技能健康检查（定期探测可用性）")
    print("  3. 集成 Prometheus metrics（匹配延迟、成功率、版本分布）")
    print("  4. 支持技能热更新（无需重启 Registry）")
    print("  5. 实现技能评分排序（综合 F1、延迟、成本、稳定性）")
    print("  6. 与 Subagent Decomposer 集成：Registry 提供技能列表，Decomposer 决定组合")
