"""
Subagent Decomposer — 复杂任务子智能体分解
自定义框架（基于 DAG Decomposition + MapReduce Pattern）

核心能力:
1. 横向分解 — 按数据维度并行拆分
2. 纵向分解 — 按处理阶段串行拆分
3. 混合分解 — 横向+纵向结合，形成 DAG
4. 执行排序 — 拓扑排序确定执行顺序

母婴电商场景: 全品类 VOC 周报生成、竞品深度对标分析
"""

from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class DecompositionType(Enum):
    """分解类型"""
    PARALLEL = "parallel"       # 横向并行
    SEQUENTIAL = "sequential"   # 纵向串行
    HYBRID = "hybrid"           # 混合


class SubtaskStatus(Enum):
    """子任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """子任务"""
    task_id: str
    name: str
    skill_name: str
    data_scope: str           # 数据范围（如"吸奶器品类"、"全量"）
    dependencies: List[str] = field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    output: Any = None
    estimated_time_ms: int = 1000


@dataclass
class TaskDAG:
    """任务依赖图"""
    subtasks: Dict[str, Subtask] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)  # task_id -> downstream task_ids

    def add_subtask(self, subtask: Subtask):
        """添加子任务"""
        self.subtasks[subtask.task_id] = subtask
        if subtask.task_id not in self.edges:
            self.edges[subtask.task_id] = set()

    def add_dependency(self, upstream: str, downstream: str):
        """添加依赖关系：downstream 依赖 upstream"""
        if upstream in self.subtasks and downstream in self.subtasks:
            self.edges[upstream].add(downstream)
            if downstream not in self.subtasks[downstream].dependencies:
                self.subtasks[downstream].dependencies.append(upstream)

    def topological_sort(self) -> List[str]:
        """拓扑排序获取执行顺序"""
        in_degree = {tid: 0 for tid in self.subtasks}
        for upstream, downstreams in self.edges.items():
            for d in downstreams:
                in_degree[d] += 1

        queue = deque([tid for tid, d in in_degree.items() if d == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for downstream in self.edges.get(node, set()):
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        return result

    def get_parallel_groups(self) -> List[List[str]]:
        """获取可并行执行的子任务组（按层）"""
        sorted_ids = self.topological_sort()
        levels: Dict[str, int] = {}

        for tid in sorted_ids:
            subtask = self.subtasks[tid]
            if not subtask.dependencies:
                levels[tid] = 0
            else:
                levels[tid] = max(levels.get(dep, 0) for dep in subtask.dependencies) + 1

        max_level = max(levels.values()) if levels else 0
        groups = []
        for l in range(max_level + 1):
            group = [tid for tid, level in levels.items() if level == l]
            if group:
                groups.append(group)
        return groups

    def get_ready_tasks(self) -> List[str]:
        """获取当前可执行的子任务（依赖全部完成）"""
        ready = []
        for tid, subtask in self.subtasks.items():
            if subtask.status == SubtaskStatus.PENDING:
                deps_satisfied = all(
                    self.subtasks[dep].status == SubtaskStatus.SUCCESS
                    for dep in subtask.dependencies
                )
                if deps_satisfied:
                    ready.append(tid)
        return ready

    def visualize(self) -> str:
        """可视化 DAG"""
        lines = ["Task DAG:"]
        for tid, subtask in self.subtasks.items():
            deps = ", ".join(subtask.dependencies) if subtask.dependencies else "none"
            downstream = ", ".join(self.edges.get(tid, set())) or "none"
            lines.append(f"  [{tid}] {subtask.name}")
            lines.append(f"    deps: {deps} | downstream: {downstream}")
        return "\n".join(lines)


class SubagentDecomposer:
    """
    子任务分解器

    将复杂 Task Blueprint 分解为可并行/串行执行的子任务 DAG。
    """

    def __init__(self):
        pass

    def decompose(self, task_description: str,
                  data_dimensions: List[str],
                  processing_stages: List[str],
                  skills: Dict[str, str]) -> TaskDAG:
        """
        混合分解：横向（按数据维度）+ 纵向（按处理阶段）

        Args:
            task_description: 任务描述
            data_dimensions: 数据维度列表（如品类列表）
            processing_stages: 处理阶段列表（如抽取→分析→汇总）
            skills: 阶段到技能的映射

        Returns:
            TaskDAG
        """
        dag = TaskDAG()

        # 阶段 1: 横向并行（每个数据维度独立处理）
        stage1_tasks = []
        for dim in data_dimensions:
            for stage in processing_stages[:-1]:  # 除最后一个汇总阶段
                task_id = f"{stage}_{dim}"
                subtask = Subtask(
                    task_id=task_id,
                    name=f"{stage} ({dim})",
                    skill_name=skills.get(stage, "default"),
                    data_scope=dim
                )
                dag.add_subtask(subtask)
                stage1_tasks.append(task_id)

        # 阶段间依赖：同一数据维度的处理阶段串行
        for dim in data_dimensions:
            for i in range(len(processing_stages) - 2):
                upstream = f"{processing_stages[i]}_{dim}"
                downstream = f"{processing_stages[i+1]}_{dim}"
                dag.add_dependency(upstream, downstream)

        # 阶段 2: 汇总阶段（依赖所有数据维度的最后阶段完成）
        final_stage = processing_stages[-1]
        final_task_id = f"{final_stage}_all"
        final_subtask = Subtask(
            task_id=final_task_id,
            name=f"{final_stage} (汇总)",
            skill_name=skills.get(final_stage, "default"),
            data_scope="all"
        )
        dag.add_subtask(final_subtask)

        # 汇总阶段依赖所有数据维度的前一阶段完成
        last_stage = processing_stages[-2]
        for dim in data_dimensions:
            upstream = f"{last_stage}_{dim}"
            dag.add_dependency(upstream, final_task_id)

        return dag

    def decompose_by_dimensions(self, task_description: str,
                                 dimensions: List[str],
                                 skill_name: str) -> TaskDAG:
        """纯横向分解：按数据维度并行"""
        dag = TaskDAG()

        for dim in dimensions:
            task_id = f"task_{dim}"
            dag.add_subtask(Subtask(
                task_id=task_id,
                name=f"{task_description} ({dim})",
                skill_name=skill_name,
                data_scope=dim
            ))

        return dag

    def decompose_by_stages(self, task_description: str,
                            stages: List[str],
                            skills: List[str]) -> TaskDAG:
        """纯纵向分解：按处理阶段串行"""
        dag = TaskDAG()

        prev_id = None
        for i, (stage, skill) in enumerate(zip(stages, skills)):
            task_id = f"stage_{i}"
            dag.add_subtask(Subtask(
                task_id=task_id,
                name=stage,
                skill_name=skill,
                data_scope="all"
            ))
            if prev_id:
                dag.add_dependency(prev_id, task_id)
            prev_id = task_id

        return dag


# ============================================
# 母婴电商场景 — 全品类 VOC 周报分解
# ============================================

def demo_weekly_report_decomposition():
    """演示全品类 VOC 周报的子任务分解"""
    print("=" * 70)
    print("Subagent Decomposer — 全品类 VOC 周报生成")
    print("=" * 70)

    decomposer = SubagentDecomposer()

    # 定义分解参数
    categories = ["吸奶器", "储奶袋", "温奶器", "推车", "安全座椅", "洗护", "喂养配件", "其他"]
    stages = ["抽取", "情感分析", "趋势检测", "汇总报告"]
    skills = {
        "抽取": "InstructUIE",
        "情感分析": "ABSA-BERT-MoE",
        "趋势检测": "TrendAnalyzer",
        "汇总报告": "ReportGenerator"
    }

    print(f"\n[任务] 生成全品类 VOC 周报")
    print(f"[数据维度] {len(categories)} 个品类")
    print(f"[处理阶段] {len(stages)} 个阶段")

    # 混合分解
    dag = decomposer.decompose(
        task_description="全品类 VOC 周报",
        data_dimensions=categories,
        processing_stages=stages,
        skills=skills
    )

    print(f"\n[分解结果]")
    print(f"  子任务总数: {len(dag.subtasks)}")

    # 并行组
    groups = dag.get_parallel_groups()
    print(f"\n[并行执行组] {len(groups)} 层")
    for i, group in enumerate(groups):
        tasks = [dag.subtasks[tid].name for tid in group]
        print(f"  层 {i}: {len(group)} 个任务并行")
        for t in tasks[:3]:
            print(f"    - {t}")
        if len(tasks) > 3:
            print(f"    ... 和 {len(tasks) - 3} 个其他任务")

    # 拓扑排序
    order = dag.topological_sort()
    print(f"\n[拓扑排序] 执行顺序 (前 10 个):")
    for tid in order[:10]:
        subtask = dag.subtasks[tid]
        deps = ", ".join(subtask.dependencies) if subtask.dependencies else "无"
        print(f"  {tid}: {subtask.name} (依赖: {deps})")
    if len(order) > 10:
        print(f"  ... 共 {len(order)} 个任务")

    # 性能估算
    parallel_time = sum(max(dag.subtasks[tid].estimated_time_ms for tid in group) for group in groups)
    serial_time = len(categories) * (len(stages) - 1) * 1000
    print(f"\n[性能估算]")
    print(f"  理论串行时间: {serial_time/1000:.1f}s")
    print(f"  并行优化时间: {parallel_time/1000:.1f}s")
    print(f"  加速比: {serial_time/max(parallel_time, 1):.1f}x")

    print("\n" + "=" * 70)


def demo_competitor_analysis_decomposition():
    """演示竞品深度分析的子任务分解"""
    print("\n" + "=" * 70)
    print("Subagent Decomposer — 竞品深度对标分析")
    print("=" * 70)

    decomposer = SubagentDecomposer()

    # 纯横向分解：按维度并行
    dimensions = ["产品规格", "用户评价", "价格竞争力", "市场份额"]
    dag = decomposer.decompose_by_dimensions(
        task_description="竞品对标分析",
        dimensions=dimensions,
        skill_name="CompetitorAnalyzer"
    )

    print(f"\n[任务] 深度对标 Spectra S1 vs Medela vs Elvie")
    print(f"[分解策略] 横向并行（按分析维度）")
    print(f"\n[子任务] {len(dag.subtasks)} 个并行任务:")
    for tid, subtask in dag.subtasks.items():
        print(f"  - {subtask.name}")

    # 纯纵向分解：按阶段串行
    stages = ["数据获取", "分析处理", "综合对比", "报告生成"]
    stage_skills = ["DataCollector", "Analyzer", "Comparator", "ReportGen"]
    dag2 = decomposer.decompose_by_stages(
        task_description="竞品分析流水线",
        stages=stages,
        skills=stage_skills
    )

    print(f"\n[分解策略] 纵向串行（按处理阶段）")
    print(f"[执行顺序]:")
    for tid in dag2.topological_sort():
        print(f"  -> {dag2.subtasks[tid].name}")

    print("\n" + "=" * 70)


def demonstrate_decomposition_patterns():
    """展示分解模式对比"""
    print("\n" + "=" * 70)
    print("分解模式对比")
    print("=" * 70)

    print(r"""
    横向分解 (Parallel):
      适用: 数据量大、按维度可独立处理
      例: 8 个品类分别分析
      结构:
        [品类1] [品类2] [品类3] ... [品类8]
           |       |       |           |
        (并行执行，无依赖)

    纵向分解 (Sequential):
      适用: 处理流程明确、阶段间有依赖
      例: 抽取 → 分析 → 汇总 → 报告
      结构:
        [抽取] -> [分析] -> [汇总] -> [报告]

    混合分解 (Hybrid DAG):
      适用: 复杂任务，既有并行又有串行
      例: 全品类 VOC 周报
      结构:
        [抽_品1] [抽_品2] ... [抽_品8]
            |       |           |
        [情_品1] [情_品2] ... [情_品8]
            |       |           |
            \       |           /
             \      |          /
              \     |         /
               \    |        /
                [汇总报告]

      优势: 最大化并行度，同时保证依赖正确
    """)


if __name__ == "__main__":
    demo_weekly_report_decomposition()
    demo_competitor_analysis_decomposition()
    demonstrate_decomposition_patterns()

    print("\n生产环境建议:")
    print("  1. 使用 Temporal / Celery 作为底层任务调度引擎")
    print("  2. 实现子任务容错（失败重试、降级策略）")
    print("  3. 监控 DAG 执行状态（完成/失败/进行中）")
    print("  4. 支持动态调整（执行中发现可进一步分解）")
    print("  5. 结果缓存（相同输入直接复用结果）")
    print("  6. 与 Skill Registry 集成：Decomposer 查询技能边界")
