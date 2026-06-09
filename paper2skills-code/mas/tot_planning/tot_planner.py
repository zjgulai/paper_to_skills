"""
Tree of Thoughts (ToT) — 树搜索式任务规划
基于论文: Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", NeurIPS 2023

核心能力:
1. Thought Decomposition — 将问题分解为中间推理步骤
2. Thought Generation — 从每个节点生成多个候选 thoughts
3. State Evaluation — 评估每个 thought 的前景
4. Search Algorithm — BFS/DFS 搜索最优路径

母婴电商场景: VOC 标签体系设计策略搜索、评论分类策略优化
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random


@dataclass
class ThoughtNode:
    """Thought 树节点"""
    content: str
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    score: Optional[float] = None
    depth: int = 0

    def path(self) -> List[str]:
        """从根到当前节点的 thought 路径"""
        if self.parent is None:
            return [self.content]
        return self.parent.path() + [self.content]


class SearchStrategy(Enum):
    """搜索策略"""
    BFS = "bfs"   # 广度优先：每层保留 top-k
    DFS = "dfs"   # 深度优先：深度探索后回溯


class ThoughtEvaluator:
    """
    Thought 评估器

    评估每个 thought 的前景（价值或置信度）。
    生产环境使用 LLM 进行评估，这里使用模拟评估。
    """

    def __init__(self, eval_func: Optional[Callable] = None):
        self.eval_func = eval_func or self._mock_evaluate

    def evaluate(self, node: ThoughtNode, problem: str) -> float:
        """评估 thought 的前景，返回 0-1 分数"""
        return self.eval_func(node, problem)

    def _mock_evaluate(self, node: ThoughtNode, problem: str) -> float:
        """模拟评估器 — 基于关键词和内容启发式打分"""
        content = node.content.lower()

        # 标签体系设计场景的启发式规则
        if "标签" in problem or "label" in problem.lower():
            if "覆盖率" in content or "coverage" in content:
                return 0.85 + random.uniform(-0.05, 0.05)
            elif "层级" in content or "hierarchy" in content:
                return 0.75 + random.uniform(-0.05, 0.05)
            elif "粒度" in content or "granularity" in content:
                return 0.70 + random.uniform(-0.05, 0.05)
            elif "兼容" in content or "compatible" in content:
                return 0.80 + random.uniform(-0.05, 0.05)

        # 分类策略场景的启发式规则
        if "分类" in problem or "classify" in problem.lower():
            if "属性" in content or "attribute" in content:
                return 0.82 + random.uniform(-0.05, 0.05)
            elif "情感" in content or "sentiment" in content:
                return 0.78 + random.uniform(-0.05, 0.05)
            elif "多标签" in content or "multi-label" in content:
                return 0.65 + random.uniform(-0.05, 0.05)

        return 0.60 + random.uniform(-0.10, 0.10)


class ThoughtGenerator:
    """
    Thought 生成器

    从当前节点生成 k 个候选 thoughts。
    """

    def __init__(self, k: int = 3, generate_func: Optional[Callable] = None):
        self.k = k
        self.generate_func = generate_func or self._mock_generate

    def generate(self, node: ThoughtNode, problem: str) -> List[str]:
        """生成 k 个候选 thoughts"""
        return self.generate_func(node, problem, self.k)

    def _mock_generate(self, node: ThoughtNode, problem: str, k: int) -> List[str]:
        """模拟 thought 生成"""
        depth = node.depth + 1 if node else 0

        # 标签体系设计场景的候选 thoughts
        if "标签" in problem or "label" in problem.lower():
            if depth == 0:
                return [
                    "按产品功能分类（喂养/洗护/出行/睡眠）",
                    "按用户旅程分类（购买前/购买中/使用后）",
                    "按用户人群分类（新手妈妈/二胎妈妈/职场妈妈）",
                ][:k]
            elif depth == 1:
                return [
                    "3层结构：品类 → 属性 → 具体特征",
                    "4层结构：大类 → 子类 → 属性 → 情感",
                    "2层扁平结构：直接到属性级别",
                ][:k]
            else:
                return [
                    f"细化方案 {i+1}: 增加跨品类通用属性（价格/质量/物流）"
                    for i in range(k)
                ]

        # 分类策略场景的候选 thoughts
        if "分类" in problem or "classify" in problem.lower():
            if depth == 0:
                return [
                    "先识别属性 → 再判断情感",
                    "先判断情感 → 再识别属性",
                    "一步多标签分类（同时输出所有标签）",
                ][:k]
            elif depth == 1:
                return [
                    "使用规则匹配做粗分类，LLM 做细分类",
                    "纯 LLM 分类，不做预分类",
                    "分层迭代：先大类 → 再子类 → 再情感",
                ][:k]
            else:
                return [
                    f"优化方案 {i+1}: 增加否定句式处理"
                    for i in range(k)
                ]

        return [f"候选方案 {i+1}" for i in range(k)]


class TreeOfThoughts:
    """
    Tree of Thoughts 规划器

    维护 thought 树，支持 BFS/DFS 搜索最优路径。
    """

    def __init__(self, k: int = 3, max_depth: int = 3,
                 search_strategy: SearchStrategy = SearchStrategy.BFS,
                 evaluator: Optional[ThoughtEvaluator] = None,
                 generator: Optional[ThoughtGenerator] = None):
        self.k = k
        self.max_depth = max_depth
        self.search_strategy = search_strategy
        self.evaluator = evaluator or ThoughtEvaluator()
        self.generator = generator or ThoughtGenerator(k=k)
        self.root: Optional[ThoughtNode] = None
        self.all_nodes: List[ThoughtNode] = []

    def solve(self, problem: str) -> Dict:
        """
        解决给定问题

        Returns:
            包含最优路径、搜索树、统计信息的字典
        """
        # 初始化根节点
        self.root = ThoughtNode(content=f"问题: {problem}")
        self.all_nodes = [self.root]

        if self.search_strategy == SearchStrategy.BFS:
            result = self._bfs(problem)
        else:
            result = self._dfs(problem)

        return result

    def _bfs(self, problem: str) -> Dict:
        """广度优先搜索"""
        current_level = [self.root]
        best_path = None
        best_score = -1

        for depth in range(self.max_depth):
            next_level = []

            for node in current_level:
                # 生成候选 thoughts
                candidates = self.generator.generate(node, problem)

                for content in candidates:
                    child = ThoughtNode(
                        content=content,
                        parent=node,
                        depth=depth + 1
                    )
                    # 评估
                    child.score = self.evaluator.evaluate(child, problem)
                    node.children.append(child)
                    self.all_nodes.append(child)
                    next_level.append(child)

            # 按分数排序，保留 top-k
            next_level.sort(key=lambda n: n.score or 0, reverse=True)
            current_level = next_level[:self.k]

            # 记录最优路径
            if current_level:
                top = current_level[0]
                if (top.score or 0) > best_score:
                    best_score = top.score or 0
                    best_path = top.path()

        return {
            "strategy": "BFS",
            "best_path": best_path or [],
            "best_score": best_score,
            "total_nodes": len(self.all_nodes),
            "max_depth_reached": min(self.max_depth, max(n.depth for n in self.all_nodes)),
            "tree": self._serialize_tree()
        }

    def _dfs(self, problem: str) -> Dict:
        """深度优先搜索（简化版）"""
        best_path = None
        best_score = -1

        def dfs_helper(node: ThoughtNode):
            nonlocal best_path, best_score

            if node.depth >= self.max_depth:
                # 到达叶子，评估完整路径
                path_score = sum(n.score or 0 for n in self.all_nodes if n in self._get_path_nodes(node)) / max(1, node.depth)
                if path_score > best_score:
                    best_score = path_score
                    best_path = node.path()
                return

            # 生成候选
            candidates = self.generator.generate(node, problem)
            for content in candidates:
                child = ThoughtNode(
                    content=content,
                    parent=node,
                    depth=node.depth + 1
                )
                child.score = self.evaluator.evaluate(child, problem)
                node.children.append(child)
                self.all_nodes.append(child)
                dfs_helper(child)

        dfs_helper(self.root)

        return {
            "strategy": "DFS",
            "best_path": best_path or [],
            "best_score": best_score,
            "total_nodes": len(self.all_nodes),
            "max_depth_reached": max(n.depth for n in self.all_nodes),
            "tree": self._serialize_tree()
        }

    def _get_path_nodes(self, node: ThoughtNode) -> List[ThoughtNode]:
        """获取从根到节点的路径上的所有节点"""
        nodes = []
        current = node
        while current:
            nodes.append(current)
            current = current.parent
        return nodes[::-1]

    def _serialize_tree(self) -> List[Dict]:
        """序列化树结构"""
        def serialize_node(node: ThoughtNode) -> Dict:
            return {
                "content": node.content[:50],
                "score": round(node.score, 3) if node.score else None,
                "depth": node.depth,
                "children": [serialize_node(c) for c in node.children]
            }
        return [serialize_node(self.root)]


# ============================================
# 母婴电商场景 — ToT 标签体系设计策略搜索
# ============================================

def demo_tot_label_design():
    """演示 ToT 在 VOC 标签体系设计中的应用"""
    print("=" * 70)
    print("Tree of Thoughts — 标签体系设计策略搜索")
    print("=" * 70)

    problem = "设计一个最优的母婴产品 VOC 评论标签体系"

    print(f"\n[问题] {problem}")
    print(f"[配置] 分支因子 k=3, 最大深度=3, 搜索策略=BFS")

    # BFS 搜索
    tot = TreeOfThoughts(k=3, max_depth=3, search_strategy=SearchStrategy.BFS)
    result = tot.solve(problem)

    print(f"\n[搜索结果]")
    print(f"  策略: {result['strategy']}")
    print(f"  搜索节点数: {result['total_nodes']}")
    print(f"  最大深度: {result['max_depth_reached']}")
    print(f"  最优路径得分: {result['best_score']:.3f}")

    print(f"\n[最优路径]")
    for i, step in enumerate(result['best_path'], 1):
        print(f"  Step {i}: {step}")

    # DFS 对比
    print(f"\n[对比] DFS 搜索")
    tot_dfs = TreeOfThoughts(k=3, max_depth=3, search_strategy=SearchStrategy.DFS)
    result_dfs = tot_dfs.solve(problem)
    print(f"  DFS 最优得分: {result_dfs['best_score']:.3f}")
    print(f"  DFS 搜索节点数: {result_dfs['total_nodes']}")

    print("\n" + "=" * 70)


def demo_tot_classification_strategy():
    """演示 ToT 在评论分类策略搜索中的应用"""
    print("\n" + "=" * 70)
    print("Tree of Thoughts — 评论分类策略搜索")
    print("=" * 70)

    problem = "搜索最优的母婴产品评论多标签分类策略"

    print(f"\n[问题] {problem}")

    tot = TreeOfThoughts(k=3, max_depth=3, search_strategy=SearchStrategy.BFS)
    result = tot.solve(problem)

    print(f"\n[搜索结果]")
    print(f"  最优路径得分: {result['best_score']:.3f}")
    print(f"\n[最优分类策略]")
    for i, step in enumerate(result['best_path'], 1):
        print(f"  Step {i}: {step}")

    print("\n" + "=" * 70)


def demonstrate_tot_vs_cot():
    """对比 ToT 和 CoT"""
    print("\n" + "=" * 70)
    print("ToT vs CoT 对比")
    print("=" * 70)

    print(r"""
    CoT (Chain-of-Thought):
      问题 → Thought1 → Thought2 → Thought3 → 答案

      局限:
        - 单一路径，无法探索替代方案
        - 一旦某步出错，无法回溯
        - 适合简单、确定性推理

    ToT (Tree of Thoughts):
           问题
          / | \
       T1a T1b T1c
       /|\   |   |
     T2a... T2b T2c
      |
     答案

      优势:
        - 多路径并行探索
        - 中间节点可评估、剪枝
        - 允许回溯，找到全局最优
        - 适合复杂、需要探索的决策问题

    母婴电商适用性:
      - 标签体系设计: 需要探索多种分类维度 → ToT
      - 评论分类策略: 需要对比多种策略效果 → ToT
      - 简单情感判断: 单一路径即可 → CoT 足够
    """)


if __name__ == "__main__":
    random.seed(42)
    demo_tot_label_design()
    demo_tot_classification_strategy()
    demonstrate_tot_vs_cot()

    print("\n生产环境建议:")
    print("  1. 使用真实 LLM API 进行 thought 生成和评估")
    print("  2. 限制搜索空间: 分支因子≤5, 深度≤4")
    print("  3. 缓存中间评估结果，避免重复计算")
    print("  4. 并行生成候选 thoughts，加速搜索")
    print("  5. 使用 Beam Search 平衡质量和成本")
    print("  6. 对实时场景，使用 A* 搜索替代 BFS/DFS")
