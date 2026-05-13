"""迭代层级分类器

对文本进行 L1→L2→L3→L4 的逐层分类，记录未覆盖文本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from taxonomy_builder import TaxonomyNode, TaxonomyTree


@dataclass
class ClassificationResult:
    """单条分类结果"""

    text: str
    path: list[str]           # 节点ID路径 [L1, L2, L3, L4]
    path_names: list[str]     # 节点名称路径
    confidence: float         # 整体置信度
    is_covered: bool          # 是否被现有 Taxonomy 覆盖
    failed_at: int = 0        # 在哪个层级失败 (0=全部通过)


class IterativeClassifier:
    """迭代层级分类器

    逐层分类: L1 → L2 → L3 → L4
    每层使用关键词匹配（简化版，生产环境可用 embedding / BERT）。
    """

    def __init__(self, taxonomy: TaxonomyTree, confidence_threshold: float = 0.3):
        self.taxonomy = taxonomy
        self.confidence_threshold = confidence_threshold

        # 为每个节点预计算关键词（从名称和描述中提取）
        self._node_keywords: dict[str, set[str]] = {}
        self._build_keywords()

    def _build_keywords(self) -> None:
        """为每个节点构建关键词集合

        中文策略: 从名称和描述中提取 2-3 字滑动窗口作为关键词
        """
        for node in self.taxonomy.nodes.values():
            keywords = set()
            # 从名称提取 2-3 字滑动窗口
            for length in range(2, 4):
                for i in range(len(node.name) - length + 1):
                    keywords.add(node.name[i:i+length])
            # 从描述提取 2-3 字滑动窗口
            for length in range(2, 4):
                for i in range(len(node.description) - length + 1):
                    kw = node.description[i:i+length]
                    if any("\u4e00" <= c <= "\u9fff" for c in kw):
                        keywords.add(kw)
            self._node_keywords[node.id] = keywords

    def classify(self, text: str) -> ClassificationResult:
        """对单条文本进行层级分类"""
        text_lower = text.lower()
        path: list[str] = []
        path_names: list[str] = []
        current_level = 1
        current_parent_id: Optional[str] = None

        while current_level <= 4:
            # 获取当前层级的候选节点
            if current_level == 1:
                candidates = [self.taxonomy.nodes[rid] for rid in self.taxonomy.root_ids]
            else:
                if current_parent_id is None:
                    break
                candidates = self.taxonomy.get_children(current_parent_id)

            if not candidates:
                break

            # 匹配最佳节点
            best_node, best_score = self._match_level(text_lower, candidates)

            if best_score < self.confidence_threshold:
                # 当前层级未匹配成功
                return ClassificationResult(
                    text=text,
                    path=path,
                    path_names=path_names,
                    confidence=best_score,
                    is_covered=False,
                    failed_at=current_level,
                )

            path.append(best_node.id)
            path_names.append(best_node.name)
            current_parent_id = best_node.id
            current_level += 1

        # 全部层级通过
        confidence = self._compute_confidence(path, text_lower)
        return ClassificationResult(
            text=text,
            path=path,
            path_names=path_names,
            confidence=confidence,
            is_covered=True,
            failed_at=0,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """批量分类"""
        return [self.classify(t) for t in texts]

    def _match_level(self, text: str, candidates: list[TaxonomyNode]) -> tuple[TaxonomyNode, float]:
        """在候选节点中匹配最佳节点"""
        best_node = candidates[0]
        best_score = 0.0

        for node in candidates:
            keywords = self._node_keywords.get(node.id, set())
            score = 0.0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in text:
                    # 精确匹配（前后有空格）权重更高
                    score += 2.0 if f" {kw_lower} " in f" {text} " else 1.0

            if score > best_score:
                best_score = score
                best_node = node

        return best_node, best_score

    def _compute_confidence(self, path: list[str], text: str) -> float:
        """计算路径的整体置信度"""
        if not path:
            return 0.0
        # 简单平均：路径越深置信度越高
        return min(len(path) / 4.0, 1.0)

    def get_uncovered_texts(self, results: list[ClassificationResult]) -> list[str]:
        """获取未被覆盖的文本"""
        return [r.text for r in results if not r.is_covered]

    def get_coverage_stats(self, results: list[ClassificationResult]) -> dict:
        """获取覆盖率统计"""
        total = len(results)
        covered = sum(1 for r in results if r.is_covered)

        # 按层级统计失败
        fail_by_level: dict[int, int] = {}
        for r in results:
            if r.failed_at > 0:
                fail_by_level[r.failed_at] = fail_by_level.get(r.failed_at, 0) + 1

        return {
            "total": total,
            "covered": covered,
            "coverage_rate": covered / total if total > 0 else 0,
            "uncovered": total - covered,
            "fail_by_level": fail_by_level,
        }


# ── 测试 ──────────────────────────────────────────────────────

def test_classifier():
    print("=" * 60)
    print("测试: IterativeClassifier")
    print("=" * 60)

    from taxonomy_builder import create_mombaby_seed_taxonomy

    tree = create_mombaby_seed_taxonomy()
    classifier = IterativeClassifier(tree, confidence_threshold=0.5)

    test_texts = [
        "这个纸尿裤晚上总是侧漏，宝宝睡不好",           # 应匹配: 纸尿裤→质量→漏尿
        "奶粉溶解性不好，有结块",                        # 应匹配: 奶粉→质量→溶解性
        "防蚊贴贴了两小时就掉了，粘性太差",              # 应匹配: 防蚊产品→质量→粘性
        "物流太慢，清关等了两周",                        # 应匹配: 纸尿裤→物流→配送时效
        "这个驱蚊液味道很刺鼻，宝宝一直哭",              # 防蚊产品→质量→? (可能未覆盖)
        "宝宝用了后起红疹，怀疑过敏",                    # 可能未覆盖
    ]

    print("\n--- 单条分类 ---")
    for text in test_texts:
        result = classifier.classify(text)
        status = "✓ 覆盖" if result.is_covered else f"✗ 失败于 L{result.failed_at}"
        print(f"\n  文本: {text}")
        print(f"  路径: {' → '.join(result.path_names) if result.path_names else '[无]'} {status}")
        print(f"  置信度: {result.confidence:.2f}")

    print("\n--- 批量统计 ---")
    results = classifier.classify_batch(test_texts)
    stats = classifier.get_coverage_stats(results)
    print(f"  总计: {stats['total']}")
    print(f"  已覆盖: {stats['covered']} ({stats['coverage_rate']:.1%})")
    print(f"  未覆盖: {stats['uncovered']}")
    print(f"  各层级失败: {stats['fail_by_level']}")

    print("\n--- 未覆盖文本 ---")
    uncovered = classifier.get_uncovered_texts(results)
    for text in uncovered:
        print(f"  - {text}")

    print("\n" + "=" * 60)
    print("迭代分类器测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_classifier()
