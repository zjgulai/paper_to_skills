"""TaxoAdapt 完整流水线

整合: 种子 Taxonomy → 迭代分类 → 扩展候选发现 → 一致性校验 → 输出
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from expander import ConsistencyChecker, DepthExpander, WidthExpander
from iterative_classifier import ClassificationResult, IterativeClassifier
from taxonomy_builder import TaxonomyNode, TaxonomyTree


@dataclass
class TaxoAdaptReport:
    """流水线执行报告"""

    initial_nodes: int
    final_nodes: int
    width_expansions: int
    depth_expansions: int
    coverage_before: float
    coverage_after: float
    expansion_details: list[dict]
    execution_time_sec: float


def _generate_node_id(tree: TaxonomyTree, level: int) -> str:
    """生成新节点ID"""
    existing = [n for n in tree.nodes.values() if n.level == level]
    return f"L{level}-{len(existing) + 1:03d}"


class TaxoAdaptPipeline:
    """TaxoAdapt 完整流水线

    一键执行:
        1. 加载/生成种子 Taxonomy
        2. 迭代层级分类
        3. 分析未覆盖文本，触发扩展
        4. 一致性校验并应用扩展
        5. 输出扩展后的 Taxonomy + 报告
    """

    def __init__(
        self,
        width_threshold: float = 0.7,
        depth_threshold: float = 0.3,
        min_uncovered: int = 3,
        min_texts_for_depth: int = 5,
        max_iterations: int = 3,
    ):
        self.width_expander = WidthExpander(
            coverage_threshold=width_threshold,
            min_uncovered=min_uncovered,
        )
        self.depth_expander = DepthExpander(
            min_texts_for_split=min_texts_for_depth,
            diversity_threshold=depth_threshold,
        )
        self.checker = ConsistencyChecker()
        self.max_iterations = max_iterations

    def run(
        self,
        taxonomy: TaxonomyTree,
        texts: list[str],
    ) -> tuple[TaxonomyTree, TaxoAdaptReport]:
        """执行 TaxoAdapt 流水线

        Args:
            taxonomy: 初始种子 Taxonomy
            texts: 待分类和扩展分析的文本

        Returns:
            (扩展后的 Taxonomy, 执行报告)
        """
        start_time = datetime.now()
        initial_nodes = len(taxonomy.nodes)

        print(f"[TaxoAdapt] 开始执行")
        print(f"  初始节点: {initial_nodes}")
        print(f"  待分析文本: {len(texts)}")

        width_count = 0
        depth_count = 0
        expansion_details: list[dict] = []

        # 迭代优化
        for iteration in range(self.max_iterations):
            print(f"\n--- 迭代 {iteration + 1}/{self.max_iterations} ---")

            # 1. 分类
            classifier = IterativeClassifier(taxonomy)
            results = classifier.classify_batch(texts)
            stats = classifier.get_coverage_stats(results)
            coverage = stats["coverage_rate"]
            print(f"  覆盖率: {coverage:.1%} ({stats['covered']}/{stats['total']})")

            if coverage >= 0.95:
                print(f"  覆盖率达标，提前终止")
                break

            # 2. 按父节点分组未覆盖文本
            uncovered = [r for r in results if not r.is_covered]
            if not uncovered:
                print(f"  全部覆盖，提前终止")
                break

            # 按失败层级和父节点分组
            grouped: dict[tuple[int, Optional[str]], list[str]] = {}
            for r in uncovered:
                key = (r.failed_at, r.path[-1] if r.path else None)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(r.text)

            # 3. 尝试宽度扩展
            expanded_this_round = False
            for (failed_level, parent_id), unc_texts in grouped.items():
                if parent_id is None:
                    continue

                # 检查是否需要宽度扩展
                parent_texts = unc_texts
                total_at_parent = sum(1 for r in results if parent_id in r.path)

                if self.width_expander.should_expand(
                    parent_id, parent_texts, max(total_at_parent, len(parent_texts))
                ):
                    candidates = self.width_expander.expand(taxonomy, parent_id, parent_texts)
                    for candidate in candidates:
                        ok, msg = self.checker.validate_expansion(taxonomy, candidate)
                        if ok:
                            # 应用扩展
                            new_id = _generate_node_id(taxonomy, failed_level)
                            new_node = TaxonomyNode(
                                id=new_id,
                                name=candidate.suggested_name,
                                description=candidate.description,
                                parent_id=candidate.parent_id,
                                level=failed_level,
                            )
                            taxonomy.add_node(new_node)
                            width_count += 1
                            expanded_this_round = True
                            expansion_details.append({
                                "type": "width",
                                "node_id": new_id,
                                "name": candidate.suggested_name,
                                "parent_id": candidate.parent_id,
                                "confidence": candidate.confidence,
                            })
                            print(f"  [宽度扩展] {new_id}: {candidate.suggested_name} "
                                  f"(父: {taxonomy.get_node(candidate.parent_id).name})")

            # 4. 尝试深度扩展（对已有标签下的文本进行细分）
            # 按叶子节点分组文本
            leaf_texts: dict[str, list[str]] = {}
            for r in results:
                if r.is_covered and r.path:
                    leaf_id = r.path[-1]
                    if leaf_id not in leaf_texts:
                        leaf_texts[leaf_id] = []
                    leaf_texts[leaf_id].append(r.text)

            for leaf_id, l_texts in leaf_texts.items():
                if self.depth_expander.should_expand(l_texts):
                    candidates = self.depth_expander.expand(taxonomy, leaf_id, l_texts)
                    for candidate in candidates:
                        ok, msg = self.checker.validate_expansion(taxonomy, candidate)
                        if ok:
                            leaf = taxonomy.get_node(leaf_id)
                            new_id = _generate_node_id(taxonomy, leaf.level + 1)
                            new_node = TaxonomyNode(
                                id=new_id,
                                name=candidate.suggested_name,
                                description=candidate.description,
                                parent_id=leaf_id,
                                level=leaf.level + 1,
                            )
                            taxonomy.add_node(new_node)
                            depth_count += 1
                            expanded_this_round = True
                            expansion_details.append({
                                "type": "depth",
                                "node_id": new_id,
                                "name": candidate.suggested_name,
                                "parent_id": leaf_id,
                                "confidence": candidate.confidence,
                            })
                            print(f"  [深度扩展] {new_id}: {candidate.suggested_name} "
                                  f"(父: {leaf.name})")

            if not expanded_this_round:
                print(f"  本轮无扩展，终止迭代")
                break

        # 最终分类统计
        final_classifier = IterativeClassifier(taxonomy)
        final_results = final_classifier.classify_batch(texts)
        final_stats = final_classifier.get_coverage_stats(final_results)

        elapsed = (datetime.now() - start_time).total_seconds()

        report = TaxoAdaptReport(
            initial_nodes=initial_nodes,
            final_nodes=len(taxonomy.nodes),
            width_expansions=width_count,
            depth_expansions=depth_count,
            coverage_before=0.0,  # 简化：不追踪初始覆盖率
            coverage_after=final_stats["coverage_rate"],
            expansion_details=expansion_details,
            execution_time_sec=elapsed,
        )

        print(f"\n[TaxoAdapt] 执行完成")
        print(f"  新增节点: {report.final_nodes - report.initial_nodes}")
        print(f"  宽度扩展: {width_count}")
        print(f"  深度扩展: {depth_count}")
        print(f"  最终覆盖率: {report.coverage_after:.1%}")
        print(f"  耗时: {elapsed:.1f}s")

        return taxonomy, report

    def export(
        self,
        taxonomy: TaxonomyTree,
        report: TaxoAdaptReport,
        output_dir: str,
    ) -> None:
        """导出结果"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 导出 Taxonomy
        taxonomy.to_json(str(out / "taxonomy_evolved.json"))

        # 导出报告
        report_dict = {
            "initial_nodes": report.initial_nodes,
            "final_nodes": report.final_nodes,
            "width_expansions": report.width_expansions,
            "depth_expansions": report.depth_expansions,
            "coverage_after": report.coverage_after,
            "execution_time_sec": report.execution_time_sec,
            "expansion_details": report.expansion_details,
        }
        with open(out / "report.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        print(f"\n结果已导出到: {output_dir}")


# ── 流式演化扩展（EvoTaxo 思想）────────────────────────────────

@dataclass
class StreamingIngestResult:
    """单批数据摄入结果"""

    texts_processed: int
    coverage_rate: float
    uncovered_added: int
    buffer_size: int
    triggered: bool
    evolution: Optional[dict] = None


class StreamingTaxoAdapt:
    """流式 Taxonomy 演化器

    持续接收新文本流，积累未覆盖样本，当满足触发条件时自动扩展 Taxonomy。
    核心思想来自 EvoTaxo: 流式数据驱动分类体系自动演化。

    触发策略（满足任一即触发）:
      1. 缓冲区满 — 未覆盖文本积累到阈值
      2. 覆盖率漂移 — 连续多批覆盖率下降超过阈值
      3. 时间窗口 — 距上次演化超过最大间隔
    """

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        buffer_capacity: int = 50,
        coverage_drop_threshold: float = 0.15,
        min_evolve_interval_hours: float = 24.0,
        max_iterations: int = 3,
    ):
        self.taxonomy = taxonomy
        self.pipeline = TaxoAdaptPipeline(max_iterations=max_iterations)
        self.buffer: list[str] = []
        self.buffer_capacity = buffer_capacity
        self.coverage_drop_threshold = coverage_drop_threshold
        self.min_evolve_interval_hours = min_evolve_interval_hours
        self.history: list[dict] = []
        self.last_evolve_time: Optional[datetime] = None
        self.evolution_count = 0

    def ingest(self, texts: list[str], batch_tag: str = "") -> StreamingIngestResult:
        """接收一批新文本，分类并可能触发演化。

        Args:
            texts: 新到达的文本列表
            batch_tag: 批次标识（如日期 "2026-04-22"）

        Returns:
            StreamingIngestResult 包含处理统计和演化结果
        """
        # 1. 分类
        classifier = IterativeClassifier(self.taxonomy)
        results = classifier.classify_batch(texts)
        stats = classifier.get_coverage_stats(results)

        # 2. 收集未覆盖文本到缓冲区
        uncovered = [r.text for r in results if not r.is_covered]
        self.buffer.extend(uncovered)

        # 3. 记录历史
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "tag": batch_tag,
            "batch_size": len(texts),
            "coverage": stats["coverage_rate"],
            "uncovered": len(uncovered),
            "buffer_size": len(self.buffer),
        })

        # 4. 检查触发条件
        triggered = self._check_trigger(stats["coverage_rate"])
        evolution_info: Optional[dict] = None

        if triggered:
            evolution_info = self._evolve()
            self.evolution_count += 1

        return StreamingIngestResult(
            texts_processed=len(texts),
            coverage_rate=stats["coverage_rate"],
            uncovered_added=len(uncovered),
            buffer_size=len(self.buffer),
            triggered=triggered,
            evolution=evolution_info,
        )

    def _check_trigger(self, current_coverage: float) -> bool:
        """检查是否满足演化触发条件。"""
        # 条件1: 缓冲区满
        if len(self.buffer) >= self.buffer_capacity:
            return True

        # 条件2: 覆盖率连续下降超过阈值（漂移检测）
        if len(self.history) >= 3:
            coverages = [h["coverage"] for h in self.history[-3:]]
            if coverages[0] - coverages[-1] > self.coverage_drop_threshold:
                return True

        # 条件3: 时间窗口（距上次演化超过阈值）
        if self.last_evolve_time is not None:
            hours_since = (datetime.now() - self.last_evolve_time).total_seconds() / 3600
            if hours_since >= self.min_evolve_interval_hours and len(self.buffer) >= 10:
                return True

        return False

    def _evolve(self) -> dict:
        """执行 Taxonomy 演化扩展。"""
        print(f"\n[StreamingTaxoAdapt] 触发演化 (#{self.evolution_count + 1})")
        print(f"  缓冲区未覆盖文本: {len(self.buffer)}")

        nodes_before = len(self.taxonomy.nodes)

        # 用缓冲区文本触发 TaxoAdapt 扩展
        self.taxonomy, report = self.pipeline.run(self.taxonomy, self.buffer)

        # 清空缓冲区
        self.buffer = []
        self.last_evolve_time = datetime.now()

        result = {
            "nodes_before": nodes_before,
            "nodes_after": len(self.taxonomy.nodes),
            "width_expansions": report.width_expansions,
            "depth_expansions": report.depth_expansions,
            "coverage_after": report.coverage_after,
            "timestamp": self.last_evolve_time.isoformat(),
        }

        print(f"  演化完成: +{result['nodes_after'] - result['nodes_before']} 节点")
        return result

    def get_drift_report(self) -> dict:
        """获取数据漂移检测报告。

        分析历史覆盖率趋势，判断 Taxonomy 是否与当前数据分布脱节。
        """
        if len(self.history) < 2:
            return {"status": "insufficient_data", "message": "至少需要2批数据"}

        coverages = [h["coverage"] for h in self.history]
        avg_coverage = sum(coverages) / len(coverages)
        latest = coverages[-1]

        # 趋势判断
        if len(coverages) >= 3:
            trend = "decreasing" if coverages[-1] < coverages[-3] else "stable"
        else:
            trend = "unknown"

        # 漂移状态
        drift_detected = (
            trend == "decreasing"
            and latest < avg_coverage - self.coverage_drop_threshold
        )

        return {
            "status": "drift_detected" if drift_detected else "stable",
            "coverage_trend": trend,
            "current_coverage": latest,
            "avg_coverage": round(avg_coverage, 3),
            "total_batches": len(self.history),
            "total_evolutions": self.evolution_count,
            "buffer_size": len(self.buffer),
            "total_uncovered_ever": sum(h["uncovered"] for h in self.history),
            "last_evolve": self.last_evolve_time.isoformat() if self.last_evolve_time else None,
        }

    def export_state(self, output_dir: str) -> None:
        """导出当前流式演化器状态（Taxonomy + 历史 + 缓冲区）。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 导出 Taxonomy
        self.taxonomy.to_json(str(out / "taxonomy_current.json"))

        # 导出历史
        with open(out / "ingest_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        # 导出缓冲区
        with open(out / "buffer.json", "w", encoding="utf-8") as f:
            json.dump(self.buffer, f, ensure_ascii=False, indent=2)

        print(f"流式状态已导出到: {output_dir}")


# ── 测试 ──────────────────────────────────────────────────────

def test_pipeline():
    print("=" * 60)
    print("测试: TaxoAdaptPipeline")
    print("=" * 60)

    from taxonomy_builder import create_mombaby_seed_taxonomy

    tree = create_mombaby_seed_taxonomy()
    print(f"\n初始 Taxonomy: {tree}")

    # 模拟春季新品评论（包含现有标签无法覆盖的内容）
    texts = [
        # 已有标签可覆盖
        "这个纸尿裤晚上总是侧漏",
        "奶粉溶解性不好，有结块",
        "物流太慢，清关等了两周",
        "防蚊贴贴了两小时就掉了",

        # 新痛点：气味问题（宽度扩展候选）
        "驱蚊液味道很刺鼻，宝宝一直哭",
        "这个防蚊喷雾气味太浓了",
        "味道太冲，不敢给宝宝用",
        "香味太重，宝宝打喷嚏",
        "气味刺鼻，换了一个牌子",

        # 新痛点：驱蚊效果不佳（宽度扩展候选）
        "晚上还是有蚊子，效果一般",
        "驱蚊效果不太好，还是被咬了",
        "蚊子照样咬，没什么用",

        # 漏尿细分：夜间漏 vs 量大漏（深度扩展候选）
        "晚上总是侧漏，床单都湿了",
        "夜间漏尿严重，宝宝醒了",
        "半夜漏尿，睡不好",
        "量大的时候漏出来",
        "尿量多的时候兜不住",
        "白天没事，晚上漏",

        # 尺码细分：偏大 vs 偏小（深度扩展候选）
        "尺码偏大，穿着松松垮垮",
        "太大了，容易漏",
        "尺码偏小，勒得不舒服",
        "太小了，宝宝腿上有勒痕",
    ]

    pipeline = TaxoAdaptPipeline(
        width_threshold=0.6,
        depth_threshold=0.25,
        min_uncovered=2,
        min_texts_for_depth=3,
        max_iterations=3,
    )

    evolved_tree, report = pipeline.run(tree, texts)

    print("\n--- 扩展后的叶子标签 ---")
    leaves = evolved_tree.get_leaves()
    for leaf in leaves:
        path = evolved_tree.get_path(leaf.id)
        print(f"  {' → '.join(n.name for n in path)}")

    # 导出
    pipeline.export(evolved_tree, report, "/tmp/taxoadapt_output")

    print("\n" + "=" * 60)
    print("TaxoAdapt 流水线测试完成 ✓")
    print("=" * 60)


def test_streaming():
    """测试流式演化"""
    print("\n" + "=" * 60)
    print("测试: StreamingTaxoAdapt")
    print("=" * 60)

    from taxonomy_builder import create_mombaby_seed_taxonomy

    taxonomy = create_mombaby_seed_taxonomy()
    print(f"\n初始 Taxonomy: {taxonomy}")

    # 初始化流式演化器（小缓冲区便于测试触发）
    streamer = StreamingTaxoAdapt(
        taxonomy=taxonomy,
        buffer_capacity=8,  # 小阈值便于触发
        coverage_drop_threshold=0.2,
        min_evolve_interval_hours=0.001,  # 几乎立即允许时间触发
        max_iterations=2,
    )

    # 模拟多批数据流入
    batches = [
        # 批次1: 现有标签可覆盖大部分
        {
            "tag": "Day1",
            "texts": [
                "纸尿裤晚上侧漏严重",
                "奶粉有结块，溶解不好",
                "物流太慢，等了两周",
                "尺码偏大，穿着松垮",
            ],
        },
        # 批次2: 出现新主题——气味问题（未覆盖）
        {
            "tag": "Day2",
            "texts": [
                "驱蚊液味道很刺鼻",
                "气味太浓，宝宝打喷嚏",
                "香味太重，不敢用",
                "味道刺鼻，换了个牌子",
            ],
        },
        # 批次3: 更多气味 + 新主题——驱蚊效果
        {
            "tag": "Day3",
            "texts": [
                "还是有蚊子咬，效果不行",
                "驱蚊效果不好",
                "味道还是太大",
                "晚上蚊子照样多",
            ],
        },
        # 批次4: 触发演化后，新文本应被更好覆盖
        {
            "tag": "Day4-PostEvolve",
            "texts": [
                "这个驱蚊喷雾气味很冲",
                "防蚊效果一般，还是有包",
            ],
        },
    ]

    print("\n--- 模拟流式数据摄入 ---")
    for batch in batches:
        result = streamer.ingest(batch["texts"], batch_tag=batch["tag"])
        status = "🔄 触发演化" if result.triggered else "📥 缓冲中"
        print(f"  [{batch['tag']}] 处理 {result.texts_processed} 条, "
              f"覆盖率 {result.coverage_rate:.0%}, "
              f"未覆盖 +{result.uncovered_added}, "
              f"缓冲区 {result.buffer_size} {status}")
        if result.evolution:
            ev = result.evolution
            print(f"    → 演化: {ev['nodes_before']} → {ev['nodes_after']} 节点 "
                  f"(+{ev['width_expansions']}宽 +{ev['depth_expansions']}深)")

    # 漂移报告
    print("\n--- 漂移检测报告 ---")
    drift = streamer.get_drift_report()
    for k, v in drift.items():
        print(f"  {k}: {v}")

    # 验证演化后的覆盖率提升
    print("\n--- 演化后 Taxonomy ---")
    print(f"最终: {streamer.taxonomy}")
    leaves = streamer.taxonomy.get_leaves()
    print(f"叶子节点 ({len(leaves)} 个):")
    for leaf in leaves:
        path = streamer.taxonomy.get_path(leaf.id)
        print(f"  {' → '.join(n.name for n in path)}")

    print("\n" + "=" * 60)
    print("流式演化测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
    test_streaming()
