"""VOC Proxy NPS × AIPL 全旅程指标落地工作流

端到端编排: VOC 数据 → 质量筛选 → 统一标签萃取 → 指标看板

整合 Skills:
- ReviewQuality-Scoring: Phase 2 质量筛选
- UnifiedLabelExtraction: Phase 3 统一打标
- DashboardGenerator: Phase 4 指标计算

Usage:
    from workflow import VOCProxyNPSWorkflow

    workflow = VOCProxyNPSWorkflow(
        tag_dict_path="tag_seeds.csv",
        quality_threshold=60.0,
    )
    results = workflow.run(voc_records)
    dashboard = workflow.generate_dashboard(results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from unified_label_extraction import (
    DashboardData,
    DashboardGenerator,
    TagSeedDictionary,
    UnifiedLabelingPipeline,
    VOCRecord,
    VOCLabelExtraction,
)


@dataclass
class WorkflowResults:
    """工作流完整输出"""

    total_input: int
    filtered_out: int
    labeled_count: int
    extractions: list[VOCLabelExtraction]
    dashboard: Optional[DashboardData] = None
    quality_report: Optional[dict] = None
    processing_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input": self.total_input,
            "filtered_out": self.filtered_out,
            "labeled_count": self.labeled_count,
            "quality_report": self.quality_report,
            "processing_stats": self.processing_stats,
            "dashboard": self.dashboard.to_dict() if self.dashboard else None,
        }

    def save(self, output_dir: str) -> None:
        """保存工作流结果到目录"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. 萃取结果
        extractions_path = out / "extractions.json"
        with open(extractions_path, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self.extractions],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 2. 看板数据
        if self.dashboard:
            dashboard_path = out / "dashboard.json"
            self.dashboard.to_json(str(dashboard_path))

        # 3. 工作流摘要
        summary_path = out / "workflow_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class VOCProxyNPSWorkflow:
    """VOC Proxy NPS × AIPL 全旅程指标落地工作流

    四阶段流水线:
    1. 加载标签字典
    2. 质量筛选（可选）
    3. 统一标签萃取
    4. 指标看板生成
    """

    def __init__(
        self,
        tag_dict_path: Optional[str] = None,
        tag_dict: Optional[TagSeedDictionary] = None,
        quality_pipeline=None,  # 外部注入的质量流水线
        quality_threshold: float = 60.0,
        enable_quality_filter: bool = True,
        enable_spam_detection: bool = True,
    ):
        """
        Args:
            tag_dict_path: 标签种子 CSV/xlsx 路径（与 tag_dict 二选一）
            tag_dict: 已加载的标签字典（与 tag_dict_path 二选一）
            quality_pipeline: 外部注入的质量流水线（可选，兼容任何有 process(text, rating) 方法的对象）
            quality_threshold: 质量分阈值
            enable_quality_filter: 是否启用质量筛选
            enable_spam_detection: 是否启用虚假检测
        """
        if tag_dict:
            self.tag_dict = tag_dict
        elif tag_dict_path:
            if tag_dict_path.endswith(".xlsx"):
                self.tag_dict = TagSeedDictionary.from_xlsx(tag_dict_path)
            else:
                self.tag_dict = TagSeedDictionary.from_csv(tag_dict_path)
        else:
            from unified_label_extraction import create_demo_tag_dictionary
            self.tag_dict = create_demo_tag_dictionary()

        self.quality_threshold = quality_threshold
        self.enable_quality_filter = enable_quality_filter
        self.enable_spam_detection = enable_spam_detection
        self._external_quality_pipeline = quality_pipeline

        # 延迟初始化质量流水线（避免循环导入）
        self._quality_pipeline = None

    def _get_quality_pipeline(self):
        """获取质量流水线（优先外部注入，其次懒加载）"""
        if self._external_quality_pipeline is not None:
            return self._external_quality_pipeline
        if self._quality_pipeline is None:
            try:
                import sys
                rqs_path = Path(__file__).parent.parent / "review_quality_scoring"
                if str(rqs_path) not in sys.path:
                    sys.path.insert(0, str(rqs_path))
                from pipeline import ReviewQualityPipeline
                self._quality_pipeline = ReviewQualityPipeline(
                    quality_threshold=self.quality_threshold,
                )
            except ImportError:
                self._quality_pipeline = None
        return self._quality_pipeline

    def run(self, vocs: list[VOCRecord]) -> WorkflowResults:
        """执行完整工作流

        Args:
            vocs: VOC 原始数据列表

        Returns:
            WorkflowResults 包含萃取结果和看板数据
        """
        total = len(vocs)
        stats: dict[str, Any] = {
            "phase": {},
            "timestamps": {},
        }

        # Phase 1: 标签字典已加载
        stats["phase"]["tag_dictionary"] = {
            "total_tags": len(self.tag_dict.get_all()),
            "summary": self.tag_dict.summary(),
        }

        # Phase 2: 质量筛选
        quality_pipeline = None
        if self.enable_quality_filter:
            quality_pipeline = self._get_quality_pipeline()

        # Phase 3: 统一标签萃取
        labeling_pipeline = UnifiedLabelingPipeline(
            tag_dict=self.tag_dict,
            quality_pipeline=quality_pipeline,
        )
        extractions = labeling_pipeline.process(vocs)

        # 统计过滤
        suspicious_count = sum(1 for e in extractions if e.is_suspicious)
        low_quality_count = sum(
            1 for e in extractions
            if not e.is_suspicious and e.quality_score > 0 and e.quality_score < self.quality_threshold
        )
        filtered = suspicious_count + low_quality_count

        # Phase 4: 指标看板
        dashboard = DashboardGenerator().build(extractions)

        # 质量报告
        quality_report = None
        if quality_pipeline:
            quality_report = {
                "threshold": self.quality_threshold,
                "suspicious_count": suspicious_count,
                "low_quality_count": low_quality_count,
                "pass_rate": (total - filtered) / total if total else 0,
            }

        stats["phase"]["extraction"] = {
            "total_processed": total,
            "suspicious_filtered": suspicious_count,
            "low_quality_filtered": low_quality_count,
        }

        return WorkflowResults(
            total_input=total,
            filtered_out=filtered,
            labeled_count=total - filtered,
            extractions=extractions,
            dashboard=dashboard,
            quality_report=quality_report,
            processing_stats=stats,
        )

    def generate_dashboard(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> DashboardData:
        """从萃取结果生成看板"""
        return DashboardGenerator().build(extractions)

    def get_tag_dictionary_summary(self) -> dict[str, Any]:
        """获取标签字典摘要"""
        return self.tag_dict.summary()


# ── 便捷入口 ────────────────────────────────────────────────────

def run_voc_proxy_nps_workflow(
    vocs: list[VOCRecord],
    tag_dict_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> WorkflowResults:
    """一键运行 VOC Proxy NPS 工作流

    Args:
        vocs: VOC 数据列表
        tag_dict_path: 标签种子 CSV 路径（None 使用演示数据）
        output_dir: 输出目录（None 不保存）

    Returns:
        WorkflowResults
    """
    workflow = VOCProxyNPSWorkflow(tag_dict_path=tag_dict_path)
    results = workflow.run(vocs)

    if output_dir:
        results.save(output_dir)

    return results


# ── 测试 ────────────────────────────────────────────────────────

def test_workflow():
    print("=" * 70)
    print("测试: VOCProxyNPSWorkflow")
    print("=" * 70)

    from unified_label_extraction import create_demo_tag_dictionary

    # 创建测试数据
    test_vocs = [
        VOCRecord(
            review_id="REV001",
            text=(
                "I was searching for a wearable pump and came across Momcozy on TikTok. "
                "Compared it with Willow and Elvie, the price is much more affordable. "
                "However, the flange size is too small and the suction feels weak. "
                "Customer service was slow to respond. Would not recommend to friends."
            ),
            source_type="review",
            platform="amazon",
            spu_code="SPU001",
            product_line="breast_pump",
            category="wearable_pump",
            rating=2.0,
        ),
        VOCRecord(
            review_id="REV002",
            text=(
                "This is my second purchase! The pump is so comfortable, "
                "I forget I'm wearing it. Highly recommend to all new moms. "
                "Way better than my old Spectra."
            ),
            source_type="trustpilot",
            platform="dtc",
            spu_code="SPU002",
            product_line="breast_pump",
            category="wearable_pump",
            rating=5.0,
        ),
        VOCRecord(
            review_id="REV003",
            text=(
                "First time mom here, feeling anxious about breastfeeding. "
                "Saw this on Instagram and decided to try. "
                "It's a bit noisy but gets the job done. "
                "Shipping took two weeks though."
            ),
            source_type="review",
            platform="amazon",
            spu_code="SPU003",
            product_line="breast_pump",
            category="wearable_pump",
            rating=4.0,
        ),
        VOCRecord(
            review_id="REV004",
            text="Great product, fast shipping. Love it!",
            source_type="review",
            platform="amazon",
            spu_code="SPU004",
            product_line="breast_pump",
            category="wearable_pump",
            rating=5.0,
        ),
    ]

    # 使用演示标签字典运行
    tag_dict = create_demo_tag_dictionary()
    # 工作流测试：关闭质量筛选，专注标签萃取
    workflow = VOCProxyNPSWorkflow(
        tag_dict=tag_dict,
        enable_quality_filter=False,
    )

    print("\n--- 标签字典摘要 ---")
    summary = workflow.get_tag_dictionary_summary()
    print(f"  总标签数: {summary['total_tags']}")
    print(f"  AIPL分布: {summary['by_aipl']}")

    print("\n--- 执行工作流 ---")
    results = workflow.run(test_vocs)

    print(f"  输入: {results.total_input}")
    print(f"  过滤: {results.filtered_out}")
    print(f"  有效: {results.labeled_count}")

    print("\n--- 萃取结果示例 ---")
    for e in results.extractions[:2]:
        print(f"\n  [{e.review_id}] AIPL={e.aipl_stage}")
        print(f"    标签: {[t.tag_en for t in e.aipl_tags]}")
        print(f"    画像: {e.persona_derived}")
        print(f"    情感: {e.sentiment_polarity:+.2f}")
        print(f"    NPS:  {e.proxy_nps_contribution}")

    print("\n--- 看板数据 ---")
    if results.dashboard:
        d = results.dashboard.to_dict()
        nps = d["proxy_nps"]["overall"]
        print(f"  Proxy NPS: {nps['proxy_nps']:.1f}")
        print(f"    推荐者: {nps['promoters']} ({nps['promoter_pct']}%)")
        print(f"    贬损者: {nps['detractors']} ({nps['detractor_pct']}%)")

        print(f"\n  AIPL 漏斗:")
        for node, info in d["aipl_funnel"].items():
            if info["count"] > 0:
                print(f"    {node}: {info['count']} 条")

        print(f"\n  画像分布:")
        for persona, info in d["persona_insights"].items():
            print(f"    {persona}: {info['penetration']:.1%} 渗透率, NPS={info['proxy_nps']['proxy_nps']:.1f}")

    # 保存测试
    print("\n--- 保存结果 ---")
    results.save("/tmp/voc_workflow_output")
    print("✓ 结果已保存到 /tmp/voc_workflow_output/")

    # 验证
    assert results.total_input == 4
    assert results.labeled_count == 4  # 关闭质量筛选，全部通过
    assert results.dashboard is not None
    print("\n✓ 工作流测试通过")

    print("\n" + "=" * 70)
    print("VOCProxyNPSWorkflow 测试完成 ✓")
    print("=" * 70)

    return results


if __name__ == "__main__":
    test_workflow()
