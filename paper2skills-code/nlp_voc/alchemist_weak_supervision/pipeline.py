"""ALCHEmist 完整流水线

整合: LLM 程序生成 → 验证筛选 → 批量标注 → 投票聚合
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aggregator import AggregationResult, MajorityVoteAggregator
from label_function import LFRegistry, LabelFunction
from program_generator import ProgramGenerator


@dataclass
class PipelineReport:
    """流水线执行报告"""

    label_name: str
    n_programs_generated: int
    n_programs_passed: int
    n_texts_labeled: int
    n_high_confidence: int
    n_uncertain: int
    avg_confidence: float
    label_distribution: dict


def _default_verify_set() -> tuple[list[str], list[str]]:
    """默认验证集（过敏反应标签）"""
    texts = [
        "宝宝用了后起红疹",
        "腰部一圈红红的",
        "大腿内侧发红",
        "红屁屁严重",
        "用了三天起小疙瘩",
        "皮肤敏感的宝宝慎买",
        "这个一用就红",
        "材质不适合敏感肌",
        "怀疑是过敏反应",
        "尺码偏小勒得不舒服",  # 非过敏
        "物流太慢了",           # 非过敏
        "面料太硬不舒服",       # 非过敏
        "价格有点贵",           # 非过敏
        "晚上漏尿严重",         # 非过敏
    ]
    labels = ["过敏反应"] * 9 + [None] * 5
    return texts, labels


class ALCHEmistPipeline:
    """ALCHEmist 完整流水线

    一键执行:
        1. 用 LLM 为标签生成多个标注程序
        2. 在验证集上评估程序质量，过滤低质量程序
        3. 用通过验证的程序批量标注未标注数据
        4. 多程序投票聚合，输出带置信度的标签
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.6,
        coverage_threshold: float = 0.2,
        min_votes: int = 2,
        model_name: str = "simulated",
    ):
        self.accuracy_threshold = accuracy_threshold
        self.coverage_threshold = coverage_threshold
        self.generator = ProgramGenerator(model_name=model_name)
        self.aggregator = MajorityVoteAggregator(min_votes=min_votes)

    def run(
        self,
        label_name: str,
        description: str,
        positive_examples: list[str],
        negative_examples: list[str],
        unlabeled_texts: list[str],
        verify_texts: Optional[list[str]] = None,
        verify_labels: Optional[list[str]] = None,
        n_programs: int = 5,
    ) -> tuple[list[AggregationResult], PipelineReport]:
        """执行 ALCHEmist 流水线

        Args:
            label_name: 目标标签名称
            description: 标签描述
            positive_examples: 正例文本（用于生成程序）
            negative_examples: 反例文本（用于生成程序）
            unlabeled_texts: 待标注的未标注文本
            verify_texts: 验证集文本（评估程序质量）
            verify_labels: 验证集标签
            n_programs: 生成程序数量

        Returns:
            (标注结果列表, 流水线报告)
        """
        print(f"[ALCHEmist] 开始为标签 '{label_name}' 生成标注程序")
        print(f"  正例: {len(positive_examples)} 条")
        print(f"  反例: {len(negative_examples)} 条")
        print(f"  待标注: {len(unlabeled_texts)} 条")

        # 1. 生成标注程序
        print(f"\n[Step 1] LLM 生成 {n_programs} 个标注程序...")
        programs = self.generator.generate(
            label_name=label_name,
            description=description,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            n_programs=n_programs,
        )
        print(f"  生成 {len(programs)} 个程序")

        # 2. 验证程序质量
        print(f"\n[Step 2] 验证程序质量 (准确率≥{self.accuracy_threshold}, 覆盖率≥{self.coverage_threshold})...")
        if verify_texts is None:
            verify_texts, verify_labels = _default_verify_set()

        valid_programs: list[LabelFunction] = []
        for prog in programs:
            stats = prog.evaluate(verify_texts, verify_labels)
            passed = (
                stats["accuracy"] >= self.accuracy_threshold
                and stats["coverage"] >= self.coverage_threshold
            )
            status = "✓ 通过" if passed else "✗ 过滤"
            print(f"  {prog.name}: 准确率={stats['accuracy']:.1%}, 覆盖率={stats['coverage']:.1%} {status}")
            if passed:
                valid_programs.append(prog)

        print(f"  通过验证: {len(valid_programs)}/{len(programs)}")

        if not valid_programs:
            print("[ALCHEmist] 警告: 没有程序通过验证，降低阈值重试...")
            self.accuracy_threshold *= 0.5
            return self.run(
                label_name, description, positive_examples,
                negative_examples, unlabeled_texts,
                verify_texts, verify_labels, n_programs,
            )

        # 3. 批量标注 + 聚合
        print(f"\n[Step 3] 批量标注 {len(unlabeled_texts)} 条文本...")
        results: list[AggregationResult] = []
        for text in unlabeled_texts:
            result = self.aggregator.aggregate(text, valid_programs)
            results.append(result)

        # 4. 生成报告
        high_conf = sum(1 for r in results if r.label is not None and r.confidence >= 0.5)
        uncertain = sum(1 for r in results if r.label is None)
        avg_conf = sum(r.confidence for r in results) / len(results) if results else 0

        label_dist: dict[str, int] = {}
        for r in results:
            if r.label:
                label_dist[r.label] = label_dist.get(r.label, 0) + 1

        report = PipelineReport(
            label_name=label_name,
            n_programs_generated=len(programs),
            n_programs_passed=len(valid_programs),
            n_texts_labeled=len(unlabeled_texts),
            n_high_confidence=high_conf,
            n_uncertain=uncertain,
            avg_confidence=avg_conf,
            label_distribution=label_dist,
        )

        print(f"\n[ALCHEmist] 流水线完成")
        print(f"  高置信度标注: {high_conf} ({high_conf/len(unlabeled_texts):.1%})")
        print(f"  不确定: {uncertain} ({uncertain/len(unlabeled_texts):.1%})")
        print(f"  平均置信度: {avg_conf:.2f}")

        return results, report

    def export_results(
        self,
        results: list[AggregationResult],
        report: PipelineReport,
        output_path: str,
    ) -> None:
        """导出结果到 JSON"""
        data = {
            "report": {
                "label_name": report.label_name,
                "n_programs_generated": report.n_programs_generated,
                "n_programs_passed": report.n_programs_passed,
                "n_texts_labeled": report.n_texts_labeled,
                "n_high_confidence": report.n_high_confidence,
                "n_uncertain": report.n_uncertain,
                "avg_confidence": report.avg_confidence,
                "label_distribution": report.label_distribution,
            },
            "results": [
                {
                    "text": r.text,
                    "label": r.label,
                    "confidence": r.confidence,
                    "vote_distribution": r.vote_distribution,
                }
                for r in results
            ],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ── 测试 ──────────────────────────────────────────────────────

def test_pipeline():
    """测试完整流水线"""
    print("=" * 60)
    print("测试: ALCHEmistPipeline")
    print("=" * 60)

    # 目标标签
    label_name = "过敏反应"
    description = "用户反馈使用后出现皮肤过敏症状"

    # 示例数据
    positive_examples = [
        "宝宝用了后起红疹，怀疑是过敏",
        "腰部一圈红红的，是不是过敏了",
        "大腿内侧发红，应该是过敏",
        "红屁屁严重，怀疑是过敏反应",
        "用了三天，屁股上起了小疙瘩",
        "之前用别的牌子没事，这个一用就红",
        "皮肤敏感的宝宝慎买",
        "材质可能不适合敏感肌",
    ]

    negative_examples = [
        "这个尺码偏小，勒得宝宝不舒服",
        "面料太硬了，摩擦得皮肤不舒服",
        "物流太慢了",
        "价格有点贵",
    ]

    # 待标注数据
    unlabeled_texts = [
        "宝宝用了后起红疹",
        "腰部一圈红红的",
        "皮肤敏感的宝宝慎买",
        "大腿内侧发红",
        "红屁屁严重",
        "用了三天起小疙瘩",
        "这个一用就红",
        "材质不适合敏感肌",
        "怀疑过敏反应",
        "尺码偏小不舒服",   # 非过敏
        "物流太慢",         # 非过敏
        "面料太硬",         # 非过敏
        "价格贵",           # 非过敏
        "晚上漏尿",         # 非过敏
    ]

    # 执行流水线
    pipeline = ALCHEmistPipeline(
        accuracy_threshold=0.5,
        coverage_threshold=0.1,
        min_votes=1,
        model_name="simulated",
    )

    results, report = pipeline.run(
        label_name=label_name,
        description=description,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        unlabeled_texts=unlabeled_texts,
        n_programs=5,
    )

    # 打印详细结果
    print("\n--- 标注结果 ---")
    for r in results:
        status = f"{r.label} (置信度: {r.confidence:.2f})" if r.label else "[不确定]"
        print(f"  '{r.text[:20]}...' -> {status}")

    # 导出
    pipeline.export_results(results, report, "/tmp/alchemist_output/results.json")
    print(f"\n--- 结果已导出到 /tmp/alchemist_output/results.json ---")

    print("\n" + "=" * 60)
    print("ALCHEmist 流水线测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
