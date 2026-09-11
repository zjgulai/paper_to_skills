"""EComStage: 电商 Agent 三阶段(Perception/Planning/Action) + 双向评估 benchmark.

参考论文:Zhao, K. et al. (2026) EComStage: Stage-wise and Orientation-specific
Benchmarking for Large Language Models in E-commerce. arxiv:2601.02752.

本实现是简化版评估管道:
- close-ended 任务:accuracy
- open-ended 任务:token-vector cosine similarity 替代真实 embedding(生产用 Qwen3-Embedding-8B)
- 7 任务的数据结构 + 三阶段/双向报告

生产环境:接入真实 LLM API 跑推理,把 cosine sim 升级为真实 embedding 向量计算.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional


# Stage / orientation enums (用字符串避免依赖) ---------------------------------

PERCEPTION = "perception"
PLANNING = "planning"
ACTION = "action"

CUSTOMER = "customer"
MERCHANT = "merchant"
BOTH = "both"


# Data structures -----------------------------------------------------------


@dataclass
class TaskSample:
    sample_id: str
    task_name: str
    stage: str
    orientation: str
    input: dict  # 例: {"query": "...", "history": [...]}
    reference: str  # ground truth answer or label
    options: Optional[list[str]] = None  # close-ended 任务的选项


@dataclass
class Prediction:
    sample_id: str
    output: str


@dataclass
class TaskResult:
    task_name: str
    stage: str
    orientation: str
    metric: str  # "accuracy" 或 "cosine"
    score: float
    n_samples: int


@dataclass
class StageReport:
    perception: dict[str, TaskResult] = field(default_factory=dict)
    planning: dict[str, TaskResult] = field(default_factory=dict)
    action: dict[str, TaskResult] = field(default_factory=dict)

    def overall(self) -> float:
        all_results = list(self.perception.values()) + list(self.planning.values()) + list(self.action.values())
        if not all_results:
            return 0.0
        return sum(r.score for r in all_results) / len(all_results)

    def stage_average(self, stage: str) -> float:
        bucket = {"perception": self.perception, "planning": self.planning, "action": self.action}[stage]
        if not bucket:
            return 0.0
        return sum(r.score for r in bucket.values()) / len(bucket)

    def orientation_average(self, orientation: str) -> float:
        all_results = list(self.perception.values()) + list(self.planning.values()) + list(self.action.values())
        relevant = [r for r in all_results if r.orientation == orientation or r.orientation == BOTH]
        if not relevant:
            return 0.0
        return sum(r.score for r in relevant) / len(relevant)


# Evaluator ----------------------------------------------------------------


class Evaluator:
    """支持 close-ended (accuracy) 与 open-ended (cosine sim) 两种任务."""

    OPEN_ENDED_TASKS = {"query_rewrite", "rag_qa"}

    def evaluate_sample(self, sample: TaskSample, prediction: Prediction) -> float:
        if sample.task_name in self.OPEN_ENDED_TASKS:
            return self._cosine_sim(prediction.output, sample.reference)
        # close-ended:精确匹配(忽略大小写与空白)
        return 1.0 if self._normalize(prediction.output) == self._normalize(sample.reference) else 0.0

    def evaluate_task(self, samples: list[TaskSample], predictions: dict[str, Prediction]) -> TaskResult:
        if not samples:
            return TaskResult(task_name="empty", stage=PERCEPTION, orientation=CUSTOMER, metric="accuracy", score=0.0, n_samples=0)
        task_name = samples[0].task_name
        metric = "cosine" if task_name in self.OPEN_ENDED_TASKS else "accuracy"
        scores = [
            self.evaluate_sample(s, predictions[s.sample_id])
            for s in samples
            if s.sample_id in predictions
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        return TaskResult(
            task_name=task_name,
            stage=samples[0].stage,
            orientation=samples[0].orientation,
            metric=metric,
            score=avg * 100,  # 百分比,与论文 Table 3 一致
            n_samples=len(samples),
        )

    @staticmethod
    def _cosine_sim(a: str, b: str) -> float:
        ca = Counter(re.findall(r"[a-z0-9]+|[一-鿿]", a.lower()))
        cb = Counter(re.findall(r"[a-z0-9]+|[一-鿿]", b.lower()))
        if not ca or not cb:
            return 0.0
        dot = sum(ca[k] * cb[k] for k in ca)
        norm_a = math.sqrt(sum(v * v for v in ca.values()))
        norm_b = math.sqrt(sum(v * v for v in cb.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()


# Benchmark runner ----------------------------------------------------------


@dataclass
class EComStageBenchmark:
    samples: list[TaskSample] = field(default_factory=list)
    evaluator: Evaluator = field(default_factory=Evaluator)

    def add(self, sample: TaskSample) -> None:
        self.samples.append(sample)

    def by_task(self) -> dict[str, list[TaskSample]]:
        bucket: dict[str, list[TaskSample]] = defaultdict(list)
        for sample in self.samples:
            bucket[sample.task_name].append(sample)
        return dict(bucket)

    def run(self, model_fn: Callable[[TaskSample], str]) -> StageReport:
        predictions = {s.sample_id: Prediction(s.sample_id, model_fn(s)) for s in self.samples}
        report = StageReport()
        for task_name, task_samples in self.by_task().items():
            result = self.evaluator.evaluate_task(task_samples, predictions)
            bucket = {
                PERCEPTION: report.perception,
                PLANNING: report.planning,
                ACTION: report.action,
            }[result.stage]
            bucket[task_name] = result
        return report


# Demo: 模拟跨境母婴电商场景的 7 任务 -------------------------------------------


def _demo_samples() -> list[TaskSample]:
    return [
        # Perception - Query Rewrite (Customer, open-ended)
        TaskSample(
            sample_id="qr_001",
            task_name="query_rewrite",
            stage=PERCEPTION,
            orientation=CUSTOMER,
            input={"history": ["怎么退?", "您好,请问要退什么呢?"], "last": "上次买的奶嘴"},
            reference="客户希望退掉上次购买的奶嘴",
        ),
        # Perception - Attitude Classification (Merchant, close-ended)
        TaskSample(
            sample_id="ac_001",
            task_name="attitude_classification",
            stage=PERCEPTION,
            orientation=MERCHANT,
            input={"merchant_msg": "客户这种问题我们已经回答过 N 次了,真烦"},
            reference="negative",
            options=["positive", "neutral", "negative"],
        ),
        # Perception - Query Match (Customer, close-ended)
        TaskSample(
            sample_id="qm_001",
            task_name="query_match",
            stage=PERCEPTION,
            orientation=CUSTOMER,
            input={"query": "纸尿裤的尺码怎么选", "candidates": ["尺码选择", "退货政策", "物流时效"]},
            reference="尺码选择",
        ),
        # Perception - Intent Recognition (Customer, close-ended)
        TaskSample(
            sample_id="ir_001",
            task_name="intent_recognition",
            stage=PERCEPTION,
            orientation=CUSTOMER,
            input={"message": "宝宝吃了奶粉过敏怎么办?"},
            reference="refund_request",
            options=["product_inquiry", "refund_request", "complaint", "other"],
        ),
        # Planning - Scenario Route (Merchant, close-ended)
        TaskSample(
            sample_id="sr_001",
            task_name="scenario_route",
            stage=PLANNING,
            orientation=MERCHANT,
            input={"chat_history": ["商家:广告被拒怎么回事?", "客服:让我看一下..."]},
            reference="content_rejection",
            options=["content_rejection", "promotion_complaint", "refund_request"],
        ),
        # Action - Solution Decision (Customer, close-ended)
        TaskSample(
            sample_id="sd_001",
            task_name="solution_decision",
            stage=ACTION,
            orientation=CUSTOMER,
            input={"query": "宝宝过敏需要退货", "options": ["initiate_return", "ask_for_photo", "ask_human"]},
            reference="ask_for_photo",
            options=["initiate_return", "ask_for_photo", "ask_human"],
        ),
        # Action - RAG-QA (Both, open-ended)
        TaskSample(
            sample_id="rq_001",
            task_name="rag_qa",
            stage=ACTION,
            orientation=BOTH,
            input={"question": "母婴产品过敏多久内可以退货?", "context": "过敏退货需在 30 天内,提供症状照片和批次号."},
            reference="30 天内,需提供症状照片和批次号",
        ),
    ]


def _mock_strong_model(sample: TaskSample) -> str:
    """模拟一个 Claude/GPT 级别的强模型:基本上都答对."""
    return sample.reference


def _mock_weak_model(sample: TaskSample) -> str:
    """模拟一个小模型/弱模型:Perception 任务还行,Action 任务一半错."""
    if sample.stage == ACTION and "rag" not in sample.task_name:
        return "ask_human"  # 弱模型常 fallback 到人工
    if sample.task_name == "rag_qa":
        return "30 天内可退"  # 漏了细节
    return sample.reference


def _format_report(name: str, report: StageReport) -> str:
    lines = [f"\n=== {name} ==="]
    for stage_name, bucket in [("Perception", report.perception), ("Planning", report.planning), ("Action", report.action)]:
        lines.append(f"\n[{stage_name}]")
        for task, r in bucket.items():
            lines.append(f"  {task:25s} score={r.score:6.2f} ({r.metric}, n={r.n_samples}, orient={r.orientation})")
    lines.append(f"\nOverall avg: {report.overall():.2f}")
    lines.append(f"Stage avg:   Perception={report.stage_average('perception'):.2f}, "
                 f"Planning={report.stage_average('planning'):.2f}, "
                 f"Action={report.stage_average('action'):.2f}")
    lines.append(f"Orient avg:  Customer={report.orientation_average(CUSTOMER):.2f}, "
                 f"Merchant={report.orientation_average(MERCHANT):.2f}")
    return "\n".join(lines)


def main() -> None:
    bench = EComStageBenchmark()
    for sample in _demo_samples():
        bench.add(sample)

    strong_report = bench.run(_mock_strong_model)
    weak_report = bench.run(_mock_weak_model)

    print(_format_report("Strong Model (mock Claude)", strong_report))
    print(_format_report("Weak Model (mock 3B baseline)", weak_report))


def test_pipeline() -> None:
    bench = EComStageBenchmark()
    for sample in _demo_samples():
        bench.add(sample)

    # Strong model 应该接近满分
    strong_report = bench.run(_mock_strong_model)
    assert strong_report.overall() >= 95.0, f"strong model overall must be high, got {strong_report.overall()}"
    assert strong_report.stage_average("perception") >= 95.0
    assert strong_report.stage_average("action") >= 90.0

    # Weak model 应该在 action 阶段明显较差
    weak_report = bench.run(_mock_weak_model)
    assert weak_report.stage_average("perception") > weak_report.stage_average("action"), (
        "weak model should be worse on action than perception"
    )
    assert weak_report.overall() < strong_report.overall(), "weak < strong"

    # 评估器单元
    evaluator = Evaluator()
    open_score = evaluator._cosine_sim("30 天内可退", "30 天内,需提供症状照片和批次号")
    assert 0 < open_score < 1, "open-ended partial overlap should produce 0 < score < 1"

    closed_score = evaluator.evaluate_sample(
        TaskSample("t", "intent_recognition", PERCEPTION, CUSTOMER, {}, "refund_request"),
        Prediction("t", "refund_request"),
    )
    assert closed_score == 1.0

    # by_task / orientation 接口
    task_buckets = bench.by_task()
    assert len(task_buckets) == 7, "demo provides 7 tasks"
    assert weak_report.orientation_average(MERCHANT) <= 100.0
    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    main()
