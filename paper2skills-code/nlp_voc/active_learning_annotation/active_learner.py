"""主动学习循环 + 不确定性采样

实现 ALCHEmist 的核心：选择"最值得人工验证"的样本。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from annotator import AnnotationResult, LLMAnnotator


@dataclass
class LabeledSample:
    """已标注样本"""

    text: str
    label: str
    source: str  # "llm" | "human"
    confidence: str = ""
    uncertainty_score: float = 0.0


class UncertaintySampler:
    """不确定性采样器

    三种策略：
        - least_confidence: 最低置信度
        - margin: 最小编距（Top2 概率差最小）
        - entropy: 最大熵
    """

    def __init__(self, strategy: str = "entropy"):
        if strategy not in ("least_confidence", "margin", "entropy"):
            raise ValueError(f"未知策略: {strategy}")
        self.strategy = strategy

    def score(self, predictions: list[dict]) -> list[float]:
        """计算每条样本的不确定性分数（越高越不确定）

        Args:
            predictions: 每条样本的预测分布 [{label1: prob1, label2: prob2, ...}, ...]

        Returns:
            不确定性分数列表（越高越需要人工验证）
        """
        scores = []
        for pred_dist in predictions:
            probs = list(pred_dist.values())
            if not probs:
                scores.append(0.0)
                continue

            if self.strategy == "least_confidence":
                score = 1.0 - max(probs)
            elif self.strategy == "margin":
                sorted_probs = sorted(probs, reverse=True)
                score = 1.0 - (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
            else:  # entropy
                score = -sum(p * math.log(p + 1e-10) for p in probs)
            scores.append(score)

        return scores

    def select(
        self,
        predictions: list[dict],
        n_samples: int,
    ) -> list[int]:
        """选择 Top-N 最不确定的样本索引"""
        scores = self.score(predictions)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in indexed[:n_samples]]


class SimpleClassifier:
    """简单分类器（基于关键词匹配）

    生产环境替换为: sklearn 分类器 / 微调 BERT / SetFit
    """

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.label_weights: dict[str, dict[str, float]] = {label: {} for label in labels}

    def train(self, samples: list[LabeledSample]) -> None:
        """从标注样本中学习关键词权重"""
        for sample in samples:
            words = set(sample.text.lower().split())
            for word in words:
                if len(word) < 2:
                    continue
                self.label_weights[sample.label][word] = (
                    self.label_weights[sample.label].get(word, 0) + 1.0
                )

    def predict_proba(self, text: str) -> dict[str, float]:
        """预测概率分布"""
        words = set(text.lower().split())
        scores: dict[str, float] = {}

        for label in self.labels:
            score = 0.0
            for word in words:
                score += self.label_weights[label].get(word, 0)
            scores[label] = score

        # Softmax 归一化
        total = sum(math.exp(s) for s in scores.values())
        if total == 0:
            return {label: 1.0 / len(self.labels) for label in self.labels}

        return {label: math.exp(s) / total for label, s in scores.items()}


class ActiveLearner:
    """主动学习循环

    ALCHEmist 核心流程:
        1. LLM 标注全部未标注样本
        2. 不确定性采样选出 Top-K
        3. 人工验证这 K 条
        4. 模型重训
        5. 重复直到收敛
    """

    def __init__(
        self,
        annotator: LLMAnnotator,
        labels: list[str],
        sampler: Optional[UncertaintySampler] = None,
        n_human_per_round: int = 10,
        max_rounds: int = 5,
        confidence_threshold: float = 0.8,
    ):
        self.annotator = annotator
        self.labels = labels
        self.sampler = sampler or UncertaintySampler("entropy")
        self.n_human_per_round = n_human_per_round
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold

        self.classifier = SimpleClassifier(labels)
        self.labeled_pool: list[LabeledSample] = []
        self.history: list[dict] = []

    def run(
        self,
        unlabeled_texts: list[str],
        seed_samples: list[LabeledSample],
        human_verifier: Optional[Callable[[str], str]] = None,
    ) -> list[LabeledSample]:
        """执行主动学习循环

        Args:
            unlabeled_texts: 未标注文本池
            seed_samples: 初始种子样本（已人工标注）
            human_verifier: 人工验证函数 (text) -> label

        Returns:
            最终标注数据集（LLM + 人工验证混合）
        """
        self.labeled_pool = seed_samples.copy()
        remaining_texts = unlabeled_texts.copy()

        print(f"[AL] 初始种子: {len(seed_samples)} 条")
        print(f"[AL] 未标注池: {len(unlabeled_texts)} 条")
        print(f"[AL] 每轮人工验证: {self.n_human_per_round} 条")
        print(f"[AL] 最大轮数: {self.max_rounds}")

        for round_idx in range(self.max_rounds):
            print(f"\n--- 第 {round_idx + 1} 轮 ---")

            # 1. 训练分类器
            self.classifier.train(self.labeled_pool)

            # 2. 对剩余未标注样本预测
            predictions = [self.classifier.predict_proba(t) for t in remaining_texts]

            # 3. 不确定性采样
            uncertain_indices = self.sampler.select(predictions, self.n_human_per_round)
            selected_texts = [remaining_texts[i] for i in uncertain_indices]

            # 4. LLM 标注全部剩余样本
            llm_results = self.annotator.annotate_batch(remaining_texts)

            # 5. 对选中的样本进行"人工验证"（模拟或真实）
            verified_samples: list[LabeledSample] = []
            for idx in uncertain_indices:
                text = remaining_texts[idx]

                if human_verifier:
                    true_label = human_verifier(text)
                else:
                    # 模拟人工：使用 LLM 结果但加入更高准确率
                    true_label = self._simulate_human_verify(text, llm_results[idx])

                verified_samples.append(LabeledSample(
                    text=text,
                    label=true_label,
                    source="human",
                    uncertainty_score=predictions[idx][max(predictions[idx], key=predictions[idx].get)],
                ))

            # 6. 高置信度 LLM 标注直接入库
            high_conf_llm: list[LabeledSample] = []
            for i, result in enumerate(llm_results):
                if i not in uncertain_indices and result.confidence == "high":
                    high_conf_llm.append(LabeledSample(
                        text=result.text,
                        label=result.label,
                        source="llm",
                        confidence=result.confidence,
                    ))

            # 7. 更新标注池和剩余文本
            self.labeled_pool.extend(verified_samples)
            self.labeled_pool.extend(high_conf_llm)

            # 从剩余池中移除已处理的文本
            processed_indices = set(uncertain_indices)
            remaining_texts = [t for i, t in enumerate(remaining_texts) if i not in processed_indices]

            # 记录本轮统计
            stats = {
                "round": round_idx + 1,
                "human_verified": len(verified_samples),
                "llm_auto": len(high_conf_llm),
                "labeled_total": len(self.labeled_pool),
                "remaining": len(remaining_texts),
            }
            self.history.append(stats)
            print(f"  人工验证: {stats['human_verified']} | LLM自动: {stats['llm_auto']} | "
                  f"累计标注: {stats['labeled_total']} | 剩余: {stats['remaining']}")

            # 停止条件检查
            if len(remaining_texts) == 0:
                print("[AL] 全部样本已处理，提前终止")
                break

            # 若连续两轮新增样本很少，也停止
            if round_idx > 0:
                prev = self.history[-2]
                curr_added = stats["human_verified"] + stats["llm_auto"]
                if curr_added < self.n_human_per_round // 2:
                    print("[AL] 新增样本不足，收敛终止")
                    break

        return self.labeled_pool

    def _simulate_human_verify(self, text: str, llm_result: AnnotationResult) -> str:
        """模拟人工验证（比 LLM 更准确）"""
        # 90% 概率接受 LLM 结果，10% 概率纠正
        if random.random() < 0.9:
            return llm_result.label
        # 随机选择另一个标签（模拟偶尔的人工错误）
        other = [l for l in self.labels if l != llm_result.label]
        return random.choice(other) if other else llm_result.label

    def get_stats(self) -> dict:
        """获取标注过程统计"""
        human_count = sum(1 for s in self.labeled_pool if s.source == "human")
        llm_count = sum(1 for s in self.labeled_pool if s.source == "llm")

        return {
            "total_labeled": len(self.labeled_pool),
            "human_verified": human_count,
            "llm_auto": llm_count,
            "human_ratio": human_count / len(self.labeled_pool) if self.labeled_pool else 0,
            "rounds": len(self.history),
            "cost_reduction": len(self.labeled_pool) / max(human_count, 1),  # 相对于全人工的节省倍数
        }


# ── 测试 ──────────────────────────────────────────────────────

def test_active_learner():
    """测试主动学习循环"""
    print("=" * 60)
    print("测试: ActiveLearner")
    print("=" * 60)

    labels = ["尺码偏差", "材质问题", "漏尿", "腰贴问题", "物流延迟", "过敏反应"]

    # 未标注文本池（模拟 100 条评论）
    unlabeled_pool = [
        "这个纸尿裤尺码偏小，建议买大一码",
        "腰贴总是粘不住，宝宝一动就开了",
        "物流太慢了，清关等了两周",
        "宝宝用了皮肤发红，是不是过敏了",
        "面料太硬了，摩擦得皮肤不舒服",
        "晚上漏尿严重，床单都湿了",
        "有一股刺鼻的化学味道",
        "包装破损，纸尿裤都露出来了",
        "这个尺寸刚刚好，穿着很合身",
        "材质很柔软，宝宝穿着舒服",
        # ... 更多样本文本
    ] * 10  # 100 条

    # 种子样本（人工标注的少量样本）
    seed_samples = [
        LabeledSample("纸尿裤太大了，宝宝穿上松松垮垮", "尺码偏差", "human"),
        LabeledSample("这个面料太硬了，摩擦宝宝皮肤", "材质问题", "human"),
        LabeledSample("晚上总是漏尿，床单都湿了", "漏尿", "human"),
        LabeledSample("腰贴粘不牢，一动就开了", "腰贴问题", "human"),
        LabeledSample("快递太慢了，等了一个星期", "物流延迟", "human"),
        LabeledSample("宝宝用了起红疹，怀疑过敏", "过敏反应", "human"),
    ]

    annotator = LLMAnnotator(labels=labels, simulate=True, simulate_accuracy=0.85)

    # 测试三种采样策略
    for strategy in ["least_confidence", "margin", "entropy"]:
        print(f"\n--- 策略: {strategy} ---")
        sampler = UncertaintySampler(strategy)
        learner = ActiveLearner(
            annotator=annotator,
            labels=labels,
            sampler=sampler,
            n_human_per_round=5,
            max_rounds=5,
        )

        result = learner.run(unlabeled_pool[:50], seed_samples)
        stats = learner.get_stats()
        print(f"  总计标注: {stats['total_labeled']} 条")
        print(f"  人工验证: {stats['human_verified']} 条 ({stats['human_ratio']:.1%})")
        print(f"  LLM 自动: {stats['llm_auto']} 条")
        print(f"  执行轮数: {stats['rounds']}")
        print(f"  成本节省: {stats['cost_reduction']:.1f}x (相对全人工)")

    print("\n" + "=" * 60)
    print("主动学习测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_active_learner()
