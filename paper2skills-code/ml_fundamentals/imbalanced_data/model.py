"""
Imbalanced Data Handling — SMOTE/ClassWeight/Threshold 不平衡数据处理
paper2skills-code: 12-ML基础 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass


@dataclass
class Sample:
    features: list[float]
    label: int


@dataclass
class ImbalancedResult:
    method: str
    original_ratio: float
    after_ratio: float
    n_samples_before: int
    n_samples_after: int
    estimated_recall_improvement: float


def smote_oversample(minority_samples: list[Sample],
                     target_count: int, k: int = 5,
                     seed: int = 42) -> list[Sample]:
    """SMOTE 过采样：合成少数类样本"""
    random.seed(seed)
    synthetic = []
    while len(synthetic) < target_count - len(minority_samples):
        s1 = random.choice(minority_samples)
        s2 = random.choice(minority_samples)
        alpha = random.random()
        new_features = [
            s1.features[i] * alpha + s2.features[i] * (1 - alpha)
            for i in range(len(s1.features))
        ]
        synthetic.append(Sample(features=new_features, label=1))
    return minority_samples + synthetic


def undersample_majority(majority_samples: list[Sample],
                         target_count: int, seed: int = 42) -> list[Sample]:
    """随机欠采样：减少多数类"""
    random.seed(seed)
    return random.sample(majority_samples, min(target_count, len(majority_samples)))


def class_weight_dict(n_pos: int, n_neg: int) -> dict[int, float]:
    """计算类别权重（sklearn 兼容格式）"""
    total = n_pos + n_neg
    return {0: total / (2 * n_neg), 1: total / (2 * n_pos)}


def optimal_threshold(pos_probs: list[float], neg_probs: list[float],
                      beta: float = 1.0) -> float:
    """最优分类阈值（F-beta 最大化）"""
    best_f = 0.0
    best_t = 0.5
    for t in [i / 100.0 for i in range(10, 91)]:
        tp = sum(1 for p in pos_probs if p >= t)
        fp = sum(1 for p in neg_probs if p >= t)
        fn = sum(1 for p in pos_probs if p < t)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f_beta = (1 + beta**2) * precision * recall / max((beta**2) * precision + recall, 1e-6)
        if f_beta > best_f:
            best_f = f_beta
            best_t = t
    return round(best_t, 2)


def run_imbalanced_demo():
    random.seed(42)
    majority = [Sample([random.gauss(0, 1) for _ in range(5)], 0) for _ in range(900)]
    minority = [Sample([random.gauss(0.5, 1) for _ in range(5)], 1) for _ in range(100)]

    original_ratio = len(minority) / len(majority)
    print("=== 不平衡数据处理（母婴 Churn 预测，正负样本 1:9）===\n")
    print(f"原始: {len(majority)} 负例 + {len(minority)} 正例 (比例 {original_ratio:.2f})")

    synthetic_minority = smote_oversample(minority, target_count=450)
    smote_ratio = len(synthetic_minority) / len(majority)
    print(f"SMOTE 后: 正例 {len(synthetic_minority)} | 比例 {smote_ratio:.2f}")

    undersampled_majority = undersample_majority(majority, target_count=300)
    under_ratio = len(minority) / len(undersampled_majority)
    print(f"欠采样后: 负例 {len(undersampled_majority)} | 比例 {under_ratio:.2f}")

    weights = class_weight_dict(len(minority), len(majority))
    print(f"类别权重: {{0: {weights[0]:.2f}, 1: {weights[1]:.2f}}}")

    pos_probs = [random.betavariate(3, 2) for _ in range(100)]
    neg_probs = [random.betavariate(2, 4) for _ in range(900)]
    t = optimal_threshold(pos_probs, neg_probs, beta=2.0)
    print(f"最优分类阈值 (F2): {t}")

    print("\n✅ 不平衡数据处理演示完成")


if __name__ == "__main__":
    run_imbalanced_demo()
