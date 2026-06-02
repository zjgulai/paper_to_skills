"""
Ensemble Methods — Bagging/Boosting/Stacking/Blending
paper2skills-code: 12-ML基础 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field


@dataclass
class WeakLearner:
    name: str
    weight: float = 1.0

    def predict(self, x: list[float], seed: int = 0) -> float:
        """模拟弱学习器预测（返回 0-1 概率）"""
        random.seed(seed + hash(self.name) % 1000)
        base = sum(v * (0.3 + random.random() * 0.4) for v in x[:3]) / (len(x) + 1)
        return min(0.99, max(0.01, base + random.gauss(0, 0.05)))


class BaggingEnsemble:
    """Bagging：并行训练，多数投票/平均"""
    def __init__(self, n_learners: int = 10):
        self.learners = [WeakLearner(f"bag_{i}") for i in range(n_learners)]

    def predict_proba(self, x: list[float]) -> float:
        preds = [l.predict(x, seed=i) for i, l in enumerate(self.learners)]
        return sum(preds) / len(preds)


class GradientBoostingEnsemble:
    """Boosting：串行训练，逐步修正残差（简化版）"""
    def __init__(self, n_stages: int = 5, learning_rate: float = 0.1):
        self.n_stages = n_stages
        self.lr = learning_rate
        self.learners = [WeakLearner(f"boost_{i}", weight=learning_rate) for i in range(n_stages)]

    def predict_proba(self, x: list[float]) -> float:
        pred = 0.5
        for i, l in enumerate(self.learners):
            delta = l.predict(x, seed=i) - 0.5
            pred += self.lr * delta
        return min(0.99, max(0.01, pred))


class StackingEnsemble:
    """Stacking：第一层模型输出作为第二层元模型输入"""
    def __init__(self):
        self.base_models = [WeakLearner(f"stack_base_{i}") for i in range(3)]
        self.meta_model = WeakLearner("meta")

    def predict_proba(self, x: list[float]) -> float:
        base_preds = [m.predict(x, seed=i) for i, m in enumerate(self.base_models)]
        return self.meta_model.predict(base_preds, seed=99)


def evaluate_ensemble(model, test_cases: list[tuple], name: str) -> dict:
    preds = [model.predict_proba(x) for x, _ in test_cases]
    labels = [y for _, y in test_cases]
    # 简化 AUC 估算（正负样本概率排序）
    pos_probs = [p for p, l in zip(preds, labels) if l == 1]
    neg_probs = [p for p, l in zip(preds, labels) if l == 0]
    if not pos_probs or not neg_probs:
        return {"name": name, "auc": 0.5}
    auc = sum(1 for p in pos_probs for n in neg_probs if p > n) / (len(pos_probs) * len(neg_probs))
    return {"name": name, "auc": round(auc, 4)}


def run_ensemble_demo():
    random.seed(42)
    test_cases = [([random.random() for _ in range(5)], random.randint(0, 1))
                  for _ in range(100)]

    models = [
        (BaggingEnsemble(), "Bagging(n=10)"),
        (GradientBoostingEnsemble(), "Boosting(n=5)"),
        (StackingEnsemble(), "Stacking(3+meta)"),
    ]

    print("=== 集成学习方法对比（母婴复购预测）===")
    for model, name in models:
        result = evaluate_ensemble(model, test_cases, name)
        print(f"  {name:25s}: AUC {result['auc']:.4f}")

    print("\n✅ 集成学习演示完成")


if __name__ == "__main__":
    run_ensemble_demo()
