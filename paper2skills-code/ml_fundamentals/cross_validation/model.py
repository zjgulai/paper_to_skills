"""
Cross Validation Strategies — K-Fold/Stratified/TimeSeries/Group 交叉验证
paper2skills-code: 12-ML基础 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterator


@dataclass
class CVResult:
    strategy: str
    fold_scores: list[float]
    mean_score: float
    std_score: float
    cv_type: str


def kfold_indices(n: int, k: int = 5) -> Iterator[tuple[list[int], list[int]]]:
    """K-Fold 索引生成"""
    fold_size = n // k
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        val_idx = list(range(val_start, val_end))
        train_idx = list(range(0, val_start)) + list(range(val_end, n))
        yield train_idx, val_idx


def stratified_kfold_indices(labels: list[int], k: int = 5) -> Iterator[tuple[list[int], list[int]]]:
    """Stratified K-Fold：保证每折类别分布一致"""
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[label].append(i)

    folds = [[] for _ in range(k)]
    for label, indices in class_indices.items():
        for j, idx in enumerate(indices):
            folds[j % k].append(idx)

    for i in range(k):
        val_idx = folds[i]
        train_idx = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        yield train_idx, val_idx


def timeseries_cv_indices(n: int, n_splits: int = 5,
                          gap: int = 0) -> Iterator[tuple[list[int], list[int]]]:
    """时序交叉验证：只用历史数据预测未来（无数据泄露）"""
    test_size = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = i * test_size
        val_start = train_end + gap
        val_end = min(val_start + test_size, n)
        if val_end > n:
            break
        yield list(range(0, train_end)), list(range(val_start, val_end))


def mock_train_eval(train_idx: list[int], val_idx: list[int],
                    seed: int = 42) -> float:
    """模拟训练+评估，返回 AUC 分数"""
    import hashlib
    h = int(hashlib.md5(str(sorted(val_idx[:3])).encode()).hexdigest(), 16)
    base = 0.82 + (h % 100) / 1000.0
    return min(0.97, max(0.70, base))


def cross_validate(strategy: str, n_samples: int = 1000,
                   labels: list[int] = None, k: int = 5) -> CVResult:
    scores = []
    if strategy == "kfold":
        for train, val in kfold_indices(n_samples, k):
            scores.append(mock_train_eval(train, val))
    elif strategy == "stratified" and labels:
        for train, val in stratified_kfold_indices(labels, k):
            scores.append(mock_train_eval(train, val))
    elif strategy == "timeseries":
        for train, val in timeseries_cv_indices(n_samples, k):
            scores.append(mock_train_eval(train, val))

    mean = sum(scores) / len(scores) if scores else 0.0
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1))
    return CVResult(strategy=strategy, fold_scores=[round(s, 4) for s in scores],
                    mean_score=round(mean, 4), std_score=round(std, 4), cv_type=strategy)


def run_cv_demo():
    print("=== 交叉验证策略对比（母婴 Churn 预测）===")
    n = 1000
    labels = [0] * 700 + [1] * 300  # 70:30 不平衡

    for strategy, kwargs in [
        ("kfold", {}),
        ("stratified", {"labels": labels}),
        ("timeseries", {}),
    ]:
        result = cross_validate(strategy, n, **kwargs)
        print(f"  {strategy:12s}: AUC {result.mean_score:.4f} (std {result.std_score:.4f})")

    print("\n✅ 交叉验证演示完成")


if __name__ == "__main__":
    run_cv_demo()
