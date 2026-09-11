"""AdaNEN 简化版流式 VOC 分类器

基于 Adaptive Neural Ensemble Network (ACM TKDD 2024) 的核心思想：
  1. 集成多个基于不同时间窗口的分类器
  2. 滑动窗口检测概念漂移
  3. 根据验证性能动态调整集成权重
  4. 漂移时自动注入新分类器

简化假设：
  - 使用 Prototype-based 分类器（与 OpenWorldClassifier 一致）
  - 漂移检测基于样本到最近原型的平均距离变化
  - 集成权重基于验证窗口准确率
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class WindowStats:
    """数据窗口统计"""

    X: np.ndarray
    y: np.ndarray
    avg_prototype_dist: float = 0.0  # 样本到最近原型的平均距离


class PrototypeClassifier:
    """基于原型的简单分类器（支持单分类器）"""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.prototypes: dict[int, np.ndarray] = {}
        self.class_counts: dict[int, int] = {}
        self.trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """从数据计算原型向量"""
        for cid in sorted(set(y)):
            mask = y == cid
            self.prototypes[cid] = X[mask].mean(axis=0)
            self.class_counts[cid] = int(mask.sum())
        self.trained = True

    def predict(self, x: np.ndarray) -> int:
        """最近原型分类"""
        if not self.prototypes:
            raise RuntimeError("模型尚未训练，无可用原型")
        dists = {cid: float(np.linalg.norm(x - p)) for cid, p in self.prototypes.items()}
        return min(dists, key=dists.get)

    def predict_proba(self, x: np.ndarray) -> dict[int, float]:
        """基于距离的 softmax 概率"""
        neg_dists = {cid: -np.linalg.norm(x - p) for cid, p in self.prototypes.items()}
        vals = np.array(list(neg_dists.values()))
        exp_vals = np.exp(vals - np.max(vals))
        probs = exp_vals / exp_vals.sum()
        return {cid: float(p) for cid, p in zip(neg_dists.keys(), probs)}

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """在验证集上的准确率"""
        if not self.trained or len(self.prototypes) == 0:
            return 0.0
        correct = sum(self.predict(x) == yi for x, yi in zip(X, y))
        return correct / len(y)

    def avg_prototype_distance(self, X: np.ndarray) -> float:
        """样本到最近原型的平均距离"""
        if not self.trained or len(self.prototypes) == 0:
            return 0.0
        dists = []
        for x in X:
            d = min(float(np.linalg.norm(x - p)) for p in self.prototypes.values())
            dists.append(d)
        return sum(dists) / len(dists) if dists else 0.0


class DriftDetector:
    """概念漂移检测器

    检测策略：比较当前窗口和参考窗口的分布差异
    - 使用样本到最近原型的平均距离作为分布特征
    - 当距离变化超过阈值时触发漂移告警
    """

    def __init__(self, drift_threshold: float = 0.3, min_window_size: int = 20):
        self.drift_threshold = drift_threshold
        self.min_window_size = min_window_size
        self.reference_stats: Optional[float] = None

    def update_reference(self, X: np.ndarray, classifier: PrototypeClassifier) -> None:
        """更新参考窗口统计（通常在无漂移时调用）"""
        if len(X) < self.min_window_size:
            return
        self.reference_stats = classifier.avg_prototype_distance(X)

    def detect(self, X: np.ndarray, classifier: PrototypeClassifier) -> tuple[bool, float]:
        """检测当前窗口是否发生漂移

        Returns:
            (是否漂移, 漂移程度分数)
        """
        if self.reference_stats is None or len(X) < self.min_window_size:
            return False, 0.0

        current_dist = classifier.avg_prototype_distance(X)
        # 相对变化率
        if self.reference_stats == 0:
            drift_score = 0.0
        else:
            drift_score = abs(current_dist - self.reference_stats) / self.reference_stats

        is_drift = drift_score > self.drift_threshold
        return is_drift, drift_score


@dataclass
class EnsembleMember:
    """集成中的一个分类器成员"""

    classifier: PrototypeClassifier
    weight: float = 1.0
    birth_time: int = 0  # 创建时的样本计数
    accuracy_history: list[float] = field(default_factory=list)


class EnsembleClassifier:
    """自适应集成分类器

    维护多个基于不同时间窗口的分类器，根据性能动态调整权重。
    核心机制：
      1. 新数据到达 → 用最新窗口训练候选分类器
      2. 验证窗口评估所有分类器 → 更新权重
      3. 权重低的分类器逐渐淘汰
    """

    def __init__(
        self,
        feature_dim: int,
        max_members: int = 5,
        weight_decay: float = 0.95,
        min_weight: float = 0.1,
    ):
        self.feature_dim = feature_dim
        self.max_members = max_members
        self.weight_decay = weight_decay
        self.min_weight = min_weight
        self.members: list[EnsembleMember] = []
        self.sample_counter = 0

    def add_member(self, X: np.ndarray, y: np.ndarray) -> None:
        """基于当前窗口数据添加新分类器成员"""
        clf = PrototypeClassifier(self.feature_dim)
        clf.fit(X, y)

        member = EnsembleMember(
            classifier=clf,
            weight=1.0,
            birth_time=self.sample_counter,
        )
        self.members.append(member)

    def prune_if_needed(self) -> None:
        """淘汰低权重成员（应在权重更新后调用）"""
        if len(self.members) > self.max_members:
            self._prune_members()

    def update_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """基于验证集更新所有成员权重"""
        for member in self.members:
            acc = member.classifier.score(X_val, y_val)
            member.accuracy_history.append(acc)
            # 新权重 = 验证准确率 × 时间衰减
            age = self.sample_counter - member.birth_time
            decay = self.weight_decay ** (age / 100)  # 每 100 样本衰减一次
            member.weight = acc * decay

        # 归一化
        total = sum(m.weight for m in self.members)
        if total > 0:
            for member in self.members:
                member.weight /= total

    def _prune_members(self) -> None:
        """淘汰权重最低的成员"""
        self.members.sort(key=lambda m: m.weight, reverse=True)
        # 保留前 max_members 个
        self.members = self.members[: self.max_members]

    def predict(self, x: np.ndarray) -> int:
        """加权集成预测"""
        if not self.members:
            raise RuntimeError("集成器为空，尚未训练")

        # 收集每个成员的预测和权重
        votes: dict[int, float] = {}
        for member in self.members:
            pred = member.classifier.predict(x)
            votes[pred] = votes.get(pred, 0.0) + member.weight

        return max(votes, key=votes.get)

    def predict_proba(self, x: np.ndarray) -> dict[int, float]:
        """加权集成概率"""
        if not self.members:
            raise RuntimeError("集成器为空，尚未训练")

        agg: dict[int, float] = {}
        for member in self.members:
            probs = member.classifier.predict_proba(x)
            for cid, p in probs.items():
                agg[cid] = agg.get(cid, 0.0) + p * member.weight

        # 归一化
        total = sum(agg.values())
        if total > 0:
            agg = {cid: p / total for cid, p in agg.items()}
        return agg

    def get_weights(self) -> dict[int, float]:
        """获取各成员的当前权重"""
        return {i: m.weight for i, m in enumerate(self.members)}


@dataclass
class StreamPrediction:
    """单条流式预测结果"""

    predicted_class: int
    confidence: float
    drift_score: float
    is_drift: bool
    active_classifiers: int


class AdaNENClassifier:
    """AdaNEN 简化版流式分类器

    支持在数据分布持续变化（概念漂移）的流式环境中自适应分类。

    Args:
        feature_dim: 特征维度
        window_size: 滑动窗口大小（每个窗口的样本数）
        drift_threshold: 漂移检测阈值（相对距离变化率）
        max_classifiers: 最大集成分类器数量
        validation_ratio: 验证集占窗口的比例（用于权重更新）
    """

    def __init__(
        self,
        feature_dim: int,
        window_size: int = 50,
        drift_threshold: float = 0.3,
        max_classifiers: int = 5,
        validation_ratio: float = 0.2,
    ):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.validation_ratio = validation_ratio

        self.ensemble = EnsembleClassifier(
            feature_dim=feature_dim,
            max_members=max_classifiers,
        )
        min_window_size = max(5, int(window_size * validation_ratio / 2))
        self.drift_detector = DriftDetector(
            drift_threshold=drift_threshold,
            min_window_size=min_window_size,
        )

        # 数据缓冲区
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[int] = []
        self._window_count = 0

        # 历史记录
        self.drift_history: list[dict] = []
        self.accuracy_history: list[float] = []

    def ingest(self, x: np.ndarray, y: int) -> Optional[StreamPrediction]:
        """摄入单条样本，当窗口满时触发训练和检测

        Returns:
            StreamPrediction（窗口满时）或 None
        """
        self._buffer_X.append(x.copy())
        self._buffer_y.append(y)
        self.ensemble.sample_counter += 1

        if len(self._buffer_X) >= self.window_size:
            return self._process_window()
        return None

    def ingest_batch(self, X: np.ndarray, y: np.ndarray) -> list[Optional[StreamPrediction]]:
        """批量摄入样本"""
        return [self.ingest(x, yi) for x, yi in zip(X, y)]

    def _process_window(self) -> StreamPrediction:
        """处理满窗口：训练、检测漂移、更新权重"""
        X = np.array(self._buffer_X)
        y = np.array(self._buffer_y)
        self._window_count += 1

        # 划分训练集和验证集
        n_val = int(len(y) * self.validation_ratio)
        if n_val < 5:
            n_val = min(5, len(y) // 2)

        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        # 1. 漂移检测（用已有最新分类器衡量新窗口的分布变化）
        if self.ensemble.members:
            latest_member = max(self.ensemble.members, key=lambda m: m.birth_time)
            latest_clf = latest_member.classifier
            is_drift, drift_score = self.drift_detector.detect(X_val, latest_clf)
        else:
            is_drift, drift_score = False, 0.0

        # 2. 用最新窗口训练新分类器并加入集成
        new_clf = PrototypeClassifier(self.feature_dim)
        new_clf.fit(X_train, y_train)
        self.ensemble.add_member(X_train, y_train)

        # 3. 用验证集更新所有成员权重
        self.ensemble.update_weights(X_val, y_val)

        # 4. 淘汰低权重成员（在权重更新后执行，确保基于真实性能）
        self.ensemble.prune_if_needed()

        # 5. 更新漂移检测的参考统计（用训练集，样本量更大更稳定）
        self.drift_detector.update_reference(X_train, new_clf)

        # 6. 记录历史
        val_acc = new_clf.score(X_val, y_val)
        self.accuracy_history.append(val_acc)
        self.drift_history.append({
            "window": self._window_count,
            "is_drift": is_drift,
            "drift_score": drift_score,
            "val_acc": val_acc,
            "n_members": len(self.ensemble.members),
            "weights": self.ensemble.get_weights(),
        })

        # 清空缓冲区
        self._buffer_X = []
        self._buffer_y = []

        # 返回最后一个样本的预测（作为代表性结果）
        last_x = X_val[-1] if len(X_val) > 0 else X_train[-1]
        pred = self.ensemble.predict(last_x)
        conf = self.ensemble.predict_proba(last_x).get(pred, 0.0)

        return StreamPrediction(
            predicted_class=pred,
            confidence=conf,
            drift_score=drift_score,
            is_drift=is_drift,
            active_classifiers=len(self.ensemble.members),
        )

    def predict(self, x: np.ndarray) -> int:
        """对单条样本预测（不触发学习）"""
        return self.ensemble.predict(x)

    def get_drift_report(self) -> dict:
        """获取漂移检测报告"""
        if not self.drift_history:
            return {"status": "no_data", "message": "尚未处理任何窗口"}

        drift_count = sum(1 for h in self.drift_history if h["is_drift"])
        avg_acc = sum(self.accuracy_history) / len(self.accuracy_history) if self.accuracy_history else 0.0

        return {
            "status": "drift_detected" if drift_count > 0 else "stable",
            "total_windows": self._window_count,
            "drift_events": drift_count,
            "drift_rate": drift_count / self._window_count if self._window_count > 0 else 0.0,
            "avg_validation_acc": round(avg_acc, 3),
            "latest_drift_score": round(self.drift_history[-1]["drift_score"], 3),
            "active_classifiers": len(self.ensemble.members),
        }

    def to_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "window_size": self.window_size,
            "window_count": self._window_count,
            "drift_history": self.drift_history,
            "accuracy_history": self.accuracy_history,
        }


# ── 测试 ──────────────────────────────────────────────────────

def _generate_streaming_data(
    n_windows: int,
    samples_per_window: int,
    feature_dim: int,
    n_classes: int = 3,
    seed: int = 42,
    drift_type: str = "abrupt",  # "abrupt" 或 "gradual"
) -> tuple[np.ndarray, np.ndarray]:
    """生成带概念漂移的合成流式数据"""
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []

    # 初始类别中心
    centers = rng.randn(n_classes, feature_dim) * 3

    for w in range(n_windows):
        # 漂移：每隔一定窗口偏移类别中心
        if drift_type == "abrupt" and w > 0 and w % 5 == 0:
            # 突变：大幅偏移
            centers += rng.randn(n_classes, feature_dim) * 2.0
        elif drift_type == "gradual" and w > 0:
            # 渐进：小幅持续偏移
            centers += rng.randn(n_classes, feature_dim) * 0.15

        for _ in range(samples_per_window):
            cid = rng.randint(0, n_classes)
            sample = centers[cid] + rng.randn(feature_dim) * 0.8
            X_list.append(sample)
            y_list.append(cid)

    return np.array(X_list), np.array(y_list)


def test_streaming_classifier():
    """测试流式分类器（含概念漂移）"""
    print("=" * 60)
    print("测试: AdaNENClassifier（流式分类 + 概念漂移检测）")
    print("=" * 60)

    feature_dim = 16
    n_windows = 12
    samples_per_window = 30
    n_classes = 3

    # 场景1: 突变漂移 (abrupt)
    print("\n--- 场景1: 突变漂移 (abrupt) ---")
    X_stream, y_stream = _generate_streaming_data(
        n_windows, samples_per_window, feature_dim, n_classes, seed=1, drift_type="abrupt"
    )

    clf = AdaNENClassifier(
        feature_dim=feature_dim,
        window_size=samples_per_window,
        drift_threshold=0.25,
        max_classifiers=4,
    )

    predictions = []
    for x, y in zip(X_stream, y_stream):
        result = clf.ingest(x, y)
        if result:
            predictions.append(result)

    # 报告
    report = clf.get_drift_report()
    print(f"  总窗口数: {report['total_windows']}")
    print(f"  漂移事件: {report['drift_events']}")
    print(f"  漂移率: {report['drift_rate']:.1%}")
    print(f"  平均验证准确率: {report['avg_validation_acc']:.3f}")
    print(f"  活跃分类器: {report['active_classifiers']}")

    # 详细窗口记录
    print("\n--- 窗口级漂移详情 ---")
    for h in clf.drift_history:
        flag = "🚨" if h["is_drift"] else "  "
        print(f"  {flag} 窗口{h['window']:2d}: 漂移分={h['drift_score']:.3f}, "
              f"验证准确率={h['val_acc']:.3f}, 成员数={h['n_members']}")

    # 场景2: 渐进漂移 (gradual)
    print("\n--- 场景2: 渐进漂移 (gradual) ---")
    X_stream2, y_stream2 = _generate_streaming_data(
        n_windows, samples_per_window, feature_dim, n_classes, seed=2, drift_type="gradual"
    )

    clf2 = AdaNENClassifier(
        feature_dim=feature_dim,
        window_size=samples_per_window,
        drift_threshold=0.2,
        max_classifiers=4,
    )

    for x, y in zip(X_stream2, y_stream2):
        clf2.ingest(x, y)

    report2 = clf2.get_drift_report()
    print(f"  总窗口数: {report2['total_windows']}")
    print(f"  漂移事件: {report2['drift_events']}")
    print(f"  漂移率: {report2['drift_rate']:.1%}")
    print(f"  平均验证准确率: {report2['avg_validation_acc']:.3f}")

    # 场景3: 无漂移（基线）
    print("\n--- 场景3: 无漂移（基线对照）---")
    # 生成无漂移数据
    rng = np.random.RandomState(99)
    centers = rng.randn(n_classes, feature_dim) * 3
    X_baseline, y_baseline = [], []
    for _ in range(n_windows * samples_per_window):
        cid = rng.randint(0, n_classes)
        sample = centers[cid] + rng.randn(feature_dim) * 0.8
        X_baseline.append(sample)
        y_baseline.append(cid)
    X_baseline = np.array(X_baseline)
    y_baseline = np.array(y_baseline)

    clf3 = AdaNENClassifier(
        feature_dim=feature_dim,
        window_size=samples_per_window,
        drift_threshold=0.25,
        max_classifiers=3,
    )
    for x, y in zip(X_baseline, y_baseline):
        clf3.ingest(x, y)

    report3 = clf3.get_drift_report()
    print(f"  总窗口数: {report3['total_windows']}")
    print(f"  漂移事件: {report3['drift_events']}")
    print(f"  漂移率: {report3['drift_rate']:.1%}")
    print(f"  平均验证准确率: {report3['avg_validation_acc']:.3f}")

    print("\n" + "=" * 60)
    print("流式分类器测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_streaming_classifier()
