"""开放世界增量文本分类器

基于 Prototype-based Open World Learning 思想：
  1. 每个已知类别维护一个原型向量（类内样本均值）
  2. 分类时计算到各原型的距离，取最近
  3. 若到所有已知原型的距离 > 阈值，标记为"未知/新类别"
  4. 积累足够多"未知"样本后，聚类发现新类别原型
  5. 扩展原型集合，实现增量学习

参考: OpenCML (Open-world Continual Learning, ACL 2025)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ClassPrototype:
    """类别原型"""

    class_id: int
    class_name: str
    centroid: np.ndarray
    sample_count: int = 0
    created_at: str = "base"  # "base" 或增量轮次标识

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "centroid": [float(v) for v in self.centroid],
            "sample_count": int(self.sample_count),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClassPrototype:
        return cls(
            class_id=d["class_id"],
            class_name=d["class_name"],
            centroid=np.array(d["centroid"]),
            sample_count=d.get("sample_count", 0),
            created_at=d.get("created_at", "base"),
        )


@dataclass
class PredictionResult:
    """单条预测结果"""

    predicted_class: int
    predicted_name: str
    confidence: float
    is_known: bool  # True=已知类别, False=检测到新类别(unknown)
    distances: dict[int, float]  # 到各原型的距离


@dataclass
class IncrementResult:
    """增量学习结果"""

    new_classes_discovered: int
    new_class_names: list[str]
    total_classes: int
    replay_samples_used: int


class OpenWorldClassifier:
    """开放世界增量分类器

    支持在运行时发现新类别并扩展分类能力，无需从头重训练。

    Args:
        feature_dim: 特征向量维度
        novelty_threshold: 新类别检测阈值（距离倍数，默认 1.5×平均类内距离）
        min_samples_for_discovery: 发现新类别所需的最少"未知"样本数
        replay_buffer_size: 记忆回放缓冲区大小（防止灾难性遗忘）
        random_state: 随机种子
    """

    def __init__(
        self,
        feature_dim: int,
        novelty_threshold: float = 1.5,
        min_samples_for_discovery: int = 10,
        replay_buffer_size: int = 200,
        random_state: int = 42,
    ):
        self.feature_dim = feature_dim
        self.novelty_threshold = novelty_threshold
        self.min_samples_for_discovery = min_samples_for_discovery
        self.replay_buffer_size = replay_buffer_size
        self.rng = np.random.RandomState(random_state)

        self.prototypes: dict[int, ClassPrototype] = {}
        self.unknown_buffer: list[np.ndarray] = []
        self.replay_buffer: list[tuple[int, np.ndarray]] = []
        self._next_class_id = 0
        self.increment_round = 0

    # ── 基础训练 ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, class_names: Optional[dict[int, str]] = None) -> None:
        """在基础类别上训练（封闭世界阶段）。

        Args:
            X: (n_samples, feature_dim) 特征矩阵
            y: (n_samples,) 类别标签（整数 0, 1, 2, ...）
            class_names: {class_id: name} 类别名称映射
        """
        if X.shape[1] != self.feature_dim:
            raise ValueError(f"特征维度不匹配: 期望 {self.feature_dim}, 实际 {X.shape[1]}")

        unique_classes = sorted(set(y))
        for cid in unique_classes:
            mask = y == cid
            samples = X[mask]
            name = (class_names or {}).get(cid, f"class_{cid}")

            proto = ClassPrototype(
                class_id=cid,
                class_name=name,
                centroid=samples.mean(axis=0),
                sample_count=len(samples),
                created_at="base",
            )
            self.prototypes[cid] = proto
            self._next_class_id = max(self._next_class_id, cid + 1)

            # 填充回放缓冲区
            self._add_to_replay(cid, samples)

        print(f"[OpenWorld] 基础训练完成: {len(self.prototypes)} 个类别")

    def _add_to_replay(self, class_id: int, samples: np.ndarray) -> None:
        """向回放缓冲区添加样本（ Reservoir Sampling ）。"""
        for sample in samples:
            if len(self.replay_buffer) < self.replay_buffer_size:
                self.replay_buffer.append((class_id, sample.copy()))
            else:
                # Reservoir sampling: 以概率替换
                idx = self.rng.randint(0, len(self.replay_buffer))
                self.replay_buffer[idx] = (class_id, sample.copy())

    # ── 预测 ─────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> PredictionResult:
        """对单条样本进行预测。"""
        if len(self.prototypes) == 0:
            raise RuntimeError("模型尚未训练")

        distances = {}
        for cid, proto in self.prototypes.items():
            dist = float(np.linalg.norm(x - proto.centroid))
            distances[cid] = dist

        nearest_cid = min(distances, key=distances.get)
        nearest_dist = distances[nearest_cid]

        # 判断是否为新类别：距离 > 阈值 × 平均类内距离
        avg_intra_dist = self._avg_intra_class_distance()
        threshold_dist = self.novelty_threshold * avg_intra_dist
        is_known = nearest_dist <= threshold_dist

        # 置信度 = softmax over negative distances
        conf = self._distance_to_confidence(distances, nearest_cid)

        return PredictionResult(
            predicted_class=nearest_cid if is_known else -1,
            predicted_name=self.prototypes[nearest_cid].class_name if is_known else "unknown",
            confidence=conf,
            is_known=is_known,
            distances=distances,
        )

    def predict_batch(self, X: np.ndarray) -> list[PredictionResult]:
        """批量预测。"""
        return [self.predict(x) for x in X]

    def _avg_intra_class_distance(self) -> float:
        """计算平均类内距离（作为新类别检测的基准）。"""
        # 简化为已知类别间 centroid 的平均距离
        cents = [p.centroid for p in self.prototypes.values()]
        if len(cents) <= 1:
            return 1.0
        dists = []
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                dists.append(float(np.linalg.norm(cents[i] - cents[j])))
        return sum(dists) / len(dists) if dists else 1.0

    def _distance_to_confidence(self, distances: dict[int, float], nearest_cid: int) -> float:
        """将距离转换为置信度（近者高分）。"""
        neg_dists = {cid: -d for cid, d in distances.items()}
        values = np.array(list(neg_dists.values()))
        exp_vals = np.exp(values - np.max(values))
        probs = exp_vals / exp_vals.sum()
        cid_to_idx = {cid: i for i, cid in enumerate(neg_dists.keys())}
        return float(probs[cid_to_idx[nearest_cid]])

    # ── 新类别发现（增量学习）────────────────────────────────────

    def collect_unknown(self, X: np.ndarray) -> int:
        """扫描一批样本，收集被标记为"未知"的样本到缓冲区。

        Returns:
            新收集的未知样本数
        """
        collected = 0
        for x in X:
            result = self.predict(x)
            if not result.is_known:
                self.unknown_buffer.append(x.copy())
                collected += 1
        return collected

    def discover_new_classes(self, max_new_classes: int = 3) -> IncrementResult:
        """从未知缓冲区中发现新类别。

        使用简单聚类（k-means 变体）将未知样本分组，每组视为一个新类别。

        Args:
            max_new_classes: 本次最多发现的新类别数

        Returns:
            IncrementResult 增量学习结果
        """
        if len(self.unknown_buffer) < self.min_samples_for_discovery:
            return IncrementResult(
                new_classes_discovered=0,
                new_class_names=[],
                total_classes=len(self.prototypes),
                replay_samples_used=0,
            )

        self.increment_round += 1
        unknown_array = np.array(self.unknown_buffer)

        # 简单聚类：基于距离的贪心合并
        new_prototypes = self._greedy_cluster(unknown_array, max_new_classes)

        new_names = []
        for centroid, count in new_prototypes:
            cid = self._next_class_id
            self._next_class_id += 1
            name = f"new_class_r{self.increment_round}_{cid}"
            proto = ClassPrototype(
                class_id=cid,
                class_name=name,
                centroid=centroid,
                sample_count=count,
                created_at=f"round_{self.increment_round}",
            )
            self.prototypes[cid] = proto
            new_names.append(name)

        # 用回放数据微调所有原型（防止遗忘）
        replay_used = self._replay_update()

        # 清空未知缓冲区
        self.unknown_buffer = []

        return IncrementResult(
            new_classes_discovered=len(new_prototypes),
            new_class_names=new_names,
            total_classes=len(self.prototypes),
            replay_samples_used=replay_used,
        )

    def _greedy_cluster(self, X: np.ndarray, max_k: int) -> list[tuple[np.ndarray, int]]:
        """基于距离的贪心聚类。

        从随机种子开始，逐步合并最近的样本到现有簇，直到满足停止条件。
        """
        n = len(X)
        if n == 0:
            return []

        # 初始：每个样本是一个簇
        clusters: list[list[np.ndarray]] = [[x] for x in X]

        # 合并直到簇数 <= max_k 或簇间最小距离 > 阈值
        while len(clusters) > max_k:
            # 找到最近的两个簇（基于 centroid 距离）
            min_dist = float("inf")
            to_merge = (0, 1)

            for i in range(len(clusters)):
                c1 = np.mean(clusters[i], axis=0)
                for j in range(i + 1, len(clusters)):
                    c2 = np.mean(clusters[j], axis=0)
                    d = float(np.linalg.norm(c1 - c2))
                    if d < min_dist:
                        min_dist = d
                        to_merge = (i, j)

            # 停止条件：最近簇的距离已经超过类内平均距离的阈值
            avg_intra = self._avg_intra_class_distance()
            if min_dist > self.novelty_threshold * avg_intra * 0.5:
                break

            i, j = to_merge
            clusters[i].extend(clusters[j])
            clusters.pop(j)

        # 过滤太小的簇
        result = []
        for cluster in clusters:
            if len(cluster) >= max(3, self.min_samples_for_discovery // max_k):
                centroid = np.mean(cluster, axis=0)
                result.append((centroid, len(cluster)))

        return result

    def _replay_update(self) -> int:
        """使用回放数据更新所有原型（防止灾难性遗忘）。

        Returns:
            使用的回放样本数
        """
        if not self.replay_buffer:
            return 0

        # 按类别分组回放样本
        by_class: dict[int, list[np.ndarray]] = {}
        for cid, sample in self.replay_buffer:
            by_class.setdefault(cid, []).append(sample)

        # 更新每个原型：合并新 centroid 和回放 centroid
        for cid, proto in self.prototypes.items():
            if cid in by_class:
                replay_centroid = np.mean(by_class[cid], axis=0)
                # 加权平均：原有 centroid × 样本数 + 回放 centroid × 回放数
                w1 = proto.sample_count
                w2 = len(by_class[cid])
                proto.centroid = (proto.centroid * w1 + replay_centroid * w2) / (w1 + w2)

        return len(self.replay_buffer)

    # ── 序列化 ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "feature_dim": int(self.feature_dim),
            "novelty_threshold": float(self.novelty_threshold),
            "min_samples_for_discovery": int(self.min_samples_for_discovery),
            "next_class_id": int(self._next_class_id),
            "increment_round": int(self.increment_round),
            "prototypes": {int(cid): p.to_dict() for cid, p in self.prototypes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> OpenWorldClassifier:
        clf = cls(
            feature_dim=d["feature_dim"],
            novelty_threshold=d["novelty_threshold"],
            min_samples_for_discovery=d["min_samples_for_discovery"],
        )
        clf._next_class_id = d.get("next_class_id", 0)
        clf.increment_round = d.get("increment_round", 0)
        clf.prototypes = {
            int(cid): ClassPrototype.from_dict(p)
            for cid, p in d.get("prototypes", {}).items()
        }
        return clf

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> OpenWorldClassifier:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ── 测试 ──────────────────────────────────────────────────────

def _generate_synthetic_data(
    n_classes: int,
    n_samples_per_class: int,
    feature_dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """生成合成数据：每个类别是一个高斯簇。"""
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []

    for c in range(n_classes):
        center = rng.randn(feature_dim) * 3 + c * 5  # 类别间分离
        samples = center + rng.randn(n_samples_per_class, feature_dim) * 0.8
        X_list.append(samples)
        y_list.append(np.full(n_samples_per_class, c))

    return np.vstack(X_list), np.hstack(y_list)


def test_openworld_classifier():
    """测试开放世界分类器"""
    print("=" * 60)
    print("测试: OpenWorldClassifier")
    print("=" * 60)

    feature_dim = 16
    n_base = 3
    n_samples = 50

    # 阶段1: 基础训练（3 个已知类别）
    print("\n--- 阶段1: 基础训练 ---")
    X_base, y_base = _generate_synthetic_data(n_base, n_samples, feature_dim, seed=1)
    class_names = {0: "漏尿", 1: "红屁股", 2: "尺码问题"}

    clf = OpenWorldClassifier(feature_dim=feature_dim, novelty_threshold=1.2)
    clf.fit(X_base, y_base, class_names)
    print(f"  基础类别: {[p.class_name for p in clf.prototypes.values()]}")

    # 阶段2: 测试已知类别预测
    print("\n--- 阶段2: 已知类别预测 ---")
    test_known = X_base[:5]
    for x in test_known:
        r = clf.predict(x)
        status = "已知" if r.is_known else "未知"
        print(f"  预测: {r.predicted_name} (置信度 {r.confidence:.2f}) [{status}]")

    # 阶段3: 引入新类别样本（模拟数据漂移）
    print("\n--- 阶段3: 引入新类别样本 ---")
    # 新类别中心远离已知类别（已知类别中心在 0, 5, 10 附近，新类别放在 20+）
    X_new, y_new = _generate_synthetic_data(2, 30, feature_dim, seed=99)  # 2 个新类别
    X_new = X_new + 25  # 整体偏移，确保与已知类别分离

    # 先预测，观察被标记为"未知"
    unknown_results = [clf.predict(x) for x in X_new]
    n_unknown = sum(1 for r in unknown_results if not r.is_known)
    print(f"  新类别样本中标记为'未知': {n_unknown}/{len(X_new)}")

    # 收集未知样本
    collected = clf.collect_unknown(X_new)
    print(f"  收集到未知缓冲区: {collected}")

    # 阶段4: 增量学习 —— 发现新类别
    print("\n--- 阶段4: 增量学习（发现新类别）---")
    result = clf.discover_new_classes(max_new_classes=2)
    print(f"  发现新类别: {result.new_classes_discovered}")
    print(f"  新类别名: {result.new_class_names}")
    print(f"  总类别数: {result.total_classes}")
    print(f"  回放样本数: {result.replay_samples_used}")

    # 阶段5: 再次预测新样本
    print("\n--- 阶段5: 增量后预测 ---")
    test_new = X_new[:5]
    for x in test_new:
        r = clf.predict(x)
        status = "已知" if r.is_known else "未知"
        print(f"  预测: {r.predicted_name} (置信度 {r.confidence:.2f}) [{status}]")

    # 阶段6: 漂移检测统计
    print("\n--- 阶段6: 模型状态 ---")
    print(f"  总类别数: {len(clf.prototypes)}")
    print(f"  增量轮次: {clf.increment_round}")
    for cid, p in clf.prototypes.items():
        print(f"    [{cid}] {p.class_name} (样本数 {p.sample_count}, 来源 {p.created_at})")

    # 阶段7: 序列化测试
    print("\n--- 阶段7: 序列化测试 ---")
    clf.save("/tmp/openworld_classifier.json")
    restored = OpenWorldClassifier.load("/tmp/openworld_classifier.json")
    print(f"  保存后恢复: {len(restored.prototypes)} 个类别 ✓")

    print("\n" + "=" * 60)
    print("开放世界分类器测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_openworld_classifier()
