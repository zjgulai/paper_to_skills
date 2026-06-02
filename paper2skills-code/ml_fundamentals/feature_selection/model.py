"""
Feature Selection — SHAP/Boruta/Permutation/RFE 特征选择
paper2skills-code: 12-ML基础 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass


@dataclass
class FeatureImportance:
    feature_name: str
    importance_score: float
    method: str
    selected: bool = False


def mock_shap_values(n_features: int, seed: int = 42) -> list[float]:
    """模拟 SHAP 值（生产替换为 shap.TreeExplainer）"""
    random.seed(seed)
    raw = [random.expovariate(1.0) for _ in range(n_features)]
    total = sum(raw)
    return [v / total for v in raw]


def shap_feature_selection(feature_names: list[str],
                           threshold: float = 0.05) -> list[FeatureImportance]:
    shap_vals = mock_shap_values(len(feature_names))
    results = []
    for name, score in zip(feature_names, shap_vals):
        results.append(FeatureImportance(
            feature_name=name, importance_score=round(score, 4),
            method="SHAP", selected=score >= threshold,
        ))
    return sorted(results, key=lambda x: -x.importance_score)


def permutation_importance(feature_names: list[str],
                           baseline_auc: float = 0.85,
                           seed: int = 42) -> list[FeatureImportance]:
    """置换重要性：随机打乱特征后 AUC 下降量"""
    random.seed(seed)
    results = []
    for i, name in enumerate(feature_names):
        drop = random.random() * 0.1 * (len(feature_names) - i) / len(feature_names)
        score = round(drop, 4)
        results.append(FeatureImportance(
            feature_name=name, importance_score=score,
            method="Permutation", selected=score > 0.01,
        ))
    return sorted(results, key=lambda x: -x.importance_score)


def rfe_selection(feature_names: list[str], n_select: int = 5) -> list[FeatureImportance]:
    """递归特征消除（RFE）：模拟版"""
    random.seed(7)
    scores = sorted([(name, random.random()) for name in feature_names],
                    key=lambda x: -x[1])
    return [FeatureImportance(
        feature_name=name, importance_score=round(score, 4),
        method="RFE", selected=(i < n_select),
    ) for i, (name, score) in enumerate(scores)]


def run_feature_selection_demo():
    features = [
        "purchase_frequency_30d", "avg_order_value", "days_since_last_order",
        "customer_age_days", "total_orders", "baby_age_months",
        "product_category_count", "review_given", "coupon_used",
        "device_type_mobile", "traffic_source_social", "sku_diversity",
    ]

    print("=== 特征选择方法对比（母婴 LTV 预测）===\n")

    for method_fn, method_name in [
        (lambda: shap_feature_selection(features, threshold=0.07), "SHAP"),
        (lambda: permutation_importance(features), "Permutation"),
        (lambda: rfe_selection(features, n_select=5), "RFE"),
    ]:
        results = method_fn()
        selected = [r.feature_name for r in results if r.selected]
        print(f"  {method_name}: 选出 {len(selected)} 个特征")
        for r in results[:5]:
            tag = "✅" if r.selected else "  "
            print(f"    {tag} {r.feature_name:35s} {r.importance_score:.4f}")
        print()

    print("✅ 特征选择演示完成")


if __name__ == "__main__":
    run_feature_selection_demo()
