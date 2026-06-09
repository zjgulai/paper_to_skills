from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FeatureContribution:
    feature_name: str
    contribution_score: float
    raw_value: Any
    description: str = ""

    def __str__(self) -> str:
        sign = "+" if self.contribution_score > 0 else ""
        return f"  {self.feature_name}: {sign}{self.contribution_score:.3f} → {self.description}"


CONSUMER_FEATURE_MAP: dict[str, dict[str, str]] = {
    "age_match": {
        "high": "适合您宝宝的月龄",
        "medium": "月龄基本匹配",
        "low": "月龄匹配度较低",
    },
    "hmo_content": {
        "yes": "含有 HMO 益生元（接近母乳配方）",
        "no": "不含 HMO 成分",
    },
    "brand_trust": {
        "high": "您之前购买过同品牌产品",
        "medium": "口碑良好的品牌",
        "low": "新品牌，暂无购买记录",
    },
    "price_tier": {
        "premium": "高端品质定位",
        "mid": "性价比均衡",
        "budget": "经济实惠选择",
    },
    "safety_cert": {
        "passed": "通过权威安全认证",
        "pending": "认证进行中",
    },
    "rating": {
        "high": "用户评价 4.5 分以上",
        "medium": "用户评价 4.0-4.5 分",
        "low": "用户评价低于 4.0 分",
    },
    "market_gap": {
        "high": "市场空缺度高（竞争机会大）",
        "medium": "市场有一定空缺",
        "low": "市场已较为饱和",
    },
    "compliance_risk": {
        "low": "合规风险低（认证成本可控）",
        "medium": "合规风险中等",
        "high": "合规门槛较高",
    },
    "competitor_count": {
        "low": "竞品数量少（<50 个，市场未饱和）",
        "medium": "竞品适中（50-100 个）",
        "high": "竞品众多（>100 个，竞争激烈）",
    },
}


def _categorize_value(feature_name: str, value: Any) -> str:
    if feature_name == "age_match":
        if isinstance(value, float):
            if value >= 0.8: return "high"
            elif value >= 0.5: return "medium"
            return "low"
    elif feature_name == "hmo_content":
        return "yes" if value else "no"
    elif feature_name == "brand_trust":
        if isinstance(value, float):
            if value >= 0.7: return "high"
            elif value >= 0.4: return "medium"
            return "low"
    elif feature_name == "price_tier":
        return str(value) if str(value) in ("premium", "mid", "budget") else "mid"
    elif feature_name == "safety_cert":
        return "passed" if value else "pending"
    elif feature_name == "rating":
        if isinstance(value, (int, float)):
            if value >= 4.5: return "high"
            elif value >= 4.0: return "medium"
            return "low"
    elif feature_name == "market_gap":
        if isinstance(value, float):
            if value >= 0.35: return "high"
            elif value >= 0.15: return "medium"
            return "low"
    elif feature_name == "compliance_risk":
        if isinstance(value, str):
            return value if value in ("low", "medium", "high") else "medium"
    elif feature_name == "competitor_count":
        if isinstance(value, int):
            if value < 50: return "low"
            elif value <= 100: return "medium"
            return "high"
    return "medium"


class LocalExplainer:
    def __init__(self, feature_weights: dict[str, float]) -> None:
        self._weights = feature_weights

    def explain(
        self,
        item_features: dict[str, Any],
        base_score: float = 0.5,
    ) -> list[FeatureContribution]:
        contributions = []
        for feat_name, feat_value in item_features.items():
            if feat_name not in self._weights:
                continue
            weight = self._weights[feat_name]
            category = _categorize_value(feat_name, feat_value)
            positive_categories = {"high", "yes", "passed", "premium", "low"}
            direction = 1.0 if category in positive_categories else -0.5
            if feat_name == "compliance_risk":
                direction = 1.0 if category == "low" else (-0.5 if category == "high" else 0.0)
            elif feat_name == "competitor_count":
                direction = 1.0 if category == "low" else (-0.5 if category == "high" else 0.2)
            contribution = weight * direction
            desc = CONSUMER_FEATURE_MAP.get(feat_name, {}).get(category, f"{feat_name}={feat_value}")
            contributions.append(FeatureContribution(
                feature_name=feat_name,
                contribution_score=contribution,
                raw_value=feat_value,
                description=desc,
            ))
        contributions.sort(key=lambda c: abs(c.contribution_score), reverse=True)
        return contributions


class ConsumerFriendlyFormatter:
    def format(
        self,
        contributions: list[FeatureContribution],
        top_k: int = 3,
        include_negative: bool = False,
    ) -> str:
        positive = [c for c in contributions if c.contribution_score > 0]
        negative = [c for c in contributions if c.contribution_score < 0]
        ordinals = ["①", "②", "③", "④", "⑤"]
        lines = ["推荐理由："]
        for i, c in enumerate(positive[:top_k]):
            lines.append(f"  {ordinals[i]} {c.description}")
        result = "\n".join(lines)
        if include_negative and negative:
            result += "\n注意事项："
            for c in negative[:2]:
                result += f"\n  ⚠ {c.description}"
        return result

    def format_selection_report(
        self,
        category_name: str,
        contributions: list[FeatureContribution],
        top_k: int = 3,
    ) -> str:
        ordinals = ["①", "②", "③", "④", "⑤"]
        top_pos = [c for c in contributions if c.contribution_score > 0][:top_k]
        top_neg = [c for c in contributions if c.contribution_score < 0][:1]
        lines = [f"推荐进入【{category_name}】品类："]
        for i, c in enumerate(top_pos):
            lines.append(f"  {ordinals[i]} {c.description}")
        if top_neg:
            lines.append(f"\n  ⚠ 风险提示：{top_neg[0].description}")
        return "\n".join(lines)


@dataclass
class RecommendationWithExplanation:
    item_id: str
    item_name: str
    recommendation_score: float
    contributions: list[FeatureContribution]
    explanation_text: str

    def __str__(self) -> str:
        return (
            f"[推荐] {self.item_name} (score={self.recommendation_score:.3f})\n"
            f"{self.explanation_text}"
        )


class ExplainableRecommender:
    def __init__(self, feature_weights: dict[str, float]) -> None:
        self._explainer = LocalExplainer(feature_weights)
        self._formatter = ConsumerFriendlyFormatter()

    def recommend_with_explanation(
        self,
        candidates: list[dict],
        top_k: int = 3,
        context: str = "consumer",
    ) -> list[RecommendationWithExplanation]:
        results = []
        for item in candidates:
            item_features = {k: v for k, v in item.items() if k not in ("id", "name")}
            contributions = self._explainer.explain(item_features)
            score = sum(max(0, c.contribution_score) for c in contributions)
            if context == "selection":
                explanation = self._formatter.format_selection_report(
                    item.get("name", item.get("id", "未知")), contributions
                )
            else:
                explanation = self._formatter.format(contributions, top_k=3)
            results.append(RecommendationWithExplanation(
                item_id=item.get("id", ""),
                item_name=item.get("name", ""),
                recommendation_score=round(score, 4),
                contributions=contributions,
                explanation_text=explanation,
            ))
        results.sort(key=lambda r: r.recommendation_score, reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    FORMULA_WEIGHTS = {
        "age_match": 0.35,
        "hmo_content": 0.25,
        "brand_trust": 0.20,
        "safety_cert": 0.12,
        "rating": 0.08,
    }
    candidates = [
        {"id": "F001", "name": "某品牌A段奶粉", "age_match": 0.95, "hmo_content": True, "brand_trust": 0.8, "safety_cert": True, "rating": 4.7},
        {"id": "F002", "name": "某品牌进口奶粉", "age_match": 0.6, "hmo_content": False, "brand_trust": 0.3, "safety_cert": True, "rating": 4.2},
        {"id": "F003", "name": "某品牌有机奶粉", "age_match": 0.9, "hmo_content": True, "brand_trust": 0.5, "safety_cert": True, "rating": 4.5},
    ]
    recommender = ExplainableRecommender(FORMULA_WEIGHTS)
    results = recommender.recommend_with_explanation(candidates, top_k=3)
    print("=== 婴儿奶粉推荐结果 ===")
    for r in results:
        print(f"\n{r}")

    assert len(results) == 3
    top = results[0]
    assert top.item_id == "F001", f"最优推荐应为 F001，实际为 {top.item_id}"
    assert "推荐理由" in top.explanation_text
    print(f"\n[✓] Top推荐: {top.item_name} | score={top.recommendation_score:.3f}")

    SELECTION_WEIGHTS = {
        "market_gap": 0.4,
        "compliance_risk": 0.3,
        "competitor_count": 0.3,
    }
    selection_candidates = [
        {"id": "C001", "name": "婴儿安抚奶嘴", "market_gap": 0.40, "compliance_risk": "medium", "competitor_count": 38},
        {"id": "C002", "name": "婴儿爬行垫", "market_gap": 0.10, "compliance_risk": "low", "competitor_count": 142},
    ]
    sel_recommender = ExplainableRecommender(SELECTION_WEIGHTS)
    sel_results = sel_recommender.recommend_with_explanation(selection_candidates, top_k=2, context="selection")
    print("\n=== WF-D 选品推荐报告 ===")
    for r in sel_results:
        print(f"\n{r}")

    assert sel_results[0].item_id == "C001", "安抚奶嘴应排名第一"
    print("\n[✓] AI Explainability Consumer Trust 全部测试通过")
