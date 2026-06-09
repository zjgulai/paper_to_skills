"""
paper-选题 论文质量评分器（真实实现）

参考评分规范见同目录 paper_quality_scorer.md。
评分总分 ≥ 7 才推荐进入 paper-萃取 流程。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


_NOVEL_TECH_KEYWORDS = {
    "transformer", "attention", "diffusion", "gnn", "graph neural",
    "reinforcement learning", "rl ", "rlhf", "self-supervised",
    "contrastive", "few-shot", "zero-shot", "in-context",
    "uplift", "causal forest", "doubly robust", "double machine learning",
    "mixture of experts", "moe", "agentic", "tool use", "mcp",
    "lora", "prompt tuning", "continual learning", "meta-learning",
}

_EXPERIMENT_SIGNALS = {
    "experiment", "benchmark", "ablation", "evaluation",
    "real-world", "production", "case study", "datasets",
}

_CODE_SIGNALS = {
    "github.com", "open-source", "open source", "code is available",
    "implementation", "released code",
}

_BUSINESS_KEYWORDS = {
    "e-commerce": 1.0, "ecommerce": 1.0, "retail": 0.9, "marketplace": 0.9,
    "recommendation": 1.0, "advertising": 0.9, "marketing": 0.9,
    "supply chain": 0.9, "inventory": 0.9, "demand forecasting": 1.0,
    "churn": 0.8, "ltv": 1.0, "ab test": 0.8, "uplift": 1.0,
    "voc": 0.7, "review": 0.5, "rating": 0.5, "user behavior": 0.6,
    "cross-border": 1.2, "international": 0.8, "amazon": 0.9, "shopify": 0.9,
    "mother": 1.5, "baby": 1.5, "maternity": 1.5, "infant": 1.5,
}


@dataclass
class DimensionResult:
    score: float
    weight: int
    reason: str


@dataclass
class ScoreResult:
    total_score: float
    recommend: bool
    details: Dict[str, DimensionResult] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total_score": round(self.total_score, 2),
            "recommend": self.recommend,
            "details": {
                k: {"score": round(v.score, 2), "weight": v.weight, "reason": v.reason}
                for k, v in self.details.items()
            },
        }


class PaperQualityScorer:
    """论文质量评分器。

    评分维度（与 paper-选题/SKILL.md 的标准对齐）：
    - 算法创新性 20%
    - 实验完整性 30%
    - 工程可落地性 30%
    - 业务适配度 20%
    """

    WEIGHTS = {
        "innovation": 20,
        "experiment": 30,
        "engineering": 30,
        "business_fit": 20,
    }

    RECOMMEND_THRESHOLD = 7.0

    def __init__(self, threshold: Optional[float] = None) -> None:
        if threshold is not None:
            self.RECOMMEND_THRESHOLD = threshold

    def score(self, paper: dict) -> dict:
        innovation = self._score_innovation(paper)
        experiment = self._score_experiment(paper)
        engineering = self._score_engineering(paper)
        business = self._score_business_fit(paper)

        details = {
            "algorithm_innovation": innovation,
            "experiment_completeness": experiment,
            "engineering_feasibility": engineering,
            "business_fit": business,
        }

        total = sum(d.score * d.weight for d in details.values()) / 100.0
        result = ScoreResult(
            total_score=total,
            recommend=total >= self.RECOMMEND_THRESHOLD,
            details=details,
        )
        return result.as_dict()

    def _score_innovation(self, paper: dict) -> DimensionResult:
        text = self._collect_text(paper).lower()
        hits = [kw for kw in _NOVEL_TECH_KEYWORDS if kw in text]
        year = self._extract_year(paper)
        age_years = max(0, datetime.now().year - year) if year else 5
        recency_bonus = max(0.0, 2.0 - age_years * 0.5)

        keyword_score = min(7.0, 4.0 + len(hits) * 0.6)
        score = min(10.0, keyword_score + recency_bonus)
        reason = (
            f"命中前沿关键词 {len(hits)} 个: {hits[:3]}; 论文年龄 {age_years} 年; "
            f"近期性加分 {recency_bonus:.1f}"
        )
        return DimensionResult(score=score, weight=self.WEIGHTS["innovation"], reason=reason)

    def _score_experiment(self, paper: dict) -> DimensionResult:
        text = self._collect_text(paper).lower()
        if paper.get("has_experiments") is False:
            return DimensionResult(
                score=2.0,
                weight=self.WEIGHTS["experiment"],
                reason="paper.has_experiments=False",
            )

        signal_hits = [s for s in _EXPERIMENT_SIGNALS if s in text]
        base = 5.0 + min(4.0, len(signal_hits) * 0.8)

        dataset_count = self._estimate_dataset_count(text)
        dataset_bonus = min(1.5, dataset_count * 0.5)

        score = min(10.0, base + dataset_bonus)
        reason = (
            f"实验关键词命中 {len(signal_hits)}; 估算数据集数 ≈ {dataset_count}; "
            f"has_experiments={paper.get('has_experiments', 'unknown')}"
        )
        return DimensionResult(score=score, weight=self.WEIGHTS["experiment"], reason=reason)

    def _score_engineering(self, paper: dict) -> DimensionResult:
        text = self._collect_text(paper).lower()
        has_code = paper.get("has_code")
        if has_code is True:
            code_score = 8.0
        elif has_code is False:
            code_score = 3.0
        else:
            code_score = 7.0 if any(s in text for s in _CODE_SIGNALS) else 5.0

        complexity_penalty = 0.0
        complexity_signals = ["theoretical", "proof", "theorem", "lemma"]
        if sum(1 for s in complexity_signals if s in text) >= 2:
            complexity_penalty = 1.5

        score = max(1.0, min(10.0, code_score - complexity_penalty))
        reason = (
            f"has_code={has_code}; code_score={code_score}; "
            f"theoretical_penalty={complexity_penalty}"
        )
        return DimensionResult(score=score, weight=self.WEIGHTS["engineering"], reason=reason)

    def _score_business_fit(self, paper: dict) -> DimensionResult:
        text = self._collect_text(paper).lower()
        keywords_input: List[str] = paper.get("keywords", []) or []
        combined = " ".join([text] + [k.lower() for k in keywords_input])

        hits: List[str] = []
        weight_sum = 0.0
        for kw, w in _BUSINESS_KEYWORDS.items():
            if kw in combined:
                hits.append(kw)
                weight_sum += w

        score = min(10.0, 3.0 + weight_sum * 1.2)
        reason = f"业务关键词命中 {len(hits)} 个 (weighted={weight_sum:.1f}): {hits[:5]}"
        return DimensionResult(score=score, weight=self.WEIGHTS["business_fit"], reason=reason)

    @staticmethod
    def _collect_text(paper: dict) -> str:
        parts = [
            paper.get("title", ""),
            paper.get("abstract", ""),
            " ".join(paper.get("keywords", []) or []),
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _extract_year(paper: dict) -> Optional[int]:
        year = paper.get("published_year") or paper.get("year")
        if isinstance(year, int):
            return year
        if isinstance(year, str):
            m = re.search(r"(20\d{2})", year)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def _estimate_dataset_count(text: str) -> int:
        matches = re.findall(r"(\d+)\s+(?:datasets?|benchmarks?)", text)
        if matches:
            return max(int(m) for m in matches)
        return text.count("dataset")


def _demo() -> None:
    scorer = PaperQualityScorer()
    paper = {
        "title": "Uplift Modeling for Cross-Border E-commerce Maternal Recommendation",
        "abstract": (
            "We propose a doubly robust uplift modeling framework with "
            "transformer encoders. Experiments on 3 real-world datasets from "
            "Amazon and Shopify show consistent ROI improvement. Code is "
            "available at https://github.com/example."
        ),
        "has_code": True,
        "has_experiments": True,
        "published_year": 2025,
        "keywords": ["uplift modeling", "causal inference", "e-commerce", "mother and baby"],
    }
    result = scorer.score(paper)
    print(f"总分: {result['total_score']}/10  推荐: {result['recommend']}")
    for k, v in result["details"].items():
        print(f"  - {k}: {v['score']}/10 ({v['weight']}%) — {v['reason']}")


if __name__ == "__main__":
    _demo()
