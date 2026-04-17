"""
Multi-Agent Actionable Advice (MAA) Pipeline
基于论文: A Multi-Agent System for Generating Actionable Business Advice
将大规模评论语料转化为具体、可执行的商业建议。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Review:
    text: str
    market: str = ""
    product: str = ""
    rating: int = 5


@dataclass
class Issue:
    theme: str
    description: str
    source_review: str


@dataclass
class Advice:
    issue: Issue
    text: str
    scores: Dict[str, int] = field(default_factory=dict)
    feasibility_score: float = 0.0


@dataclass
class Cluster:
    reviews: List[Review]
    representative: Review = field(default=None)


class ReviewEmbedder:
    """Simple TF-IDF style embedder for demonstration."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower())

    def fit(self, reviews: List[Review]) -> None:
        docs = [self._tokenize(r.text) for r in reviews]
        all_terms = set()
        for d in docs:
            all_terms.update(d)
        self.vocab = {t: i for i, t in enumerate(sorted(all_terms))}
        n = len(docs)
        for t in self.vocab:
            df = sum(1 for d in docs if t in d)
            self.idf[t] = math.log((n + 1) / (df + 1)) + 1

    def transform(self, reviews: List[Review]) -> List[List[float]]:
        vectors = []
        for r in reviews:
            tokens = self._tokenize(r.text)
            vec = [0.0] * len(self.vocab)
            tf = Counter(tokens)
            for t, count in tf.items():
                if t in self.vocab:
                    vec[self.vocab[t]] = count * self.idf.get(t, 1.0)
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class ClusteringAgent:
    """Embeds reviews, clusters them, and selects representative review per cluster."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.embedder = ReviewEmbedder()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _kmeans(self, vectors: List[List[float]]) -> List[int]:
        # Simple k-means++ initialization and iterations
        if not vectors:
            return []
        k = min(self.n_clusters, len(vectors))
        centroids = [vectors[0][:]]
        for _ in range(1, k):
            dists = []
            for v in vectors:
                min_d = max(self._cosine_similarity(v, c) for c in centroids)
                dists.append(1 - min_d)
            total = sum(dists) or 1.0
            probs = [d / total for d in dists]
            # pick by probability (deterministic for test: choose max dist)
            idx = probs.index(max(probs))
            centroids.append(vectors[idx][:])
        labels = [0] * len(vectors)
        for _ in range(20):
            new_labels = []
            for v in vectors:
                sims = [self._cosine_similarity(v, c) for c in centroids]
                new_labels.append(sims.index(max(sims)))
            if new_labels == labels:
                break
            labels = new_labels
            # recompute centroids
            for j in range(k):
                members = [vectors[i] for i, lbl in enumerate(labels) if lbl == j]
                if members:
                    centroids[j] = [sum(m[i] for m in members) / len(members) for i in range(len(vectors[0]))]
        return labels

    def run(self, reviews: List[Review]) -> List[Cluster]:
        self.embedder.fit(reviews)
        vectors = self.embedder.transform(reviews)
        labels = self._kmeans(vectors)
        clusters: Dict[int, List[Review]] = {}
        for r, lbl in zip(reviews, labels):
            clusters.setdefault(lbl, []).append(r)
        result = []
        for lbl, revs in clusters.items():
            vecs = self.embedder.transform(revs)
            # compute centroid
            centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(len(vecs[0]))]
            # select representative: closest to centroid
            best_idx = max(range(len(vecs)), key=lambda i: self._cosine_similarity(vecs[i], centroid))
            result.append(Cluster(reviews=revs, representative=revs[best_idx]))
        return result


class IssueAgent:
    """Extracts themes and issues from representative reviews (LLM-simulated)."""

    KEYWORD_THEMES: Dict[str, List[str]] = {
        "noise": ["静音性能", "马达噪音", "夜间使用干扰"],
        "battery": ["续航能力", "充电便捷性", "外出使用"],
        "comfort": ["佩戴舒适度", "吸力调节", "乳头疼痛"],
        "ease of use": ["操作简便性", "清洗难度", "配件安装"],
        "sterilization": ["消毒效果", "烘干功能", "容量大小"],
        "heating": ["加热均匀性", "温控精准度", "加热速度"],
    }

    def run(self, clusters: List[Cluster]) -> List[Issue]:
        issues = []
        for cl in clusters:
            rep = cl.representative
            text_lower = rep.text.lower()
            matched_theme = None
            for theme, keywords in self.KEYWORD_THEMES.items():
                if any(kw in text_lower or kw in rep.text for kw in keywords + [theme]):
                    matched_theme = theme
                    break
            if matched_theme is None:
                matched_theme = "general"
            cn_themes = {
                "noise": "静音与噪音",
                "battery": "续航与便携",
                "comfort": "舒适与吸力",
                "ease of use": "易用与清洗",
                "sterilization": "消毒与烘干",
                "heating": "加热与温控",
                "general": "综合体验",
            }
            issue_desc = self._summarize_issue(rep.text)
            issues.append(Issue(
                theme=cn_themes.get(matched_theme, matched_theme),
                description=issue_desc,
                source_review=rep.text,
            ))
        return issues

    def _summarize_issue(self, text: str) -> str:
        # Heuristic: take first sentence or up to 60 chars
        s = text.split("。")[0]
        if len(s) > 80:
            s = s[:80] + "..."
        return s


class RecommendationAgent:
    """Generates actionable advice per issue (LLM-simulated)."""

    TEMPLATES: Dict[str, List[str]] = {
        "静音与噪音": [
            "升级马达减震结构，引入静音认证测试；在详情页突出分贝数据。",
            "推出夜间静音模式，降低低频噪音对母婴休息的干扰。",
        ],
        "续航与便携": [
            "升级电池容量或推出快充版本，明确标注单次充电使用次数。",
            "开发车载/移动电源兼容配件，拓展外出背奶场景。",
        ],
        "舒适与吸力": [
            "增加吸力档位细分（如9档→15档），提供柔软亲肤罩尺寸选择。",
            "在包装内附赠乳头测量卡，帮助用户快速选对罩口尺寸。",
        ],
        "易用与清洗": [
            "简化配件数量，推广一体式可拆卸设计，减少清洗死角。",
            "拍摄30秒安装/清洗视频教程，降低首次使用门槛。",
        ],
        "消毒与烘干": [
            "扩大消毒仓容量，支持同时容纳吸奶器+奶瓶+奶嘴。",
            "优化烘干风道设计，缩短烘干时间并降低运行噪音。",
        ],
        "加热与温控": [
            "引入NTC精准温控芯片，实现±1°C温控并在屏幕实时显示。",
            "增加母乳解冻模式，避免高温破坏营养成分。",
        ],
        "综合体验": [
            "建立用户反馈快速响应机制，针对高频差评问题30天内给出改进计划。",
            "在产品包装中增加中英德多语言快速入门卡片。",
        ],
    }

    def run(self, issues: List[Issue]) -> List[Advice]:
        advices = []
        for issue in issues:
            templates = self.TEMPLATES.get(issue.theme, self.TEMPLATES["综合体验"])
            for t in templates:
                advices.append(Advice(issue=issue, text=t))
        return advices


class EvaluationAgent:
    """Scores advice on Specificity, Relevance, Actionability, Concision (SRAC)."""

    def run(self, advices: List[Advice]) -> List[Advice]:
        for adv in advices:
            text = adv.text
            # Heuristic scoring based on text characteristics
            s = 4 if any(k in text for k in ["升级", "推出", "增加", "优化", "引入"]) else 3
            r = 4 if adv.issue.theme in text or any(k in text for k in adv.issue.description[:10].split()) else 3
            a = 4 if any(k in text for k in ["降低", "缩短", "帮助", "减少", "实现"]) else 3
            c = 4 if 20 <= len(text) <= 60 else 3
            adv.scores = {"S": s, "R": r, "A": a, "C": c}
        return advices


class RankingAgent:
    """Ranks advices by weighted composite score and feasibility."""

    def run(self, advices: List[Advice]) -> List[Advice]:
        for adv in advices:
            srac = sum(adv.scores.values())
            # Feasibility heuristic: shorter and concrete actions score higher
            feasibility = 0.7 + 0.05 * srac
            adv.feasibility_score = round(feasibility, 2)
        advices.sort(key=lambda a: a.feasibility_score, reverse=True)
        return advices


class ActionableAdviceGenerator:
    """End-to-end MAA pipeline."""

    def __init__(self, n_clusters: int = 3):
        self.clustering = ClusteringAgent(n_clusters=n_clusters)
        self.issue = IssueAgent()
        self.recommendation = RecommendationAgent()
        self.evaluation = EvaluationAgent()
        self.ranking = RankingAgent()

    def generate(self, reviews: List[Review]) -> Dict:
        clusters = self.clustering.run(reviews)
        issues = self.issue.run(clusters)
        advices = self.recommendation.run(issues)
        advices = self.evaluation.run(advices)
        advices = self.ranking.run(advices)
        return {
            "clusters": len(clusters),
            "issues": [{"theme": i.theme, "description": i.description} for i in issues],
            "top_advices": [
                {
                    "theme": a.issue.theme,
                    "advice": a.text,
                    "SRAC": a.scores,
                    "feasibility": a.feasibility_score,
                }
                for a in advices[:6]
            ],
        }


def build_demo_reviews() -> List[Review]:
    return [
        Review(
            text="这款Momcozy吸奶器夜间使用马达声音太大，影响宝宝睡眠，希望能改进静音性能。",
            market="德国",
            product="Momcozy M5 吸奶器",
            rating=3,
        ),
        Review(
            text="吸奶器续航能力一般，外出背奶时经常没电，建议提升电池容量或支持快充。",
            market="美国",
            product="Momcozy M5 吸奶器",
            rating=3,
        ),
        Review(
            text="吸力档位不够精细，最大档有点痛，希望能增加更多柔和档位选择。",
            market="美国",
            product="Momcozy M5 吸奶器",
            rating=4,
        ),
        Review(
            text="Momcozy消毒器容量太小了，一次放不下吸奶器全套配件，烘干时间也很长。",
            market="德国",
            product="Momcozy 紫外线消毒器",
            rating=3,
        ),
        Review(
            text="暖奶器加热不均匀，有时候外热内冷，温控不够精准，担心影响母乳质量。",
            market="美国",
            product="Momcozy 智能暖奶器",
            rating=2,
        ),
        Review(
            text="产品配件太多，清洗安装很麻烦，新手妈妈用起来很有压力，需要简化设计。",
            market="中国",
            product="Momcozy M5 吸奶器",
            rating=3,
        ),
    ]


def demo():
    reviews = build_demo_reviews()
    generator = ActionableAdviceGenerator(n_clusters=3)
    result = generator.generate(reviews)
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ------------------ Tests ------------------

def test_embedding_basic():
    reviews = build_demo_reviews()
    emb = ReviewEmbedder()
    emb.fit(reviews)
    vecs = emb.transform(reviews)
    assert len(vecs) == len(reviews)
    assert all(len(v) == len(emb.vocab) for v in vecs)


def test_clustering_selects_representative():
    reviews = build_demo_reviews()
    agent = ClusteringAgent(n_clusters=2)
    clusters = agent.run(reviews)
    assert len(clusters) <= 2
    for c in clusters:
        assert c.representative is not None
        assert c.representative in c.reviews


def test_end_to_end_output_structure():
    reviews = build_demo_reviews()
    generator = ActionableAdviceGenerator(n_clusters=3)
    result = generator.generate(reviews)
    assert "clusters" in result
    assert "issues" in result
    assert "top_advices" in result
    assert len(result["top_advices"]) > 0
    for adv in result["top_advices"]:
        assert "theme" in adv
        assert "advice" in adv
        assert "SRAC" in adv
        assert all(k in adv["SRAC"] for k in ["S", "R", "A", "C"])


def test_evaluation_scores_in_range():
    reviews = build_demo_reviews()
    generator = ActionableAdviceGenerator(n_clusters=3)
    result = generator.generate(reviews)
    for adv in result["top_advices"]:
        for score in adv["SRAC"].values():
            assert 1 <= score <= 5


def test_ranking_is_sorted():
    reviews = build_demo_reviews()
    generator = ActionableAdviceGenerator(n_clusters=3)
    result = generator.generate(reviews)
    scores = [a["feasibility"] for a in result["top_advices"]]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    demo()
