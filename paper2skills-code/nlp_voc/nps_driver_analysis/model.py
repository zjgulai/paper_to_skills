"""
NPS Driver Analysis: From Reviews to Actionable Insights

基于论文 "From Reviews to Actionable Insights" (Boughamni et al., 2025)
核心思路：从评论中提取属性级情感，通过 SHAP 归因识别 NPS 驱动因素，
量化各属性对满意度的贡献，输出可操作的改进优先级。

母婴出海场景：将消费者对母婴产品的评论转化为 NPS 驱动因素洞察，
支撑产品改进和运营决策。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. 数据模型
# ---------------------------------------------------------------------------

@dataclass
class AspectSentiment:
    """单个属性的情感评分"""
    aspect: str
    sentiment: float  # -1.0 (极度负面) ~ +1.0 (极度正面)
    mention_count: int = 1
    evidence: list[str] = field(default_factory=list)


@dataclass
class ReviewAnalysis:
    """单条评论的分析结果"""
    review_id: str
    text: str
    overall_rating: float  # 原始评分，如 1-5 星
    aspects: list[AspectSentiment] = field(default_factory=list)
    nps_label: int = 0  # -1=贬损者, 0=被动者, +1=推荐者


@dataclass
class DriverInsight:
    """单个驱动因素的洞察"""
    aspect: str
    shap_value: float
    avg_sentiment: float
    mention_rate: float
    impact_score: float
    recommendation: str


# ---------------------------------------------------------------------------
# 2. 示例数据生成 (母婴出海场景)
# ---------------------------------------------------------------------------

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "产品质量": ["质量", "材质", "做工", "耐用", "安全", "无异味", "柔软", "舒适"],
    "产品安全": ["安全", "无害", "认证", "检测", "环保", "无毒", "防过敏"],
    "产品设计": ["设计", "外观", "颜色", "图案", "可爱", "时尚", "实用", "方便"],
    "物流速度": ["物流", "配送", "快递", "发货", "到达", "速度", "时效"],
    "包装体验": ["包装", "盒子", "保护", "完好", "精致", "送礼", "仪式感"],
    "客服服务": ["客服", "服务", "态度", "耐心", "专业", "回复", "售后"],
    "价格价值": ["价格", "性价比", "划算", "贵", "便宜", "优惠", "值得"],
    "使用体验": ["使用", "穿戴", "贴合", "吸水", "透气", "方便换", "不起球"],
}

POSITIVE_PATTERNS: dict[str, list[str]] = {
    "产品质量": [
        "质量很好，摸起来很柔软",
        "材质不错，宝宝穿着舒服",
        "做工精细，没有线头",
        "很耐用，洗了很多次不变形",
    ],
    "产品安全": [
        "有安全认证，放心给宝宝用",
        "无异味，材质很安全",
        "通过了严格检测，值得信赖",
        "环保材质，不怕宝宝啃咬",
    ],
    "产品设计": [
        "设计很贴心，穿脱方便",
        "颜色很好看，宝宝很喜欢",
        "图案可爱，送礼很有面子",
        "实用性强，很多细节考虑周到",
    ],
    "物流速度": [
        "物流很快，第二天就到了",
        "发货迅速，配送及时",
        "快递小哥服务态度好",
        "海外仓发货，比预期快",
    ],
    "包装体验": [
        "包装很精致，像礼物一样",
        "保护得很好，没有任何破损",
        "包装盒很有设计感",
        "送礼很有仪式感",
    ],
    "客服服务": [
        "客服态度很好，耐心解答",
        "售后很给力，有问题及时解决",
        "客服很专业，推荐尺码很准",
        "回复速度快，服务周到",
    ],
    "价格价值": [
        "性价比很高，比实体店便宜",
        "质量不错，这个价位很划算",
        "活动买的，特别值",
        "虽然贵一点但物有所值",
    ],
    "使用体验": [
        "吸水性好，宝宝屁股很干爽",
        "透气性不错，不闷热",
        "穿戴方便，一拉就穿好",
        "贴合度好，不漏尿",
    ],
}

NEGATIVE_PATTERNS: dict[str, list[str]] = {
    "产品质量": [
        "质量一般，洗几次就起球了",
        "材质偏硬，宝宝不喜欢穿",
        "做工粗糙，有线头",
        "不耐用，拉链很快就坏了",
    ],
    "产品安全": [
        "有点异味，需要多洗几次",
        "没有安全认证标识，不太放心",
        "材质摸起来不太舒服",
        "担心有荧光剂",
    ],
    "产品设计": [
        "设计不太合理，穿脱麻烦",
        "颜色跟图片有差距",
        "尺码偏小，需要买大一码",
        "款式一般，不够时尚",
    ],
    "物流速度": [
        "物流太慢了，等了一周",
        "发货延迟，耽误使用",
        "快递途中包装破损",
        "清关时间太长",
    ],
    "包装体验": [
        "包装太简单，像地摊货",
        "盒子压扁了，影响送礼",
        "没有保护措施",
        "包装不环保",
    ],
    "客服服务": [
        "客服回复慢，等很久",
        "售后推来推去，不负责任",
        "客服不专业，答非所问",
        "态度冷淡，体验不好",
    ],
    "价格价值": [
        "价格偏贵，性价比不高",
        "质量配不上这个价格",
        "活动价格虚高",
        "同类产品价格更便宜",
    ],
    "使用体验": [
        "吸水性一般，容易漏",
        "不透气，宝宝红屁股",
        "穿戴麻烦，容易滑落",
        "尺码不准，偏大",
    ],
}


def generate_synthetic_reviews(
    n_reviews: int = 500,
    seed: int = 42,
) -> list[ReviewAnalysis]:
    """生成母婴出海场景合成评论数据，含方面情感标签。"""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    reviews: list[ReviewAnalysis] = []
    aspects = list(ASPECT_KEYWORDS.keys())

    for i in range(n_reviews):
        n_aspects = rng.randint(2, 5)
        selected_aspects = rng.sample(aspects, n_aspects)

        aspect_sentiments: list[AspectSentiment] = []
        texts: list[str] = []
        total_sentiment = 0.0

        for aspect in selected_aspects:
            sentiment = np_rng.normal(0.0, 0.4)
            sentiment = max(-1.0, min(1.0, sentiment))

            if sentiment > 0.2:
                pattern = rng.choice(POSITIVE_PATTERNS[aspect])
            elif sentiment < -0.2:
                pattern = rng.choice(NEGATIVE_PATTERNS[aspect])
            else:
                patterns = (
                    POSITIVE_PATTERNS[aspect][:1] + NEGATIVE_PATTERNS[aspect][:1]
                )
                pattern = rng.choice(patterns)

            texts.append(pattern)
            total_sentiment += sentiment

            aspect_sentiments.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=round(sentiment, 3),
                    mention_count=1,
                    evidence=[pattern],
                )
            )

        # 综合评分由平均情感 + 噪声决定
        avg_sentiment = total_sentiment / len(selected_aspects)
        rating = 3.0 + avg_sentiment * 1.8 + np_rng.normal(0, 0.3)
        rating = float(np.clip(rating, 1.0, 5.0))

        # NPS 分类
        if rating >= 4.0:
            nps = 1
        elif rating <= 2.5:
            nps = -1
        else:
            nps = 0

        reviews.append(
            ReviewAnalysis(
                review_id=f"review_{i:04d}",
                text=";".join(texts),
                overall_rating=round(rating, 2),
                aspects=aspect_sentiments,
                nps_label=nps,
            )
        )

    return reviews


# ---------------------------------------------------------------------------
# 3. 方面提取器 (规则版，生产环境可替换为 LLM/ABSA 模型)
# ---------------------------------------------------------------------------

class AspectExtractor:
    """从评论文本中提取属性级情感。"""

    def __init__(self, aspect_keywords: dict[str, list[str]] | None = None) -> None:
        self.keywords = aspect_keywords or ASPECT_KEYWORDS
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.patterns: dict[str, re.Pattern[str]] = {}
        for aspect, keywords in self.keywords.items():
            pattern = "|".join(re.escape(kw) for kw in keywords)
            self.patterns[aspect] = re.compile(pattern)

    def extract(self, text: str) -> list[AspectSentiment]:
        """从文本中提取属性情感（规则版，基于关键词匹配和简单情感词典）。"""
        sentiments: list[AspectSentiment] = []
        sentences = [s.strip() for s in text.split(";") if s.strip()]

        for aspect, pattern in self.patterns.items():
            matched_sentences = [s for s in sentences if pattern.search(s)]
            if not matched_sentences:
                continue

            sentiment = self._score_sentiment(matched_sentences)
            sentiments.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment,
                    mention_count=len(matched_sentences),
                    evidence=matched_sentences,
                )
            )

        return sentiments

    def _score_sentiment(self, sentences: list[str]) -> float:
        """基于情感词典打分（简化版）。"""
        positive_words = [
            "好", "不错", "满意", "喜欢", "推荐", "值得", "棒", "优秀", "精致",
            "快", "耐心", "专业", "给力", "划算", "物有所值", "放心", "安全",
            "舒适", "柔软", "可爱", "时尚", "实用", "方便", "贴心", "耐用",
            "great", "good", "excellent", "love", "perfect", "amazing",
        ]
        negative_words = [
            "差", "不好", "失望", "慢", "贵", "不划算", "不值", "一般", "粗糙",
            "麻烦", "不方便", "不满意", "漏", "破损", "异味", "担心", "偏小",
            "bad", "poor", "terrible", "disappointed", "slow", "expensive",
            "worst", "horrible", "cheap",
        ]

        total = 0.0
        for sentence in sentences:
            pos = sum(1 for w in positive_words if w in sentence)
            neg = sum(1 for w in negative_words if w in sentence)
            total += (pos - neg) / max(pos + neg, 1)

        return round(float(np.clip(total / len(sentences), -1.0, 1.0)), 3)


# ---------------------------------------------------------------------------
# 4. NPS 驱动因素分析器 (基于 SHAP 思想)
# ---------------------------------------------------------------------------

class NPSDriverAnalyzer:
    """基于方面情感的 NPS/满意度驱动因素分析。"""

    def __init__(self, aspect_keywords: dict[str, list[str]] | None = None) -> None:
        self.aspects = list((aspect_keywords or ASPECT_KEYWORDS).keys())
        self.n_aspects = len(self.aspects)
        self.aspect_to_idx = {a: i for i, a in enumerate(self.aspects)}

    def vectorize(
        self, reviews: list[ReviewAnalysis]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """将评论转换为方面情感矩阵 X 和 NPS 标签 y。"""
        X = np.zeros((len(reviews), self.n_aspects), dtype=np.float64)
        y = np.zeros(len(reviews), dtype=np.float64)

        for i, review in enumerate(reviews):
            for asp in review.aspects:
                if asp.aspect in self.aspect_to_idx:
                    X[i, self.aspect_to_idx[asp.aspect]] = asp.sentiment
            y[i] = review.nps_label

        return X, y

    def compute_shap_attribution(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> dict[str, float]:
        """
        计算各属性对 NPS 的归因值。

        方法：拟合线性模型 Xβ ≈ y，系数 β_j 即为属性 j 的边际贡献。
        同时计算置换重要性验证稳定性。
        """
        # 1. 线性回归系数（带 L2 正则化避免共线性）
        # 使用解析解: β = (X^T X + λI)^-1 X^T y
        lam = 0.1
        XtX = X.T @ X
        XtX[np.diag_indices_from(XtX)] += lam
        beta = np.linalg.solve(XtX, X.T @ y)

        # 2. 置换重要性：逐个属性置零，观察 MSE 变化
        # 完整模型预测
        pred_full = X @ beta
        mse_full = float(np.mean((pred_full - y) ** 2))

        shap_values: dict[str, float] = {}
        for aspect, idx in self.aspect_to_idx.items():
            # 仅将该属性置零
            X_perm = X.copy()
            X_perm[:, idx] = 0.0
            pred_perm = X_perm @ beta
            mse_perm = float(np.mean((pred_perm - y) ** 2))
            # 贡献 = 移除该属性后 MSE 增加量（越大越重要）
            perm_importance = mse_perm - mse_full
            # 综合归因 = 系数（方向） × 置换重要性（幅度）
            shap_values[aspect] = beta[idx] * (1 + perm_importance * 10)

        return shap_values

    def analyze_drivers(
        self,
        reviews: list[ReviewAnalysis],
    ) -> list[DriverInsight]:
        """分析驱动因素并生成洞察。"""
        X, y = self.vectorize(reviews)
        shap_values = self.compute_shap_attribution(X, y)

        # 计算各属性的统计指标
        n = len(reviews)
        aspect_stats: dict[str, dict[str, float]] = {}

        for aspect in self.aspects:
            sentiments = []
            mentions = 0
            for review in reviews:
                for asp in review.aspects:
                    if asp.aspect == aspect:
                        sentiments.append(asp.sentiment)
                        mentions += 1

            if sentiments:
                aspect_stats[aspect] = {
                    "avg_sentiment": float(np.mean(sentiments)),
                    "mention_rate": mentions / n,
                    "sentiment_std": float(np.std(sentiments)),
                }
            else:
                aspect_stats[aspect] = {
                    "avg_sentiment": 0.0,
                    "mention_rate": 0.0,
                    "sentiment_std": 0.0,
                }

        # 生成洞察
        insights: list[DriverInsight] = []
        for aspect in self.aspects:
            stats = aspect_stats[aspect]
            shap = shap_values.get(aspect, 0.0)

            # 影响分 = SHAP 贡献 × 提及率（考虑影响广度）
            impact = abs(shap) * (1 + stats["mention_rate"])

            # 生成建议
            rec = self._generate_recommendation(
                aspect, stats["avg_sentiment"], shap, stats["mention_rate"]
            )

            insights.append(
                DriverInsight(
                    aspect=aspect,
                    shap_value=round(shap, 4),
                    avg_sentiment=round(stats["avg_sentiment"], 3),
                    mention_rate=round(stats["mention_rate"], 3),
                    impact_score=round(impact, 4),
                    recommendation=rec,
                )
            )

        # 按影响分降序
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        return insights

    def _generate_recommendation(
        self,
        aspect: str,
        avg_sentiment: float,
        shap_value: float,
        mention_rate: float,
    ) -> str:
        """基于属性状态生成改进建议。"""
        if abs(shap_value) < 0.005:
            return f"[{aspect}] 当前对 NPS 影响较小，保持现有水平即可。"

        if avg_sentiment < -0.2:
            priority = "高优先级"
            action = "立即整改"
        elif avg_sentiment < 0.0:
            priority = "中优先级"
            action = "针对性改进"
        elif avg_sentiment < 0.3:
            priority = "观察"
            action = "微调优化"
        else:
            priority = "保持"
            action = "维持优势"

        aspects_map: dict[str, str] = {
            "产品质量": "加强品控、提升材质标准",
            "产品安全": "完善安全认证、增加检测报告",
            "产品设计": "收集用户反馈优化设计细节",
            "物流速度": "优化供应链、增加海外仓覆盖",
            "包装体验": "升级包装设计、增加防护措施",
            "客服服务": "加强客服培训、缩短响应时间",
            "价格价值": "优化定价策略、增加组合优惠",
            "使用体验": "改进产品功能、优化使用流程",
        }

        return (
            f"[{aspect}] {priority} | 情感均值: {avg_sentiment:+.2f} | "
            f"提及率: {mention_rate:.1%} | 建议: {action} - {aspects_map.get(aspect, '')}"
        )


# ---------------------------------------------------------------------------
# 5. 核心函数
# ---------------------------------------------------------------------------

def analyze_nps_drivers(
    reviews: list[ReviewAnalysis] | None = None,
    n_reviews: int = 500,
) -> dict[str, Any]:
    """
    主入口：分析 NPS 驱动因素。

    返回包含以下内容的字典:
        - n_reviews: 评论总数
        - nps_distribution: 贬损者/被动者/推荐者分布
        - top_drivers: 按影响分排序的驱动因素列表
        - summary: 关键发现摘要
    """
    if reviews is None:
        reviews = generate_synthetic_reviews(n_reviews)

    analyzer = NPSDriverAnalyzer()
    insights = analyzer.analyze_drivers(reviews)

    # NPS 分布
    nps_dist = {"贬损者": 0, "被动者": 0, "推荐者": 0}
    for r in reviews:
        if r.nps_label == -1:
            nps_dist["贬损者"] += 1
        elif r.nps_label == 0:
            nps_dist["被动者"] += 1
        else:
            nps_dist["推荐者"] += 1

    # 计算 NPS 值
    nps_score = (
        (nps_dist["推荐者"] - nps_dist["贬损者"]) / len(reviews) * 100
        if reviews
        else 0.0
    )

    # 摘要
    top_positive = [i for i in insights if i.shap_value > 0][:3]
    top_negative = [i for i in insights if i.shap_value <= 0][:3]

    summary = {
        "总评论数": len(reviews),
        "NPS 得分": round(nps_score, 1),
        "推荐者占比": f"{nps_dist['推荐者'] / len(reviews):.1%}",
        "贬损者占比": f"{nps_dist['贬损者'] / len(reviews):.1%}",
        "核心驱动因素": [i.aspect for i in insights[:3]],
        "最大提升机会": insights[0].aspect if insights else "无",
    }

    return {
        "n_reviews": len(reviews),
        "nps_distribution": nps_dist,
        "nps_score": round(nps_score, 1),
        "top_drivers": [
            {
                "属性": i.aspect,
                "SHAP归因值": i.shap_value,
                "平均情感": i.avg_sentiment,
                "提及率": f"{i.mention_rate:.1%}",
                "影响分": i.impact_score,
                "建议": i.recommendation,
            }
            for i in insights
        ],
        "top_positive_drivers": [
            {"属性": i.aspect, "归因值": i.shap_value} for i in top_positive
        ],
        "top_negative_drivers": [
            {"属性": i.aspect, "归因值": i.shap_value} for i in top_negative
        ],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 6. 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("NPS 驱动因素分析 - 母婴出海场景")
    print("=" * 70)

    # 生成数据并分析
    result = analyze_nps_drivers(n_reviews=500)

    print(f"\n📊 NPS 概览")
    print(f"   总评论数: {result['n_reviews']}")
    print(f"   NPS 得分: {result['nps_score']}")
    print(f"   分布: {result['nps_distribution']}")

    print(f"\n🔍 驱动因素排名 (按影响分)")
    print("-" * 70)
    for i, driver in enumerate(result["top_drivers"][:5], 1):
        print(f"\n  #{i} {driver['属性']}")
        print(f"     SHAP归因值: {driver['SHAP归因值']:+.4f}")
        print(f"     平均情感: {driver['平均情感']:+.3f}")
        print(f"     提及率: {driver['提及率']}")
        print(f"     影响分: {driver['影响分']:.4f}")
        print(f"     💡 {driver['建议']}")

    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)
