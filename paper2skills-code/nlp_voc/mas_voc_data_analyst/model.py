"""
Multi-Agent VOC Data Analyst
基于论文: arXiv:2402.01386

核心思想: 用多智能体协作系统自动执行VOC定性数据分析，
将论文中的27-Agent定性分析框架适配到母婴出海VOC场景。

每个Agent负责特定分析任务，通过管道协作完成从原始VOC数据到结构化洞察的转换：
数据摄入 → 主题分析 → 情感分析 → 编码手册 → 模式识别 → 洞察综合 → 质量验证 → 报告生成

本实现为规则基线版，保留LLM扩展接口。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


# ────────────────────────── 数据模型 ──────────────────────────

@dataclass
class VOCRecord:
    """VOC数据记录"""
    record_id: str
    text: str
    source: str = ""       # 'amazon', 'reddit', 'trustpilot', 'zendesk'
    rating: float = 0.0
    category: str = ""     # 'product', 'service', 'shipping'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """单条分析结果"""
    record_id: str
    themes: list[str] = field(default_factory=list)
    sentiment: str = "neutral"   # 'positive', 'negative', 'neutral'
    sentiment_score: float = 0.0
    codes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class ThemeCluster:
    """主题聚类"""
    theme_id: str
    theme_name: str
    description: str
    keywords: list[str]
    record_ids: list[str] = field(default_factory=list)
    sentiment_distribution: dict[str, int] = field(default_factory=dict)
    representative_quotes: list[str] = field(default_factory=list)


@dataclass
class Pattern:
    """识别出的模式"""
    pattern_type: str      # 'co_occurrence', 'trend', 'anomaly'
    description: str
    supporting_evidence: list[str]
    confidence: float


@dataclass
class Insight:
    """洞察"""
    insight_id: str
    insight_type: str      # 'pain_point', 'opportunity', 'trend', 'risk'
    title: str
    description: str
    evidence_count: int
    affected_records: list[str]
    severity: float        # 0-10
    action_suggestion: str = ""


@dataclass
class VOCAnalysisReport:
    """VOC分析报告"""
    report_id: str
    dataset_name: str
    total_records: int
    theme_clusters: list[ThemeCluster] = field(default_factory=list)
    sentiment_summary: dict[str, Any] = field(default_factory=dict)
    patterns: list[Pattern] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    codebook: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    raw_results: list[AnalysisResult] = field(default_factory=list)


# ────────────────────────── Agent 基类 ──────────────────────────

class BaseAgent:
    """Agent 基类"""

    def __init__(self, agent_id: str, name: str) -> None:
        self.agent_id = agent_id
        self.name = name
        self.memory: list[dict[str, Any]] = []

    def log(self, action: str, data: dict[str, Any]) -> None:
        self.memory.append({"agent": self.name, "action": action, "data": data})

    def process(self, data: Any) -> Any:
        raise NotImplementedError


# ────────────────────────── 情感词典 ──────────────────────────

SENTIMENT_DICT = {
    "positive": [
        "love", "great", "excellent", "amazing", "perfect", "best", "recommend",
        "happy", "satisfied", "awesome", "wonderful", "fantastic", "good", "nice",
        "easy", "comfortable", "convenient", "helpful", "useful", "quality",
        "喜欢", "好用", "推荐", "满意", "不错", "方便", "舒适", "质量好",
    ],
    "negative": [
        "hate", "terrible", "awful", "worst", "bad", "disappointing", "broken",
        "useless", "difficult", "hard", "problem", "issue", "defect", "poor",
        "annoying", "frustrating", "waste", "return", "refund", "complaint",
        "不喜欢", "难用", "糟糕", "失望", "问题", "坏了", "差", "退货", "麻烦",
    ],
}


# ────────────────────────── 主题词典（母婴出海场景） ──────────────────────────

THEME_KEYWORDS = {
    "noise": {
        "keywords": ["noise", "quiet", "silent", "loud", "sound", "motor", "buzzing",
                     "噪音", "静音", "声音", "吵", "安静"],
        "description": "Product noise level concerns",
    },
    "suction": {
        "keywords": ["suction", "suck", "pump", "power", "strength", "pressure",
                     "flow", "milk", "expression", "吸力", "吸奶", "力度"],
        "description": "Suction power and milk expression effectiveness",
    },
    "portability": {
        "keywords": ["portable", "travel", "compact", "lightweight", "carry", "bag",
                     "size", "便携", "轻便", "携带", "外出", "旅行", "小巧"],
        "description": "Product portability and travel convenience",
    },
    "battery": {
        "keywords": ["battery", "charge", "charging", "power", "cordless", "续航",
                     "电池", "充电", "电量", "续航", "无线"],
        "description": "Battery life and charging experience",
    },
    "cleaning": {
        "keywords": ["clean", "wash", "sterilize", "dishwasher", "parts", "assemble",
                     "清洗", "清洁", "消毒", "拆装", "零件"],
        "description": "Ease of cleaning and maintenance",
    },
    "comfort": {
        "keywords": ["comfortable", "soft", "gentle", "pain", "hurt", "sore",
                     "nipple", "breast", "舒适", "柔软", "疼痛", "乳头", "乳房"],
        "description": "User comfort and physical experience",
    },
    "price": {
        "keywords": ["price", "expensive", "cheap", "cost", "value", "money", "afford",
                     "价格", "贵", "便宜", "划算", "性价比", "值得"],
        "description": "Price perception and value for money",
    },
    "shipping": {
        "keywords": ["shipping", "delivery", "arrive", "fast", "slow", "package",
                     "物流", "快递", "发货", "到达", "包装"],
        "description": "Shipping and delivery experience",
    },
    "customer_service": {
        "keywords": ["service", "support", "help", "response", "warranty", "return",
                     "客服", "售后", "服务", "回复", "保修"],
        "description": "Customer service and support experience",
    },
    "design": {
        "keywords": ["design", "look", "appearance", "color", "style", "modern",
                     "设计", "外观", "颜色", "款式", "时尚"],
        "description": "Product design and aesthetics",
    },
}


# ────────────────────────── Agent 实现 ──────────────────────────

class DataIngestionAgent(BaseAgent):
    """
    Agent 1: 数据摄入Agent
    清洗和预处理原始VOC数据
    """

    def __init__(self) -> None:
        super().__init__("agent_001", "DataIngestionAgent")

    def process(self, records: list[dict[str, Any]]) -> list[VOCRecord]:
        cleaned: list[VOCRecord] = []
        for i, raw in enumerate(records):
            text = raw.get("text", raw.get("content", ""))
            # 基础清洗
            text = text.strip()
            text = re.sub(r"\s+", " ", text)

            record = VOCRecord(
                record_id=raw.get("id", f"voc_{i:04d}"),
                text=text,
                source=raw.get("source", "unknown"),
                rating=float(raw.get("rating", 0)),
                category=raw.get("category", ""),
                metadata=raw.get("metadata", {}),
            )
            cleaned.append(record)

        self.log("ingest", {"input_count": len(records), "output_count": len(cleaned)})
        return cleaned


class ThematicAnalysisAgent(BaseAgent):
    """
    Agent 2-4: 主题分析Agent（对应论文的Summary + Coders + Themes）
    提取VOC记录中的主题标签
    """

    def __init__(self) -> None:
        super().__init__("agent_002", "ThematicAnalysisAgent")
        self.theme_dict = THEME_KEYWORDS

    def process(self, records: list[VOCRecord]) -> list[AnalysisResult]:
        results: list[AnalysisResult] = []
        for record in records:
            text_lower = record.text.lower()
            themes: list[str] = []
            keywords_found: list[str] = []
            codes: list[str] = []

            for theme_id, theme_info in self.theme_dict.items():
                matched = False
                for kw in theme_info["keywords"]:
                    if kw.lower() in text_lower:
                        matched = True
                        keywords_found.append(kw)
                if matched:
                    themes.append(theme_id)
                    codes.append(f"CODE_{theme_id.upper()}")

            # 情感分析
            sentiment, score = self._analyze_sentiment(record.text)

            result = AnalysisResult(
                record_id=record.record_id,
                themes=themes,
                sentiment=sentiment,
                sentiment_score=score,
                codes=codes,
                keywords=list(set(keywords_found)),
            )
            results.append(result)

        self.log("thematic_analysis", {
            "records_processed": len(results),
            "themes_found": len(set(t for r in results for t in r.themes)),
        })
        return results

    def _analyze_sentiment(self, text: str) -> tuple[str, float]:
        """基于词典的情感分析"""
        text_lower = text.lower()
        pos_count = sum(1 for w in SENTIMENT_DICT["positive"] if w.lower() in text_lower)
        neg_count = sum(1 for w in SENTIMENT_DICT["negative"] if w.lower() in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return "neutral", 0.0

        score = (pos_count - neg_count) / max(total, 1)
        if score > 0.1:
            return "positive", round(score, 3)
        elif score < -0.1:
            return "negative", round(score, 3)
        return "neutral", round(score, 3)


class CodebookAgent(BaseAgent):
    """
    Agent 5: 编码手册Agent（对应论文的Agent Codebook）
    基于分析结果生成结构化编码手册
    """

    def __init__(self) -> None:
        super().__init__("agent_005", "CodebookAgent")

    def process(
        self, records: list[VOCRecord], results: list[AnalysisResult]
    ) -> dict[str, Any]:
        codebook: dict[str, Any] = {
            "version": "1.0",
            "generated_at": "2026-04-29",
            "codes": {},
        }

        # 统计每个code的频率和分布
        code_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "sentiment": Counter(), "sources": Counter(), "examples": []}
        )

        for record, result in zip(records, results):
            for code in result.codes:
                code_stats[code]["count"] += 1
                code_stats[code]["sentiment"][result.sentiment] += 1
                code_stats[code]["sources"][record.source] += 1
                if len(code_stats[code]["examples"]) < 3:
                    code_stats[code]["examples"].append(record.text[:100])

        for code, stats in code_stats.items():
            theme_id = code.replace("CODE_", "").lower()
            theme_info = THEME_KEYWORDS.get(theme_id, {})
            codebook["codes"][code] = {
                "description": theme_info.get("description", ""),
                "keywords": theme_info.get("keywords", []),
                "frequency": stats["count"],
                "sentiment_distribution": dict(stats["sentiment"]),
                "source_distribution": dict(stats["sources"]),
                "examples": stats["examples"],
            }

        self.log("codebook_generation", {"codes_defined": len(codebook["codes"])})
        return codebook


class PatternRecognitionAgent(BaseAgent):
    """
    Agent 6-7: 模式识别Agent（对应论文的Pattern Recognition + Verify）
    识别主题共现、趋势和异常
    """

    def __init__(self) -> None:
        super().__init__("agent_006", "PatternRecognitionAgent")

    def process(
        self, records: list[VOCRecord], results: list[AnalysisResult]
    ) -> list[Pattern]:
        patterns: list[Pattern] = []

        # 1. 主题共现模式
        cooccurrence = self._find_cooccurrence_patterns(results)
        patterns.extend(cooccurrence)

        # 2. 情感-主题关联模式
        sentiment_theme = self._find_sentiment_theme_patterns(results)
        patterns.extend(sentiment_theme)

        # 3. 来源差异模式
        source_patterns = self._find_source_patterns(records, results)
        patterns.extend(source_patterns)

        self.log("pattern_recognition", {"patterns_found": len(patterns)})
        return patterns

    def _find_cooccurrence_patterns(
        self, results: list[AnalysisResult]
    ) -> list[Pattern]:
        """发现经常同时出现的主题对"""
        cooc: dict[tuple[str, str], int] = Counter()
        for r in results:
            themes = sorted(r.themes)
            for i in range(len(themes)):
                for j in range(i + 1, len(themes)):
                    pair = (themes[i], themes[j])
                    cooc[pair] += 1

        patterns: list[Pattern] = []
        for (t1, t2), count in cooc.most_common(5):
            if count >= 2:
                patterns.append(Pattern(
                    pattern_type="co_occurrence",
                    description=f"'{t1}' and '{t2}' frequently co-occur ({count} times)",
                    supporting_evidence=[f"{t1}+{t2}"],
                    confidence=min(1.0, count / max(len(results) * 0.1, 1)),
                ))
        return patterns

    def _find_sentiment_theme_patterns(
        self, results: list[AnalysisResult]
    ) -> list[Pattern]:
        """发现特定主题的情感偏向"""
        theme_sentiment: dict[str, Counter] = defaultdict(Counter)
        for r in results:
            for theme in r.themes:
                theme_sentiment[theme][r.sentiment] += 1

        patterns: list[Pattern] = []
        for theme, sentiment_counts in theme_sentiment.items():
            total = sum(sentiment_counts.values())
            if total < 2:
                continue
            neg_ratio = sentiment_counts.get("negative", 0) / total
            pos_ratio = sentiment_counts.get("positive", 0) / total

            if neg_ratio > 0.6:
                patterns.append(Pattern(
                    pattern_type="trend",
                    description=f"'{theme}' is predominantly negative ({neg_ratio*100:.0f}%)",
                    supporting_evidence=[f"neg={sentiment_counts.get('negative',0)}/total={total}"],
                    confidence=neg_ratio,
                ))
            elif pos_ratio > 0.7:
                patterns.append(Pattern(
                    pattern_type="trend",
                    description=f"'{theme}' is predominantly positive ({pos_ratio*100:.0f}%)",
                    supporting_evidence=[f"pos={sentiment_counts.get('positive',0)}/total={total}"],
                    confidence=pos_ratio,
                ))
        return patterns

    def _find_source_patterns(
        self, records: list[VOCRecord], results: list[AnalysisResult]
    ) -> list[Pattern]:
        """发现不同来源的数据差异"""
        source_sentiment: dict[str, Counter] = defaultdict(Counter)
        for record, result in zip(records, results):
            source_sentiment[record.source][result.sentiment] += 1

        patterns: list[Pattern] = []
        for source, sentiment_counts in source_sentiment.items():
            total = sum(sentiment_counts.values())
            neg_ratio = sentiment_counts.get("negative", 0) / max(total, 1)
            if neg_ratio > 0.5 and total >= 3:
                patterns.append(Pattern(
                    pattern_type="anomaly",
                    description=f"Source '{source}' has high negative rate ({neg_ratio*100:.0f}%)",
                    supporting_evidence=[f"source={source}, neg_ratio={neg_ratio:.2f}"],
                    confidence=neg_ratio,
                ))
        return patterns


class InsightSynthesisAgent(BaseAgent):
    """
    Agent 8: 洞察综合Agent
    从模式中提取可行动的洞察
    """

    def __init__(self) -> None:
        super().__init__("agent_008", "InsightSynthesisAgent")

    def process(
        self,
        records: list[VOCRecord],
        results: list[AnalysisResult],
        patterns: list[Pattern],
        codebook: dict[str, Any],
    ) -> list[Insight]:
        insights: list[Insight] = []

        # 1. 从负面主题中提取痛点
        pain_points = self._extract_pain_points(results)
        insights.extend(pain_points)

        # 2. 从正面主题中提取机会
        opportunities = self._extract_opportunities(results)
        insights.extend(opportunities)

        # 3. 从模式中提取风险
        risks = self._extract_risks(patterns, results)
        insights.extend(risks)

        self.log("insight_synthesis", {"insights_generated": len(insights)})
        return insights

    def _extract_pain_points(self, results: list[AnalysisResult]) -> list[Insight]:
        """提取高频负面主题作为痛点"""
        theme_neg_counts: dict[str, list[str]] = defaultdict(list)
        for r in results:
            if r.sentiment == "negative":
                for theme in r.themes:
                    theme_neg_counts[theme].append(r.record_id)

        insights: list[Insight] = []
        for theme, record_ids in sorted(theme_neg_counts.items(), key=lambda x: -len(x[1]))[:3]:
            theme_name = theme.replace("_", " ").title()
            insights.append(Insight(
                insight_id=f"PAIN_{theme.upper()}",
                insight_type="pain_point",
                title=f"High negative sentiment in '{theme_name}'",
                description=f"Users frequently express dissatisfaction related to {theme_name}",
                evidence_count=len(record_ids),
                affected_records=record_ids[:10],
                severity=min(10, len(record_ids) * 2),
                action_suggestion=f"Investigate {theme_name} issues and develop improvement plan",
            ))
        return insights

    def _extract_opportunities(self, results: list[AnalysisResult]) -> list[Insight]:
        """提取高频正面主题作为机会"""
        theme_pos_counts: dict[str, list[str]] = defaultdict(list)
        for r in results:
            if r.sentiment == "positive":
                for theme in r.themes:
                    theme_pos_counts[theme].append(r.record_id)

        insights: list[Insight] = []
        for theme, record_ids in sorted(theme_pos_counts.items(), key=lambda x: -len(x[1]))[:2]:
            theme_name = theme.replace("_", " ").title()
            insights.append(Insight(
                insight_id=f"OPP_{theme.upper()}",
                insight_type="opportunity",
                title=f"Strong positive reception for '{theme_name}'",
                description=f"Users consistently praise {theme_name} - leverage in marketing",
                evidence_count=len(record_ids),
                affected_records=record_ids[:10],
                severity=3.0,
                action_suggestion=f"Highlight {theme_name} in product messaging",
            ))
        return insights

    def _extract_risks(
        self, patterns: list[Pattern], results: list[AnalysisResult]
    ) -> list[Insight]:
        """从异常模式中提取风险"""
        insights: list[Insight] = []
        for pattern in patterns:
            if pattern.pattern_type == "anomaly":
                insights.append(Insight(
                    insight_id=f"RISK_{len(insights):03d}",
                    insight_type="risk",
                    title=f"Anomaly: {pattern.description[:50]}",
                    description=pattern.description,
                    evidence_count=len(pattern.supporting_evidence),
                    affected_records=[],
                    severity=min(10, pattern.confidence * 10),
                    action_suggestion="Investigate root cause of anomaly",
                ))
        return insights


class QualityVerificationAgent(BaseAgent):
    """
    Agent 9: 质量验证Agent（对应论文的Agent Verify）
    检查分析结果的一致性和完整性
    """

    def __init__(self) -> None:
        super().__init__("agent_009", "QualityVerificationAgent")

    def process(
        self,
        records: list[VOCRecord],
        results: list[AnalysisResult],
        insights: list[Insight],
    ) -> dict[str, Any]:
        checks = {
            "coverage": self._check_coverage(records, results),
            "consistency": self._check_consistency(results),
            "insight_validity": self._check_insight_validity(insights, results),
        }

        # 综合质量分
        scores = [checks["coverage"]["score"], checks["consistency"]["score"], checks["insight_validity"]["score"]]
        overall = round(sum(scores) / len(scores), 2)

        self.log("quality_verification", {"overall_score": overall, "checks": list(checks.keys())})
        return {
            "overall_score": overall,
            "checks": checks,
            "passed": overall >= 0.6,
        }

    def _check_coverage(
        self, records: list[VOCRecord], results: list[AnalysisResult]
    ) -> dict[str, Any]:
        """检查覆盖率"""
        analyzed = len(results)
        total = len(records)
        coverage = analyzed / max(total, 1)

        # 检查未分类记录比例
        uncategorized = sum(1 for r in results if not r.themes)
        uncategorized_ratio = uncategorized / max(analyzed, 1)

        score = coverage * 0.5 + (1 - uncategorized_ratio) * 0.5
        return {
            "score": round(score, 2),
            "coverage_ratio": round(coverage, 2),
            "uncategorized_ratio": round(uncategorized_ratio, 2),
            "issues": ["High uncategorized ratio"] if uncategorized_ratio > 0.3 else [],
        }

    def _check_consistency(self, results: list[AnalysisResult]) -> dict[str, Any]:
        """检查一致性"""
        # 检查情感分数与标签的一致性
        inconsistent = 0
        for r in results:
            if r.sentiment == "positive" and r.sentiment_score < 0:
                inconsistent += 1
            elif r.sentiment == "negative" and r.sentiment_score > 0:
                inconsistent += 1

        inconsistency_ratio = inconsistent / max(len(results), 1)
        score = 1 - inconsistency_ratio
        return {
            "score": round(score, 2),
            "inconsistency_ratio": round(inconsistency_ratio, 2),
            "issues": ["Sentiment-label inconsistency detected"] if inconsistency_ratio > 0.1 else [],
        }

    def _check_insight_validity(
        self, insights: list[Insight], results: list[AnalysisResult]
    ) -> dict[str, Any]:
        """检查洞察有效性"""
        valid_insights = sum(1 for i in insights if i.evidence_count >= 2)
        total_insights = len(insights)
        validity = valid_insights / max(total_insights, 1)
        return {
            "score": round(validity, 2),
            "valid_insights": valid_insights,
            "total_insights": total_insights,
            "issues": ["Low insight validity"] if validity < 0.5 else [],
        }


class ReportGeneratorAgent(BaseAgent):
    """
    Agent 10: 报告生成Agent（对应论文的Agent Finalize）
    生成最终结构化报告
    """

    def __init__(self) -> None:
        super().__init__("agent_010", "ReportGeneratorAgent")

    def process(
        self,
        dataset_name: str,
        records: list[VOCRecord],
        results: list[AnalysisResult],
        codebook: dict[str, Any],
        patterns: list[Pattern],
        insights: list[Insight],
        quality: dict[str, Any],
    ) -> VOCAnalysisReport:
        # 生成主题聚类
        theme_clusters = self._build_theme_clusters(records, results)

        # 情感摘要
        sentiment_summary = self._build_sentiment_summary(results)

        report = VOCAnalysisReport(
            report_id=f"voc_report_{dataset_name}",
            dataset_name=dataset_name,
            total_records=len(records),
            theme_clusters=theme_clusters,
            sentiment_summary=sentiment_summary,
            patterns=patterns,
            insights=insights,
            codebook=codebook,
            quality_score=quality["overall_score"],
            raw_results=results,
        )

        self.log("report_generation", {
            "report_id": report.report_id,
            "themes": len(theme_clusters),
            "insights": len(insights),
        })
        return report

    def _build_theme_clusters(
        self, records: list[VOCRecord], results: list[AnalysisResult]
    ) -> list[ThemeCluster]:
        """构建主题聚类"""
        theme_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"record_ids": [], "sentiment": Counter(), "quotes": []}
        )

        record_map = {r.record_id: r for r in records}
        for result in results:
            for theme in result.themes:
                theme_data[theme]["record_ids"].append(result.record_id)
                theme_data[theme]["sentiment"][result.sentiment] += 1
                record = record_map.get(result.record_id)
                if record and len(theme_data[theme]["quotes"]) < 3:
                    # 取包含关键词的句子作为代表性引用
                    sentences = record.text.split(".")
                    for s in sentences:
                        if any(kw in s.lower() for kw in THEME_KEYWORDS.get(theme, {}).get("keywords", [])):
                            theme_data[theme]["quotes"].append(s.strip())
                            break

        clusters: list[ThemeCluster] = []
        for theme_id, data in theme_data.items():
            theme_info = THEME_KEYWORDS.get(theme_id, {})
            clusters.append(ThemeCluster(
                theme_id=theme_id,
                theme_name=theme_id.replace("_", " ").title(),
                description=theme_info.get("description", ""),
                keywords=theme_info.get("keywords", [])[:5],
                record_ids=data["record_ids"],
                sentiment_distribution=dict(data["sentiment"]),
                representative_quotes=data["quotes"][:3],
            ))

        # 按记录数排序
        clusters.sort(key=lambda c: len(c.record_ids), reverse=True)
        return clusters

    def _build_sentiment_summary(self, results: list[AnalysisResult]) -> dict[str, Any]:
        """构建情感摘要"""
        sentiment_counts = Counter(r.sentiment for r in results)
        total = len(results)
        avg_score = sum(r.sentiment_score for r in results) / max(total, 1)

        return {
            "distribution": dict(sentiment_counts),
            "percentages": {
                k: round(v / max(total, 1) * 100, 1)
                for k, v in sentiment_counts.items()
            },
            "average_score": round(avg_score, 3),
            "total_analyzed": total,
        }


# ────────────────────────── 主协调管道 ──────────────────────────

class MultiAgentVOCPipeline:
    """
    多智能体VOC分析管道

    协调10个Agent完成端到端VOC分析：
    DataIngestion → ThematicAnalysis → Codebook → PatternRecognition → InsightSynthesis → QualityVerification → ReportGenerator
    """

    def __init__(self) -> None:
        self.agents: list[BaseAgent] = [
            DataIngestionAgent(),
            ThematicAnalysisAgent(),
            CodebookAgent(),
            PatternRecognitionAgent(),
            InsightSynthesisAgent(),
            QualityVerificationAgent(),
            ReportGeneratorAgent(),
        ]

    def run(self, raw_data: list[dict[str, Any]], dataset_name: str = "voc_dataset") -> VOCAnalysisReport:
        """运行完整分析管道"""
        print(f"\n{'='*60}")
        print(f"Multi-Agent VOC Analysis Pipeline")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        # Step 1: 数据摄入
        print(f"\n[Step 1/7] Data Ingestion...")
        records = self.agents[0].process(raw_data)
        print(f"   Cleaned {len(records)} records")

        # Step 2: 主题分析
        print(f"\n[Step 2/7] Thematic Analysis...")
        analysis_results = self.agents[1].process(records)
        theme_count = len(set(t for r in analysis_results for t in r.themes))
        print(f"   Identified {theme_count} unique themes across {len(analysis_results)} records")

        # Step 3: 编码手册
        print(f"\n[Step 3/7] Codebook Generation...")
        codebook = self.agents[2].process(records, analysis_results)
        print(f"   Generated codebook with {len(codebook.get('codes', {}))} codes")

        # Step 4: 模式识别
        print(f"\n[Step 4/7] Pattern Recognition...")
        patterns = self.agents[3].process(records, analysis_results)
        print(f"   Found {len(patterns)} patterns")

        # Step 5: 洞察综合
        print(f"\n[Step 5/7] Insight Synthesis...")
        insights = self.agents[4].process(records, analysis_results, patterns, codebook)
        print(f"   Generated {len(insights)} insights")

        # Step 6: 质量验证
        print(f"\n[Step 6/7] Quality Verification...")
        quality = self.agents[5].process(records, analysis_results, insights)
        print(f"   Overall quality score: {quality['overall_score']:.2f}")
        print(f"   Passed: {quality['passed']}")

        # Step 7: 报告生成
        print(f"\n[Step 7/7] Report Generation...")
        report = self.agents[6].process(
            dataset_name, records, analysis_results, codebook, patterns, insights, quality
        )
        print(f"   Report generated: {report.report_id}")

        print(f"\n{'='*60}")
        print(f"Pipeline Complete")
        print(f"{'='*60}")

        return report


# ────────────────────────── 工厂方法和演示 ──────────────────────────

def create_demo_voc_data() -> list[dict[str, Any]]:
    """创建演示VOC数据（Momcozy吸奶器场景）"""
    return [
        {
            "id": "voc_001",
            "text": "I absolutely love this pump! The suction is strong and I can get 150ml in 10 minutes. However, the noise is a bit loud when using it at work.",
            "source": "amazon",
            "rating": 4.0,
            "category": "product",
        },
        {
            "id": "voc_002",
            "text": "Battery life is amazing, lasts all day. Very portable and easy to carry in my bag. Cleaning is a bit tedious with all the small parts.",
            "source": "amazon",
            "rating": 4.0,
            "category": "product",
        },
        {
            "id": "voc_003",
            "text": "Terrible experience. The pump stopped working after 2 weeks. Customer service was unresponsive. Had to return it.",
            "source": "trustpilot",
            "rating": 1.0,
            "category": "service",
        },
        {
            "id": "voc_004",
            "text": "Great value for the price. Much cheaper than Medela but works just as well. Suction is comfortable and not painful.",
            "source": "reddit",
            "rating": 5.0,
            "category": "product",
        },
        {
            "id": "voc_005",
            "text": "Shipping was fast but the packaging was damaged. The product itself works fine though. No complaints about the suction power.",
            "source": "amazon",
            "rating": 3.0,
            "category": "shipping",
        },
        {
            "id": "voc_006",
            "text": "Love how quiet this is compared to my old pump. I can use it during conference calls. The portable design is perfect for traveling.",
            "source": "amazon",
            "rating": 5.0,
            "category": "product",
        },
        {
            "id": "voc_007",
            "text": "The suction is too weak for me. I have oversupply and this pump can't keep up. Also the battery dies quickly.",
            "source": "reddit",
            "rating": 2.0,
            "category": "product",
        },
        {
            "id": "voc_008",
            "text": "Customer service was helpful when I had questions about assembly. Quick response and friendly support team.",
            "source": "zendesk",
            "rating": 5.0,
            "category": "service",
        },
        {
            "id": "voc_009",
            "text": "Design is sleek and modern. The LED display is a nice touch. Easy to clean compared to other pumps I've tried.",
            "source": "amazon",
            "rating": 4.0,
            "category": "product",
        },
        {
            "id": "voc_010",
            "text": "Very disappointed. Expected better quality for this price. The motor makes weird noises and suction is inconsistent.",
            "source": "trustpilot",
            "rating": 2.0,
            "category": "product",
        },
        {
            "id": "voc_011",
            "text": "这个吸奶器吸力很强，10分钟就能吸150ml。就是噪音有点大，在公司用有点尴尬。",
            "source": "amazon",
            "rating": 4.0,
            "category": "product",
        },
        {
            "id": "voc_012",
            "text": "电池续航很好，用一整天没问题。很便携，放包里不占地方。清洗小零件有点麻烦。",
            "source": "amazon",
            "rating": 4.0,
            "category": "product",
        },
        {
            "id": "voc_013",
            "text": "太糟糕了，用了两周就坏了。客服也不回复。只能退货。",
            "source": "trustpilot",
            "rating": 1.0,
            "category": "service",
        },
        {
            "id": "voc_014",
            "text": "性价比很高，比美德乐便宜但效果一样好。吸力舒适不疼。",
            "source": "reddit",
            "rating": 5.0,
            "category": "product",
        },
        {
            "id": "voc_015",
            "text": "物流很快但包装破损。产品本身没问题，吸力很好。",
            "source": "amazon",
            "rating": 3.0,
            "category": "shipping",
        },
        {
            "id": "voc_016",
            "text": "比我旧泵安静多了，开会时也能用。便携设计出差用很合适。",
            "source": "amazon",
            "rating": 5.0,
            "category": "product",
        },
        {
            "id": "voc_017",
            "text": "吸力对我太弱了，我奶多这个泵跟不上。电池也不耐用。",
            "source": "reddit",
            "rating": 2.0,
            "category": "product",
        },
        {
            "id": "voc_018",
            "text": "客服很耐心，解答了我关于组装的问题。回复快态度好。",
            "source": "zendesk",
            "rating": 5.0,
            "category": "service",
        },
        {
            "id": "voc_019",
            "text": "设计很时尚，LED显示屏很贴心。比我用过的其他泵好清洗。",
            "source": "amazon",
            "rating": 4.0,
            "category": "product",
        },
        {
            "id": "voc_020",
            "text": "很失望，这个价位期望更好的质量。电机有异响，吸力不稳定。",
            "source": "trustpilot",
            "rating": 2.0,
            "category": "product",
        },
    ]


def print_report(report: VOCAnalysisReport) -> None:
    """打印结构化报告"""
    print(f"\n{'='*60}")
    print(f"VOC ANALYSIS REPORT: {report.dataset_name}")
    print(f"{'='*60}")

    print(f"\n📊 Overview")
    print(f"   Total Records: {report.total_records}")
    print(f"   Quality Score: {report.quality_score:.2f}")

    print(f"\n💭 Sentiment Summary")
    for sentiment, count in report.sentiment_summary.get("distribution", {}).items():
        pct = report.sentiment_summary.get("percentages", {}).get(sentiment, 0)
        print(f"   {sentiment.capitalize()}: {count} ({pct}%)")
    print(f"   Average Score: {report.sentiment_summary.get('average_score', 0):.3f}")

    print(f"\n📌 Top Themes")
    for cluster in report.theme_clusters[:5]:
        total = sum(cluster.sentiment_distribution.values())
        neg = cluster.sentiment_distribution.get("negative", 0)
        pos = cluster.sentiment_distribution.get("positive", 0)
        print(f"   {cluster.theme_name}: {len(cluster.record_ids)} mentions")
        print(f"      Sentiment: +{pos}/-{neg}")
        if cluster.representative_quotes:
            print(f"      Quote: \"{cluster.representative_quotes[0][:80]}...\"")

    print(f"\n🔍 Patterns Detected ({len(report.patterns)})")
    for p in report.patterns[:5]:
        print(f"   [{p.pattern_type}] {p.description}")
        print(f"      Confidence: {p.confidence:.2f}")

    print(f"\n💡 Key Insights ({len(report.insights)})")
    for insight in report.insights[:5]:
        emoji = {"pain_point": "🔴", "opportunity": "🟢", "trend": "🔵", "risk": "🟡"}.get(insight.insight_type, "⚪")
        print(f"   {emoji} [{insight.insight_type.upper()}] {insight.title}")
        print(f"      Evidence: {insight.evidence_count} records")
        print(f"      Severity: {insight.severity:.1f}/10")
        if insight.action_suggestion:
            print(f"      Action: {insight.action_suggestion}")

    print(f"\n{'='*60}")


# ────────────────────────── 主流程 ──────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent VOC Data Analyst")
    print("Based on: arXiv:2402.01386")
    print("Scenario: Momcozy Cross-border M&B VOC Analysis")
    print("=" * 60)

    # 创建演示数据
    demo_data = create_demo_voc_data()

    # 运行管道
    pipeline = MultiAgentVOCPipeline()
    report = pipeline.run(demo_data, dataset_name="momcozy_voc_q2_2026")

    # 打印报告
    print_report(report)

    # 导出JSON
    output = {
        "report_id": report.report_id,
        "dataset_name": report.dataset_name,
        "total_records": report.total_records,
        "quality_score": report.quality_score,
        "sentiment_summary": report.sentiment_summary,
        "themes": [
            {
                "name": c.theme_name,
                "mentions": len(c.record_ids),
                "sentiment": c.sentiment_distribution,
                "keywords": c.keywords,
            }
            for c in report.theme_clusters
        ],
        "insights": [
            {
                "type": i.insight_type,
                "title": i.title,
                "severity": i.severity,
                "action": i.action_suggestion,
            }
            for i in report.insights
        ],
    }

    print(f"\n📄 JSON Export (summary):")
    print(json.dumps(output, indent=2, ensure_ascii=False)[:800] + "...")

    print("\n" + "=" * 60)
    print("Analysis complete. Verify: no errors raised.")
    print("=" * 60)
