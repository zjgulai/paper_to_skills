"""WF-C 客服 Case 分诊 + WF-E Review 监控 Skill 工具.

涵盖:
  - Case 分类 (退款/换货/咨询/投诉/差评)
  - 语种检测 + 情感分析 (规则版)
  - 决策树 (复用 customer_journey_tree 思想)
  - GraphRAG 知识检索 (stub)
  - Review 情感聚类 + 差评归因
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


REFUND_KEYWORDS = {"refund", "退款", "money back", "return", "退货", "remboursement"}
EXCHANGE_KEYWORDS = {"exchange", "换货", "swap", "replace", "替换"}
COMPLAINT_KEYWORDS = {"defective", "broken", "leak", "缺陷", "破损", "投诉", "complaint", "damaged"}
INQUIRY_KEYWORDS = {"how", "怎么", "what", "什么", "when", "时候", "where", "comment", "qué"}
NEGATIVE_KEYWORDS = {"awful", "terrible", "差", "烂", "失望", "horrible", "worst", "very bad"}

LANG_PATTERNS = {
    "zh": re.compile(r"[\u4e00-\u9fff]"),
    "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]"),
    "de": re.compile(r"\b(ist|der|die|das|nicht|sehr|sehr)\b", re.I),
    "fr": re.compile(r"\b(le|la|les|est|très|n'est)\b", re.I),
    "es": re.compile(r"\b(el|la|es|muy|no|está)\b", re.I),
}


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    for lang, pat in LANG_PATTERNS.items():
        if pat.search(text):
            return lang
    return "en"


def classify_case_intent(text: str) -> Dict[str, Any]:
    text_low = text.lower() if text else ""

    if any(kw.lower() in text_low for kw in REFUND_KEYWORDS):
        intent = "refund"
    elif any(kw.lower() in text_low for kw in EXCHANGE_KEYWORDS):
        intent = "exchange"
    elif any(kw.lower() in text_low for kw in COMPLAINT_KEYWORDS):
        intent = "complaint"
    elif any(kw.lower() in text_low for kw in INQUIRY_KEYWORDS):
        intent = "inquiry"
    else:
        intent = "other"

    return {"skill": "case_intent_classifier", "intent": intent, "confidence": 0.85}


def sentiment_analysis(text: str) -> Dict[str, Any]:
    if not text:
        return {"sentiment": "neutral", "score": 0.0}
    text_low = text.lower()
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw.lower() in text_low)
    pos_signals = sum(1 for kw in ["love", "great", "perfect", "excellent", "好", "棒"] if kw in text_low)

    if neg_count > pos_signals:
        sentiment = "negative"
        score = -min(1.0, 0.3 + neg_count * 0.2)
    elif pos_signals > neg_count:
        sentiment = "positive"
        score = min(1.0, 0.3 + pos_signals * 0.2)
    else:
        sentiment = "neutral"
        score = 0.0

    return {"skill": "sentiment_analysis", "sentiment": sentiment, "score": round(score, 2)}


def graphrag_lookup_faq(intent: str, language: str) -> Dict[str, Any]:
    faq_db = {
        ("refund", "en"): "Refunds for unopened items are processed within 7 business days under our 30-day policy.",
        ("refund", "zh"): "未拆封商品在 30 天内可申请退款,7 个工作日内到账。",
        ("exchange", "en"): "Exchange is available within 30 days; please share order number and product images.",
        ("exchange", "zh"): "30 天内可申请换货,请提供订单号和商品图片。",
        ("complaint", "en"): "We are sorry to hear about the issue. Please provide images and we will investigate within 24h.",
        ("complaint", "zh"): "非常抱歉给您带来困扰,请提供问题图片,我们将在 24 小时内调查。",
        ("inquiry", "en"): "We are happy to help. Please share more details about your question.",
        ("inquiry", "zh"): "很乐意帮您解答,请告知您的具体问题。",
        ("other", "en"): "Could you please provide more details so we can better assist you?",
        ("other", "zh"): "请您提供更多信息以便我们更好地协助您。",
    }
    snippet = faq_db.get((intent, language)) or faq_db.get((intent, "en")) or "We will follow up shortly."
    return {"skill": "kg_graphrag", "snippet": snippet, "confidence": 0.8}


def journey_tree_decision(
    intent: str,
    days_since_order: int,
    user_complaint_history: int,
    order_amount: float,
) -> Dict[str, Any]:
    if intent == "refund":
        if order_amount > 500:
            action = "manual_review"
            response_template = "退款金额较大,已转专员审核,24 小时内回复。"
            estimated_cost = order_amount
        elif days_since_order <= 7:
            action = "auto_approve"
            response_template = "7 天内全额退款已批准,款项 1-3 工作日到账。"
            estimated_cost = order_amount
        elif days_since_order <= 30 and user_complaint_history <= 2:
            action = "auto_approve"
            response_template = "30 天内退款已批准,款项 1-3 工作日到账。"
            estimated_cost = order_amount
        else:
            action = "transfer_human"
            response_template = "为您转接资深客服处理。"
            estimated_cost = 0
    elif intent == "exchange":
        action = "auto_approve" if days_since_order <= 30 else "transfer_human"
        response_template = "换货申请已批准,预计 7-15 天到货。" if action == "auto_approve" else "为您转接人工。"
        estimated_cost = 0
    elif intent == "complaint":
        action = "manual_review" if user_complaint_history >= 2 else "auto_response_with_followup"
        response_template = "非常抱歉,我们会立即跟进处理。"
        estimated_cost = 0
    else:
        action = "auto_response"
        response_template = "感谢您的咨询。"
        estimated_cost = 0

    return {
        "skill": "data_customer_journey_tree",
        "action": action,
        "response_template": response_template,
        "estimated_cost": estimated_cost,
        "confidence": 0.88,
    }


def run_full_wfc_analysis(case: Dict[str, Any]) -> Dict[str, Any]:
    text = case.get("text", "")
    language = detect_language(text)
    intent_result = classify_case_intent(text)
    sentiment_result = sentiment_analysis(text)
    faq_snippet = graphrag_lookup_faq(intent_result["intent"], language)

    decision = journey_tree_decision(
        intent=intent_result["intent"],
        days_since_order=int(case.get("days_since_order", 0)),
        user_complaint_history=int(case.get("user_complaint_history", 0)),
        order_amount=float(case.get("order_amount", 0)),
    )

    risk_level = "high" if (sentiment_result["score"] <= -0.5 or decision["estimated_cost"] > 500) else "low"

    overall_conf = (intent_result["confidence"] + faq_snippet["confidence"] + decision["confidence"]) / 3

    return {
        "case_id": case.get("case_id", "unknown"),
        "language": language,
        "intent": intent_result["intent"],
        "sentiment": sentiment_result,
        "faq_snippet": faq_snippet["snippet"],
        "decision": decision,
        "risk_level": risk_level,
        "estimated_cost": decision["estimated_cost"],
        "skill_chain": [
            "case_intent_classifier",
            "sentiment_analysis",
            "kg_graphrag",
            "data_customer_journey_tree",
        ],
        "confidence": round(overall_conf, 3),
    }


def cluster_review_themes(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    theme_keywords = {
        "quality": {"defective", "broken", "leak", "缺陷", "破损", "damage"},
        "fit": {"size", "small", "large", "尺码", "fit"},
        "shipping": {"slow", "late", "shipping", "物流", "迟"},
        "smell": {"smell", "odor", "气味"},
        "value": {"price", "expensive", "贵", "value"},
    }
    counts: Dict[str, int] = {k: 0 for k in theme_keywords}
    for r in reviews:
        text_low = (r.get("text", "") or "").lower()
        for theme, kws in theme_keywords.items():
            if any(k in text_low for k in kws):
                counts[theme] += 1

    negative_reviews = [r for r in reviews if int(r.get("rating", 5)) <= 2]
    avg_rating = sum(int(r.get("rating", 5)) for r in reviews) / max(len(reviews), 1)

    return {
        "skill": "review_theme_clustering",
        "themes": counts,
        "total_reviews": len(reviews),
        "negative_count": len(negative_reviews),
        "avg_rating": round(avg_rating, 2),
        "top_theme": max(counts, key=counts.get) if counts else None,
        "confidence": 0.82,
    }
