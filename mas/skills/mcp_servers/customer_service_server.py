from __future__ import annotations

from mas.skills.mcp_servers.base import BaseMCPServer
from mas.skills.customer_service_tools import (
    classify_case_intent,
    sentiment_analysis,
    graphrag_lookup_faq,
    journey_tree_decision,
    cluster_review_themes,
    detect_language,
)


class CustomerServiceServer(BaseMCPServer):
    server_name = "customer_service_server"
    domain = "customer_service"

    def _register_tools(self) -> None:
        self._add("case_intent_classifier", "Case 意图分类,退款/换货/投诉/咨询/其他",
                  lambda **kw: classify_case_intent(kw.get("text", "")))
        self._add("sentiment_analysis", "情感分析,规则版负面/正面/中性打分",
                  lambda **kw: sentiment_analysis(kw.get("text", "")))
        self._add("kg_graphrag", "GraphRAG FAQ 检索,按意图+语种返回回复模板",
                  lambda **kw: graphrag_lookup_faq(kw.get("intent", "other"),
                                                   kw.get("language", "en")))
        self._add("data_customer_journey_tree", "决策树分诊,自动批准/人工审核/转人工",
                  lambda **kw: journey_tree_decision(
                      kw.get("intent", "other"),
                      int(kw.get("days_since_order", 0)),
                      int(kw.get("user_complaint_history", 0)),
                      float(kw.get("order_amount", 0)),
                  ))
        self._add("review_theme_clustering", "Review 主题聚类,品质/物流/气味/价值五维",
                  lambda **kw: cluster_review_themes(kw.get("reviews", [])))
        self._add("language_detection", "语种检测,支持 zh/ja/de/fr/es/en",
                  lambda **kw: {"language": detect_language(kw.get("text", ""))})
