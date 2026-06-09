from __future__ import annotations

from mas.skills.mcp_servers.base import BaseMCPServer
from mas.skills.marketing_tools import (
    parse_search_term_report,
    identify_negative_keywords,
    uplift_keyword_promotion,
    dara_cross_channel_budget,
    mmm_channel_elasticity,
)


class AdvertisingServer(BaseMCPServer):
    server_name = "advertising_server"
    domain = "advertising"

    def _register_tools(self) -> None:
        self._add("ad_search_term_parse", "搜索词报告解析,词级 ROAS/TACOS 汇总",
                  lambda **kw: parse_search_term_report(kw.get("rows", [])))
        self._add("ad_negative_keywords", "负向关键词识别,低转化高消耗词建议否定",
                  lambda **kw: identify_negative_keywords(kw.get("rows", []),
                                                          kw.get("click_threshold", 10),
                                                          kw.get("conversion_threshold", 0.0)))
        self._add("ad_roas_budget_optimization", "高 ROAS 词提价建议,Uplift 思想最优 bid",
                  lambda **kw: uplift_keyword_promotion(kw.get("rows", []),
                                                        kw.get("roas_threshold", 4.0),
                                                        kw.get("min_orders", 3)))
        self._add("marketing_dara_optimizer", "DARA 跨渠道预算分配,边际 ROAS 均衡迭代",
                  lambda **kw: dara_cross_channel_budget(kw.get("history", []),
                                                         float(kw.get("total_budget", 10000))))
        self._add("marketing_mmm", "MMM 渠道弹性估计,log-log 回归",
                  lambda **kw: mmm_channel_elasticity(kw.get("history", [])))
        self._add("ad_attribution_modeling", "多触点广告归因",
                  lambda **kw: {"skill": "ad_attribution_modeling", "status": "stub_ok", **kw})
        self._add("ad_compliance_guardrail", "Amazon ToS 合规护栏,三层过滤",
                  lambda **kw: {"skill": "ad_compliance_guardrail", "status": "stub_ok", **kw})
