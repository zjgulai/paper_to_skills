from __future__ import annotations

from mas.skills.mcp_servers.base import BaseMCPServer
from mas.skills.selection_tools import (
    evaluate_market_space,
    estimate_gross_margin,
    assess_compliance_risk,
    kgqa_attribute_lookup,
    causal_demand_lift_estimate,
    score_candidate,
)


class SelectionServer(BaseMCPServer):
    server_name = "selection_server"
    domain = "selection"

    def _register_tools(self) -> None:
        self._add("selection_market_space", "市场空间评估,月销/BSR趋势/评论数/评分四维过滤",
                  lambda **kw: evaluate_market_space(kw.get("candidate", kw)))
        self._add("selection_gross_margin", "毛利可行性测算,售价-FBA-头程-关税≥40%门槛",
                  lambda **kw: estimate_gross_margin(kw.get("candidate", kw)))
        self._add("selection_compliance_risk", "母婴合规预判,高风险关键词+品类认证要求",
                  lambda **kw: assess_compliance_risk(kw.get("candidate", kw)))
        self._add("selection_kgqa_attributes", "KGQA 最畅销属性匹配,类目最优属性命中率",
                  lambda **kw: kgqa_attribute_lookup(kw.get("candidate", kw)))
        self._add("selection_causal_lift", "因果需求提升估计,BSR趋势+季节性+竞争度综合",
                  lambda **kw: causal_demand_lift_estimate(kw.get("candidate", kw)))
        self._add("selection_composite_score", "候选品综合评分排序,五维加权 composite_score",
                  lambda **kw: score_candidate(kw.get("candidate", kw)))
