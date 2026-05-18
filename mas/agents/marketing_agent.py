"""MarketingAgent: WF-B 广告关键词优化决策 Agent.

输入 payload:
  - search_term_rows: 搜索词报告 list[{search_term, clicks, orders, spend, revenue, bid}]
  - channel_history: 跨渠道历史 list[{channel, spend, revenue, roas}]
  - total_budget: 周期总预算
  - target_tacos: 目标 TACOS (默认 0.15 = 15%)

输出 estimated_cost = 否定词省下 + 加价投入,触发金额阈值 HITL.
"""

from __future__ import annotations

from typing import Any, Dict

from mas.skills.marketing_tools import run_full_wfb_analysis
from mas.state.schema import WorkflowContext


SOFT_BUDGET_RESERVE = 500


class MarketingAgent:
    name = "marketing_agent"
    role_description = "母婴跨境广告优化 Agent: 搜索词清洗 + 否定词 + 提价 + 跨渠道预算"

    def __call__(self, state: WorkflowContext) -> Dict[str, Any]:
        remaining = state.get("token_budget", 0) - state.get("token_usage", 0)
        if remaining < SOFT_BUDGET_RESERVE:
            return {
                "skill_outputs": [{"agent": self.name, "status": "skipped_budget_low"}],
                "error": "token_budget_low",
            }

        payload = state.get("payload", {})
        search_terms = payload.get("search_term_rows", [])
        channel_hist = payload.get("channel_history", [])
        total_budget = float(payload.get("total_budget", 10_000))
        target_tacos = float(payload.get("target_tacos", 0.15))

        analysis = run_full_wfb_analysis(
            search_term_rows=search_terms,
            channel_history=channel_hist,
            total_budget=total_budget,
            target_tacos=target_tacos,
        )

        msg = self._format_decision(analysis)
        return {
            "messages": [{"role": "assistant", "content": msg, "agent": self.name}],
            "skill_outputs": [{
                "agent": self.name,
                "skill_name": "wfb_chain[search_term+negatives+uplift+mmm+dara]",
                "output": analysis,
                "confidence": analysis["confidence"],
                "estimated_cost": analysis["estimated_cost"],
                "status": "ok",
            }],
            "token_usage": state.get("token_usage", 0) + 1000,
        }

    @staticmethod
    def _format_decision(analysis: Dict[str, Any]) -> str:
        s = analysis["summary"]
        neg = analysis["negatives"]
        promo = analysis["promotions"]
        dara = analysis["dara"]

        lines = [
            "【广告优化分析】",
            f"  TACOS: {s['overall_tacos']:.2%} (目标 {analysis['target_tacos']:.0%}) → {analysis['tacos_status']}",
            f"  否定词建议: {neg['negative_count']} 条, 预期节省 ¥{neg['spend_wasted']:,.0f}",
            f"  高 ROAS 词提价: {promo['promote_count']} 条",
            f"  跨渠道预算重分配 (¥{dara['total_budget']:,.0f}): {dara['allocation']}",
            f"  预估总调整金额: ¥{analysis['estimated_cost']:,.0f}",
            f"  置信度: {analysis['confidence']}",
        ]
        return "\n".join(lines)
