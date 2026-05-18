"""SelectionAgent: WF-D 选品扫描 Agent.

输入 payload.candidates: list[{id, name, category, monthly_sales_usd, review_count,
  avg_rating, bsr_trend_30d, selling_price_usd, cogs_usd, fba_fee_usd, freight_usd,
  duty_rate, description, seasonality_factor, competitor_count}]
输出: 评分排序的 top_picks (estimated_cost=0,选品不直接花钱,只产生候选清单)
"""

from __future__ import annotations

from typing import Any, Dict

from mas.skills.selection_tools import run_full_wfd_analysis
from mas.state.schema import WorkflowContext


SOFT_BUDGET_RESERVE = 500


class SelectionAgent:
    name = "selection_agent"
    role_description = "母婴跨境选品 Agent: 市场空间 + 毛利 + 合规 + KG 属性 + 因果 lift"

    def __call__(self, state: WorkflowContext) -> Dict[str, Any]:
        remaining = state.get("token_budget", 0) - state.get("token_usage", 0)
        if remaining < SOFT_BUDGET_RESERVE:
            return {
                "skill_outputs": [{"agent": self.name, "status": "skipped_budget_low"}],
                "error": "token_budget_low",
            }

        candidates = state.get("payload", {}).get("candidates", [])
        analysis = run_full_wfd_analysis(candidates, top_n=10)

        msg = self._format_decision(analysis)
        return {
            "messages": [{"role": "assistant", "content": msg, "agent": self.name}],
            "skill_outputs": [{
                "agent": self.name,
                "skill_name": "wfd_chain[market+margin+compliance+kgqa+causal]",
                "output": analysis,
                "confidence": analysis["confidence"],
                "estimated_cost": 0.0,
                "status": "ok",
            }],
            "token_usage": state.get("token_usage", 0) + 1200,
        }

    @staticmethod
    def _format_decision(analysis: Dict[str, Any]) -> str:
        top = analysis["top_picks"]
        lines = [
            f"【选品扫描报告】共 {analysis['total_candidates']} 候选, 推荐 {analysis['recommend_count']} 个进入下一轮",
            "Top 5 候选:",
        ]
        for i, p in enumerate(top[:5], 1):
            lines.append(
                f"  {i}. {p['name']} (id={p['candidate_id']}) 评分 {p['composite_score']} "
                f"| 毛利率 {p['margin']['margin_pct']:.1%} | 合规 {p['compliance']['risk_level']}"
            )
        lines.append(f"  置信度: {analysis['confidence']}")
        return "\n".join(lines)
