"""CustomerServiceAgent: 复用于 WF-C (客服分诊) 与 WF-E (Review 监控).

通过 workflow_type 区分:
  - customer_ops / case_dispatch: 走 case 分诊路径
  - review_monitor: 走 Review 聚类路径
"""

from __future__ import annotations

from typing import Any, Dict

from mas.skills.customer_service_tools import cluster_review_themes, run_full_wfc_analysis
from mas.state.schema import WorkflowContext


SOFT_BUDGET_RESERVE = 500


class CustomerServiceAgent:
    name = "customer_service_agent"
    role_description = "母婴跨境客服 Agent: Case 分诊 + 多语种回复 + Review 监控"

    def __call__(self, state: WorkflowContext) -> Dict[str, Any]:
        remaining = state.get("token_budget", 0) - state.get("token_usage", 0)
        if remaining < SOFT_BUDGET_RESERVE:
            return {
                "skill_outputs": [{"agent": self.name, "status": "skipped_budget_low"}],
                "error": "token_budget_low",
            }

        workflow_type = state.get("workflow_type", "")
        payload = state.get("payload", {})

        if workflow_type == "review_monitor":
            return self._handle_review(state, payload)
        return self._handle_case(state, payload)

    def _handle_case(self, state: WorkflowContext, payload: Dict[str, Any]) -> Dict[str, Any]:
        case = payload.get("case", payload)
        analysis = run_full_wfc_analysis(case)
        msg = self._format_case_message(analysis)

        return {
            "messages": [{"role": "assistant", "content": msg, "agent": self.name}],
            "skill_outputs": [{
                "agent": self.name,
                "skill_name": "wfc_chain[intent+sentiment+graphrag+journey_tree]",
                "output": analysis,
                "confidence": analysis["confidence"],
                "estimated_cost": analysis["estimated_cost"],
                "status": "ok",
            }],
            "token_usage": state.get("token_usage", 0) + 600,
        }

    def _handle_review(self, state: WorkflowContext, payload: Dict[str, Any]) -> Dict[str, Any]:
        reviews = payload.get("reviews", [])
        clusters = cluster_review_themes(reviews)

        negative_pct = clusters["negative_count"] / max(clusters["total_reviews"], 1) * 100
        needs_action = clusters["avg_rating"] < 4.0 or negative_pct > 20

        estimated_cost = 200.0 if needs_action else 0.0

        msg_lines = [
            f"【Review 健康度报告】共 {clusters['total_reviews']} 条评论",
            f"  平均评分: {clusters['avg_rating']} 星",
            f"  负面评论: {clusters['negative_count']} 条 ({negative_pct:.1f}%)",
            f"  主要主题: {clusters['top_theme']} (共出现 {clusters['themes'].get(clusters['top_theme'], 0)} 次)" if clusters['top_theme'] else "",
            f"  主题分布: {clusters['themes']}",
            f"  需要行动: {'是' if needs_action else '否'}",
        ]

        return {
            "messages": [{"role": "assistant", "content": "\n".join(msg_lines), "agent": self.name}],
            "skill_outputs": [{
                "agent": self.name,
                "skill_name": "review_theme_clustering",
                "output": {**clusters, "needs_action": needs_action, "negative_pct": round(negative_pct, 2)},
                "confidence": clusters["confidence"],
                "estimated_cost": estimated_cost,
                "status": "ok",
            }],
            "token_usage": state.get("token_usage", 0) + 500,
        }

    @staticmethod
    def _format_case_message(analysis: Dict[str, Any]) -> str:
        d = analysis["decision"]
        lines = [
            f"【Case {analysis['case_id']} 分诊结果】",
            f"  语言: {analysis['language']} | 意图: {analysis['intent']} | "
            f"情感: {analysis['sentiment']['sentiment']} ({analysis['sentiment']['score']})",
            f"  风险等级: {analysis['risk_level']}",
            f"  建议动作: {d['action']}",
            f"  回复模板: {d['response_template']}",
            f"  预估成本: ¥{d['estimated_cost']:,.0f}",
            f"  置信度: {analysis['confidence']}",
        ]
        return "\n".join(lines)
