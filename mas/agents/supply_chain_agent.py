"""SupplyChainAgent: WF-A 智能补货决策 Agent.

职责:
  - 读取 WorkflowContext.payload (SKU 列表 + 历史销售数据)
  - 依序调用 supply_chain_tools 完成需求预测 → 反事实 → 安全库存 → 补货建议
  - 输出 estimated_cost 供 HITL gate 判断
"""

from __future__ import annotations

from typing import Any, Dict

from mas.skills.supply_chain_tools import run_full_wfa_analysis
from mas.state.schema import WorkflowContext


SOFT_BUDGET_RESERVE = 500


class SupplyChainAgent:
    name = "supply_chain_agent"
    role_description = "母婴跨境电商供应链决策 Agent: 需求预测 + 反事实 + 安全库存 + 补货建议"

    def __call__(self, state: WorkflowContext) -> Dict[str, Any]:
        remaining = state.get("token_budget", 0) - state.get("token_usage", 0)
        if remaining < SOFT_BUDGET_RESERVE:
            return {
                "skill_outputs": [{"agent": self.name, "status": "skipped_budget_low"}],
                "error": "token_budget_low",
            }

        payload = state.get("payload", {})
        sku_id = payload.get("sku_id", "UNKNOWN")
        history = payload.get("history_daily_sales", [])
        current_stock = int(payload.get("current_stock", 0))
        in_transit = int(payload.get("in_transit", 0))
        lead_time_days = float(payload.get("lead_time_days", 30))
        season_multiplier = float(payload.get("season_multiplier", 1.0))
        interventions = payload.get("interventions", [])
        moq = int(payload.get("moq", 500))
        unit_cost_rmb = float(payload.get("unit_cost_rmb", 50.0))
        service_level = float(payload.get("service_level", 0.95))

        analysis = run_full_wfa_analysis(
            sku_id=sku_id,
            history_daily_sales=history,
            current_stock=current_stock,
            in_transit=in_transit,
            lead_time_days=lead_time_days,
            season_multiplier=season_multiplier,
            interventions=interventions,
            moq=moq,
            unit_cost_rmb=unit_cost_rmb,
            service_level=service_level,
        )

        rec = analysis["recommendation"]
        msg_content = self._format_decision_message(analysis)

        return {
            "messages": [{"role": "assistant", "content": msg_content, "agent": self.name}],
            "skill_outputs": [{
                "agent": self.name,
                "skill_name": "wfa_chain[demand_forecast+gcf+safety+drl]",
                "output": analysis,
                "confidence": analysis["confidence"],
                "estimated_cost": rec["estimated_cost"],
                "status": "ok",
            }],
            "token_usage": state.get("token_usage", 0) + 800,
        }

    @staticmethod
    def _format_decision_message(analysis: Dict[str, Any]) -> str:
        rec = analysis["recommendation"]
        forecast = analysis["forecast"]
        counterfactual = analysis["counterfactual"]
        safety = analysis["safety"]

        lines = [
            f"【SKU {analysis['sku_id']} 补货分析】",
            f"  需求预测: 未来 {forecast['horizon_days']} 天 = {forecast['forecast_total']} 件 (日均 {forecast['daily_mean']})",
            f"  反事实需求 (含 {counterfactual['interventions_count']} 项干预): "
            f"日均提升 {counterfactual['intervention_lift_pct']}%",
            f"  安全库存: {safety['safety_stock']} 件, 再订货点: {safety['reorder_point']} 件 "
            f"(服务水平 {safety['service_level']})",
            f"  补货建议: {rec['recommendation']}",
            f"  补货量: {rec['recommended_qty']} 件 (MOQ={rec['moq']})",
            f"  预估成本: ¥{rec['estimated_cost']:,.0f}",
            f"  整体置信度: {analysis['confidence']}",
        ]
        return "\n".join(lines)
