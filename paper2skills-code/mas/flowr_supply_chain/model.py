"""
Flowr Supply Chain MAS — 多 Agent 供应链协作框架（母婴出海版）
arXiv:2604.05987 | Python 3.14+ | 仅标准库，无需额外安装

参考论文: Flowr -- Scaling Up Retail Supply Chain Operations Through Agentic AI
作者: Eranga Bandara et al. (2026-04)
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class SKU:
    sku_id: str
    name: str
    current_stock: int        # 件
    daily_sales_avg: float    # 件/日
    lead_time_days: int       # 补货提前期（天）
    moq: int                  # 最小起订量
    unit_cost: float          # 元/件


@dataclass
class AgentResult:
    agent: str
    status: str               # "ok" | "warning" | "error"
    data: dict[str, Any]
    message: str = ""


@dataclass
class HITLGate:
    """人工监督门控记录"""
    gate_id: str
    agent: str
    decision_summary: str
    approved: bool = False
    approver: str = ""


# ── 6 个专业化 Agent ─────────────────────────────────────────────────────────

class DemandForecastingAgent:
    """需求预测 Agent — 基于移动平均 + 季节因子（生产环境可替换为 TFT/Prophet）"""

    def run(self, sku: SKU, horizon_days: int = 30, seasonality: float = 1.0) -> AgentResult:
        forecast_daily = sku.daily_sales_avg * seasonality
        forecast_total = round(forecast_daily * horizon_days)
        ci_lower = round(forecast_total * 0.8)
        ci_upper = round(forecast_total * 1.25)
        return AgentResult(
            agent="DemandForecasting",
            status="ok",
            data={
                "horizon_days": horizon_days,
                "forecast_total": forecast_total,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "daily_rate": round(forecast_daily, 2),
            },
            message=f"预测 {horizon_days} 天总需求: {forecast_total} 件 (80% CI: [{ci_lower}, {ci_upper}])",
        )


class InventoryMonitoringAgent:
    """库存监控 Agent — 计算库存覆盖天数与缺货风险"""

    RISK_THRESHOLD_DAYS = 14  # 低于 14 天覆盖则预警

    def run(self, sku: SKU, demand_result: AgentResult) -> AgentResult:
        daily_rate = demand_result.data["daily_rate"]
        coverage_days = sku.current_stock / daily_rate if daily_rate > 0 else 999
        reorder_point = daily_rate * sku.lead_time_days * 1.3  # 安全系数 1.3

        risk = "high" if coverage_days < sku.lead_time_days else \
               "medium" if coverage_days < self.RISK_THRESHOLD_DAYS else "low"

        return AgentResult(
            agent="InventoryMonitoring",
            status="warning" if risk != "low" else "ok",
            data={
                "coverage_days": round(coverage_days, 1),
                "reorder_point": round(reorder_point),
                "risk_level": risk,
                "current_stock": sku.current_stock,
            },
            message=f"库存覆盖 {coverage_days:.1f} 天，风险等级: {risk.upper()}",
        )


class ProcurementOrderingAgent:
    """采购订购 Agent — MOQ 批量优化，计算经济订购量"""

    def run(self, sku: SKU, demand_result: AgentResult, inventory_result: AgentResult) -> AgentResult:
        forecast_total = demand_result.data["forecast_total"]
        current_stock = inventory_result.data["current_stock"]
        net_demand = max(0, forecast_total - current_stock)

        # MOQ 向上取整
        if net_demand <= 0:
            order_qty = 0
            status = "ok"
            msg = "库存充足，无需补货"
        else:
            batches = -(-net_demand // sku.moq)  # ceiling division
            order_qty = batches * sku.moq
            order_cost = order_qty * sku.unit_cost
            status = "ok"
            msg = f"建议采购 {order_qty} 件（{batches} 批 × MOQ {sku.moq}），预计成本 ¥{order_cost:,.0f}"

        return AgentResult(
            agent="ProcurementOrdering",
            status=status,
            data={
                "net_demand": net_demand,
                "order_qty": order_qty,
                "order_cost": order_qty * sku.unit_cost,
            },
            message=msg,
        )


class SupplierCoordinationAgent:
    """供应商协调 Agent — 生成采购确认草稿（生产环境接邮件/ERP API）"""

    def run(self, sku: SKU, procurement_result: AgentResult) -> AgentResult:
        order_qty = procurement_result.data["order_qty"]
        if order_qty == 0:
            return AgentResult(
                agent="SupplierCoordination",
                status="ok",
                data={"draft": None},
                message="无采购需求，跳过供应商协调",
            )

        delivery_date = (date.today() + timedelta(days=sku.lead_time_days)).isoformat()
        draft = (
            f"[采购草稿] SKU: {sku.sku_id} ({sku.name})\n"
            f"数量: {order_qty} 件 | 单价: ¥{sku.unit_cost} | "
            f"总额: ¥{order_qty * sku.unit_cost:,.0f}\n"
            f"要求到货日期: {delivery_date}\n"
            f"[待人工审核后发送供应商]"
        )
        return AgentResult(
            agent="SupplierCoordination",
            status="ok",
            data={"draft": draft, "delivery_date": delivery_date},
            message=f"供应商函草稿已生成，需 HITL 审核后发送",
        )


class DCReplenishmentPlanningAgent:
    """DC 补货规划 Agent — 海外仓分仓补货分配"""

    WAREHOUSES = ["US-LA", "US-NJ", "DE-FRA"]  # 海外仓节点

    def run(self, sku: SKU, procurement_result: AgentResult) -> AgentResult:
        order_qty = procurement_result.data["order_qty"]
        if order_qty == 0:
            return AgentResult(
                agent="DCReplenishmentPlanning",
                status="ok",
                data={"allocation": {}},
                message="无库存需分配",
            )

        # 简化分配策略：按固定比例（生产环境接实时需求权重）
        ratios = [0.5, 0.3, 0.2]
        allocation = {
            wh: round(order_qty * r)
            for wh, r in zip(self.WAREHOUSES, ratios)
        }
        # 修正取整误差
        diff = order_qty - sum(allocation.values())
        allocation[self.WAREHOUSES[0]] += diff

        return AgentResult(
            agent="DCReplenishmentPlanning",
            status="ok",
            data={"allocation": allocation, "total": order_qty},
            message=f"仓库分配方案: {allocation}",
        )


class ExceptionAlertAgent:
    """异常预警 Agent — 断货风险 / 质量异常 / 超阈值预警"""

    def run(
        self,
        sku: SKU,
        inventory_result: AgentResult,
        procurement_result: AgentResult,
        review_score: float = 4.5,
        return_rate: float = 0.02,
    ) -> AgentResult:
        alerts: list[str] = []

        if inventory_result.data["risk_level"] == "high":
            alerts.append(f"断货风险: 库存覆盖仅 {inventory_result.data['coverage_days']} 天，低于提前期 {sku.lead_time_days} 天")

        if review_score < 4.0:
            alerts.append(f"评分下滑: 当前评分 {review_score:.1f}，建议启动质量排查")

        if return_rate > 0.05:
            alerts.append(f"退货率异常: {return_rate:.1%}，超过阈值 5%")

        if not alerts:
            alerts.append("无紧急异常，常规补货")

        status = "error" if inventory_result.data["risk_level"] == "high" else \
                 "warning" if (review_score < 4.0 or return_rate > 0.05) else "ok"

        return AgentResult(
            agent="ExceptionAlert",
            status=status,
            data={"alerts": alerts, "alert_count": len(alerts)},
            message=" | ".join(alerts),
        )


# ── Orchestrator（含 HITL 门控）────────────────────────────────────────────────

class FlowrOrchestrator:
    """
    Flowr 中央编排器 — 协调 6 个专业化 Agent，管理 HITL 门控
    生产环境中 central_llm 为 GPT-4 级别 Reasoning LLM，此处使用规则代理
    """

    def __init__(self, auto_approve_hitl: bool = False):
        self.demand_agent = DemandForecastingAgent()
        self.inventory_agent = InventoryMonitoringAgent()
        self.procurement_agent = ProcurementOrderingAgent()
        self.supplier_agent = SupplierCoordinationAgent()
        self.dc_agent = DCReplenishmentPlanningAgent()
        self.exception_agent = ExceptionAlertAgent()
        self.auto_approve_hitl = auto_approve_hitl  # True=测试模式，False=生产模式
        self.hitl_log: list[HITLGate] = []

    def _hitl_gate(self, gate_id: str, agent: str, summary: str) -> bool:
        """HITL 门控：生产中展示给供应链经理，测试中自动批准"""
        gate = HITLGate(gate_id=gate_id, agent=agent, decision_summary=summary)
        if self.auto_approve_hitl:
            gate.approved = True
            gate.approver = "AUTO_TEST"
        else:
            # 生产环境：此处集成 MCP 接口，等待人工确认
            print(f"\n[HITL] {gate_id}: {summary}")
            resp = input("  供应链经理审批 (y/n): ").strip().lower()
            gate.approved = resp == "y"
            gate.approver = "MANAGER"

        self.hitl_log.append(gate)
        return gate.approved

    def run(self, sku: SKU, horizon_days: int = 30, review_score: float = 4.5, return_rate: float = 0.02) -> dict:
        """
        执行完整的 Flowr 供应链工作流
        Returns: 各 Agent 结果汇总 + HITL 日志 + 最终决策
        """
        results: dict[str, AgentResult] = {}

        # Step 1: 需求预测
        results["demand"] = self.demand_agent.run(sku, horizon_days)

        # Step 2: 库存监控
        results["inventory"] = self.inventory_agent.run(sku, results["demand"])

        # Step 3: 采购订购
        results["procurement"] = self.procurement_agent.run(sku, results["demand"], results["inventory"])

        # Step 4: HITL 门控 - 采购确认（采购额 > 5 万元时强制审批）
        order_cost = results["procurement"].data["order_cost"]
        if order_cost > 50000:
            approved = self._hitl_gate(
                gate_id="PROCUREMENT_APPROVAL",
                agent="ProcurementOrdering",
                summary=f"采购 {results['procurement'].data['order_qty']} 件，总额 ¥{order_cost:,.0f}，请确认",
            )
            if not approved:
                results["procurement"].data["order_qty"] = 0
                results["procurement"].message = "[人工取消] 采购订单未获批准"

        # Step 5: 供应商协调
        results["supplier"] = self.supplier_agent.run(sku, results["procurement"])

        # Step 6: HITL 门控 - 供应商发函审批
        if results["supplier"].data.get("draft"):
            approved = self._hitl_gate(
                gate_id="SUPPLIER_EMAIL_APPROVAL",
                agent="SupplierCoordination",
                summary=f"供应商确认函，要求 {results['supplier'].data['delivery_date']} 到货",
            )
            if not approved:
                results["supplier"].message = "[人工取消] 供应商函未发送"

        # Step 7: DC 补货规划
        results["dc"] = self.dc_agent.run(sku, results["procurement"])

        # Step 8: 异常预警
        results["exception"] = self.exception_agent.run(
            sku, results["inventory"], results["procurement"], review_score, return_rate
        )

        return {
            "sku_id": sku.sku_id,
            "results": {k: {"status": v.status, "message": v.message, "data": v.data}
                        for k, v in results.items()},
            "hitl_log": [{"gate_id": g.gate_id, "approved": g.approved, "approver": g.approver}
                         for g in self.hitl_log],
            "overall_status": "error" if any(v.status == "error" for v in results.values()) else
                              "warning" if any(v.status == "warning" for v in results.values()) else "ok",
        }


# ── 测试用例（模拟母婴 SKU 补货场景）─────────────────────────────────────────

def test_flowr_normal_replenishment():
    """测试常规补货场景（库存充足，无风险）"""
    sku = SKU(
        sku_id="BB-STROLLER-001",
        name="轻便折叠婴儿车 A 款",
        current_stock=800,
        daily_sales_avg=15.0,
        lead_time_days=25,
        moq=100,
        unit_cost=280.0,
    )
    orchestrator = FlowrOrchestrator(auto_approve_hitl=True)
    result = orchestrator.run(sku, horizon_days=30, review_score=4.7, return_rate=0.02)

    assert result["overall_status"] in ("ok", "warning"), f"预期 ok/warning，实际: {result['overall_status']}"
    assert result["results"]["demand"]["data"]["forecast_total"] > 0
    # 库存 800 件，日销 15 件 → 覆盖 53 天，无缺货风险
    assert result["results"]["inventory"]["data"]["coverage_days"] > 14, "库存覆盖应大于 14 天"
    print(f"[PASS] 常规场景: overall={result['overall_status']}, "
          f"coverage={result['results']['inventory']['data']['coverage_days']} 天")
    return result


def test_flowr_stockout_risk():
    """测试高断货风险场景（库存仅剩 5 天覆盖）"""
    sku = SKU(
        sku_id="BB-DIAPER-002",
        name="超薄纸尿裤 L 码",
        current_stock=120,
        daily_sales_avg=25.0,
        lead_time_days=20,
        moq=500,
        unit_cost=45.0,
    )
    orchestrator = FlowrOrchestrator(auto_approve_hitl=True)
    result = orchestrator.run(sku, horizon_days=30, review_score=4.2, return_rate=0.03)

    assert result["results"]["inventory"]["data"]["risk_level"] == "high", "应检测到高风险"
    assert result["results"]["procurement"]["data"]["order_qty"] > 0, "应生成采购订单"
    alerts = result["results"]["exception"]["data"]["alerts"]
    assert any("断货" in a for a in alerts), "应有断货预警"
    print(f"[PASS] 断货风险场景: overall={result['overall_status']}, "
          f"order_qty={result['results']['procurement']['data']['order_qty']}")
    return result


def test_flowr_quality_alert():
    """测试质量异常场景（评分下滑 + 高退货率）"""
    sku = SKU(
        sku_id="BB-BOTTLE-003",
        name="耐热玻璃奶瓶 240ml",
        current_stock=600,
        daily_sales_avg=8.0,
        lead_time_days=18,
        moq=200,
        unit_cost=65.0,
    )
    orchestrator = FlowrOrchestrator(auto_approve_hitl=True)
    result = orchestrator.run(sku, horizon_days=30, review_score=3.6, return_rate=0.09)

    alerts = result["results"]["exception"]["data"]["alerts"]
    assert any("评分" in a for a in alerts), "应有评分下滑预警"
    assert any("退货" in a for a in alerts), "应有退货率预警"
    print(f"[PASS] 质量异常场景: alerts={len(alerts)} 条, status={result['overall_status']}")
    return result


def test_flowr_batch_run():
    """批量测试：模拟 5 个母婴 SKU 同时运行"""
    skus_data = [
        ("BB-WIPES-001", "婴儿湿巾 80 片", 3000, 120.0, 15, 1000, 8.5),
        ("BB-FORMULA-001", "有机奶粉 800g", 450, 18.0, 30, 50, 320.0),
        ("BB-SEAT-001", "安全座椅 0-4 岁", 60, 2.5, 45, 20, 980.0),
        ("BB-TOY-001", "早教益智积木套装", 200, 12.0, 20, 100, 85.0),
        ("BB-MONITOR-001", "婴儿监视器 WiFi 版", 35, 3.0, 35, 10, 560.0),
    ]
    orchestrator = FlowrOrchestrator(auto_approve_hitl=True)
    all_results = []
    for sku_id, name, stock, daily, lead, moq, cost in skus_data:
        sku = SKU(sku_id, name, stock, daily, lead, moq, cost)
        r = orchestrator.run(sku)
        all_results.append((sku_id, r["overall_status"]))

    assert len(all_results) == 5, "应处理 5 个 SKU"
    status_map = {sku_id: status for sku_id, status in all_results}
    print(f"[PASS] 批量运行: {status_map}")
    return all_results


if __name__ == "__main__":
    print("=" * 60)
    print("Flowr Supply Chain MAS — 母婴出海补货场景测试")
    print("=" * 60)

    r1 = test_flowr_normal_replenishment()
    r2 = test_flowr_stockout_risk()
    r3 = test_flowr_quality_alert()
    r4 = test_flowr_batch_run()

    print("\n" + "=" * 60)
    print("全部 4 个测试通过")
    print("\n[断货风险场景] 详细输出:")
    print(json.dumps(r2, ensure_ascii=False, indent=2))
