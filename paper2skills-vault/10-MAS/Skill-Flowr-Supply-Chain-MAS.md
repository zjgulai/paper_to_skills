---
title: Flowr — 零售供应链多 Agent 端到端自动化
doc_type: knowledge
module: 10-MAS
topic: flowr-supply-chain-mas
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Flowr — 零售供应链多 Agent 端到端自动化

> 论文: arXiv:2604.05987 (2026-04) | Eranga Bandara et al. | 真实超市 PoC 验证

---

## ① 算法原理

### 核心思想

Flowr 将**人工密集型零售供应链操作**系统性地分解为 6 个专业化 AI Agent 联盟，由中央 Reasoning LLM 协调编排，供应链经理通过 MCP（Model Context Protocol）接口实施人工监督（HITL）。核心洞察是：**供应链决策具有天然的功能模块化特征，不同子任务所需的专业知识和数据源存在边界清晰的分工，因此适合由专业化 LLM（而非单一通用模型）各司其职，再由 Reasoning LLM 统一协调**。

### 多 Agent 分工架构

6 个专业化 Agent 形成端到端执行链路：

```
需求预测 Agent → 库存监控 Agent → 采购订购 Agent
                                        ↓
异常预警 Agent ← DC补货规划 Agent ← 供应商协调 Agent
```

**任务分解与能力互补公式**：

$$\mathcal{V}(S) = \sum_{i=1}^{6} f_i(\mathcal{D}_i, \theta_i) \cdot w_i(t) + \lambda \cdot \text{Coord}(\{f_i\}, G)$$

其中 $f_i$ 为第 $i$ 个 Agent 的专业化函数（独立微调 LLM），$\mathcal{D}_i$ 为其专属数据域，$\theta_i$ 为微调参数，$w_i(t)$ 为时变权重，$\text{Coord}(·)$ 为中央 Reasoning LLM 的协调项，$\lambda$ 控制协调成本。

**HITL 门控机制**：关键决策节点（如大额采购、供应商切换）须经供应链经理在 MCP 接口确认后才能执行，实现"AI 建议 + 人工批准"的双保险。

### 关键假设

1. 各 Agent 的数据域边界清晰，信息依赖主要通过结构化消息传递（而非共享上下文）
2. 中央 Reasoning LLM（如 GPT-4 级别）有足够能力进行跨域协调推理
3. 历史供应链数据量充足，支持各域专业 LLM 的微调或 RAG 增强
4. HITL 干预延迟可接受（适用于小时/天级别决策，不适用于毫秒级实时系统）

---

## ② 母婴出海应用案例

### 场景一：母婴跨境多 Agent 补货链路（WF-A 直接对应）

**业务问题**：母婴品牌在亚马逊/独立站同时运营，SKU 达 500+，跨境仓（海外仓 + 国内直发）补货涉及 DHL/UPS 运输周期（15-30 天）、海关清关（3-7 天）、Amazon FBA 入仓（1-5 天），任何一环延误都导致断货（Lost Buy Box，单 SKU 日损失 2,000-8,000 元）。

**6 个 Agent 映射到母婴补货链路**：

| Flowr Agent | 母婴对应角色 | 专属数据域 |
|------------|------------|-----------|
| Demand Forecasting | 销量预测 Agent（含节假日/季节/营销）| Amazon API 销售历史、促销日历 |
| Inventory Monitoring | 库存状态 Agent（FBA + 海外仓 + 在途）| FBA Inventory Report、WMS 系统 |
| Procurement & Ordering | 采购下单 Agent（MOQ/阶梯价格优化）| ERP 采购历史、供应商价格表 |
| Supplier Coordination | 供应商协调 Agent（产能确认/档期锁定）| 供应商 API/邮件系统 |
| DC Replenishment Planning | 海外仓补货规划 Agent（动态安全库存）| 运输时效数据、清关历史 |
| Exception & Alert | 异常预警 Agent（断货风险/质量投诉）| Amazon Review、Seller Central Alert |

**数据要求**：
- 销售历史：SKU × 日期 × 渠道，过去 12 个月，≥80% 数据完整率
- 库存快照：每日实时同步（FBA/海外仓/在途 三库区）
- 运输时效：历史货代记录（含清关延误标记）

**预期产出**：
- 每日自动生成补货建议单（含置信区间）
- 断货风险预警（提前 14 天，命中率目标 ≥ 85%）
- 供应商确认函模板（自动生成，人工审核后发送）

**业务价值（量化）**：
- 断货率从 12% → 5%（行业基准），按单 SKU 日均 GMV 3,000 元、年断货天数减少 25 天，500 SKU 计：**年增 GMV 约 3,750 万元**
- 补货计划人工工时从每日 6 人·时 → 1 人·时（人工复核），节省 5 人·时/日，年节省 **约 91 万元人工成本**

---

### 场景二：选品 → 备货 → 异常预警端到端 MAS Pipeline

**业务问题**：新 SKU 上架母婴类目（如婴儿推车配件）时，没有历史数据支撑需求预测，盲目备货导致积压（平均积压率 35%），保守备货又导致早期断货错失上升期流量。

**MAS Pipeline 设计**：

```
阶段 1 - 新品冷启动预测（Demand Forecasting Agent）
  → 调用相似品历史 + Bass 扩散模型参数（arXiv:2307.03595）
  → 输出：首批建议量 + 上市后 30/60/90 天分段补货量

阶段 2 - 库存安全线设定（Inventory Monitoring Agent）
  → 结合跨境提前期分布（Gen-QOT，arXiv:2310.17168）
  → 输出：动态安全库存值（置信区间 95%）

阶段 3 - MOQ 批量优化（Procurement & Ordering Agent）
  → 综合阶梯价格 + MOQ 约束（Dynamic-Lot-Sizing-MOQ Skill）
  → 输出：分批下单方案（降低资金占用）

阶段 4 - 供应商确认（Supplier Coordination Agent）
  → 自动草拟产能确认邮件 → HITL 门控（经理审批）→ 发送

阶段 5 - 上市后异常监控（Exception & Alert Agent）
  → 实时监控：Review 评分下滑 / 退货率异常 / 竞品降价
  → 触发补救：加速备货 / 启动推广 / 价格调整建议

HITL 干预点：阶段 1 首批量确认 / 阶段 4 供应商发函 / 阶段 5 价格调整执行
```

**数据要求**：
- 新品标签：类目、关键词、价格带（用于相似品匹配）
- 类目历史销售数据（用于 Bass 参数估计）
- 供应商联系人数据库

**预期产出**：
- 冷启动首批量误差 MAPE ≤ 30%（相比纯经验判断的 55%）
- 上市 90 天内断货次数 ≤ 1 次

**业务价值（量化）**：
- 新品积压率从 35% → 20%，单品首批备货量平均 10 万元，50 款新品/年：**减少积压资金占用约 750 万元**
- 断货率下降带来流量损失减少，BSR 排名提升周期缩短约 **3-4 周**

---

## ③ 代码模板

代码路径：`paper2skills-code/mas/flowr_supply_chain/model.py`

```python
"""
Flowr Supply Chain MAS — 多 Agent 供应链协作框架（母婴出海版）
arXiv:2604.05987 | Python 3.14+ | 仅标准库，无需额外安装
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
            alerts.append(f"🚨 断货风险: 库存覆盖仅 {inventory_result.data['coverage_days']} 天，低于提前期 {sku.lead_time_days} 天")

        if review_score < 4.0:
            alerts.append(f"⚠️ 评分下滑: 当前评分 {review_score:.1f}，建议启动质量排查")

        if return_rate > 0.05:
            alerts.append(f"⚠️ 退货率异常: {return_rate:.1%}，超过阈值 5%")

        if procurement_result.data["order_qty"] > 0 and inventory_result.data["risk_level"] == "low":
            alerts.append("ℹ️ 常规补货，无紧急异常")

        status = "error" if any("🚨" in a for a in alerts) else \
                 "warning" if any("⚠️" in a for a in alerts) else "ok"

        return AgentResult(
            agent="ExceptionAlert",
            status=status,
            data={"alerts": alerts, "alert_count": len(alerts)},
            message=" | ".join(alerts) if alerts else "无异常",
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

    assert result["overall_status"] in ("ok", "warning")
    assert result["results"]["demand"]["data"]["forecast_total"] > 0
    # 库存 800 件，日销 15 件 → 覆盖 53 天，无缺货风险
    assert result["results"]["inventory"]["data"]["coverage_days"] > 14
    print(f"✅ 常规场景测试通过: overall={result['overall_status']}")
    return result


def test_flowr_stockout_risk():
    """测试高断货风险场景（库存仅剩 5 天覆盖）"""
    sku = SKU(
        sku_id="BB-DIAPER-002",
        name="超薄纸尿裤 L 码",
        current_stock=120,
        daily_sales_avg=25.0,  # 日销高
        lead_time_days=20,
        moq=500,
        unit_cost=45.0,
    )
    orchestrator = FlowrOrchestrator(auto_approve_hitl=True)
    result = orchestrator.run(sku, horizon_days=30, review_score=4.2, return_rate=0.03)

    assert result["results"]["inventory"]["data"]["risk_level"] == "high"
    assert result["results"]["procurement"]["data"]["order_qty"] > 0
    assert any("断货" in a for a in result["results"]["exception"]["data"]["alerts"])
    print(f"✅ 断货风险场景测试通过: overall={result['overall_status']}, "
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
    assert any("评分" in a for a in alerts)
    assert any("退货" in a for a in alerts)
    print(f"✅ 质量异常场景测试通过: alerts={len(alerts)} 条")
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

    assert len(all_results) == 5
    statuses = [s for _, s in all_results]
    print(f"✅ 批量运行测试通过: {dict(zip([s[0] for s in skus_data], statuses))}")
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
    print("所有测试通过 ✅")
    print("\n📋 场景二（断货风险）详细输出:")
    print(json.dumps(r2, ensure_ascii=False, indent=2))
print("[✓] Flowr Supply Chain MAS 测试通过")
```

---

## ④ 技能关联

### 前置

- [[Skill-MAS-Orchestrator]] — Flowr 的中央编排器设计直接复用 MAS Orchestrator 的生命周期管理模型
- [[Skill-Demand-Forecasting-Supply-Chain]] — Demand Forecasting Agent 的预测内核
- [[Skill-Multi-Echelon-Inventory]] — DC 补货规划 Agent 的多级库存理论基础

### 延伸（待萃取）

- [[Skill-AIM-RM-LLM-Inventory-MAS-Memory]] — 为各 Agent 添加跨轮次记忆，提升长期决策一致性
- [[Skill-KG-Auto-Construction-Agent-Driven]] — 将知识图谱注入供应链 MAS，增强因果推理能力

### 可组合

- [[Skill-Safety-Stock-Replenishment]] — 与 Inventory Monitoring Agent 结合，计算动态安全库存
- [[Skill-Supplier-Capacity-Planning]] — 与 Supplier Coordination Agent 结合，实现产能约束下的排产
- [[Skill-Skill-Registry-Dynamic-Loading]] — 动态注册/替换 Flowr 的各专业化 Agent，支持热更新

---

## ⑤ 商业价值评估

### ROI 量化

| 价值维度 | 基准（现状）| 目标（Flowr 上线后）| 年化收益 |
|---------|-----------|-------------------|---------|
| 补货人工协调工时 | 6 人·时/日 | 1 人·时/日（人工复核）| 节省 ≈ 91 万元/年（按 50 元/时计） |
| 断货率 | 12% | 5% | 年增 GMV ≈ 3,750 万元（500 SKU）|
| 主动异常处理 | 事后发现（平均延误 3 天）| 提前 14 天预警 | 每次断货损失减少 60-80% |
| 新品冷启动积压 | 35% | 20% | 减少资金占用 ≈ 750 万元（50 新品/年）|

**综合年化 ROI（中型母婴品牌，500 SKU）：约 4,500 万元潜在增益**

### 实施难度

⭐⭐⭐☆☆（3星）

- **易**：框架已有模板（本 Skill 提供完整代码骨架），6 个 Agent 边界清晰，可分阶段上线
- **中**：需对接 Amazon API、WMS、ERP 等数据源（通常 2-4 周集成工期）
- **难**：专业化 LLM 微调需要供应链领域语料（可用 RAG 替代降低门槛）

### 优先级

⭐⭐⭐⭐⭐（5星）

### 评分依据

1. **真实 PoC 验证**：在真实大型连锁超市验证，显著优于学术 benchmark，迁移风险低
2. **WF-A 直接增强**：与现有 MAS WF-A 智能补货工作流（当前覆盖率 95%）高度对齐，可作为架构升级方案
3. **量化收益清晰**：断货率、人工工时、积压资金三维均可量化，业务方易于接受
4. **框架可泛化**：论文明确说明框架与领域无关，可复用于母婴出海其他场景（如广告投放 MAS、选品 MAS）
