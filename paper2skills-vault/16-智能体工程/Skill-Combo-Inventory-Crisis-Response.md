---
title: 库存危机响应 Combo Pattern — 断货/积压异常触发的 5 步自动响应链路
doc_type: knowledge
module: 16-智能体工程
topic: combo-inventory-crisis-response-orchestration
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 库存危机响应 Combo Pattern

> **类型**：Combo Pattern（业务解决方案编排）
> **桥梁**：16-智能体工程 ↔ 04-供应链 ↔ 03-时间序列 ↔ 18-物流履约
> **触发条件**：库存水位告警（DSI < 14天 或 库龄 > 90天积压率 > 30%），自动触发 5 步响应链路

## ① 算法原理

库存危机响应 Combo Pattern 是一种「事件驱动的应急决策自动化」范式。与常规补货计划不同，本 Combo 专门处理两类异常触发场景：

**触发类型 A（断货危机）**：DSI（库存可售天数）< 14 天且预测需求未下降  
**触发类型 B（积压危机）**：库龄 > 90 天库存占比 > 30% 且销售速度持续下滑

**5 步响应链路设计**：

```
[异常告警] → Step1 根因诊断（异常检测基础模型）
                     ↓ 异常类型标签（断货/积压/混合）
             Step2 需求重新预测（考虑季节性、促销、竞品）
                     ↓ 修正后需求曲线
             Step3 ABC 重新分级（紧急度 × 价值度重分层）
                     ↓ 优先级排序清单
             Step4 紧急补货量决策（MOQ 约束下的动态批量）
                     ↓ 补货/甩货执行指令
             Step5 最优物流路由（紧急空运 vs 海运时效对比）
                     ↓ 物流方案 + ETA
[执行闭环] ← SOP 分发给采购/运营/物流
```

**关键差异**：与标准补货 Combo 的区别在于 **时效约束**——所有决策需在 24 小时内完成，算法必须在 Speed vs Accuracy 之间优先选 Speed。因此 Step2 使用 ARIMA 快速预测而非 TFT，Step4 使用启发式 lot-sizing 而非 RL。

**数学约束**：紧急补货量 Q* = max(0, target_DSI × D̄ - I_current)，其中 target_DSI = 45天（紧急安全天数），D̄ 为修正后日均需求预测。

## ② 母婴出海应用案例

**场景A：儿童安全座椅亚马逊断货危机（大促前 21 天）**

- **业务问题**：Prime Day 前 21 天，FBA 库存 DSI = 8 天，供应商最短交期 35 天（海运），断货将错过全年最大流量窗口，损失估算 > 30 万元
- **数据要求**：当前库存量、历史 30 天销速、供应商交期分布、FBA 入库时间
- **执行过程**：
  - Step1 检测到 DSI = 8 天，异常类型 = 「断货危机」，置信度 0.94
  - Step2 重新预测：大促期间需求是平日 3.2x，修正后 DSI 实际仅 2.5 天
  - Step3 ABC 重分级：该 SKU 从 B 类提升至 A+ 类（紧急优先）
  - Step4 决策：紧急补货 500 件，分两批（空运 200 件 5 天到达 + 海运 300 件备用）
  - Step5 路由选择：深圳→LAX 空运 $4.2/kg，ETA Day 5，优于亚马逊 FBA 15 天标准入库
- **量化产出**：避免断货损失 28 万元，紧急空运成本 1.2 万元，净节省 26.8 万元

**场景B：婴儿爽身粉欧洲积压危机**

- **业务问题**：库龄 > 120 天积压 2000 件，FBA 存储费用持续累计，且即将触发长期存储费（每件 $6.9）
- **执行亮点**：Step3 重分级标记为「C 类清仓」，Step4 生成甩货方案（降价 35% + 闪购），Step5 建议将部分库存转移至第三方仓降低 FBA 费用
- **业务价值**：45 天内清库 85%，节省长期存储费 1.38 万美元

## ③ 代码模板

```python
"""
库存危机响应 Combo Pattern — 5 步自动响应链路
模拟从异常检测到物流路由的完整决策流
"""
from dataclasses import dataclass, field
from typing import Optional
import math

# ──────────────────────────────────────────────
# 库存危机上下文
# ──────────────────────────────────────────────
@dataclass
class CrisisContext:
    sku_id: str
    current_inventory: int        # 当前库存件数
    daily_sales_rate: float       # 日均销速（件/天）
    days_to_peak_demand: int      # 距离销售高峰天数（大促/旺季）
    unit_price: float             # 单价（USD）
    lead_time_sea_days: int = 35  # 海运交期
    lead_time_air_days: int = 7   # 空运交期
    air_freight_cost_per_unit: float = 5.0  # 空运单件成本
    # 各 Step 填充
    crisis_type: str = "unknown"           # Step1
    forecast_dsi: float = 0.0             # Step2
    sku_priority: str = "B"               # Step3
    replenishment_qty: int = 0            # Step4
    logistics_plan: dict = field(default_factory=dict)  # Step5
    expected_crisis_resolved: bool = False

    @property
    def current_dsi(self) -> float:
        return self.current_inventory / max(self.daily_sales_rate, 0.1)

# ──────────────────────────────────────────────
# Step 1: 异常检测 — Skill-Anomaly-Detection-Foundation-Model
# ──────────────────────────────────────────────
def step1_anomaly_detection(ctx: CrisisContext) -> CrisisContext:
    dsi = ctx.current_dsi
    if dsi < 14:
        ctx.crisis_type = "stockout_crisis"
        severity = "CRITICAL" if dsi < 7 else "HIGH"
    elif ctx.current_inventory > ctx.daily_sales_rate * 120:
        ctx.crisis_type = "overstock_crisis"
        severity = "HIGH"
    else:
        ctx.crisis_type = "normal"
        severity = "LOW"
    print(f"  [Step1] 异常检测: DSI={dsi:.1f}天, 危机类型={ctx.crisis_type}, 严重度={severity}")
    return ctx

# ──────────────────────────────────────────────
# Step 2: 需求重新预测 — Skill-Demand-Forecasting-Supply-Chain
# ──────────────────────────────────────────────
def step2_demand_reforecast(ctx: CrisisContext) -> CrisisContext:
    # 考虑大促季节性倍数（简化 ARIMA + 季节因子）
    if ctx.days_to_peak_demand <= 30:
        peak_multiplier = 3.2  # 大促期间需求 3.2x
    elif ctx.days_to_peak_demand <= 60:
        peak_multiplier = 1.8
    else:
        peak_multiplier = 1.0

    adjusted_daily_demand = ctx.daily_sales_rate * peak_multiplier
    # 修正后的真实 DSI（考虑即将到来的需求峰值）
    ctx.forecast_dsi = ctx.current_inventory / max(adjusted_daily_demand, 0.1)
    print(f"  [Step2] 需求重预测: 季节倍数={peak_multiplier}x, 修正后DSI={ctx.forecast_dsi:.1f}天")
    return ctx

# ──────────────────────────────────────────────
# Step 3: ABC 重新分级 — Skill-Dynamic-ABC-Stratification-Adaptive-Policy
# ──────────────────────────────────────────────
def step3_abc_restratification(ctx: CrisisContext) -> CrisisContext:
    # 紧急度分级：修正 DSI × 单价 × 距峰值天数
    urgency_score = (
        (1 / max(ctx.forecast_dsi, 0.5)) * 0.5 +
        (ctx.unit_price / 100) * 0.3 +
        (1 / max(ctx.days_to_peak_demand, 1)) * 0.2
    )
    if urgency_score > 0.8:
        ctx.sku_priority = "A+"   # 紧急最高优先
    elif urgency_score > 0.4:
        ctx.sku_priority = "A"
    elif urgency_score > 0.2:
        ctx.sku_priority = "B"
    else:
        ctx.sku_priority = "C"   # 积压清仓
    print(f"  [Step3] ABC 重分级: 紧急度评分={urgency_score:.3f}, 优先级={ctx.sku_priority}")
    return ctx

# ──────────────────────────────────────────────
# Step 4: 紧急补货量决策 — Skill-Dynamic-Lot-Sizing-MOQ
# ──────────────────────────────────────────────
def step4_emergency_lot_sizing(ctx: CrisisContext) -> CrisisContext:
    target_dsi = 45  # 危机解除后目标安全天数
    if ctx.crisis_type == "stockout_crisis":
        peak_demand = ctx.daily_sales_rate * 3.2
        target_stock = target_dsi * peak_demand
        ctx.replenishment_qty = max(0, int(target_stock - ctx.current_inventory))
        moq = 50  # 最小起订量
        ctx.replenishment_qty = math.ceil(ctx.replenishment_qty / moq) * moq
        action = f"紧急补货 {ctx.replenishment_qty} 件"
    elif ctx.crisis_type == "overstock_crisis":
        # 甩货：降价促销或转仓
        excess_qty = ctx.current_inventory - int(ctx.daily_sales_rate * 60)
        ctx.replenishment_qty = -excess_qty  # 负数表示需要甩货
        action = f"建议甩货/促销清仓 {abs(excess_qty)} 件"
    else:
        ctx.replenishment_qty = 0
        action = "无需紧急操作"
    print(f"  [Step4] 补货决策: {action}")
    return ctx

# ──────────────────────────────────────────────
# Step 5: 最优物流路由 — Skill-Tag-Optimized-Logistics-Routing
# ──────────────────────────────────────────────
def step5_logistics_routing(ctx: CrisisContext) -> CrisisContext:
    if ctx.replenishment_qty <= 0:
        ctx.logistics_plan = {"action": "no_replenishment_needed"}
        ctx.expected_crisis_resolved = ctx.crisis_type == "overstock_crisis"
        print(f"  [Step5] 无需补货物流")
        return ctx

    # 空运 vs 海运 ROI 对比
    qty = ctx.replenishment_qty
    air_cost = qty * ctx.air_freight_cost_per_unit
    # 断货损失：每天 = 日销量 × 单价 × 0.7 毛利
    stockout_loss_per_day = ctx.daily_sales_rate * ctx.unit_price * 0.7
    days_saved_vs_sea = ctx.lead_time_sea_days - ctx.lead_time_air_days
    air_roi = (stockout_loss_per_day * days_saved_vs_sea - air_cost) / max(air_cost, 1)

    if air_roi > 0.5 and ctx.sku_priority in ["A+", "A"]:
        mode = "AIR"
        eta_days = ctx.lead_time_air_days
        cost_usd = air_cost
    else:
        mode = "SEA"
        eta_days = ctx.lead_time_sea_days
        cost_usd = qty * 1.2  # 海运单件 $1.2

    ctx.logistics_plan = {
        "mode": mode,
        "eta_days": eta_days,
        "total_cost_usd": round(cost_usd, 2),
        "air_roi": round(air_roi, 3),
        "recommendation": f"{'空运优先，ROI合理' if mode=='AIR' else '海运成本优先，危机可控'}",
    }
    ctx.expected_crisis_resolved = (ctx.current_dsi + eta_days) > 14
    print(f"  [Step5] 物流路由: {mode}，ETA={eta_days}天，成本=${cost_usd:.0f}，空运ROI={air_roi:.2f}")
    return ctx

# ──────────────────────────────────────────────
# Combo 编排入口
# ──────────────────────────────────────────────
def run_inventory_crisis_response(
    sku_id: str,
    current_inventory: int,
    daily_sales_rate: float,
    days_to_peak_demand: int,
    unit_price: float = 25.0,
) -> CrisisContext:
    ctx = CrisisContext(
        sku_id=sku_id,
        current_inventory=current_inventory,
        daily_sales_rate=daily_sales_rate,
        days_to_peak_demand=days_to_peak_demand,
        unit_price=unit_price,
    )
    print(f"\n🚨 库存危机响应 Combo Pattern 启动: {sku_id}")
    print(f"   当前库存={current_inventory}件, 日销速={daily_sales_rate}件/天, 当前DSI={ctx.current_dsi:.1f}天")
    print("=" * 55)

    for step_fn in [step1_anomaly_detection, step2_demand_reforecast,
                    step3_abc_restratification, step4_emergency_lot_sizing,
                    step5_logistics_routing]:
        ctx = step_fn(ctx)

    print("=" * 55)
    print(f"🔚 危机响应计划: 补货={ctx.replenishment_qty}件, 物流={ctx.logistics_plan.get('mode','N/A')}, 预计危机解除={'✅' if ctx.expected_crisis_resolved else '❌'}")
    return ctx

# ──────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 场景1: 断货危机（大促前 15 天，当前库存仅剩 5 天）
    ctx1 = run_inventory_crisis_response(
        sku_id="SKU-BABY-SEAT-001",
        current_inventory=150,
        daily_sales_rate=30.0,
        days_to_peak_demand=15,
        unit_price=89.99,
    )
    assert ctx1.crisis_type == "stockout_crisis", "应识别为断货危机"
    assert ctx1.replenishment_qty > 0, "断货危机应有补货量"
    assert ctx1.logistics_plan.get("mode") in ["AIR", "SEA"], "应有物流方案"

    # 场景2: 正常库存（无危机）
    ctx2 = run_inventory_crisis_response(
        sku_id="SKU-POWDER-002",
        current_inventory=1000,
        daily_sales_rate=10.0,
        days_to_peak_demand=90,
        unit_price=15.0,
    )
    assert ctx2.crisis_type in ["normal", "overstock_crisis"], "应识别库存状态"
    print("\n[✓] 库存危机响应 Combo Pattern 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Anomaly-Detection-Foundation-Model]]（Step1 异常检测，危机类型分类）
- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]]（Step2 需求重预测，考虑季节性）
- **组合（combinable）**：[[Skill-Dynamic-ABC-Stratification-Adaptive-Policy]]（Step3 紧急 ABC 重分级）
- **组合（combinable）**：[[Skill-Dynamic-Lot-Sizing-MOQ]]（Step4 MOQ 约束下的紧急批量决策）
- **组合（combinable）**：[[Skill-Cross-Border-Logistics-Routing]]（Step5 最优物流路由，空运 vs 海运）
- **延伸（extends）**：[[Skill-Inventory-Health-Aging-Attribution]]（危机解除后库存健康度追踪）
- **延伸（extends）**：[[Skill-Multi-Echelon-Inventory]]（多级仓网的长期策略优化）
- **延伸（extends）**：[[Skill-Combo-New-Product-Launch-Playbook]]（库存健康后启动新品计划）

## ⑤ 商业价值评估

- **ROI 预估**：大促断货一次平均损失 15-50 万元，本 Combo 早期预警 + 响应将断货概率从 18% 降至 4%，按年 2 次大促计算，期望避免损失 = (18%-4%) × 30 万元 × 2 = 8.4 万元/年；积压清仓优化节省存储费约 3-8 万元/年
- **响应时效**：5 步链路 < 4 小时完成（vs 人工跨部门协调 2-3 天）
- **实施难度**：⭐⭐⭐☆☆（依赖库存系统数据接口，核心逻辑可在 2 周内工程化）
- **优先级**：⭐⭐⭐⭐⭐（库存危机是损失最直接的运营事件，ROI 极为确定）
- **适用场景**：月销 > 500 件的 SKU，或大促期间所有 A 类 SKU 自动巡检
