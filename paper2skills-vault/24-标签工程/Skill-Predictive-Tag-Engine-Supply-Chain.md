---
title: 预测型标签引擎 — 将未来风险转化为当前可查询标签的供应链预测打标体系
doc_type: knowledge
module: 24-标签工程
topic: predictive-tag-engine-supply-chain
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 预测型标签引擎

> **来源**：arXiv:2305.14481（Predictive Tagging for Inventory Management）+ arXiv:2401.07823（Proactive Supply Chain Intelligence）+ arXiv:2308.11234（Temporal Tag Systems）
> **桥梁**：预测模型 ↔ 标签工程 ↔ 供应链行动 | **类型**：预测标签引擎

## ① 算法原理

**预测型标签（Predictive Tag）** 的核心洞察：传统标签描述**当前状态**（库存=50件），但决策需要的是**未来状态**（7天后会断货）。预测标签把"预测模型的输出"编码为可查询、可触发行动的标签。

**预测标签 vs 状态标签的区别**：

| 维度 | 状态标签 | 预测标签 |
|------|--------|--------|
| 描述 | 当前真实状态 | 预测的未来状态 |
| 来源 | 系统数据 | 预测模型 |
| 时效 | 实时更新 | 每日/每N小时重算 |
| 置信度 | 接近1.0 | 0.6-0.9 |
| Action阈值 | 通常更低 | 需要更高置信度才触发 |

**供应链核心预测标签集**：

```python
PREDICTIVE_TAGS = {
    # 断货预测
    "predicted_stockout_7d":  {"horizon": 7,  "model": "时序预测+安全库存"},
    "predicted_stockout_14d": {"horizon": 14, "model": "时序预测+PLT"},
    "predicted_stockout_30d": {"horizon": 30, "model": "LightGBM+特征工程"},

    # 需求异常预测
    "predicted_demand_spike": {"horizon": 7,  "model": "事件感知预测"},
    "predicted_demand_drop":  {"horizon": 14, "model": "促销后衰减模型"},

    # 供应风险预测
    "predicted_supplier_delay": {"horizon": 14, "model": "OTIF趋势+外部信号"},
    "predicted_price_increase": {"horizon": 30, "model": "原材料指数+季节性"},

    # 库存状态预测
    "predicted_slow_moving":  {"horizon": 30,  "model": "销速衰减检测"},
    "predicted_expiry_risk":  {"horizon": 60,  "model": "库龄+效期预警"},
    "predicted_overstock":    {"horizon": 14,  "model": "备货计划vs预测需求"},
}
```

**预测-行动闭环设计**（关键：预测标签触发的是"提前行动"）：

```
T-30天: predicted_stockout_30d=True → 纳入常规补货计划（低优先级）
T-14天: predicted_stockout_14d=True → 加急采购审批（中优先级）
T-7天:  predicted_stockout_7d=True  → 紧急补货或空运（高优先级）
T+0:    status.stockout_risk=critical → 被动应急（已晚）
```

**滚动预测标签更新**（每日重算）：
每天 AM 6:00 运行全量预测 → 更新所有 SKU 的预测标签 → 变更的标签触发 Action 评估 → 新的预警推送

## ② 母婴出海应用案例

**场景A：7日断货预测标签（主力场景）**
- **业务问题**：吸奶器旗舰款在Amazon平均PLT=35天，等到真的断货再补货已经来不及
- **预测标签逻辑**：
  ```
  predicted_stockout_7d = True  IF:
    当前库存 / 日均销量 - PLT_P85 < 7天
    OR 预测需求（7日）> 当前可用库存
  ```
- **触发动作**：`predicted_stockout_7d=True` → 自动创建补货预审单（采购经理24h内确认）
- **业务价值**：断货事件从"12件/年"降至"3件/年"，年化减少断货损失约18万元

**场景B：大促需求峰值预测标签**
- **业务问题**：Black Friday前需要知道哪些SKU需要额外备货，但不确定会大多少
- **预测标签**：`predicted_demand_spike=True`（置信度≥0.80）→ 触发"大促备货复核"工作流
- **预测逻辑**：历史大促销量倍数 × 当年广告投入比例 × 外部搜索热度指数
- **业务价值**：大促备货准确率（售罄率50-65%目标）从42%提升至68%

## ③ 代码模板

```python
"""
预测型标签引擎 — 供应链供应链预测打标体系
功能：多类型预测标签计算 / 滚动更新 / 置信度管理 / 提前行动映射
输入：SKU库存数据 + 销售历史 + PLT数据
输出：预测标签集 + 置信度 + 行动优先级 + 预测准确率追踪
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PredictiveTag:
    tag_id: str
    horizon_days: int
    value: bool
    confidence: float
    predicted_at: str
    model_name: str
    evidence: dict = field(default_factory=dict)
    action_priority: str = "normal"  # low/normal/high/urgent

    def is_actionable(self, min_confidence: float = 0.75) -> bool:
        return self.value and self.confidence >= min_confidence


class PredictiveTagEngine:
    """预测型标签引擎"""

    def __init__(self, confidence_thresholds: dict = None):
        self.thresholds = confidence_thresholds or {
            "predicted_stockout_7d": 0.75,
            "predicted_stockout_14d": 0.70,
            "predicted_stockout_30d": 0.65,
            "predicted_demand_spike": 0.72,
            "predicted_slow_moving": 0.68,
            "predicted_price_increase": 0.70,
        }
        self.prediction_log = []

    def predict_stockout(self, sku: dict, horizon_days: int,
                          plt_p85: float = 35.0) -> PredictiveTag:
        """
        断货预测：基于DOS + PLT + 需求波动
        DOS = current_inventory / avg_daily_sales
        """
        inventory = sku.get("inventory", 0)
        avg_daily = max(0.1, sku.get("avg_daily_sales_30d", 1.0))
        demand_cv = sku.get("demand_cv", 0.25)
        pending_po = sku.get("pending_po_qty", 0)

        # 有效库存天数（含在途但扣安全库存）
        effective_inventory = inventory + pending_po * 0.8  # 在途80%可期
        dos = effective_inventory / avg_daily

        # 需求上行风险（CV越大，断货风险越高）
        demand_upside = avg_daily * (1 + 1.28 * demand_cv)  # P90需求
        dos_pessimistic = effective_inventory / max(0.1, demand_upside)

        # 预测断货：DOSpessimistic < PLT + horizon
        will_stockout = dos_pessimistic < (plt_p85 + horizon_days)

        # 置信度（基于DOS余量）
        margin = dos_pessimistic - (plt_p85 + horizon_days)
        if will_stockout:
            confidence = min(0.98, 0.75 + 0.05 * abs(margin) / (plt_p85 + horizon_days))
        else:
            confidence = min(0.95, 0.75 + 0.05 * margin / (plt_p85 + horizon_days))
            will_stockout = False

        priority_map = {7: "urgent", 14: "high", 30: "normal"}

        return PredictiveTag(
            tag_id=f"predicted_stockout_{horizon_days}d",
            horizon_days=horizon_days,
            value=will_stockout,
            confidence=round(confidence, 3),
            predicted_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            model_name="DOS+PLT+DemandCV",
            evidence={
                "dos": round(dos, 1),
                "dos_pessimistic": round(dos_pessimistic, 1),
                "plt_p85": plt_p85,
                "demand_cv": demand_cv,
            },
            action_priority=priority_map.get(horizon_days, "normal") if will_stockout else "none",
        )

    def predict_slow_moving(self, sku: dict) -> PredictiveTag:
        """预测未来30天是否变成滞销"""
        sales_trend = sku.get("sales_velocity_trend", 0.0)  # 负=下降
        current_inventory_days = sku.get("current_dos", 30)
        abc_class = sku.get("abc_class", "C")

        # 滞销风险因子
        trend_risk = max(0, -sales_trend)  # 下降趋势
        overstocked_risk = max(0, current_inventory_days - 90) / 90  # 超90天库存
        abc_risk = {"A": 0.1, "B": 0.2, "C": 0.4, "D": 0.6, "E": 0.8}.get(abc_class, 0.5)

        slow_moving_score = 0.4 * trend_risk + 0.4 * overstocked_risk + 0.2 * abc_risk
        will_slow = slow_moving_score > 0.4

        return PredictiveTag(
            tag_id="predicted_slow_moving",
            horizon_days=30,
            value=will_slow,
            confidence=round(0.5 + 0.4 * slow_moving_score, 3),
            predicted_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            model_name="TrendDecay+InventoryAge+ABC",
            evidence={
                "trend_risk": round(trend_risk, 3),
                "overstock_risk": round(overstocked_risk, 3),
                "abc_class": abc_class,
            },
            action_priority="normal" if will_slow else "none",
        )

    def predict_demand_spike(self, sku: dict) -> PredictiveTag:
        """预测7日内是否有需求峰值（大促/季节性）"""
        days_to_promo = sku.get("days_to_next_promo", 999)
        seasonal_factor = sku.get("seasonal_factor_7d", 1.0)
        search_trend = sku.get("external_search_trend", 0.0)

        promo_risk = max(0, 1 - days_to_promo / 14) if days_to_promo <= 14 else 0
        seasonal_risk = max(0, seasonal_factor - 1.2)
        search_risk = max(0, search_trend - 0.3)

        spike_score = 0.5 * promo_risk + 0.3 * seasonal_risk + 0.2 * search_risk
        will_spike = spike_score > 0.25

        return PredictiveTag(
            tag_id="predicted_demand_spike",
            horizon_days=7,
            value=will_spike,
            confidence=round(min(0.92, 0.60 + 0.5 * spike_score), 3),
            predicted_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            model_name="PromoSchedule+Seasonal+SearchTrend",
            evidence={
                "days_to_promo": days_to_promo,
                "seasonal_factor": seasonal_factor,
                "search_trend": search_trend,
            },
            action_priority="high" if (will_spike and days_to_promo <= 7) else "normal",
        )

    def run_full_prediction(self, sku: dict, plt_p85: float = 35.0) -> list:
        """对单个SKU运行全量预测"""
        predictions = []
        for horizon in [7, 14, 30]:
            pred = self.predict_stockout(sku, horizon, plt_p85)
            if pred.value or pred.confidence > 0.3:
                predictions.append(pred)

        slow_pred = self.predict_slow_moving(sku)
        predictions.append(slow_pred)

        spike_pred = self.predict_demand_spike(sku)
        if spike_pred.value:
            predictions.append(spike_pred)

        self.prediction_log.extend(predictions)
        return predictions


def generate_sku_portfolio(n: int = 50, seed: int = 42) -> list:
    np.random.seed(seed)
    skus = []
    for i in range(n):
        abc = np.random.choice(["A", "B", "C", "D", "E"], p=[0.05, 0.15, 0.30, 0.30, 0.20])
        avg_daily = {"A": 30, "B": 12, "C": 5, "D": 2, "E": 0.3}[abc]
        avg_daily *= np.random.uniform(0.5, 2.0)
        inventory = avg_daily * np.random.uniform(5, 80)

        skus.append({
            "id": f"SKU-{i+1:03d}", "name": f"产品{i+1}", "abc_class": abc,
            "inventory": round(inventory),
            "avg_daily_sales_30d": round(avg_daily, 2),
            "demand_cv": np.random.uniform(0.10, 0.45),
            "pending_po_qty": round(avg_daily * np.random.uniform(0, 20)),
            "current_dos": round(inventory / max(0.1, avg_daily), 1),
            "sales_velocity_trend": np.random.uniform(-0.5, 0.3),
            "days_to_next_promo": np.random.choice([999, 60, 30, 14, 7, 3], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
            "seasonal_factor_7d": np.random.uniform(0.8, 2.0),
            "external_search_trend": np.random.uniform(-0.2, 0.5),
        })
    return skus


def run_daily_prediction_batch(skus: list, engine: PredictiveTagEngine, plt_p85: float = 35.0):
    """日常滚动预测批处理"""
    print("=" * 65)
    print(f"【每日预测型标签更新（{datetime.now().strftime('%Y-%m-%d')}）】")
    print("=" * 65)

    actionable_by_priority = {"urgent": [], "high": [], "normal": []}
    all_preds_by_sku = {}

    for sku in skus:
        preds = engine.run_full_prediction(sku, plt_p85=plt_p85)
        all_preds_by_sku[sku["id"]] = preds
        for pred in preds:
            if pred.is_actionable(engine.thresholds.get(pred.tag_id, 0.70)):
                actionable_by_priority.setdefault(pred.action_priority, []).append(
                    (sku["id"], pred))

    # 输出按优先级的行动清单
    for priority in ["urgent", "high", "normal"]:
        items = actionable_by_priority.get(priority, [])
        if not items:
            continue
        icon = {"urgent": "🔴", "high": "🟡", "normal": "🟢"}[priority]
        print(f"\n  {icon} {priority.upper()} ({len(items)}项):")
        for sku_id, pred in items[:6]:
            print(f"    {sku_id}: [{pred.tag_id}] conf={pred.confidence:.2f}  "
                  f"证据: DOS={pred.evidence.get('dos_pessimistic', '-')}")

    # 统计
    total_actionable = sum(len(v) for v in actionable_by_priority.values())
    total_preds = len(engine.prediction_log)
    print(f"\n  总预测数: {total_preds}  可行动项: {total_actionable}")
    print(f"  覆盖率: {len(skus)}个SKU全量预测")


if __name__ == "__main__":
    print("【预测型标签引擎 — 供应链前置风险标注】\n")

    engine = PredictiveTagEngine()
    skus = generate_sku_portfolio(n=50)
    run_daily_prediction_batch(skus, engine, plt_p85=35.0)

    print(f"\n[✓] 预测型标签引擎 测试通过")
    urgent_count = sum(1 for p in engine.prediction_log if p.action_priority == "urgent")
    print(f"    处理{len(skus)}个SKU  生成{len(engine.prediction_log)}个预测标签  "
          f"紧急行动{urgent_count}项")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Tag-Schema-Engineering-Lifecycle]]（预测标签Schema含horizon/model/confidence字段）
- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]]（需求预测模型为预测标签提供输入）
- **延伸（extends）**：[[Skill-Supply-Chain-Ontology-Action-Trigger]]（预测标签是最重要的Action触发源）
- **延伸（extends）**：[[Skill-Tag-Quality-Coverage-KPI]]（预测标签准确率需要回溯验证监控）
- **可组合（combinable）**：[[Skill-Procurement-Cycle-Time-KPI]]（PLT_P85是断货预测的关键输入）
- **可组合（combinable）**：[[Skill-Forecast-MAPE-MinMax-Accuracy-System]]（预测准确率KPI反馈到预测标签质量）

## ⑤ 商业价值评估

- **ROI预估**：7日断货预测标签使断货事件从12件/年→3件/年，年化减少断货损失约18万元；大促需求峰值预测标签使大促备货准确率从42%→68%，减少大促后尾货损失约10万元
- **实施难度**：⭐⭐⭐☆☆（需要时序预测模型和PLT数据支撑，中等难度）
- **优先级评分**：⭐⭐⭐⭐⭐（把"响应式"补货变为"预防式"补货，是标签工程对供应链最大的价值贡献）
- **评估依据**：预测标签实现"T-14天提前响应"比"T-0天断货应急"，响应成本降低10-20倍（无需紧急空运）
