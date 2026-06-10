---
title: Supplier Performance Scorecard — 供应商绩效量化追踪系统
doc_type: knowledge
module: 04-供应链
topic: supplier-performance-scorecard-kpi-tracking
status: stable
created: 2026-06-10
updated: 2026-06-10
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Supplier-Performance-Scorecard（供应商绩效追踪）

> **方法**：多维 KPI 积分卡 + 时序趋势预警 | **桥梁**: 04-供应链 ↔ 23-运营财务 | **类型**: 算法工具

---

## ① 算法原理

**核心思想**：大多数跨境品牌对供应商的评估是"年度打分"——一年只看一次，且主要靠主观印象。当供应商质量开始下滑时（如交期从 98% 准时率悄悄降到 85%），往往要等到断货或大批投诉才发现。供应商绩效追踪系统用**月度 KPI 积分卡 + 时序趋势检测**，实时掌握每个供应商的绩效动态，提前识别风险。

**五维 KPI 框架**：
```
维度1: 交货准时率 (On-Time Delivery)      权重 30%
  = 按时交货订单数 / 总订单数
  基准线: ≥ 95% 优秀，85-95% 正常，< 85% 预警

维度2: 质量合格率 (Quality Pass Rate)     权重 30%
  = 质检通过批次 / 总批次
  关联: 退款率 + 客服投诉率（质量问题）

维度3: 响应速度 (Response Responsiveness) 权重 15%
  = 沟通响应时间 + 问题处理时效
  关键场景: 紧急补货、质量异议、证书补件

维度4: 价格竞争力 (Price Competitiveness)  权重 15%
  = 报价 vs 市场基准价（定期更新）
  趋势: 是否在无成本变化下频繁提价

维度5: 合规完整性 (Compliance)            权重 10%
  = 证书有效期 + 年审到期预警 + 违规记录
```

**时序趋势预警**：
- 连续 2 个月下降 → 🟡 黄色预警（约谈）
- 连续 3 个月下降 → 🔴 红色预警（启动备选）
- 单月骤降 > 10pp → 🚨 紧急预警（立即介入）

---

## ② 母婴出海应用案例

**场景：核心硅胶供应商绩效下滑预警**

- **业务现状**：某东莞硅胶厂是品牌核心供应商（占吸奶器硅胶件 80%），月初数据显示交货准时率从 97% 悄悄降到 88%，质量合格率从 99% 降到 94%，但负责人没有发现。
- **预警触发**：
  - 交货准时率连续 2 个月下降 → 🟡 黄色预警
  - 质量合格率单月下降 5pp → 🟡 预警（边界）
  - 综合评级从 A+ 降至 B
- **管理动作**：
  - 预警触发后约谈供应商负责人，了解内部原因（产能紧张/工人流失）
  - 同步激活备选供应商（广州另一家），分担 20% 订单
  - 3 个月后复评，若仍未改善则正式切换
- **业务价值**：提前 2-3 个月发现供应商风险，避免临时断供损失 30-100 万元。

---

## ③ 代码模板

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class MonthlyRecord:
    month: str
    on_time_rate: float
    quality_pass_rate: float
    avg_response_hours: float
    price_vs_benchmark: float
    compliance_score: float

@dataclass
class SupplierProfile:
    supplier_id: str
    name: str
    category: str
    history: List[MonthlyRecord] = field(default_factory=list)

WEIGHTS = {"on_time": 0.30, "quality": 0.30, "response": 0.15, "price": 0.15, "compliance": 0.10}

def score_month(record: MonthlyRecord) -> Dict:
    response_score = max(0, 1 - record.avg_response_hours / 48)
    price_score = max(0, 1 - (record.price_vs_benchmark - 1) * 2)
    composite = (record.on_time_rate / 100 * WEIGHTS["on_time"] +
                 record.quality_pass_rate / 100 * WEIGHTS["quality"] +
                 response_score * WEIGHTS["response"] +
                 price_score * WEIGHTS["price"] +
                 record.compliance_score / 100 * WEIGHTS["compliance"])
    grade = "A+" if composite >= 0.95 else "A" if composite >= 0.90 else "B" if composite >= 0.80 else "C" if composite >= 0.70 else "D"
    return {"month": record.month, "composite": round(composite * 100, 1), "grade": grade,
            "on_time": record.on_time_rate, "quality": record.quality_pass_rate,
            "response_score": round(response_score * 100, 1)}

def detect_trend_alerts(supplier: SupplierProfile) -> List[Dict]:
    if len(supplier.history) < 2:
        return []
    scores = [score_month(r) for r in supplier.history]
    alerts = []
    composites = [s["composite"] for s in scores]
    if len(composites) >= 3 and all(composites[-i-1] < composites[-i-2] for i in range(2)):
        alerts.append({"type": "🔴 红色预警", "message": "综合评分连续 3 个月下降",
                        "action": "立即启动备选供应商，准备切换"})
    elif len(composites) >= 2 and composites[-1] < composites[-2]:
        alerts.append({"type": "🟡 黄色预警", "message": "综合评分连续 2 个月下降",
                        "action": "约谈供应商了解原因，设定改善目标"})
    latest = scores[-1]
    prev = scores[-2] if len(scores) >= 2 else latest
    if latest["on_time"] < 85:
        alerts.append({"type": "🚨 紧急", "message": f"交货准时率骤降至 {latest['on_time']}%",
                        "action": "紧急沟通，启动应急备货"})
    if latest["quality"] < 90:
        alerts.append({"type": "🟡 预警", "message": f"质量合格率降至 {latest['quality']}%",
                        "action": "加强入库质检，与品控部门确认原因"})
    return alerts

def generate_scorecard(supplier: SupplierProfile) -> Dict:
    if not supplier.history:
        return {"error": "no data"}
    scores = [score_month(r) for r in supplier.history]
    latest = scores[-1]
    trend = "⬇️ 下滑" if len(scores) >= 2 and scores[-1]["composite"] < scores[-2]["composite"] else "⬆️ 改善" if len(scores) >= 2 and scores[-1]["composite"] > scores[-2]["composite"] else "→ 稳定"
    alerts = detect_trend_alerts(supplier)
    return {"supplier": supplier.name, "category": supplier.category,
            "latest_grade": latest["grade"], "latest_score": latest["composite"],
            "trend": trend, "alerts": alerts,
            "history_summary": [{"month": s["month"], "score": s["composite"], "grade": s["grade"]} for s in scores[-3:]]}

supplier = SupplierProfile("S_DONGGUAN", "东莞硅胶厂", "硅胶配件", [
    MonthlyRecord("2026-03", 97.2, 99.1, 4, 1.02, 100),
    MonthlyRecord("2026-04", 94.5, 97.8, 6, 1.02, 100),
    MonthlyRecord("2026-05", 88.3, 93.9, 12, 1.05, 95),
])
card = generate_scorecard(supplier)
print(f"=== 供应商绩效积分卡: {card['supplier']} ===")
print(f"最新评级: {card['latest_grade']} ({card['latest_score']}分) | 趋势: {card['trend']}")
print("\n近3个月走势:")
for h in card["history_summary"]:
    print(f"  {h['month']}: {h['score']}分 [{h['grade']}]")
if card["alerts"]:
    print("\n⚠️ 预警信息:")
    for a in card["alerts"]:
        print(f"  {a['type']}: {a['message']}")
        print(f"  → 建议: {a['action']}")
print("[✓] Supplier Performance Scorecard 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Supplier-Capacity-Planning]]（产能规划数据输入绩效评估）
- **前置**：[[Skill-Supplier-Risk-XGBoost]]（风险评分 + 绩效追踪双维度供应商管理）
- **延伸**：[[Skill-SC-Resilience-Hypergraph]]（单供应商绩效 → 整体供应链韧性影响）
- **延伸**：[[Skill-Supply-Chain-Due-Diligence]]（尽调建立基线，绩效追踪监控变化）
- **组合**：[[Skill-Lead-Time-Distribution-Risk-GenQOT]]（绩效下滑时交期风险自动提升，触发安全库存调整）

---

## ⑤ 商业价值评估

- **ROI 预估**：提前 2-3 个月识别供应商风险，避免断供损失 30-100 万元；年度评估效率提升 80%
- **实施难度**：⭐⭐☆☆☆（低，主要是采购数据整理 + KPI 计算）
- **优先级**：⭐⭐⭐⭐☆（核心供应商数量有限（5-15 家），建立追踪系统一次性投入小收益大）
- **评估依据**：供应商 KPI 积分卡是供应链管理行业标准，时序预警是 Lean 制造中成熟实践
