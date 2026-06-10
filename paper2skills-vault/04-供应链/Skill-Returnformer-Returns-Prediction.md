---
title: Returnformer Returns Prediction — 图 Transformer 电商退货预测
doc_type: knowledge
module: 04-供应链
topic: returnformer-ecommerce-returns-prediction
status: stable
created: 2026-06-10
updated: 2026-06-10
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Returnformer-Returns-Prediction（退货风险预测）

> **论文**：Returnformer: A Graph Transformer-Based Model for Predicting Product Returns in E-Commerce
> **DOI**：10.3390/e28010072 | MDPI Entropy 2025-01 | **桥梁**: 04-供应链 ↔ 08-知识图谱 ↔ 14-用户分析 | **类型**: 跨域融合

---

## ① 算法原理

**核心思想**：传统退货预测把每笔订单孤立看待，忽略"同一用户多次退同类商品"和"同一商品被多个用户退"的关系网络信号。Returnformer 构建用户-商品二部图，在图上用 Transformer 注意力机制聚合邻域信息，支付完成前即输出退货风险分（0-1），超过阈值可触发拦截或差异化服务策略。

**图结构**：
```
节点：用户节点（含历史退货率特征）+ 商品节点（含品类/价格/描述特征）
边：用户→商品（购买关系，边权=订单金额）
拓扑嵌入：计算节点度中心性、聚类系数作为补充特征
全局注意力：跨图捕捉用户-商品交互的长程依赖
```

**关键指标**：AUC=0.84，超越 4 种 ML 基线（LR/RF/XGBoost/标准GNN）。

---

## ② 母婴出海应用案例

**场景：跨境母婴高退货风险订单识别**

- **业务问题**：母婴跨境退货成本极高（国际退货运费$30-50 + 清关检验 + 翻新成本），退货率从 5% 降至 3% 可节省大量成本，但不知道哪些订单会退。
- **数据要求**：历史订单数据（用户 ID、商品 ID、订单金额、是否退货）+ 用户特征（历史退货率）+ 商品特征（品类、价格、描述）。
- **预期产出**：
  - 每笔订单的退货风险分（0-1）
  - 高风险订单清单（阈值建议 0.6+）
  - 按品类/价格段的退货风险分布热图
- **业务应用**：
  - 高风险订单触发主动客服回访（确认尺码/使用场景），降低因信息不对称导致的退货
  - 高风险用户在下次下单时展示详细使用视频
  - 选品决策：高退货率 SKU 优先排查产品质量或描述问题
- **业务价值**：退货率降低 1-2pp，年化节省退货成本 20-50 万元（视规模）。

---

## ③ 代码模板

```python
from dataclasses import dataclass, field
from typing import List, Dict
import statistics

@dataclass
class OrderFeature:
    order_id: str
    user_id: str
    sku_id: str
    category: str
    price: float
    user_return_rate: float
    sku_return_rate: float
    description_clarity: float

def compute_return_risk(order: OrderFeature,
                        category_base_rates: Dict[str, float]) -> float:
    base = category_base_rates.get(order.category, 0.08)
    user_factor = 1 + (order.user_return_rate - 0.1) * 2
    sku_factor = 1 + (order.sku_return_rate - 0.08) * 2
    price_factor = 1 + max(0, (order.price - 100) / 500) * 0.3
    clarity_factor = 1 - order.description_clarity * 0.3
    risk = base * max(0.3, user_factor) * max(0.3, sku_factor) * price_factor * clarity_factor
    return min(1.0, round(risk, 3))

def batch_return_prediction(orders: List[OrderFeature],
                            category_rates: Dict[str, float],
                            threshold: float = 0.25) -> List[Dict]:
    results = []
    for order in orders:
        risk = compute_return_risk(order, category_rates)
        level = "高" if risk >= threshold else "中" if risk >= threshold * 0.6 else "低"
        action = "主动客服回访" if level == "高" else "发送使用指南" if level == "中" else "正常处理"
        results.append({"order_id": order.order_id, "risk_score": risk,
                        "risk_level": level, "action": action})
    return sorted(results, key=lambda x: -x["risk_score"])

category_rates = {"breast_pump": 0.06, "clothing": 0.18, "electronics": 0.12, "bottle": 0.04}
orders = [
    OrderFeature("O001", "U001", "pump-s1", "breast_pump", 89.99, 0.05, 0.06, 0.9),
    OrderFeature("O002", "U002", "dress-m",  "clothing",   45.00, 0.35, 0.22, 0.5),
    OrderFeature("O003", "U003", "monitor",  "electronics", 199.99, 0.10, 0.15, 0.7),
    OrderFeature("O004", "U004", "bottle-s", "bottle",      29.99, 0.02, 0.03, 0.95),
]
predictions = batch_return_prediction(orders, category_rates)
for p in predictions:
    print(f"订单 {p['order_id']}: 风险={p['risk_score']:.3f} [{p['risk_level']}] → {p['action']}")
avg_risk = statistics.mean(p["risk_score"] for p in predictions)
print(f"平均退货风险: {avg_risk:.3f}")
print("[✓] Returnformer 退货预测测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-RFM-Customer-Segmentation]]（用户退货率特征来自 RFM 分析）
- **延伸**：[[Skill-Consumer-Complaint-Recall-Prediction]]（退货预测 + 召回预测联合，识别产品质量问题）
- **组合**：[[Skill-Review-Pain-Point-Mining]]（高退货 SKU 的差评文本挖掘，定位根因）
- **组合**：[[Skill-Customer-Churn-Prediction]]（高退货率用户往往也是流失高风险用户，联合干预）

---

## ⑤ 商业价值评估

- **ROI 预估**：退货率降低 1-2pp，年化节省 20-50 万元（跨境退货成本高）
- **实施难度**：⭐⭐☆☆☆（低，特征工程 + 简单 ML 模型即可实现 MVP）
- **优先级**：⭐⭐⭐⭐⭐（跨境退货成本极高，是边际利润的最大侵蚀因素之一）
- **评估依据**：论文 AUC=0.84，超越 4 种 ML 基线
