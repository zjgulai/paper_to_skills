---
title: Post-Purchase-Review-Request-Dispatcher — 订单完成后按满意度预测分层分发评论邀请
doc_type: knowledge
module: 14-用户分析
topic: post-purchase-review-request-dispatcher
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Post-Purchase-Review-Request-Dispatcher

> **配对分析层**：[[Skill-CSAT-NPS-Survey-Analysis]]
> **决策类型**: 自动触发型 | **触发条件**: 订单完成后第7天 | **执行动作**: 按预测CSAT分层——高满意→直接邀请评论；低满意→先客服处理再邀请

## ① 算法原理

核心是「交付后状态检测 + CSAT 代理预测 + 差异化分发」：

1. **触发时机**：订单标记「已送达」后 7 天，系统自动扫描并触发分发决策。
2. **CSAT 代理预测**：用配送时效（承诺 vs 实际）、历史评分、退货记录、客服联系次数等特征估算满意度概率：
   - 预测 CSAT ≥ 4.2：高满意，直接发送评论邀请
   - 3.5 ≤ 预测 CSAT < 4.2：中性，先发满意度调查问卷，再根据反馈决定是否邀请评论
   - 预测 CSAT < 3.5：低满意，优先触发客服主动联系解决问题，问题解决后再邀请评论
3. **时间窗口保护**：Amazon Review Policy 要求不可在交付 90 天后邀请，需在 7-30 天窗口内完成。
4. **防刷保护**：同一订单只触发一次评论邀请；若用户已主动评论则跳过。

## ② 母婴出海应用案例

**场景：婴儿吸奶器购后评论管理（目标评分 ≥ 4.5★）**
- 触发条件：订单 ORD-20260615 配送 7 天，预测 CSAT 3.2（配送延误 3 天 + 包装破损投诉记录）
- 执行动作：
  - T+0h：客服主动联系「我们关注到您的配送体验，能否告知问题？提供补偿方案」
  - T+48h：客服确认问题解决后，系统自动触发评论邀请（附 $5 积分）
- 对照组（预测 CSAT 4.6）：订单完成 7 天后直接发评论邀请邮件
- 业务价值：将差评率从 18% 降至 6%，平均评分从 4.1→4.6★，年化转化率提升 ~3 个百分点（评分 4.5→4.8 对应 CTR 增加 12%）

## ③ 代码模板

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def post_purchase_review_request_dispatcher(
    orders: List[Dict],
    now: Optional[datetime] = None,
    trigger_days: int = 7,
    high_csat_threshold: float = 4.2,
    low_csat_threshold: float = 3.5,
    review_window_days: int = 90
) -> Dict:
    """
    购后评论邀请分层分发器
    
    参数:
        orders: [{
            "order_id": str, "user_id": str,
            "delivered_at": str (ISO8601),
            "predicted_csat": float (1-5),
            "has_reviewed": bool,
            "cs_contacted": bool,
            "cs_resolved": bool
        }]
        trigger_days: 交付后多少天触发（默认7天）
        high_csat_threshold: 高满意度阈值（直接邀请评论）
        low_csat_threshold: 低满意度阈值（先处理问题）
        review_window_days: 评论邀请有效窗口（Amazon政策90天）
    
    返回:
        {"dispatches": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    dispatches = []
    
    for order in orders:
        oid = order["order_id"]
        uid = order["user_id"]
        delivered_at = datetime.fromisoformat(order["delivered_at"])
        predicted_csat = order.get("predicted_csat", 4.0)
        has_reviewed = order.get("has_reviewed", False)
        cs_contacted = order.get("cs_contacted", False)
        cs_resolved = order.get("cs_resolved", False)
        
        days_since_delivery = (now - delivered_at).days
        
        # 已评论：跳过
        if has_reviewed:
            dispatches.append({"order_id": oid, "action": "SKIP", "reason": "用户已自行评论"})
            continue
        
        # 未到触发时间
        if days_since_delivery < trigger_days:
            dispatches.append({"order_id": oid, "action": "WAIT",
                                "reason": f"仅{days_since_delivery}天，等待{trigger_days}天触发"})
            continue
        
        # 超出评论窗口
        if days_since_delivery > review_window_days:
            dispatches.append({"order_id": oid, "action": "EXPIRED",
                                "reason": f"超出{review_window_days}天评论窗口"})
            continue
        
        # 分层分发逻辑
        if predicted_csat >= high_csat_threshold:
            # 高满意：直接邀请评论
            action = {
                "order_id": oid,
                "user_id": uid,
                "action": "REVIEW_REQUEST_DIRECT",
                "predicted_csat": predicted_csat,
                "channel": "email",
                "incentive": "$5积分",
                "message": f"感谢您的购买！您的评价对其他妈妈很重要，分享您的体验获得$5积分",
                "send_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "days_since_delivery": days_since_delivery
            }
        elif predicted_csat >= low_csat_threshold:
            # 中性：先发调查问卷
            action = {
                "order_id": oid,
                "user_id": uid,
                "action": "SURVEY_FIRST",
                "predicted_csat": predicted_csat,
                "channel": "email",
                "message": "您对本次购物体验满意吗？（1-5分）",
                "follow_up": "根据问卷反馈决定是否推进评论邀请",
                "send_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "days_since_delivery": days_since_delivery
            }
        else:
            # 低满意：先客服介入
            if not cs_contacted:
                action = {
                    "order_id": oid,
                    "user_id": uid,
                    "action": "CS_INTERVENTION_FIRST",
                    "predicted_csat": predicted_csat,
                    "channel": "cs_proactive",
                    "sla_hours": 24,
                    "message": "主动联系客户了解问题，提供补偿方案",
                    "review_request_after": "客服确认问题解决后48h内发评论邀请",
                    "send_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                    "days_since_delivery": days_since_delivery
                }
            elif cs_resolved:
                # 客服已解决，可以邀请评论
                action = {
                    "order_id": oid,
                    "user_id": uid,
                    "action": "REVIEW_REQUEST_POST_CS",
                    "predicted_csat": predicted_csat,
                    "channel": "email",
                    "incentive": "$8积分（感谢耐心）",
                    "message": "感谢您的耐心，问题已解决。您的反馈对我们改进非常重要",
                    "send_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                    "days_since_delivery": days_since_delivery
                }
            else:
                # 客服介入但未解决
                action = {
                    "order_id": oid,
                    "user_id": uid,
                    "action": "HOLD_CS_PENDING",
                    "predicted_csat": predicted_csat,
                    "message": "等待客服完成问题处理后再触发评论邀请",
                    "days_since_delivery": days_since_delivery
                }
        
        dispatches.append(action)
    
    # 统计
    action_counts = {}
    for d in dispatches:
        a = d.get("action", "UNKNOWN")
        action_counts[a] = action_counts.get(a, 0) + 1
    
    return {
        "total_orders": len(orders),
        "dispatches": dispatches,
        "stats": action_counts,
        "direct_review_rate": action_counts.get("REVIEW_REQUEST_DIRECT", 0) / max(len(orders), 1)
    }


# 测试
orders = [
    {"order_id": "ORD001", "user_id": "U001", "delivered_at": "2026-06-15T10:00:00",
     "predicted_csat": 4.6, "has_reviewed": False, "cs_contacted": False, "cs_resolved": False},
    {"order_id": "ORD002", "user_id": "U002", "delivered_at": "2026-06-14T10:00:00",
     "predicted_csat": 3.2, "has_reviewed": False, "cs_contacted": False, "cs_resolved": False},
    {"order_id": "ORD003", "user_id": "U003", "delivered_at": "2026-06-14T10:00:00",
     "predicted_csat": 3.2, "has_reviewed": False, "cs_contacted": True, "cs_resolved": True},
    {"order_id": "ORD004", "user_id": "U004", "delivered_at": "2026-06-15T10:00:00",
     "predicted_csat": 3.8, "has_reviewed": False, "cs_contacted": False, "cs_resolved": False},
    {"order_id": "ORD005", "user_id": "U005", "delivered_at": "2026-06-15T10:00:00",
     "predicted_csat": 4.8, "has_reviewed": True, "cs_contacted": False, "cs_resolved": False},  # 已评论
]

now = datetime(2026, 6, 22, 12, 0, 0)
result = post_purchase_review_request_dispatcher(orders, now=now)

assert result["total_orders"] == 5
dispatches = {d["order_id"]: d["action"] for d in result["dispatches"]}
assert dispatches["ORD001"] == "REVIEW_REQUEST_DIRECT"     # 高满意
assert dispatches["ORD002"] == "CS_INTERVENTION_FIRST"     # 低满意，未联系客服
assert dispatches["ORD003"] == "REVIEW_REQUEST_POST_CS"    # 低满意，客服已解决
assert dispatches["ORD004"] == "SURVEY_FIRST"              # 中性
assert dispatches["ORD005"] == "SKIP"                      # 已评论

print("[✓] Post Purchase Review Request Dispatcher 测试通过")
print(f"  总订单: {result['total_orders']}，直接邀请率: {result['direct_review_rate']:.0%}")
print(f"  分发详情: {result['stats']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-CSAT-NPS-Survey-Analysis]]（构建 CSAT 预测特征工程）
- **延伸（extends）**：[[Skill-High-Value-Customer-Alert-Action]]（对高价值用户单独策略）
- **可组合（combinable）**：[[Skill-VOC-Sentiment-Dispatcher]]（评论内容质量分析反哺预测模型）

## ⑤ 商业价值评估
- **ROI量化**：差评率从 18% → 6%，平均评分 4.1 → 4.6★，CTR 提升约 12%，年化 GMV 增量 $50,000+
- **实施难度**：⭐⭐☆☆☆（需配送 API + 客服 CRM + 邮件平台对接）
- **优先级**：⭐⭐⭐⭐⭐（评分直接影响搜索排名和转化率，核心竞争力）
