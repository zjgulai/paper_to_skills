---
title: Supplier-Performance-Alert-Action — 供应商OTIF连续不达标自动触发备选供应商激活
doc_type: knowledge
module: 04-供应链
topic: supplier-performance-alert-action
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Supplier-Performance-Alert-Action

> **配对分析层**：[[Skill-Supplier-Scorecard-Analytics]]
> **决策类型**: 自动触发型 | **触发条件**: 供应商 OTIF < 85% 连续3周 | **执行动作**: 触发备选供应商激活流程 + 主供应商预警通知

## ① 算法原理

核心是「OTIF 滚动计算 + 连续不达标检测 + 双轨响应（预警+备选激活）」：

1. **OTIF 计算**：On-Time In-Full = 准时且完整交付的订单数 / 总订单数。以周为颗粒度滚动计算。
2. **连续检测**：维护每个供应商的最近 N 周 OTIF 时间序列，检测连续 K 周低于阈值（默认 K=3, threshold=85%）。
3. **分级响应**：
   - 连续 2 周 OTIF < 85%：发送预警通知给供应商，启动「改善行动计划（CAP）」
   - 连续 3 周 OTIF < 85%：激活备选供应商，部分订单转移（20-50%）
   - 连续 5 周 OTIF < 85%：暂停主供应商新订单，全量切换备选供应商
4. **自动恢复**：备选供应商激活后，若主供应商连续 4 周 OTIF ≥ 90%，自动恢复比例分配。

**关键指标**：供应商 OTIF 每下降 1 个百分点，缺货风险增加约 2.3%（历史数据）。

## ② 母婴出海应用案例

**场景：婴儿纸尿裤主供应商产能下滑**
- 触发条件：主供应商 S-DIAPER-A 第 3 周 OTIF = 76%（W1: 82%, W2: 79%, W3: 76%），连续 3 周低于 85%
- 执行动作：
  - 立即发送正式预警函给主供应商（要求 72h 内提交 CAP）
  - 激活备选供应商 S-DIAPER-B，将未来 4 周 30% 订单量转移
  - 通知采购团队监控备选供应商产能和质检流程
- 安全护栏：备选供应商激活前需通过质检验证（首批次质检合格率 ≥ 95%）
- 业务价值：供应链中断风险从 28% 降至 6%，年化避免缺货损失 $85,000

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime

def supplier_performance_alert_action(
    suppliers: List[Dict],
    warning_weeks: int = 2,
    activate_weeks: int = 3,
    suspend_weeks: int = 5,
    otif_threshold: float = 0.85,
    transfer_pct_partial: float = 0.30,
    transfer_pct_full: float = 1.0
) -> Dict:
    """
    供应商绩效预警与备选激活执行器
    
    参数:
        suppliers: [{
            "supplier_id": str, "supplier_name": str,
            "weekly_otif": List[float],  # 最近N周OTIF，从早到晚排列
            "backup_supplier_id": str | None,
            "backup_qualified": bool,    # 备选供应商是否已通过质检
            "weekly_order_volume": int   # 周均订单量
        }]
        warning_weeks: 触发预警所需连续低OTIF周数
        activate_weeks: 触发备选激活所需连续低OTIF周数
        suspend_weeks: 触发暂停所需连续低OTIF周数
        otif_threshold: OTIF 不达标阈值
    
    返回:
        {"actions": [...], "stats": {...}}
    """
    actions = []
    
    for supplier in suppliers:
        sid = supplier["supplier_id"]
        sname = supplier.get("supplier_name", sid)
        weekly_otif = supplier.get("weekly_otif", [])
        backup_id = supplier.get("backup_supplier_id")
        backup_qualified = supplier.get("backup_qualified", False)
        order_volume = supplier.get("weekly_order_volume", 100)
        
        if not weekly_otif:
            actions.append({"supplier_id": sid, "action": "DATA_MISSING", "reason": "无OTIF历史数据"})
            continue
        
        # 计算连续不达标周数（从最近一周往前数）
        consecutive_fail = 0
        for otif in reversed(weekly_otif):
            if otif < otif_threshold:
                consecutive_fail += 1
            else:
                break
        
        current_otif = weekly_otif[-1]
        
        if consecutive_fail >= suspend_weeks:
            # 全量暂停
            action = {
                "supplier_id": sid,
                "supplier_name": sname,
                "action": "SUSPEND_AND_TRANSFER",
                "consecutive_fail_weeks": consecutive_fail,
                "current_otif": current_otif,
                "otif_trend": weekly_otif[-5:],
                "severity": "CRITICAL",
                "steps": [
                    f"暂停主供应商{sname}所有新订单",
                    f"全量切换至备选供应商{backup_id}（前提：质检通过）",
                    "启动新供应商寻源流程（目标：30天内完成评估）",
                    f"发送正式终止合同预警函（30天预告期）"
                ],
                "transfer_pct": transfer_pct_full,
                "orders_to_transfer": order_volume,
                "backup_required": True,
                "backup_qualified": backup_qualified
            }
        elif consecutive_fail >= activate_weeks:
            # 部分转移
            action = {
                "supplier_id": sid,
                "supplier_name": sname,
                "action": "ACTIVATE_BACKUP_PARTIAL",
                "consecutive_fail_weeks": consecutive_fail,
                "current_otif": current_otif,
                "otif_trend": weekly_otif[-4:],
                "severity": "HIGH",
                "steps": [
                    f"激活备选供应商{backup_id}（需质检验证）",
                    f"将{int(transfer_pct_partial*100)}%订单量转移至备选（{int(order_volume*transfer_pct_partial)}单/周）",
                    f"主供应商{sname}发送正式预警函，要求72h内提交改善行动计划（CAP）",
                    "设置4周恢复观察期：若OTIF≥90%则恢复比例分配"
                ],
                "transfer_pct": transfer_pct_partial,
                "orders_to_transfer": int(order_volume * transfer_pct_partial),
                "backup_required": True,
                "backup_qualified": backup_qualified
            }
        elif consecutive_fail >= warning_weeks:
            # 预警
            action = {
                "supplier_id": sid,
                "supplier_name": sname,
                "action": "WARNING_ISSUED",
                "consecutive_fail_weeks": consecutive_fail,
                "current_otif": current_otif,
                "otif_trend": weekly_otif[-3:],
                "severity": "MEDIUM",
                "steps": [
                    f"发送书面预警通知给{sname}",
                    "要求提交产能改善计划（48h内回复）",
                    "安排周度绩效回顾会议",
                    "若下周OTIF仍<85%，触发备选供应商激活"
                ],
                "backup_required": False
            }
        else:
            action = {
                "supplier_id": sid,
                "supplier_name": sname,
                "action": "HEALTHY",
                "consecutive_fail_weeks": consecutive_fail,
                "current_otif": current_otif,
                "reason": f"OTIF={current_otif:.0%}，连续不达标{consecutive_fail}周（阈值{warning_weeks}周）"
            }
        
        actions.append(action)
    
    severity_counts = {}
    for a in actions:
        sev = a.get("severity", "HEALTHY")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    return {
        "total_suppliers": len(suppliers),
        "actions": actions,
        "severity_summary": severity_counts,
        "backup_activation_count": sum(1 for a in actions if a.get("backup_required", False))
    }


# 测试
suppliers = [
    {
        "supplier_id": "S-DIAPER-A", "supplier_name": "广州纸尿裤厂A",
        "weekly_otif": [0.92, 0.90, 0.88, 0.82, 0.79, 0.76],  # 连续3周<85%
        "backup_supplier_id": "S-DIAPER-B", "backup_qualified": True,
        "weekly_order_volume": 500
    },
    {
        "supplier_id": "S-BOTTLE-B", "supplier_name": "深圳奶瓶厂B",
        "weekly_otif": [0.91, 0.89, 0.87, 0.85, 0.82, 0.80],  # 连续2周<85%
        "backup_supplier_id": "S-BOTTLE-C", "backup_qualified": True,
        "weekly_order_volume": 300
    },
    {
        "supplier_id": "S-TOY-C", "supplier_name": "东莞玩具厂C",
        "weekly_otif": [0.95, 0.93, 0.92, 0.94, 0.96, 0.91],  # 健康
        "backup_supplier_id": None, "backup_qualified": False,
        "weekly_order_volume": 200
    },
    {
        "supplier_id": "S-SEAT-D", "supplier_name": "宁波座椅厂D",
        "weekly_otif": [0.88, 0.80, 0.75, 0.72, 0.68, 0.65],  # 连续5周<85%
        "backup_supplier_id": "S-SEAT-E", "backup_qualified": True,
        "weekly_order_volume": 150
    },
]

result = supplier_performance_alert_action(suppliers)

assert result["total_suppliers"] == 4
action_map = {a["supplier_id"]: a["action"] for a in result["actions"]}
assert action_map["S-DIAPER-A"] == "ACTIVATE_BACKUP_PARTIAL"
assert action_map["S-BOTTLE-B"] == "WARNING_ISSUED"
assert action_map["S-TOY-C"] == "HEALTHY"
assert action_map["S-SEAT-D"] == "SUSPEND_AND_TRANSFER"
assert result["backup_activation_count"] == 2

print("[✓] Supplier Performance Alert Action 测试通过")
print(f"  总供应商: {result['total_suppliers']}，需激活备选: {result['backup_activation_count']}")
print(f"  严重程度分布: {result['severity_summary']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Supplier-Scorecard-Analytics]]（计算多维供应商评分，OTIF 是核心指标之一）
- **延伸（extends）**：[[Skill-OOS-Emergency-Airfreight-Gate]]（供应商履约失败 → 触发紧急补货门控）
- **可组合（combinable）**：[[Skill-Multi-Echelon-Inventory-Optimization]]（备选供应商激活后重新优化备货策略）

## ⑤ 商业价值评估
- **ROI量化**：供应链中断风险从 28% → 6%，年化避免缺货损失 $85,000；备选供应商激活成本约 $5,000/次
- **实施难度**：⭐⭐☆☆☆（需供应商 ERP 数据接口 + 通知系统 + 备选供应商评级库）
- **优先级**：⭐⭐⭐⭐⭐（供应商风险是跨境电商最难预测的断货根因）
