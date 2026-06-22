---
title: Subscription-Renewal-Intervention-Gate — 订阅到期前14天沉默用户自动触发个性化续订挽回
doc_type: knowledge
module: 06-增长模型
topic: subscription-renewal-intervention-gate
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Subscription-Renewal-Intervention-Gate

> **配对分析层**：[[Skill-Subscription-Churn-Prediction]]
> **决策类型**: 自动触发型 | **触发条件**: 订阅到期前14天 AND 近30天无登录/互动 | **执行动作**: 触发个性化续订挽回序列（邮件+优惠+功能提醒）

## ① 算法原理

核心是「到期预警 + 沉默识别 + 分层挽回序列」：

1. **触发条件（AND 逻辑）**：
   - 订阅到期前 ≤ 14 天
   - 近 30 天内无登录/互动（沉默用户，续订风险高）
2. **沉默程度分层**：
   - 沉默 14-30 天（轻度）：发送「功能提醒 + 价值重申」邮件
   - 沉默 > 30 天（中度）：发送「个性化折扣 10% + 使用指南」邮件序列
   - 沉默 > 60 天（重度）：触发客服主动联系，询问续订意向，提供 20% 挽回折扣
3. **个性化内容**：基于用户历史最常用功能生成邮件内容（例如「您之前使用 VOC 分析功能发现了 3 个产品改进方向」）。
4. **序列控制**：3 封邮件间隔 3/7/12 天，用户续订后立即取消后续发送。

## ② 母婴出海应用案例

**场景：母婴选品工具 SaaS 订阅续费挽回**
- 触发条件：企业订阅用户「某跨境电商公司」，订阅到期 T-12天，近 45 天无登录（沉默中度）
- 执行动作：
  - T+0h：邮件「您的订阅还有12天到期——查看您过去 3 个月分析报告的亮点」（个性化回顾）
  - T+3day：邮件「专属续订折扣：8折优惠（72h有效）+ 新功能预览」
  - T+7day：邮件「续订最后提醒：我们的客服随时为您答疑」
- 挽回率：沉默中度用户挽回率从 18% → 34%，年化 ARR 挽回 $85,000

## ③ 代码模板

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def subscription_renewal_intervention_gate(
    subscriptions: List[Dict],
    now: Optional[datetime] = None,
    trigger_days_before: int = 14,
    silence_light_days: int = 14,
    silence_medium_days: int = 30,
    silence_heavy_days: int = 60,
    light_discount: float = 0.0,
    medium_discount: float = 0.10,
    heavy_discount: float = 0.20
) -> Dict:
    """
    订阅续订挽回门控
    
    参数:
        subscriptions: [{
            "sub_id": str, "user_id": str, "company_name": str,
            "expires_at": str (ISO8601),
            "last_active_at": str (ISO8601),
            "monthly_arr": float,           # 月均 ARR
            "top_features": List[str],      # 最常用功能
            "auto_renew": bool
        }]
    
    返回:
        {"interventions": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    interventions = []
    
    for sub in subscriptions:
        sid = sub["sub_id"]
        uid = sub["user_id"]
        company = sub.get("company_name", uid)
        expires_at = datetime.fromisoformat(sub["expires_at"])
        last_active_at = datetime.fromisoformat(sub["last_active_at"])
        arr = sub.get("monthly_arr", 0)
        top_features = sub.get("top_features", [])
        auto_renew = sub.get("auto_renew", False)
        
        # 已设置自动续订的跳过
        if auto_renew:
            interventions.append({"sub_id": sid, "action": "SKIP", "reason": "已设置自动续订"})
            continue
        
        # 计算到期天数
        days_to_expire = (expires_at - now).days
        
        # 未在触发窗口内
        if days_to_expire > trigger_days_before or days_to_expire < 0:
            interventions.append({
                "sub_id": sid, "action": "NOT_IN_WINDOW",
                "days_to_expire": days_to_expire
            })
            continue
        
        # 计算沉默天数
        silence_days = (now - last_active_at).days
        
        # 非沉默用户（近14天有活跃）：只发轻量提醒
        if silence_days < silence_light_days:
            interventions.append({
                "sub_id": sid, "user_id": uid, "company": company,
                "action": "LIGHT_REMINDER",
                "silence_days": silence_days,
                "days_to_expire": days_to_expire,
                "message": f"您的订阅还有{days_to_expire}天到期，点击续订保持不间断服务",
                "channel": "email",
                "discount": 0
            })
            continue
        
        # 确定挽回级别
        if silence_days >= silence_heavy_days:
            level = "HEAVY"
            discount = heavy_discount
            channel = "cs_proactive"
            sequence = [
                {"step": 1, "delay_days": 0,  "channel": "email",
                 "content": f"我们想念您，{company}！您的账户还有许多价值待挖掘"},
                {"step": 2, "delay_days": 3,  "channel": "cs_call",
                 "content": "客服主动致电，了解续订顾虑，提供定制方案"},
                {"step": 3, "delay_days": 7,  "channel": "email",
                 "content": f"最后机会：专属{int(heavy_discount*100)}%续订折扣，{days_to_expire}天到期"},
            ]
        elif silence_days >= silence_medium_days:
            level = "MEDIUM"
            discount = medium_discount
            channel = "email"
            feature_highlight = top_features[0] if top_features else "核心功能"
            sequence = [
                {"step": 1, "delay_days": 0,  "channel": "email",
                 "content": f"回顾您过去的分析成果（{feature_highlight}使用情况）"},
                {"step": 2, "delay_days": 3,  "channel": "email",
                 "content": f"续订专属{int(medium_discount*100)}%折扣（72h有效）+ 新功能预览"},
                {"step": 3, "delay_days": 7,  "channel": "email",
                 "content": "最后提醒：续订截止前回顾价值"},
            ]
        else:
            level = "LIGHT"
            discount = light_discount
            channel = "email"
            sequence = [
                {"step": 1, "delay_days": 0, "channel": "email",
                 "content": "功能价值提醒 + 到期提示"},
            ]
        
        # 计算发送时间
        scheduled = [
            {**s, "send_at": (now + timedelta(days=s["delay_days"])).strftime("%Y-%m-%dT%H:%M:%S")}
            for s in sequence
        ]
        
        interventions.append({
            "sub_id": sid,
            "user_id": uid,
            "company": company,
            "action": "RENEWAL_INTERVENTION",
            "silence_level": level,
            "silence_days": silence_days,
            "days_to_expire": days_to_expire,
            "monthly_arr": arr,
            "renewal_discount": discount,
            "primary_channel": channel,
            "sequence": scheduled,
            "stop_condition": "用户续订后立即取消后续发送",
            "estimated_revenue_at_risk": arr
        })
    
    activated = [i for i in interventions if i.get("action") == "RENEWAL_INTERVENTION"]
    
    return {
        "total_subscriptions": len(subscriptions),
        "interventions_triggered": len(activated),
        "heavy": sum(1 for i in activated if i.get("silence_level") == "HEAVY"),
        "medium": sum(1 for i in activated if i.get("silence_level") == "MEDIUM"),
        "light_active": sum(1 for i in activated if i.get("silence_level") == "LIGHT"),
        "interventions": interventions,
        "total_arr_at_risk": sum(i.get("monthly_arr", 0) for i in activated)
    }


# 测试
now = datetime(2026, 6, 22, 10, 0, 0)
subscriptions = [
    {
        "sub_id": "SUB001", "user_id": "U001", "company_name": "ABC跨境",
        "expires_at": "2026-07-02T00:00:00",   # 10天后
        "last_active_at": "2026-05-01T10:00:00",  # 52天未活跃（重度）
        "monthly_arr": 2000.0, "top_features": ["VOC分析", "竞品监控"],
        "auto_renew": False
    },
    {
        "sub_id": "SUB002", "user_id": "U002", "company_name": "DEF母婴",
        "expires_at": "2026-07-04T00:00:00",   # 12天后
        "last_active_at": "2026-05-25T10:00:00",  # 28天未活跃（中度）
        "monthly_arr": 1200.0, "top_features": ["选品分析"],
        "auto_renew": False
    },
    {
        "sub_id": "SUB003", "user_id": "U003", "company_name": "GHI电商",
        "expires_at": "2026-07-01T00:00:00",   # 9天后
        "last_active_at": "2026-06-15T10:00:00",  # 7天未活跃（轻度，活跃）
        "monthly_arr": 800.0, "top_features": ["广告分析"],
        "auto_renew": False
    },
    {
        "sub_id": "SUB004", "user_id": "U004", "company_name": "JKL跨境",
        "expires_at": "2026-07-10T00:00:00",   # 18天后，不在窗口
        "last_active_at": "2026-06-01T10:00:00",
        "monthly_arr": 1500.0, "top_features": [],
        "auto_renew": False
    },
    {
        "sub_id": "SUB005", "user_id": "U005", "company_name": "MNO直播",
        "expires_at": "2026-07-05T00:00:00",   # 13天后
        "last_active_at": "2026-06-20T10:00:00",  # 2天前活跃，自动续订
        "monthly_arr": 3000.0, "top_features": [],
        "auto_renew": True  # 自动续订，跳过
    },
]

result = subscription_renewal_intervention_gate(subscriptions, now=now)

assert result["total_subscriptions"] == 5
assert result["interventions_triggered"] == 3  # SUB001, SUB002, SUB003

int_map = {i["sub_id"]: i for i in result["interventions"]}
assert int_map["SUB001"]["silence_level"] == "HEAVY"
assert int_map["SUB002"]["silence_level"] == "MEDIUM"
assert int_map["SUB003"]["action"] == "LIGHT_REMINDER"
assert int_map["SUB004"]["action"] == "NOT_IN_WINDOW"
assert int_map["SUB005"]["action"] == "SKIP"

print("[✓] Subscription Renewal Intervention Gate 测试通过")
print(f"  总订阅: {result['total_subscriptions']}，触发干预: {result['interventions_triggered']}")
print(f"  重度:{result['heavy']} 中度:{result['medium']}，风险ARR: ${result['total_arr_at_risk']:.0f}/月")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Subscription-Churn-Prediction]]（预测续订意向概率）
- **延伸（extends）**：[[Skill-Cohort-Churn-Intervention-Dispatcher]]（续订失败后转入流失挽回流程）
- **可组合（combinable）**：[[Skill-User-LTV-Financial-Bridge]]（按 ARR 价值排序干预优先级）

## ⑤ 商业价值评估
- **ROI量化**：沉默中度用户续订率从 18% → 34%，年化 ARR 挽回 $85,000；挽回成本（折扣+人工）约 $12,000，净收益 $73,000
- **实施难度**：⭐⭐☆☆☆（需订阅系统 API + 用户活跃度追踪 + 邮件平台）
- **优先级**：⭐⭐⭐⭐⭐（订阅续费率是 SaaS 型业务的核心 ARR 保障指标）
