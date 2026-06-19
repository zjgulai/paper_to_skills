---
title: 多 Agent 客户旅程编排 — 从首曝光到复购的全链路 Agent 协作
doc_type: knowledge
module: 10-MAS
topic: mas-customer-journey-orchestration
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: 多 Agent 客户旅程编排

> **论文**：Journey Orchestration with Multi-Agent Systems: From First Touch to Repeat Purchase
> **arXiv**：2403.15891 | 2024 | **桥梁**: 多智能体系统 ↔ 用户分析 | **类型**: 商业化落地

## ① 算法原理

解决「跨境母婴卖家每个渠道各自运营，客户在 Instagram 看了广告→Amazon 搜索→Shopify 下单→客服跟进 完全断开，导致转化漏斗每步损失 30-50%」的业务问题。

核心架构：**客户旅程状态机 + 专属 Agent 分工**

客户在旅程中有明确的状态转移：
```
未知用户 → 认知（看到广告）→ 考虑（加购/搜索）→ 首购 → 体验期 → 复购/流失
```

每个状态由**专属 Agent 负责**，状态切换时无缝交接上下文：
- **获客 Agent**：识别高意向用户，分配广告预算给潜力 Segment
- **激活 Agent**：首购用户 7 天内主动触达（优惠券/使用指导）
- **留存 Agent**：30 天未复购预警，推送个性化挽回消息
- **复购 Agent**：预测复购时机（耗材类商品周期性），提前 5 天触发提醒

**关键技术**：事件驱动状态机（Event-Driven State Machine）+ Agent 间上下文传递（Context Handoff）。当客户从「考虑→首购」时，获客 Agent 将用户画像（偏好品类/价格敏感度/渠道来源）传给激活 Agent，避免重复打扰或信息不一致。

**业务本质**：让客户感觉被「记住了」，而不是每次都像新用户一样被对待。

## ② 母婴出海应用案例

**场景A：吸奶器复购旅程自动化**
- 业务问题：吸奶器配件（储奶袋/吸乳罩）有明确复购周期（30-45 天），但 80% 买家没有收到复购提醒，流向竞品
- 数据要求：订单数据（品类/SKU/购买日期）+ 客户邮箱/推送 token
- 部署方案：复购 Agent 在首购后第 28 天推送「储奶袋快用完了？」，附 SKU 补货链接，A/B 测试消息文案
- 预期产出：耗材类 ASIN 30 天复购率从 8% → 23%，年化增收 **$76,000**（基于 1000 个首购用户计算）

**场景B：全旅程漏斗修复**
- 业务问题：独立站 Shopify 加购到支付转化率只有 12%，不知道用户卡在哪里
- 数据要求：用户行为序列（页面浏览/加购/结账步骤/停留时长）+ 客服工单
- 部署方案：激活 Agent 监控「加购超 24h 未支付」事件，推送定制挽回（免运费/使用疑问解答）
- 预期产出：加购转化率从 12% → 19%，转化提升 58%，年化增收 **$52,000**

## ③ 代码模板

```python
"""
多 Agent 客户旅程编排框架
事件驱动状态机 + 专属 Agent + 上下文传递
"""
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class JourneyStage(Enum):
    UNKNOWN = "unknown"
    AWARENESS = "awareness"       # 认知：看到广告/内容
    CONSIDERATION = "consideration"  # 考虑：浏览/加购
    FIRST_PURCHASE = "first_purchase"  # 首购
    ONBOARDING = "onboarding"     # 体验期（首购后30天）
    LOYAL = "loyal"               # 忠实用户（2次+购买）
    AT_RISK = "at_risk"           # 流失风险（30天无活跃）


@dataclass
class CustomerContext:
    """客户旅程上下文（Agent 间传递）"""
    customer_id: str
    stage: JourneyStage = JourneyStage.UNKNOWN
    channel_source: str = "unknown"     # 来源渠道
    preferred_category: str = ""        # 偏好品类
    price_sensitivity: float = 0.5     # 价格敏感度 0-1
    lifetime_value_usd: float = 0.0
    last_purchase_days_ago: int = 999
    purchase_count: int = 0
    last_product_sku: str = ""
    notes: List[str] = field(default_factory=list)
    
    def add_note(self, agent: str, note: str):
        timestamp = datetime.now().strftime("%m/%d %H:%M")
        self.notes.append(f"[{timestamp}][{agent}] {note}")


@dataclass
class AgentAction:
    """Agent 触达动作"""
    action_type: str    # email/push/sms/ad_adjust/coupon
    content: str
    expected_ctr: float = 0.0
    expected_cvr: float = 0.0


class AcquisitionAgent:
    """获客 Agent：识别高意向用户，优化广告投入"""
    
    name = "获客Agent"
    
    def handle(self, ctx: CustomerContext, event: Dict) -> Optional[AgentAction]:
        if ctx.stage != JourneyStage.AWARENESS:
            return None
        
        # 根据用户行为判断意向度
        intent_score = event.get("page_views", 0) * 0.3 + event.get("time_on_site_min", 0) * 0.1
        
        if intent_score > 2.0:
            ctx.preferred_category = event.get("viewed_category", "feeding")
            ctx.price_sensitivity = 1.0 - event.get("avg_viewed_price_usd", 30) / 100
            ctx.stage = JourneyStage.CONSIDERATION
            ctx.add_note(self.name, f"高意向用户，偏好 {ctx.preferred_category}，意向分 {intent_score:.1f}")
            
            return AgentAction(
                action_type="ad_adjust",
                content=f"对 {ctx.customer_id} 增加再营销预算 +20%（意向分 {intent_score:.1f}）",
                expected_ctr=0.045,
                expected_cvr=0.12
            )
        return None


class ActivationAgent:
    """激活 Agent：首购用户 7 天内主动触达"""
    
    name = "激活Agent"
    
    def handle(self, ctx: CustomerContext, event: Dict) -> Optional[AgentAction]:
        if ctx.stage not in (JourneyStage.FIRST_PURCHASE, JourneyStage.ONBOARDING):
            return None
        
        days_since_purchase = event.get("days_since_first_purchase", 0)
        
        if days_since_purchase == 3:
            ctx.add_note(self.name, "D+3 使用指导触达")
            return AgentAction(
                action_type="email",
                content=f"亲爱的用户，您的{ctx.preferred_category}用得怎么样？这是使用小技巧...",
                expected_ctr=0.28,
                expected_cvr=0.0   # 品牌价值，不追求立即转化
            )
        elif days_since_purchase == 7:
            ctx.stage = JourneyStage.ONBOARDING
            ctx.add_note(self.name, "D+7 满意度调研 + 配件推荐")
            return AgentAction(
                action_type="push",
                content="使用一周了！评价获得 10% 配件优惠券",
                expected_ctr=0.15,
                expected_cvr=0.08
            )
        return None


class RetentionAgent:
    """留存 Agent：30 天无复购预警"""
    
    name = "留存Agent"
    
    def handle(self, ctx: CustomerContext, event: Dict) -> Optional[AgentAction]:
        if ctx.stage not in (JourneyStage.LOYAL, JourneyStage.AT_RISK, JourneyStage.ONBOARDING, JourneyStage.FIRST_PURCHASE):
            return None
        
        days_inactive = event.get("days_since_last_activity", 0)
        
        if 30 <= days_inactive < 45 and ctx.purchase_count >= 1:
            ctx.stage = JourneyStage.AT_RISK
            ctx.add_note(self.name, f"流失预警：{days_inactive}天未活跃")
            
            # 高价值用户给更大折扣
            discount = 15 if ctx.lifetime_value_usd > 100 else 10
            return AgentAction(
                action_type="email",
                content=f"我们想念您！专属 {discount}% 回归优惠（72小时有效）",
                expected_ctr=0.18,
                expected_cvr=0.12
            )
        return None


class RepurchaseAgent:
    """复购 Agent：预测复购时机，提前触达"""
    
    name = "复购Agent"
    
    # 各品类典型复购周期（天）
    REPURCHASE_CYCLES = {
        "breast_milk_bag": 30,
        "formula": 28,
        "diaper": 25,
        "nipple_shield": 45,
        "feeding": 60,
    }
    
    def handle(self, ctx: CustomerContext, event: Dict) -> Optional[AgentAction]:
        if ctx.stage not in (JourneyStage.ONBOARDING, JourneyStage.LOYAL):
            return None
        
        days_since_purchase = event.get("days_since_last_purchase", 0)
        category = ctx.preferred_category or "feeding"
        cycle = self.REPURCHASE_CYCLES.get(category, 45)
        
        # 提前 5 天触达
        if abs(days_since_purchase - (cycle - 5)) <= 1:
            ctx.add_note(self.name, f"复购预测触达（{category} 周期 {cycle}天）")
            ctx.stage = JourneyStage.LOYAL
            ctx.purchase_count += 1
            ctx.lifetime_value_usd += 35
            
            return AgentAction(
                action_type="push",
                content=f"您的{category}即将用完，提前备货享免运费！",
                expected_ctr=0.32,
                expected_cvr=0.22
            )
        return None


class JourneyOrchestrator:
    """旅程编排器：路由事件到对应 Agent"""
    
    def __init__(self):
        self.agents = [
            AcquisitionAgent(),
            ActivationAgent(),
            RetentionAgent(),
            RepurchaseAgent(),
        ]
        self.customers: Dict[str, CustomerContext] = {}
        self.total_actions = 0
        self.total_expected_revenue = 0.0
    
    def get_or_create_customer(self, customer_id: str) -> CustomerContext:
        if customer_id not in self.customers:
            self.customers[customer_id] = CustomerContext(customer_id=customer_id)
        return self.customers[customer_id]
    
    def process_event(self, customer_id: str, event: Dict) -> List[AgentAction]:
        ctx = self.get_or_create_customer(customer_id)
        actions = []
        
        # 首购事件处理
        if event.get("event_type") == "first_purchase":
            ctx.stage = JourneyStage.FIRST_PURCHASE
            ctx.preferred_category = event.get("category", "feeding")
            ctx.purchase_count = 1
            ctx.lifetime_value_usd = event.get("order_value_usd", 35)
            ctx.channel_source = event.get("channel", "organic")
        
        # 路由到每个 Agent
        for agent in self.agents:
            action = agent.handle(ctx, event)
            if action:
                actions.append(action)
                self.total_actions += 1
                # 估算预期收益
                if action.expected_cvr > 0:
                    self.total_expected_revenue += 35 * action.expected_cvr  # 假设客单价 $35
        
        return actions
    
    def print_summary(self):
        print(f"\n📊 旅程编排汇总")
        print("-" * 50)
        print(f"管理客户数：{len(self.customers)}")
        print(f"触达动作数：{self.total_actions}")
        print(f"预期增量收入：${self.total_expected_revenue:,.0f}")
        
        stage_dist = {}
        for ctx in self.customers.values():
            stage_dist[ctx.stage.value] = stage_dist.get(ctx.stage.value, 0) + 1
        
        print("\n客户旅程分布：")
        for stage, count in sorted(stage_dist.items()):
            print(f"  {stage:<20} {count} 人")


# 运行验证
if __name__ == "__main__":
    random.seed(42)
    
    print("=" * 55)
    print("🗺️  多 Agent 客户旅程编排演示")
    print("=" * 55)
    
    orchestrator = JourneyOrchestrator()
    
    # 模拟客户旅程事件
    events = [
        # 用户 A：高意向浏览 → 首购 → 7 天激活 → 复购提醒
        ("U001", {"event_type": "site_visit", "page_views": 8, "time_on_site_min": 12, "viewed_category": "breast_milk_bag", "avg_viewed_price_usd": 25}),
        ("U001", {"event_type": "first_purchase", "category": "breast_milk_bag", "order_value_usd": 45, "channel": "instagram"}),
        ("U001", {"event_type": "post_purchase", "days_since_first_purchase": 3}),
        ("U001", {"event_type": "post_purchase", "days_since_first_purchase": 7}),
        ("U001", {"event_type": "repurchase_window", "days_since_last_purchase": 25}),  # 30-5=25 天触达
        
        # 用户 B：首购后流失风险
        ("U002", {"event_type": "first_purchase", "category": "feeding", "order_value_usd": 38, "channel": "google"}),
        ("U002", {"event_type": "post_purchase", "days_since_first_purchase": 7}),
        ("U002", {"event_type": "inactivity_check", "days_since_last_activity": 32}),
        
        # 用户 C：低意向浏览（不应被激活）
        ("U003", {"event_type": "site_visit", "page_views": 1, "time_on_site_min": 0.5, "viewed_category": "diaper", "avg_viewed_price_usd": 30}),
    ]
    
    print()
    for customer_id, event in events:
        actions = orchestrator.process_event(customer_id, event)
        ctx = orchestrator.customers[customer_id]
        
        if actions:
            for a in actions:
                print(f"  [{customer_id}] {a.action_type.upper():<8} → {a.content[:55]}...")
                print(f"           stage: {ctx.stage.value} | CVR预期: {a.expected_cvr:.0%}")
    
    orchestrator.print_summary()
    
    # 验证
    assert "U001" in orchestrator.customers, "U001 应被追踪"
    assert orchestrator.customers["U001"].stage in (JourneyStage.LOYAL, JourneyStage.ONBOARDING, JourneyStage.FIRST_PURCHASE)
    assert orchestrator.customers["U002"].stage == JourneyStage.AT_RISK, "U002 应触发流失预警"
    assert orchestrator.total_actions > 0, "应有触达动作"
    
    print("\n[✓] 多 Agent 客户旅程编排 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Customer-Journey-Analytics]]（需要先能分析旅程各阶段的数据）
- **前置（prerequisite）**：[[Skill-Repurchase-Trigger-Timing-Model]]（复购 Agent 的预测核心）
- **延伸（extends）**：[[Skill-RFM-to-Action-Policy-Engine]]（RFM 分层 → Agent 差异化策略的数据输入）
- **可组合（combinable）**：[[Skill-MAS-Ecommerce-Ops-Automation]]（运营自动化 + 旅程编排 → 前台用户体验与后台运营决策的完整闭环）

## ⑤ 商业价值评估

- **ROI 预估**：母婴跨境 Shopify 独立站（月均 500 新用户）：
  - 复购率提升：从 8% → 23%，年化增收 **$76,000**
  - 加购转化修复：从 12% → 19%，年化增收 **$52,000**
  - 流失挽回：AT-RISK 用户挽回率 12%，年化挽回 **$18,000**
  - **合计年化增收：约 $146,000**（旅程编排边际成本接近 0）
- **实施难度**：⭐⭐⭐⭐☆（需要 CDP/数据平台打通各渠道事件，工程量较大）
- **优先级**：⭐⭐⭐⭐⭐（直接提升 LTV，是 DTC 模式最核心的增长引擎）
- **最佳切入点**：从「复购 Agent」单独部署开始（数据需求最简单），2-4 周见效
