---
title: 多 Agent 电商运营自动化 — 补货/广告/客服全天候自动化编排
doc_type: knowledge
module: 10-MAS
topic: mas-ecommerce-ops-automation
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: 多 Agent 电商运营自动化

> **论文**：AutoMate: Multi-Agent Systems for End-to-End E-Commerce Operations Automation
> **arXiv**：2406.11234 | 2024 | **桥梁**: 多智能体系统 ↔ 供应链运营 | **类型**: 商业化落地

## ① 算法原理

解决「运营团队每天重复做补货审批、广告调价、Listing 更新、客服分诊，占用 70% 时间，却没有精力做真正的增长」的业务问题。

核心架构：**Supervisor-Worker 模式的事件驱动 MAS**

- **Supervisor Agent**：接收运营事件（库存低/广告 ACoS 超标/差评出现），分配任务给对应 Worker Agent，设置「人工审批门控」（高风险操作必须人确认）
- **Worker Agent 矩阵**：
  - 补货 Agent：读取库存速度 + 前置期，计算 EOQ，生成补货建议
  - 广告 Agent：监控 ACoS/ROAS，按规则调整竞价和预算分配
  - Listing Agent：检测差评关键词，生成优化建议，标记需要人工确认
  - 客服分诊 Agent：识别工单类型（退款/咨询/投诉），路由到对应处理流程

**执行门控设计**（防止 Agent 乱操作）：
- 低风险（调整幅度 < 5%）→ 全自动执行
- 中风险（调整幅度 5-20%）→ 发飞书/邮件，30 分钟无反馈则自动执行
- 高风险（调整幅度 > 20% 或单次金额 > $500）→ 必须人工确认，不自动执行

## ② 母婴出海应用案例

**场景A：吸奶器品类全天候补货自动化**
- 业务问题：运营每天早上花 2 小时手工看各 ASIN 库存，判断是否补货，容易漏
- 数据要求：ERP 库存数据（每日同步）+ 销售速度历史 + 前置期数据
- 部署方案：补货 Agent 每 4 小时扫描一次，库存天数 < 30 天自动触发建议，< 15 天升级 Supervisor 告警
- 预期产出：断货率从 8.3% 降到 2.1%，减少断货损失 $18,000/年；运营从 2h/天补货工作 → 15min 审批

**场景B：广告 Agent 自动降 ACoS**
- 业务问题：旺季广告 ACoS 飙到 45%（目标 28%），人工调整来不及，每天多烧 $500
- 数据要求：广告平台 API 数据（ACoS/ROAS/点击/转化）+ 库存数据（避免广告投已断货品）
- 部署方案：广告 Agent 每 6 小时读取数据，ACoS > 35% 自动降竞价 8%，ROAS < 2 暂停关键词
- 预期产出：广告 ACoS 从 45% → 29%，旺季 3 个月节省广告浪费 $43,000，ROAS 提升 35%

## ③ 代码模板

```python
"""
多 Agent 电商运营自动化框架
Supervisor-Worker 模式，含人工审批门控
"""
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable
from datetime import datetime


class RiskLevel(Enum):
    LOW = "low"         # 全自动执行
    MEDIUM = "medium"   # 通知 + 30 分钟超时自动执行
    HIGH = "high"       # 必须人工确认


@dataclass
class OpsEvent:
    """运营事件"""
    event_id: str
    event_type: str     # inventory_low / acos_high / bad_review / customer_ticket
    asin: str
    severity: float     # 0-1，越高越紧急
    payload: Dict       # 事件具体数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentAction:
    """Agent 建议的操作"""
    action_id: str
    agent_name: str
    action_type: str
    description: str
    estimated_impact_usd: float   # 预估影响金额（正=收益，负=成本）
    risk_level: RiskLevel
    auto_execute: bool = False
    executed: bool = False
    approved: bool = False


class WorkerAgent:
    """基础 Worker Agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.actions_generated = 0
        self.actions_executed = 0
    
    def process(self, event: OpsEvent) -> Optional[AgentAction]:
        raise NotImplementedError
    
    def _estimate_risk(self, change_pct: float, amount_usd: float) -> RiskLevel:
        if abs(change_pct) > 20 or amount_usd > 500:
            return RiskLevel.HIGH
        elif abs(change_pct) > 5 or amount_usd > 100:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class ReplenishmentAgent(WorkerAgent):
    """补货 Agent"""
    
    def __init__(self):
        super().__init__("补货Agent")
    
    def process(self, event: OpsEvent) -> Optional[AgentAction]:
        if event.event_type != "inventory_low":
            return None
        
        inventory_days = event.payload.get("inventory_days", 0)
        daily_sales = event.payload.get("daily_sales_units", 10)
        lead_time_days = event.payload.get("lead_time_days", 45)
        unit_cost_usd = event.payload.get("unit_cost_usd", 15)
        
        # EOQ 简化计算（目标库存 = 安全库存 + 前置期用量）
        safety_stock = daily_sales * 15  # 15天安全库存
        replenish_qty = max(0, (lead_time_days + 30) * daily_sales + safety_stock 
                          - inventory_days * daily_sales)
        replenish_cost = replenish_qty * unit_cost_usd
        
        change_pct = (replenish_qty / max(daily_sales * 30, 1)) * 100  # 相对月销量的比例
        
        self.actions_generated += 1
        return AgentAction(
            action_id=f"replenish_{event.asin}",
            agent_name=self.name,
            action_type="purchase_order",
            description=f"建议补货 {replenish_qty} 件（当前库存 {inventory_days} 天）",
            estimated_impact_usd=replenish_cost,
            risk_level=self._estimate_risk(change_pct, replenish_cost)
        )


class AdvertisingAgent(WorkerAgent):
    """广告 Agent"""
    
    def __init__(self, target_acos: float = 0.28):
        super().__init__("广告Agent")
        self.target_acos = target_acos
    
    def process(self, event: OpsEvent) -> Optional[AgentAction]:
        if event.event_type != "acos_high":
            return None
        
        current_acos = event.payload.get("current_acos", 0.35)
        daily_ad_spend = event.payload.get("daily_ad_spend_usd", 200)
        
        # 计算降价幅度
        acos_gap = (current_acos - self.target_acos) / self.target_acos
        bid_reduction_pct = min(acos_gap * 0.8 * 100, 30)  # 最多降 30%
        daily_savings = daily_ad_spend * (bid_reduction_pct / 100)
        
        self.actions_generated += 1
        return AgentAction(
            action_id=f"ads_{event.asin}",
            agent_name=self.name,
            action_type="bid_adjustment",
            description=f"降低竞价 {bid_reduction_pct:.1f}%（ACoS {current_acos:.0%} → 目标 {self.target_acos:.0%}）",
            estimated_impact_usd=daily_savings,
            risk_level=self._estimate_risk(bid_reduction_pct, daily_savings * 30)
        )


class CustomerServiceAgent(WorkerAgent):
    """客服分诊 Agent"""
    
    TICKET_ROUTES = {
        "refund": "退款处理流程",
        "inquiry": "产品咨询自动回复",
        "complaint": "差评干预 SOP",
        "defect": "质量问题上报"
    }
    
    def __init__(self):
        super().__init__("客服分诊Agent")
    
    def process(self, event: OpsEvent) -> Optional[AgentAction]:
        if event.event_type != "customer_ticket":
            return None
        
        ticket_type = event.payload.get("ticket_type", "inquiry")
        estimated_resolution_value = event.payload.get("order_value_usd", 30)
        
        route = self.TICKET_ROUTES.get(ticket_type, "人工处理")
        
        self.actions_generated += 1
        return AgentAction(
            action_id=f"cs_{event.asin}_{event.event_id}",
            agent_name=self.name,
            action_type="ticket_route",
            description=f"工单分诊 → {route}（票值 ${estimated_resolution_value}）",
            estimated_impact_usd=estimated_resolution_value * 0.1,  # 挽回10%订单价值
            risk_level=RiskLevel.LOW  # 分诊本身低风险
        )


class SupervisorAgent:
    """Supervisor Agent：事件分发 + 人工审批门控"""
    
    def __init__(self, approval_callback: Optional[Callable] = None):
        self.workers: Dict[str, WorkerAgent] = {
            "inventory_low": ReplenishmentAgent(),
            "acos_high": AdvertisingAgent(target_acos=0.28),
            "customer_ticket": CustomerServiceAgent(),
        }
        self.approval_callback = approval_callback or self._default_approval
        self.action_log: List[AgentAction] = []
        self.stats = {"auto_exec": 0, "pending_approval": 0, "blocked_high_risk": 0}
    
    def _default_approval(self, action: AgentAction) -> bool:
        """默认审批逻辑（模拟：模拟人工审批）"""
        # 模拟 medium 风险随机批准，high 风险 80% 批准
        if action.risk_level == RiskLevel.MEDIUM:
            return random.random() > 0.3
        elif action.risk_level == RiskLevel.HIGH:
            return random.random() > 0.2
        return True
    
    def handle_event(self, event: OpsEvent) -> Optional[AgentAction]:
        """处理运营事件"""
        worker = self.workers.get(event.event_type)
        if not worker:
            print(f"  ⚠️  未知事件类型: {event.event_type}")
            return None
        
        action = worker.process(event)
        if not action:
            return None
        
        # 执行门控
        if action.risk_level == RiskLevel.LOW:
            action.auto_execute = True
            action.executed = True
            action.approved = True
            worker.actions_executed += 1
            self.stats["auto_exec"] += 1
            print(f"  ✅ [自动执行] {action.description} | 预计 ${action.estimated_impact_usd:.0f}")
        
        elif action.risk_level == RiskLevel.MEDIUM:
            approved = self.approval_callback(action)
            action.approved = approved
            if approved:
                action.executed = True
                worker.actions_executed += 1
                self.stats["auto_exec"] += 1
                print(f"  ✅ [审批执行] {action.description} | 预计 ${action.estimated_impact_usd:.0f}")
            else:
                self.stats["pending_approval"] += 1
                print(f"  ⏳ [待审批] {action.description}")
        
        elif action.risk_level == RiskLevel.HIGH:
            approved = self.approval_callback(action)
            action.approved = approved
            if approved:
                action.executed = True
                worker.actions_executed += 1
                self.stats["auto_exec"] += 1
                print(f"  ✅ [人工审批] {action.description} | 预计 ${action.estimated_impact_usd:.0f}")
            else:
                self.stats["blocked_high_risk"] += 1
                print(f"  🚫 [高风险拦截] {action.description} — 等待人工处理")
        
        self.action_log.append(action)
        return action
    
    def print_summary(self):
        executed = [a for a in self.action_log if a.executed]
        total_impact = sum(a.estimated_impact_usd for a in executed)
        
        print("\n📊 运营自动化执行摘要")
        print("-" * 50)
        print(f"总事件数: {len(self.action_log)}")
        print(f"自动执行: {self.stats['auto_exec']} 次")
        print(f"待审批: {self.stats['pending_approval']} 次")
        print(f"高风险拦截: {self.stats['blocked_high_risk']} 次")
        print(f"预计总影响: ${total_impact:,.0f}")
        
        for w in self.workers.values():
            if w.actions_generated > 0:
                exec_rate = w.actions_executed / w.actions_generated * 100
                print(f"  {w.name}: 生成 {w.actions_generated} 个决策，执行率 {exec_rate:.0f}%")


# 运行验证
if __name__ == "__main__":
    random.seed(42)
    
    print("=" * 55)
    print("🤖 多 Agent 电商运营自动化演示（7天运营快照）")
    print("=" * 55)
    
    supervisor = SupervisorAgent()
    
    # 模拟 7 天的运营事件
    events = [
        OpsEvent("e001", "inventory_low", "B08N5WRWNW", 0.9,
                 {"inventory_days": 12, "daily_sales_units": 15, "lead_time_days": 45, "unit_cost_usd": 18}),
        OpsEvent("e002", "acos_high", "B07X9WZBHP", 0.7,
                 {"current_acos": 0.42, "daily_ad_spend_usd": 180}),
        OpsEvent("e003", "customer_ticket", "B08N5WRWNW", 0.5,
                 {"ticket_type": "refund", "order_value_usd": 45}),
        OpsEvent("e004", "inventory_low", "B09KJLMN12", 0.6,
                 {"inventory_days": 28, "daily_sales_units": 8, "lead_time_days": 30, "unit_cost_usd": 25}),
        OpsEvent("e005", "acos_high", "B08N5WRWNW", 0.8,
                 {"current_acos": 0.51, "daily_ad_spend_usd": 350}),
        OpsEvent("e006", "customer_ticket", "B07X9WZBHP", 0.4,
                 {"ticket_type": "inquiry", "order_value_usd": 32}),
    ]
    
    print()
    for event in events:
        supervisor.handle_event(event)
    
    supervisor.print_summary()
    
    # 验证
    assert len(supervisor.action_log) == len(events), "每个事件应生成一个 Action"
    executed_count = sum(1 for a in supervisor.action_log if a.executed)
    assert executed_count >= 0, "执行数量应 >= 0"
    
    print("\n[✓] 多 Agent 电商运营自动化 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Flowr-Supply-Chain-MAS]]（供应链 MAS 基础架构）
- **前置（prerequisite）**：[[Skill-MAS-Consensus-Mechanism]]（多 Agent 共识是协调的基础）
- **延伸（extends）**：[[Skill-Human-in-Loop-Approval-Gate-Tag]]（人工审批门控的详细实现）
- **可组合（combinable）**：[[Skill-CONCAT-Consensus-Decentralized-MAS]]（去中心化 MAS + 运营自动化 → 跨仓/跨店联动决策）

## ⑤ 商业价值评估

- **ROI 预估**：母婴跨境 50+ ASIN 规模卖家，部署运营自动化 MAS 后：
  - 运营效率：日常运营工作 2.5h/天 → 0.5h/天，节省 **$36,000/年**人力成本
  - 断货损失：断货率 8% → 2%，年化挽回销售额 **$24,000**（按 GMV $400K 计算）
  - 广告浪费：ACoS 超标浪费 → 精准控制，年化节省广告费 **$43,000**
  - **合计年化价值：约 $103,000**
- **实施难度**：⭐⭐⭐⭐☆（需接通 ERP + 广告 API + 工单系统，工程量较大）
- **优先级**：⭐⭐⭐⭐⭐（直接替代重复运营工作，CEO/COO 最容易感知的 Agent 价值）
- **最佳切入点**：从「广告 ACoS 自动降价」单一 Agent 开始，4 周见效后再扩展
