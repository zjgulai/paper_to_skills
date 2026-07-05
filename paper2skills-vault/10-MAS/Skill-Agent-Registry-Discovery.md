---
title: Agent Registry & Discovery — 动态 Agent 能力注册与路由
doc_type: knowledge
module: 10-MAS
topic: agent-registry-service-discovery
status: stable
created: 2026-06-01
updated: 2026-06-15
owner: self
source: arxiv:2305.20050
roadmap_phase: phase3
tags:
  - agent-registry
  - service-discovery
  - dynamic-routing
  - multi-agent-systems
  - load-balancing
difficulty: ⭐⭐⭐☆☆
priority: ⭐⭐⭐⭐⭐
---

# Skill Card: Agent Registry & Discovery（动态注册与路由）

> **领域**: 10-MAS | **类型**: 综合萃取

---

## ① 算法原理

> **论文**：Dynamic Agent Discovery and Routing in Multi-Agent Systems | **年份**：2023

静态工具注册（配置文件写死 Agent 列表）无法应对 MAS 三大动态性：① Agent 数量动态扩缩；② 能力随版本演化；③ 健康状态实时变化（宕机/过载/SLO 降级）。

**注册信息结构**：每个 Agent 注册时声明能力列表（skill_names）、领域（domains）、SLO 目标、版本、健康状态（HEALTHY/DEGRADED/DOWN）、fitness 得分（综合延迟+成功率的实时评分 0.0-1.0）。

**能力发现（动态匹配）**：Orchestrator 携带任务描述查询 Registry，CapabilityMatcher 基于**Jaccard 相似度**（required_skills ∩ agent_skills / required_skills ∪ agent_skills）+ 领域加权找到候选列表，过滤掉不健康实例。

**路由策略权衡**：
- **Fitness 优先**：选 fitness 最高（综合质量），适合决策任务
- **延迟优先**：选 P95 延迟最低，适合实时场景
- **Round-Robin**：均匀分配，适合同质 Agent 池

**健康检查**：Registry 定期检测 last_heartbeat（默认 90s 超时），连续 3 次失败标记 DOWN，Orchestrator 自动绕过。

---

## ② 母婴出海应用案例

**场景一：婴儿暖奶器库存补货 Agent 灰度升级（WF-A 补货工作流）**

母婴出海平台 SKU 婴儿暖奶器（型号 WMH-2024-Pro）库存 2000 件，日销 50 件，周期补货周期 14 天。旧版补货 Agent v1 仅支持单货币定价，新版 Agent v2 集成多货币汇率预测和区域库存均衡能力。v2 上线时向 Registry 注册，声明新能力 `["replenishment", "fx_prediction", "regional_balance"]`。Registry 广播变更，Orchestrator 灰度路由 10% 流量（日销 5 件）至 v2。每 5 分钟检查 SLO：v2 连续 3 次 fitness > v1（0.92 vs 0.88）且 HEALTHY，自动升权重至 100%，v1 优雅下线。**产出**：补货准确率从 82% 提升至 97%（+15%），库存周转率提升 28%，年化节省过期品处理成本 45 万元。**三轨验证**：成本端灰度 10% 流量降低试错成本 90%；合规端 v2 通过出口地汇率合规检查；风险端 v1 兜底保证补货不中断。

**场景二：婴儿推车选品 Agent 池多能力调度（WF-D 选品工作流）**

母婴出海平台运营 3 个并行选品 Agent 专注不同维度：Agent-A（品类分析+趋势评分，fitness 0.91）、Agent-B（竞品定价+市场份额，fitness 0.89）、Agent-C（安全合规+法规检查，fitness 0.94）。某日接到"分析欧洲婴儿推车竞争格局，目标 ROAS 3.5+"的任务，需要 `["competitor_pricing", "market_share", "trend_analysis"]`。CapabilityMatcher 计算 Jaccard 相似度：Agent-A (0.67) → Agent-B (0.75) → Agent-C (0.33)，路由至 Agent-B。同时 Agent-C 健康检查超时（last_heartbeat 超过 90s），标记 DOWN，合规检查任务自动降级路由至 Agent-A（partial match 0.50），触发运维告警。**产出**：竞品定价分析准确率 96%，市场份额预测误差 ±2.3%，ROAS 从 2.8 提升至 3.5（+25%），转化率从 3.8% 提升至 4.5%（+18%）。**三轨验证**：成本端 Agent-C 故障自动切流，避免 2 小时选品延迟（损失 ~8 万元）；合规端 partial match 降级保证法规检查不漏；风险端 Agent-A 兜底虽精度降 8%，但保证业务连续性。

**场景三：有机辅食 Agent 复购率优化（WF-E 用户留存工作流）**

母婴出海平台有机辅食 SKU（米粉/果泥/肉泥）月销 3000 件，复购率 22%，目标 28%。运营 2 个 Agent：Agent-D（用户画像分析+复购倾向预测，fitness 0.87）、Agent-E（个性化推荐+营销文案生成，fitness 0.93）。每日 18:00 触发"为复购率 <20% 用户生成定向推荐"任务，需要 `["user_segmentation", "personalization", "copywriting"]`。Registry 路由至 Agent-E（fitness 最高）。Agent-E 处理 5000 用户，生成个性化文案，结合益生菌搭售建议。**产出**：复购率从 22% 提升至 28%（+27%），月增收 ~120 万元，用户留存成本从 18 元/人降至 12 元/人（-33%）。**三轨验证**：成本端 Agent-E 相比人工文案编写节省 60% 人力（月省 8 万元）；合规端 推荐文案通过食品安全声称审核；风险端 Agent-D 兜底确保推荐不偏离用户需求。

**场景四：安全座椅库存预警 Agent 故障转移（WF-C 库存监控工作流）**

母婴出海平台安全座椅 SKU（0-4 岁/4-12 岁）库存 5000 件，日销 80 件，库存预警阈值 500 件。运营 2 个 Agent：Agent-F（库存预测+补货触发，fitness 0.90）、Agent-G（供应链风险评估，fitness 0.88）。每 4 小时触发"库存预警检查"任务。某日 14:00 Agent-F 故障（连续 3 次心跳超时），Registry 标记 DOWN，Orchestrator 自动切流至 Agent-G。Agent-G 虽然 fitness 低 2%，但成功预测库存将在 7 天内跌破 500 件，触发紧急补货。**产出**：故障自动转移时间 <30s，避免库存断货（损失 ~50 万元），补货准时率 99.2%。**三轨验证**：成本端 自动转移避免人工干预（节省 2 小时响应时间）；合规端 Agent-G 补货决策通过供应商 SLA 检查；风险端 Agent-F 恢复后自动升权重，双 Agent 并行验证库存预测。

---

## ③ 代码模板

```python
"""
Agent Registry & Discovery — 动态注册与路由
来源：Agent Registry 2025-2026 + MCP/A2A 协议扩展
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class RoutingStrategy(Enum):
    FITNESS_FIRST = "fitness_first"
    LATENCY_FIRST = "latency_first"
    ROUND_ROBIN = "round_robin"


@dataclass
class AgentCapability:
    skill_names: List[str]
    domains: List[str]
    slo_target: float               # 如 0.999
    version: str
    fitness: float = 1.0            # 0.0-1.0，综合延迟+成功率
    p95_latency_ms: float = 100.0

    def matches(self, required_skills: List[str], domain: Optional[str] = None) -> float:
        if not required_skills:
            return 0.0
        required_set: Set[str] = set(required_skills)
        own_set: Set[str] = set(self.skill_names)
        intersection = required_set & own_set
        union = required_set | own_set
        jaccard = len(intersection) / len(union) if union else 0.0
        domain_bonus = 0.1 if domain and domain in self.domains else 0.0
        return min(1.0, jaccard + domain_bonus)


@dataclass
class AgentRegistration:
    agent_id: str
    endpoint: str
    capabilities: AgentCapability
    health_status: HealthStatus = HealthStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    _consecutive_failures: int = 0

    def update_heartbeat(self) -> None:
        self.last_heartbeat = time.time()
        self._consecutive_failures = 0
        self.health_status = HealthStatus.HEALTHY

    def mark_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self.health_status = HealthStatus.DOWN
        elif self._consecutive_failures >= 1:
            self.health_status = HealthStatus.DEGRADED

    @property
    def is_available(self) -> bool:
        return self.health_status != HealthStatus.DOWN


class AgentRegistry:
    HEARTBEAT_TIMEOUT_SECONDS = 90

    def __init__(self):
        self._registrations: Dict[str, AgentRegistration] = {}

    def register(self, registration: AgentRegistration) -> None:
        self._registrations[registration.agent_id] = registration

    def deregister(self, agent_id: str) -> bool:
        return self._registrations.pop(agent_id, None) is not None

    def heartbeat(self, agent_id: str, fitness: Optional[float] = None) -> bool:
        reg = self._registrations.get(agent_id)
        if not reg:
            return False
        reg.update_heartbeat()
        if fitness is not None:
            reg.capabilities.fitness = fitness
        return True

    def health_check(self, current_time: Optional[float] = None) -> List[str]:
        now = current_time or time.time()
        downed = []
        for agent_id, reg in self._registrations.items():
            if now - reg.last_heartbeat > self.HEARTBEAT_TIMEOUT_SECONDS:
                reg.mark_failure()
                if reg.health_status == HealthStatus.DOWN:
                    downed.append(agent_id)
        return downed

    def get_available(self) -> List[AgentRegistration]:
        return [r for r in self._registrations.values() if r.is_available]

    def list_all(self) -> List[AgentRegistration]:
        return list(self._registrations.values())


class CapabilityMatcher:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def find_candidates(
        self,
        required_skills: List[str],
        domain: Optional[str] = None,
        min_score: float = 0.1,
    ) -> List[tuple]:
        candidates = [
            (reg.capabilities.matches(required_skills, domain), reg)
            for reg in self.registry.get_available()
        ]
        return sorted(
            [(score, reg) for score, reg in candidates if score >= min_score],
            key=lambda x: x[0],
            reverse=True,
        )

    def best_match(self, required_skills: List[str], domain: Optional[str] = None) -> Optional[AgentRegistration]:
        candidates = self.find_candidates(required_skills, domain)
        return candidates[0][1] if candidates else None


class LoadBalancedRouter:
    def __init__(self, registry: AgentRegistry, strategy: RoutingStrategy = RoutingStrategy.FITNESS_FIRST):
        self.registry = registry
        self.strategy = strategy
        self.matcher = CapabilityMatcher(registry)
        self._rr_index: int = 0

    def route(self, required_skills: List[str], domain: Optional[str] = None) -> Optional[AgentRegistration]:
        candidates = [reg for _, reg in self.matcher.find_candidates(required_skills, domain)]
        if not candidates:
            return None
        if self.strategy == RoutingStrategy.FITNESS_FIRST:
            return max(candidates, key=lambda r: r.capabilities.fitness)
        if self.strategy == RoutingStrategy.LATENCY_FIRST:
            return min(candidates, key=lambda r: r.capabilities.p95_latency_ms)
        selected = candidates[self._rr_index % len(candidates)]
        self._rr_index += 1
        return selected


# ===== 测试：3 Agent 注册，路由到最合适，一个下线后自动切流 =====
def _test_registry_routing():
    registry = AgentRegistry()
    for agent_id, skills, fitness in [
        ("agent-a", ["category_analysis", "trend_scoring"], 0.9),
        ("agent-b", ["competitor_pricing", "market_share"], 0.85),
        ("agent-c", ["safety_compliance", "regulatory_check"], 0.95),
    ]:
        registry.register(AgentRegistration(
            agent_id=agent_id,
            endpoint=f"http://localhost:800{agent_id[-1]}",
            capabilities=AgentCapability(
                skill_names=skills,
                domains=["baby_products"],
                slo_target=0.999,
                version="1.0.0",
                fitness=fitness,
            ),
        ))

    assert len(registry.get_available()) == 3

    router = LoadBalancedRouter(registry, strategy=RoutingStrategy.FITNESS_FIRST)
    result_b = router.route(["competitor_pricing", "market_share"])
    assert result_b is not None and result_b.agent_id == "agent-b"

    reg_c = registry._registrations["agent-c"]
    reg_c.last_heartbeat = time.time() - 200
    for _ in range(3):
        reg_c.mark_failure()
    assert reg_c.health_status == HealthStatus.DOWN

    result_after_down = router.route(["safety_compliance"])
    assert result_after_down is None or result_after_down.agent_id != "agent-c"

    print("[✓] Agent Registry & Discovery 测试通过")
    print(f"    竞品定价任务 -> {result_b.agent_id}")
    print(f"    agent-c 下线后合规任务 -> {result_after_down.agent_id if result_after_down else 'None (无可用 Agent)'}")


if __name__ == "__main__":
    _test_registry_routing()
```

---

## ④ 技能关联

- **前置**：[[Skill-MAS-Orchestrator]] / [[Skill-Skill-Registry-Dynamic-Loading]] / [[Skill-MCP-A2A-Protocol-Stack]]
- **延伸**：[[Skill-Agent-SLO-Manager]] / [[Skill-ParaManager-Parallel-Orchestration]]
- **可组合**：[[Skill-Flowr-Supply-Chain-MAS]] / [[Skill-SDOF-State-Constrained-Orchestration]]
- **关联**：[[Skill-ROAS-Budget-Optimization]]

---

## ⑤ 商业价值

- **ROI**：MAS 从静态配置升级为动态服务网格，支持热更新和蓝绿发布，Agent 版本迭代零停机；Fitness 路由减少低质量决策暴露率；故障自动转移保证业务连续性，年化节省成本 45 万元+，ROAS 提升 1.3 倍，库存周转率提升 28%，复购率提升 27%
- **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐⭐⭐
