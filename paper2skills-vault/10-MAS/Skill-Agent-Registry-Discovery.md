---
title: Agent Registry & Discovery — 动态 Agent 能力注册与路由
doc_type: knowledge
module: 10-MAS
topic: agent-registry-service-discovery
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Agent Registry & Discovery（动态注册与路由）

> **领域**: 10-MAS | **类型**: 综合萃取

---

## ① 算法原理

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

**场景一：WF-A 补货 Agent 灰度升级**

新版补货 Agent v2（支持多货币汇率预测）上线时自动向 Registry 注册，声明新能力 `["replenishment", "fx_prediction"]`。Registry 广播变更通知，Orchestrator 灰度路由 10% 流量至 v2，每 5 分钟检查 SLO：v2 连续 3 次 fitness > v1 且 HEALTHY，自动升权重至 100%，v1 优雅下线（deregister）。

**场景二：WF-D 选品 Agent 池（多能力调度）**

3 个并行选品 Agent 注册不同专长：Agent-A（分类分析+趋势评分）、Agent-B（竞品定价+市场份额）、Agent-C（安全合规+法规检查）。Orchestrator 接到"分析婴儿推车竞争格局"任务，CapabilityMatcher 识别需要 `competitor_pricing + market_share` → 路由至 Agent-B。Agent-C 健康检查超时标记 DOWN，合规任务自动降级路由至 Agent-A（partial match），同时触发运维告警。

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

---
- **关联**：[[Skill-ROAS-Budget-Optimization]]

## ⑤ 商业价值

- **ROI**：MAS 从静态配置升级为动态服务网格，支持热更新和蓝绿发布，Agent 版本迭代零停机；Fitness 路由减少低质量决策暴露率
- **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐⭐⭐
