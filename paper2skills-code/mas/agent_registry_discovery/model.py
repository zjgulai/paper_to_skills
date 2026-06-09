"""
Agent Registry & Discovery -- 动态注册与路由
来源：Agent Registry 2025-2026 + MCP/A2A 协议扩展
"""
import time
import hashlib
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
        """计算能力匹配得分（Jaccard 相似度）"""
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
    """Agent 服务注册中心"""

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
        """定期健康检查，返回标记为 DOWN 的 agent_id 列表"""
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
    """基于能力声明匹配最合适的 Agent"""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def find_candidates(
        self,
        required_skills: List[str],
        domain: Optional[str] = None,
        min_score: float = 0.1,
    ) -> List[tuple]:
        """返回 (score, registration) 列表，按得分降序排列"""
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
    """负载均衡路由器：支持 Fitness优先 / 延迟优先 / RoundRobin"""

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
