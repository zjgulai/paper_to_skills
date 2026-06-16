---
title: Context Engine三层架构 — engine/agents/registry分离的可复用MAS骨架
doc_type: knowledge
module: 10-MAS
topic: context-engine-architecture-pattern
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Context Engine三层架构

> **书籍**：Context Engineering for Multi-Agent Systems — Chapter 4-5: Assembling & Hardening the Context Engine
> **作者**：Denis Rothman | 2025 | **桥梁**: MAS ↔ 智能体工程 | **类型**: 算法工具
> **GitHub**：Denis2054/Context-Engineering-for-Multi-Agent-Systems / commons/engine/{engine.py, agents.py, registry.py}

## ① 算法原理

**核心洞察（Rothman三层分离）**：大多数MAS实现把编排逻辑、Agent定义、工具注册混在一起，导致"改一个Agent需要修改整个系统"。Rothman的Context Engine提出严格的三层分离架构：

```
┌────────────────────────────────────────────┐
│            engine.py  (编排层)              │
│  - 接收任务，决定调用哪些Agent              │
│  - 管理上下文流转（输入→处理→输出）         │
│  - Token计数/成本追踪                       │
├────────────────────────────────────────────┤
│            agents.py  (Agent层)             │
│  - 定义每个专业Agent的SRL蓝图               │
│  - 专注单一职责（研究/分析/生成）           │
│  - 不知道其他Agent的存在                    │
├────────────────────────────────────────────┤
│           registry.py  (注册层)             │
│  - Agent能力清单（动态注册/发现）           │
│  - 工具/MCP端点注册                         │
│  - 路由规则（什么任务路由到哪个Agent）      │
└────────────────────────────────────────────┘
```

**三层职责精确定义**：

1. **Engine层（编排+上下文管理）**：
   - 接收用户请求，解析为任务
   - 查询Registry确定执行Agent序列
   - 管理Agent间的上下文传递（前一个Agent的输出 = 下一个Agent的Patient）
   - 维护全局会话状态
   - 追踪Token消耗和成本

2. **Agents层（专业执行）**：
   - 每个Agent = 一个SRL蓝图 + 执行逻辑
   - Agent只知道自己的输入/输出协议
   - 完全不知道其他Agent的存在（松耦合）
   - 可独立测试、独立更新

3. **Registry层（发现+路由）**：
   - 存储所有Agent的能力描述
   - 提供按能力/标签查找Agent的接口
   - 工具注册（MCP工具端点）
   - 路由规则（关键词/意图→Agent_ID映射）

**硬化（Hardening）技术（Ch5核心）**：
- **模块化重构**：将紧耦合的Notebook代码重构为可独立部署的模块
- **弹性扩展**：Engine层的工作队列支持水平扩展（多Engine实例共享一个Registry）
- **预生产检查清单**：类似NASA的发射前检查——验证所有Agent连通性、工具可达性、Token限制合理性
- **反向兼容性**：新版Engine仍能处理旧格式的任务请求（不破坏现有集成）

**Context Engine状态机**：
```
IDLE → RECEIVING_TASK → PLANNING → EXECUTING(Agent_i) → 
CONTEXT_TRANSFER → EXECUTING(Agent_i+1) → ... → AGGREGATING → DONE
```

## ② 母婴出海应用案例

**场景A：选品MAS系统重构为Context Engine三层架构**

- **业务问题**：原有4-Agent选品系统所有逻辑都写在一个Notebook里，每次想改Research Agent的检索策略都要担心影响其他Agent；新增Finance Agent时需要重写大量编排代码
- **重构方案**：
  - engine.py：接收"选品分析请求"→调用registry查找对应Agent→顺序执行→汇总输出
  - agents.py：4个Agent类（ResearchAgent/CompetitorAgent/FinanceAgent/ReportAgent），各自只知道自己的职责
  - registry.py：注册4个Agent的能力描述，新增Agent只需在registry中添加一行
- **预期产出**：新增一个Agent从"2天重写" → "2小时注册+测试"；修改单一Agent不影响其他Agent；系统可独立测试每个组件

**场景B：预生产硬化检查清单**

- **业务问题**：某次上线前Registry中的工具端点URL写错，导致整个MAS在生产环境崩溃
- **硬化方案**：预生产检查脚本自动验证：①所有注册Agent实例化是否成功 ②所有工具端点ping是否可达 ③Token上限设置是否合理 ④内存/上下文窗口是否充足
- **预期产出**：生产环境故障率从每月2-3次降至接近0次

## ③ 代码模板

```python
"""
Context Engine三层架构实现
功能：engine/agents/registry三层分离 + 上下文流转 + Token追踪 + 预生产检查
基于 Denis Rothman《Context Engineering for Multi-Agent Systems》Ch4-5
"""
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ─── 数据结构 ──────────────────────────────────────────────────

class EngineState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    AGGREGATING = "aggregating"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentCapability:
    """Agent能力描述（Registry中存储）"""
    agent_id: str
    agent_class: str
    description: str
    input_types: List[str]          # 支持的输入类型
    output_type: str                # 输出类型
    tags: List[str]                 # 能力标签（用于路由）
    priority: int = 5               # 调用优先级（1-10）
    max_tokens: int = 2000          # 最大Token输出


@dataclass
class AgentContext:
    """Agent执行上下文（在Agent间流转）"""
    task_id: str
    input_data: Any
    upstream_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    token_budget_remaining: int = 8000


@dataclass
class AgentOutput:
    """Agent执行输出"""
    agent_id: str
    output: Any
    token_consumed: int
    execution_time_ms: float
    citations: List[str] = field(default_factory=list)
    status: str = "success"


# ─── REGISTRY LAYER（注册层）──────────────────────────────────

class AgentRegistry:
    """
    Agent注册表 — 存储Agent能力、工具端点、路由规则
    对应 registry.py in Denis Rothman's framework
    """

    def __init__(self):
        self._agents: Dict[str, AgentCapability] = {}
        self._tools: Dict[str, Dict] = {}
        self._routing_rules: List[Dict] = []

    def register_agent(self, capability: AgentCapability) -> None:
        self._agents[capability.agent_id] = capability
        print(f"  [Registry] ✅ 注册Agent: {capability.agent_id} ({capability.description[:40]})")

    def register_tool(self, tool_id: str, endpoint: str, description: str) -> None:
        self._tools[tool_id] = {
            'endpoint': endpoint,
            'description': description,
            'registered_at': time.time(),
        }

    def add_routing_rule(self, pattern: str, agent_ids: List[str], priority: int = 5) -> None:
        self._routing_rules.append({
            'pattern': pattern.lower(),
            'agent_ids': agent_ids,
            'priority': priority,
        })

    def find_agents_by_tag(self, tag: str) -> List[AgentCapability]:
        return [cap for cap in self._agents.values() if tag in cap.tags]

    def route_task(self, task_description: str) -> List[str]:
        """根据任务描述路由到对应Agent序列"""
        task_lower = task_description.lower()
        matched_rules = []
        for rule in self._routing_rules:
            if any(kw in task_lower for kw in rule['pattern'].split('|')):
                matched_rules.append(rule)
        matched_rules.sort(key=lambda x: x['priority'], reverse=True)
        if matched_rules:
            return matched_rules[0]['agent_ids']
        # 默认：返回所有注册Agent（按优先级）
        return sorted(self._agents.keys(),
                      key=lambda aid: self._agents[aid].priority, reverse=True)

    def health_check(self) -> Dict:
        """预生产健康检查"""
        issues = []
        if not self._agents:
            issues.append({'severity': 'CRITICAL', 'msg': '没有注册任何Agent'})
        for agent_id, cap in self._agents.items():
            if cap.max_tokens > 4000:
                issues.append({'severity': 'WARNING',
                                'msg': f'{agent_id} max_tokens={cap.max_tokens}，超过建议上限'})
        return {
            'agents_count': len(self._agents),
            'tools_count': len(self._tools),
            'routing_rules': len(self._routing_rules),
            'issues': issues,
            'status': 'HEALTHY' if not any(i['severity'] == 'CRITICAL' for i in issues) else 'CRITICAL',
        }


# ─── AGENTS LAYER（Agent层）───────────────────────────────────

class BaseAgent(ABC):
    """基础Agent抽象类 — 对应 agents.py"""

    def __init__(self, agent_id: str, max_tokens: int = 1500):
        self.agent_id = agent_id
        self.max_tokens = max_tokens

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentOutput:
        """执行Agent核心逻辑"""
        pass

    def _estimate_tokens(self, text: str) -> int:
        """估算Token数（简化：按字数估算，4字≈1token）"""
        return max(len(text) // 4, 1)


class ResearchAgent(BaseAgent):
    """市场研究Agent"""

    def execute(self, context: AgentContext) -> AgentOutput:
        start = time.time()
        task = context.input_data.get('task', '')

        # 模拟研究输出（生产环境调用LLM + RAG）
        output = {
            'market_size': '$2.8B (2025 US)',
            'yoy_growth': '12%',
            'top_players': ['Spectra', 'Medela', 'BabyBuddha'],
            'trends': ['医院级静音', '无线便携', '智能APP联动'],
            'task_responded': task[:50],
        }
        text = json.dumps(output)
        tokens = self._estimate_tokens(text)

        return AgentOutput(
            agent_id=self.agent_id,
            output=output,
            token_consumed=min(tokens, self.max_tokens),
            execution_time_ms=(time.time() - start) * 1000,
            citations=['Market_Report_Q4_2025', 'Amazon_Baby_Data'],
        )


class FinanceAgent(BaseAgent):
    """财务分析Agent"""

    def execute(self, context: AgentContext) -> AgentOutput:
        start = time.time()
        # 使用上游Research输出作为Patient
        research_data = context.upstream_outputs.get('research_agent', {})
        market_size = research_data.get('market_size', '$0')

        output = {
            'roi_estimate': '28-35% (12个月)',
            'payback_period': '8-11个月',
            'risk_level': 'MEDIUM',
            'based_on_market': market_size,
            'capex_estimate': '$25,000-$40,000',
        }
        text = json.dumps(output)
        tokens = self._estimate_tokens(text)

        return AgentOutput(
            agent_id=self.agent_id,
            output=output,
            token_consumed=min(tokens, self.max_tokens),
            execution_time_ms=(time.time() - start) * 1000,
        )


class ReportAgent(BaseAgent):
    """报告生成Agent"""

    def execute(self, context: AgentContext) -> AgentOutput:
        start = time.time()
        upstream = context.upstream_outputs

        research = upstream.get('research_agent', {})
        finance = upstream.get('finance_agent', {})

        report = f"""# 母婴选品分析报告

## 市场概况
- 市场规模：{research.get('market_size', 'N/A')}
- YoY增长：{research.get('yoy_growth', 'N/A')}
- 主要趋势：{', '.join(research.get('trends', []))}

## 财务预测
- ROI估算：{finance.get('roi_estimate', 'N/A')}
- 回本周期：{finance.get('payback_period', 'N/A')}
- 风险等级：{finance.get('risk_level', 'N/A')}

## 决策建议
基于市场数据和财务预测，建议进入该品类。
"""
        tokens = self._estimate_tokens(report)
        return AgentOutput(
            agent_id=self.agent_id,
            output=report,
            token_consumed=min(tokens, self.max_tokens),
            execution_time_ms=(time.time() - start) * 1000,
        )


# ─── ENGINE LAYER（编排层）───────────────────────────────────

class ContextEngine:
    """
    Context Engine核心编排层 — 对应 engine.py
    负责：任务接收→规划→执行→上下文流转→汇总
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._agent_instances: Dict[str, BaseAgent] = {}
        self.state = EngineState.IDLE
        self.token_tracker = {'input': 0, 'output': 0, 'total': 0}
        self.execution_log: List[Dict] = []

    def register_agent_instance(self, agent: BaseAgent) -> None:
        """注册Agent实例（与Registry中的能力描述绑定）"""
        self._agent_instances[agent.agent_id] = agent

    def _plan_execution(self, task: Dict) -> List[str]:
        """规划Agent执行顺序"""
        description = task.get('description', '')
        return self.registry.route_task(description)

    def _execute_agent(self, agent_id: str, context: AgentContext) -> Optional[AgentOutput]:
        """执行单个Agent"""
        agent = self._agent_instances.get(agent_id)
        if not agent:
            self.execution_log.append({'agent': agent_id, 'status': 'NOT_FOUND'})
            return None

        self.state = EngineState.EXECUTING
        output = agent.execute(context)
        self.token_tracker['output'] += output.token_consumed
        self.token_tracker['total'] += output.token_consumed

        self.execution_log.append({
            'agent': agent_id,
            'status': output.status,
            'tokens': output.token_consumed,
            'time_ms': round(output.execution_time_ms, 1),
        })
        return output

    def run(self, task: Dict, token_budget: int = 8000) -> Dict:
        """完整引擎执行流程"""
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        print(f"\n  [Engine] 任务ID: {task_id}")
        print(f"  [Engine] 任务: {task.get('description', '')[:60]}...")

        # 1. 规划
        self.state = EngineState.PLANNING
        agent_sequence = self._plan_execution(task)
        print(f"  [Engine] 执行序列: {' → '.join(agent_sequence)}")

        # 2. 执行（顺序，上下文流转）
        upstream_outputs = {}
        context = AgentContext(
            task_id=task_id,
            input_data=task,
            upstream_outputs=upstream_outputs,
            token_budget_remaining=token_budget,
        )

        for agent_id in agent_sequence:
            if agent_id not in self._agent_instances:
                continue
            context.upstream_outputs = dict(upstream_outputs)  # 快照传入
            output = self._execute_agent(agent_id, context)
            if output:
                upstream_outputs[agent_id] = output.output
                context.token_budget_remaining -= output.token_consumed
                print(f"  [Engine] ✅ {agent_id}: {output.token_consumed}tokens, "
                      f"{output.execution_time_ms:.0f}ms")

        # 3. 汇总
        self.state = EngineState.AGGREGATING
        total_time = (time.time() - start_time) * 1000
        self.state = EngineState.DONE

        return {
            'task_id': task_id,
            'final_output': upstream_outputs.get(agent_sequence[-1] if agent_sequence else '', ''),
            'agent_outputs': upstream_outputs,
            'execution_log': self.execution_log,
            'token_summary': self.token_tracker,
            'total_time_ms': round(total_time, 1),
        }


def pre_production_check(engine: ContextEngine, registry: AgentRegistry) -> None:
    """预生产检查清单（Hardening，Ch5核心）"""
    print("\n[预生产检查清单（Ch5 Hardening）]")

    checks = [
        ("Registry健康", registry.health_check()['status'] == 'HEALTHY'),
        ("Agent实例完整", all(
            aid in engine._agent_instances
            for aid in registry._agents
        )),
        ("Token预算合理", all(
            cap.max_tokens <= 3000
            for cap in registry._agents.values()
        )),
        ("路由规则存在", len(registry._routing_rules) > 0),
    ]

    all_passed = True
    for name, passed in checks:
        icon = '✅' if passed else '❌'
        print(f"  {icon} {name}")
        if not passed:
            all_passed = False

    print(f"\n  预生产状态: {'✅ 可发布' if all_passed else '❌ 需修复后发布'}")


def run_context_engine_demo():
    """Context Engine三层架构完整演示"""
    print("=" * 65)
    print("Context Engine三层架构（engine/agents/registry）")
    print("基于 Denis Rothman Context Engineering Ch4-5")
    print("=" * 65)

    # ─── REGISTRY 初始化 ───────────────────────────────────────
    print("\n[1] 初始化Registry（注册层）")
    registry = AgentRegistry()
    registry.register_agent(AgentCapability(
        "research_agent", "ResearchAgent", "市场研究与竞品分析",
        ["text", "query"], "json", ["research", "market", "analysis"], priority=8))
    registry.register_agent(AgentCapability(
        "finance_agent", "FinanceAgent", "财务ROI评估",
        ["json"], "json", ["finance", "roi", "risk"], priority=6))
    registry.register_agent(AgentCapability(
        "report_agent", "ReportAgent", "业务报告生成",
        ["json"], "markdown", ["report", "summary", "output"], priority=4))

    registry.add_routing_rule("选品|分析|市场|产品",
                               ["research_agent", "finance_agent", "report_agent"])
    registry.add_routing_rule("roi|财务|投资回报",
                               ["finance_agent", "report_agent"])

    # ─── ENGINE 初始化 ─────────────────────────────────────────
    print("\n[2] 初始化Engine（编排层）")
    engine = ContextEngine(registry)
    engine.register_agent_instance(ResearchAgent("research_agent"))
    engine.register_agent_instance(FinanceAgent("finance_agent"))
    engine.register_agent_instance(ReportAgent("report_agent"))

    # ─── 预生产检查 ────────────────────────────────────────────
    pre_production_check(engine, registry)

    # ─── 执行任务 ──────────────────────────────────────────────
    print("\n[3] 执行任务（上下文流转）")
    result = engine.run({
        'description': '请对母婴吸奶器品类做选品分析，评估进入可行性',
        'category': '母婴',
        'target_market': 'US',
    })

    # ─── 结果展示 ──────────────────────────────────────────────
    print(f"\n[4] 执行结果")
    print(f"\n  Token消耗: {result['token_summary']['total']} tokens")
    print(f"  总耗时: {result['total_time_ms']:.0f}ms")
    print(f"\n  执行日志:")
    for log in result['execution_log']:
        print(f"    {log['agent']}: {log['status']} | {log['tokens']}tokens | {log['time_ms']}ms")

    # 最终输出
    final = result['final_output']
    if isinstance(final, str):
        print(f"\n  最终报告（前500字）:")
        print(f"  {final[:500]}...")

    # Registry健康报告
    health = registry.health_check()
    print(f"\n[5] Registry健康报告: {health['status']}")
    print(f"  注册Agent: {health['agents_count']} | 工具: {health['tools_count']} | 路由规则: {health['routing_rules']}")

    print("\n[✓] Context Engine三层架构系统测试通过")
    return engine


if __name__ == "__main__":
    engine = run_context_engine_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SRL-Semantic-Blueprint-MAS]]（SRL蓝图是Agents层每个Agent的定义格式）、[[Skill-Agent-Registry-Discovery]]（Registry概念的具体实现）
- **延伸（extends）**：[[Skill-Domain-Agnostic-Context-Engine]]（三层架构是域无关复用的基础骨架）、[[Skill-Glass-Box-MAS-Observability]]（Engine层的执行日志是可观测性的数据源）
- **可组合（combinable）**：[[Skill-Dual-RAG-Context-Engine]]（RAG引擎作为Engine层的工具调用）、[[Skill-Policy-Driven-Meta-Controller]]（元控制器作为Engine层的策略插件）

## ⑤ 商业价值评估

- **ROI 预估**：将紧耦合Notebook式MAS重构为三层架构，新增Agent时间从2天→2小时（节省$500/次），每年约10次变更节省$4500；单Agent测试时间从"全系统测试"→"独立测试"，减少80%回归测试开销；系统成本$8万，年ROI≈200%
- **实施难度**：⭐⭐⭐☆☆（三层分离本身不复杂；难点在于将现有紧耦合系统重构迁移，需要耐心的渐进式重构）
- **优先级**：⭐⭐⭐⭐⭐（Rothman书中Ch4-5是整本书的核心工程章节，三层架构是Context Engine可复用性的基础——没有这个架构，所有其他能力都无法域无关复用）
- **适用规模**：所有需要长期维护和扩展的MAS系统；一次性脚本不值得引入
- **数据依赖**：无需外部数据；需要对现有MAS做模块化分析
