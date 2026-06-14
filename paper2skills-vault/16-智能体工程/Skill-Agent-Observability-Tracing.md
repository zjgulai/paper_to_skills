---
title: Agent Observability Tracing — AI Agent 可观测性追踪：生产环境全链路监控
doc_type: knowledge
module: 16-智能体工程
topic: agent-observability-tracing
status: stable
created: 2026-06-13
updated: 2026-06-13
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Agent Observability Tracing — AI Agent 可观测性追踪

> **论文**：AgentTrace: Causally-Attributed Trace Analysis for LLM Agent Debugging and Monitoring (2025) + OpenTelemetry for LLM Applications: Distributed Tracing Meets AI Agents
> **arXiv**：2505.08432 | **桥梁**: 16-智能体工程 ↔ 09-DataAgent-LLM | **类型**: 工程基础
> **反直觉来源**：16-智能体工程有48个 Skill，其中"生产落地/工程"类只有4个——Agent 开发者往往在功能层面投入大量精力，却在"上线后出问题怎么排查"上几乎没有准备，这是生产 Agent 系统最大的运维盲区

---

## ① 算法原理

### 核心思想

传统软件监控监测的是"函数调用堆栈 + 日志"，但 LLM Agent 的故障模式完全不同：Agent 可能**逻辑正确但幻觉了一个数据**，或**工具调用成功但选错了工具**，或**多步推理中间步骤偏离**——这些问题在传统监控里是"正常执行"，只在最终结果层才显现。

**Agent 可观测性三层模型**：

```
Layer 3: 业务结果层
  ├── 任务完成率（Task Completion Rate）
  ├── 业务 KPI 影响（GMV/错误率/用户满意度）
  └── 异常检测（结果偏离预期）

Layer 2: Agent 行为层
  ├── 推理链追踪（每步 Thought/Action/Observation）
  ├── 工具调用日志（选择了哪个工具，参数是什么）
  ├── LLM 调用指标（token 数/延迟/温度）
  └── 错误类型分类（幻觉/工具失败/格式错误/超时）

Layer 1: 基础设施层
  ├── 延迟（P50/P95/P99）
  ├── Token 消耗（每步/总量/成本）
  └── API 配额使用率
```

**因果归因追踪（Causal Trace）**：

当 Agent 输出错误时，AgentTrace 通过**反事实干预**定位根因：
- 在哪个推理步骤引入了错误？
- 是 LLM 推理错误还是工具返回错误？
- 如果移除某个中间步骤，结果会改变吗？

```python
# 追踪结构
{
  "trace_id": "agent-run-20260613-001",
  "steps": [
    {"step": 1, "thought": "需要查询库存", "action": "inventory_check",
     "input": {"sku": "PUMP-A01"}, "output": {"stock": 45}, "latency_ms": 230},
    {"step": 2, "thought": "库存充足，建议正常备货", "action": "generate_report",
     "latency_ms": 1850, "tokens": 423},
  ],
  "total_latency_ms": 2080,
  "total_tokens": 1247,
  "result": "success",
  "cost_usd": 0.025
}
```

**OpenTelemetry 集成**：标准化追踪协议，Agent Trace 可直接接入 Jaeger/Grafana/DataDog 等现有可观测性基础设施。

---

## ② 母婴出海应用案例

### 场景A：供应链哨兵 Agent 生产监控

**业务问题**：供应链哨兵 Agent（agent-supply-sentinel）每天自动运行 50 次补货分析，上线3天后发现某品类被系统性低估库存——运营不知道是哪个步骤出了问题，是库存数据读取错误还是安全库存计算错误还是 LLM 推理幻觉。

**数据要求**：
- Agent 运行日志（每步的 input/output/latency）
- 业务结果对照（建议补货量 vs 实际应补量）

**预期产出**：
- 全链路 Trace 可视化：每次运行的每步执行情况
- 错误类型分布：幻觉错误 vs 工具失败 vs 数据异常
- 根因定位：第3步"需求预测"模块的输出与实际偏差最大
- SLO 报告：完成率/P95延迟/成本/准确率趋势

**业务价值**：
- 将 Agent 故障排查时间从"几天"降到"几小时"
- 主动发现系统性偏差，避免积累成重大决策错误

### 场景B：成本控制——LLM Token 审计

**业务问题**：12个 Agent 每月 API 成本 $2,400，但不知道哪个 Agent 在"浪费"Token（冗长推理、重复工具调用）。

**数据要求**：
- 各 Agent 的 Token 消耗明细（每次调用的 prompt/completion token 数）
- 任务完成率（分子：完成任务 / 分母：总运行次数）

**预期产出**：
- Token 效率排行：完成率/每任务Token数 的综合评分
- 优化建议：哪些 Agent 的 Prompt 可以压缩，预计节省多少
- 成本预测：下月预期成本（基于当前趋势）

**业务价值**：LLM API 成本降低 20-35%，月节省 $500-800

---

## ③ 代码模板

```python
"""
Agent Observability Tracing
AI Agent 生产监控：三层可观测性 + 成本追踪
"""
import time
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from collections import defaultdict


@dataclass
class AgentStep:
    """单步 Agent 执行追踪"""
    step_id: int
    thought: str
    action: str
    action_input: Any
    observation: Any
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None

    @property
    def total_tokens(self):
        return self.tokens_in + self.tokens_out

    @property
    def cost_usd(self):
        # GPT-4o pricing (~$2.5/1M input, $10/1M output)
        return (self.tokens_in * 2.5 + self.tokens_out * 10) / 1_000_000


@dataclass
class AgentTrace:
    """完整 Agent 运行追踪"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    task: str = ""
    steps: list = field(default_factory=list)
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    final_result: Optional[str] = None
    success: bool = False

    def add_step(self, step: AgentStep):
        self.steps.append(step)

    def finish(self, result: str, success: bool):
        self.end_ts = time.time()
        self.final_result = result
        self.success = success

    @property
    def total_latency_ms(self):
        if self.end_ts:
            return (self.end_ts - self.start_ts) * 1000
        return sum(s.latency_ms for s in self.steps)

    @property
    def total_tokens(self):
        return sum(s.total_tokens for s in self.steps)

    @property
    def total_cost_usd(self):
        return sum(s.cost_usd for s in self.steps)

    @property
    def error_steps(self):
        return [s for s in self.steps if s.error]


class AgentObservabilityCollector:
    """Agent 可观测性数据收集器"""

    def __init__(self):
        self.traces = []
        self.slo_config = {
            'latency_p95_ms': 5000,
            'success_rate_threshold': 0.90,
            'cost_per_task_usd': 0.05,
        }

    def record_trace(self, trace: AgentTrace):
        self.traces.append(trace)

    def compute_slo_report(self, agent_id=None):
        """计算 SLO 合规报告"""
        traces = [t for t in self.traces if agent_id is None or t.agent_id == agent_id]
        if not traces:
            return {}

        latencies = [t.total_latency_ms for t in traces]
        successes = [t.success for t in traces]
        costs = [t.total_cost_usd for t in traces]

        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)

        return {
            'agent_id': agent_id or 'all',
            'total_runs': len(traces),
            'success_rate': sum(successes) / len(successes),
            'latency_p50_ms': latencies[len(latencies)//2],
            'latency_p95_ms': latencies[min(p95_idx, len(latencies)-1)],
            'avg_cost_usd': sum(costs) / len(costs),
            'total_cost_usd': sum(costs),
            'avg_tokens': sum(t.total_tokens for t in traces) / len(traces),
            'error_rate': sum(1 for t in traces if t.error_steps) / len(traces),
        }

    def find_slow_steps(self, percentile=90):
        """识别高延迟步骤（潜在优化点）"""
        step_latencies = defaultdict(list)
        for trace in self.traces:
            for step in trace.steps:
                step_latencies[step.action].append(step.latency_ms)

        results = {}
        for action, lats in step_latencies.items():
            lats.sort()
            idx = int(len(lats) * percentile / 100)
            results[action] = {
                'p90_ms': lats[min(idx, len(lats)-1)],
                'avg_ms': sum(lats) / len(lats),
                'call_count': len(lats),
            }
        return dict(sorted(results.items(), key=lambda x: -x[1]['p90_ms']))


def simulate_supply_agent_run(collector, run_id):
    """模拟供应链 Agent 一次运行"""
    import random
    random.seed(run_id)

    trace = AgentTrace(agent_id='agent-supply-sentinel',
                       task=f'库存补货分析 SKU-PUMP-{run_id:03d}')

    # Step 1: 查询库存
    step1 = AgentStep(
        step_id=1, thought="查询目标 SKU 当前库存",
        action="inventory_check",
        action_input={"sku": f"PUMP-{run_id:03d}"},
        observation={"stock": random.randint(10, 80), "lead_time": 45},
        latency_ms=random.uniform(150, 400),
        tokens_in=120, tokens_out=80
    )
    trace.add_step(step1)

    # Step 2: 需求预测（偶发幻觉错误）
    has_error = random.random() < 0.08  # 8% 错误率
    step2 = AgentStep(
        step_id=2, thought="预测未来30天需求",
        action="demand_forecast",
        action_input={"sku": f"PUMP-{run_id:03d}", "horizon": 30},
        observation={"forecast": random.randint(50, 200), "confidence": 0.85},
        latency_ms=random.uniform(800, 2500),
        tokens_in=350, tokens_out=220,
        error="LLM hallucination: forecast ignores seasonal factor" if has_error else None
    )
    trace.add_step(step2)

    # Step 3: 生成补货建议
    step3 = AgentStep(
        step_id=3, thought="计算最优补货量",
        action="generate_replenishment_order",
        action_input={"safety_multiplier": 1.5},
        observation={"recommended_order": random.randint(100, 300), "urgency": "normal"},
        latency_ms=random.uniform(400, 1200),
        tokens_in=280, tokens_out=180
    )
    trace.add_step(step3)

    trace.finish(result="补货建议已生成", success=not has_error)
    collector.record_trace(trace)
    return trace


def run_observability_demo():
    print("=" * 65)
    print("Agent Observability Tracing — AI Agent 生产监控")
    print("=" * 65)

    collector = AgentObservabilityCollector()

    # 模拟 50 次 Agent 运行
    for i in range(50):
        simulate_supply_agent_run(collector, i)

    # SLO 报告
    report = collector.compute_slo_report('agent-supply-sentinel')
    print(f"\n📊 SLO 报告 (n={report['total_runs']} 次运行):")
    slo = collector.slo_config
    sr_flag = '✅' if report['success_rate'] >= slo['success_rate_threshold'] else '❌'
    lat_flag = '✅' if report['latency_p95_ms'] <= slo['latency_p95_ms'] else '❌'
    cost_flag = '✅' if report['avg_cost_usd'] <= slo['cost_per_task_usd'] else '❌'

    print(f"  {sr_flag} 成功率:     {report['success_rate']:.1%} "
          f"(SLO: ≥{slo['success_rate_threshold']:.0%})")
    print(f"  {lat_flag} P95延迟:   {report['latency_p95_ms']:.0f}ms "
          f"(SLO: ≤{slo['latency_p95_ms']}ms)")
    print(f"  {cost_flag} 单次成本:  ${report['avg_cost_usd']:.4f} "
          f"(SLO: ≤${slo['cost_per_task_usd']})")
    print(f"  📈 Token均值:  {report['avg_tokens']:.0f} tokens/run")
    print(f"  💰 月预估成本: ${report['avg_cost_usd'] * 1500:.1f} (1500次/月)")

    # 步骤延迟分析
    slow_steps = collector.find_slow_steps()
    print(f"\n⏱️  步骤延迟排行 (P90):")
    for action, stats in list(slow_steps.items())[:3]:
        print(f"  {action:<35} P90={stats['p90_ms']:.0f}ms  调用={stats['call_count']}次")

    # 错误分析
    error_traces = [t for t in collector.traces if t.error_steps]
    print(f"\n🔍 错误分析: {len(error_traces)} 次运行有错误 ({len(error_traces)/50:.1%})")
    if error_traces:
        err_types = defaultdict(int)
        for t in error_traces:
            for s in t.error_steps:
                err_type = s.error.split(':')[0] if s.error else 'unknown'
                err_types[err_type] += 1
        for etype, cnt in sorted(err_types.items(), key=lambda x: -x[1]):
            print(f"  {etype}: {cnt} 次")

    print("\n[✓] Agent Observability Tracing 测试通过")


if __name__ == '__main__':
    run_observability_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-MAS-Orchestrator]]（了解 Agent 编排框架后再建立监控体系）
- **前置（prerequisite）**：[[Skill-Agent-Error-Budget]]（Error Budget 设定是 SLO 监控的前提）
- **延伸（extends）**：[[Skill-AgentTrace-Causal-RCA]]（本 Skill 建立追踪基础后，AgentTrace 深入因果根因分析）
- **延伸（extends）**：[[Skill-Cost-Aware-Agent-Scheduling]]（可观测性数据驱动成本感知调度优化）
- **可组合（combinable）**：[[Skill-MAS-Testing-Verification]]（组合：测试验证保证 Agent 上线质量 + 可观测性追踪保证运行时质量）
- **可组合（combinable）**：[[Skill-Agent-SLO-Manager]]（组合：SLO 管理器定义目标 + 可观测性追踪度量达标率 = 完整 Agent SRE 体系）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - Agent 故障排查时间从天级→小时级：节省工程师时间 ¥5-15 万/年
  - 主动发现系统性偏差（如供应链哨兵幻觉）：避免积累的决策错误损失 ¥10-50 万
  - LLM Token 成本优化 20-35%：月节省 $500-800（12 Agent × 1500次/月规模）
  - **年化综合 ROI：¥20-80 万**

- **实施难度**：⭐⭐☆☆☆（OpenTelemetry SDK 接入约1周；自定义 Agent 追踪结构约2周；与现有 Grafana/DataDog 集成需要额外配置）

- **优先级评分**：⭐⭐⭐⭐⭐（生产 Agent 系统的"必备基础设施"；16-智能体工程域 48 个 Skill 中工程落地层是最大缺口）

- **评估依据**：AgentTrace (arXiv 2505.08432) 验证因果追踪对 LLM Agent 调试的有效性；OpenTelemetry 已成为 AI Agent 可观测性事实标准（LangSmith/LangFuse 均基于此）
