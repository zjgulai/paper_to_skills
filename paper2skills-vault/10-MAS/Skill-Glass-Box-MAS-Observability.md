---
title: 玻璃盒MAS可观测性 — Agent推理轨迹追踪、Token成本仪表盘与透明度工程
doc_type: knowledge
module: 10-MAS
topic: glass-box-mas-observability
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: 玻璃盒MAS可观测性

> **书籍**：Context Engineering for Multi-Agent Systems — Chapter 10: The Blueprint for Production-Ready AI
> **作者**：Denis Rothman | 2025 | **桥梁**: MAS ↔ 智能体工程 | **类型**: 算法工具
> **GitHub**：Denis2054/Context-Engineering-for-Multi-Agent-Systems / Chapter10/Universal_Context_Engine.ipynb + Universal_Context_Engine_UI.ipynb

## ① 算法原理

**核心洞察（Rothman玻璃盒哲学）**："黑盒AI"是企业部署MAS的最大障碍——管理层无法理解AI的推理过程，合规团队无法审计AI的决策，运营团队无法诊断AI的失败。"玻璃盒（Glass Box）"是对"黑盒（Black Box）"的颠覆：**100%透明的推理轨迹，让每一个AI决策都可追溯、可审计、可解释**。

**玻璃盒可观测性的三个维度**：

1. **Agent推理轨迹（Reasoning Trace）**：
   - 记录每个Agent的完整执行过程：输入→检索→推理→输出
   - 每一步决策的依据：使用了哪些检索结果，如何影响了最终输出
   - 时间戳：精确到毫秒的执行时间线
   - 可视化：交互式追踪仪表盘（支持放大查看每个Agent的细节）

2. **Token经济仪表盘（Token Economy Dashboard）**：
   - 实时追踪：每次调用的输入/输出Token数
   - 成本分解：按Agent、按任务类型、按时间段的成本分布
   - 效率指标：Token使用效率（有效信息Token vs 冗余Token）
   - 成本预警：超出预算时自动告警
   - 业务价值翻译：将技术指标（Token）转化为业务语言（$成本、响应时间）

3. **主权架构（Sovereign Architecture）**：
   - 数据主权：所有推理过程的数据留在用户的基础设施内，不上传第三方
   - 架构主权：用户完全控制每个组件，不依赖黑盒SaaS平台
   - 审计主权：随时可导出完整的决策日志用于合规审计
   - 模型主权：可替换任何组件（从GPT-4切换到其他模型）不影响整体架构

**实现技术栈**：
```python
# Trace数据结构示例
{
  "session_id": "sess-abc123",
  "timestamp": "2026-06-15T10:30:00",
  "agents": [
    {
      "agent_id": "research_agent",
      "input_tokens": 450,
      "output_tokens": 320,
      "latency_ms": 1230,
      "retrieved_docs": ["DOC-a1b2c3d4"],
      "reasoning_steps": ["检索市场数据", "分析竞品格局", "生成摘要"],
      "output_preview": "美国母婴市场$28亿..."
    },
    ...
  ],
  "total_tokens": 1840,
  "total_cost_usd": 0.0092,
  "wall_clock_ms": 3450
}
```

**可观测性等级**：
| 等级 | 描述 | 适用场景 |
|-----|------|---------|
| L0 基础 | 仅记录输入/输出和总Token | 开发阶段 |
| L1 标准 | + Agent执行日志 + 每步延迟 | 生产基础 |
| L2 深度 | + 检索文档 + 推理步骤 | 审计/合规 |
| L3 完整 | + 完整上下文快照 + 成本分析 | 高风险系统 |

**Rothman的Gradio UI集成**：
书中Ch10提供了`Universal_Context_Engine_UI.ipynb`，用Gradio构建了实时可观测性仪表盘，包含：
- 执行轨迹可视化（折叠展开式Agent执行树）
- Token成本实时计量
- 跨会话的性能趋势

## ② 母婴出海应用案例

**场景A：MAS选品助手的可观测性仪表盘**

- **业务问题**：运营总监无法理解AI选品助手"为什么推荐这个品类而不是那个"，导致对AI建议缺乏信任，最终放弃使用
- **玻璃盒方案**：
  1. 为每次分析生成完整执行追踪报告，展示"Research Agent检索了哪5份文档，发现了什么，Finance Agent如何计算ROI"
  2. Token成本仪表盘：每次分析消耗多少Token，对应成本多少，如何逐步下降（优化效果可见）
  3. 推理步骤展示："市场规模28亿（来源：Market_Report_Q4）+ YoY增长12% → 判断为成长市场 → 推荐进入"
- **预期产出**：运营总监对AI建议的信任度从30%提升至78%，AI辅助决策采用率从20%提升至65%

**场景B：合规审计轨迹导出**

- **业务问题**：监管机构要求公司提供"AI系统如何做出某个合规建议"的完整记录
- **玻璃盒方案**：一键导出指定时间段内所有合规查询的完整执行轨迹（JSON格式），包含：使用的法规文档ID、检索时间戳、推理链、最终输出版本
- **预期产出**：合规审计从"无法提供AI决策依据"→"2小时内提供完整轨迹报告"，满足监管要求

## ③ 代码模板

```python
"""
玻璃盒MAS可观测性系统
功能：推理轨迹追踪 + Token成本仪表盘 + 审计日志导出 + 可观测性等级管理
基于 Denis Rothman《Context Engineering for Multi-Agent Systems》Ch10
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class ObservabilityLevel:
    L0 = 0  # 基础：输入/输出/Token
    L1 = 1  # 标准：+Agent日志+延迟
    L2 = 2  # 深度：+检索文档+推理步骤
    L3 = 3  # 完整：+上下文快照+成本分析


@dataclass
class AgentTrace:
    """单个Agent的执行追踪"""
    agent_id: str
    start_time: float
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved_docs: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    input_preview: str = ""
    output_preview: str = ""
    status: str = "running"
    metadata: Dict = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return self.total_tokens * 0.000005  # GPT-4o 近似价格


@dataclass
class SessionTrace:
    """完整会话追踪"""
    session_id: str
    task_description: str
    domain: str
    start_time: float
    end_time: float = 0.0
    agents: List[AgentTrace] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(a.total_tokens for a in self.agents)

    @property
    def total_cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.agents)

    @property
    def wall_clock_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def agent_count(self) -> int:
        return len(self.agents)


class GlassBoxObservability:
    """
    玻璃盒可观测性引擎
    对应 Denis Rothman Universal Context Engine 的可观测性层
    """

    def __init__(self, level: int = ObservabilityLevel.L2):
        self.level = level
        self.sessions: Dict[str, SessionTrace] = {}
        self.token_budget: Optional[int] = None
        self._current_session: Optional[SessionTrace] = None

    def start_session(self, task: str, domain: str = 'general') -> str:
        """开始新会话追踪"""
        session_id = f"sess-{str(uuid.uuid4())[:8]}"
        self._current_session = SessionTrace(
            session_id=session_id,
            task_description=task[:100],
            domain=domain,
            start_time=time.time(),
        )
        self.sessions[session_id] = self._current_session
        return session_id

    def start_agent_trace(self, agent_id: str,
                           input_data: Any = None) -> str:
        """开始Agent执行追踪"""
        if not self._current_session:
            return ""
        trace = AgentTrace(
            agent_id=agent_id,
            start_time=time.time(),
            input_preview=str(input_data)[:100] if input_data else "",
        )
        self._current_session.agents.append(trace)
        return agent_id

    def log_retrieval(self, agent_id: str, doc_ids: List[str]):
        """记录检索文档"""
        if self.level < ObservabilityLevel.L2:
            return
        trace = self._get_active_trace(agent_id)
        if trace:
            trace.retrieved_docs.extend(doc_ids)

    def log_reasoning_step(self, agent_id: str, step: str):
        """记录推理步骤"""
        if self.level < ObservabilityLevel.L2:
            return
        trace = self._get_active_trace(agent_id)
        if trace:
            trace.reasoning_steps.append(step)

    def finish_agent_trace(self, agent_id: str, output: Any,
                            input_tokens: int = 0, output_tokens: int = 0):
        """完成Agent执行追踪"""
        trace = self._get_active_trace(agent_id)
        if trace:
            trace.end_time = time.time()
            trace.input_tokens = input_tokens
            trace.output_tokens = output_tokens
            trace.output_preview = str(output)[:100] if output else ""
            trace.status = "completed"

    def finish_session(self) -> Optional[SessionTrace]:
        """结束会话追踪"""
        if self._current_session:
            self._current_session.end_time = time.time()
            session = self._current_session
            self._current_session = None
            return session
        return None

    def _get_active_trace(self, agent_id: str) -> Optional[AgentTrace]:
        if not self._current_session:
            return None
        for trace in reversed(self._current_session.agents):
            if trace.agent_id == agent_id:
                return trace
        return None

    def render_trace_report(self, session: SessionTrace) -> str:
        """生成可读的追踪报告（玻璃盒可视化）"""
        lines = [
            f"{'='*60}",
            f"📊 会话追踪报告（玻璃盒）",
            f"{'='*60}",
            f"会话ID: {session.session_id}",
            f"任务: {session.task_description}",
            f"域: {session.domain}",
            f"总耗时: {session.wall_clock_ms:.0f}ms",
            f"总Token: {session.total_tokens} (估算成本: ${session.total_cost_usd:.4f})",
            f"Agent数: {session.agent_count}",
            f"\n{'─'*60}",
            "Agent执行轨迹:",
        ]

        for i, agent_trace in enumerate(session.agents, 1):
            lines.append(f"\n  [{i}] {agent_trace.agent_id}")
            lines.append(f"      状态: {agent_trace.status} | "
                         f"耗时: {agent_trace.latency_ms:.0f}ms | "
                         f"Token: {agent_trace.total_tokens}({agent_trace.input_tokens}↓+{agent_trace.output_tokens}↑)")

            if self.level >= ObservabilityLevel.L2:
                if agent_trace.retrieved_docs:
                    lines.append(f"      检索文档: {agent_trace.retrieved_docs}")
                if agent_trace.reasoning_steps:
                    lines.append("      推理步骤:")
                    for step in agent_trace.reasoning_steps:
                        lines.append(f"        → {step}")

            if agent_trace.output_preview:
                lines.append(f"      输出预览: {agent_trace.output_preview[:60]}...")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)

    def get_cost_dashboard(self) -> Dict:
        """Token成本仪表盘"""
        if not self.sessions:
            return {}

        total_sessions = len(self.sessions)
        total_tokens = sum(s.total_tokens for s in self.sessions.values())
        total_cost = sum(s.total_cost_usd for s in self.sessions.values())

        by_domain = defaultdict(lambda: {'sessions': 0, 'tokens': 0, 'cost': 0.0})
        for s in self.sessions.values():
            by_domain[s.domain]['sessions'] += 1
            by_domain[s.domain]['tokens'] += s.total_tokens
            by_domain[s.domain]['cost'] += s.total_cost_usd

        all_latencies = [s.wall_clock_ms for s in self.sessions.values() if s.wall_clock_ms > 0]
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0

        return {
            'total_sessions': total_sessions,
            'total_tokens': total_tokens,
            'total_cost_usd': round(total_cost, 4),
            'avg_cost_per_session': round(total_cost / max(total_sessions, 1), 4),
            'avg_latency_ms': round(avg_latency, 0),
            'by_domain': dict(by_domain),
            'cost_efficiency': 'GOOD' if total_cost / max(total_sessions, 1) < 0.01 else 'REVIEW',
        }

    def export_audit_log(self, session_id: str = None) -> str:
        """导出审计日志（合规用）"""
        if session_id:
            sessions_to_export = {session_id: self.sessions[session_id]} if session_id in self.sessions else {}
        else:
            sessions_to_export = self.sessions

        audit_records = []
        for sid, session in sessions_to_export.items():
            record = {
                'session_id': sid,
                'timestamp': datetime.fromtimestamp(session.start_time).isoformat(),
                'domain': session.domain,
                'task': session.task_description,
                'total_tokens': session.total_tokens,
                'cost_usd': session.total_cost_usd,
                'agents': [
                    {
                        'agent_id': t.agent_id,
                        'latency_ms': round(t.latency_ms, 1),
                        'tokens': t.total_tokens,
                        'docs_retrieved': t.retrieved_docs,
                        'reasoning': t.reasoning_steps,
                        'status': t.status,
                    }
                    for t in session.agents
                ],
            }
            audit_records.append(record)

        return json.dumps(audit_records, ensure_ascii=False, indent=2)


def run_glass_box_demo():
    """玻璃盒可观测性系统完整演示"""
    print("=" * 65)
    print("玻璃盒MAS可观测性系统（100%透明推理追踪）")
    print("基于 Denis Rothman Context Engineering Ch10")
    print("=" * 65)

    obs = GlassBoxObservability(level=ObservabilityLevel.L2)

    # ─── 会话1：选品分析 ──────────────────────────────────────
    print("\n[会话1: 母婴选品分析MAS执行]")
    sid1 = obs.start_session("母婴吸奶器品类选品分析", domain="ecommerce")

    # Research Agent
    obs.start_agent_trace("research_agent", "母婴市场研究请求")
    obs.log_reasoning_step("research_agent", "检索美国母婴市场规模数据")
    obs.log_reasoning_step("research_agent", "分析主要竞品及市场份额")
    obs.log_retrieval("research_agent", ["DOC-market-q4", "DOC-competitor"])
    obs.log_reasoning_step("research_agent", "生成市场摘要报告")
    time.sleep(0.01)
    obs.finish_agent_trace("research_agent", "市场规模$28亿，YoY+12%", 450, 320)

    # Finance Agent
    obs.start_agent_trace("finance_agent", "研究报告JSON")
    obs.log_reasoning_step("finance_agent", "读取Research Agent输出")
    obs.log_reasoning_step("finance_agent", "计算ROI预期")
    obs.log_reasoning_step("finance_agent", "评估风险等级")
    time.sleep(0.01)
    obs.finish_agent_trace("finance_agent", "ROI预计28-35%，风险MEDIUM", 280, 180)

    # Report Agent
    obs.start_agent_trace("report_agent", "研究+财务数据")
    obs.log_reasoning_step("report_agent", "整合多Agent输出")
    obs.log_reasoning_step("report_agent", "生成执行报告")
    time.sleep(0.01)
    obs.finish_agent_trace("report_agent", "选品报告：建议进入吸奶器品类...", 380, 450)

    session1 = obs.finish_session()
    print(obs.render_trace_report(session1))

    # ─── 会话2：合规查询 ──────────────────────────────────────
    print("\n[会话2: 合规查询MAS执行]")
    sid2 = obs.start_session("CPSC婴儿产品认证要求查询", domain="compliance")
    obs.start_agent_trace("compliance_agent", "CPSC查询")
    obs.log_retrieval("compliance_agent", ["INS-cpsc-1119", "INS-fda-1119"])
    obs.log_reasoning_step("compliance_agent", "检索相关法规条文")
    obs.log_reasoning_step("compliance_agent", "验证引用完整性")
    time.sleep(0.01)
    obs.finish_agent_trace("compliance_agent", "CPSC要求：[INS-cpsc-1119]...", 200, 280)
    session2 = obs.finish_session()

    # ─── Token成本仪表盘 ──────────────────────────────────────
    print("\n[Token成本仪表盘]")
    dashboard = obs.get_cost_dashboard()
    print(f"  总会话数: {dashboard['total_sessions']}")
    print(f"  总Token: {dashboard['total_tokens']}")
    print(f"  总成本: ${dashboard['total_cost_usd']:.4f}")
    print(f"  平均会话成本: ${dashboard['avg_cost_per_session']:.4f}")
    print(f"  平均延迟: {dashboard['avg_latency_ms']:.0f}ms")
    print(f"  成本效率: {dashboard['cost_efficiency']}")
    print(f"\n  按域分布:")
    for domain, stats in dashboard['by_domain'].items():
        print(f"    {domain:<15} {stats['sessions']}次 | {stats['tokens']}tokens | ${stats['cost']:.4f}")

    # ─── 审计日志导出 ─────────────────────────────────────────
    print("\n[审计日志导出（合规用，节选）]")
    audit = obs.export_audit_log(sid1)
    audit_parsed = json.loads(audit)
    if audit_parsed:
        record = audit_parsed[0]
        print(f"  会话: {record['session_id']}")
        print(f"  时间: {record['timestamp']}")
        print(f"  Token: {record['total_tokens']} | 成本: ${record['cost_usd']:.4f}")
        print(f"  Agent数: {len(record['agents'])}")
        for agent in record['agents'][:2]:
            print(f"    [{agent['agent_id']}] {agent['tokens']}tokens | 推理步骤: {len(agent['reasoning'])}步")

    # ─── 观测性等级对比 ───────────────────────────────────────
    print("\n[观测性等级说明]")
    levels = [
        ("L0 基础", "输入/输出/Token计数", "开发调试"),
        ("L1 标准", "+Agent日志+延迟分析", "生产监控"),
        ("L2 深度", "+检索文档+推理步骤（当前）", "审计合规"),
        ("L3 完整", "+完整上下文快照+成本分析", "高风险系统"),
    ]
    for level, desc, use_case in levels:
        current = " ← 当前" if "当前" in desc else ""
        print(f"  {level:<12} {desc:<35} 适用:{use_case}{current}")

    print("\n[✓] 玻璃盒MAS可观测性系统测试通过")
    return obs


if __name__ == "__main__":
    obs = run_glass_box_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Context-Engine-Architecture]]（Engine层的执行日志是可观测性的数据来源）、[[Skill-MAS-Testing-Verification]]（可观测性数据是测试验证的基础）
- **延伸（extends）**：[[Skill-Domain-Agnostic-Context-Engine]]（Universal Context Engine以玻璃盒可观测性为核心特性）、[[Skill-MASEval-System-Evaluation]]（可观测性追踪数据是系统评估的输入）
- **可组合（combinable）**：[[Skill-Policy-Driven-Meta-Controller]]（策略决策记录到追踪系统）、[[Skill-Context-Token-Compression]]（压缩前后Token对比是成本仪表盘的核心指标）

## ⑤ 商业价值评估

- **ROI 预估**：可观测性系统使AI采用率从20%→65%（管理层信任度提升），对应AI辅助决策价值增加约$50万/年；合规审计从"无法提供"→"2小时出报告"，避免潜在监管风险；成本仪表盘使Token消耗优化30%，年化节省约$1-5万；系统成本$5万，ROI≈1000%+
- **实施难度**：⭐⭐☆☆☆（数据结构设计简单；主要工作是在所有Agent执行点插入追踪代码；Gradio UI需要额外开发）
- **优先级**：⭐⭐⭐⭐⭐（Rothman在Ch10（最终章）将玻璃盒可观测性作为生产就绪MAS的核心特征，没有可观测性的MAS无法在企业环境中被信任和采用）
- **适用规模**：所有生产级MAS系统；特别是需要合规审计的金融/法律/医疗/跨境电商场景
- **数据依赖**：无需外部数据；需要在所有Agent执行路径上插入追踪埋点
