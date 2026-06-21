---
title: Agent能力评估基准 — 任务完成率与工具调用准确率评测
doc_type: knowledge
module: 09-DataAgent-LLM
topic: agent-capability-evaluation
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Agent能力评估基准

> **论文/方法来源**：AgentBench: Evaluating LLMs as Agents (Liu et al., 2023) + ToolBench: Facilitating Large Language Models to Master 16000+ Real-world APIs (Qin et al., 2023) + GAIA Benchmark (Mialon et al., 2023)
> **领域**：09-DataAgent-LLM ↔ 16-智能体工程 | **类型**: 算法工具

## ① 算法原理

Agent 评估比模型评估更复杂，因为 Agent 的"正确"不只是最终答案，还包括**过程质量**（工具调用路径、中间推理、资源消耗）。

**核心评估维度（5D框架）**：
1. **任务完成率（TCR）**：最终是否达成目标（二值）
2. **工具调用准确率（TCA）**：工具选择 + 参数填充的综合准确性
3. **幻觉率（HR）**：调用不存在工具 / 参数值凭空捏造的比例
4. **效率分（ES）**：最小步骤完成任务的比例（避免冗余工具调用）
5. **鲁棒性（RS）**：工具报错后恢复 / 重试 / 换策略的成功率

**评分公式**：
$$\text{AgentScore} = 0.4 \cdot TCR + 0.25 \cdot TCA + 0.15 \cdot (1-HR) + 0.1 \cdot ES + 0.1 \cdot RS$$

**使用场景**：Agent 版本迭代比较（v1 vs v2）；新模型 baseline 建立；回归测试防止退化。

## ② 母婴出海应用案例

**场景A：补货 Agent 版本迭代评测**
- 业务问题：补货 Agent v2 修改了提示词，不知道是否改善了多步骤工具调用准确性，需要量化对比
- 数据要求：标准测试集（50 个补货场景），每个场景有 ground-truth 工具调用序列，Agent 实际调用记录
- 预期产出：v1 vs v2 的 5D 评分对比报告，定位退化点（如 v2 的幻觉率从 3% 升至 8%）
- 业务价值：防止无意识的 Agent 能力退化，每次迭代有数据支撑，减少上线事故

**场景B：多 Agent 横向能力基准**
- 业务问题：比较 GPT-4o / Claude-3.5 / DeepSeek-V3 在公司业务 Agent 场景的实际表现差异
- 数据要求：30 个标准业务 Task，各模型的完整调用 trace
- 预期产出：多模型 Radar Chart 对比，定量支撑模型选型决策
- 业务价值：选对模型节省 40% API 成本（约 5 万元/月），同时保证任务完成率

## ③ 代码模板

```python
"""
Agent 能力评估基准 — 5D 评估框架 + 自动化评测流水线
"""
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """单次工具调用记录"""
    tool_name: str
    parameters: Dict
    result: Optional[str] = None
    success: bool = True


@dataclass
class AgentTrace:
    """Agent 完整执行轨迹"""
    task_id: str
    task_description: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None
    task_completed: bool = False


@dataclass
class GroundTruth:
    """任务标准答案"""
    task_id: str
    expected_tool_sequence: List[str]          # 期望工具调用顺序
    expected_params_keys: List[List[str]]       # 每次调用的期望参数键
    valid_final_answers: List[str] = field(default_factory=list)
    max_steps_allowed: int = 5


def evaluate_tool_call_accuracy(
    trace: AgentTrace,
    ground_truth: GroundTruth
) -> Tuple[float, float]:
    """
    返回 (工具选择准确率, 参数准确率)
    """
    if not ground_truth.expected_tool_sequence:
        return 1.0, 1.0

    expected = ground_truth.expected_tool_sequence
    actual = [tc.tool_name for tc in trace.tool_calls]

    # 工具序列匹配（允许顺序交换，取最长公共子序列比例）
    def lcs_ratio(a, b):
        m, n = len(a), len(b)
        if max(m, n) == 0:
            return 1.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n] / max(m, n)

    tool_accuracy = lcs_ratio(actual, expected)

    # 参数键准确率
    param_scores = []
    for i, tc in enumerate(trace.tool_calls):
        if i < len(ground_truth.expected_params_keys):
            expected_keys = set(ground_truth.expected_params_keys[i])
            actual_keys = set(tc.parameters.keys())
            if expected_keys:
                overlap = len(expected_keys & actual_keys) / len(expected_keys)
                param_scores.append(overlap)

    param_accuracy = sum(param_scores) / len(param_scores) if param_scores else 1.0
    return round(tool_accuracy, 4), round(param_accuracy, 4)


def evaluate_hallucination_rate(
    trace: AgentTrace,
    valid_tools: List[str]
) -> float:
    """幻觉率：调用不存在工具的比例"""
    if not trace.tool_calls:
        return 0.0
    hallucinated = sum(1 for tc in trace.tool_calls if tc.tool_name not in valid_tools)
    return round(hallucinated / len(trace.tool_calls), 4)


def evaluate_efficiency(
    trace: AgentTrace,
    ground_truth: GroundTruth
) -> float:
    """效率分：实际步数 vs 最优步数"""
    optimal_steps = len(ground_truth.expected_tool_sequence)
    actual_steps = len(trace.tool_calls)
    if actual_steps == 0 or optimal_steps == 0:
        return 0.0
    return round(min(optimal_steps / actual_steps, 1.0), 4)


def compute_agent_score(
    trace: AgentTrace,
    ground_truth: GroundTruth,
    valid_tools: List[str],
    weights: Optional[Dict[str, float]] = None
) -> Dict:
    """计算综合 Agent Score"""
    if weights is None:
        weights = {"TCR": 0.40, "TCA": 0.25, "HR": 0.15, "ES": 0.10, "RS": 0.10}

    # 各维度计算
    tcr = float(trace.task_completed)
    tool_acc, param_acc = evaluate_tool_call_accuracy(trace, ground_truth)
    tca = (tool_acc + param_acc) / 2
    hr = evaluate_hallucination_rate(trace, valid_tools)
    es = evaluate_efficiency(trace, ground_truth)
    # 鲁棒性：成功处理工具错误的比例
    failed_calls = sum(1 for tc in trace.tool_calls if not tc.success)
    rs = 1.0 if failed_calls == 0 else float(trace.task_completed)

    agent_score = (
        weights["TCR"] * tcr +
        weights["TCA"] * tca +
        weights["HR"] * (1 - hr) +
        weights["ES"] * es +
        weights["RS"] * rs
    )

    return {
        "task_id": trace.task_id,
        "TCR": tcr,
        "TCA": round(tca, 4),
        "HR": round(hr, 4),
        "ES": round(es, 4),
        "RS": round(rs, 4),
        "AgentScore": round(agent_score, 4),
        "detail": {
            "tool_selection_accuracy": tool_acc,
            "param_accuracy": param_acc,
            "hallucinated_tools": [tc.tool_name for tc in trace.tool_calls if tc.tool_name not in valid_tools],
        }
    }


def benchmark_agents(
    agent_results: Dict[str, List[AgentTrace]],
    ground_truths: List[GroundTruth],
    valid_tools: List[str]
) -> Dict:
    """多 Agent 横向评测汇总"""
    gt_map = {gt.task_id: gt for gt in ground_truths}
    summary = {}
    for agent_name, traces in agent_results.items():
        scores = []
        for trace in traces:
            if trace.task_id in gt_map:
                score = compute_agent_score(trace, gt_map[trace.task_id], valid_tools)
                scores.append(score)
        if scores:
            summary[agent_name] = {
                "avg_agent_score": round(sum(s["AgentScore"] for s in scores) / len(scores), 4),
                "task_completion_rate": round(sum(s["TCR"] for s in scores) / len(scores), 4),
                "avg_hallucination_rate": round(sum(s["HR"] for s in scores) / len(scores), 4),
                "n_tasks": len(scores)
            }
    return summary


# ===== 测试 =====
if __name__ == "__main__":
    valid_tools = ["query_inventory", "query_sales", "adjust_price", "generate_report", "query_returns"]

    # 构造测试 trace（模拟补货 Agent 调用）
    trace = AgentTrace(
        task_id="task_001",
        task_description="检查 B08X 库存并预测是否需要补货",
        tool_calls=[
            ToolCall("query_inventory", {"asin": "B08X", "market": "US"}, result="库存: 120"),
            ToolCall("query_sales", {"asin": "B08X", "start_date": "2026-06-14", "end_date": "2026-06-21"}, result="日均销量: 15"),
        ],
        final_answer="当前库存可维持 8 天，建议本周补货 200 件",
        task_completed=True
    )

    ground_truth = GroundTruth(
        task_id="task_001",
        expected_tool_sequence=["query_inventory", "query_sales"],
        expected_params_keys=[["asin", "market"], ["asin", "start_date", "end_date"]],
        valid_final_answers=["补货"],
        max_steps_allowed=3
    )

    score = compute_agent_score(trace, ground_truth, valid_tools)
    print("=== 单任务评估 ===")
    for k, v in score.items():
        if k != "detail":
            print(f"  {k}: {v}")
    print(f"  详情: {score['detail']}")

    # 多 Agent 横向比较
    trace_v2 = AgentTrace(
        task_id="task_001",
        task_description="检查 B08X 库存并预测是否需要补货",
        tool_calls=[
            ToolCall("query_inventory", {"asin": "B08X", "market": "US"}),
            ToolCall("hallucinated_tool", {"asin": "B08X"}, success=False),  # 幻觉调用
            ToolCall("query_sales", {"asin": "B08X", "start_date": "2026-06-14", "end_date": "2026-06-21"}),
        ],
        task_completed=True
    )

    benchmark_result = benchmark_agents(
        {"agent_v1": [trace], "agent_v2": [trace_v2]},
        [ground_truth],
        valid_tools
    )
    print("\n=== 多Agent横向评测 ===")
    for agent, stats in benchmark_result.items():
        print(f"  {agent}: Score={stats['avg_agent_score']}, TCR={stats['task_completion_rate']}, HR={stats['avg_hallucination_rate']}")

    assert score["AgentScore"] > 0.5, f"Agent Score 过低: {score['AgentScore']}"
    assert score["TCR"] == 1.0
    assert benchmark_result["agent_v1"]["avg_agent_score"] > benchmark_result["agent_v2"]["avg_agent_score"], \
        "v1 应优于 v2（v2 有幻觉调用）"

    print("\n[✓] Agent能力评估基准测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-Agent-Observability-Tracing]]（评估数据来自 Trace 采集）
- **前置**：[[Skill-Agent-Error-Budget]]（错误预算与评估阈值的关系）
- **延伸**：[[Skill-Agent-Fault-Tolerance]]（评估鲁棒性维度）
- **可组合**：[[Skill-LLM-Tool-Selection-Router]]（路由准确率是评估的核心维度）
- **可组合**：[[Skill-DeepAnalyze-Autonomous-Data-Science-Agent]]（自主 Agent 的能力评测）

## ⑤ 商业价值评估

- ROI 预估：防止 Agent 退化导致的运营事故（单次事故损失 5-20 万元），年化规避风险价值 50 万元+
- 实施难度：⭐⭐⭐☆☆（需要构建标准测试集，评测框架本身开发 2-3 天）
- 优先级：⭐⭐⭐⭐☆
- 评估依据：任何上生产的 Agent 都需要持续评测，没有评测就没有迭代方向；5D 框架覆盖 Agent 特有的失败模式（幻觉/冗余/不鲁棒）
