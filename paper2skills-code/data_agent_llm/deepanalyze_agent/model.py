"""
DeepAnalyze Autonomous Data Science Agent — 自治数据科学 Agent
paper2skills-code: 09-DataAgent-LLM | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field


@dataclass
class AnalysisTask:
    task_id: str
    question: str
    data_description: str
    priority: str = "medium"


@dataclass
class AnalysisStep:
    step_name: str
    tool_used: str
    input_summary: str
    output_summary: str
    confidence: float


@dataclass
class AnalysisReport:
    task_id: str
    question: str
    steps: list[AnalysisStep]
    answer: str
    confidence: float
    follow_up_questions: list[str]
    code_snippet: str


class DataProfiler:
    """数据概览工具"""
    def profile(self, data: list[dict]) -> dict:
        if not data:
            return {}
        keys = list(data[0].keys())
        profile = {}
        for k in keys:
            vals = [row[k] for row in data if k in row]
            numeric = [v for v in vals if isinstance(v, (int, float))]
            if numeric:
                profile[k] = {
                    "type": "numeric",
                    "mean": round(sum(numeric)/len(numeric), 2),
                    "min": min(numeric), "max": max(numeric),
                    "n": len(numeric),
                }
            else:
                from collections import Counter
                profile[k] = {"type": "categorical",
                               "top": Counter(vals).most_common(3), "n": len(vals)}
        return profile


class CorrelationAnalyzer:
    """相关性分析工具"""
    def correlate(self, x: list[float], y: list[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        mx = sum(x)/len(x); my = sum(y)/len(y)
        num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
        dx = math.sqrt(sum((xi-mx)**2 for xi in x))
        dy = math.sqrt(sum((yi-my)**2 for yi in y))
        return round(num / (dx*dy + 1e-9), 4)


class HypothesisTester:
    """假设检验工具（简化 t-test）"""
    def test(self, group_a: list[float], group_b: list[float]) -> dict:
        if not group_a or not group_b:
            return {"significant": False, "p_value": 1.0}
        ma = sum(group_a)/len(group_a)
        mb = sum(group_b)/len(group_b)
        sa = math.sqrt(sum((x-ma)**2 for x in group_a)/max(len(group_a)-1,1))
        sb = math.sqrt(sum((x-mb)**2 for x in group_b)/max(len(group_b)-1,1))
        se = math.sqrt(sa**2/len(group_a) + sb**2/len(group_b)) + 1e-9
        t_stat = abs(ma - mb) / se
        p_approx = max(0.001, 2 * (1 - min(0.999, t_stat/4)))
        return {"t_stat": round(t_stat, 3), "p_value": round(p_approx, 4),
                "significant": p_approx < 0.05,
                "mean_a": round(ma, 2), "mean_b": round(mb, 2)}


class DeepAnalyzeAgent:
    """自治数据科学 Agent：自动规划分析步骤并执行"""

    def __init__(self):
        self.profiler = DataProfiler()
        self.correlator = CorrelationAnalyzer()
        self.hypothesis = HypothesisTester()

    def analyze(self, task: AnalysisTask, data: list[dict]) -> AnalysisReport:
        steps = []

        profile = self.profiler.profile(data)
        steps.append(AnalysisStep(
            step_name="数据概览",
            tool_used="DataProfiler",
            input_summary=f"{len(data)} 条记录, {len(profile)} 个字段",
            output_summary=f"发现 {sum(1 for v in profile.values() if v['type']=='numeric')} 个数值字段",
            confidence=0.95,
        ))

        numeric_cols = [k for k, v in profile.items() if v.get("type") == "numeric"]
        corr_insight = ""
        if len(numeric_cols) >= 2:
            x = [row.get(numeric_cols[0], 0) for row in data]
            y = [row.get(numeric_cols[1], 0) for row in data]
            corr = self.correlator.correlate(x, y)
            corr_insight = f"{numeric_cols[0]} 与 {numeric_cols[1]} 相关系数: {corr}"
            steps.append(AnalysisStep(
                step_name="相关性分析",
                tool_used="CorrelationAnalyzer",
                input_summary=f"分析 {numeric_cols[0]} vs {numeric_cols[1]}",
                output_summary=corr_insight,
                confidence=0.85,
            ))

        text_cols = [k for k, v in profile.items() if v.get("type") == "categorical"]
        hypothesis_insight = ""
        if text_cols and len(numeric_cols) >= 1:
            col = text_cols[0]; metric = numeric_cols[0]
            groups: dict[str, list[float]] = {}
            for row in data:
                key = str(row.get(col, "other"))
                groups.setdefault(key, []).append(float(row.get(metric, 0)))
            if len(groups) >= 2:
                keys = list(groups.keys())
                result = self.hypothesis.test(groups[keys[0]], groups[keys[1]])
                sig = "显著" if result["significant"] else "不显著"
                hypothesis_insight = (f"{col} 分组在 {metric} 上差异{sig} "
                                      f"(p={result['p_value']}, "
                                      f"均值: {keys[0]}={result['mean_a']}, "
                                      f"{keys[1]}={result['mean_b']})")
                steps.append(AnalysisStep(
                    step_name="假设检验",
                    tool_used="HypothesisTester",
                    input_summary=f"比较 {col} 分组的 {metric} 差异",
                    output_summary=hypothesis_insight,
                    confidence=0.90,
                ))

        answer_parts = [f"数据集包含 {len(data)} 条记录"]
        if corr_insight:
            answer_parts.append(corr_insight)
        if hypothesis_insight:
            answer_parts.append(hypothesis_insight)

        avg_confidence = sum(s.confidence for s in steps) / max(len(steps), 1)

        return AnalysisReport(
            task_id=task.task_id,
            question=task.question,
            steps=steps,
            answer="；".join(answer_parts),
            confidence=round(avg_confidence, 3),
            follow_up_questions=[
                f"是否需要对 {numeric_cols[0] if numeric_cols else '关键指标'} 做时序预测？",
                "是否需要识别异常值并生成预警？",
            ],
            code_snippet=(f"import pandas as pd"
                          f"df = pd.DataFrame(data)"
                          f"print(df[{numeric_cols[:2]}].corr())" if numeric_cols else ""),
        )


def run_deepanalyze_demo():
    random.seed(42)
    data = [
        {"sku": f"SKU-{i}", "category": ["奶粉","湿巾","玩具"][i%3],
         "revenue": random.uniform(1000, 15000),
         "orders": random.randint(10, 200),
         "return_rate": random.uniform(0.01, 0.12)}
        for i in range(30)
    ]
    task = AnalysisTask("T001", "分析各品类的营收和退货率关系", "母婴店铺30天销售数据")
    agent = DeepAnalyzeAgent()
    report = agent.analyze(task, data)

    print(f"=== 自治数据分析报告 ===问题: {report.question}")
    for i, step in enumerate(report.steps, 1):
        print(f"步骤 {i} [{step.tool_used}]: {step.output_summary}")
    print(f"结论: {report.answer}")
    print(f"整体置信度: {report.confidence:.2f}")
    print(f"后续建议: {report.follow_up_questions[0]}")
    print("✅ 分析完成")
if __name__ == "__main__":
    run_deepanalyze_demo()
