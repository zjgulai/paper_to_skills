"""
Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization
基于两阶段多Agent架构的智能可视化生成

论文: Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization
arXiv: 2505.23695
开源: https://github.com/77bvC/D2D_Data2Dashboard

⚠️ 安全警告：本原型使用 exec() 执行LLM生成的代码。
生产环境必须：
1. 使用 Docker 沙箱或 RestrictedPython 限制执行环境
2. 仅暴露白名单API（禁止文件系统写操作、网络访问）
3. 设置执行超时（如30秒）
4. 禁用危险内置函数（__import__, open, eval, exec）
"""

import os
import re
import io
import json
import contextlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI


class AgentRole(Enum):
    DOMAIN_DETECTOR = "domain_detector"
    CONCEPT_EXTRACTOR = "concept_extractor"
    PERSPECTIVE_ANALYZER = "perspective_analyzer"
    SELF_REFLECTOR = "self_reflector"
    VISUALIZATION_SELECTOR = "visualization_selector"
    CHART_GENERATOR = "chart_generator"


@dataclass
class Insight:
    """数据洞察"""
    perspective: str  # 视角: trend/distribution/correlation/anomaly
    description: str  # 洞察描述
    confidence: float  # 置信度 0-1
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartSpec:
    """图表规格"""
    insight: Insight
    chart_type: str  # line/bar/scatter/heatmap/etc
    title: str
    x_label: str
    y_label: str
    code: str  # 可执行图表代码
    score: float = 0.0  # 专家评分


class DataToDashboardAgent:
    """
    Data-to-Dashboard 多Agent可视化生成器

    两阶段架构：
    - Stage 1: Data-to-Insight（4个Agent协作生成洞察）
    - Stage 2: Insight-to-Chart（2个Agent协作生成图表）
    """

    PERSPECTIVES = ["trend", "distribution", "correlation", "anomaly", "composition"]

    CHART_TYPES = {
        "trend": ["line", "area", "candlestick"],
        "distribution": ["histogram", "bar", "box"],
        "correlation": ["scatter", "heatmap", "bubble"],
        "anomaly": ["line_with_markers", "box", "scatter"],
        "composition": ["pie", "stacked_bar", "treemap"],
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_dashboard(self, data: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """
        端到端仪表板生成

        Args:
            data: 输入数据框
            context: 业务上下文描述

        Returns:
            {"insights": [...], "charts": [...], "summary": "..."}
        """
        # Stage 1: Data-to-Insight
        domain = self._detect_domain(data, context)
        concepts = self._extract_concepts(data, domain)
        insights = self._multi_perspective_analysis(data, concepts)
        refined_insights = self._self_reflection(insights, data)

        # Stage 2: Insight-to-Chart
        charts = []
        for insight in refined_insights[:5]:  # 最多生成5张图
            candidates = self._generate_chart_candidates(insight, data)
            best_chart = self._expert_consensus(candidates, data)
            if best_chart:
                charts.append(best_chart)

        # 执行图表代码生成图片
        chart_paths = self._execute_charts(charts)

        summary = self._generate_summary(refined_insights, charts)

        return {
            "domain": domain,
            "concepts": concepts,
            "insights": [self._insight_to_dict(i) for i in refined_insights],
            "charts": [self._chart_to_dict(c) for c in charts],
            "chart_paths": chart_paths,
            "summary": summary,
        }

    def _detect_domain(self, data: pd.DataFrame, context: str) -> str:
        """Agent 1: 领域检测"""
        columns = ', '.join(data.columns.tolist())
        prompt = f"""你是数据分析领域的专家。基于以下数据列名和业务上下文，判断这份数据属于哪个业务领域。

数据列名: {columns}
业务上下文: {context}

请只输出一个领域标签，从以下选择：
- e-commerce-sales（电商销售）
- user-behavior（用户行为）
- supply-chain（供应链）
- marketing（营销）
- customer-service（客服/VOC）
- finance（财务）

输出格式: DOMAIN: <领域标签>"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = response.choices[0].message.content
        match = re.search(r'DOMAIN:\s*(\S+)', content)
        return match.group(1) if match else "general"

    def _extract_concepts(self, data: pd.DataFrame, domain: str) -> Dict[str, List[str]]:
        """Agent 2: 概念提取"""
        columns = ', '.join(data.columns.tolist())
        dtypes = {c: str(t) for c, t in data.dtypes.items()}

        prompt = f"""你是{domain}领域的业务分析师。从以下数据中提取关键业务概念。

数据列名: {columns}
数据类型: {dtypes}

请输出以下概念分类（JSON格式）：
{{
  "metrics": ["数值型指标，如GMV, revenue, sales_qty"],
  "dimensions": ["分维度字段，如platform, sku, region"],
  "time_fields": ["时间字段"],
  "id_fields": ["ID字段"]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return {"metrics": [], "dimensions": [], "time_fields": [], "id_fields": []}

    def _multi_perspective_analysis(self, data: pd.DataFrame,
                                     concepts: Dict[str, List[str]]) -> List[Insight]:
        """Agent 3: 多角度分析"""
        insights = []
        data_summary = self._compute_data_summary(data)

        for perspective in self.PERSPECTIVES:
            prompt = f"""你是数据洞察专家。从"{perspective}"视角分析以下数据，找出有价值的业务洞察。

数据摘要:
{data_summary}

关键概念:
- 指标: {concepts.get('metrics', [])}
- 维度: {concepts.get('dimensions', [])}
- 时间: {concepts.get('time_fields', [])}

要求：
1. 找出1-2个该视角下的关键洞察
2. 每个洞察必须有数据支撑
3. 用中文描述洞察
4. 评估置信度（0-1）

输出格式（每行一个洞察）:
INSIGHT: <描述>
CONFIDENCE: <0-1>
SUPPORTING: <支持数据>"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            content = response.choices[0].message.content
            perspective_insights = self._parse_insights(content, perspective)
            insights.extend(perspective_insights)

        return insights

    def _self_reflection(self, insights: List[Insight],
                         data: pd.DataFrame) -> List[Insight]:
        """Agent 4: 自反思迭代优化"""
        if len(insights) <= 3:
            return insights

        insights_text = '\n'.join([
            f"{i+1}. [{insp.perspective}] {insp.description} (置信度: {insp.confidence})"
            for i, insp in enumerate(insights)
        ])

        prompt = f"""你是资深数据分析师。请审查以下洞察列表，去除低质量或重复的洞察，保留最有价值的5个。

洞察列表:
{insights_text}

评估标准:
1. 是否具备业务 actionable 价值
2. 是否有数据支撑
3. 是否与其他洞察互补（避免重复视角）
4. 置信度是否足够高（>0.5）

输出要保留的洞察编号（用逗号分隔）:
SELECTED: <编号列表>"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content
        match = re.search(r'SELECTED:\s*([\d,\s]+)', content)

        if match:
            selected = [int(x.strip()) - 1 for x in match.group(1).split(',') if x.strip().isdigit()]
            selected = [i for i in selected if 0 <= i < len(insights)]
            if selected:
                return [insights[i] for i in selected]

        # 降级：按置信度排序取前5
        return sorted(insights, key=lambda x: x.confidence, reverse=True)[:5]

    def _generate_chart_candidates(self, insight: Insight,
                                    data: pd.DataFrame) -> List[ChartSpec]:
        """为单个洞察生成多种图表候选"""
        candidates = []
        chart_types = self.CHART_TYPES.get(insight.perspective, ["bar", "line"])

        for chart_type in chart_types[:3]:  # 每种视角最多3种图表
            prompt = f"""你是数据可视化专家。为以下洞察生成一个{chart_type}图表的Python代码（使用matplotlib）。

洞察: {insight.description}
视角: {insight.perspective}

数据列名: {', '.join(data.columns.tolist())}

要求：
1. 代码必须完整可执行
2. 使用 matplotlib.pyplot (已导入为 plt)
3. 数据通过变量 `df` 访问（pandas DataFrame）
4. 设置合适的中文标题（使用英文标签避免字体问题）
5. 保存图片到 `/tmp/d2d_chart_{chart_type}.png`
6. 只输出代码块，不要其他解释

输出格式:
```python
<代码>
```"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            content = response.choices[0].message.content
            code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
            code = code_blocks[0].strip() if code_blocks else content.strip()

            candidates.append(ChartSpec(
                insight=insight,
                chart_type=chart_type,
                title=f"{insight.perspective.title()}: {insight.description[:50]}",
                x_label="",
                y_label="",
                code=code
            ))

        return candidates

    def _expert_consensus(self, candidates: List[ChartSpec],
                          data: pd.DataFrame) -> Optional[ChartSpec]:
        """专家共识：评估并选择最佳图表"""
        if not candidates:
            return None

        best = candidates[0]
        best_score = -1

        for candidate in candidates:
            # 执行代码并评分
            score = self._evaluate_chart(candidate, data)
            candidate.score = score
            if score > best_score:
                best_score = score
                best = candidate

        return best if best_score > 0 else candidates[0]

    def _evaluate_chart(self, chart: ChartSpec, data: pd.DataFrame) -> float:
        """评估图表质量（启发式评分）"""
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'sum': sum, 'min': min, 'max': max, 'abs': abs,
            'round': round, 'float': float, 'int': int, 'str': str,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'print': print, 'isinstance': isinstance,
        }
        env = {'__builtins__': safe_builtins, 'pd': pd, 'np': np, 'plt': plt, 'df': data}

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                compiled = compile(chart.code, '<d2d_chart>', 'exec')
                exec(compiled, env)
            return 1.0  # 成功执行 = 满分
        except Exception:
            return 0.0  # 执行失败 = 0分

    def _execute_charts(self, charts: List[ChartSpec]) -> List[str]:
        """执行所有图表代码并返回文件路径"""
        paths = []
        for i, chart in enumerate(charts):
            path = f"/tmp/d2d_output_{i}_{chart.chart_type}.png"
            safe_builtins = {
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'float': float, 'int': int, 'str': str,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'print': print, 'isinstance': isinstance,
            }
            env = {'__builtins__': safe_builtins, 'pd': pd, 'np': np, 'plt': plt}

            # 查找代码中的 savefig 路径，替换为统一路径
            code = chart.code.replace(f"/tmp/d2d_chart_{chart.chart_type}.png", path)

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    compiled = compile(code, '<d2d_chart>', 'exec')
                    exec(compiled, env)
                # 如果没有显式savefig，补充保存
                if 'savefig' not in code:
                    plt.savefig(path)
                    plt.close()
                paths.append(path)
            except Exception as e:
                paths.append(f"ERROR: {e}")

        return paths

    def _generate_summary(self, insights: List[Insight],
                          charts: List[ChartSpec]) -> str:
        """生成仪表板摘要"""
        return f"""Data-to-Dashboard 生成报告
============================
共发现 {len(insights)} 个关键洞察，生成 {len(charts)} 张可视化图表。

关键发现:
{chr(10).join(f"- [{i.perspective}] {i.description}" for i in insights[:3])}

图表列表:
{chr(10).join(f"- {c.chart_type}: {c.title}" for c in charts)}
"""

    # ========== 辅助方法 ==========

    def _compute_data_summary(self, data: pd.DataFrame) -> str:
        lines = [f"形状: {data.shape[0]} 行 x {data.shape[1]} 列"]
        lines.append(f"列名: {', '.join(data.columns.tolist())}")

        numeric = data.select_dtypes(include=[np.number])
        if not numeric.empty:
            desc = numeric.describe().to_string()
            lines.append(f"数值列统计:\n{desc}")

        return '\n'.join(lines)

    def _parse_insights(self, content: str, perspective: str) -> List[Insight]:
        """解析LLM输出的洞察"""
        insights = []
        pattern = r'INSIGHT:\s*(.+?)(?=CONFIDENCE:|$)'
        conf_pattern = r'CONFIDENCE:\s*([\d.]+)'

        for match in re.finditer(pattern, content, re.DOTALL):
            desc = match.group(1).strip()
            conf_match = re.search(conf_pattern, content[match.end():])
            confidence = float(conf_match.group(1)) if conf_match else 0.5

            insights.append(Insight(
                perspective=perspective,
                description=desc,
                confidence=min(max(confidence, 0.0), 1.0)
            ))

        return insights

    def _insight_to_dict(self, insight: Insight) -> Dict[str, Any]:
        return {
            "perspective": insight.perspective,
            "description": insight.description,
            "confidence": insight.confidence,
        }

    def _chart_to_dict(self, chart: ChartSpec) -> Dict[str, Any]:
        return {
            "type": chart.chart_type,
            "title": chart.title,
            "score": chart.score,
            "insight": chart.insight.description,
        }


# ============ 测试用例 ============

def create_test_data() -> pd.DataFrame:
    """生成测试用的母婴出海多平台销售数据"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=90, freq='D')

    data = {
        'date': dates,
        'platform': np.random.choice(['Amazon', 'Shopify', 'SHEIN'], 90),
        'sku': np.random.choice(['Bottle-001', 'Diaper-002', 'Stroller-003', 'Pump-004'], 90),
        'sales_qty': np.random.poisson(50, 90),
        'revenue': np.random.normal(500, 150, 90).round(2),
        'ad_spend': np.random.normal(100, 30, 90).round(2),
        'returns': np.random.poisson(3, 90),
    }
    df = pd.DataFrame(data)
    df['roi'] = ((df['revenue'] - df['ad_spend']) / df['ad_spend']).round(2)
    df['return_rate'] = (df['returns'] / df['sales_qty']).round(3)

    # 注入异常
    mask = (df['platform'] == 'Shopify') & (df.index >= 60) & (df.index <= 65)
    df.loc[mask, 'returns'] = 20
    df.loc[mask, 'return_rate'] = (df.loc[mask, 'returns'] / df.loc[mask, 'sales_qty']).round(3)

    return df


def test_dashboard_agent():
    """测试 Data-to-Dashboard Agent"""
    df = create_test_data()
    print("=" * 60)
    print("Data-to-Dashboard 智能可视化生成测试")
    print("=" * 60)
    print(f"测试数据: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"列名: {', '.join(df.columns.tolist())}")
    print()

    agent = DataToDashboardAgent(model="gpt-4o-mini")

    result = agent.generate_dashboard(
        data=df,
        context="母婴品牌出海跨境电商销售数据，覆盖Amazon/Shopify/SHEIN三个平台"
    )

    print(f"\n检测到业务领域: {result['domain']}")
    print(f"\n提取的关键概念:")
    for k, v in result['concepts'].items():
        print(f"  {k}: {v}")

    print(f"\n发现 {len(result['insights'])} 个洞察:")
    for i, insight in enumerate(result['insights'], 1):
        print(f"  {i}. [{insight['perspective']}] {insight['description'][:80]}...")
        print(f"     置信度: {insight['confidence']}")

    print(f"\n生成 {len(result['charts'])} 张图表:")
    for i, chart in enumerate(result['charts'], 1):
        print(f"  {i}. [{chart['type']}] {chart['title'][:60]}")

    print(f"\n图表文件:")
    for path in result['chart_paths']:
        print(f"  - {path}")

    print(f"\n{'=' * 60}")
    print(result['summary'])

    return result


if __name__ == "__main__":
    test_dashboard_agent()
