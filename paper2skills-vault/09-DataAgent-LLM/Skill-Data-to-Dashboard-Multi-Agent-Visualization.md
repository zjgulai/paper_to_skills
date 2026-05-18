---
title: Data-to-Dashboard — 多Agent智能可视化生成
name: Skill-Data-to-Dashboard-Multi-Agent-Visualization
description: 基于两阶段多Agent架构，将原始数据自动转化为商业洞察可视化仪表板，无需人工定义图表模板
module: data-agent-llm
topic: multi-agent-visualization
version: 0.1.0
status: stable
created: 2026-04-26
updated: 2026-04-26
paper: arXiv:2505.23695
source: ai
---

# Data-to-Dashboard — 多Agent智能可视化生成

## 1. 算法原理

Data-to-Dashboard 的核心思想是**模拟商业分析师的工作流**——不是让 LLM 直接生成图表，而是先理解数据背后的业务洞察，再基于洞察选择最合适的可视化表达方式。

**两阶段架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Data-to-Dashboard Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Data-to-Insight                                      │
│    ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│    │Domain       │──►│Concept      │──►│Multi-Perspective│   │
│    │Detection    │   │Extraction   │   │Analysis         │   │
│    └─────────────┘   └─────────────┘   └─────────────────┘   │
│                            │                                  │
│                            ▼                                  │
│                     ┌─────────────┐                           │
│                     │Self-Reflect │  (迭代优化洞察质量)        │
│                     └─────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Insight-to-Chart                                     │
│    ┌─────────────────┐   ┌─────────────────┐                │
│    │Tree-of-Thoughts │──►│Expert Consensus │                │
│    │Visualization    │   │Chart Selection  │                │
│    │Reasoning        │   │                 │                │
│    └─────────────────┘   └─────────────────┘                │
│                            │                                  │
│                            ▼                                  │
│                     ┌─────────────┐                           │
│                     │Chart Code   │  (生成可执行可视化代码)     │
│                     │Generation   │                           │
│                     └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

**Stage 1 — Data-to-Insight（洞察生成）**：

| Agent | 职责 | 输出 |
|-------|------|------|
| **Domain Detection** | 识别数据所属业务领域 | 领域标签（销售/用户/库存等） |
| **Concept Extraction** | 提取关键指标和维度 | 概念列表（GMV、留存率、SKU等） |
| **Multi-Perspective** | 从多个视角分析数据 | 多角度洞察（趋势/分布/关联/异常） |
| **Self-Reflection** | 评估洞察质量并迭代优化 | 精炼后的高质量洞察集合 |

关键创新：**不依赖封闭本体或预定义问题模板**。传统 BI 工具需要人工配置维度/度量/图表类型，D2D 通过 Agent 自主推理，从数据中"发现"应该看什么。

**Stage 2 — Insight-to-Chart（图表生成）**：

采用 **Tree-of-Thoughts（ToT）** 推理机制：

```
洞察输入
    → 分支1: 柱状图候选（清晰度评分=0.8, 信息密度=0.7）
    → 分支2: 折线图候选（清晰度评分=0.9, 信息密度=0.6）
    → 分支3: 热力图候选（清晰度评分=0.5, 信息密度=0.9）
         ↓
    专家共识评估（多维度加权）
         ↓
    剪枝低分分支（热力图因清晰度不足被剪枝）
         ↓
    选择最优路径（折线图综合得分最高）
         ↓
    生成可执行图表代码
```

具体步骤：
1. **候选分支生成**：为每个洞察生成 2-3 种可视化方案，形成搜索树的根节点到叶节点路径
2. **专家评估打分**：多专家 Agent 并行评估每个候选，维度包括"清晰度"（能否直观传达洞察）、"信息密度"（是否包含足够数据维度）、"美学"（是否符合企业报告规范）
3. **低分剪枝**：低于阈值（如综合评分 < 0.6）的分支被剪除，减少后续计算
4. **最优路径选择**：剩余分支中按加权总分排序，选择 Top-1
5. **代码生成**：将最优方案转化为可直接执行的 matplotlib/plotly 代码

**Stage 1 → Stage 2 的数据流**：
- 输入：`Insight` 对象（perspective + description + confidence + supporting_data）
- 转化：LLM 将自然语言洞察描述翻译为图表语义（x轴 → 时间维度，y轴 → 指标值，颜色 → 分类维度）
- 输出：`ChartSpec` 对象（chart_type + title + labels + executable_code）

**评估结果**（相比 GPT-4o 单提示基线）：
- 洞察深度（Insightfulness）：+12%
- 新颖性（Novelty）：+28%
- 深度（Depth）：+31%

## 2. 业务应用

### 场景A：母婴出海周报仪表板自动生成

**背景**：运营团队每周需要整合 Amazon/Shopify/SHEIN 三平台数据制作周报仪表板，涉及 15+ 张图表，人工制作耗时 3-4 小时。

**Agent工作流**：
1. **Domain Detection**：识别数据领域 —— 跨境电商销售分析
2. **Concept Extraction**：提取关键指标 —— GMV、ROI、退货率、广告 spend、SKU 动销率
3. **Multi-Perspective Analysis**：
   - 时间视角：各平台 GMV 周趋势对比
   - 分布视角：SKU 销量分布（帕累托分析）
   - 关联视角：广告 spend 与 ROI 相关性
   - 异常视角：退货率异常时段识别
4. **Self-Reflection**：检查洞察 —— "Shopify 退货率突增是否与其他指标联动？"
5. **Insight-to-Chart**：
   - GMV 趋势 → 多线折线图
   - SKU 分布 → 帕累托柱状图
   - 广告-ROI → 散点图 + 趋势线
   - 退货异常 → 异常标记折线图
6. **输出**：完整仪表板代码 + 洞察摘要

**预期效果**：3-4 小时/周 → 5 分钟，且图表选择更符合分析目的。

### 场景B：VOC 评论洞察可视化

**背景**：从 Trustpilot/Amazon 收集的用户评论经 NLP 分析后，需要向产品团队展示洞察。

**Agent工作流**：
1. **Domain Detection**：用户声音/产品反馈分析
2. **Concept Extraction**：情感分数、关键词、产品型号、时间
3. **Multi-Perspective Analysis**：
   - 情感趋势时间线
   - 产品型号差评分布
   - 关键词共现网络
   - 情感-评分偏离分析
4. **Self-Reflection**："差评集中在近 30 天，是否与大促物流延迟有关？"
5. **Insight-to-Chart**：
   - 情感趋势 → 面积图
   - 型号差评 → 横向柱状图
   - 关键词网络 → 力导向图
   - 偏离分析 → 散点图

## 3. 代码模板

代码位置：`paper2skills-code/data_agent_llm/data_to_dashboard_agent/dashboard_agent.py`

```python
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
    perspective: str
    description: str
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartSpec:
    """图表规格"""
    insight: Insight
    chart_type: str
    title: str
    x_label: str
    y_label: str
    code: str
    score: float = 0.0


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
        """端到端仪表板生成"""
        # Stage 1: Data-to-Insight
        domain = self._detect_domain(data, context)
        concepts = self._extract_concepts(data, domain)
        insights = self._multi_perspective_analysis(data, concepts)
        refined_insights = self._self_reflection(insights, data)

        # Stage 2: Insight-to-Chart
        charts = []
        for insight in refined_insights[:5]:
            candidates = self._generate_chart_candidates(insight, data)
            best_chart = self._expert_consensus(candidates, data)
            if best_chart:
                charts.append(best_chart)

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
        # 调用 LLM 识别数据所属业务领域
        pass

    def _extract_concepts(self, data: pd.DataFrame, domain: str) -> Dict[str, List[str]]:
        """Agent 2: 概念提取"""
        # 调用 LLM 提取关键指标和维度
        pass

    def _multi_perspective_analysis(self, data: pd.DataFrame,
                                     concepts: Dict[str, List[str]]) -> List[Insight]:
        """Agent 3: 多角度分析"""
        # 从5个视角分别分析数据
        pass

    def _self_reflection(self, insights: List[Insight],
                         data: pd.DataFrame) -> List[Insight]:
        """Agent 4: 自反思迭代优化"""
        # 评估洞察质量，去除低质量/重复项
        pass

    def _generate_chart_candidates(self, insight: Insight,
                                    data: pd.DataFrame) -> List[ChartSpec]:
        """为单个洞察生成多种图表候选"""
        pass

    def _expert_consensus(self, candidates: List[ChartSpec],
                          data: pd.DataFrame) -> Optional[ChartSpec]:
        """专家共识：评估并选择最佳图表"""
        pass

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
            code = chart.code.replace(f"/tmp/d2d_chart_{chart.chart_type}.png", path)

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    compiled = compile(code, '<d2d_chart>', 'exec')
                    exec(compiled, env)
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
        pass

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
```

完整可运行代码见 `paper2skills-code/data_agent_llm/data_to_dashboard_agent/dashboard_agent.py`。

运行测试：

```bash
cd paper2skills-code/data_agent_llm/data_to_dashboard_agent
export OPENAI_API_KEY=your_key
python dashboard_agent.py
```

## 4. 技能关系

### 前置技能
- **Python数据分析基础**：Pandas、NumPy、Matplotlib 基本操作
- **LLM API 调用**：OpenAI API 或兼容接口的使用

### 关联技能
- [Skill-DeepAnalyze-Autonomous-Data-Science-Agent](paper2skills-vault/09-DataAgent-LLM/Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md) — DeepAnalyze 提供数据分析能力，D2D 提供可视化呈现能力，两者结合形成"分析+展示"完整链路
- [Skill-Argos-Agentic-Anomaly-Detection](paper2skills-vault/09-DataAgent-LLM/Skill-Argos-Agentic-Anomaly-Detection.md) — Argos 识别异常后，D2D 自动生成异常监控仪表板
- [Skill-Aspect-Based-Sentiment-Analysis](paper2skills-vault/07-NLP-VOC/Skill-Aspect-Based-Sentiment-Analysis.md) — ABSA 提供结构化情感数据，D2D 将其可视化
- [Skill-Demand-Forecasting](paper2skills-vault/03-时间序列/Skill-Demand-Forecasting.md) — 预测结果通过 D2D 生成预测趋势仪表板

### 扩展方向
- **+ DeepAnalyze** → 数据分析 + 可视化双链路：DeepAnalyze 做深度分析，D2D 做图表呈现
- **+ Argos** → 异常检测 → 自动异常监控仪表板
- **+ 前端渲染** → 将 matplotlib 输出升级为交互式 Plotly/Dash 仪表板
- **+ 定时调度** → 周报/日报自动生成并推送


## ④ 技能关联

### 前置技能
- [Skill-SQL-Agent-Text-to-SQL](../09-DataAgent-LLM/Skill-SQL-Agent-Text-to-SQL.md) — 可视化的数据来源依赖 SQL Agent

### 延伸技能
- [Skill-DeepAnalyze-Autonomous-Data-Science-Agent](../09-DataAgent-LLM/Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md) — 可视化嵌入自治数据分析 Agent

### 可组合
- [Skill-MAS-Orchestrator](../10-MAS/Skill-MAS-Orchestrator.md) — 可视化作为多 Agent 编排的最终输出环节

## 5. 商业价值评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **ROI** | ★★★★★ | 将仪表板制作从3-4小时/周压缩至5分钟，人力成本节省显著 |
| **实施难度** | ★★★☆☆ | 原型可基于开源代码快速搭建；生产级需要解决安全沙箱、图表定制需求 |
| **业务匹配度** | ★★★★★ | 直接解决母婴出海多平台周报制作痛点，且图表选择更符合分析目的 |
| **技术成熟度** | ★★★★☆ | KDD 2025 Workshop 论文，有开源代码；但多Agent调用的API成本需控制 |
| **优先级** | **P2** | 高ROI + 完整开源实现，作为 Data Agent 方向第三个落地技能，与 DeepAnalyze 形成互补 |

**量化ROI估算**：
- 假设运营团队3人，每人每周制作仪表板耗时3小时
- 人力成本按¥200/小时计算
- 使用Agent后：3人 x 3小时 x ¥200 = ¥1800/周 → ¥150/周（API成本）
- **年节省：约 ¥8.6万**

**与 DeepAnalyze 的协同价值**：
- DeepAnalyze 解决"分析什么"（生成洞察报告）
- Data-to-Dashboard 解决"怎么看"（生成可视化图表）
- 两者结合形成完整的"数据→洞察→可视化"决策链路，预期年节省可达 **¥25-30万**
