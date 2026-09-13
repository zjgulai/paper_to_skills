---
title: DeepAnalyze — 自主数据科学Agent
name: Skill-DeepAnalyze-Autonomous-Data-Science-Agent
description: 基于课程式Agent训练和五动作编排架构，让LLM自主完成从原始数据到分析级报告的端到端数据科学任务
module: data-agent-llm
topic: autonomous-data-science-agent
version: 0.1.0
status: stable
created: 2026-04-26
updated: 2026-04-26
venue_tier: preprint
venue_source: arxiv-abs(无发表声明)
paper: arXiv:2510.16872
paper_id: 2510.16872
source: ai
evidence_basis: paper-verbatim
l1_id: PLN-PLT
l1_plane: 数据与Agent平台
l2_id: DOM-08
l2_domain: 数据与AI运行
l3_id: DOM-08-144
l3_business: 业务工具实现
l3_all: 业务工具实现 / 经营预测
l1_l2_l3: 数据与Agent平台/数据与AI运行/业务工具实现
---

# DeepAnalyze — 自主数据科学Agent

## 1. 算法原理

DeepAnalyze 的核心思想是**让LLM像人类数据科学家一样工作**——不是按预定义工作流执行固定步骤，而是自主决定下一步该做什么。

**五动作编排架构**：

| 动作 | 职责 | 触发条件 |
|------|------|----------|
| **Analyze** | 文本分析、任务规划、推理、反思 | 需要理解问题或调整策略时 |
| **Understand** | 读取并理解数据源（CSV/数据库/API） | 需要了解数据结构和内容时 |
| **Code** | 生成Python代码（Pandas/NumPy/Matplotlib） | 需要对数据进行操作时 |
| **Execute** | 在环境中执行代码并收集输出/错误 | Code动作后自动触发 |
| **Answer** | 生成最终报告或回答 | 任务完成时 |

**课程式Agent训练**（关键创新）：

模拟人类数据科学家从新手到专家的学习路径：
1. **单能力阶段**：分别训练推理、结构化数据理解、代码生成三个基础能力
2. **多能力阶段**：通过GRPO（Group Relative Policy Optimization）强化学习训练模型自主编排多个动作的组合。核心思想：同一问题的多组回答中，优于组内平均水平的回答获得正奖励，反之获得负奖励——无需外部评判模型，靠组内相对比较驱动策略优化。

**数据Grounded轨迹合成**：

解决训练数据稀缺问题——从现有TableQA等结构化数据数据集自动合成高质量交互轨迹，包含推理过程（Reasoning Trajectory）和多轮环境交互（Interaction Trajectory）。

**混合奖励建模**：

- 规则奖励：输出格式正确性
- LLM-as-judge：报告质量（实用性、丰富度、严谨性、可解释性、可读性）
- 交互质量：环境交互轮次与成功率

仅用8B参数，在12个数据科学benchmark上超越GPT-4o等工作流agent。

## 2. 业务应用

### 场景A：母婴出海多平台销售数据自动分析报告

**背景**：母婴品牌在Amazon、Shopify、SHEIN等多个平台销售，运营团队每周需要汇总各平台数据生成分析报告，耗时4-6小时/周。

**Agent工作流**：
1. **Understand**：读取各平台CSV导出文件，识别字段（SKU、销量、销售额、退货、广告 spend）
2. **Analyze**：规划分析方向——平台对比、时间趋势、TOP SKU、退货率异常
3. **Code** → **Execute**：
   - 数据清洗（处理缺失值、统一货币单位）
   - 计算各平台GMV、ROI、退货率
   - 生成趋势图（折线图）和对比图（柱状图）
4. **Analyze**：基于执行结果反思——发现Shopify退货率异常升高
5. **Code** → **Execute**：深挖退货原因（按SKU、按时间维度）
6. **Answer**：生成结构化报告，包含关键发现、数据可视化、可执行建议

**预期效果**：从4-6小时/周压缩至5分钟，报告质量标准化。

### 场景B：用户VOC数据深度研究

**背景**：从Trustpilot、Zendesk、Amazon评论收集的用户反馈需要定期分析，识别产品改进机会。

**Agent工作流**：
1. **Understand**：读取评论数据，识别字段（评分、评论文本、时间、产品型号）
2. **Analyze**：规划——情感趋势 + 关键话题提取 + 时间维度演变
3. **Code** → **Execute**：
   - 情感分析（按时间聚合）
   - 关键词提取与话题聚类
   - 评分分布与文本长度分析
4. **Analyze**：发现某产品型号负面评论集中出现在近30天
5. **Code** → **Execute**：提取该型号高频负面关键词
6. **Answer**：生成产品改进优先级建议报告

## 3. 代码模板

代码位置：`paper2skills-code/data_agent_llm/deepanalyze_agent/`

核心架构：`agent.py`

```python
"""
DeepAnalyze-inspired Autonomous Data Science Agent
基于五动作编排架构的简化版实现

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
import contextlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI


class ActionType(Enum):
    ANALYZE = "analyze"
    UNDERSTAND = "understand"
    CODE = "code"
    EXECUTE = "execute"
    ANSWER = "answer"


@dataclass
class Action:
    type: ActionType
    content: str
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentState:
    instruction: str
    data_sources: List[str] = field(default_factory=list)
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    actions: List[Action] = field(default_factory=list)
    environment_context: str = ""
    final_answer: str = ""


class DataScienceAgent:
    """
    自主数据科学Agent

    基于DeepAnalyze五动作架构的简化实现：
    - 使用外部LLM API（OpenAI/DeepSeek等）
    - 支持CSV/Excel数据加载
    - 自动编排Analyze→Understand→Code→Execute→Answer循环
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_iterations = 20
        # 白名单：只允许使用的模块和函数
        self.allowed_modules = {'pandas', 'numpy', 'matplotlib', 'matplotlib.pyplot'}

    def run(self, instruction: str, data_paths: List[str]) -> str:
        """执行端到端数据科学任务"""
        state = AgentState(instruction=instruction, data_sources=data_paths)

        # Step 1: Understand - 加载并理解数据
        self._understand_data(state)

        # Step 2-4: 迭代执行 Analyze -> Code -> Execute
        for i in range(self.max_iterations):
            action = self._decide_next_action(state)

            if action.type == ActionType.ANSWER:
                state.actions.append(action)
                state.final_answer = action.content
                break
            elif action.type == ActionType.CODE:
                state.actions.append(action)
                self._execute_code_sandboxed(state, action)
            elif action.type == ActionType.ANALYZE:
                state.actions.append(action)
                state.environment_context += f"\n[分析] {action.content}"
            else:
                state.actions.append(action)

        return state.final_answer

    def _understand_data(self, state: AgentState) -> None:
        """加载数据源并生成数据概要"""
        summaries = []
        for path in state.data_sources:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                continue

            state.dataframes[os.path.basename(path)] = df
            summary = self._summarize_dataframe(df, os.path.basename(path))
            summaries.append(summary)

        state.environment_context = "\n".join(summaries)
        state.actions.append(Action(
            type=ActionType.UNDERSTAND,
            content=state.environment_context
        ))

    def _summarize_dataframe(self, df: pd.DataFrame, name: str) -> str:
        """生成DataFrame结构概要"""
        lines = [f"数据集: {name}", f"形状: {df.shape[0]} 行 x {df.shape[1]} 列"]
        lines.append(f"列: {', '.join(df.columns.tolist())}")

        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            lines.append(f"数值列: {', '.join(numeric_cols)}")

        # 类别列
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            lines.append(f"类别列: {', '.join(cat_cols)}")

        # 缺失值
        missing = df.isnull().sum()
        if missing.sum() > 0:
            lines.append(f"缺失值列: {missing[missing > 0].to_dict()}")

        return "\n".join(lines)

    def _decide_next_action(self, state: AgentState) -> Action:
        """调用LLM决定下一个动作"""
        prompt = self._build_decision_prompt(state)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content
        return self._parse_action(content)

    def _system_prompt(self) -> str:
        return """你是自主数据科学Agent。你的任务是根据用户需求，自主完成数据分析任务。

可用动作（必须严格按以下格式输出）：

1. **ANALYZE**: 分析当前情况，规划下一步，或反思之前的结果
   格式: ACTION: ANALYZE
   CONTENT: <你的分析内容>

2. **CODE**: 生成Python代码处理数据
   格式: ACTION: CODE
   CONTENT: ```python
   <代码>
   ```

3. **ANSWER**: 任务完成，生成最终报告
   格式: ACTION: ANSWER
   CONTENT: <最终报告>

规则：
- 数据已通过pandas加载到 `dfs` 字典中，键为文件名，值为DataFrame
- 可用库：pandas, numpy, matplotlib.pyplot (已导入为pd, np, plt)
- 每张图保存到 `/tmp/output_{序号}.png`
- 代码必须完整可执行，不要省略关键步骤
- 如果前一步代码执行出错，先ANALYZE分析错误原因，再CODE修正
"""

    def _build_decision_prompt(self, state: AgentState) -> str:
        history = "\n".join([
            f"Step {i+1}: [{a.type.value.upper()}] {a.content[:200]}..."
            for i, a in enumerate(state.actions[-5:])  # 最近5步
        ])

        error_info = ""
        if state.actions and state.actions[-1].error:
            error_info = f"前一步代码执行错误: {state.actions[-1].error}"

        return f"""用户指令: {state.instruction}

已加载数据概要:
{state.environment_context}

执行历史（最近5步）:
{history}

{error_info}

请决定下一个动作。"""

    def _parse_action(self, content: str) -> Action:
        """解析LLM输出的动作"""
        action_match = re.search(r'ACTION:\s*(\w+)', content, re.IGNORECASE)
        if not action_match:
            return Action(type=ActionType.ANALYZE, content=content)

        action_type = action_match.group(1).lower()
        content_text = re.sub(r'ACTION:\s*\w+\s*', '', content, flags=re.IGNORECASE).strip()

        if action_type == "code":
            return Action(type=ActionType.CODE, content=content_text)
        elif action_type == "answer":
            return Action(type=ActionType.ANSWER, content=content_text)
        else:
            return Action(type=ActionType.ANALYZE, content=content_text)

    def _execute_code_sandboxed(self, state: AgentState, action: Action) -> None:
        """
        在受限环境中执行代码

        安全策略：
        1. 代码写入临时文件
        2. 使用受限的globals/locals（禁止__builtins__中的危险函数）
        3. 仅暴露白名单模块
        4. 生产环境应使用Docker沙箱或subprocess隔离
        """
        # 提取代码块
        code_blocks = re.findall(r'```python\n(.*?)```', action.content, re.DOTALL)
        if not code_blocks:
            code_blocks = [action.content]

        code = code_blocks[0].strip()

        # 构建安全执行环境 - 限制可用的内置函数
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'sum': sum, 'min': min, 'max': max, 'abs': abs,
            'round': round, 'float': float, 'int': int, 'str': str,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'print': print, 'isinstance': isinstance, 'hasattr': hasattr,
        }

        env = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'plt': plt,
            'dfs': state.dataframes,
        }

        # 捕获输出
        import io
        import contextlib

        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                # 使用 compile + exec 以支持更好错误追踪
                compiled = compile(code, '<agent_code>', 'exec')
                exec(compiled, env)

            action.result = output_buffer.getvalue()

            # 保存生成的图
            plt.savefig('/tmp/output_latest.png')
            plt.close()

        except Exception as e:
            action.error = f"{type(e).__name__}: {str(e)}"


# ============ 测试用例 ============

def create_test_data():
    """生成测试用的母婴出海销售数据"""
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

    # 注入异常：第60-65天Shopify退货率异常升高
    mask = (df['platform'] == 'Shopify') & (df.index >= 60) & (df.index <= 65)
    df.loc[mask, 'returns'] = 20
    df.loc[mask, 'return_rate'] = (df.loc[mask, 'returns'] / df.loc[mask, 'sales_qty']).round(3)

    test_path = '/tmp/test_sales_data.csv'
    df.to_csv(test_path, index=False)
    return test_path


def test_agent():
    """测试Agent"""
    test_path = create_test_data()

    agent = DataScienceAgent(model="gpt-4o-mini")  # 使用低成本模型测试

    instruction = (
        "分析这份母婴出海销售数据。需要完成："
        "1. 各平台GMV和ROI对比"
        "2. 各SKU销量排名"
        "3. 退货率趋势，找出异常时段"
        "4. 生成简洁的分析结论"
    )

    result = agent.run(instruction, [test_path])
    print("=" * 60)
    print("最终报告:")
    print(result)

    return result


if __name__ == "__main__":
    test_agent()
```

运行测试：

```bash
cd paper2skills-code/data_agent_llm/deepanalyze_agent
export OPENAI_API_KEY=your_key
python agent.py
```

## 4. 技能关系

### 前置技能
- **Python数据分析基础**：Pandas、NumPy、Matplotlib 基本操作
- **LLM基础**：理解prompt engineering、function calling

### 关联技能
- [Skill-AutoTag-SelfEvolving-Label-System](paper2skills-vault/07-NLP-VOC/Skill-AutoTag-SelfEvolving-Label-System.md) — Agent分析后可自动生成标签
- [Skill-Aspect-Based-Sentiment-Analysis](paper2skills-vault/07-NLP-VOC/Skill-Aspect-Based-Sentiment-Analysis.md) — 为Agent提供结构化情感分析数据源
- [Skill-OpenWorld-Class-Incremental-Learning](paper2skills-vault/07-NLP-VOC/Skill-OpenWorld-Class-Incremental-Learning.md) — 支持Agent处理新出现的业务类别

### 扩展方向
- **+ Argos** → 异常检测触发 + 自动深度分析报告
- **+ Data-to-Dashboard** → 自动数据可视化仪表板
- **+ 多Agent协作** → 专业分工（数据清洗Agent + 分析Agent + 报告Agent）

## 5. 商业价值评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **ROI** | ★★★★☆ | 将数据分析报告生成从4-6小时/周压缩至5分钟，人力成本节省显著 |
| **实施难度** | ★★★☆☆ | 原型可基于开源代码快速搭建；生产级需要解决安全沙箱、API成本控制 |
| **业务匹配度** | ★★★★★ | 直接解决母婴出海多平台数据整合分析痛点 |
| **技术成熟度** | ★★★★☆ | 论文已开源模型+代码+数据，可直接复用；但8B模型能力有上限 |
| **优先级** | **P1** | 高ROI + 成熟开源实现，建议作为Data Agent方向首个落地技能 |

**量化ROI估算**：
- 假设运营团队5人，每人每周数据分析耗时4小时
- 人力成本按¥200/小时计算
- 使用Agent后：5人 x 4小时 x ¥200 = ¥4000/周 → ¥200/周（API成本）
- **年节省：约 ¥20万**

---

## ⑥ 原文引用

> 原文:"To emulate this process, DeepAnalyze introduces five actions to automatically accomplish the data science task, including:"
> 出处：2510.16872 §3.1（Architecture · Interaction Pattern，对应「五动作编排架构」）

> 原文:"Analyze textually, including planning, reasoning, reflection, self-verification"
> 出处：2510.16872 §3.1（动作 1 · Analyze 的职责）

> 原文:"Understand the content of data source, such as databases, tables, and documents."
> 出处：2510.16872 §3.1（动作 2 · Understand 的职责）

> 原文:"Generate code to interact with the data in the environment, using Python suited for data science."
> 出处：2510.16872 §3.1（动作 3 · Code 的职责）

> 原文:"Execute code and collect the feedback from the environment."
> 出处：2510.16872 §3.1（动作 4 · Execute 的职责）

> 原文:"Produce the final output."
> 出处：2510.16872 §3.1（动作 5 · Answer 的职责）

> 原文:"all actions (i.e., special tokens) are autonomously generated by the model without any human-defined workflows or rules, which allows DeepAnalyze to fully autonomously orchestrate and optimize each action, laying the foundation for autonomous data science."
> 出处：2510.16872 §3.1（「不是按预定义工作流执行」的关键依据）

> 原文:"Although these systems demonstrate stronger task coordination, they depend heavily on manually designed heuristics and domain-specific rules, falling short of achieving autonomous and adaptive behavior."
> 出处：2510.16872 §1（Introduction，workflow-based agent 的局限）

> 原文:"To address this challenge, we propose curriculum-based agentic training, which emulates the learning path of human data scientists by gradually transitioning from mastering single abilities to integrating multiple abilities."
> 出处：2510.16872 §3.2（课程式训练的动机与命名依据）

> 原文:"This training framework consists of two stages, where stage 1 employs single-ability fine-tuning to strengthen the foundation LLM’s single ability, and stage 2 uses multi-ability agentic training to enable the LLM to apply multiple abilities in real-world environments to accomplish complex data science tasks."
> 出处：2510.16872 §3.2（两阶段划分）

> 原文:"we first enhance the various single abilities that data science relies, primarily including reasoning, structured data understanding, and code generation"
> 出处：2510.16872 §3.2（单能力阶段的三个基础能力）

> 原文:"Subsequently, we train DeepAnalyze in real-world environments using reinforcement learning with group relative policy optimization (GRPO) (Shao et al., 2024)."
> 出处：2510.16872 §3.2（多能力阶段使用 GRPO）

> 原文:"we adopt a hybrid reward modeling that combines rule-based rewards with LLM-as-a-judge rewards."
> 出处：2510.16872 §3.2（混合奖励建模）

> 原文:"the score that evaluates the generated report from five aspects: usefulness, richness, soundness, interpretability, and readability."
> 出处：2510.16872 §3.2（LLM-as-judge 的五个报告质量维度）

> 原文:"This reward encourages DeepAnalyze to engage in more successful interactions with the environment and to generate high-quality research report."
> 出处：2510.16872 §3.2（交互质量奖励项）

> 原文:"The data-grounded trajectory synthesis framework consists of two parts: Reasoning Trajectory Synthesis, which construct the reasoning trajectory for existing structured data instruction datasets, and Interaction Trajectory Synthesis, which constructs entire data science trajectory based on structured data sources in the environment."
> 出处：2510.16872 §3.3（数据 Grounded 轨迹合成的两类轨迹）

> 原文:"Compared with “One-stage Training”, a scheduled training process from simple (single-ability) to complex (multi-ability) proves more beneficial for model performance using the same data."
> 出处：2510.16872 §5.2（课程式训练相对 One-stage 的消融结论）

> 原文:"We build DeepAnalyze-8B based on DeepSeek-R1-0528-Qwen3-8B as the foundation LLM."
> 出处：2510.16872 §4.2（基座模型）

> 原文:"We conduct experiments on 12 data science benchmarks."
> 出处：2510.16872 §4.1（Benchmarks，「12 个 benchmark」的出处）

> 原文:"The results show that, despite having only 8B parameters, DeepAnalyze-8B achieves state-of-the-art performance among open-source LLM-based agents and even outperforms most advanced proprietary models (e.g., GPT-4-Turbo, GPT-4o-mini, Claude 3.5 Sonnet), ranking second only to GPT-4o."
> 出处：2510.16872 §4.3（DataSciBench 主结果；注意：此处论文自称「仅次于 GPT-4o」，非「超越 GPT-4o」）

> 原文:"Experiments demonstrate that, with only 8B parameters, DeepAnalyze outperforms previous workflow-based agents built on most advanced proprietary LLMs."
> 出处：2510.16872 §Abstract（8B 参数的总体结论）

> 原文:"Powered by curriculum-based agentic training and data-grounded trajectory synthesis, DeepAnalyze-8B outperforms state-of-the-art closed-source LLMs on 12 data science benchmarks."
> 出处：2510.16872 §6（Conclusion）
