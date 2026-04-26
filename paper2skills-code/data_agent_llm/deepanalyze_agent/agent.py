"""
DeepAnalyze-inspired Autonomous Data Science Agent
基于五动作编排架构的简化版实现

论文: DeepAnalyze: Agentic Large Language Models for Autonomous Data Science
arXiv: 2510.16872
开源: https://github.com/ruc-datalab/DeepAnalyze

⚠️ 安全警告：本原型使用 exec() 执行LLM生成的代码。
生产环境必须使用 Docker 沙箱或 RestrictedPython 限制执行环境。
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
    基于DeepAnalyze五动作架构的简化实现
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_iterations = 20

    def run(self, instruction: str, data_paths: List[str]) -> str:
        """执行端到端数据科学任务"""
        state = AgentState(instruction=instruction, data_sources=data_paths)
        self._understand_data(state)

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
        summaries = []
        for path in state.data_sources:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                continue
            state.dataframes[os.path.basename(path)] = df
            summaries.append(self._summarize_dataframe(df, os.path.basename(path)))
        state.environment_context = "\n".join(summaries)
        state.actions.append(Action(type=ActionType.UNDERSTAND, content=state.environment_context))

    def _summarize_dataframe(self, df: pd.DataFrame, name: str) -> str:
        lines = [f"数据集: {name}", f"形状: {df.shape[0]} 行 x {df.shape[1]} 列"]
        lines.append(f"列: {', '.join(df.columns.tolist())}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            lines.append(f"数值列: {', '.join(numeric_cols)}")
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            lines.append(f"类别列: {', '.join(cat_cols)}")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            lines.append(f"缺失值列: {missing[missing > 0].to_dict()}")
        return "\n".join(lines)

    def _decide_next_action(self, state: AgentState) -> Action:
        prompt = self._build_decision_prompt(state)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return self._parse_action(response.choices[0].message.content)

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
            for i, a in enumerate(state.actions[-5:])
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
        在受限环境中执行代码。
        ⚠️ 安全策略：限制内置函数，生产环境应使用Docker沙箱。
        """
        code_blocks = re.findall(r'```python\n(.*?)```', action.content, re.DOTALL)
        code = code_blocks[0].strip() if code_blocks else action.content.strip()

        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'sum': sum, 'min': min, 'max': max, 'abs': abs,
            'round': round, 'float': float, 'int': int, 'str': str,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'print': print, 'isinstance': isinstance, 'hasattr': hasattr,
        }
        env = {'__builtins__': safe_builtins, 'pd': pd, 'np': np, 'plt': plt, 'dfs': state.dataframes}

        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                compiled = compile(code, '<agent_code>', 'exec')
                exec(compiled, env)
            action.result = output_buffer.getvalue()
            plt.savefig('/tmp/output_latest.png')
            plt.close()
        except Exception as e:
            action.error = f"{type(e).__name__}: {str(e)}"


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
    mask = (df['platform'] == 'Shopify') & (df.index >= 60) & (df.index <= 65)
    df.loc[mask, 'returns'] = 20
    df.loc[mask, 'return_rate'] = (df.loc[mask, 'returns'] / df.loc[mask, 'sales_qty']).round(3)
    test_path = '/tmp/test_sales_data.csv'
    df.to_csv(test_path, index=False)
    return test_path


def test_agent():
    """测试Agent"""
    test_path = create_test_data()
    agent = DataScienceAgent(model="gpt-4o-mini")
    instruction = (
        "分析这份母婴出海销售数据。需要完成："
        "1. 各平台GMV和ROI对比 2. 各SKU销量排名 3. 退货率趋势，找出异常时段 4. 生成简洁的分析结论"
    )
    result = agent.run(instruction, [test_path])
    print("=" * 60)
    print("最终报告:")
    print(result)
    return result


if __name__ == "__main__":
    test_agent()
