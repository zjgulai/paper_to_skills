"""
Argos-inspired Agentic Time-Series Anomaly Detection
基于三Agent协作的简化版实现

论文: ARGOS: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via LLMs
arXiv: 2501.14170

⚠️ 安全警告：本原型使用 exec() 编译并执行LLM生成的代码。
生产环境必须使用 Docker 沙箱或 RestrictedPython 限制执行环境。
"""

import os
import re
import io
import contextlib
import textwrap
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openai import OpenAI


@dataclass
class DetectionRule:
    """异常检测规则"""
    code: str
    description: str
    f1_score: float = 0.0
    is_valid: bool = False


class ArgosAnomalyDetector:
    """
    Agentic时序异常检测器
    三Agent协作：Detection Agent -> Repair Agent -> Review Agent
    """

    CODE_TEMPLATE = '''import numpy as np

def inference(sample: np.ndarray, threshold: float) -> np.ndarray:
    """检测时序数据中的异常点"""
    values = sample[:, 0]
    labels = np.zeros(len(values), dtype=int)
    {rule_code}
    return labels
'''

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_repair_iterations = 3
        self.max_review_iterations = 3
        self.k_best = 3

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray,
            val_data: np.ndarray, val_labels: np.ndarray) -> DetectionRule:
        """训练异常检测规则"""
        candidate_rules = self._detection_agent_generate(train_data, train_labels, n_candidates=self.k_best)
        valid_rules = []
        for rule in candidate_rules:
            repaired = self._repair_agent_fix(rule, train_data)
            if repaired.is_valid:
                valid_rules.append(repaired)
        if not valid_rules:
            raise RuntimeError("所有候选规则均无法通过语法检查")

        best_rule = None
        best_f1 = 0.0
        for rule in valid_rules:
            improved_rule = self._review_agent_evaluate_and_improve(rule, val_data, val_labels)
            if improved_rule.f1_score > best_f1:
                best_f1 = improved_rule.f1_score
                best_rule = improved_rule
        return best_rule

    def predict(self, data: np.ndarray, rule: DetectionRule) -> np.ndarray:
        detector = self._compile_rule(rule.code)
        return detector(data, threshold=2.0)

    def _detection_agent_generate(self, data: np.ndarray, labels: np.ndarray,
                                   n_candidates: int = 3) -> List[DetectionRule]:
        normal_samples = data[labels == 0][:10]
        anomaly_samples = data[labels == 1][:5]
        prompt = f"""你是异常检测专家。基于以下时序数据特征，生成Python异常检测规则。
数据特征:
- 正常样本统计: 均值={np.mean(normal_samples[:,0]):.2f}, 标准差={np.std(normal_samples[:,0]):.2f}
- 异常样本统计: 均值={np.mean(anomaly_samples[:,0]):.2f}, 标准差={np.std(anomaly_samples[:,0]):.2f}
- 数据点总数: {len(data)}, 异常率: {np.mean(labels)*100:.1f}%
要求:
1. 只写 `def inference` 函数体内的代码（不需要import和函数签名）
2. 规则必须可解释，添加中文注释说明逻辑
3. 考虑多种异常模式：突增、突降、持续偏离
4. 返回numpy数组 labels
只输出代码，不要其他解释。"""

        rules = []
        for i in range(n_candidates):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的时序异常检测工程师。"},
                    {"role": "user", "content": prompt + f"\n\n这是第{i+1}个不同角度的规则："}
                ],
                temperature=0.5 + i * 0.2
            )
            code = response.choices[0].message.content
            code_blocks = re.findall(r'```python\n(.*?)```', code, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
            code = code.strip()
            # 给函数体代码添加缩进，使其与模板匹配
            indented_code = textwrap.indent(code, '    ')
            rules.append(DetectionRule(
                code=self.CODE_TEMPLATE.format(rule_code=indented_code),
                description=f"Candidate rule {i+1}"
            ))
        return rules

    def _repair_agent_fix(self, rule: DetectionRule, test_data: np.ndarray) -> DetectionRule:
        code = rule.code
        for attempt in range(self.max_repair_iterations):
            detector = self._compile_rule(code)
            try:
                dummy = test_data[:min(10, len(test_data))]
                result = detector(dummy, threshold=2.0)
                if not isinstance(result, np.ndarray):
                    raise TypeError(f"返回值类型错误: {type(result)}")
                if len(result) != len(dummy):
                    raise ValueError(f"返回长度不匹配")
                return DetectionRule(code=code, description=rule.description, is_valid=True)
            except Exception as e:
                if attempt == self.max_repair_iterations - 1:
                    return DetectionRule(code=code, description=rule.description, is_valid=False)
                fix_prompt = f"""以下Python代码执行时报错：\n错误: {type(e).__name__}: {str(e)}\n代码:\n```python\n{code}\n```\n请修正代码中的错误，只输出修正后的完整代码。"""
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是Python代码调试专家。"},
                        {"role": "user", "content": fix_prompt}
                    ],
                    temperature=0.1
                )
                fixed_code = response.choices[0].message.content
                code_blocks = re.findall(r'```python\n(.*?)```', fixed_code, re.DOTALL)
                code = code_blocks[0] if code_blocks else fixed_code
        return DetectionRule(code=code, description=rule.description, is_valid=False)

    def _review_agent_evaluate_and_improve(self, rule: DetectionRule,
                                            val_data: np.ndarray,
                                            val_labels: np.ndarray) -> DetectionRule:
        detector = self._compile_rule(rule.code)
        current_code = rule.code
        for attempt in range(self.max_review_iterations):
            predictions = detector(val_data, threshold=2.0)
            f1 = self._calculate_f1(val_labels, predictions)
            if f1 >= 0.8:
                return DetectionRule(code=current_code, description=rule.description, f1_score=f1, is_valid=True)
            fp_mask = (predictions == 1) & (val_labels == 0)
            fn_mask = (predictions == 0) & (val_labels == 1)
            feedback = f"""当前规则F1={f1:.3f}，需要改进。\n错误分析:\n- 误报 (FP): {fp_mask.sum()} 个\n- 漏报 (FN): {fn_mask.sum()} 个\n请基于以上反馈改进异常检测逻辑。只输出改进后的代码。"""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是异常检测优化专家。"},
                    {"role": "user", "content": feedback + f"\n\n当前代码:\n```python\n{current_code}\n```"}
                ],
                temperature=0.2
            )
            improved = response.choices[0].message.content
            code_blocks = re.findall(r'```python\n(.*?)```', improved, re.DOTALL)
            current_code = code_blocks[0] if code_blocks else improved

        predictions = detector(val_data, threshold=2.0)
        final_f1 = self._calculate_f1(val_labels, predictions)
        return DetectionRule(code=current_code, description=rule.description, f1_score=final_f1, is_valid=True)

    def _compile_rule(self, code: str) -> Callable:
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter,
            'sum': sum, 'min': min, 'max': max, 'abs': abs,
            'round': round, 'float': float, 'int': int, 'str': str,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'print': print, 'isinstance': isinstance,
            '__import__': __import__,  # 允许导入（原型环境）
        }
        local_ns = {'np': np, '__builtins__': safe_builtins}
        exec(code, local_ns)
        return local_ns['inference']

    @staticmethod
    def _calculate_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def create_synthetic_ts(n: int = 200, anomaly_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    t = np.arange(n)
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 2, n)
    values = 50 + trend + seasonal + noise
    labels = np.zeros(n, dtype=int)
    n_anomalies = int(n * anomaly_ratio)
    anomaly_indices = np.random.choice(n, n_anomalies, replace=False)
    for idx in anomaly_indices:
        if np.random.rand() > 0.5:
            values[idx] += np.random.uniform(20, 40)
        else:
            values[idx] -= np.random.uniform(20, 40)
        labels[idx] = 1
    return values.reshape(-1, 1), labels


def test_argos():
    print("生成合成时序数据...")
    data, labels = create_synthetic_ts(n=200, anomaly_ratio=0.05)
    split = int(0.7 * len(data))
    train_data, val_data = data[:split], data[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    print(f"训练集: {len(train_data)} 样本, 异常率: {train_labels.mean()*100:.1f}%")
    print(f"验证集: {len(val_data)} 样本, 异常率: {val_labels.mean()*100:.1f}%")
    detector = ArgosAnomalyDetector(model="gpt-4o-mini")
    print("\n开始训练（Detection -> Repair -> Review）...")
    try:
        rule = detector.fit(train_data, train_labels, val_data, val_labels)
        print(f"\n最佳规则F1: {rule.f1_score:.3f}")
        predictions = detector.predict(val_data, rule)
        final_f1 = ArgosAnomalyDetector._calculate_f1(val_labels, predictions)
        print(f"验证集最终F1: {final_f1:.3f}")
    except Exception as e:
        print(f"训练失败: {e}")


if __name__ == "__main__":
    test_argos()
