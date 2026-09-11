---
title: Argos — Agentic时序异常检测
name: Skill-Argos-Agentic-Anomaly-Detection
description: 利用LLM自主生成可解释、可复现的时序异常检测规则，通过三Agent协作确保准确率与效率
module: data-agent-llm
topic: agentic-time-series-anomaly-detection
version: 0.1.0
status: stable
created: 2026-04-26
updated: 2026-04-26
paper: arXiv:2501.14170
source: ai
---

# Argos — Agentic时序异常检测

## 1. 算法原理

Argos 解决的核心矛盾是：**LLM能生成异常检测规则，但无法同时保证可解释性、可复现性和准确率**。

传统方法的缺陷：
- **深度学习模型**（FCVAE、LSTMAD）：黑盒，不可解释
- **人工规则**：可解释但无法自适应，维护成本高
- **直接LLM生成**：输出方差大，8.8%语法错误，无准确率保证

**Argos 三阶段架构**：

```
+--------------+    +------------------------------+    +--------------+
| 数据预处理    | -> | 规则训练（三Agent协作）        | -> |   部署       |
|              |    |                              |    |              |
| • 缩放索引   |    |  Detection Agent ----> 生成规则 |    | Anomaly      |
| • 分块      |    |       |                        |    | Detector     |
| • Tokenizer |    |  Repair Agent <---- 修正语法    |    |  +           |
|   预处理    |    |       |                        |    | Aggregator   |
|              |    |  Review Agent <---- 评估准确率  |    | (模型融合)   |
+--------------+    +------------------------------+    +--------------+
```

**三Agent协作机制**（核心创新）：

1. **Detection Agent**：接收时序数据样本和ground-truth标签，用代码模板生成Python异常检测规则
   - 输出格式：`def inference(sample: np.ndarray, threshold: float) -> np.ndarray`
   - 规则包含自然语言注释解释逻辑

2. **Repair Agent**：检查规则语法错误，用dummy data执行验证，自动修正
   - 将正确率从~40%提升到~95%

3. **Review Agent**：在验证数据上评估规则F1分数，与上一轮对比
   - 如果准确率下降，提出改进建议反馈给Detection Agent
   - 迭代直到收敛

**准确率保证机制**：

通过Aggregator将LLM生成规则与已有成熟检测器（如统计模型）融合：
- **为什么分两套规则？** 单一规则容易偏向——只优化漏报会牺牲误报，反之亦然。分别从false negatives（漏检样本）和false positives（误报样本）中提取对比样本，训练出互补的两套规则，类似集成学习中"互补基学习器"的思想。
- 融合后保证准确率不低于基线检测器
- 无Aggregator时KPI数据集准确率下降3.5%，有Aggregator时无下降

**效率增强**：

Top-k规则选择：每次Detection Agent提出多条候选规则，只选最优k条，避免LLM方差问题。

## 2. 业务应用

### 场景A：跨境电商平台销售异常实时监控

**背景**：母婴品牌在Amazon US、Amazon EU、Shopify、SHEIN四平台销售，需要监控每日销量、广告ROI、退货率等指标异常。现有阈值规则误报率高，深度学习模型不可解释。

**Argos应用**：
1. 加载历史90天销售数据（分平台、分SKU）
2. Detection Agent为每个指标生成异常检测规则：
   ```python
   # 示例生成规则（可解释）
   def inference(sample, threshold):
       # 规则：如果销量连续3天低于过去7天均值的threshold倍，标记异常
       values = sample[:, 0]
       labels = np.zeros(len(values))
       for i in range(3, len(values)):
           avg_7d = np.mean(values[i-7:i])
           if values[i] < threshold * avg_7d and values[i-1] < threshold * avg_7d and values[i-2] < threshold * avg_7d:
               labels[i] = 1
       return labels
   ```
3. Repair Agent验证语法，Review Agent在验证集上评估
4. Aggregator融合已有统计基线，保证不误报
5. 部署后实时检测，异常触发告警

**预期效果**：F1提升9.5%~28.3%，推理速度提升3x~34x，规则可解释（业务人员能看懂触发条件）。

### 场景B：供应链物流时效异常预警

**背景**：跨境物流涉及头程、清关、尾程多个环节，时效波动大。需要自动检测物流延迟异常，提前预警补货。

**Argos应用**：
- 监控指标：头程时效、清关时长、尾程配送时效、库存水位
- 自动识别季节性模式（如黑五期间正常延迟 vs 真正的异常延迟）
- 为每个物流线路生成自适应规则
- 异常时自动触发补货建议

**预期效果**：自动适应不同线路、不同季节模式，减少人工调参。

## 3. 代码模板

代码位置：`paper2skills-code/data_agent_llm/argos_agent/`

核心实现：`anomaly_detector.py`

```python
"""
Argos-inspired Agentic Time-Series Anomaly Detection
基于三Agent协作的简化版实现

⚠️ 安全警告：本原型使用 exec() 编译并执行LLM生成的代码。
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

    三Agent协作：
    1. Detection Agent: 生成异常检测规则
    2. Repair Agent: 修正语法错误
    3. Review Agent: 评估准确率
    """

    CODE_TEMPLATE = '''import numpy as np

def inference(sample: np.ndarray, threshold: float) -> np.ndarray:
    """
    检测时序数据中的异常点

    Args:
        sample: 时序样本，shape (n_samples, n_features)
        threshold: 异常检测阈值

    Returns:
        labels: 异常标签，1表示异常，0表示正常，shape (n_samples,)
    """
    values = sample[:, 0]  # 取第一个特征
    labels = np.zeros(len(values), dtype=int)

    # === 你的异常检测逻辑 ===
    # 提示：基于均值、标准差、滑动窗口等统计量设计规则
    # 确保规则可解释（添加注释说明逻辑）

    {rule_code}

    return labels
'''

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_repair_iterations = 3
        self.max_review_iterations = 3
        self.k_best = 3  # top-k规则选择

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray,
            val_data: np.ndarray, val_labels: np.ndarray) -> DetectionRule:
        """
        训练异常检测规则

        Args:
            train_data: 训练时序数据 (n_samples, n_features)
            train_labels: 训练标签 (n_samples,)
            val_data: 验证时序数据
            val_labels: 验证标签
        """
        # Step 1: Detection Agent 生成候选规则
        candidate_rules = self._detection_agent_generate(
            train_data, train_labels, n_candidates=self.k_best
        )

        # Step 2: Repair Agent 修正语法
        valid_rules = []
        for rule in candidate_rules:
            repaired = self._repair_agent_fix(rule, train_data)
            if repaired.is_valid:
                valid_rules.append(repaired)

        if not valid_rules:
            raise RuntimeError("所有候选规则均无法通过语法检查")

        # Step 3: Review Agent 评估并迭代优化
        best_rule = None
        best_f1 = 0.0

        for rule in valid_rules:
            improved_rule = self._review_agent_evaluate_and_improve(
                rule, val_data, val_labels
            )
            if improved_rule.f1_score > best_f1:
                best_f1 = improved_rule.f1_score
                best_rule = improved_rule

        return best_rule

    def predict(self, data: np.ndarray, rule: DetectionRule) -> np.ndarray:
        """使用训练好的规则预测异常"""
        detector = self._compile_rule(rule.code)
        # 使用默认threshold=2.0，实际应用中可调整
        return detector(data, threshold=2.0)

    def _detection_agent_generate(self, data: np.ndarray, labels: np.ndarray,
                                   n_candidates: int = 3) -> List[DetectionRule]:
        """Detection Agent: 生成异常检测规则"""

        # 数据特征描述
        normal_samples = data[labels == 0][:10]
        anomaly_samples = data[labels == 1][:5]

        prompt = f"""你是异常检测专家。基于以下时序数据特征，生成Python异常检测规则。

数据特征:
- 正常样本统计: 均值={np.mean(normal_samples[:,0]):.2f}, 标准差={np.std(normal_samples[:,0]):.2f}
- 异常样本统计: 均值={np.mean(anomaly_samples[:,0]):.2f}, 标准差={np.std(anomaly_samples[:,0]):.2f}
- 数据点总数: {len(data)}
- 异常率: {np.mean(labels)*100:.1f}%

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
                temperature=0.5 + i * 0.2  # 不同温度生成多样化规则
            )

            code = response.choices[0].message.content
            # 提取代码块
            code_blocks = re.findall(r'```python\n(.*?)```', code, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]

            # 清理代码
            code = code.strip()
            # 给函数体代码添加缩进，使其与模板匹配
            indented_code = textwrap.indent(code, '    ')

            rules.append(DetectionRule(
                code=self.CODE_TEMPLATE.format(rule_code=indented_code),
                description=f"Candidate rule {i+1}"
            ))

        return rules

    def _repair_agent_fix(self, rule: DetectionRule, test_data: np.ndarray) -> DetectionRule:
        """Repair Agent: 检查并修正语法错误"""
        code = rule.code

        for attempt in range(self.max_repair_iterations):
            # 尝试执行
            detector = self._compile_rule(code)

            try:
                # 用dummy data测试
                dummy = test_data[:min(10, len(test_data))]
                result = detector(dummy, threshold=2.0)

                # 检查输出格式
                if not isinstance(result, np.ndarray):
                    raise TypeError(f"返回值类型错误: {type(result)}")
                if len(result) != len(dummy):
                    raise ValueError(f"返回长度不匹配: {len(result)} vs {len(dummy)}")

                # 成功
                return DetectionRule(
                    code=code,
                    description=rule.description,
                    is_valid=True
                )

            except Exception as e:
                if attempt == self.max_repair_iterations - 1:
                    return DetectionRule(code=code, description=rule.description, is_valid=False)

                # 请求LLM修复
                fix_prompt = f"""以下Python代码执行时报错：

错误: {type(e).__name__}: {str(e)}

代码:
```python
{code}
```

请修正代码中的错误，只输出修正后的完整代码。"""

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
                if code_blocks:
                    code = code_blocks[0]
                else:
                    code = fixed_code

        return DetectionRule(code=code, description=rule.description, is_valid=False)

    def _review_agent_evaluate_and_improve(self, rule: DetectionRule,
                                            val_data: np.ndarray,
                                            val_labels: np.ndarray) -> DetectionRule:
        """Review Agent: 评估规则准确率并迭代优化"""
        detector = self._compile_rule(rule.code)
        current_code = rule.code

        for attempt in range(self.max_review_iterations):
            # 评估
            predictions = detector(val_data, threshold=2.0)
            f1 = self._calculate_f1(val_labels, predictions)

            if f1 >= 0.8:  # 满意阈值
                return DetectionRule(
                    code=current_code,
                    description=rule.description,
                    f1_score=f1,
                    is_valid=True
                )

            # 分析错误
            fp_mask = (predictions == 1) & (val_labels == 0)
            fn_mask = (predictions == 0) & (val_labels == 1)

            feedback = f"""当前规则F1={f1:.3f}，需要改进。

错误分析:
- 误报 (FP): {fp_mask.sum()} 个
- 漏报 (FN): {fn_mask.sum()} 个

请基于以上反馈改进异常检测逻辑。只输出改进后的代码。"""

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
            if code_blocks:
                current_code = code_blocks[0]
            else:
                current_code = improved

        # 返回最佳结果
        predictions = detector(val_data, threshold=2.0)
        final_f1 = self._calculate_f1(val_labels, predictions)

        return DetectionRule(
            code=current_code,
            description=rule.description,
            f1_score=final_f1,
            is_valid=True
        )

    def _compile_rule(self, code: str) -> Callable:
        """编译规则代码为可调用函数"""
        # 安全限制：只暴露必要的内置函数
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
        """计算F1分数"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


# ============ 测试用例 ============

def create_synthetic_ts(n: int = 200, anomaly_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成时序数据（含异常）"""
    np.random.seed(42)

    # 基础趋势 + 季节性
    t = np.arange(n)
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 2, n)
    values = 50 + trend + seasonal + noise

    # 注入异常
    labels = np.zeros(n, dtype=int)
    n_anomalies = int(n * anomaly_ratio)
    anomaly_indices = np.random.choice(n, n_anomalies, replace=False)

    for idx in anomaly_indices:
        if np.random.rand() > 0.5:
            values[idx] += np.random.uniform(20, 40)  # 突增
        else:
            values[idx] -= np.random.uniform(20, 40)  # 突降
        labels[idx] = 1

    # 构造 (n_samples, 1) 格式
    data = values.reshape(-1, 1)
    return data, labels


def test_argos():
    """测试Argos异常检测器"""
    print("生成合成时序数据...")
    data, labels = create_synthetic_ts(n=200, anomaly_ratio=0.05)

    # 划分训练/验证集
    split = int(0.7 * len(data))
    train_data, val_data = data[:split], data[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    print(f"训练集: {len(train_data)} 样本, 异常率: {train_labels.mean()*100:.1f}%")
    print(f"验证集: {len(val_data)} 样本, 异常率: {val_labels.mean()*100:.1f}%")

    # 训练
    detector = ArgosAnomalyDetector(model="gpt-4o-mini")
    print("\n开始训练（Detection -> Repair -> Review）...")

    try:
        rule = detector.fit(train_data, train_labels, val_data, val_labels)
        print(f"\n最佳规则F1: {rule.f1_score:.3f}")
        print(f"规则描述: {rule.description}")

        # 测试预测
        predictions = detector.predict(val_data, rule)
        final_f1 = ArgosAnomalyDetector._calculate_f1(val_labels, predictions)
        print(f"验证集最终F1: {final_f1:.3f}")

    except Exception as e:
        print(f"训练失败: {e}")


if __name__ == "__main__":
    test_argos()
```

运行测试：

```bash
cd paper2skills-code/data_agent_llm/argos_agent
export OPENAI_API_KEY=your_key
python anomaly_detector.py
```

## 4. 技能关系

### 前置技能
- **时序分析基础**：理解趋势、季节性、异常值概念
- **Python代码执行安全**：了解`exec()`的风险与沙箱方案

### 关联技能
- [Skill-AdaNEN-Streaming-Classifier](paper2skills-vault/07-NLP-VOC/Skill-AdaNEN-Streaming-Classifier.md) — 流式分类与实时异常检测结合
- [Skill-ALCHEmist-Weak-Supervision](paper2skills-vault/07-NLP-VOC/Skill-ALCHEmist-Weak-Supervision.md) — 弱监督标签生成可作为Argos的ground-truth来源
- [Skill-DeepAnalyze-Autonomous-Data-Science-Agent](paper2skills-vault/09-DataAgent-LLM/Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md) — 异常检测触发后自动生成深度分析报告

### 扩展方向
- **+ 根因分析Agent** → 检测到异常后自动归因（如"退货率异常升高 -> 某SKU质量问题"）
- **+ 多模态异常检测** → 结合文本评论情绪变化检测隐性异常
- **+ 自动阈值调优** → 基于业务反馈自动调整threshold参数

## 5. 商业价值评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **ROI** | ★★★★☆ | 减少人工规则维护成本（1-2人天/月），提升异常检测准确率直接减少损失 |
| **实施难度** | ★★★☆☆ | 原型搭建简单；生产部署需解决LLM API成本、实时性、沙箱安全 |
| **业务匹配度** | ★★★★★ | 跨境电商天然需要监控多平台多指标，Argos的自适应特性完美匹配 |
| **技术成熟度** | ★★★☆☆ | 方法论清晰但论文未明确开源代码，需自行实现核心逻辑 |
| **优先级** | **P1** | 异常检测是数据监控的基础设施，建议与DeepAnalyze组合落地 |

**量化ROI估算**：
- 假设监控20个核心指标，人工调参1人天/月
- 异常漏检导致库存断货或积压，单次损失¥5-10万
- Argos提升F1 9.5%~28.3%，直接减少漏检损失
- **年节省：人工 ¥3-5万 + 异常损失减少 ¥20-50万**
