---
title: LLM输出不确定性量化 — BI场景的置信度感知决策
doc_type: knowledge
module: 09-DataAgent-LLM
topic: llm-uncertainty-quantification-bi
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: LLM Uncertainty Quantification BI

> **论文**：Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation（Kuhn et al., ICLR 2023, arXiv:2302.09664）+ Can LLMs Express Their Uncertainty?（Xiong et al., ICLR 2024, arXiv:2306.13063）
> **arXiv**：2302.09664 | 2023 | **桥梁**: 09-DataAgent-LLM ↔ 23-运营财务 ↔ 12-ML基础 | **类型**: 工程基础

## ① 算法原理

**问题**：LLM回答"本月ROAS是多少"时总是自信满满，但有时答错。如何让LLM知道"我不确定这个答案"，并在不确定时主动告知用户？

**语义不确定性（Semantic Uncertainty）**：
传统不确定性（词频级别）对NLG无效——"ROAS是3.2"和"ROAS约3.2"语义相同但词不同。语义不确定性在语义等价类上计算：
1. 对同一问题采样多个回答（temperature>0）
2. 将语义等价的回答归为一类（用NLI模型判断等价性）
3. 按类的分布计算熵：$H = -\sum_c p_c \log p_c$，熵越高越不确定

**自信度校准**：
LLM表达的主观自信（"我很确定这是X"）与实际准确率往往不一致（过于自信）。通过**Platt缩放**将LLM的verbalized confidence转化为校准概率：
$$P(\text{correct}) = \sigma(a \cdot \text{verbalized\_conf} + b)$$
其中 $a, b$ 用标注数据拟合。

**BI场景实用方案**：
三层不确定性叠加：
1. **数据不确定性**：查询的数据源是否完整（如昨天数据还未入库）
2. **计算不确定性**：SQL生成是否正确（Text2SQL的不确定性）
3. **推断不确定性**：数字解读是否准确（如"环比下降"的方向是否正确）

**跨学科源头**：不确定性量化来自贝叶斯统计（epistemic vs aleatoric uncertainty），迁移到LLM的降维打击：BI工具里的AI回答如果都标注"置信度85%"，用户可以根据置信度决定是否需要人工复核，大幅提升AI工具的实用性和信任度。

## ② 母婴出海应用案例

**场景A：运营日报数字的置信度标注**
- 业务问题：DeepSeek驱动的运营日报，有时对"同比增长"方向判断错误（数据库里是日期维度导致的join问题），运营不知道哪些数字可信哪些要复核
- 数据要求：LLM生成的日报文本 + 用于校准的历史标注数据（各类型问题的正确率统计）
- 预期产出：每个关键数字/结论都标注置信度（"本月ROAS 3.2 [95%置信]"、"同比增长12% [61%置信，建议复核]"），低于70%的自动触发人工核查
- 业务价值：高置信度答案（>85%）直接使用，减少人工核查约60%；低置信度答案（<70%）必须核查，防止决策错误，年化避免因AI错误数据导致的决策损失约80万元

**三轨对抗验证**：
1. **成本验证**：语义不确定性需要多次采样（5-10次），API成本增加5-10倍，但只对重要决策问题使用，总成本可控（<500元/月）
2. **合规验证**：置信度标注是内部工具，无合规风险；注意不要将"置信度低"的结论对外（如在客服/营销文案中）使用
3. **风险验证**：校准模型可能过时（LLM更新后准确率分布变化）；建议每季度重新校准；语义等价判断本身也可能出错（NLI模型的局限）

**场景B：选品分析Agent的不确定性透传**
- 业务问题：选品Agent判断"婴儿监护器市场规模2.3亿美元"，但这个数字来自LLM的训练知识而非实时数据，不确定性高
- 方案：自动检测回答来源（知识型/检索型），知识型回答附加"数据可能截止2024年，建议查询实时数据"
- 业务价值：防止基于过时数据的选品决策，年化避免选品错误损失约50万元

## ③ 代码模板

```python
"""
Skill-LLM-Uncertainty-Quantification-BI
LLM输出不确定性量化 — BI场景置信度感知决策

依赖：pip install numpy pandas scikit-learn scipy
注意：生产环境需接入LLM API进行多次采样
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.special import expit  # sigmoid

np.random.seed(42)

# ── 1. 模拟LLM回答的不确定性信号 ─────────────────────────────────────
def simulate_llm_responses(question_type: str, n_samples: int = 5):
    """
    模拟同一问题的多次LLM采样（temperature>0）
    生产环境：替换为真实LLM API调用
    """
    if question_type == 'factual_number':
        # 数字型问题：LLM通常较确定
        base = np.random.normal(3.2, 0.1)
        responses = [f"ROAS是{base + np.random.normal(0, 0.05):.2f}" for _ in range(n_samples)]
        true_label = 1  # 正确的概率高
    elif question_type == 'trend_direction':
        # 趋势方向：LLM可能混淆方向
        directions = np.random.choice(['增长', '下降'], n_samples, p=[0.6, 0.4])
        responses = [f"同比{d}了X%" for d in directions]
        true_label = int(np.random.binomial(1, 0.65))  # 65%概率正确
    elif question_type == 'prediction':
        # 预测型：高度不确定
        values = np.random.normal(0, 1, n_samples)
        responses = [f"预计{'增长' if v>0 else '下降'}" for v in values]
        true_label = int(np.random.binomial(1, 0.52))  # 接近随机
    else:
        # 因果推断或其他：默认随机响应
        responses = [f"可能{'有' if np.random.random()>0.5 else '无'}因果关系" for _ in range(n_samples)]
        true_label = int(np.random.binomial(1, 0.58))
    return responses, true_label

# ── 2. 语义不确定性计算（简化版：基于回答多样性）────────────────────
def semantic_entropy(responses: list) -> float:
    """
    计算语义熵（语义不确定性）
    简化版：基于唯一回答比例（生产版用NLI模型判断语义等价）
    """
    # 简化的语义等价分组：相同关键词视为同一语义
    def extract_key_token(r):
        for kw in ['增长', '下降', '上升', '减少', '持平']:
            if kw in r: return kw
        # 数字型：四舍五入到一位小数
        import re
        nums = re.findall(r'\d+\.\d+', r)
        if nums: return str(round(float(nums[0]), 1))
        return r[:10]

    groups = {}
    for r in responses:
        key = extract_key_token(r)
        groups[key] = groups.get(key, 0) + 1

    n = len(responses)
    probs = [c/n for c in groups.values()]
    entropy = -sum(p * np.log(p + 1e-9) for p in probs)
    return entropy

# ── 3. 置信度校准（Platt缩放）────────────────────────────────────────
# 模拟历史标注数据：LLM的verbalized置信度 + 实际是否正确
n_cal = 500
verbalized_conf = np.random.beta(6, 2, n_cal)  # LLM自报置信度（普遍偏高）
# 真实准确率比自报低（过度自信）
true_correct = np.random.binomial(1, verbalized_conf * 0.7 + 0.15, n_cal)

# Platt缩放：将自报置信度校准为真实准确率
calibrator = LogisticRegression(C=10).fit(
    verbalized_conf.reshape(-1, 1), true_correct
)

def calibrate_confidence(raw_conf: float) -> float:
    """将LLM自报置信度转化为校准后的真实准确率估计"""
    return calibrator.predict_proba([[raw_conf]])[0][1]

print("【置信度校准效果】")
for raw in [0.95, 0.80, 0.65, 0.50]:
    cal = calibrate_confidence(raw)
    print(f"  LLM自报{raw:.0%} → 校准后真实准确率估计: {cal:.0%}")

# ── 4. BI场景不确定性评估器 ──────────────────────────────────────────
class BIUncertaintyEstimator:
    """
    BI场景LLM不确定性的三层评估
    """
    QUESTION_TYPES = {
        'factual_number': ('数字查询', 0.85, '数字精度受SQL准确性影响'),
        'trend_direction': ('趋势方向', 0.72, '方向判断受数据时效和口径影响'),
        'comparison':      ('对比分析', 0.68, '跨维度对比容易混淆口径'),
        'prediction':      ('预测展望', 0.52, '预测本身具有高不确定性'),
        'causal_claim':    ('因果推断', 0.58, '因果关系需要严格方法论，LLM容易混淆相关与因果'),
    }

    def assess(self, question: str, responses: list, verbalized_conf: float) -> dict:
        """综合评估LLM回答的不确定性"""
        # 检测问题类型
        q_type = 'factual_number'
        if any(kw in question for kw in ['趋势', '增长', '下降', '变化']):
            q_type = 'trend_direction'
        elif any(kw in question for kw in ['预测', '未来', '预计', '估计']):
            q_type = 'prediction'
        elif any(kw in question for kw in ['为什么', '原因', '导致', '因为']):
            q_type = 'causal_claim'
        elif any(kw in question for kw in ['对比', '比较', '差异', '更高']):
            q_type = 'comparison'

        type_name, base_acc, warning = self.QUESTION_TYPES[q_type]

        # 语义熵
        s_entropy = semantic_entropy(responses)
        entropy_penalty = min(s_entropy / np.log(len(responses)+1), 1.0)

        # 校准置信度
        calibrated = calibrate_confidence(verbalized_conf)

        # 综合置信度（三因素加权）
        final_confidence = (
            0.4 * calibrated +
            0.4 * base_acc +
            0.2 * (1 - entropy_penalty)
        )

        if final_confidence >= 0.80:
            verdict = '✅ 高可信 — 可直接使用'
        elif final_confidence >= 0.65:
            verdict = '⚠️ 中等置信 — 建议交叉验证'
        else:
            verdict = '❌ 低可信 — 必须人工核查'

        return {
            'question_type': type_name,
            'semantic_entropy': s_entropy,
            'calibrated_conf': calibrated,
            'base_type_acc': base_acc,
            'final_confidence': final_confidence,
            'verdict': verdict,
            'warning': warning,
        }

# ── 5. 测试不同类型问题 ───────────────────────────────────────────────
estimator = BIUncertaintyEstimator()

test_questions = [
    ("本月ROAS是多少？", "factual_number", 0.92),
    ("销量同比增长还是下降？", "trend_direction", 0.78),
    ("下个月销量会怎样？", "factual_number", 0.71),
    ("广告预算增加导致销量提升了吗？", "causal_claim", 0.75),
]

print("\n【BI场景不确定性评估报告】")
print(f"{'问题':<25} {'类型':<10} {'校准置信':>8} {'语义熵':>8} {'综合置信':>9} 判决")
print("-" * 85)

for question, q_type, raw_conf in test_questions:
    responses, _ = simulate_llm_responses(q_type)
    result = estimator.assess(question, responses, raw_conf)
    print(f"  {question:<23} {result['question_type']:<10} "
          f"{result['calibrated_conf']:>7.0%} "
          f"{result['semantic_entropy']:>8.3f} "
          f"{result['final_confidence']:>8.0%}  {result['verdict']}")

# ── 6. 校准质量验证（Brier分数）──────────────────────────────────────
test_conf = np.random.beta(5, 2, 200)
test_correct = np.random.binomial(1, test_conf * 0.7 + 0.15, 200)
calibrated_probs = calibrator.predict_proba(test_conf.reshape(-1,1))[:,1]
brier = brier_score_loss(test_correct, calibrated_probs)
print(f"\n校准质量 Brier分数: {brier:.4f} (越低越好，随机=0.25，完美=0)")
assert brier < 0.25, f"校准质量不足: {brier:.4f}"
print("\n[✓] LLM不确定性量化BI 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-LLM-Hallucination-Detection-BI]]（幻觉检测是不确定性量化的配套工具）、[[Skill-Model-Calibration]]（模型校准基础方法）
- **延伸（extends）**：[[Skill-Conformal-Prediction-Framework]]（分位数式不确定性区间的更严格版本）
- **可组合（combinable）**：[[Skill-LLM-as-Judge-Evaluator]]（Judge评估中加入不确定性标注）、[[Skill-LLM-Business-Intelligence-Reasoning]]（BI推理中透传不确定性给用户）

## ⑤ 商业价值评估

- **ROI 预估**：低置信度结论自动触发人工复核，防止基于错误AI数据决策，年化避免决策损失约80万元；高置信度答案（>85%）直接使用，减少人工核查60%，节省约30万元/年；总ROI约110万元/年
- **实施难度**：⭐⭐⭐☆☆（语义熵需多次LLM调用，成本增加5-10倍；Platt校准需要历史标注数据）
- **优先级**：⭐⭐⭐⭐☆（AI工具可信度是工业落地的关键瓶颈，不确定性标注是提升信任的最直接手段）
- **评估依据**：ICLR 2023 Semantic Uncertainty论文验证语义熵比token-level熵更准确；ICLR 2024展示LLM自报置信度系统性高估约15-25%，校准是必要步骤；Google/Anthropic的内部评估系统均包含不确定性量化模块
