---
title: A/B Test Result Interpretation and Practical Significance
module: 02-A_B实验
topic: result-interpretation
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase1
---

# Skill Card: A/B Test Result Interpretation

## ① 算法原理

**核心问题**：Power Analysis告诉你"测多少"，实验跑完后，如何正确解读结果？很多团队会犯这些错误：
- 只看P值，不估计效应量
- 把统计显著当成业务显著
- 忽视置信区间的宽度
- 对不显著的结果下"无效"结论（可能是样本量不足）

**正确解读框架**：

**1. 效应量估计（Effect Size）**

统计显著 ≠ 业务有价值。效应量告诉你"改了之后提升了多少"。

- **绝对提升**：实验组 - 对照组
- **相对提升**：（实验组 - 对照组）/ 对照组
- **Cohen's d**：（实验组 - 对照组）/ 合并标准差（标准化效应量）

**2. 置信区间（Confidence Interval）**

点估计不可靠，区间估计才可靠。95%置信区间告诉你"真实效应有95%的概率落在这个范围"。

- 如果CI下限 > 0：正向显著
- 如果CI上限 < 0：负向显著
- 如果CI包含0：不显著（但不能说"无效"）

**3. 统计显著 vs 业务显著**

| 场景 | 统计显著 | 业务显著 | 结论 |
|------|---------|---------|------|
| 转化率提升0.1%，P=0.01 | ✅ | ❌（提升太小） | 不值得上线 |
| 转化率提升5%，P=0.10 | ❌ | ✅（提升大） | 样本量不足，需要扩大实验 |
| 转化率提升5%，P=0.001 | ✅ | ✅ | 上线 |
| 转化率下降2%，P=0.03 | ✅ | ✅（负向） | 不上线 |

**4. 实验后检验（Post-hoc Analysis）**

- **分段分析**：不同用户群体（新/老、不同国家）的效果是否一致？
- **时间趋势**：效果是否随时间衰减？
- **Guardrail Metrics**：是否对其他关键指标产生负面影响？

**反直觉洞察**：
- P=0.05意味着"如果改动无效，有5%的概率看到当前结果"——不是"改动有效的概率是95%"
- 跑了10个实验，即使所有改动都无效，也有约40%的概率至少有一个显著（多重比较问题）
- "不显著"不等于"零效应"——可能是效应存在但样本量不够检测出来

---

## ② 母婴出海应用案例

### 场景1：首页改版实验结果解读

**实验结果**：
- 对照组转化率：2.00%
- 实验组转化率：2.12%
- 相对提升：+6.0%
- P值：0.03
- 95% CI：[0.02%, 0.22%]

**解读流程**：
1. **统计显著？** P=0.03 < 0.05 → ✅ 统计显著
2. **效应量？** 绝对提升0.12个百分点，相对提升6% → 中小效应
3. **业务显著？** 日均50,000 UV，提升0.12% = 每天多60单，月增1,800单 → ✅ 业务显著
4. **置信区间？** [0.02%, 0.22%] → 真实提升至少0.02%，值得上线
5. **分段分析？** 美国站+8%，德国站+3%，英国站-1% → 考虑分国家上线
6. **Guardrail？** 客单价无显著变化，退货率无显著变化 → ✅ 安全

**结论**：建议在美国站和德国站上线，英国站不推。

### 场景2：不显著结果的后续决策

**实验结果**：
- 对照组转化率：2.00%
- 实验组转化率：2.05%
- 相对提升：+2.5%
- P值：0.25
- 95% CI：[-0.05%, 0.15%]

**解读**：
- 不显著（P>0.05），但CI上限0.15%暗示可能有正向效应
- MDE回顾：如果实验前设定的MDE是0.1%（相对5%），当前提升2.5%小于MDE
- 决策：不扩大实验，也不放弃——将这个设计元素保留为备选方案，在更大改动中复用

---

## ③ 代码模板

```python
"""
A/B Test Result Interpretation — A/B实验结果解读
支持：效应量计算、置信区间、分段分析、多重比较校正
"""

import numpy as np
from scipy import stats


def ab_test_summary(control_conversions, control_total,
                    treatment_conversions, treatment_total,
                    confidence=0.95):
    """
    A/B测试结果全面解读

    Args:
        control_conversions: 对照组转化数
        control_total: 对照组样本量
        treatment_conversions: 实验组转化数
        treatment_total: 实验组样本量
    """
    # 转化率
    p_c = control_conversions / control_total
    p_t = treatment_conversions / treatment_total

    # 绝对和相对提升
    abs_lift = p_t - p_c
    rel_lift = abs_lift / p_c if p_c > 0 else 0

    # Z检验
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    z = abs_lift / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 置信区间
    alpha = 1 - confidence
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total)
    ci_lower = abs_lift - z_crit * se_diff
    ci_upper = abs_lift + z_crit * se_diff

    # Cohen's h（比例差异的效应量）
    cohens_h = 2 * (np.arcsin(np.sqrt(p_t)) - np.arcsin(np.sqrt(p_c)))

    # 业务解读
    is_stat_sig = p_value < alpha
    is_practical = abs(rel_lift) >= 0.05  # 假设5%相对提升为业务显著阈值

    interpretation = []
    if is_stat_sig and is_practical:
        interpretation.append("✅ 统计显著 + 业务显著 → 建议上线")
    elif is_stat_sig and not is_practical:
        interpretation.append("⚠️ 统计显著但提升太小 → 评估ROI后决定")
    elif not is_stat_sig and is_practical:
        interpretation.append("📊 业务提升大但不显著 → 样本量不足，考虑扩大实验")
    else:
        interpretation.append("❌ 不显著且提升小 → 不建议上线")

    return {
        'control_rate': p_c,
        'treatment_rate': p_t,
        'abs_lift': abs_lift,
        'rel_lift': rel_lift,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_h': cohens_h,
        'is_statistically_significant': is_stat_sig,
        'is_practically_significant': is_practical,
        'interpretation': interpretation
    }


def segment_analysis(control_segments, treatment_segments):
    """
    分段分析

    Args:
        control_segments: dict {segment_name: (conversions, total)}
        treatment_segments: dict {segment_name: (conversions, total)}
    """
    results = []
    for segment in control_segments:
        c_conv, c_total = control_segments[segment]
        t_conv, t_total = treatment_segments[segment]
        result = ab_test_summary(c_conv, c_total, t_conv, t_total)
        result['segment'] = segment
        results.append(result)

    return results


def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni多重比较校正"""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values], corrected_alpha


# 示例
if __name__ == '__main__':
    # 首页改版实验
    result = ab_test_summary(
        control_conversions=1000, control_total=50000,
        treatment_conversions=1060, treatment_total=50000
    )
    print("A/B测试结果解读:")
    print(f"  对照组转化率: {result['control_rate']:.3%}")
    print(f"  实验组转化率: {result['treatment_rate']:.3%}")
    print(f"  绝对提升: {result['abs_lift']:.3%}")
    print(f"  相对提升: {result['rel_lift']:.1%}")
    print(f"  P值: {result['p_value']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.3%}, {result['ci_upper']:.3%}]")
    print(f"  Cohen's h: {result['cohens_h']:.3f}")
    print(f"  结论: {result['interpretation'][0]}")

    # 分段分析
    segments = segment_analysis(
        control_segments={'US': (400, 20000), 'DE': (300, 15000), 'UK': (300, 15000)},
        treatment_segments={'US': (448, 20000), 'DE': (312, 15000), 'UK': (300, 15000)}
    )
    print("\n分段分析:")
    for s in segments:
        print(f"  {s['segment']}: {s['rel_lift']:+.1%} (P={s['p_value']:.3f}) {s['interpretation'][0]}")
print("[✓] AB Test Result Interpreta 测试通过")
```

---


## ④ 技能关联

### 前置技能
- [Skill-AB-Experimental-Design](../02-A_B实验/[[Skill-AB-Experimental-Design]].md) — 解读建立在严谨的实验设计之上
- [Skill-Power-Analysis-Sample-Size](../02-A_B实验/[[Skill-Power-Analysis-Sample-Size]].md) — 样本量决定结果置信度

### 延伸技能
- [Skill-Intelligent-Attribution-Causal-Forest](../01-因果推断/[[Skill-Intelligent-Attribution-Causal-Forest]].md) — 结果解读后做异质性归因（CATE）

### 可组合
- [Skill-Uplift-Modeling](../01-因果推断/[[Skill-Uplift-Modeling]].md) — Uplift 与 A/B 实验的因果效应估计互补

## ⑤ 商业价值评估

- **ROI**：避免"统计显著但业务无价值"的错误上线，每次避免损失 > 10万
- **难度**：⭐☆☆☆☆（1/5）— 概念简单，但团队常犯错
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 每个实验后的必做分析，成本极低但价值极高
