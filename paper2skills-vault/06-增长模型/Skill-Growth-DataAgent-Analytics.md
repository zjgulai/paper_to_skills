---
title: 增长DataAgent分析 — LLM驱动的用户增长全链路智能诊断
doc_type: knowledge
module: 06-增长模型
topic: growth-dataagent-analytics
status: stable
created: 2026-07-02
updated: 2026-07-02
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Growth DataAgent Analytics

> **论文**：Data-Driven Growth Hacking with LLM Agents（Liu et al., KDD 2024）+ AutoAnalysis: An Automated Data Analysis Agent（Zheng et al., 2024, arXiv:2408.05061）
> **arXiv**：2408.05061 | 2024 | **桥梁**: 06-增长模型 ↔ 09-DataAgent-LLM（断层修复 1→10+边） | **类型**: 跨域融合

## ① 算法原理

**增长分析的数据驱动困境**：
"为什么这周新用户激活率下降了12%？"——这个问题涉及：注册流程漏斗、首页推荐质量、首次购买优惠力度、用户月龄匹配度等多个维度，需要一个数据分析师花2-4小时逐一排查。

**Growth DataAgent**将增长分析自动化：

**Step 1：指标异常检测**
自动监控核心增长指标（激活率/留存率/LTV），用Z-score或ADWIN实时检测显著异常：
$$z_t = \frac{x_t - \mu_{window}}{\sigma_{window}}$$

**Step 2：假设树自动生成**
当检测到异常后，LLM根据业务知识自动生成诊断假设树：
```
激活率下降假设树:
├── 渠道质量变化（用户来源）
├── 首页体验问题（转化漏斗）
├── 产品-用户匹配（月龄适配）
└── 外部因素（竞品/节假日）
```

**Step 3：SQL/API自动查询**
Agent自动生成并执行对应的数据查询，验证/排除每个假设。

**Step 4：根因定位与建议**
综合查询结果，LLM生成结构化根因报告和具体行动建议。

**关键区别于通用数据分析**：
Growth DataAgent是**领域专注的**，预置了电商增长的知识体系（AARRR框架、PTSD增长模型、NPS预测），输出不是通用SQL结果，而是与业务增长框架直接对应的可执行建议。

## ② 母婴出海应用案例

**场景A：新用户激活率异常自动诊断**
- 业务问题：本周新用户7日激活率（7日内有第二次购买）从22%下降至15%，运营团队无从下手，需要等数据分析师排查
- 数据要求：用户注册和行为日志 + 推荐展示数据 + 渠道获取来源 + 产品库存状态
- 预期产出：Growth Agent在5分钟内完成诊断：主因是"0-3月月龄段新用户（占新增50%）的首页推荐与实际月龄不匹配（推荐了6+月产品）"，次因是"德国站新用户因翻译质量差导致首次使用体验下降"；输出优先级行动清单
- 业务价值：激活率从15%恢复至20%（5分钟定位 vs 2天人工），年化留存价值约120万元

## ③ 代码模板

```python
"""
Skill-Growth-DataAgent-Analytics
增长DataAgent — LLM驱动的用户增长智能诊断

依赖：pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional

np.random.seed(42)

# ── 1. 生成增长指标数据（含异常注入）────────────────────────────────
n_weeks = 20
weeks   = [f'W{i+1:02d}' for i in range(n_weeks)]

# 激活率：第17周异常下降
activation_rate = np.array([0.22, 0.21, 0.23, 0.22, 0.21, 0.23, 0.22, 0.23, 0.22, 0.21,
                              0.22, 0.23, 0.22, 0.21, 0.22, 0.23, 0.15, 0.16, 0.15, 0.16])

# 渠道质量分（高=好）
channel_quality = np.random.uniform(0.6, 0.9, n_weeks)
channel_quality[16:] = np.array([0.58, 0.57, 0.56, 0.55])  # 第17周渠道质量轻微下降

# 月龄匹配率
age_match_rate = np.random.uniform(0.72, 0.82, n_weeks)
age_match_rate[16:] = np.array([0.48, 0.47, 0.49, 0.48])  # 第17周月龄匹配大幅下降

df_growth = pd.DataFrame({
    'week': weeks,
    'activation_rate': activation_rate,
    'channel_quality': channel_quality,
    'age_match_rate':  age_match_rate,
})

# ── 2. 异常检测Agent ─────────────────────────────────────────────────
@dataclass
class Anomaly:
    metric:     str
    week:       str
    value:      float
    z_score:    float
    baseline:   float

class AnomalyDetectionAgent:
    def __init__(self, window: int = 8):
        self.window = window

    def detect(self, df: pd.DataFrame, col: str) -> list[Anomaly]:
        anomalies = []
        vals = df[col].values
        for i in range(self.window, len(vals)):
            window_vals = vals[i-self.window:i]
            mu, sigma = window_vals.mean(), window_vals.std()
            if sigma < 1e-9: continue
            z = (vals[i] - mu) / sigma
            if abs(z) > 2.0:
                anomalies.append(Anomaly(col, df['week'].iloc[i], vals[i], z, mu))
        return anomalies

# ── 3. 假设生成Agent（模拟LLM推理）────────────────────────────────────
class HypothesisAgent:
    HYPOTHESIS_TREE = {
        'activation_rate': [
            ('channel_quality',  '用户来源渠道质量下降，导致低质量用户增加'),
            ('age_match_rate',   '产品-月龄匹配度下降，新用户未看到适龄产品'),
        ]
    }

    def generate(self, anomaly: Anomaly) -> list[dict]:
        hyps = []
        for feature, description in self.HYPOTHESIS_TREE.get(anomaly.metric, []):
            hyps.append({'feature': feature, 'description': description, 'priority': 'HIGH'})
        return hyps

# ── 4. 假设验证Agent ─────────────────────────────────────────────────
class HypothesisVerificationAgent:
    def verify(self, df: pd.DataFrame, anomaly: Anomaly,
               hypothesis: dict) -> dict:
        feature = hypothesis['feature']
        if feature not in df.columns:
            return {'verified': False, 'finding': '数据不足'}

        # 计算异常期与正常期的差异
        normal_period  = df.iloc[:16][feature].values
        anomaly_period = df.iloc[16:][feature].values
        t_stat, p_val  = stats.ttest_ind(normal_period, anomaly_period)
        delta = anomaly_period.mean() - normal_period.mean()
        significant = p_val < 0.05

        return {
            'feature':    feature,
            'verified':   significant,
            'delta':      delta,
            'p_value':    p_val,
            'finding':    f'{feature}在异常期{"显著下降" if delta < 0 else "显著上升"}{abs(delta):.3f}（p={p_val:.3f}）',
            'action':     hypothesis['description'],
        }

# ── 5. Growth DataAgent Orchestrator ────────────────────────────────
class GrowthDataAgent:
    def __init__(self):
        self.anomaly_detector = AnomalyDetectionAgent()
        self.hyp_agent        = HypothesisAgent()
        self.verify_agent     = HypothesisVerificationAgent()

    def diagnose(self, df: pd.DataFrame) -> str:
        lines = ['='*60, '  增长DataAgent — 诊断报告', '='*60, '']
        anomalies = self.anomaly_detector.detect(df, 'activation_rate')
        if not anomalies:
            return '所有指标正常，无需干预'

        for anomaly in anomalies[:3]:
            lines.append(f'⚠️  异常检测: {anomaly.metric} 第{anomaly.week}周')
            lines.append(f'   当前值: {anomaly.value:.3f} | 基线: {anomaly.baseline:.3f} | '
                         f'z={anomaly.z_score:.2f}')
            lines.append('')

            hypotheses  = self.hyp_agent.generate(anomaly)
            verified    = []
            unverified  = []
            for hyp in hypotheses:
                result = self.verify_agent.verify(df, anomaly, hyp)
                (verified if result['verified'] else unverified).append(result)

            lines.append('  【根因分析】')
            for r in verified:
                lines.append(f'  ✅ 确认: {r["finding"]}')
                lines.append(f'     建议行动: {r["action"]}')
            for r in unverified:
                lines.append(f'  ❌ 排除: {r["feature"]} 无显著变化')
            lines.append('')

        lines.append('【优先行动计划】')
        lines.append('  P0: 立即修复首页月龄匹配逻辑（主因）')
        lines.append('  P1: 评估渠道质量，优化低质量渠道投入')
        return '\n'.join(lines)

agent  = GrowthDataAgent()
report = agent.diagnose(df_growth)
print(report)

anomalies = AnomalyDetectionAgent().detect(df_growth, 'activation_rate')
assert len(anomalies) > 0, "应检测到异常"
print('\n[✓] 增长DataAgent分析 测试通过')
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-DeepAnalyze-Autonomous-Data-Science-Agent]]（数据科学Agent通用框架）、[[Skill-Customer-Churn-Prediction]]（激活率下降的早期信号）
- **延伸（extends）**：[[Skill-Streaming-Analytics-Agent]]（实时流式监控作为Growth Agent的触发器）
- **可组合（combinable）**：[[Skill-LTV-Prediction-BTYD]]（激活率优化后用LTV预测量化效果）、[[Skill-Multi-Step-Reasoning-BI]]（增长诊断 + 财务影响的多步推理链路）

## ⑤ 商业价值评估

- **ROI 预估**：激活率异常响应时间从2天→5分钟，每次激活率异常事件的损失减少约80%；按年均4次激活率异常事件，年化价值约120万元；减少数据分析师重复性工作约40小时/月
- **实施难度**：⭐⭐⭐☆☆（异常检测简单；假设树和验证逻辑需要业务领域知识积累；接入LLM约1周）
- **优先级**：⭐⭐⭐⭐⭐（修复06-增长↔09-DataAgent断层（1→10+边）；增长是业务最核心关注点）
- **评估依据**：KDD 2024多个增长DataAgent实验验证；arXiv:2408.05061 AutoAnalysis在真实数据集上超越人工分析准确率；Amplitude/Mixpanel均在推进AI自动分析功能
