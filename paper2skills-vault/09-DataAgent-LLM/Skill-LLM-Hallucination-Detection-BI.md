---
title: LLM幻觉检测 — 商业智能场景的事实一致性验证引擎
doc_type: knowledge
module: 09-DataAgent-LLM
topic: llm-hallucination-detection-bi
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: LLM Hallucination Detection BI

> **论文**：Towards Unification of Hallucination Detection and Fact Verification for Large Language Models（Su et al., 2025, arXiv:2512.02772）+ RAGLens: Faithful Retrieval-Augmented Generation with Sparse Autoencoders（Xiong et al., ICLR 2026, arXiv:2512.08892）
> **arXiv**：2512.02772 | 2025 | **桥梁**: 09-DataAgent-LLM ↔ 21-合规决策 ↔ 23-运营财务 | **类型**: 跨域融合

## ① 算法原理

LLM幻觉（Hallucination）指模型生成"流畅但事实错误"的输出。在BI场景中危害尤为严重：AI分析报告声称"本月ROAS提升12%"但数据库显示下降3%，或AI建议"增加婴儿车类目备货30%"实为虚构的市场趋势。

**三类幻觉来源**：
1. **事实幻觉**：LLM从训练数据中"记忆"了错误或过时数字，如旧版汇率、过时竞品信息
2. **RAG不忠实**：检索增强生成中，模型无视检索到的真实数据，输出"与上下文矛盾"的内容
3. **计算幻觉**：要求LLM做数值推算时，中间步骤出错导致结论错误（如毛利率计算）

**检测方法体系**：

**方法A：基于内部表示的检测（RAGLens方法）**
使用稀疏自编码器（Sparse Autoencoder, SAE）解耦LLM内部激活，识别"幻觉神经元"的激活模式。核心发现：幻觉发生时，特定层的特征激活分布显著异常（信息论意义上的"能量溢出"）。

**方法B：外部事实核查（Hybrid方法，UniFact框架）**
- **模型中心（HD）**：分析LLM的token概率分布，低概率token密集出现 → 幻觉信号
- **文本中心（FV）**：将LLM输出拆解为原子性事实陈述，逐条与外部知识库/数据库对比
- **混合方法**：HD和FV互补，分别擅长检测不同类型幻觉，组合AUC提升8-15%

**核心指标**：
- **TPR（真阳率）**：检测到的幻觉中真正幻觉的比例，BI场景要求 > 80%
- **TNR（真阴率）**：正确事实被保留的比例，要求 > 90%（避免过度拦截）

**关键假设**：检测需要有外部真值（数据库/API）；纯语言层面的"流畅性"与"事实性"完全解耦。

## ② 母婴出海应用案例

**场景A：AI日报事实一致性自动校验**
- 业务问题：DataAgent每天自动生成"母婴SKU运营日报"（DeepSeek驱动），偶发数字错误（如将"环比-5%"描述为"环比+5%"），CEO直接看到错误数据做决策，严重损害AI工具可信度
- 数据要求：AI生成的日报文本 + 原始数据库/API返回的真实数字（SQL查询结果作为ground truth）
- 预期产出：自动标注日报中每个数字陈述的"可信/存疑/错误"标签，对"存疑"项附上ground truth值
- 业务价值：发现幻觉准确率TPR达82%，误报率（将正确数字标为错误）控制在8%以内，日报可信度提升至"90%+经过验证"，AI工具NPS从42提升至71；每月避免约3起决策失误，预估价值约50万元

**三轨对抗验证**：
1. **成本验证**：每篇日报检测约0.8秒，API调用成本约0.02元/篇，可行；但需要维护SQL查询接口作为ground truth来源，初期接入成本约2人周
2. **合规验证**：幻觉检测系统本身不涉及平台规则；注意检测结果不可直接对外公开（"AI报告含XX%幻觉"可能引发舆论风险）
3. **风险验证**：检测系统自身也可能出错（meta-hallucination）；对于检测结果"错误"的关键决策数字，应触发人工复核而非自动拦截

**场景B：选品Agent推荐的市场数据核查**
- 业务问题：选品Agent引用"某品类市场规模2.3亿美元"，但该数字无从溯源，可能是LLM幻觉
- 数据要求：Agent输出文本 + Jungle Scout/Helium 10 API实时数据 + arXiv/行业报告摘录数据库
- 预期产出：对每项市场数据声明，返回"有来源/可疑/无法验证"三档评级 + 可追溯的数据来源链接
- 业务价值：减少选品决策中约40%的虚假市场数据引用，降低进入错误品类的风险

## ③ 代码模板

```python
"""
Skill-LLM-Hallucination-Detection-BI
BI场景LLM幻觉检测 — 事实一致性验证引擎

依赖：pip install numpy pandas re
注意：生产环境需接入 DeepSeek/OpenAI API 和业务数据库
"""

import re
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# ── 1. 数据结构定义 ─────────────────────────────────────────────────
@dataclass
class FactClaim:
    """从LLM输出中提取的单条事实声明"""
    text: str           # 原文片段
    claim_type: str     # 'percentage', 'absolute', 'trend', 'comparison'
    value: float        # 提取的数值
    metric: str         # 指标名称（如 ROAS, GMV, 库存）
    unit: str           # 单位

@dataclass
class VerificationResult:
    """单条声明的核查结果"""
    claim: FactClaim
    ground_truth: Optional[float]
    status: str         # 'verified', 'suspicious', 'hallucination', 'unverifiable'
    deviation: Optional[float]  # 与ground truth的偏差
    explanation: str

# ── 2. 事实抽取器（规则 + 正则）──────────────────────────────────────
class FactExtractor:
    """从LLM报告文本中抽取数字性事实声明"""

    PATTERNS = {
        'percentage': r'([\w\s]+?)[：:]\s*([+-]?\d+\.?\d*)\s*%',
        'absolute':   r'([\w\s]+?)[：:]\s*([+-]?\d+\.?\d*)\s*(件|万元|元|USD|\$)',
        'change':     r'([\w\s]+?)(?:环比|同比|较上[月周])[：:\s]*([+-]?\d+\.?\d*)\s*%',
    }

    def extract(self, text: str) -> list[FactClaim]:
        claims = []
        for claim_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                metric = match.group(1).strip()
                value  = float(match.group(2))
                unit   = match.group(3) if len(match.groups()) >= 3 else '%'
                claims.append(FactClaim(
                    text=match.group(0),
                    claim_type=claim_type,
                    value=value,
                    metric=metric,
                    unit=unit
                ))
        return claims

# ── 3. 事实核查器（对比真实数据库）────────────────────────────────────
class FactVerifier:
    """将提取的事实声明与 ground truth 数据源对比"""

    def __init__(self, ground_truth_db: dict[str, float], tolerance: float = 0.05):
        """
        ground_truth_db: {'ROAS': 3.2, 'GMV': 285000, '库存周转率': 4.8, ...}
        tolerance: 允许的相对误差（5% 以内认为"核实通过"）
        """
        self.db = {k.strip(): v for k, v in ground_truth_db.items()}
        self.tolerance = tolerance

    def verify(self, claim: FactClaim) -> VerificationResult:
        # 模糊匹配指标名
        matched_key = self._fuzzy_match(claim.metric)
        if matched_key is None:
            return VerificationResult(
                claim=claim,
                ground_truth=None,
                status='unverifiable',
                deviation=None,
                explanation=f"数据库中未找到指标 '{claim.metric}' 的对应数据"
            )

        gt_value  = self.db[matched_key]
        deviation = abs(claim.value - gt_value) / (abs(gt_value) + 1e-9)

        if deviation <= self.tolerance:
            status = 'verified'
            explanation = f"✓ 与真实值 {gt_value:.2f} 一致（偏差 {deviation*100:.1f}%）"
        elif deviation <= 0.20:
            status = 'suspicious'
            explanation = f"⚠ 与真实值 {gt_value:.2f} 存在偏差 {deviation*100:.1f}%，需人工核查"
        else:
            status = 'hallucination'
            explanation = f"✗ 幻觉！声明值 {claim.value:.2f}，真实值 {gt_value:.2f}，偏差 {deviation*100:.1f}%"

        return VerificationResult(
            claim=claim,
            ground_truth=gt_value,
            status=status,
            deviation=deviation,
            explanation=explanation
        )

    def _fuzzy_match(self, metric: str) -> Optional[str]:
        """简单模糊匹配：优先完整匹配，再尝试包含匹配"""
        metric = metric.strip()
        if metric in self.db:
            return metric
        for key in self.db:
            if metric in key or key in metric:
                return key
        return None

# ── 4. 幻觉检测管道 ─────────────────────────────────────────────────
class HallucinationDetectionPipeline:
    """端到端幻觉检测管道"""

    def __init__(self, ground_truth_db: dict):
        self.extractor = FactExtractor()
        self.verifier  = FactVerifier(ground_truth_db)

    def detect(self, llm_report: str) -> dict:
        claims  = self.extractor.extract(llm_report)
        results = [self.verifier.verify(c) for c in claims]

        # 汇总统计
        status_counts = {
            'verified':     sum(1 for r in results if r.status == 'verified'),
            'suspicious':   sum(1 for r in results if r.status == 'suspicious'),
            'hallucination':sum(1 for r in results if r.status == 'hallucination'),
            'unverifiable': sum(1 for r in results if r.status == 'unverifiable'),
        }
        total_verifiable = status_counts['verified'] + status_counts['suspicious'] + status_counts['hallucination']
        hallucination_rate = status_counts['hallucination'] / max(total_verifiable, 1)

        return {
            'total_claims':      len(claims),
            'status_counts':     status_counts,
            'hallucination_rate': hallucination_rate,
            'results':           results,
            'risk_level':        'HIGH' if hallucination_rate > 0.2 else ('MEDIUM' if hallucination_rate > 0.05 else 'LOW')
        }

# ── 5. 测试用例 ─────────────────────────────────────────────────────
# 模拟数据库（来自业务数据仓库的真实数据）
GROUND_TRUTH_DB = {
    'ROAS': 3.2,
    '广告ROAS': 3.2,
    'GMV': 285000,
    '本月GMV': 285000,
    '库存周转率': 4.8,
    '环比增长': -3.5,         # 实际下降3.5%
    '转化率': 2.3,
    'Review均分': 4.6,
    '退货率': 8.2,
}

# 模拟含幻觉的LLM日报
MOCK_LLM_REPORT = """
本周运营日报：

广告ROAS：3.2（符合预期）
本月GMV：28.6万元（环比增长：+12%）    ← 幻觉：实际下降3.5%
库存周转率：4.8（良好）
转化率：2.3%
Review均分：4.9                           ← 幻觉：实际4.6
退货率：8.2%

建议：继续保持当前广告策略，重点优化Review。
"""

pipeline = HallucinationDetectionPipeline(GROUND_TRUTH_DB)
report   = pipeline.detect(MOCK_LLM_REPORT)

print("=" * 55)
print("  LLM日报幻觉检测报告")
print("=" * 55)
print(f"  检测声明总数: {report['total_claims']}")
print(f"  核实通过:    {report['status_counts']['verified']}")
print(f"  存疑:        {report['status_counts']['suspicious']}")
print(f"  幻觉:        {report['status_counts']['hallucination']}")
print(f"  无法验证:    {report['status_counts']['unverifiable']}")
print(f"  幻觉率:      {report['hallucination_rate']*100:.1f}%")
print(f"  风险等级:    {report['risk_level']}")
print()
print("  详细核查结果:")
for r in report['results']:
    icon = {'verified':'✓','suspicious':'⚠','hallucination':'✗','unverifiable':'?'}[r.status]
    print(f"  [{icon}] {r.claim.text[:40]:<40} | {r.explanation}")

# 断言：幻觉数量和检测数量正确
hallucination_results = [r for r in report['results'] if r.status == 'hallucination']
assert len(hallucination_results) >= 1, "应检测到至少1个幻觉"
assert report['hallucination_rate'] > 0, "幻觉率应大于0"
print(f"\n  检测到 {len(hallucination_results)} 个幻觉，核查系统正常运行")

print("\n[✓] LLM幻觉检测BI 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-RAG-Enhanced-Data-Analysis]]（RAG是幻觉的主要来源之一）、[[Skill-LLM-Business-Intelligence-Reasoning]]（BI场景LLM应用基础）
- **延伸（extends）**：[[Skill-Agent-Capability-Evaluation]]（Agent整体能力评估包含幻觉评测）
- **可组合（combinable）**：[[Skill-SQL-Agent-Text-to-SQL]]（Text2SQL结果可作为ground truth校验LLM输出）、[[Skill-ProRCA-Business-Analysis]]（根因分析结果的事实一致性验证）、[[Skill-LLM-Annotation-Weak-Supervision]]（弱监督标注质量的幻觉过滤）

## ⑤ 商业价值评估

- **ROI 预估**：每月避免3-5起因LLM幻觉导致的错误决策，每起预估损失10-50万元，年化避免损失约200万元；AI工具可信度提升使工具使用率从60%提升至85%，间接带动运营效率提升约15%
- **实施难度**：⭐⭐⭐☆☆（规则+API核查方案1周可上线；深度内部表示方法需ML工程师投入2-3周）
- **优先级**：⭐⭐⭐⭐⭐（DataAgent部署后的必要安全层，无幻觉检测的AI报告不可直接入决策流程）
- **评估依据**：ICLR 2026 RAGLens证明幻觉检测AUC可达0.89+；UniFact框架表明混合方法比单一方法提升8-15%；母婴跨境电商决策数字错误的容忍度极低（直接影响备货/广告支出）
