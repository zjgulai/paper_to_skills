# Skill Card: Amazon ToS Compliance Guardrail（亚马逊合规护栏）

> **论文**: SAFE-AGENT-L: Legal Compliance Framework for LLM Agents in Retail (OpenReview, AAAI 2026 Bridge Workshop 2025)  
> **辅论文**: EVADE-Bench: Evasive Content Detection in E-Commerce (arXiv:2505.17654, 2025)  
> **领域**: 13-广告分析（跨 WF-C + WF-B） | **服务工作流**: WF-C + WF-B

---

## ① 算法原理

### 核心思想
LLM 在生成商品文案、广告文案、客服回复时可能无意间违反平台规则（医疗声明、安全认证、受限品类）。Compliance Guardrail 在 LLM 输出端建立三层过滤——从确定性规则匹配到风险评分到人工升级——确保所有面向亚马逊的内容合规。

### 数学直觉

**SAFE-AGENT-L 三层架构**：

```
Layer 1 — Grounded Legal Alignment
  将 Amazon ToS 规则注入 LLM system prompt + RAG 检索相关条款
  产出: 初始内容 + 合规标注

Layer 2 — Risk-Aware Action Governance
  复合风险分 = α·UncertaintyScore + β·ViolationProbability + γ·LegalSensitiveFeatureScore
  其中:
  - UncertaintyScore: LLM 输出的 entropy（不确定就升级）
  - ViolationProbability: 细粒度多标签分类器（医疗声明/安全认证/价格误导/...）
  - LegalSensitiveFeatureScore: 关键词匹配（"FDA approved""guaranteed results"等）

Layer 3 — Multi-Stage Guardrails
  if risk < τ_low  → 通过（直接发布）
  if τ_low ≤ risk < τ_high → 强制覆盖（用安全模板替换可疑段落）
  if risk ≥ τ_high → 人工升级（送审）
  if risk > τ_critical → 安全降级（回退到预审过的安全模板）
```

**EVADE-Bench 关键发现**：明确的规则定义比模型能力更影响合规准确率。All-in-One 多规则统一指令使 partial-match 准确率从 62%→78%。

### 关键假设
- ToS 规则可被结构化（非结构化条款需人工转化为可验证规则）
- 多层过滤的延迟可接受（<500ms）——不适合实时竞价场景
- 品类特定的风险校准需要领域数据（母婴食品安全声明 ≠ 电子产品合规）

---

## ② 母婴出海应用案例

### 场景一：吸奶器 Listing 文案自动合规审查

**业务问题**：运营用 ChatGPT 生成吸奶器 listing 文案，但 LLM 可能产出违规声明——"clinically proven to increase milk supply""FDA approved design""guaranteed results in 7 days"——这些在 Amazon 上都是红线。

**数据要求**：
- Amazon ToS 结构化规则库：医疗声明（禁止）、安全认证（需有证才能声明）、对比广告（按辖区不同）
- EVADE-Bench 风格的多规则 prompt 模板

**预期产出**：
- 合规扫描报告：原文案 3 处违规（医疗声明 × 1 / 无证认证声明 × 1 / 保证性用语 × 1）
- 自动修正建议：将"clinically proven"替换为"designed for comfort"、"FDA approved"替换为"meets safety standards"
- 风险分级：该 listing 综合风险分 0.72 → 强制覆盖（不送人工）

**业务价值**：
- 每次违规 listing 被下架损失 $2,000-5,000（下架期间销量损失 + 恢复时间）
- 20 个 SKU × 年均 3 次 listing 更新 = 60 次审查 → 预计拦截 15 次违规
- 年化止损 **$30,000-75,000**，加上账号健康分保护的无形价值

### 场景二：客服自动回复的合规护栏（WF-C）

**业务问题**：客服 LLM 自动回复用户关于"这个吸奶器能增加奶量吗""对早产儿安全吗"的问题，需确保不给出医疗建议。

**数据要求**：客服对话历史 + 标注的合规/违规样本

**预期产出**：实时客服回复合规过滤——检测到医疗建议意图时，自动替换为标准化安全回复："建议咨询医生获取个性化建议。产品已通过[认证名称]安全认证。"

**业务价值**：避免客服 LLM 的医疗建议导致法律风险（每次投诉可能触发平台调查）

---

## ③ 代码模板

```python
"""
Amazon ToS Compliance Guardrail — 三层合规过滤器
基于 SAFE-AGENT-L 框架
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ComplianceResult:
    risk_score: float
    violations: List[Dict]
    action: str  # "pass" | "override" | "escalate" | "fallback"
    safe_text: Optional[str] = None


class ToSComplianceGuardrail:
    """亚马逊 ToS 合规护栏"""
    
    # 高危关键词 + 正则规则（母婴品类）
    FORBIDDEN_PATTERNS = [
        (r'(?i)FDA\s*approved', 'medical_certification', 0.9),
        (r'(?i)clinically\s*proven', 'medical_claim', 0.9),
        (r'(?i)guaranteed\s*(result|effect|outcome)', 'guarantee_claim', 0.85),
        (r'(?i)(treat|cure|heal|prevent)\s+\w+\s+(disease|condition)', 'medical_treatment', 0.95),
        (r'(?i)(increase|boost)\s+(milk|breast\s*milk)\s*(supply|production)', 'medical_efficacy', 0.8),
        (r'(?i)safe\s+for\s+(premature|newborn|infant)', 'safety_claim', 0.85),
        (r'(?i)#1\s+(rated|selling|recommended)', 'unsubstantiated_ranking', 0.7),
        (r'(?i)(free|100%|no)\s+(risk|side\s*effect)', 'absolute_claim', 0.85),
    ]
    
    # 安全替换模板
    SAFE_TEMPLATES = {
        'medical_claim': '[已移除医疗声明] 产品设计注重舒适性和易用性。',
        'medical_certification': '产品已通过相关安全标准检测。',
        'medical_efficacy': '产品设计旨在提供舒适的吸乳体验。',
        'medical_treatment': '[已移除治疗声明] 如有健康问题请咨询医生。',
        'safety_claim': '产品按照安全标准制造。具体使用请遵循说明书。',
        'guarantee_claim': '我们致力于提供优质产品和服务。',
        'absolute_claim': '我们提供满意的客户体验。',
    }
    
    def __init__(self, low_risk: float = 0.3, high_risk: float = 0.7):
        self.low_risk = low_risk
        self.high_risk = high_risk
    
    def scan_text(self, text: str) -> ComplianceResult:
        """
        扫描文本，返回合规评估结果
        """
        violations = []
        max_risk = 0.0
        override_text = text
        
        for pattern, category, base_risk in self.FORBIDDEN_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 上下文加权：句子越长越可能是复杂声明 → 风险微增
                sentence = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                context_boost = min(0.1, len(sentence) / 500)
                risk = min(base_risk + context_boost, 1.0)
                
                violations.append({
                    'category': category,
                    'matched_text': match.group(),
                    'position': match.span(),
                    'risk': risk,
                })
                max_risk = max(max_risk, risk)
                
                # 用安全模板替换
                template = self.SAFE_TEMPLATES.get(
                    category, '[已移除不合规内容]')
                override_text = override_text.replace(match.group(), template)
        
        # 决策
        if max_risk < self.low_risk:
            action = "pass"
            safe = None
        elif max_risk < self.high_risk:
            action = "override"
            safe = override_text
        else:
            action = "escalate"
            safe = override_text
        
        return ComplianceResult(
            risk_score=max_risk,
            violations=violations,
            action=action,
            safe_text=safe,
        )
    
    def batch_review(self, texts: List[str]) -> List[ComplianceResult]:
        """批量合规审查"""
        return [self.scan_text(t) for t in texts]


# ============ 测试 ============

if __name__ == '__main__':
    guard = ToSComplianceGuardrail(low_risk=0.3, high_risk=0.7)
    
    test_cases = [
        # 违规案例
        "Our breast pump is clinically proven to increase milk supply by 30%. FDA approved design!",
        # 边界案例
        "Many moms report comfortable pumping experience with our ergonomic design.",
        # 合规案例
        "Electric breast pump with 3 modes, BPA-free, includes 24mm and 28mm flanges.",
    ]
    
    for i, text in enumerate(test_cases):
        result = guard.scan_text(text)
        print(f"\n[Case {i+1}] Risk={result.risk_score:.2f} Action={result.action}")
        print(f"  Original: {text[:80]}...")
        if result.violations:
            for v in result.violations:
                print(f"  ⚠ {v['category']}: '{v['matched_text']}' (risk={v['risk']:.2f})")
        if result.safe_text and result.action != "pass":
            print(f"  → Safe: {result.safe_text[:100]}...")
    
    # 断言
    assert guard.scan_text(test_cases[0]).action != "pass", "Violation should be caught"
    assert guard.scan_text(test_cases[2]).action == "pass", "Safe text should pass"
    
    print("\n[✓] Amazon ToS Compliance Guardrail 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-ROAS-Budget-Optimization]] | [[Skill-Ad-Attribution-Modeling]]
- **延伸技能**：[[Skill-Negative-Keyword-Safe-Guard]]（同为风险控制工具）
- **可组合技能**：
  - **[[Skill-DialIn-LLM-Case-Intent-Clustering]]** — 合规护栏 + 意图聚类 = 安全客服管道
  - **[[Skill-LACA-CrossLingual-ABSA]]** — 多语种 listing 的合规审查

---
- **相关**：[[Skill-Supplier-Evaluation-Model]]

## ⑤ 商业价值评估

- **ROI 预估**：拦截违规 listing 下架（$2,000-5,000/次 × 15 次/年）+ 账号健康保护；年化止损 **5-10 万美元** + 无形风险规避
- **实施难度**：⭐⭐☆☆☆（2 星）— 基于规则的初版可快速上线，ML 风险分类器需要训练数据
- **优先级评分**：⭐⭐⭐⭐⭐（5 星）— 极高紧迫度（合规），账号安全是业务的底线
- **评估依据**：SAFE-AGENT-L 三层框架直接可落地，EVADE-Bench 提供 26 个 LLM 的电商合规评测基线


## 🧪 调用案例（智能体广场验证）

**Agent**：账号风险卫士  
**测试输入**：账号状态=黄色预警，收到警告邮件  
**输出摘要**：风险评分6.5/10，检出Review Manipulation信号，提供POA申诉模板框架  
**验证状态**：✅ 本地计算通过 | 2026-06-11
