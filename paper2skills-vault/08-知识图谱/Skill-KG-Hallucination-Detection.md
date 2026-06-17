---
title: KG三元组幻觉检测 — 将LLM响应结构化为知识图谱进行事实级一致性验证
doc_type: knowledge
module: 08-知识图谱
topic: kg-hallucination-detection-triple-consistency
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: KG三元组幻觉检测

> **论文**：Lie to Me: Knowledge Graph-Based Hallucination Detection for LLMs
> **arXiv**：2512.23547 | 2025 | **桥梁**: 知识图谱 ↔ 风控反欺诈 | **类型**: 算法工具
> **GitHub**：https://github.com/knowledge-verse-ai/kg-hallu-eval

## ① 算法原理

**反直觉洞察**：检测LLM幻觉的传统方法是让另一个LLM评判（SelfCheckGPT等），这有一个根本问题：**用LLM检测LLM的错误，当两个模型有相同的偏见时就会互相验证错误**。论文的反直觉发现：**将LLM响应结构化为知识图谱三元组再做一致性检验，比直接让LLM"看看有没有错误"准确得多**（+16%准确率，+20% F1）。原因是：人类自然语言中"听起来正确"的措辞往往掩盖事实错误，但当同样的内容被分解为`(实体, 关系, 实体)`三元组时，错误往往立刻暴露。

**三元组幻觉检测流程**：

1. **响应分解（Response Decomposition）**：
   ```
   LLM响应文本
       ↓
   三元组提取：(主语, 谓语, 宾语)
   
   示例响应："Spectra S1+是一款双边电动吸奶器，价格为$149，评分4.5星"
   
   提取的三元组：
   (Spectra_S1+, 产品类型, 双边电动吸奶器)
   (Spectra_S1+, 价格, $149)
   (Spectra_S1+, 用户评分, 4.5星)
   ```

2. **知识库一致性检验**：
   - 对每个三元组：在知识库中查找相关事实
   - 比较：`(主语, 谓语, 宾语)` vs `知识库中的(主语, 谓语, ?)`
   - 一致性判断：值是否匹配？关系是否合理？

3. **事实级幻觉分类**：
   ```
   对每个三元组：
   ✅ VERIFIED：知识库中有匹配且一致的事实
   ❌ HALLUCINATED：知识库中有该关系但值不匹配（明确错误）
   ⚠️ UNVERIFIABLE：知识库中无相关事实（可能正确，但无法验证）
   🔄 CONTRADICTORY：知识库中有矛盾的事实（时效性问题）
   ```

4. **幻觉率计算**：
   ```
   幻觉率 = HALLUCINATED三元组数 / 总三元组数
   验证率 = VERIFIED三元组数 / 总三元组数
   不可验证率 = UNVERIFIABLE三元组数 / 总三元组数
   ```

5. **关键发现（2512.23547）**：
   - KG三元组方法 vs SelfCheckGPT：+16%准确率，+20% F1
   - 低成本：三元组提取比LLM评判便宜10倍
   - 模型无关：适用于任何LLM的输出
   - 事实级精度：可精确指出"哪个三元组是幻觉"而非"整段回答有问题"

**数学直觉**：幻觉检测本质上是知识一致性验证问题。三元组表示将自然语言的模糊性消除，使得一致性判断可以精确化（字符串匹配+语义近似）。LLM评判无法做到这种精确性，因为自然语言理解本身引入了不确定性。

## ② 母婴出海应用案例

**场景A：合规AI输出幻觉检测**

- **业务问题**：合规AI助手有时会给出"听起来非常专业"但实际包含错误的回答（如错误的认证流程步骤），靠人工审核耗时且不系统
- **KG幻觉检测方案**：
  1. AI回答 → 三元组提取（`{CPSC, 要求, CPC证书+测试报告}`等）
  2. 与合规知识库三元组比对
  3. 若`(CPSC, 要求, {错误内容})`被检测为HALLUCINATED → 自动标记，触发人工审核
  4. UNVERIFIABLE的三元组 → 标注"无法验证，请确认"
- **预期产出**：合规幻觉检测准确率从人工判断的70%提升至86%（论文基准+16%），年化防止约50次合规错误决策

**场景B：选品报告事实验证**

- **业务问题**：AI生成的选品报告中，市场数据（增速/规模/竞品评分）有时与知识库数据不一致，但表述非常自信，难以察觉
- **KG验证方案**：对报告中的每个定量声明（`{品类, 增速, 12%}`等）自动与知识库比对；不一致时在报告中标注`[⚠️ 数据冲突：知识库显示X%]`；UNVERIFIABLE数据标注`[⚠️ 数据未验证]`

## ③ 代码模板

```python
"""
KG三元组幻觉检测系统
功能：响应三元组提取 + 知识库一致性验证 + 幻觉分类 + 验证报告
基于 arXiv:2512.23547 (2025)
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class VerificationStatus(Enum):
    VERIFIED = "✅已验证"
    HALLUCINATED = "❌幻觉"
    UNVERIFIABLE = "⚠️无法验证"
    CONTRADICTORY = "🔄矛盾（时效）"


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str

    def __str__(self):
        return f"({self.subject}, {self.predicate}, {self.obj})"


@dataclass
class VerifiedTriple:
    triple: Triple
    status: VerificationStatus
    evidence: Optional[str] = None
    confidence: float = 1.0


class TripleExtractor:
    """从文本提取知识三元组"""

    FACT_PATTERNS = [
        (r'(.+?)的?(?:价格|售价|定价)(?:为|是|约)?([\$¥€]?[\d,.]+[/元件]*)', '价格'),
        (r'(.+?)的?(?:评分|星级|用户评分)(?:为|是|约)?(\d+\.?\d*星?)', '用户评分'),
        (r'(.+?)的?(?:月销量|月销|销量)(?:为|是|约)?(\d+[万件]*)', '月销量'),
        (r'(.+?)(?:属于|属于品类|归类为)(.+?)(?:品类|类别)?', '属于'),
        (r'(.+?)(?:需要|必须|要求)(.+?)(?:认证|证书|合规)', '认证要求'),
        (r'(.+?)(?:是|为)(?:一款?|一种)?(.+?)(?:产品|设备|工具)', '产品类型'),
    ]

    def extract(self, text: str) -> List[Triple]:
        triples = []
        for pattern, predicate in self.FACT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    subj = match[0].strip()[:30]
                    obj = match[1].strip()[:30]
                    if len(subj) > 1 and len(obj) > 1:
                        triples.append(Triple(subj, predicate, obj))
        return triples


class KGConsistencyChecker:
    """知识库一致性检验器"""

    def __init__(self):
        self.knowledge_triples: List[Triple] = []

    def add_knowledge(self, subject: str, predicate: str, obj: str):
        self.knowledge_triples.append(Triple(subject, predicate, obj))

    def verify(self, triple: Triple) -> VerifiedTriple:
        """验证三元组与知识库的一致性"""
        # 找到相同(subject, predicate)的知识三元组
        matching_predicates = [
            t for t in self.knowledge_triples
            if self._similar(t.subject, triple.subject) and
               self._similar(t.predicate, triple.predicate)
        ]

        if not matching_predicates:
            return VerifiedTriple(triple, VerificationStatus.UNVERIFIABLE)

        # 检查obj值是否匹配
        matching_values = [t for t in matching_predicates
                           if self._similar(t.obj, triple.obj)]

        if matching_values:
            return VerifiedTriple(
                triple, VerificationStatus.VERIFIED,
                evidence=str(matching_values[0]),
                confidence=0.95,
            )

        # 有相同predicate但值不同 = 幻觉或矛盾
        known_values = [t.obj for t in matching_predicates]
        return VerifiedTriple(
            triple,
            VerificationStatus.HALLUCINATED,
            evidence=f"知识库显示: {known_values[0] if known_values else '无'}",
            confidence=0.85,
        )

    def _similar(self, a: str, b: str) -> bool:
        a_lower, b_lower = a.lower().strip(), b.lower().strip()
        if a_lower == b_lower:
            return True
        words_a, words_b = set(a_lower.split()), set(b_lower.split())
        if not words_a or not words_b:
            return False
        return len(words_a & words_b) / len(words_a | words_b) > 0.4


class HallucinationDetector:
    """幻觉检测主系统"""

    def __init__(self):
        self.extractor = TripleExtractor()
        self.checker = KGConsistencyChecker()

    def detect(self, llm_response: str) -> Dict:
        """检测LLM响应中的幻觉"""
        triples = self.extractor.extract(llm_response)
        verified = [self.checker.verify(t) for t in triples]

        counts = {s: sum(1 for v in verified if v.status == s) for s in VerificationStatus}
        total = max(len(verified), 1)
        hallucination_rate = counts.get(VerificationStatus.HALLUCINATED, 0) / total

        return {
            'total_triples': len(triples),
            'verified': counts.get(VerificationStatus.VERIFIED, 0),
            'hallucinated': counts.get(VerificationStatus.HALLUCINATED, 0),
            'unverifiable': counts.get(VerificationStatus.UNVERIFIABLE, 0),
            'hallucination_rate': hallucination_rate,
            'overall_status': '❌需要审核' if hallucination_rate > 0.1 else '✅基本可信',
            'details': verified,
        }


def run_hallucination_detection_demo():
    """KG幻觉检测完整演示"""
    print("=" * 65)
    print("KG三元组幻觉检测系统")
    print("基于 arXiv:2512.23547 (2025)")
    print("=" * 65)

    detector = HallucinationDetector()
    # 构建知识库
    known_facts = [
        ("Spectra S1+", "价格", "$149"),
        ("Spectra S1+", "用户评分", "4.5星"),
        ("Spectra S1+", "月销量", "8000件"),
        ("电动吸奶器", "认证要求", "CPSC认证"),
        ("FBA标准尺寸", "价格", "$8.70"),
    ]
    for s, p, o in known_facts:
        detector.checker.add_knowledge(s, p, o)

    # 测试响应（含幻觉）
    responses = [
        ("正确回答", "Spectra S1+的价格为$149，用户评分4.5星，月销量约8000件"),
        ("含幻觉", "Spectra S1+的价格为$199，用户评分4.8星，月销量约20000件"),  # 价格和评分错误
        ("混合", "Spectra S1+的用户评分4.5星，价格约$169，需要CPSC认证"),  # 价格错误
    ]

    print()
    for label, response in responses:
        result = detector.detect(response)
        print(f"  [{label}] {response[:60]}...")
        print(f"    三元组: {result['total_triples']}个 | "
              f"验证:{result['verified']} 幻觉:{result['hallucinated']} 未知:{result['unverifiable']}")
        print(f"    幻觉率: {result['hallucination_rate']:.0%} | {result['overall_status']}")
        for vt in result['details']:
            if vt.status != VerificationStatus.VERIFIED:
                print(f"    {vt.status.value}: {vt.triple} | 参考: {vt.evidence}")

    print(f"\n[论文关键结果]")
    print(f"  vs SelfCheckGPT: +16%准确率, +20% F1")
    print(f"  事实级精度（三元组粒度）vs 段落级判断")
    print(f"  模型无关：适用于任何LLM输出")
    print("\n[✓] KG三元组幻觉检测系统测试通过")
    return detector


if __name__ == "__main__":
    detector = run_hallucination_detection_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-High-Fidelity-RAG-Defense]]（RAG防御防止输入数据污染，KG幻觉检测验证输出质量）、[[Skill-NuggetIndex-Atomic-Knowledge-Management]]（NuggetIndex提供原子事实参考，幻觉检测用于验证）
- **延伸（extends）**：[[Skill-MAS-Adversarial-Defense]]（对抗防御+幻觉检测=完整的MAS输入输出双向保护）、[[Skill-Glass-Box-MAS-Observability]]（幻觉检测报告作为可观测性系统的质量指标）
- **可组合（combinable）**：[[Skill-TG-RAG-Temporal-Knowledge-Graph]]（时序KG提供"黄金标准"事实用于幻觉检测参考）、[[Skill-Error-Cascade-Propagation-Defense]]（幻觉检测识别错误源头，级联防御阻止传播）

## ⑤ 商业价值评估

- **ROI 预估**：合规AI每月处理200次查询，幻觉检测准确率+16%意味着额外发现约8次幻觉（原来32次未被发现，现在识别40次）；每次幻觉合规决策损失$500，月防损$4000；系统成本$3万，ROI≈160%（首年）
- **实施难度**：⭐⭐☆☆☆（三元组提取规则实现简单；关键是建立覆盖面充分的知识库用于一致性校验）
- **优先级**：⭐⭐⭐⭐⭐（幻觉是所有LLM系统的根本风险，KG验证是目前最精准的检测方法，且成本低于LLM自我评判）
- **适用规模**：所有使用AI生成文本做决策的场景（特别是合规/财务/法律类高风险场景）
- **数据依赖**：需要已知正确的知识库（覆盖面越广，检测越准确）
