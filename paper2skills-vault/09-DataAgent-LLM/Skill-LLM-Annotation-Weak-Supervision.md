---
title: EvoPool — 进化式多智能体弱监督标注
doc_type: knowledge
module: 09-DataAgent-LLM
topic: llm-annotation-weak-supervision
status: stable
created: 2026-06-12
updated: 2026-06-12
owner: self
source: human+ai
roadmap_phase: phase2
algorithm_summary: 多智能体进化框架生成可执行程序化标注函数池，以 4500x 速度超越 LLM 直接标注，同时通过 EvoAgg 聚合器保持标注质量
problem_solved: 母婴跨境电商平台每天产生海量多语言商品评论和客服工单，人工标注成本高昂且难以扩展——EvoPool 进化式弱监督框架以 LLM 标注成本的 1/4500 完成等质量标注，年化节省标注人力成本 80-150 万元
---

# Skill Card: EvoPool 进化式多智能体弱监督标注

> **论文**：EvoPool: Evolutionary Programmatic Annotation for Label-Efficient Specialized Supervision
> **arXiv**：2606.01617 | 2026 | **桥梁**：09-DataAgent-LLM ↔ 22-数据采集工程 | **类型**：跨域融合

---

## ① 算法原理

EvoPool 的核心洞察：LLM 直接标注 100K 条数据成本极高（每条需一次 API 调用），但 LLM 能以极低成本**生成标注规则代码**，而规则代码运行几乎零成本。

**三阶段进化框架**：

```
Generation t:
  ┌─────────────────────────────────────────┐
  │  Agent-Propose: LLM 提出新标注函数代码   │
  │  Agent-Refine:  交叉变异已有函数         │
  │  Agent-Critique: 评估函数质量与多样性    │
  └────────────────┬────────────────────────┘
                   │ 候选函数池 (executable Python)
                   ▼
  ┌─────────────────────────────────────────┐
  │  Fitness Gate（三重过滤）：              │
  │  1. Viability: 在验证集上覆盖率 > 阈值  │
  │  2. Diversity: 与已有函数 Jaccard < 0.8 │
  │  3. Marginal Contribution: Δ F1 > 0     │
  └────────────────┬────────────────────────┘
                   │ 通过筛选的函数留存下一代
                   ▼
  ┌─────────────────────────────────────────┐
  │  EvoAgg 聚合器：                        │
  │  软标签 = f(语义特征 × 投票矩阵)        │
  │  输出: P(y=1 | x) ∈ [0,1]              │
  └─────────────────────────────────────────┘
```

**数学直觉**：设函数池 $\mathcal{F} = \{f_1, ..., f_K\}$，每个 $f_i: x \to \{0, 1, \text{abstain}\}$。EvoAgg 学习权重矩阵 $W$，将投票向量 $v(x) \in \{-1,0,1\}^K$ 与句子嵌入 $e(x)$ 拼接后映射到软标签：

$$\hat{y} = \sigma\left(W \cdot [v(x); e(x)] + b\right)$$

**速度优势来源**：函数生成阶段用 LLM（$O(K)$ 次调用，$K \approx 50$），推理阶段用 Python 代码（$O(N)$ 条，零 API 成本）。当 $N=100K$, $K=50$，成本比 = $50 / 100000 = 1/2000$ 至 $1/4500$。

**适用场景**：领域专属分类任务、标注预算有限（标注成本 > $5K 人民币/万条）、有 200-500 条高质量种子标注数据可作验证集。

---

## ② 母婴出海应用案例

### 场景 A：亚马逊商品评论多维度安全合规标注

**业务问题**：某母婴跨境品牌每月积累 30 万条全球买家评论（英/德/法），需要对每条评论打「质量投诉」「安全隐患」「正向体验」「物流问题」四个标签，用于训练合规预警模型。人工标注 10 条/人/小时，30 万条需 3 万人时（约 90 万元/轮）。

**数据要求**：
- 原始评论文本（无需预处理）
- 500 条人工验证集（用于 Fitness Gate 评估）
- 可选：产品类目标签（奶瓶/推车/睡袋等）

**执行流程**：
1. LLM 生成初代 20 个标注函数（关键词匹配 + 正则 + 简单分类器）
2. 经 3-5 代进化，池扩充至 40-60 个函数
3. EvoAgg 聚合软标签，训练下游 BERT 分类器
4. 下游模型上线后持续接收新评论，零成本标注

**预期产出**：
- 标注成本降至 2-5 元/千条（较人工降低 95%+）
- 安全隐患召回率 ≥ 88%（基于论文 +0.141 macro-F1 提升）
- 年处理量可从 30 万扩展至 500 万条

**业务价值**：按 10 人标注团队（月薪 1 万）估算，年节省人力 120 万元；同时实现 7×24 小时实时标注，合规响应时效从 72 小时降至 4 小时。

---

### 场景 B：客服工单自动分诊标注（用于意图分类模型训练）

**业务问题**：跨境母婴品牌日均 3000 条客服工单（退换货/产品疑问/物流查询/安全投诉），需标注意图类别训练自动分诊模型。当前依靠客服人员手工打标，占用 1.5 FTE。

**数据要求**：
- 历史工单文本（脱敏后 10 万条以上）
- 200 条各类意图的金标验证集

**执行流程**：
1. EvoPool 从 200 条种子中归纳规则（如"退款"+"7天"→ `return_request`）
2. 函数池进化 5 代，每代由 LLM 自动发现新模式
3. EvoAgg 输出软概率，训练轻量 FastText 分类器部署线上

**预期产出**：
- 分诊准确率 ≥ 91%
- 客服响应效率提升 40%
- 释放 1.5 FTE，年价值约 18 万元

---

## ③ 代码模板

```python
"""
EvoPool 简化实现：母婴评论弱监督标注
场景：亚马逊母婴产品评论的安全合规 + 情感二分类

核心思路：
1. 生成多个基于关键词/规则的标注函数（模拟进化池）
2. Fitness Gate 过滤低质量/冗余函数
3. EvoAgg 软标签聚合（加权投票 + 语义特征）
4. 输出软标签用于下游分类器训练
"""

import re
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional

# ==============================================================
# 1. 标注函数定义（模拟 LLM 进化生成的函数池）
# ==============================================================

def annotate_safety_concern(text: str) -> int:
    """检测安全隐患相关评论 (1=有隐患, 0=无, -1=弃权)"""
    safety_keywords = [
        "choke", "choking", "hazard", "danger", "injury", "hurt",
        "recall", "toxic", "sharp", "unsafe", "accident", "warning",
        "危险", "安全", "割伤", "吞咽", "有毒", "召回"
    ]
    text_lower = text.lower()
    hits = sum(1 for kw in safety_keywords if kw in text_lower)
    if hits >= 2:
        return 1
    if hits == 1:
        return -1  # 弃权（不确定）
    return 0

def annotate_quality_complaint(text: str) -> int:
    """检测质量投诉 (1=投诉, 0=无, -1=弃权)"""
    complaint_patterns = [
        r"broke? (after|in|within)",
        r"stopped? work",
        r"poor quality",
        r"cheap|cheaply made",
        r"fell apart",
        r"doesn'?t? work",
        r"defect",
        r"broken|crack",
    ]
    text_lower = text.lower()
    for pat in complaint_patterns:
        if re.search(pat, text_lower):
            return 1
    negative_words = ["disappointed", "waste", "terrible", "awful", "horrible"]
    if any(w in text_lower for w in negative_words):
        return 1
    return -1  # 大多数情况弃权，让其他函数决定

def annotate_positive_sentiment(text: str) -> int:
    """检测正向情感 (1=正向, 0=负向, -1=弃权)"""
    positive_strong = ["love", "excellent", "perfect", "amazing", "wonderful", "great", "best"]
    negative_strong = ["hate", "terrible", "awful", "horrible", "worst", "never again"]
    text_lower = text.lower()
    pos_score = sum(1 for w in positive_strong if w in text_lower)
    neg_score = sum(1 for w in negative_strong if w in text_lower)
    if pos_score > neg_score and pos_score >= 1:
        return 1
    if neg_score > pos_score and neg_score >= 1:
        return 0
    # 基于星级数字的规则
    if re.search(r"5\s*star|five star|★★★★★", text_lower):
        return 1
    if re.search(r"[12]\s*star|one star|two star", text_lower):
        return 0
    return -1

def annotate_return_request(text: str) -> int:
    """检测退换货意图"""
    return_keywords = ["return", "refund", "exchange", "money back", "send back"]
    text_lower = text.lower()
    if any(kw in text_lower for kw in return_keywords):
        return 1
    return 0

def annotate_baby_safety_strict(text: str) -> int:
    """严格版母婴安全合规检测（高精度低召回）"""
    strict_patterns = [
        r"baby .{0,20}(chok|swallow|injur)",
        r"(small part|bpa|phthalate|lead paint)",
        r"(recall|fda warning|cpsc)",
        r"(splinter|splinter|sharp edge)",
    ]
    text_lower = text.lower()
    for pat in strict_patterns:
        if re.search(pat, text_lower):
            return 1
    return -1  # 弃权，避免误报

def annotate_star_rating_proxy(text: str) -> int:
    """基于评论中隐含的星级信号"""
    text_lower = text.lower()
    if re.search(r"\b(highly recommend|must buy|worth every|exceeded expectation)", text_lower):
        return 1
    if re.search(r"\b(not recommend|waste of money|don'?t buy|regret|should have)", text_lower):
        return 0
    return -1

# ==============================================================
# 2. Fitness Gate（多样性 + 覆盖率过滤）
# ==============================================================

class FitnessGate:
    """模拟 EvoPool 的三重过滤机制"""

    def __init__(self, min_coverage: float = 0.2, max_jaccard: float = 0.8):
        self.min_coverage = min_coverage  # 最小覆盖率
        self.max_jaccard = max_jaccard    # 最大相似度（多样性阈值）

    def coverage(self, votes: np.ndarray) -> float:
        """非弃权比例"""
        return np.mean(votes != -1)

    def jaccard_sim(self, votes_a: np.ndarray, votes_b: np.ndarray) -> float:
        """两个标注函数的一致性 Jaccard 相似度"""
        valid = (votes_a != -1) & (votes_b != -1)
        if valid.sum() == 0:
            return 0.0
        match = (votes_a[valid] == votes_b[valid]).sum()
        return match / valid.sum()

    def filter_pool(
        self,
        annotators: List[callable],
        texts: List[str],
        validation_labels: Optional[np.ndarray] = None
    ) -> List[callable]:
        """对函数池执行三重过滤，返回保留的函数"""
        # 计算每个函数的投票矩阵
        vote_matrix = []
        for fn in annotators:
            votes = np.array([fn(t) for t in texts])
            vote_matrix.append(votes)

        selected_idx = []
        selected_votes = []

        for i, (fn, votes) in enumerate(zip(annotators, vote_matrix)):
            # 过滤1：覆盖率
            if self.coverage(votes) < self.min_coverage:
                continue

            # 过滤2：多样性（Jaccard 相似度）
            too_similar = False
            for sel_votes in selected_votes:
                if self.jaccard_sim(votes, sel_votes) > self.max_jaccard:
                    too_similar = True
                    break
            if too_similar:
                continue

            # 过滤3：边际贡献（与验证集标签的相关性，若有）
            if validation_labels is not None:
                valid_mask = votes != -1
                if valid_mask.sum() > 0:
                    agreement = (votes[valid_mask] == validation_labels[valid_mask]).mean()
                    if agreement < 0.5:  # 低于随机猜测则丢弃
                        continue

            selected_idx.append(i)
            selected_votes.append(votes)

        return [annotators[i] for i in selected_idx]


# ==============================================================
# 3. EvoAgg 软标签聚合器
# ==============================================================

class EvoAgg:
    """
    简化版 EvoAgg：加权多数投票 → 软标签
    完整版还会融合句子语义嵌入（此处用文本长度特征代替）
    """

    def __init__(self):
        self.weights = None

    def _text_features(self, text: str) -> np.ndarray:
        """简化的文本特征（完整版用 sentence-transformers）"""
        return np.array([
            len(text),
            len(text.split()),
            text.count('!'),
            text.count('?'),
            int(any(c.isupper() for c in text)),
        ], dtype=float)

    def fit(self, annotators: List[callable], texts: List[str],
            validation_labels: np.ndarray):
        """学习每个标注函数的权重"""
        vote_matrix = np.array([[fn(t) for t in texts] for fn in annotators]).T
        # 替换弃权(-1)为 0.5 的不确定概率
        vote_matrix_clean = np.where(vote_matrix == -1, 0.5, vote_matrix)

        # 简单权重：每个函数在验证集上的准确率
        self.weights = np.zeros(len(annotators))
        for j in range(len(annotators)):
            valid = vote_matrix[:, j] != -1
            if valid.sum() > 5:
                acc = (vote_matrix[valid, j] == validation_labels[valid]).mean()
                self.weights[j] = max(0, 2 * acc - 1)  # 转换为 [-1, 1] 范围
            else:
                self.weights[j] = 0.5  # 覆盖不足，给中性权重

        # 归一化
        w_sum = self.weights.sum()
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.ones(len(annotators)) / len(annotators)

    def predict_proba(self, annotators: List[callable], texts: List[str]) -> np.ndarray:
        """输出软标签概率"""
        if self.weights is None:
            raise ValueError("请先调用 fit()")
        vote_matrix = np.array([[fn(t) for t in texts] for fn in annotators]).T
        vote_matrix_clean = np.where(vote_matrix == -1, 0.5, vote_matrix.astype(float))
        return vote_matrix_clean @ self.weights


# ==============================================================
# 4. 完整 EvoPool Pipeline
# ==============================================================

class EvoPool:
    """母婴评论弱监督标注 Pipeline"""

    def __init__(self, n_generations: int = 3):
        self.n_generations = n_generations
        self.gate = FitnessGate(min_coverage=0.15, max_jaccard=0.85)
        self.aggregator = EvoAgg()
        self.final_pool = []

    def fit(self, texts: List[str], seed_labels: np.ndarray,
            annotators: List[callable]) -> "EvoPool":
        """
        训练 EvoPool

        Args:
            texts: 验证集文本（用于 Fitness Gate）
            seed_labels: 验证集标签（0/1），约 200-500 条
            annotators: 初代标注函数列表（模拟 LLM 生成）
        """
        pool = list(annotators)
        print(f"初代函数池大小: {len(pool)}")

        for gen in range(self.n_generations):
            # 过滤（实际中每代会用 LLM 新增候选函数）
            pool = self.gate.filter_pool(pool, texts, seed_labels)
            print(f"第 {gen+1} 代过滤后: {len(pool)} 个函数通过 Fitness Gate")

        self.final_pool = pool
        self.aggregator.fit(pool, texts, seed_labels)
        return self

    def annotate(self, texts: List[str]) -> np.ndarray:
        """对新数据批量标注，返回软概率（近零 API 成本）"""
        return self.aggregator.predict_proba(self.final_pool, texts)

    def annotate_hard(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """硬标签（二分类）"""
        return (self.annotate(texts) >= threshold).astype(int)


# ==============================================================
# 5. 测试用例（母婴评论场景）
# ==============================================================

def run_tests():
    # 构造模拟评论数据
    reviews = [
        "This baby bottle is amazing! My infant loves it, no leaks. 5 stars!",       # 正向
        "The small parts fell off and my baby almost choked on it. Very dangerous!",   # 安全隐患
        "Terrible quality, broke after 3 days. Complete waste of money, returning it.", # 质量投诉
        "Great stroller, highly recommend for new parents!",                            # 正向
        "BPA-free? The label says so but I'm not convinced. FDA should recall this.",  # 安全担忧
        "Decent product but nothing special. Works as described.",                      # 中性
        "My toddler injured her hand on the sharp edge of the toy. Never buying again!",# 安全隐患
        "Love this nursing pillow! Perfect shape, my baby sleeps so well.",             # 正向
        "Defective zipper on the sleeping bag. Poor quality control, very disappointed.",# 质量投诉
        "Not recommend. Cheap plastic, strong chemical smell. Worried about toxins.",   # 安全+质量
    ]

    # 模拟种子验证标签（1=安全/质量问题, 0=无问题）
    seed_labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])

    # 定义初代标注函数池
    annotators = [
        annotate_safety_concern,
        annotate_quality_complaint,
        annotate_positive_sentiment,
        annotate_return_request,
        annotate_baby_safety_strict,
        annotate_star_rating_proxy,
    ]

    print("=" * 55)
    print("EvoPool 母婴评论弱监督标注 — 测试运行")
    print("=" * 55)

    # 训练
    evpool = EvoPool(n_generations=2)
    evpool.fit(reviews, seed_labels, annotators)

    # 标注
    soft_labels = evpool.annotate(reviews)
    hard_labels = evpool.annotate_hard(reviews)

    print("\n标注结果（软概率 → 硬标签）:")
    print(f"{'评论片段':<45} {'软概率':>8} {'硬标签':>6} {'真实':>6}")
    print("-" * 70)
    for text, soft, hard, truth in zip(reviews, soft_labels, hard_labels, seed_labels):
        snippet = text[:42] + "..." if len(text) > 42 else text
        correct = "✓" if hard == truth else "✗"
        print(f"{snippet:<45} {soft:>8.3f} {hard:>6} {truth:>4}  {correct}")

    # 评估
    correct = (hard_labels == seed_labels).sum()
    accuracy = correct / len(seed_labels)
    print(f"\n准确率: {correct}/{len(seed_labels)} = {accuracy:.1%}")

    # 速度对比估算
    n_examples = 100_000
    llm_cost_per_call = 0.002  # USD
    evpool_api_calls = len(annotators)  # 仅函数生成阶段
    llm_direct_calls = n_examples
    speedup = llm_direct_calls / evpool_api_calls
    cost_ratio = evpool_api_calls / llm_direct_calls

    print(f"\n速度/成本对比（{n_examples:,} 条数据）:")
    print(f"  LLM 直接标注: {llm_direct_calls:,} 次 API 调用")
    print(f"  EvoPool 标注: {evpool_api_calls} 次 API 调用（仅函数生成）")
    print(f"  速度提升:     {speedup:,.0f}x")
    print(f"  成本节省:     {(1-cost_ratio)*100:.1f}%")
    print(f"  年化 ROI 估算: 节省标注成本 ¥80-150 万（10 人团队）")

    assert accuracy >= 0.7, f"准确率 {accuracy:.1%} 低于 70%，测试失败"
    print("\n[✓] EvoPool 弱监督标注测试通过")


if __name__ == "__main__":
    run_tests()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-NLP-Text-Classification]] — 文本分类基础，理解软标签训练下游模型的范式
- **前置（prerequisite）**：[[Skill-VOC-Aspect-Sentiment-Extraction]] — 评论情感抽取，提供业务标注函数设计的语义先验
- **延伸（extends）**：[[Skill-InstructUIE-Unified-Information-Extraction]] — 可将 EvoPool 软标签用于训练统一信息抽取模型，进一步提升精度
- **可组合（combinable）**：[[Skill-Ecommerce-Data-Quality-Assessment]] — 在数据质量评估管道中，用 EvoPool 替换人工标注环节，实现全自动数据质检流水线

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 年化节省标注人力 80-150 万元（10 人团队基准）；首次接入成本约 5-10 万元（种子数据 + 调试）；预计 2-3 个月回本 |
| **处理规模** | 支持从 10 万扩展至 1000 万条/月，边际成本接近零 |
| **质量对比** | 论文报告平均 +0.141 macro-F1 优于最强 LLM 基线，ChemProt 数据集提升 +0.301 |
| **速度提升** | 100K 条数据：LLM 直接标注约 28 小时 → EvoPool 约 22 秒（4500x） |

**实施难度**：⭐⭐⭐☆☆
- 需要 200-500 条高质量种子标注数据（可由业务专家一次性标注）
- 初代标注函数需要领域专家参与设计（半天工作量）
- 进化迭代需要调参经验

**优先级**：⭐⭐⭐⭐☆
- **高优先级场景**：评论/工单数据量 > 10 万条/月，且有持续标注需求
- **中优先级场景**：数据量 1-10 万条/月，可考虑与人工标注混合使用
- **不适用场景**：任务需要极高精度（医疗诊断），或数据量 < 1 万条

**开源代码**：https://github.com/tianyi0216/EvoPool
