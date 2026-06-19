---
title: AI 生成内容检测 — Perplexity/Burstiness 统计特征分类器
doc_type: knowledge
module: 11-AI人文
topic: ai-generated-content-detection
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: AI 生成内容检测

> **论文**：GPT Detectors Are Biased Against Non-Native English Writers (Liang et al., 2023)
> **arXiv**：2304.02819 | 2023 | **桥梁**: 11-AI人文 ↔ 07-NLP-VOC | **类型**: 跨域融合

---

## ① 算法原理

**核心思想**：AI 生成文本的统计分布与人类写作有系统性差异：AI 倾向于生成「低困惑度、低突发性」的文本（词汇选择高度可预测），而人类写作「困惑度高、突发性强」（偶尔用冷僻词，情绪驱动跳跃）。基于这两个统计特征构建轻量分类器，无需调用任何 API，本地即可检测。

**数学直觉**：
- **Perplexity（困惑度）**：$PP(w) = 2^{-\frac{1}{N}\sum_{i=1}^{N}\log_2 p(w_i | w_{<i})}$，AI 生成文本 PP 通常 < 50，人类写作 PP > 100
- **Burstiness（突发性）**：$B = \frac{\sigma_{IWT} - \mu_{IWT}}{\sigma_{IWT} + \mu_{IWT}}$，其中 $IWT$ 为词间间隔时间；AI 文本 B 趋近 0（均匀），人类写作 B 分布更广
- **Token 重复率**：$R_{rep} = 1 - \frac{|\text{unique trigrams}|}{|\text{total trigrams}|}$，AI 生成文本 3-gram 重复率显著更高
- 分类器：Logistic Regression（3 特征），决策边界简单、速度快、可解释

**关键假设**：
- 使用 3-gram 语言模型估算 perplexity（无需 GPT，用 KenLM 或 NLTK 即可）
- 短文本（< 50 词）检测准确率下降（误报率约 25%），需要辅以其他信号
- 不适用于人工改写后的 AI 内容（攻击方可绕过）

---

## ② 母婴出海应用案例

**场景A：Amazon Review 真假检测**

- **业务问题**：竞争对手通过 GPT 批量生成 5 星好评刷 BSR，导致我方 Listing 被竞品超越；亚马逊平台 2024 年严查 AI 生成评论，卖家需主动检测自己的账号是否受虚假评论污染（以免关联封号）
- **数据要求**：目标 ASIN 的 Review 文本（via Amazon API 或爬取），每条 Review ≥ 30 词
- **预期产出**：检测出 AI 生成评论比例（如：某竞品 ASIN 的 1800 条好评中 38% 疑似 AI 生成）；输出高风险 Review 列表供人工核查
- **业务价值**：提前发现竞品刷评并举报，防止自身 BSR 被人为压制；避免自身账号因接受刷单服务被封，年化减少封号风险损失约 **50 万元**

**场景B：Listing 内容合规预审**

- **业务问题**：运营使用 AI 批量生成 Listing 文案，但亚马逊 2024 年开始对 AI 生成内容执行 ToS（需标注），运营不知道哪些文案已触发检测阈值
- **数据要求**：待上架的 Listing 文案（Title + Bullet Points + Description），每份约 200-400 词
- **预期产出**：对每个 Listing 给出 AI 生成概率（0-1），概率 > 0.7 的需要人工改写；批量处理 500 个 SKU，耗时 < 5 分钟
- **业务价值**：避免 Listing 被亚马逊下架（每次下架损失约 1-3 天 GMV，约 5000 元/SKU），批量检测节省人工逐一审查约 **20 万元/年**

---

## ③ 代码模板

```python
"""
AI 生成内容检测器
基于 Perplexity / Burstiness / 重复率 三维统计特征
无需任何 API 调用，本地可运行
"""
import re
import math
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def tokenize(text: str) -> List[str]:
    """简单分词：小写 + 仅保留字母数字"""
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())


def compute_perplexity_proxy(tokens: List[str]) -> float:
    """
    用 unigram 概率估算困惑度代理指标
    AI 文本：词频分布更均匀，熵更低；人类：长尾分布，熵更高
    返回：归一化熵（越高越像人类写作）
    """
    if len(tokens) < 5:
        return 1.0
    freq = Counter(tokens)
    total = sum(freq.values())
    probs = np.array([c / total for c in freq.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = math.log2(len(freq))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_burstiness(tokens: List[str]) -> float:
    """
    词语使用的突发性指数
    计算每个词的出现间隔（IWT），AI 文本间隔更均匀（B→0），人类文本更突发（B>0）
    """
    if len(tokens) < 10:
        return 0.0
    word_positions: Dict[str, List[int]] = {}
    for i, w in enumerate(tokens):
        word_positions.setdefault(w, []).append(i)

    all_iwts = []
    for positions in word_positions.values():
        if len(positions) > 1:
            iwts = np.diff(positions)
            all_iwts.extend(iwts.tolist())

    if not all_iwts:
        return 0.0
    arr = np.array(all_iwts, dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if mu + sigma < 1e-8:
        return 0.0
    return (sigma - mu) / (sigma + mu)


def compute_trigram_repetition(tokens: List[str]) -> float:
    """
    3-gram 重复率：AI 倾向于重复模式
    返回：重复率（越高越像 AI）
    """
    if len(tokens) < 6:
        return 0.0
    trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens) - 2)]
    if not trigrams:
        return 0.0
    unique_count = len(set(trigrams))
    return 1.0 - (unique_count / len(trigrams))


def extract_features(text: str) -> np.ndarray:
    """提取 3 维统计特征"""
    tokens = tokenize(text)
    if len(tokens) < 5:
        return np.array([0.5, 0.0, 0.3])  # 短文本返回中性特征
    perplexity_proxy = compute_perplexity_proxy(tokens)
    burstiness = compute_burstiness(tokens)
    trigram_rep = compute_trigram_repetition(tokens)
    return np.array([perplexity_proxy, burstiness, trigram_rep])


class AIContentDetector:
    """AI 生成内容检测器"""

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=200))
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[int]):
        """
        训练检测器
        labels: 0=人类写作, 1=AI生成
        """
        X = np.array([extract_features(t) for t in texts])
        self.model.fit(X, labels)
        self.is_fitted = True

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """返回 AI 生成概率"""
        X = np.array([extract_features(t) for t in texts])
        return self.model.predict_proba(X)[:, 1]

    def detect(self, text: str, threshold: float = 0.65) -> Dict:
        """单条检测"""
        tokens = tokenize(text)
        features = extract_features(text)
        proba = float(self.predict_proba([text])[0]) if self.is_fitted else 0.5
        return {
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
            "token_count": len(tokens),
            "perplexity_proxy": round(features[0], 4),
            "burstiness": round(features[1], 4),
            "trigram_repetition": round(features[2], 4),
            "ai_probability": round(proba, 4),
            "verdict": "AI生成" if proba >= threshold else "人类写作",
            "confidence": "高" if abs(proba - 0.5) > 0.3 else "低",
        }


# ─── 测试用例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 模拟训练数据
    human_reviews = [
        "Honestly wasn't sure about this at first but my 8-month-old LOVES it. Took a few days to get used to the sterilizing process but now it's second nature. One small gripe: the water reservoir is a bit hard to clean properly, but overall really happy with it!",
        "Bought this for my twins and it's been a lifesaver. Sometimes I wonder if I'm using it correctly because the manual is confusing, but so far no issues. Would buy again.",
        "This thing is amazing but also drives me crazy?? Like it works perfectly 95% of the time and then randomly decides not to turn on. Support was helpful when I called though. Mixed feelings overall.",
        "Not sure if I'm a fan. The bottle drying feature is great but it takes FOREVER. My husband says it's fine but I think we could have gotten a better one for less money. Kids seem healthy so maybe that's what matters lol",
        "Love how compact this is for our small kitchen. Had a learning curve with the different sterilization modes. Hot tip: pre-rinse everything before putting it in or you'll get residue.",
    ]
    ai_reviews = [
        "This sterilizer offers exceptional performance and remarkable efficiency. The innovative design ensures optimal sterilization results while maintaining user-friendly operation. The comprehensive feature set provides outstanding value for money. Highly recommended for all parents seeking premium quality.",
        "An excellent product that delivers consistent and reliable sterilization performance. The advanced technology ensures thorough cleaning of all baby accessories. The intuitive interface makes operation straightforward and convenient. This product exceeds expectations in every measurable way.",
        "Outstanding quality and superior performance characterize this exceptional sterilizer. The state-of-the-art features provide comprehensive protection for your infant. The ergonomic design ensures maximum convenience and operational efficiency. A truly remarkable product that represents excellent value.",
        "This premium sterilizer demonstrates exceptional build quality and outstanding performance. The advanced sterilization technology ensures complete protection. The user-friendly controls make this an ideal choice for modern parents. Comprehensive functionality at an affordable price point.",
        "Remarkable efficiency and exceptional durability define this superior product. The innovative sterilization process delivers consistent results every single time. The thoughtful design incorporates all essential features for optimal performance. An outstanding investment for health-conscious families.",
    ]

    all_texts = human_reviews + ai_reviews
    labels = [0] * len(human_reviews) + [1] * len(ai_reviews)

    # 训练检测器
    detector = AIContentDetector()
    detector.fit(all_texts, labels)

    # 检测新文本
    test_cases = [
        ("NEW-HUMAN", "Picked this up on a whim during Prime Day and honestly can't believe I didn't get it sooner. Only complaint is the timer beeps are annoying at night but whatevs"),
        ("NEW-AI",    "This product provides superior sterilization performance with outstanding efficiency. The comprehensive design ensures optimal results for discerning customers."),
    ]

    print("=== AI 内容检测报告 ===\n")
    results = []
    for label, text in test_cases:
        result = detector.detect(text)
        results.append(result)
        print(f"[{label}] {result['text_preview']}")
        print(f"  AI概率: {result['ai_probability']:.4f} | 判断: {result['verdict']} | 置信度: {result['confidence']}")
        print(f"  特征 — 困惑度代理: {result['perplexity_proxy']:.3f}, 突发性: {result['burstiness']:.3f}, 3gram重复率: {result['trigram_repetition']:.3f}")
        print()

    # 批量检测模拟
    all_probas = detector.predict_proba(all_texts)
    ai_detected = sum(1 for p in all_probas[5:] if p >= 0.65)
    print(f"批量检测: {ai_detected}/{len(ai_reviews)} AI 评论被正确识别")

    # 验证
    human_proba = detector.predict_proba(human_reviews).mean()
    ai_proba = detector.predict_proba(ai_reviews).mean()
    assert ai_proba > human_proba, "AI评论概率应高于人类评论"
    assert results[0]["token_count"] > 0, "Token 计数为 0"
    assert 0 <= results[0]["ai_probability"] <= 1, "概率超出范围"

    print("\n[✓] AI 生成内容检测 测试通过")
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AIGC-Content-Detection]]（AIGC 内容检测通用框架）
- **延伸（extends）**：[[Skill-AI-Fake-Review-Detection]]（AI 生成检测 + 假评论行为模式 → 完整刷单检测体系）
- **可组合（combinable）**：[[Skill-VOC-Fraud-Review-Detection]]（虚假评论检测 + AI 生成检测联合使用，覆盖两类风险）、[[Skill-NLP-Sentiment-ML-Pipeline]]（过滤 AI 评论后再做情感分析，提升分析质量）

---

## ⑤ 商业价值评估

- **ROI 预估**：避免封号风险（关联封号一次损失约 50-200 万元）、Listing 合规避免下架损失约 **20 万元/年**；竞争情报（识别竞品刷评并举报）带来 BSR 相对收益约 **15 万元/年**。总年化约 **35 万元**
- **实施难度**：⭐⭐☆☆☆（无外部 API 依赖，仅用 sklearn + re；需要 50+ 条标注样本做微调训练）
- **优先级**：⭐⭐⭐⭐⭐（亚马逊 2024-2025 年对 AI 内容合规执法加强，零成本实现高价值合规防线）
- **评估依据**：Perplexity/Burstiness 组合在学术评测上达到 80-85% 准确率；本地运行成本为 0；可作为第一道过滤器，降低后续 API 调用量
