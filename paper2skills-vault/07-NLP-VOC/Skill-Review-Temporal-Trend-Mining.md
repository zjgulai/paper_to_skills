---
title: Review 时序趋势挖掘 — LDA 滑动窗口演化分析
doc_type: knowledge
module: 07-NLP-VOC
topic: review-temporal-trend-mining
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Review 时序趋势挖掘

> **论文**：Dynamic Topic Models (Blei & Lafferty, 2006) + Temporal Topic Modeling for E-Commerce Reviews
> **arXiv**：2308.07019 | 2023 | **桥梁**: 07-NLP-VOC ↔ 03-时间序列 | **类型**: 跨域融合

---

## ① 算法原理

**核心思想**：将产品 Review 按时间窗口分组，对每个窗口独立训练 LDA 主题模型，追踪同一主题（如「漏液问题」）随时间的概率变化，识别出「近期新出现的投诉点」和「已解决的历史问题」，为产品迭代决策提供时序 VOC 证据。

**数学直觉**：
- **LDA 生成模型**：文档 $d$ 的每个词 $w$ 由主题分布 $\theta_d$ 和主题-词分布 $\phi_k$ 共同决定：$p(w | d) = \sum_k p(w | k) p(k | d) = \sum_k \phi_{kw} \theta_{dk}$
- **时间窗口 LDA**：将 $T$ 个时间段的 Review 分组 $\{D_1, D_2, ..., D_T\}$，对每组独立拟合 LDA，得到时变主题分布序列 $\{\Phi_1, ..., \Phi_T\}$
- **主题对齐**：用 Jensen-Shannon 散度衡量跨窗口主题相似性 $\text{JSD}(\phi_k^t, \phi_k^{t+1})$，找到时序一致的主题链
- **趋势检测**：某主题在窗口 $t$ 的均值概率 $\bar{\theta}_k^t = \frac{1}{|D_t|}\sum_{d \in D_t} \theta_{dk}$，连续上升则为新兴投诉点

**关键假设**：
- 每个时间窗口内 Review 数量 ≥ 50（否则 LDA 不稳定）
- 主题数 K 预先设定（通常 5-15 个），可用 Coherence Score 自动选择
- 语言一致性（混语言 Review 需先分语言处理）

---

## ② 母婴出海应用案例

**场景A：产品上架后质量问题演化追踪**

- **业务问题**：吸奶器 V2 版上市 6 个月后，近 3 个月差评率突然从 5% 上升到 12%，运营不知道是哪个具体问题恶化了，无法定位到产品缺陷
- **数据要求**：该 ASIN 近 12 个月 Review（约 2000 条），含评论文本和提交日期；按月切分为 12 个窗口
- **预期产出**：发现「马达噪音投诉」主题在最近 3 个月激增（从月均 5% → 22%），指向「第 6 周生产批次的马达供应商变更」，提供精确溯源
- **业务价值**：从发现问题到定位原因时间从 3 周 → 2 天，避免问题持续恶化损失约 **20 万元**（差评率继续上升的 GMV 损失）；精准召回问题批次，节省售后成本约 **5 万元**

**场景B：新品上市预警系统**

- **业务问题**：竞品 ASIN 上市 4 个月，运营想监控竞品的 Review 动态，判断竞品是否存在质量下坡路（从而抓住机会窗口）
- **数据要求**：竞品 ASIN 近 6 个月 Review（按月分组，每月至少 50 条）
- **预期产出**：竞品 Review 趋势报告：「防漏性能」主题 2 个月前开始下滑，「包装损坏」主题近 1 个月新出现，竞品开始走下坡
- **业务价值**：提前 2-3 个月发现竞品弱点，及时加大广告投入抢占市场份额，预计增量 GMV 约 **30 万元**

---

## ③ 代码模板

```python
"""
Review 时序趋势挖掘
LDA Topic Model + 时间窗口滑动 → 演化趋势分析
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import re


# ─── 轻量 LDA 实现（无需 gensim，纯 numpy）──────────────────────────────────────
class LightLDA:
    """
    轻量版 Latent Dirichlet Allocation
    用 Gibbs Sampling 实现
    """

    def __init__(self, n_topics: int = 8, alpha: float = 0.1, beta: float = 0.01,
                 n_iter: int = 50, random_state: int = 42):
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.rng = np.random.default_rng(random_state)

    def fit(self, docs: List[List[int]], vocab_size: int):
        """
        训练 LDA
        docs: 词 id 列表的列表
        vocab_size: 词汇表大小
        """
        self.V = vocab_size
        D = len(docs)

        # 初始化计数矩阵
        n_dt = np.zeros((D, self.K), dtype=np.int32)    # 文档-主题计数
        n_kt = np.zeros((self.K, self.V), dtype=np.int32)  # 主题-词计数
        n_k = np.zeros(self.K, dtype=np.int32)             # 主题总词数

        # 随机初始化主题分配
        z_assignments = []
        for d, doc in enumerate(docs):
            z = self.rng.integers(0, self.K, size=len(doc))
            z_assignments.append(z.tolist())
            for w, k in zip(doc, z):
                n_dt[d, k] += 1
                n_kt[k, w] += 1
                n_k[k] += 1

        # Gibbs 采样
        for _ in range(self.n_iter):
            for d, doc in enumerate(docs):
                for i, w in enumerate(doc):
                    k_old = z_assignments[d][i]
                    # 移除当前词的贡献
                    n_dt[d, k_old] -= 1
                    n_kt[k_old, w] -= 1
                    n_k[k_old] -= 1

                    # 计算新主题分布
                    p_k = (n_dt[d] + self.alpha) * (n_kt[:, w] + self.beta) / (n_k + self.V * self.beta)
                    p_k = np.maximum(p_k, 0)
                    p_sum = p_k.sum()
                    if p_sum > 0:
                        p_k /= p_sum
                    else:
                        p_k = np.ones(self.K) / self.K

                    k_new = int(self.rng.choice(self.K, p=p_k))
                    z_assignments[d][i] = k_new
                    n_dt[d, k_new] += 1
                    n_kt[k_new, w] += 1
                    n_k[k_new] += 1

        # 归一化
        self.phi = (n_kt + self.beta) / (n_k[:, np.newaxis] + self.V * self.beta)
        self.theta = (n_dt + self.alpha) / (n_dt.sum(axis=1, keepdims=True) + self.K * self.alpha)
        return self


def preprocess(text: str, stopwords: set = None) -> List[str]:
    """简单预处理：小写 + 分词 + 停用词过滤"""
    stopwords = stopwords or {"the", "a", "an", "is", "it", "this", "that", "and",
                              "to", "for", "of", "in", "my", "i", "was", "be", "with"}
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [t for t in tokens if t not in stopwords]


def build_vocab(all_docs: List[List[str]]) -> Tuple[Dict, List[str]]:
    """构建词汇表"""
    all_words = [w for doc in all_docs for w in doc]
    freq = Counter(all_words)
    vocab = [w for w, c in freq.most_common(500) if c >= 2]
    word2id = {w: i for i, w in enumerate(vocab)}
    return word2id, vocab


def docs_to_ids(docs: List[List[str]], word2id: Dict) -> List[List[int]]:
    """转换为 id 表示"""
    return [[word2id[w] for w in doc if w in word2id] for doc in docs]


def get_top_words(phi: np.ndarray, vocab: List[str], n: int = 8) -> List[List[str]]:
    """每个主题的 Top-N 词"""
    return [
        [vocab[j] for j in phi[k].argsort()[::-1][:n]]
        for k in range(phi.shape[0])
    ]


def analyze_temporal_trends(
    reviews: List[Dict],
    window: str = "month",
    n_topics: int = 6,
    top_n_words: int = 6
) -> Dict:
    """
    时序趋势分析
    reviews: [{"text": ..., "date": "YYYY-MM", "rating": ...}]
    返回：各时间窗口主题分布 + 趋势变化
    """
    # 按时间窗口分组
    windows = defaultdict(list)
    for r in reviews:
        key = r["date"][:7] if window == "month" else r["date"][:4]
        windows[key].append(r["text"])

    sorted_windows = sorted(windows.keys())

    # 建立全局词汇表
    all_texts_tokenized = [preprocess(t) for w in sorted_windows for t in windows[w]]
    word2id, vocab = build_vocab(all_texts_tokenized)

    window_results = {}
    for w_key in sorted_windows:
        docs_tokenized = [preprocess(t) for t in windows[w_key]]
        doc_ids = [d for d in docs_to_ids(docs_tokenized, word2id) if len(d) >= 3]
        if len(doc_ids) < 5:
            continue

        lda = LightLDA(n_topics=n_topics, n_iter=30, random_state=42)
        lda.fit(doc_ids, len(vocab))

        top_words = get_top_words(lda.phi, vocab, top_n_words)
        topic_weights = lda.theta.mean(axis=0).tolist()  # 该窗口内各主题平均权重

        window_results[w_key] = {
            "n_reviews": len(doc_ids),
            "topic_weights": topic_weights,
            "top_words": top_words,
        }

    # 趋势检测：找最近一个窗口与平均相比增幅最大的主题
    if len(sorted_windows) >= 3:
        available_windows = [w for w in sorted_windows if w in window_results]
        if len(available_windows) >= 2:
            latest = available_windows[-1]
            earlier = available_windows[:-1]
            latest_weights = np.array(window_results[latest]["topic_weights"])
            avg_earlier = np.mean([window_results[w]["topic_weights"] for w in earlier], axis=0)
            trend_delta = latest_weights - avg_earlier
            rising_topic = int(np.argmax(trend_delta))
            return {
                "windows": window_results,
                "trend_alert": {
                    "rising_topic_id": rising_topic,
                    "rising_topic_words": window_results[latest]["top_words"][rising_topic],
                    "delta": round(float(trend_delta[rising_topic]), 4),
                    "latest_window": latest,
                }
            }

    return {"windows": window_results, "trend_alert": None}


# ─── 测试用例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 模拟吸奶器 Review（含时序信号：近 3 个月噪音投诉增加）
    reviews = []
    # 早期 Review：主要是漏液投诉
    for i in range(40):
        date = "2026-01" if i < 20 else "2026-02"
        reviews.append({
            "text": f"The bottle leaks sometimes, silicone seal might be defective. Motor works fine though.",
            "date": date, "rating": 3
        })
    # 中期 Review：多样问题
    for i in range(40):
        date = "2026-03" if i < 20 else "2026-04"
        reviews.append({
            "text": f"Generally good product. Cleaning is easy. Sterilization works well. Happy customer.",
            "date": date, "rating": 4
        })
    # 近期 Review：噪音问题新出现
    for i in range(40):
        date = "2026-05" if i < 20 else "2026-06"
        reviews.append({
            "text": f"Motor is very noisy loud recently. Vibration noise annoying. Quality decreased.",
            "date": date, "rating": 2
        })

    print("=== Review 时序趋势分析 ===\n")
    result = analyze_temporal_trends(reviews, window="month", n_topics=5)

    for w_key, data in result["windows"].items():
        top_topics = sorted(
            enumerate(zip(data["topic_weights"], data["top_words"])),
            key=lambda x: x[1][0], reverse=True
        )[:2]
        print(f"[{w_key}] {data['n_reviews']} 条 Review")
        for tid, (weight, words) in top_topics:
            print(f"  Topic-{tid} ({weight:.3f}): {' | '.join(words[:4])}")

    if result["trend_alert"]:
        alert = result["trend_alert"]
        print(f"\n⚠️  趋势预警：窗口 [{alert['latest_window']}] 主题异常上升")
        print(f"  上升幅度: +{alert['delta']:.4f}")
        print(f"  关键词: {' | '.join(alert['rising_topic_words'])}")

    # 验证
    assert len(result["windows"]) >= 3, "时间窗口数量不足"
    assert result["trend_alert"] is not None, "未检测到趋势预警"
    assert result["trend_alert"]["delta"] > 0, "趋势变化方向错误"

    print("\n[✓] Review 时序趋势挖掘 测试通过")
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-VOC-Trend-Signal-Forecasting]]（VOC 趋势预测基础）
- **延伸（extends）**：[[Skill-NLP-Sentiment-ML-Pipeline]]（趋势识别后对高权重主题做细粒度情感分析）
- **可组合（combinable）**：[[Skill-Multilingual-NLP-Pipeline]]（多语言覆盖 + 时序趋势 → 全球 VOC 监控体系）、[[Skill-VOC-Aspect-Sentiment-Extraction]]（主题趋势挖掘 + 方面情感提取 → 精准问题定位）

---

## ⑤ 商业价值评估

- **ROI 预估**：产品质量问题早发现（快 2 周）避免持续差评损失约 **20 万元**；竞品弱点早发现带来增量 GMV 约 **30 万元**。总年化约 **50 万元**
- **实施难度**：⭐⭐⭐☆☆（LDA 需要一定调参经验；每月 50+ 条 Review 才能稳定运行；代码已封装，直接调用 `analyze_temporal_trends` 即可）
- **优先级**：⭐⭐⭐⭐☆（有 Review 历史数据的 ASIN 立即可用，无冷启动问题）
- **评估依据**：LDA 是 VOC 分析标准方法，行业验证充分；时序窗口方法已在电商质量监控中广泛应用；问题提前发现的 ROI 显著
