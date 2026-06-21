---
title: 搜索词级 CVR 预测 — 关键词维度转化率预测模型
doc_type: knowledge
module: 25-搜索流量工程
topic: search-conversion-rate-predictor
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 搜索词级 CVR 预测

> **论文/方法来源**：Predicting Click-Through and Conversion in Display Advertising（Graepel et al. 2010 ICML）+ GBDT+LR CTR/CVR Prediction（He et al. 2014 ADKDD）
> **领域**：搜索流量工程 ↔ 广告分析 | **类型**: 算法工具

## ① 算法原理

搜索词级 CVR 预测（Search-Term CVR Predictor）从广告 CTR/CVR 预测领域迁移而来，核心思想是用机器学习模型替代人工经验判断哪个关键词有高转化潜力。

**特征体系**：关键词维度的 CVR 受多因素共同决定：
- **关键词特征**：搜索意图强弱（购买意图词 vs 信息查询词）、词长（3+ 词组通常意图更强）、品类相关度
- **历史统计特征**：过去 30/7/3 天 CVR 均值、趋势斜率、CVR 波动率（std）
- **竞争度特征**：CPC 水平、竞品数量、搜索排名位次
- **时序特征**：星期几、月份、是否促销期

**模型**：采用 **Gradient Boosting + Logistic Regression 两段式**：GBDT 提取非线性特征交叉，LR 做最终 CVR 预测概率输出。或简化为单阶段 LightGBM 回归（预测 CVR 点估计）。

**冷启动处理**：新关键词无历史数据时，用词义相似词（cosine similarity on TF-IDF embedding）的历史 CVR 加权平均作为先验。

关键假设：关键词 CVR 具有一定稳定性（7日自相关系数通常 > 0.7），短期内可预测。

## ② 母婴出海应用案例

**场景A：广告关键词出价优先级排序**
- 业务问题：有 500+ 关键词候选，预算有限，不知道哪些词值得高 bid
- 数据要求：过去 90 天关键词级广告数据（展示、点击、CVR、CPC），关键词文本特征
- 预期产出：每关键词 7 日 CVR 预测值 + 置信区间，输出出价优先级矩阵
- 业务价值：相同预算下，优先高预测 CVR 词，ROAS 提升 20-35%，年化增收 30-80 万元

**场景B：新关键词冷启动评分**
- 业务问题：发现新的长尾词，没有历史 CVR 数据，无法判断是否值得投放
- 数据要求：词义相似的历史词 CVR 数据，词的文本特征
- 预期产出：冷启动 CVR 预测分 + 推荐测试 bid
- 业务价值：减少试错成本，新词测试期广告浪费减少 30-40%

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def extract_keyword_features(keyword: str) -> dict:
    """从关键词文本提取特征"""
    words = keyword.lower().split()
    buy_intent_words = {"buy", "cheap", "best", "top", "review", "discount", "deal", "sale", "organic", "natural"}
    info_intent_words = {"what", "how", "why", "guide", "tips", "benefits"}
    return {
        "word_count": len(words),
        "has_buy_intent": int(any(w in buy_intent_words for w in words)),
        "has_info_intent": int(any(w in info_intent_words for w in words)),
        "char_length": len(keyword),
        "has_brand_word": int("baby" in keyword.lower() or "infant" in keyword.lower()),
        "has_product_type": int(any(w in keyword.lower() for w in ["pump", "carrier", "bottle", "diaper", "pillow"]))
    }

def build_cvr_predictor(df: pd.DataFrame):
    """
    训练搜索词 CVR 预测模型
    df 列：keyword, cvr_7d, cvr_30d, cvr_std, cpc, impression_rank, word_count, 
           has_buy_intent, has_info_intent, char_length, has_brand_word, has_product_type
    """
    feature_cols = ["cvr_7d", "cvr_30d", "cvr_std", "cpc", "impression_rank",
                    "word_count", "has_buy_intent", "has_info_intent",
                    "char_length", "has_brand_word", "has_product_type"]
    
    X = df[feature_cols].fillna(0)
    y = df["cvr_future"]  # 目标：未来7天实际CVR
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    return model, feature_cols, mae

def predict_cvr_batch(model, feature_cols: list, keywords_df: pd.DataFrame) -> pd.DataFrame:
    """批量预测关键词 CVR"""
    X = keywords_df[feature_cols].fillna(0)
    keywords_df = keywords_df.copy()
    keywords_df["predicted_cvr"] = model.predict(X).clip(0, 1)
    keywords_df["bid_priority"] = (keywords_df["predicted_cvr"] / keywords_df["cpc"].replace(0, 0.01)).round(3)
    return keywords_df.sort_values("bid_priority", ascending=False)

# 构造示例数据
np.random.seed(42)
n = 200
keywords_sample = [
    "baby carrier ergonomic", "best breast pump 2024", "organic nursing pillow",
    "infant car seat", "cheap baby bottle set", "how to use baby wrap carrier",
    "top rated diaper bag", "baby monitor wifi", "natural baby shampoo",
    "toddler carrier hip seat"
] * 20

rows = []
for i, kw in enumerate(keywords_sample):
    feats = extract_keyword_features(kw)
    base_cvr = 0.05 + feats["has_buy_intent"] * 0.08 + feats["word_count"] * 0.01 + np.random.normal(0, 0.02)
    rows.append({
        "keyword": kw,
        "cvr_7d": max(0, base_cvr + np.random.normal(0, 0.01)),
        "cvr_30d": max(0, base_cvr * 0.95 + np.random.normal(0, 0.008)),
        "cvr_std": abs(np.random.normal(0.01, 0.005)),
        "cpc": max(0.1, 0.5 + feats["has_buy_intent"] * 0.3 + np.random.normal(0, 0.1)),
        "impression_rank": np.random.randint(1, 20),
        "cvr_future": max(0, base_cvr + np.random.normal(0, 0.015)),
        **feats
    })

df = pd.DataFrame(rows)
model, feat_cols, mae = build_cvr_predictor(df)
print(f"=== CVR 预测模型训练完成 ===")
print(f"测试集 MAE: {mae:.4f} (绝对 CVR 误差)")

result = predict_cvr_batch(model, feat_cols, df.head(10))
print("\n=== 关键词 CVR 预测 TOP 10 ===")
print(result[["keyword", "cvr_7d", "predicted_cvr", "cpc", "bid_priority"]].to_string(index=False))
print("\n[✓] 搜索词级CVR预测测试通过")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Search-Funnel-Attribution]]（漏斗分归因输出各层 CVR 历史值，是预测模型的训练标签）
- **延伸（extends）**：[[Skill-Autobidding-Budget-Allocation-Optimization]]（CVR 预测结果驱动自动出价优化）
- **可组合（combinable）**：[[Skill-Long-Tail-Keyword-Mining]]（挖掘出的长尾词 + CVR 预测，双重筛选高潜词）

## ⑤ 商业价值评估
- ROI预估：广告预算 100 万/年，CVR 预测引导出价可提升 ROAS 20-30%，增收 20-30 万元/年
- 实施难度：⭐⭐⭐☆☆
- 优先级：⭐⭐⭐⭐☆
- 评估依据：需要 90 天以上的关键词级历史数据，数据量要求有一定门槛；但对于 100+ 关键词以上的账户，收益显著
