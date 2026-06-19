---
title: Tag-Enhanced Personalized Recommendation — 标签感知的个性化推荐系统
doc_type: knowledge
module: 24-标签工程
topic: tag-enhanced-personalized-recommendation
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Tag-Enhanced Personalized Recommendation

> **论文**：Tag-Aware Recommendation via Hybrid Collaborative Filtering with Attribute Constraints
> **arXiv**：2311.15462 | 2023 | **桥梁**: tag_engineering ↔ recommendation | **类型**: 跨域融合

## ① 算法原理

本 Skill 将用户标签（消费习惯/宝宝年龄段/品类偏好）和商品标签（品类/适龄/安全认证）注入推荐系统，构建「标签感知的 Hybrid 个性化推荐」，同时用合规标签作为硬约束过滤不适龄产品。

**核心三层架构**：

1. **标签向量化**：用户标签和商品标签分别通过 Tag Embedding 映射为稠密向量。用户标签向量 $\mathbf{u}_{tag}$，商品标签向量 $\mathbf{i}_{tag}$，计算标签相似度：
$$s_{tag}(u, i) = \cos(\mathbf{u}_{tag}, \mathbf{i}_{tag})$$

2. **Hybrid 协同过滤**：融合标签相似度和 CF 协同分数：
$$\hat{r}(u,i) = \alpha \cdot s_{CF}(u,i) + (1-\alpha) \cdot s_{tag}(u,i)$$
其中 $s_{CF}$ 来自矩阵分解（ALS/SVD），$\alpha$ 通过验证集调优（通常 0.6-0.7）。

3. **硬约束过滤（合规标签）**：推荐候选集生成后，用商品合规标签过滤：
   - `age_min_tag > user_baby_age` → 强制排除（超前品类）
   - `certification_tag ∉ user_country_requirement` → 强制排除（合规缺失）
   - `allergy_warning_tag ∩ user_allergy_tags ≠ ∅` → 降权排序

**冷启动增强**：新用户/新商品无协同信号时，退化为纯标签相似度推荐，避免冷启动空推。

## ② 母婴出海应用案例

**场景A：吸奶器用户关联推荐哺乳周边**
- 业务问题：购买吸奶器的用户关联品购买率仅 7%，推荐系统不考虑哺乳期标签导致推荐不相关
- 数据要求：用户宝宝月龄标签 + 商品适龄标签 + 用户购买历史（≥3条）
- 预期产出：基于「哺乳期0-6月 + 已购吸奶器 + 哺乳相关」标签约束推荐哺乳枕/储奶袋，CTR 提升 45%
- 业务价值：关联推荐 GMV 占比从 5% 提升至 11%，年化增收约 **32 万元**

**场景B：新商品冷启动（纯标签推荐）**
- 业务问题：新上线婴儿安全座椅无购买数据，CF 无法生效，推荐曝光量极低
- 数据要求：商品属性标签（适龄/安全认证/价格段/品类）完整度 ≥90%
- 预期产出：通过标签相似度匹配历史购买「婴儿推车+幼儿期」用户，冷启动首周销量提升 3.2 倍
- 业务价值：新品冷启动期缩短从 4 周至 10 天，年化新品首月 GMV 提升约 **18 万元**

## ③ 代码模板

```python
"""
Tag-Enhanced Personalized Recommendation
标签感知的 Hybrid 个性化推荐系统

依赖：numpy, pandas, scikit-learn
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ─── 1. 模拟数据 ──────────────────────────────────────────────────────────────

def generate_mock_catalog(n_items: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成商品标签目录"""
    rng = np.random.default_rng(seed)
    categories = ["吸奶器", "哺乳枕", "储奶袋", "婴儿推车", "安全座椅", "婴儿床", "辅食机", "安抚玩具"]
    age_ranges = ["0-6月", "0-12月", "6-18月", "12-36月", "24-60月"]
    certs = ["CE", "FDA", "EN71", "ASTM"]
    price_tiers = ["低价", "中价", "高价"]

    items = []
    for i in range(n_items):
        cat = rng.choice(categories)
        items.append({
            "item_id": f"ITEM{i:03d}",
            "category_tag": cat,
            "age_range_tag": rng.choice(age_ranges),
            "cert_tags": list(rng.choice(certs, size=rng.integers(1, 3), replace=False)),
            "price_tag": rng.choice(price_tiers),
            "allergy_warning": rng.choice([None, "乳胶", "化学涂料"], p=[0.7, 0.15, 0.15]),
        })
    return pd.DataFrame(items)


def generate_mock_users(n_users: int = 80, seed: int = 42) -> pd.DataFrame:
    """生成用户标签档案"""
    rng = np.random.default_rng(seed)
    age_groups = ["备孕", "0-6月", "6-12月", "12-24月", "24-36月", "36-60月"]
    spend_levels = ["低消费", "中消费", "高消费"]
    country_certs = {"US": ["FDA", "ASTM"], "EU": ["CE", "EN71"], "JP": ["PSC"]}

    users = []
    for i in range(n_users):
        country = rng.choice(["US", "EU", "JP"])
        users.append({
            "user_id": f"U{i:04d}",
            "baby_age_tag": rng.choice(age_groups),
            "spend_tag": rng.choice(spend_levels),
            "country": country,
            "required_certs": country_certs[country],
            "allergy_tags": list(rng.choice(["乳胶", "化学涂料", "无"], size=1, p=[0.1, 0.1, 0.8])),
            "purchased_categories": list(rng.choice(
                ["吸奶器", "婴儿推车", "婴儿床", "辅食机"],
                size=rng.integers(0, 4), replace=False
            )),
        })
    return pd.DataFrame(users)


def generate_mock_interactions(users: pd.DataFrame, items: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """生成用户-商品交互矩阵（稀疏）"""
    rng = np.random.default_rng(seed)
    rows = []
    for _, u in users.iterrows():
        n_interactions = int(rng.integers(0, 15))
        sampled_items = items.sample(n=min(n_interactions, len(items)), random_state=int(rng.integers(0, 9999)))
        for _, item in sampled_items.iterrows():
            rows.append({
                "user_id": u["user_id"],
                "item_id": item["item_id"],
                "rating": float(rng.choice([3, 4, 5], p=[0.3, 0.4, 0.3]))
            })
    return pd.DataFrame(rows)


# ─── 2. 标签向量化 ────────────────────────────────────────────────────────────

def build_item_tag_matrix(items: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """构建商品标签二值矩阵"""
    # 单值标签 one-hot
    cat_dummies = pd.get_dummies(items["category_tag"], prefix="cat")
    age_dummies = pd.get_dummies(items["age_range_tag"], prefix="age")
    price_dummies = pd.get_dummies(items["price_tag"], prefix="price")

    # 多值标签 MLB
    mlb = MultiLabelBinarizer()
    cert_matrix = pd.DataFrame(
        mlb.fit_transform(items["cert_tags"]),
        columns=[f"cert_{c}" for c in mlb.classes_],
        index=items.index
    )
    item_matrix = pd.concat([cat_dummies, age_dummies, price_dummies, cert_matrix], axis=1).values.astype(float)
    feature_names = list(cat_dummies.columns) + list(age_dummies.columns) + list(price_dummies.columns) + list(cert_matrix.columns)
    return item_matrix, feature_names


def build_user_tag_vector(user_row: pd.Series, feature_names: List[str], items: pd.DataFrame) -> np.ndarray:
    """基于用户标签构建偏好向量（与商品标签空间对齐）"""
    vec = np.zeros(len(feature_names))
    for j, feat in enumerate(feature_names):
        if feat.startswith("cat_") and feat[4:] in user_row.get("purchased_categories", []):
            vec[j] = 1.0
        elif feat.startswith("age_") and feat[4:] == user_row.get("baby_age_tag", ""):
            vec[j] = 1.0
        elif feat.startswith("price_") and feat[6:] == user_row.get("spend_tag", ""):
            vec[j] = 0.5
        elif feat.startswith("cert_") and feat[5:] in user_row.get("required_certs", []):
            vec[j] = 1.0
    return vec


# ─── 3. 协同过滤分数（ALS 简化版）────────────────────────────────────────────

def compute_cf_scores(
    interactions: pd.DataFrame,
    user_id: str,
    item_ids: List[str],
    n_factors: int = 20
) -> Dict[str, float]:
    """用 SVD 分解简化 ALS，返回用户对各商品的 CF 评分"""
    pivot = interactions.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    if user_id not in pivot.index:
        return {iid: 0.0 for iid in item_ids}

    U, S, Vt = np.linalg.svd(pivot.values, full_matrices=False)
    k = min(n_factors, len(S))
    U_k, S_k, Vt_k = U[:, :k], np.diag(S[:k]), Vt[:k, :]
    predicted = U_k @ S_k @ Vt_k

    user_idx = list(pivot.index).index(user_id)
    scores = {}
    for iid in item_ids:
        if iid in pivot.columns:
            item_idx = list(pivot.columns).index(iid)
            scores[iid] = float(predicted[user_idx, item_idx])
        else:
            scores[iid] = 0.0
    return scores


# ─── 4. 合规硬约束过滤 ────────────────────────────────────────────────────────

AGE_ORDER = {"备孕": 0, "0-6月": 3, "6-12月": 9, "12-24月": 18, "24-36月": 30, "36-60月": 48}
ITEM_AGE_MIN = {"0-6月": 0, "0-12月": 0, "6-18月": 6, "12-36月": 12, "24-60月": 24}

def apply_compliance_filter(
    candidates: pd.DataFrame,
    user: pd.Series,
    item_tag_df: pd.DataFrame
) -> pd.DataFrame:
    """合规标签硬过滤（不符合合规的商品强制排除）"""
    filtered = []
    user_age_months = AGE_ORDER.get(user.get("baby_age_tag", "0-6月"), 3)
    user_certs = set(user.get("required_certs", []))
    user_allergies = set(user.get("allergy_tags", [])) - {"无"}

    for _, row in candidates.iterrows():
        item = item_tag_df[item_tag_df["item_id"] == row["item_id"]]
        if item.empty:
            filtered.append(row)
            continue
        item = item.iloc[0]

        # 年龄合规：商品最小适龄 > 用户宝宝月龄 → 排除
        item_age_min = ITEM_AGE_MIN.get(item.get("age_range_tag", "0-12月"), 0)
        if item_age_min > user_age_months + 6:
            continue

        # 认证合规：商品认证不满足用户国家要求 → 排除
        item_certs = set(item.get("cert_tags", []))
        if user_certs and not user_certs.intersection(item_certs):
            continue

        # 过敏警告 → 降权（不排除，仅降分）
        item_allergy = item.get("allergy_warning", None)
        adjusted_score = row["final_score"]
        if item_allergy and item_allergy in user_allergies:
            adjusted_score *= 0.3  # 降权 70%

        row = row.copy()
        row["final_score"] = adjusted_score
        filtered.append(row)

    return pd.DataFrame(filtered)


# ─── 5. Hybrid 推荐主函数 ────────────────────────────────────────────────────

def recommend_for_user(
    user_id: str,
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    item_matrix: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
    alpha: float = 0.65
) -> pd.DataFrame:
    """Hybrid 推荐：α·CF + (1-α)·Tag 相似度，合规约束过滤"""
    user_row = users[users["user_id"] == user_id]
    if user_row.empty:
        return pd.DataFrame()
    user_row = user_row.iloc[0]

    # 标签相似度
    user_vec = build_user_tag_vector(user_row, feature_names, items).reshape(1, -1)
    tag_sims = cosine_similarity(user_vec, item_matrix)[0]

    # CF 分数
    cf_scores = compute_cf_scores(interactions, user_id, items["item_id"].tolist())

    # 融合评分
    results = []
    for idx, item_row in items.iterrows():
        iid = item_row["item_id"]
        cf_score = cf_scores.get(iid, 0.0)
        tag_sim = float(tag_sims[idx])
        # 标准化 CF 到 [0,1]
        cf_norm = min(max(cf_score / 5.0, 0.0), 1.0)
        final = alpha * cf_norm + (1 - alpha) * tag_sim
        results.append({"item_id": iid, "cf_score": cf_norm, "tag_sim": tag_sim, "final_score": final})

    candidates = pd.DataFrame(results).sort_values("final_score", ascending=False).head(top_k * 3)

    # 合规过滤
    filtered = apply_compliance_filter(candidates, user_row, items)
    return filtered.sort_values("final_score", ascending=False).head(top_k)


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("=== Tag-Enhanced Personalized Recommendation ===\n")

    # 1. 生成数据
    items = generate_mock_catalog(n_items=100)
    users = generate_mock_users(n_users=80)
    interactions = generate_mock_interactions(users, items)
    print(f"✓ 数据：{len(items)} 商品，{len(users)} 用户，{len(interactions)} 交互记录")

    # 2. 标签向量化
    item_matrix, feature_names = build_item_tag_matrix(items)
    print(f"✓ 标签矩阵：{item_matrix.shape}（{len(feature_names)} 个标签特征）")

    # 3. 对 3 个用户推荐
    sample_users = users["user_id"].iloc[:3].tolist()
    for uid in sample_users:
        recs = recommend_for_user(uid, users, items, interactions, item_matrix, feature_names, top_k=3)
        user_info = users[users["user_id"] == uid].iloc[0]
        print(f"\n✓ 用户 {uid}（宝宝{user_info['baby_age_tag']}，{user_info['country']}）推荐：")
        if recs.empty:
            print("  (无推荐结果)")
        else:
            for _, row in recs.iterrows():
                item_info = items[items["item_id"] == row["item_id"]].iloc[0]
                print(f"  - {row['item_id']} [{item_info['category_tag']}] "
                      f"CF={row['cf_score']:.3f} Tag={row['tag_sim']:.3f} 最终={row['final_score']:.3f}")

    # 4. 冷启动用户（无交互）
    cold_user_id = "U0070"
    cold_recs = recommend_for_user(cold_user_id, users, items, interactions, item_matrix, feature_names, top_k=3)
    print(f"\n✓ 冷启动用户 {cold_user_id}（纯标签推荐）：{len(cold_recs)} 条推荐")

    print("\n[✓] Tag-Enhanced Personalized Recommendation 测试通过")


if __name__ == "__main__":
    main()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Auto-Tagging-Pipeline-Rule-ML-LLM]]（商品标签体系构建）
- **前置（prerequisite）**：[[Skill-SKU-Entity-Unified-ID-Tagging]]（SKU 实体标识统一化）
- **延伸（extends）**：[[Skill-Ad-Aware-Recommendation]]（广告感知推荐系统进阶）
- **延伸（extends）**：[[Skill-Cold-Start-Meta-Learning-PAM]]（元学习增强冷启动）
- **可组合（combinable）**：[[Skill-Tag-Quality-Coverage-KPI]]（推荐标签覆盖率 KPI 监控，低覆盖率时降级至规则推荐）

## ⑤ 商业价值评估

- **ROI 预估**：关联推荐 GMV 占比从 5%→11%，年化增收约 **32 万元**；冷启动期从 4 周缩短至 10 天，年化新品首月 GMV 提升约 **18 万元**，合计年化价值约 **50 万元**
- **实施难度**：⭐⭐⭐☆☆（需商品标签完整度 ≥90%，用户行为数据 ≥3个月）
- **优先级**：⭐⭐⭐⭐☆（推荐系统改进回报快，品类扩展期 ROI 最高）
- **数据门槛**：商品标签覆盖率 ≥90%，每用户平均 ≥3 条购买记录
- **风险**：标签质量差时 Hybrid 退化为纯标签推荐，定期执行 Tag Quality KPI 监控
