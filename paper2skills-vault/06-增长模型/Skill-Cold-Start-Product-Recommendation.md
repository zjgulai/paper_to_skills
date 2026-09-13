---
title: Skill-Cold-Start-Product-Recommendation
module: 06-增长模型
topic: 用 LLM 模拟用户-商品交互，把冷商品直接转成「伪热商品」再走标准嵌入优化
status: draft
created: 2026-05-15
updated: 2026-09-13
owner: self
source: ai
venue_tier: CCF-B
venue_source: arxiv-abs(comments='10 pages, accepted by WSDM 2025')
paper_id: 2402.09176
paper: Large Language Model Simulator for Cold-Start Recommendation
evidence_basis: paper-verbatim
verified_by: quote_check.py (引文逐字核验 VERBATIM) + gate_check.py G2
verified_at: 2026-09-12
related: Skill-New-Product-Opportunity-Mining.md, Skill-Uplift-Churn-Prediction.md
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-04
l2_domain: 渠道经营
l3_id: DOM-04-070
l3_business: 转化优化
l3_all: 转化优化
l1_l2_l3: 业务运营/渠道经营/转化优化
---

# Skill Card: Cold-Start Product Recommendation (冷启动商品推荐)

---

## ① 算法原理

### 核心思想

冷启动商品推荐解决的核心问题是：**新商品没有历史交互数据时，如何精准推荐给用户**。传统方法通过内容特征生成合成嵌入向量，但这会造成冷商品和热商品之间的表征差距。本框架采用**LLM用户行为模拟**的全新思路：让大语言模型基于世界知识和推理能力，模拟用户可能如何与新商品交互。

该框架源自 [ColdLLM: Large Language Model Simulator for Cold-Start Recommendation](https://arxiv.org/abs/2402.09176)，WSDM 2025 接受论文。

### 数学直觉

**传统方法 vs LLM模拟方法**

传统方法（合成嵌入）：
$$
\mathbf{e}_i^{cold} = f_{DNN}(\mathbf{c}_i)
$$

LLM模拟方法：
$$
\hat{s}_i = \{(u_1, i, r_1), (u_2, i, r_2), ..., (u_n, i, r_n)\}
$$
$$
\mathbf{e}_i^{cold} = Emb_{opt}(\mathbf{c}_i, \hat{s}_i, \mathbf{E})
$$

其中 $\hat{s}_i$ 是LLM模拟的用户-商品交互集合，$Emb_{opt}$ 是标准的行为嵌入优化。

**Coupled Funnel 双阶段架构**

阶段1 - Filtering（过滤）：
$$
\mathbf{f}_i = \mathcal{F}_{\mathcal{I}}(LLM_{emb}(\mathbf{c}_i))
$$
$$
TopK = \{u | \mathbf{f}_i^T \cdot \mathbf{e}_u > \theta\}
$$

- 从十亿级用户筛选到百级候选
- 使用FAISS索引，复杂度 O(1)，耗时 ~60ms
- 基于LLM提取的内容嵌入进行相似度计算

阶段2 - Refining（精筛）：
$$
P(interact|u, i) = LLM_{fine-tuned}(Prompt(u, i, \mathcal{H}_u))
$$

- 使用微调LLaMA-7B模型
- 构建用户上下文：历史交互商品（Top-L相似度筛选）
- 输出交互概率（Yes/No）
- 处理时间 200-400ms/用户

**Item Embedding 提取**

$$
\mathbf{e}_i = MeanPool(LLM_{last\_layer}(\mathbf{c}_i))
$$

- 使用LLM最后一层token嵌入
- 均值池化获得商品内容表征
- 维度：论文将嵌入维度统一设为 **200**（§5.1.3；v1 与 v2 措辞一致）

**Embedding 优化**

模拟交互后，冷商品转化为"伪热商品"，可使用标准优化：

$$
\mathcal{L} = \sum_{(u,i) \in \hat{s}_i} -\log \sigma(\mathbf{e}_u^T \mathbf{e}_i) - \sum_{(u,j) \in \mathcal{N}} \log \sigma(-\mathbf{e}_u^T \mathbf{e}_j)
$$

### Coupled Funnel 框架流程

```
┌─────────────────────────────────────────────────────────────┐
│                     ColdLLM 框架                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  新商品内容特征 (描述、图片、类目、价格)                        │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────┐                   │
│  │  Stage 1: Filtering Simulation      │                   │
│  │  • LLM提取内容嵌入                   │                   │
│  │  • FAISS相似度检索 (十亿→百级用户)    │                   │
│  │  • 耗时: ~60ms                      │                   │
│  └─────────────────────────────────────┘                   │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────┐                   │
│  │  Stage 2: Refining Simulation       │                   │
│  │  • 构建用户上下文 (历史交互)          │                   │
│  │  • LLM预测交互概率 (Yes/No)          │                   │
│  │  • 选择Top-20模拟交互               │                   │
│  │  • 耗时: <8秒/商品                  │                   │
│  └─────────────────────────────────────┘                   │
│                     │                                       │
│                     ▼                                       │
│  模拟交互集合 + 标准Embedding优化 → 冷商品变为"伪热商品"       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键假设

- **LLM理解用户偏好**：大语言模型具备足够的世界知识理解用户可能的兴趣
- **内容特征充分**：商品标题、描述、图片、类目、价格等信息完整
- **用户历史可靠**：用户的历史交互记录能反映其偏好
- **可扩展性**：Coupled Funnel架构使十亿级用户场景可行

---

## ② 吸奶器出海应用案例

### 场景一：新品吸奶器冷启动推荐

**业务问题**：
一款新上市的智能穿戴式吸奶器上架亚马逊，0销量、0评价、0浏览历史。如何在海量用户中找到最可能购买的目标用户，实现从0到1的突破？

**数据要求**：
- **商品内容特征**：
  - 标题："Wearable Breast Pump Hands-Free, Smart APP Control"
  - 描述：静音设计、APP追踪奶量、适合职场妈妈
  - 类目：母婴 > 喂养 > 吸奶器
  - 价格：$129.99
  - 图片：产品图、使用场景图

- **用户历史数据**：
  - 用户画像：孕期阶段、宝宝月龄
  - 历史交互：浏览过的商品、购买记录
  - 偏好标签：科技爱好者、职场妈妈、价格敏感度

**预期产出**：
- 模拟生成的Top-20潜在购买用户列表
- 每个用户的交互概率评分
- 冷商品Embedding向量（可与热商品统一推荐）
- 推荐的冷启动展示位置优化建议

**业务价值**：
- 冷商品推荐效果提升 **21.69%**：论文报告的**离线推荐指标**（Recall / NDCG）相对提升 —— ⚠️ 这是排序质量指标，**不是转化率**
- 线上冷启动期 GMV：论文 §5.5 的两周 A/B 测试报告 Cold-GMV 相对 Random **+23.80%**
- 降低无效曝光：精准触达高潜用户，减少广告浪费
- ⚠️ 原卡所写的「缩短冷启动周期」一条在 v1 与 v2 中**均无出处**（v2 §5.5 反而把冷启动期定义为「商品发布后至发布后两小时」），已删除

---

### 场景二：季节性新品批量冷启动

**业务问题**：
母婴产品具有强季节性（如夏季婴儿游泳池、冬季暖奶器）。每季度上新50+SKU，每个新品都面临冷启动问题。如何批量处理大规模冷商品上新的推荐冷启动？

**数据要求**：
- **批量商品内容**：
  - 所有新品的结构化描述（标题、卖点、规格）
  - 类目层级和属性标签
  - 定价策略和目标人群

- **系统性能要求**：
  - 吞吐能力：8,640冷商品/小时（基于论文3×8×A100配置）
  - 单商品处理时间：<8秒
  - 日处理能力：~20万冷商品

**预期产出**：
- 批量冷商品的模拟交互数据
- 冷商品Embedding批量更新
- 与现有推荐系统的无缝集成
- 实时冷启动推荐能力提升

**业务价值**：
- **规模化处理能力**：支撑高频上新、快速迭代的业务（本卡场景假设，非论文结论）
- **系统ROI**：⚠️ 论文未给出与人工冷启动策略的效率对比；原卡所写的「自动化处理效率提升」倍数估算无任何出处，已删除
- **持续优化**：冷商品随模拟数据积累快速"变热"

---

## ③ 代码模板

```python
"""
Cold-Start Product Recommendation using LLM Simulator
冷启动商品推荐 - 基于LLM用户行为模拟

基于论文: "Large Language Model Simulator for Cold-Start Recommendation" (WSDM 2025)
GitHub: https://github.com/ColdLLM-Team/ColdLLM-repo
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Product:
    """商品数据结构"""
    item_id: str
    title: str
    description: str
    category: str
    price: float
    attributes: Dict[str, str]


@dataclass
class User:
    """用户数据结构"""
    user_id: str
    history_items: List[str]  # 历史交互商品ID
    history_ratings: List[float]  # 历史评分
    demographics: Dict[str, str]  # 人口统计信息


class ColdStartRecommender:
    """
    冷启动商品推荐器 - ColdLLM简化版
    """

    def __init__(self, n_filter_candidates: int = 100,
                 n_simulate_interactions: int = 20):
        """
        初始化冷启动推荐器

        Args:
            n_filter_candidates: 过滤阶段候选用户数
            n_simulate_interactions: 最终模拟交互数
        """
        self.n_filter_candidates = n_filter_candidates
        self.n_simulate_interactions = n_simulate_interactions

        # 存储
        self.users = {}
        self.products = {}
        self.user_embeddings = {}
        self.interaction_matrix = None

    def add_user(self, user: User, embedding: np.ndarray):
        """添加用户"""
        self.users[user.user_id] = user
        self.user_embeddings[user.user_id] = embedding

    def add_product(self, product: Product):
        """添加商品"""
        self.products[product.item_id] = product

    def extract_product_features(self, product: Product) -> np.ndarray:
        """
        提取商品特征（简化版，实际使用LLM嵌入）

        Args:
            product: 商品对象

        Returns:
            商品特征向量
        """
        # 基于商品属性构建简单特征
        features = []

        # 价格特征（归一化）
        features.append(min(product.price / 200.0, 1.0))

        # 类目编码（简化）
        category_map = {
            'breast_pump': [1, 0, 0, 0],
            'bottle': [0, 1, 0, 0],
            'nipple_care': [0, 0, 1, 0],
            'maternity_clothes': [0, 0, 0, 1]
        }
        features.extend(category_map.get(product.category, [0, 0, 0, 0]))

        # 标题关键词特征
        title_lower = product.title.lower()
        features.append(1.0 if 'smart' in title_lower else 0.0)
        features.append(1.0 if 'wearable' in title_lower else 0.0)
        features.append(1.0 if 'hands-free' in title_lower else 0.0)

        return np.array(features)

    def filtering_stage(self, product: Product) -> List[str]:
        """
        阶段1: 过滤候选用户

        Args:
            product: 新商品

        Returns:
            候选用户ID列表
        """
        product_features = self.extract_product_features(product)

        # 计算用户与商品的相似度
        similarities = []
        for user_id, user_emb in self.user_embeddings.items():
            # 简化：使用余弦相似度
            similarity = np.dot(product_features[:len(user_emb)], user_emb) / (
                np.linalg.norm(product_features[:len(user_emb)]) * np.linalg.norm(user_emb) + 1e-9
            )
            similarities.append((user_id, similarity))

        # 排序并取Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = [uid for uid, _ in similarities[:self.n_filter_candidates]]

        return candidates

    def simulate_user_interaction(self, user: User, product: Product) -> float:
        """
        模拟用户与商品的交互概率（简化版LLM模拟）

        Args:
            user: 用户
            product: 商品

        Returns:
            交互概率 (0-1)
        """
        prob = 0.3  # 基础概率

        # 基于用户画像调整
        demo = user.demographics

        # 类目匹配
        if product.category == 'breast_pump':
            if demo.get('stage') == 'pregnant':
                prob += 0.25
            if demo.get('lifestyle') == 'working_mom':
                prob += 0.20

        # 价格敏感度
        if product.price > 100:
            if demo.get('price_sensitivity') == 'high':
                prob -= 0.15
            elif demo.get('price_sensitivity') == 'low':
                prob += 0.10

        # 历史行为匹配（简化）
        user_cats = set()
        for item_id in user.history_items:
            if item_id in self.products:
                user_cats.add(self.products[item_id].category)

        if product.category in user_cats:
            prob += 0.15

        # 添加随机噪声
        noise = np.random.normal(0, 0.05)
        prob = np.clip(prob + noise, 0, 1)

        return prob

    def refining_stage(self, product: Product,
                      candidate_users: List[str]) -> List[Tuple[str, float]]:
        """
        阶段2: 精筛模拟交互

        Args:
            product: 新商品
            candidate_users: 候选用户列表

        Returns:
            [(user_id, interaction_prob), ...]
        """
        interactions = []

        for user_id in candidate_users:
            user = self.users.get(user_id)
            if not user:
                continue

            prob = self.simulate_user_interaction(user, product)

            if prob > 0.4:  # 阈值
                interactions.append((user_id, prob))

        # 排序取Top-N
        interactions.sort(key=lambda x: x[1], reverse=True)
        return interactions[:self.n_simulate_interactions]

    def recommend_cold_product(self, product: Product) -> Dict:
        """
        完整冷商品推荐流程

        Args:
            product: 新商品

        Returns:
            推荐结果字典
        """
        # Stage 1: 过滤
        candidates = self.filtering_stage(product)

        # Stage 2: 精筛
        interactions = self.refining_stage(product, candidates)

        # 生成冷商品嵌入（基于模拟交互）
        if interactions:
            # 使用高交互概率用户的嵌入加权平均
            weights = np.array([prob for _, prob in interactions])
            user_embs = np.array([self.user_embeddings[uid] for uid, _ in interactions])
            product_embedding = np.average(user_embs, axis=0, weights=weights)
        else:
            product_embedding = self.extract_product_features(product)

        return {
            'item_id': product.item_id,
            'title': product.title,
            'n_candidates': len(candidates),
            'n_simulated': len(interactions),
            'interactions': interactions,
            'embedding': product_embedding,
            'avg_prob': np.mean([p for _, p in interactions]) if interactions else 0
        }

    def recommend_for_user(self, user_id: str, cold_products: List[Product],
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        为用户推荐冷商品

        Args:
            user_id: 用户ID
            cold_products: 冷商品列表
            top_k: 推荐数量

        Returns:
            [(item_id, score), ...]
        """
        if user_id not in self.users:
            return []

        user = self.users[user_id]
        scores = []

        for product in cold_products:
            prob = self.simulate_user_interaction(user, product)
            scores.append((product.item_id, prob))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==================== 示例代码 ====================

def generate_sample_data(n_users: int = 500, n_warm_products: int = 100):
    """生成模拟数据"""
    np.random.seed(42)

    # 生成热商品
    warm_products = {}
    categories = ['breast_pump', 'bottle', 'nipple_care', 'maternity_clothes']

    for i in range(n_warm_products):
        cat = np.random.choice(categories)
        if cat == 'breast_pump':
            title = f"Breast Pump Model {i}"
            desc = "Efficient breast pump for nursing mothers"
            price = np.random.uniform(50, 200)
        elif cat == 'bottle':
            title = f"Baby Bottle {i}"
            desc = "BPA-free baby feeding bottle"
            price = np.random.uniform(10, 40)
        else:
            title = f"Product {i}"
            desc = f"Quality {cat} product"
            price = np.random.uniform(20, 100)

        warm_products[f"warm_{i}"] = Product(
            item_id=f"warm_{i}",
            title=title,
            description=desc,
            category=cat,
            price=price,
            attributes={}
        )

    # 生成用户
    users = []
    lifestyles = ['working_mom', 'stay_at_home', 'first_time_parent']
    stages = ['pregnant', 'newborn', 'infant', 'toddler']
    price_sens = ['high', 'medium', 'low']

    for i in range(n_users):
        # 生成历史交互
        n_history = np.random.randint(5, 20)
        history_items = np.random.choice(list(warm_products.keys()), n_history, replace=False).tolist()
        history_ratings = np.random.uniform(3, 5, n_history).tolist()

        user = User(
            user_id=f"user_{i}",
            history_items=history_items,
            history_ratings=history_ratings,
            demographics={
                'lifestyle': np.random.choice(lifestyles),
                'stage': np.random.choice(stages),
                'price_sensitivity': np.random.choice(price_sens)
            }
        )
        users.append(user)

    return warm_products, users


def demo_cold_start():
    """冷启动推荐演示"""
    print("=" * 60)
    print("冷启动商品推荐演示 - ColdLLM简化版")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    warm_products, users = generate_sample_data(n_users=500, n_warm_products=100)
    print(f"   用户数: {len(users)}")
    print(f"   热商品数: {len(warm_products)}")

    # 2. 初始化推荐器
    print("\n[2] 初始化推荐器...")
    recommender = ColdStartRecommender(
        n_filter_candidates=100,
        n_simulate_interactions=20
    )

    # 添加商品
    for product in warm_products.values():
        recommender.add_product(product)

    # 添加用户（生成简单嵌入）
    for user in users:
        # 基于历史类目的简单嵌入
        cat_counts = {}
        for item_id in user.history_items:
            if item_id in warm_products:
                cat = warm_products[item_id].category
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # 创建7维嵌入
        embedding = np.array([
            cat_counts.get('breast_pump', 0),
            cat_counts.get('bottle', 0),
            cat_counts.get('nipple_care', 0),
            cat_counts.get('maternity_clothes', 0),
            user.history_ratings[-1] if user.history_ratings else 3,
            1.0 if user.demographics['lifestyle'] == 'working_mom' else 0.0,
            1.0 if user.demographics['stage'] == 'pregnant' else 0.0
        ])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

        recommender.add_user(user, embedding)

    # 3. 模拟冷商品
    print("\n[3] 模拟冷商品...")
    cold_products = [
        Product(
            item_id="cold_001",
            title="Smart Wearable Breast Pump with APP Control",
            description="Hands-free wearable breast pump with smart APP for milk tracking",
            category="breast_pump",
            price=129.99,
            attributes={'feature': 'smart'}
        ),
        Product(
            item_id="cold_002",
            title="Eco-Friendly Glass Baby Bottle Set",
            description="Natural glass bottles with silicone sleeve",
            category="bottle",
            price=35.99,
            attributes={'feature': 'eco'}
        ),
        Product(
            item_id="cold_003",
            title="Premium Nipple Cream Organic",
            description="100% organic nipple care cream for nursing mothers",
            category="nipple_care",
            price=18.99,
            attributes={'feature': 'organic'}
        )
    ]

    for product in cold_products:
        result = recommender.recommend_cold_product(product)
        print(f"\n   📦 {result['title'][:40]}...")
        print(f"      候选用户: {result['n_candidates']}, 模拟交互: {result['n_simulated']}")
        print(f"      平均交互概率: {result['avg_prob']:.3f}")
        if result['interactions']:
            top_user, top_prob = result['interactions'][0]
            print(f"      最高意向用户: {top_user} ({top_prob:.3f})")

    # 4. 为用户推荐冷商品
    print("\n[4] 为用户推荐冷商品...")
    test_user = users[0]
    print(f"\n   测试用户: {test_user.user_id}")
    print(f"   画像: {test_user.demographics}")
    print(f"   历史类目: {set(warm_products[i].category for i in test_user.history_items[:5] if i in warm_products)}")

    recommendations = recommender.recommend_for_user(
        test_user.user_id, cold_products, top_k=3
    )

    print("\n   Top 3 冷商品推荐:")
    for rank, (item_id, score) in enumerate(recommendations, 1):
        product = next(p for p in cold_products if p.item_id == item_id)
        print(f"   {rank}. {product.title[:35]}... ({score:.3f})")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)

    return recommender, cold_products


def main():
    """主函数"""
    recommender, cold_products = demo_cold_start()

    print("\n核心要点:")
    print("• Stage 1 (Filtering): 快速筛选高潜用户 (十亿→百级)")
    print("• Stage 2 (Refining): LLM精细模拟交互 (百级→Top-20)")
    print("• 冷商品通过模拟交互获得推荐能力")
    print("• 论文效果: NDCG提升21.69%, 处理<8秒/商品")


if __name__ == '__main__':
    main()
```

---

## ④ 技能关系

### 前置依赖技能
- **新品机会挖掘** - 先预测新品成功概率，再对高潜力新品进行冷启动推荐
- **矩阵分解推荐** - 理解协同过滤基础原理
- **Embedding学习** - 理解用户/商品表征学习

### 可组合技能
- **新品机会挖掘** - 结合成功预测筛选值得冷启动投入的商品
- **多臂老虎机** - 冷启动期结合在线学习优化展示策略
- **因果推断** - 评估冷启动推荐的真实业务增量效果

### 后续扩展技能
- **序列推荐** - 冷启动后积累行为数据，转为序列推荐
- **实时特征平台** - 支持冷商品的实时Embedding更新
- **A/B测试平台** - 冷启动策略效果评估

---

## ⑤ 业务价值评估

| 维度 | 评估 |
|------|------|
| **ROI估算** | 论文 §5.5 两周线上 A/B 测试：冷启动期 Cold-GMV 相对 Random **+23.80%**、Cold-PCTR **+5.60%**、Cold-PV **+11.45%**。⚠️ 论文未给任何 ROI / 金额口径推算；原卡的「冷启动周期」与「预计 ROI 提升」两条估算在 v1 与 v2 中均无出处，已删除 |
| **实施难度** | ★★★★☆ (4/5) - 需要LLM推理能力、GPU资源、FAISS索引基础设施 |
| **数据需求** | ★★★☆☆ (3/5) - 需要商品内容特征（标题、描述）、用户历史交互 |
| **模型复杂度** | ★★★★☆ (4/5) - 双阶段架构较复杂，但开源代码可参考 |
| **可解释性** | ★★★☆☆ (3/5) - LLM决策可解释性一般，但可输出模拟交互用户列表 |
| **优先级评分** | 90/100 - 高频上新业务的必备能力，直接解决0到1突破难题 |

### 适用场景
- ✅ 月均上新50+ SKU的快速迭代品类
- ✅ 依赖算法推荐驱动销售的业务（如亚马逊、独立站首页推荐）
- ✅ 有GPU/LLM推理资源的团队
- ✅ 商品内容信息（标题、描述、图片）完整的业务

### 不适用场景
- ❌ 商品内容信息缺失（如只有商品ID，无描述）
- ❌ 低频上新（季度上新<10 SKU，可用人工策略）
- ❌ 缺乏GPU计算资源的轻量级团队
- ❌ 冷启动期极短的业务（如秒杀、闪购，上架即爆）

### 性能指标参考（基于论文）

| 指标 | 数值 |
|------|------|
| 离线NDCG提升 | +21.69%（冷商品） |
| 整体NDCG提升 | +10.79% |
| 单商品处理时间 | <8秒 |
| 吞吐能力 | 8,640商品/小时 |
| 候选用户筛选 | 十亿→百级 |
| 模拟交互数量 | Top-20/商品 |

---

## ⑥ 原文引用

> ⚠️ **底本版本声明**：本卡依据 **arXiv 2402.09176v2**，即 **WSDM 2025 正式发表版**（arXiv 摘要页 `Comments` 字段写明该文录用于 WSDM 2025；v2 首页含 WSDM 会议头与 DOI）。本仓库底本 `fulltext.md` 已同步为 v2，旧 v1 存档保留在同目录 `fulltext.v1.md`。v1 预印本标题为 *Large Language Model **Interaction** Simulator for Cold-Start **Item** Recommendation*、方法名 `LLM-InS`；**v2 起更名为 `ColdLLM`**，故本卡正文使用 `ColdLLM`。以下引文全部逐字取自 v2。

> 原文："Recommending cold items remains a significant challenge in billion-scale online recommendation systems. While warm items benefit from historical user behaviors, cold items rely solely on content features, limiting their recommendation performance and impacting user experience and revenue."
> 出处：2402.09176v2 §Abstract

> 原文："Current models generate synthetic behavioral embeddings from content features but fail to address the core issue: the absence of historical behavior data."
> 出处：2402.09176v2 §Abstract

> 原文："To manage the computational complexity, we propose a coupled funnel ColdLLM framework for online recommendation. ColdLLM efficiently reduces the number of candidate users from billions to hundreds using a trained coupled filter, allowing the LLM to operate efficiently and effectively on the filtered set."
> 出处：2402.09176v2 §Abstract —— 卡片 ① 段「十亿→百级候选」与 ② 段「Coupled Funnel 双阶段架构」的出处

> 原文："Extensive experiments show that ColdLLM significantly surpasses baselines in cold-start recommendations, including Recall and NDCG metrics. A two-week A/B test also validates that ColdLLM can effectively increase the cold-start period GMV."
> 出处：2402.09176v2 §Abstract —— 卡片 ② 段「两周 A/B 测试 … GMV」的出处

> 原文："In this subsection, we propose the coupled-funnel ColdLLM to incorporate coupled filter models efficiently and effectively simulate cold item behaviors."
> 出处：2402.09176v2 §4.2 Coupled Funnel ColdLLM

> 原文："To filter users who are likely to interact with the cold item, we consider both content embeddings and behavioral embeddings. We use the dot product of the mapped user embedding and the mapped item embedding to identify the top-$K$ highest score candidates"
> 出处：2402.09176v2 §4.2.1 Filtering Simulation 式 (7)

> 原文："We opted for a top-k value of 20."
> 出处：2402.09176 §5.1.3 Hyperparameter Setting（PDF 第 6 页）

> 原文："This top-$K$ computation can be accelerated using efficient similarity search platforms, such as FAISS (Johnson et al., 2019), resulting in ${\mathcal{O}}(1)$ computational complexity."
> 出处：2402.09176v2 §4.2.1 Filtering Simulation

> 原文："Coupled Filtering: Leveraging similarity indexing frameworks like FAISS, we can efficiently downscale the candidate users from billions to hundreds with a complexity of $\mathcal{O}(1)$ in approximately 60 ms."
> 出处：2402.09176v2 §4.4 Implementation Strategy → Complexity Analysis —— 卡片 ① 段「FAISS / O(1) / ~60ms」的出处

> 原文："Coupled Refining: We refine the filtered candidates using fine-tuned LLaMA-7B models to identify 20 qualified users. This process takes about 200-400 ms for each user-item pair. In total, the LLM-refining stage requires less than 8 seconds."
> 出处：2402.09176v2 §4.4 Implementation Strategy → Complexity Analysis —— 卡片 ① 段「微调 LLaMA-7B / 200-400ms / <8 秒」的出处

> 原文："When equipped with three 8$\times$A100 GPU machines, our system can handle a load of 8,640 cold items per hour."
> 出处：2402.09176v2 §4.4 Implementation Strategy —— 卡片 ② 段「3×8×A100 / 8,640 商品/小时」的出处

> 原文："The dimension of the embeddings was standardized to 200 for all models."
> 出处：2402.09176v2 §5.1.3 Hyperparameter Setting —— 卡片 ① 段「嵌入维度 200」的出处

> 原文："Specifically, we employ Recall@$K$ and NDCG@$K$ as our primary metrics, where $k=20$."
> 出处：2402.09176v2 §5.1.4 Evaluation Metrics

> 原文："ColdLLM brings an average NDCG improvement of 10.79% and 37.10% on overall and cold item recommendations over LightGCN."
> 出处：2402.09176v2 §5.2 Main Results (RQ1) —— 卡片 ⑤ 段「整体 NDCG 提升 +10.79%」的出处

> 原文："We conduct extensive offline experiments, demonstrating that our model outperforms existing solutions by 21.69% in cold recommendation performance."
> 出处：2402.09176v2 §1 Introduction —— 卡片 ② / ⑤ 段「冷商品 +21.69%」的出处

> 原文："*Table 1. Comparison results on overall, cold, and warm item recommendations over three backbone models (MF, NGCF, LightGCN). The best and second-best results in each column are highlighted in bold font and underlined.*"
> 出处：2402.09176v2 §5.2 Main Results (RQ1) 表 1 标题

> 原文："| Improvement | 43.48% | 42.75% | 3.28% | 3.18% | 20.28% | 19.35% | 63.41% | 49.33% | 5.84% | 7.88% | 2.64% | 2.66% |"
> 出处：2402.09176v2 §5.2 表 1（NGCF 骨干网络 Improvement 行）。列序 = Overall / Cold / Warm × CiteULike / MovieLens × Recall / NDCG。⚠️ **v2 起该行数值与 v1 不同**，卡片 ⑤ 段引用的 21.69% 与 10.79% **不出自本行**，其出处见上方 §1 与 §5.2 两条

> 原文："To validate the effectiveness of ColdLLM in an industrial setting, we conducted an online A/B test on one of the largest e-commerce platforms. The experiment ran for two consecutive weeks, involving 5% of users for each group."
> 出处：2402.09176v2 §5.5 Online Evaluation (RQ4)

> 原文："We define the cold-start period as the interval from the item’s publication to two hours after its release."
> 出处：2402.09176v2 §5.5 Online Evaluation (RQ4) —— 「冷启动期」的口径定义

> 原文："Specifically, compared to the random baseline, ColdLLM achieves substantial improvements of 11.45% in Cold-PV, 5.60% in Cold-PCTR, and 23.80% in Cold-GMV for cold items."
> 出处：2402.09176v2 §5.5 Online Evaluation (RQ4) 表 3 —— 卡片 ② / ⑤ 段线上冷启动期 Cold-GMV / Cold-PCTR / Cold-PV 提升的出处

> **版本错配记录（2026-09-13 核实）**：本卡曾被依据 v1 预印本（*Large Language Model Interaction Simulator for Cold-Start Item Recommendation*，方法名 `LLM-InS`）判为「方法名与论文不符 / 数字疑似编造」。经逐项核实：卡片的 `ColdLLM`、`WSDM 2025`、`Coupled Funnel`、`10.79%`、`LLaMA-7B`、`FAISS`、`A100`、`8,640` 等**全部是 v2（WSDM 2025 正式版）的事实**，在 v1 中的零命中是**版本差异**所致，**不是编造**。此前依赖 v1 措辞的引文已在上方替换为 v2 逐字引文；v1 存档保留为 `fulltext.v1.md` 以备对照。

---

## 参考论文

- **Large Language Model Simulator for Cold-Start Recommendation**, WSDM 2025, arXiv:2402.09176
- 作者：Feiran Huang, Yuanchen Bei, Zhenghang Yang, et al.
- GitHub: https://github.com/ColdLLM-Team/ColdLLM-repo
- 核心创新：Coupled Funnel双阶段架构 + LLM用户行为模拟
- 实验数据：十亿级用户规模，两周在线A/B测试验证GMV提升
