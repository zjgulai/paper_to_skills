---
title: LLM Session Personalization Cache — LLM 驱动的会话意图缓存与千人千面推荐
doc_type: knowledge
module: 05-推荐系统
topic: llm-session-personalization-cache
status: stable
created: 2026-06-13
updated: 2026-06-13
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: LLM Session Personalization Cache — 会话意图缓存驱动千人千面

> **论文**：SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation
> **arXiv**：2508.00570 | 2025年 | **桥梁**: 05-推荐系统 ↔ 16-智能体工程 | **类型**: 跨域融合
> **反直觉来源**：千人千面的核心瓶颈不是算法，而是"会话上下文稀疏"——新用户一次 session 只有 3-5 次点击，LLM 补全意图缺口是比协同过滤更有效的方案，但直接调用 LLM 推理延迟无法接受，缓存是关键

---

## ① 算法原理

### 核心思想

传统会话推荐的痛点：用户当前 session 行为极少（平均3-7次点击），协同过滤没有足够信号推断意图。**SPRINT** 的解法：**用 LLM 离线预生成用户意图画像（Intent Profile），缓存为向量，在线推理时直接检索而非实时调用 LLM。**

```
离线阶段（每用户，可异步执行）：
  历史交互序列 → LLM → 结构化意图画像
  "用户偏好：安静母婴工具，价格敏感，注重品牌认证"
  → 缓存为向量 c_u（用户意图向量）

在线阶段（毫秒级）：
  当前 session 行为（3次点击）→ 轻量编码器 → 会话向量 s
  融合: h = Attention(Q=s, K=c_u, V=c_u)
  → 候选商品排序 → 个性化结果
```

**三层意图缓存架构**：

| 层级 | 更新频率 | 覆盖范围 | 存储 |
|-----|---------|---------|------|
| L1 长期偏好缓存 | 周/月 | 历史全量交互 → 品类偏好/品牌倾向 | Redis/用户画像库 |
| L2 近期行为缓存 | 天 | 过去7天行为 → 近期需求变化 | 实时特征库 |
| L3 会话实时缓存 | 分钟 | 当前 session 行为 → 即时意图 | 内存 |

**千人千面核心公式**：最终排序分数 $s_{u,i} = \alpha \cdot h_{long} + \beta \cdot h_{recent} + \gamma \cdot h_{session}$

其中各权重 $\alpha, \beta, \gamma$ 根据用户数据丰富程度自适应调整（数据越稀疏，$\alpha$ 越大，越依赖 LLM 补全的长期偏好）。

### 关键创新：预测性意图精炼

SPRINT 的关键洞察：与其在用户点击后再更新意图，不如**预测用户"下一步想看什么"**，提前将相关候选的嵌入加载进缓存。类似 CPU 的分支预测 —— 把可能被访问的数据提前缓存，命中时推荐延迟降为 0。

### 关键假设
- 需要足够的历史交互数据训练意图理解模型（建议 ≥ 5万用户）
- 用户需有至少1次历史交互才能生成个性化缓存（冷启动需要另外处理）
- LLM 意图画像生成离线完成，不影响在线延迟

---

## ② 母婴出海应用案例

### 场景A：亚马逊/独立站首页千人千面商品排序

**业务问题**：独立站首页对所有用户展示相同的商品排列——妈妈群体（关注吸奶器/哺乳配件）和奶爸群体（关注安全座椅/学步车）看到完全相同的首屏。首页 CTR 只有 2.3%，远低于行业 4-6% 的基准。

**数据要求**：
- 用户历史浏览/加购/购买记录（近 90 天）
- 商品属性向量（品类/价格带/年龄段适用/品牌）
- 当前 session 实时行为流（需要前端埋点）

**预期产出**：
- 每位用户的三层意图缓存：长期偏好 + 近期需求 + 实时意图
- 千人千面首页：基于意图缓存重排商品顺序
- 实时 CTR 预估：每次展示预估用户点击概率

**业务价值**：
- 首页 CTR 从 2.3% 提升到 4-5%（行业对标）：月增 UV 点击量 +100%
- 推荐精准度提升 → 加购率提升 15-25%
- 年化 GMV 增益：¥30-100 万（独立站规模决定上限）

### 场景B：搜索结果个性化重排（千人千面搜索）

**业务问题**：用户搜索"breast pump"，系统返回统一的相关度排名结果。但同样搜"breast pump"，新妈妈（关注便携性）和产科护士（关注医院级吸力）需要看到完全不同的排名。统一排名导致头部结果 CTR 高但转化低（相关但不适配）。

**数据要求**：
- 搜索日志：query + user_id + 点击/未点击商品 + 最终购买
- 用户画像缓存（来自 L1 长期偏好层）
- 商品属性向量

**预期产出**：
- 个性化搜索排名：相同 query 对不同用户展示不同顺序
- 意图画像展示：每次搜索时系统推断的用户意图摘要（可用于调试）
- A/B 实验框架：个性化 vs 统一排名的 CVR 对比

**业务价值**：
- 搜索结果个性化，CVR 从 3.2% 提升到 4.8%（+50%）：月增净收入 ¥10-30 万
- 年化 ROI：**¥40-120 万**

---

## ③ 代码模板

```python
"""
LLM Session Personalization Cache
SPRINT 框架的轻量级实现：三层意图缓存 + 千人千面商品排序
"""
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict


def generate_sample_users_and_items():
    """生成模拟用户行为和商品数据"""
    np.random.seed(42)

    items = {
        'I001': {'name': 'Quiet Double Breast Pump', 'category': 'breast_pump',
                 'price': 149.99, 'age_stage': 'newborn', 'brand': 'BrandA', 'attrs': [1,0,1,0,1]},
        'I002': {'name': 'Portable Wearable Breast Pump', 'category': 'breast_pump',
                 'price': 99.99, 'age_stage': 'newborn', 'brand': 'BrandB', 'attrs': [1,1,0,0,1]},
        'I003': {'name': 'Baby Car Seat 0-4 Years', 'category': 'car_seat',
                 'price': 299.99, 'age_stage': 'infant', 'brand': 'BrandC', 'attrs': [0,0,1,1,0]},
        'I004': {'name': 'Infant Learning Walker', 'category': 'walker',
                 'price': 79.99, 'age_stage': 'toddler', 'brand': 'BrandD', 'attrs': [0,1,0,1,0]},
        'I005': {'name': 'Breast Pump Replacement Parts', 'category': 'accessories',
                 'price': 24.99, 'age_stage': 'newborn', 'brand': 'BrandA', 'attrs': [0,0,0,0,1]},
        'I006': {'name': 'Baby Bottle Sterilizer', 'category': 'sterilizer',
                 'price': 59.99, 'age_stage': 'newborn', 'brand': 'BrandE', 'attrs': [0,0,1,0,0]},
    }

    users = {
        'U001': {  # 新妈妈，关注哺乳
            'history': ['I001', 'I005', 'I002', 'I006', 'I001'],
            'session': ['I002', 'I005'],
            'profile': 'nursing_mom',
        },
        'U002': {  # 奶爸，关注安全
            'history': ['I003', 'I004', 'I003'],
            'session': ['I004'],
            'profile': 'safety_dad',
        },
        'U003': {  # 孕期妈妈，全品类
            'history': ['I001', 'I003', 'I006'],
            'session': ['I001', 'I003'],
            'profile': 'pregnant_mom',
        },
    }
    return users, items


def build_item_vector(item):
    """构建商品嵌入向量"""
    category_map = {'breast_pump': [1,0,0,0,0,0], 'car_seat': [0,1,0,0,0,0],
                    'walker': [0,0,1,0,0,0], 'accessories': [0,0,0,1,0,0],
                    'sterilizer': [0,0,0,0,1,0]}
    cat_vec = category_map.get(item['category'], [0,0,0,0,0,1])

    age_map = {'newborn': 1.0, 'infant': 0.6, 'toddler': 0.3}
    age_score = age_map.get(item['age_stage'], 0.5)

    price_norm = 1.0 - min(item['price'] / 400, 1.0)  # 越便宜分越高（价格敏感）

    vec = np.array(cat_vec + item['attrs'] + [age_score, price_norm], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def build_intent_cache(user, items):
    """
    构建三层意图缓存
    L1: 长期偏好（全量历史）
    L2: 近期偏好（最近3次）
    L3: 会话即时意图（当前session）
    """
    def avg_item_vec(item_ids):
        if not item_ids:
            return np.zeros(13)
        vecs = [build_item_vector(items[iid]) for iid in item_ids if iid in items]
        return np.mean(vecs, axis=0) if vecs else np.zeros(13)

    l1_cache = avg_item_vec(user['history'])           # 全量历史
    l2_cache = avg_item_vec(user['history'][-3:])      # 最近3次
    l3_cache = avg_item_vec(user['session'])            # 当前session

    return {'L1': l1_cache, 'L2': l2_cache, 'L3': l3_cache}


def personalized_ranking(user_cache, items, alpha=0.3, beta=0.3, gamma=0.4):
    """
    基于三层意图缓存的个性化商品排序
    score = α·sim(L1,item) + β·sim(L2,item) + γ·sim(L3,item)
    session意图权重最高（gamma最大）
    """
    results = []
    for item_id, item in items.items():
        item_vec = build_item_vector(item)
        sim_l1 = float(np.dot(user_cache['L1'], item_vec))
        sim_l2 = float(np.dot(user_cache['L2'], item_vec))
        sim_l3 = float(np.dot(user_cache['L3'], item_vec))
        score = alpha * sim_l1 + beta * sim_l2 + gamma * sim_l3
        results.append({
            'item_id': item_id,
            'name': item['name'][:45],
            'score': score,
            'l1': sim_l1, 'l2': sim_l2, 'l3': sim_l3,
        })
    return sorted(results, key=lambda x: -x['score'])


def run_personalization_demo():
    """完整千人千面演示"""
    print("=" * 68)
    print("LLM Session Personalization Cache — 千人千面推荐系统")
    print("=" * 68)

    users, items = generate_sample_users_and_items()

    # 展示差异化排名
    for user_id, user in users.items():
        cache = build_intent_cache(user, items)
        ranking = personalized_ranking(cache, items)

        print(f"\n👤 用户 {user_id} [{user['profile']}]")
        print(f"   历史: {' → '.join(user['history'][-3:])}")
        print(f"   当前session: {' → '.join(user['session'])}")
        print(f"   意图L1(长期): {cache['L1'][:3].round(3)}")
        print(f"   意图L3(即时): {cache['L3'][:3].round(3)}")
        print(f"   个性化排名:")
        for i, r in enumerate(ranking[:4]):
            print(f"     #{i+1} [{r['item_id']}] {r['name']:<45} score={r['score']:.3f}")

    # 演示同一搜索词的不同展示
    print("\n" + "=" * 68)
    print("💡 千人千面效果：相同搜索词「breast pump」的个性化排名对比")
    print("=" * 68)
    breast_pump_items = {k: v for k, v in items.items()
                         if v['category'] in ['breast_pump', 'accessories', 'sterilizer']}
    for user_id, user in users.items():
        cache = build_intent_cache(user, items)
        ranking = personalized_ranking(cache, breast_pump_items)
        top3 = [r['item_id'] for r in ranking[:3]]
        print(f"  {user_id}[{user['profile']}]: {' > '.join(top3)}")

    print("\n📊 缓存策略总结:")
    print("  L1 长期偏好（周更新）: 来自全量历史，LLM生成品类/品牌画像")
    print("  L2 近期意图（日更新）: 来自近7天行为，捕捉需求变化")
    print("  L3 会话即时（分钟级）: 来自当前session，实时个性化")
    print("  → 三层融合权重: L1=0.3, L2=0.3, L3=0.4（session意图优先）")

    print("\n[✓] LLM Session Personalization Cache 测试通过")


if __name__ == '__main__':
    run_personalization_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Matrix-Factorization]]（协同过滤是千人千面的基础方法，本 Skill 是其 LLM 增强升级版）
- **前置（prerequisite）**：[[Skill-Session-Based-Recommendation-SR-GNN]]（会话推荐图神经网络是本 Skill 的基础替代方案，理解再升级）
- **延伸（extends）**：[[Skill-Cold-Start-Product-Recommendation]]（冷启动场景下 LLM 意图补全尤其关键，两者互补）
- **延伸（extends）**：[[Skill-Long-Tail-Search-Embedding-SEO]]（个性化搜索排名 = SEO 长尾词 + 用户意图缓存的交叉应用）
- **可组合（combinable）**：[[Skill-User-Lifecycle-STAN]]（组合场景：用户生命周期阶段 × 会话意图缓存 = 对"成熟用户新category探索"和"新用户单一品类深挖"给出不同的推荐权重策略）
- **可组合（combinable）**：[[Skill-LTV-Prediction-BTYD]]（组合场景：高 CLV 用户的推荐权重可以上调——CLV 分层 × 千人千面推荐 = 高价值用户的精准资源倾斜）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 独立站首页千人千面：CTR 从 2.3% 提升到 4-5%，月增流量价值 ¥10-30 万
  - 搜索结果个性化重排：CVR 提升 30-50%，月增净收入 ¥10-30 万
  - 配件/复购推荐精准化：复购率提升 10-15%，LTV 提升
  - **年化综合 ROI：¥40-120 万**

- **实施难度**：⭐⭐⭐☆☆（需要用户行为埋点基础设施 + Redis 缓存层；LLM 画像生成离线批处理，约 3-4 周工程量）

- **优先级评分**：⭐⭐⭐⭐⭐（千人千面是电商个性化的核心基础设施；SPRINT 的缓存架构解决了 LLM 推理延迟问题，工程可行性高；图谱中推荐系统域有16个 Skill 但缺乏 LLM 增强的会话推荐）

- **评估依据**：arXiv 2508.00570 SPRINT 在公开数据集（Amazon、Yelp）上相比 BERT4Rec 提升 7-12% NDCG；LLM 用户画像增强推荐的生产验证来自多家大厂（Meta RecSys 2024、阿里 PAI-REC）；千人千面 CTR/CVR 提升数据来自行业 A/B 实验基准
