# Skill Card: Deep Learning Recommendation with Heterogeneous Inference

---

## ① 算法原理

### 核心思想
传统协同过滤只利用**正向交互**（用户点击、购买）来建模，忽略了**负向信号**（用户不点击、跳过）的价值。Heterogeneous Inference (HI) 通过同时建模两种推理模式：
- **p2p (positive-to-positive)**：正向到正向的收敛性推理
- **n2p (negative-to-positive)**：负向到正向的发散性推理

实现"既精准又多样"的推荐效果，解决推荐系统中常见的"信息茧房"和"过滤气泡"问题。

### 数学直觉
**传统矩阵分解**只建模用户-物品正反馈矩阵 X：
- 目标：找到低秩分解 X^TX ≈ PQ
- 局限：推荐结果收敛到热门物品，多样性不足

**HI方法**同时建模两个通道：
1. **Y通道（负反馈矩阵）**：Y = O - X，其中O是全1矩阵
2. **p2p通道**：捕捉"喜欢A的用户也喜欢B"的收敛相关性
3. **n2p通道**：捕捉"不喜欢A的用户喜欢B"的发散相关性

**损失函数**：
L = L+ + αL- + γL*
- L+：p2p损失（正向相似度）
- L-：n2p损失（负向-正向转换）
- L*：p2p与n2p交互损失

### 关键假设
1. 负反馈可解释：用户未交互的物品包含有效负信号
2. 反馈可分离：正负反馈的联合分布可被分解为p2p和n2p两个子模型
3. 隐语义存在：存在低维隐空间能同时表达用户偏好和物品属性

---

## ② 母婴出海应用案例

### 场景1：跨境电商首页个性化推荐（解决同质化问题）

**业务问题**
某母婴出海APP使用传统协同过滤进行首页推荐，发现：
- 新用户总是被推荐奶粉、纸尿裤等标品，难以发现长尾商品
- 用户反馈"推荐太单一"，复购率停滞在25%
- 想推的新品牌（如本土有机棉品牌）得不到曝光机会

**数据要求**
- user_id, item_id, interaction_type, timestamp
- user_features: baby_age, location
- item_features: category, brand

**预期产出**
- 每位用户的个性化推荐列表（Top 50）
- 多样性指标提升：平均ILS下降30%
- 长尾商品曝光占比从15%提升至35%

**业务价值**
- 品类渗透：帮助用户发现母婴全品类需求
- 新品牌孵化：给本土/新兴品牌公平的曝光机会

### 场景2：召回层优化（候选集生成）

**业务问题**
召回层需要从10万+SKU中筛选候选集。传统i2i召回只基于共现，导致召回结果集中在热门品类。

**预期产出**
- 召回候选集的类目覆盖数提升2倍
- 冷启动商品在召回中的占比达到10%

---

## ③ 代码模板

```python
"""
Heterogeneous Inference for Recommendation
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

class HINetwork(nn.Module):
    """Heterogeneous Inference Network"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, alpha=0.5, gamma=0.3):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_p2p = nn.Embedding(n_items, embedding_dim)
        self.item_embedding_n2p = nn.Embedding(n_items, embedding_dim)
        self.W = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.alpha = alpha
        self.gamma = gamma
        
        # Init
        for emb in [self.user_embedding, self.item_embedding_p2p, self.item_embedding_n2p]:
            nn.init.normal_(emb.weight, std=0.01)
    
    def compute_loss(self, users, pos_items, neg_items):
        """Compute HI loss: L = L+ + αL- + γL*"""
        u = self.user_embedding(users)
        i_pos_p2p = self.item_embedding_p2p(pos_items)
        i_pos_n2p = self.item_embedding_n2p(pos_items)
        
        # p2p score
        score_p2p_pos = torch.sum(u * i_pos_p2p, dim=1)
        
        # n2p score
        score_n2p_pos = torch.sum(u * i_pos_n2p, dim=1)
        
        # Negative samples
        batch_size, n_neg = neg_items.shape
        neg_items_flat = neg_items.view(-1)
        i_neg_p2p = self.item_embedding_p2p(neg_items_flat).view(batch_size, n_neg, -1)
        i_neg_n2p = self.item_embedding_n2p(neg_items_flat).view(batch_size, n_neg, -1)
        
        u_expanded = u.unsqueeze(1)
        score_neg_p2p = torch.sum(u_expanded * i_neg_p2p, dim=2)
        score_neg_n2p = torch.sum(u_expanded * i_neg_n2p, dim=2)
        
        # BPR losses
        loss_p2p = -torch.mean(torch.log(torch.sigmoid(score_p2p_pos.unsqueeze(1) - score_neg_p2p) + 1e-10))
        loss_n2p = -torch.mean(torch.log(torch.sigmoid(score_n2p_pos.unsqueeze(1) - score_neg_n2p) + 1e-10))
        
        # Interaction loss
        interaction = torch.sum(i_pos_p2p * torch.mm(i_pos_n2p, self.W.t()), dim=1)
        loss_inter = -torch.mean(torch.log(torch.sigmoid(interaction) + 1e-10))
        
        return loss_p2p + self.alpha * loss_n2p + self.gamma * loss_inter
    
    def recommend(self, user_id, top_k=50, exclude_interacted=None):
        """Generate recommendations for a user"""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            user_tensor = torch.LongTensor([user_id]).to(device)
            u = self.user_embedding(user_tensor)
            
            all_items_p2p = self.item_embedding_p2p.weight
            all_items_n2p = self.item_embedding_n2p.weight
            
            scores_p2p = torch.mm(u, all_items_p2p.t()).squeeze()
            scores_n2p = torch.mm(u, all_items_n2p.t()).squeeze()
            
            # Combine p2p and n2p
            final_scores = scores_p2p + 0.3 * scores_n2p
            
            if exclude_interacted is not None:
                mask = torch.ones(len(final_scores), dtype=torch.bool)
                mask[exclude_interacted] = False
                final_scores = final_scores[mask]
                original_indices = torch.where(mask)[0]
            else:
                original_indices = torch.arange(len(final_scores))
            
            top_scores, top_indices = torch.topk(final_scores, min(top_k, len(final_scores)))
            return original_indices[top_indices].cpu().numpy()


def generate_maternity_data(n_users=1000, n_items=5000, n_interactions=50000):
    """Generate simulated maternity e-commerce data"""
    np.random.seed(42)
    random.seed(42)
    
    categories = ['奶粉', '纸尿裤', '辅食', '玩具', '童装', '孕妇装']
    items_per_cat = n_items // len(categories)
    
    item_info = {}
    for cat_idx, cat in enumerate(categories):
        for i in range(items_per_cat):
            item_id = cat_idx * items_per_cat + i
            item_info[item_id] = {'category': cat, 'popularity': np.random.exponential(1.0)}
    
    segments = ['孕期', '0-6月', '6-12月', '1-2岁', '2岁+']
    segment_prefs = {
        '孕期': {'孕妇装': 0.4, '奶粉': 0.3, '哺育用品': 0.2, '玩具': 0.1},
        '0-6月': {'奶粉': 0.4, '纸尿裤': 0.3, '哺育用品': 0.2, '玩具': 0.1},
        '6-12月': {'辅食': 0.35, '奶粉': 0.25, '纸尿裤': 0.2, '玩具': 0.2},
        '1-2岁': {'辅食': 0.3, '玩具': 0.3, '童装': 0.2, '奶粉': 0.2},
        '2岁+': {'玩具': 0.35, '童装': 0.35, '辅食': 0.2, '奶粉': 0.1}
    }
    
    user_segment = {u: random.choice(segments) for u in range(n_users)}
    
    interactions = []
    for _ in range(n_interactions):
        user = random.randint(0, n_users - 1)
        segment = user_segment[user]
        prefs = segment_prefs[segment]
        cat = random.choices(list(prefs.keys()), weights=list(prefs.values()))[0]
        cat_items = [i for i, info in item_info.items() if info['category'] == cat]
        cat_pops = [item_info[i]['popularity'] for i in cat_items]
        item = random.choices(cat_items, weights=cat_pops)[0]
        interactions.append({'user_id': user, 'item_id': item, 'category': cat})
    
    return pd.DataFrame(interactions), item_info, user_segment


def train_model():
    """Train HI model"""
    print("HI Recommendation Model Training")
    print("=" * 50)
    
    n_users, n_items = 1000, 5000
    df, item_info, user_segment = generate_maternity_data(n_users, n_items)
    
    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HINetwork(n_users, n_items, embedding_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training loop
    for epoch in range(20):
        model.train()
        # Sample batch
        batch = df.sample(256)
        users = torch.LongTensor(batch['user_id'].values).to(device)
        pos_items = torch.LongTensor(batch['item_id'].values).to(device)
        
        # Negative sampling
        neg_items = torch.randint(0, n_items, (256, 5)).to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Recommend for sample user
    user_id = 0
    recs = model.recommend(user_id, top_k=20)
    print(f"\nRecommendations for user {user_id} ({user_segment[user_id]}):")
    for i, item in enumerate(recs[:10], 1):
        print(f"  {i}. Item {item} [{item_info[item]['category']}]")


if __name__ == "__main__":
    train_model()
```

---

## ④ 技能关联

### 前置技能
- 协同过滤基础：矩阵分解、隐语义模型
- 负采样技术：动态负采样策略
- 深度学习基础：Embedding层、BPR损失

### 延伸技能
- 多目标学习：同时优化点击率和多样性
- 图神经网络推荐：利用用户-物品二部图

### 可组合
- Cold-Start Product Recommendation
- LTV Prediction
- Thompson Sampling (for online exploration)

---

## ⑤ 商业价值评估

### ROI预估
| 指标 | 预估 |
|------|------|
| 长尾商品GMV | +30-50% |
| 新客留存 | +10-20% |
| 品类渗透率 | +25% |

### 实施难度
⭐⭐⭐☆☆ (3/5)

### 优先级评分
⭐⭐⭐⭐☆ (4/5)

---

## 参考资料
- Liu, Y., et al. (2020). "Simultaneous Relevance and Diversity." arXiv:2009.12969.
