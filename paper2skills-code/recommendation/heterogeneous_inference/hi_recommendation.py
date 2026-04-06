"""
Deep Learning Recommendation with Heterogeneous Inference (HI)
母婴出海场景：首页个性化推荐、召回层优化

基于论文：Liu et al. (2020) "Simultaneous Relevance and Diversity"
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from collections import defaultdict
import random


class HINetwork(nn.Module):
    """Heterogeneous Inference Network"""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 alpha: float = 0.5, gamma: float = 0.3):
        super(HINetwork, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.gamma = gamma

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_p2p = nn.Embedding(n_items, embedding_dim)
        self.item_embedding_n2p = nn.Embedding(n_items, embedding_dim)
        self.W = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

        for emb in [self.user_embedding, self.item_embedding_p2p, self.item_embedding_n2p]:
            nn.init.normal_(emb.weight, std=0.01)

    def compute_loss(self, users, items_pos, items_neg):
        """Compute HI loss: L = L+ + alpha*L- + gamma*L*"""
        u = self.user_embedding(users)
        i_pos_p2p = self.item_embedding_p2p(items_pos)
        i_pos_n2p = self.item_embedding_n2p(items_pos)

        score_p2p_pos = torch.sum(u * i_pos_p2p, dim=1)
        score_n2p_pos = torch.sum(u * i_pos_n2p, dim=1)

        batch_size, n_neg = items_neg.shape
        neg_items_flat = items_neg.view(-1)
        i_neg_p2p = self.item_embedding_p2p(neg_items_flat).view(batch_size, n_neg, -1)
        i_neg_n2p = self.item_embedding_n2p(neg_items_flat).view(batch_size, n_neg, -1)

        u_expanded = u.unsqueeze(1)
        score_neg_p2p = torch.sum(u_expanded * i_neg_p2p, dim=2)
        score_neg_n2p = torch.sum(u_expanded * i_neg_n2p, dim=2)

        loss_p2p = -torch.mean(torch.log(torch.sigmoid(score_p2p_pos.unsqueeze(1) - score_neg_p2p) + 1e-10))
        loss_n2p = -torch.mean(torch.log(torch.sigmoid(score_n2p_pos.unsqueeze(1) - score_neg_n2p) + 1e-10))

        interaction = torch.sum(i_pos_p2p * torch.mm(i_pos_n2p, self.W.t()), dim=1)
        loss_inter = -torch.mean(torch.log(torch.sigmoid(interaction) + 1e-10))

        return loss_p2p + self.alpha * loss_n2p + self.gamma * loss_inter

    def recommend(self, user_id, top_k=50, exclude_interacted=None):
        """Generate recommendations for a user"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            u = self.user_embedding(user_tensor)

            all_items_p2p = self.item_embedding_p2p.weight
            all_items_n2p = self.item_embedding_n2p.weight

            scores_p2p = torch.mm(u, all_items_p2p.t()).squeeze()
            scores_n2p = torch.mm(u, all_items_n2p.t()).squeeze()

            final_scores = scores_p2p + 0.3 * scores_n2p

            if exclude_interacted is not None:
                mask = torch.ones(self.n_items, dtype=torch.bool)
                mask[exclude_interacted] = False
                final_scores = final_scores[mask]
                original_indices = torch.where(mask)[0]
            else:
                original_indices = torch.arange(self.n_items)

            top_scores, top_indices = torch.topk(final_scores, min(top_k, len(final_scores)))
            return original_indices[top_indices].numpy()


def generate_maternity_data(n_users=1000, n_items=5000, n_interactions=50000):
    """Generate simulated maternity e-commerce data"""
    np.random.seed(42)
    random.seed(42)

    categories = ['奶粉', '纸尿裤', '辅食', '玩具', '童装', '孕妇装', '哺育用品']
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
        interactions.append({'user_id': user, 'item_id': item, 'category': cat, 'segment': segment})

    return pd.DataFrame(interactions), item_info, user_segment


def train_model():
    """Train HI model"""
    print("HI Recommendation Model Training")
    print("=" * 50)

    n_users, n_items = 1000, 5000
    df, item_info, user_segment = generate_maternity_data(n_users, n_items)

    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")

    train_df = df.sample(frac=0.8, random_state=42)

    user_item_set = defaultdict(set)
    for _, row in train_df.iterrows():
        user_item_set[int(row['user_id'])].add(int(row['item_id']))

    model = HINetwork(n_users, n_items, embedding_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n开始训练...")
    n_epochs = 20
    batch_size = 256

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for _ in range(50):
            batch = train_df.sample(batch_size)
            users = torch.LongTensor(batch['user_id'].values)
            pos_items = torch.LongTensor(batch['item_id'].values)

            neg_items = []
            for u in batch['user_id']:
                negatives = []
                while len(negatives) < 5:
                    neg = random.randint(0, n_items - 1)
                    if neg not in user_item_set[int(u)]:
                        negatives.append(neg)
                neg_items.append(negatives)
            neg_items = torch.LongTensor(neg_items)

            optimizer.zero_grad()
            loss, _, _, _ = model.compute_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss={total_loss/n_batches:.4f}")

    print("\n" + "=" * 60)
    print("推荐效果评估")
    print("=" * 60)

    user_id = 0
    recs = model.recommend(user_id, top_k=20, exclude_interacted=list(user_item_set[user_id]))

    print(f"\nRecommendations for user {user_id} ({user_segment[user_id]}):")
    for i, item in enumerate(recs[:10], 1):
        print(f"  {i}. Item {item} [{item_info[int(item)]['category']}]")

    return model, item_info


if __name__ == "__main__":
    train_model()
