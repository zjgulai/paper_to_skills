"""
Popularity-Aware Meta-Learning (PAM) for Cold-Start Recommendation
Skill: Skill-Cold-Start-Meta-Learning-PAM.md
"""

import torch
import torch.nn as nn


class PAMModel(nn.Module):
    """流行度感知的元学习推荐模型"""

    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return torch.sigmoid((u * i).sum(dim=-1))


def cold_start_adapt(model, new_item_id, few_interactions, inner_steps=5, inner_lr=0.01):
    """新品快速适应"""
    adapted = PAMModel(model.user_emb.num_embeddings, model.item_emb.num_embeddings, 64)
    adapted.load_state_dict(model.state_dict())
    opt = torch.optim.SGD(adapted.parameters(), lr=inner_lr)

    for _ in range(inner_steps):
        users = torch.LongTensor([x[0] for x in few_interactions])
        items = torch.LongTensor([x[1] for x in few_interactions])
        ratings = torch.FloatTensor([x[2] for x in few_interactions])

        loss = nn.BCELoss()(adapted(users, items), ratings)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return adapted


if __name__ == '__main__':
    model = PAMModel(n_users=100, n_items=50)
    # 模拟少量交互: [(user_id, item_id, rating), ...]
    few = [(0, 0, 1.0), (1, 0, 1.0), (2, 0, 0.0), (3, 0, 1.0)]
    adapted = cold_start_adapt(model, 0, few)
    print("Cold-start adaptation done")
