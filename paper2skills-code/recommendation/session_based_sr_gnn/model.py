"""
SR-GNN: Session-based Recommendation with Graph Neural Networks
基于论文: Session-based Recommendation with Graph Neural Networks (AAAI 2019)
arXiv: 1811.00855
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SessionGraph:
    """将 session 序列构建为有向图"""

    def __init__(self, session_items: List[int]):
        self.session_items = session_items
        self.nodes: Set[int] = set(session_items)
        self.edges: Dict[Tuple[int, int], int] = defaultdict(int)
        self.in_neighbors: Dict[int, List[int]] = defaultdict(list)
        self.out_neighbors: Dict[int, List[int]] = defaultdict(list)
        self._build_graph()

    def _build_graph(self):
        for i in range(len(self.session_items) - 1):
            src = self.session_items[i]
            dst = self.session_items[i + 1]
            self.edges[(src, dst)] += 1
            self.out_neighbors[src].append(dst)
            self.in_neighbors[dst].append(src)

    def get_neighbors_with_weights(self, node: int):
        """返回邻居及其归一化权重"""
        neighbors = self.out_neighbors[node]
        if not neighbors:
            return []
        weights = defaultdict(float)
        for n in neighbors:
            weights[n] += self.edges[(node, n)]
        total = sum(weights.values())
        return [(n, w / total) for n, w in weights.items()]


class SRGNN(nn.Module):
    """SR-GNN 核心模型"""

    def __init__(self, n_items: int, hidden_dim: int = 100, n_layers: int = 1):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.item_embedding = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        self.W_gnn_1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gnn_2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_attn = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _gnn_propagate(self, session_graphs: List[SessionGraph]):
        batch_node_reprs = []
        device = next(self.parameters()).device
        for graph in session_graphs:
            node_reprs = {}
            for node in graph.nodes:
                node_reprs[node] = self.item_embedding(torch.tensor(node, device=device))
            for _ in range(self.n_layers):
                new_reprs = {}
                for node in graph.nodes:
                    neighbors = graph.get_neighbors_with_weights(node)
                    if neighbors:
                        neighbor_embeds = torch.stack([node_reprs[n] * w for n, w in neighbors])
                        agg = neighbor_embeds.sum(dim=0)
                    else:
                        agg = torch.zeros(self.hidden_dim, device=device)
                    self_repr = self.W_gnn_1(node_reprs[node])
                    neighbor_repr = self.W_gnn_2(agg)
                    new_reprs[node] = F.relu(self_repr + neighbor_repr)
                node_reprs = new_reprs
            batch_node_reprs.append(node_reprs)
        return batch_node_reprs

    def _compute_session_repr(self, node_reprs: Dict[int, torch.Tensor], session_items: List[int]):
        all_nodes = torch.stack(list(node_reprs.values()))
        s_global = all_nodes.mean(dim=0)
        node_list = list(node_reprs.values())
        attn_input = torch.stack(node_list)
        attn_scores = self.q(torch.sigmoid(self.W_attn(attn_input) + s_global))
        attn_weights = F.softmax(attn_scores, dim=0)
        return (attn_weights * attn_input).sum(dim=0)

    def forward(self, session_graphs: List[SessionGraph], session_items_list: List[List[int]]):
        batch_node_reprs = self._gnn_propagate(session_graphs)
        session_reprs = []
        for node_reprs, session_items in zip(batch_node_reprs, session_items_list):
            session_reprs.append(self._compute_session_repr(node_reprs, session_items))
        session_batch = self.W_out(torch.stack(session_reprs))
        return torch.matmul(session_batch, self.item_embedding.weight.t())


class SessionDataset(Dataset):
    def __init__(self, sessions: List[List[int]], max_len: int = 19):
        self.sessions = sessions
        self.max_len = max_len
    def __len__(self):
        return len(self.sessions)
    def __getitem__(self, idx):
        session = self.sessions[idx]
        if len(session) < 2:
            session = [0] + session
        return session[:-1][-self.max_len:], session[-1]


def collate_fn(batch):
    sessions, targets = zip(*batch)
    return list(sessions), list(targets)


def generate_mombaby_sessions(n_sessions: int = 5000, n_items: int = 500):
    random.seed(42)
    np.random.seed(42)
    categories = {
        "feeding": list(range(1, 51)),
        "diaper": list(range(51, 101)),
        "clothing": list(range(101, 151)),
        "skincare": list(range(151, 201)),
        "gear": list(range(201, 251)),
        "toys": list(range(251, 301)),
        "safety": list(range(301, 351)),
        "mom": list(range(351, 401)),
        "accessories": list(range(401, 451)),
        "nutrition": list(range(451, 501)),
    }
    patterns = [
        ("feeding", "accessories", "feeding"),
        ("feeding", "nutrition", "feeding"),
        ("gear", "accessories", "gear"),
        ("diaper", "skincare", "diaper"),
        ("clothing", "clothing", "accessories"),
        ("safety", "safety", "accessories"),
    ]
    sessions = []
    for _ in range(n_sessions):
        if random.random() < 0.7:
            pattern = random.choice(patterns)
            session = []
            for cat in pattern:
                session.extend(random.sample(categories[cat], random.randint(1, 3)))
            if random.random() < 0.3 and len(session) > 2:
                session.insert(random.randint(1, len(session)-1), session[0])
        else:
            cats = random.sample(list(categories.keys()), random.randint(2, 4))
            session = [random.choice(categories[cat]) for cat in cats]
        if len(session) >= 2:
            sessions.append(session)
    return sessions


def train_srgnn(sessions, n_items, epochs=10, batch_size=100, lr=0.001, hidden_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = SRGNN(n_items=n_items, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = SessionDataset(sessions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_sessions, batch_targets in loader:
            session_graphs = [SessionGraph(s) for s in batch_sessions]
            optimizer.zero_grad()
            scores = model(session_graphs, batch_sessions)
            loss = criterion(scores, torch.tensor(batch_targets, device=device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model


def evaluate(model, test_sessions, k=20):
    device = next(model.parameters()).device
    model.eval()
    recalls, mrrs = [], []
    with torch.no_grad():
        for session in test_sessions:
            if len(session) < 2:
                continue
            graph = SessionGraph(session[:-1])
            scores = model([graph], [session[:-1]])
            probs = F.softmax(scores[0], dim=0)
            top_k = torch.topk(probs, k).indices.cpu().numpy()
            target = session[-1]
            recalls.append(1.0 if target in top_k else 0.0)
            if target in top_k:
                mrrs.append(1.0 / (list(top_k).index(target) + 1))
            else:
                mrrs.append(0.0)
    return np.mean(recalls), np.mean(mrrs)


def predict_next_items(model, session_items, item_id_to_name, top_k=5):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        graph = SessionGraph(session_items)
        scores = model([graph], [session_items])
        probs = F.softmax(scores[0], dim=0)
        top_k_indices = torch.topk(probs, top_k).indices.cpu().numpy()
        top_k_scores = probs[top_k_indices].cpu().numpy()
    return [(item_id_to_name.get(int(i), f"Item_{i}"), float(s)) for i, s in zip(top_k_indices, top_k_scores)]


if __name__ == "__main__":
    print("=" * 60)
    print("SR-GNN: Session-Based Recommendation")
    print("=" * 60)
    N_ITEMS, N_TRAIN, N_TEST = 501, 4000, 1000
    print("\n[1] 生成模拟 session 数据...")
    all_sessions = generate_mombaby_sessions(n_sessions=N_TRAIN + N_TEST, n_items=N_ITEMS)
    train_sessions, test_sessions = all_sessions[:N_TRAIN], all_sessions[N_TRAIN:]
    print(f"    训练集: {len(train_sessions)} sessions, 测试集: {len(test_sessions)} sessions")
    print("\n[2] 训练 SR-GNN...")
    model = train_srgnn(train_sessions, n_items=N_ITEMS, epochs=5, batch_size=50)
    print("\n[3] 评估模型...")
    recall, mrr = evaluate(model, test_sessions, k=20)
    print(f"    Recall@20: {recall:.4f}, MRR@20: {mrr:.4f}")
    print("\n[4] 实时推荐演示...")
    item_names = {
        1: "吸奶器配件", 2: "储奶袋", 3: "温奶器", 4: "母乳保鲜包",
        5: "便携冰袋", 6: "背奶包", 7: "防溢乳垫", 8: "奶瓶消毒器",
        10: "电动吸奶器", 15: "奶嘴刷", 20: "奶粉盒", 25: "奶瓶",
        30: "安抚奶嘴", 40: "辅食机", 45: "咬咬乐", 50: "学饮杯",
        205: "婴儿推车", 210: "推车雨罩", 215: "推车挂钩", 220: "安全座椅",
        405: "推车杯架", 410: "储物袋", 415: "遮阳棚",
    }
    for session in [[1, 2, 3], [205, 210, 205], [25, 30, 25]]:
        names = [item_names.get(i, f"Item_{i}") for i in session]
        print(f"\n    当前 Session: {' -> '.join(names)}")
        for name, score in predict_next_items(model, session, item_names, top_k=5):
            print(f"      -> {name} (置信度: {score:.3f})")
    print(f"\n{'=' * 60}\nSR-GNN 测试完成\n{'=' * 60}")
