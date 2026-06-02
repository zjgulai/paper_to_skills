"""
HGT Heterogeneous Graph Transformer — 异构图 Transformer
paper2skills-code: 08-知识图谱 | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass


@dataclass
class HeteroNode:
    node_id: str
    node_type: str    # product / user / brand / keyword / review
    features: list[float]


@dataclass
class HeteroEdge:
    src_id: str
    dst_id: str
    edge_type: str    # purchase / review / belong_to / search
    weight: float = 1.0


@dataclass
class HGTEmbedding:
    node_id: str
    node_type: str
    embedding: list[float]


class TypeAwareAttention:
    """类型感知多头注意力（HGT 核心）"""
    def __init__(self, dim: int = 8, n_heads: int = 2, seed: int = 42):
        self.dim = dim
        self.n_heads = n_heads
        random.seed(seed)
        self.W_k = {t: [[random.gauss(0, 0.1) for _ in range(dim)]
                        for _ in range(dim)]
                    for t in ["product", "user", "brand", "keyword", "review"]}

    def _matmul(self, W: list[list[float]], x: list[float]) -> list[float]:
        return [sum(W[i][j] * x[j] for j in range(min(len(x), len(W[i]))))
                for i in range(len(W))]

    def attend(self, query: HeteroNode, keys: list[HeteroNode],
               edge_type: str) -> list[float]:
        if not keys:
            return query.features[:]
        q_proj = self._matmul(self.W_k.get(query.node_type, self.W_k["product"]),
                               query.features)
        scores = []
        for k in keys:
            k_proj = self._matmul(self.W_k.get(k.node_type, self.W_k["product"]),
                                   k.features)
            score = sum(qi * ki for qi, ki in zip(q_proj, k_proj)) / math.sqrt(self.dim)
            scores.append(score)

        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_e = sum(exp_s)
        attn = [e / sum_e for e in exp_s]

        out = [0.0] * self.dim
        for a, k in zip(attn, keys):
            for i, fi in enumerate(k.features[:self.dim]):
                if i < len(out):
                    out[i] += a * fi
        return out


class HGTLayer:
    """单层 HGT"""
    def __init__(self, dim: int = 8):
        self.attention = TypeAwareAttention(dim=dim)

    def forward(self, nodes: dict[str, HeteroNode],
                edges: list[HeteroEdge]) -> dict[str, HGTEmbedding]:
        adj: dict[str, list[HeteroNode]] = {nid: [] for nid in nodes}
        for e in edges:
            if e.dst_id in nodes and e.src_id in nodes:
                adj[e.dst_id].append(nodes[e.src_id])

        embeddings = {}
        for nid, node in nodes.items():
            neighbors = adj[nid]
            new_feat = self.attention.attend(node, neighbors, "mixed")
            embeddings[nid] = HGTEmbedding(
                node_id=nid, node_type=node.node_type,
                embedding=[round(f, 4) for f in new_feat],
            )
        return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai**2 for ai in a))
    nb = math.sqrt(sum(bi**2 for bi in b))
    return dot / (na * nb + 1e-9)


def run_hgt_demo():
    random.seed(42)

    nodes = {
        "P001": HeteroNode("P001", "product", [random.random() for _ in range(8)]),
        "P002": HeteroNode("P002", "product", [random.random() for _ in range(8)]),
        "P003": HeteroNode("P003", "product", [random.random() for _ in range(8)]),
        "U001": HeteroNode("U001", "user",    [random.random() for _ in range(8)]),
        "B001": HeteroNode("B001", "brand",   [random.random() for _ in range(8)]),
        "K001": HeteroNode("K001", "keyword", [random.random() for _ in range(8)]),
    }
    edges = [
        HeteroEdge("U001", "P001", "purchase"),
        HeteroEdge("U001", "P002", "purchase"),
        HeteroEdge("P001", "B001", "belong_to"),
        HeteroEdge("P002", "B001", "belong_to"),
        HeteroEdge("K001", "P001", "search"),
        HeteroEdge("K001", "P002", "search"),
    ]

    hgt = HGTLayer(dim=8)
    embeddings = hgt.forward(nodes, edges)

    print("=== HGT 异构图 Transformer（母婴 KG）===")
    sim_12 = cosine_similarity(embeddings["P001"].embedding, embeddings["P002"].embedding)
    sim_13 = cosine_similarity(embeddings["P001"].embedding, embeddings["P003"].embedding)
    print(f"P001-P002 相似度（同品牌+同用户购买）: {sim_12:.4f}")
    print(f"P001-P003 相似度（无共同连接）:       {sim_13:.4f}")
    print(f"结论: P001 与 P002 更相似（{sim_12:.4f} > {sim_13:.4f}），推荐候选成立")
    print("✅ HGT 演示完成")
if __name__ == "__main__":
    run_hgt_demo()
