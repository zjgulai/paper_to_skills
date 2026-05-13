"""
HGT (Heterogeneous Graph Transformer) — 电商异构图表示学习
基于论文: Hu et al. "Heterogeneous Graph Transformer", WWW 2020

核心能力:
1. 异构互注意力 — 基于 meta relation (源类型, 边类型, 目标类型) 参数化注意力
2. 异构消息传递 — 类型相关的消息投影与传递
3. 目标特定聚合 — 按目标节点类型聚合邻居信息
4. HGSampling — 保持类型平衡的异构子图采样

母婴电商场景: 用户-产品-评论-属性 异构图的表示学习与节点分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Set


class HGTLayer(nn.Module):
    """
    单层 Heterogeneous Graph Transformer

    核心创新: 用 meta relation (源类型, 边类型, 目标类型) 分解注意力权重,
    而非为每种边类型单独设参数, 实现参数共享与泛化。
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int,
                 node_types: List[str], edge_types: List[Tuple[str, str, str]],
                 dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.node_types = node_types
        self.edge_types = edge_types

        self.k_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim) for ntype in node_types
        })
        self.q_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim) for ntype in node_types
        })
        self.v_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim) for ntype in node_types
        })

        self.msg_linears = nn.ModuleDict({
            f"{src}__{rel}__{dst}": nn.Linear(out_dim, out_dim)
            for src, rel, dst in edge_types
        })

        self.priors = nn.ParameterDict({
            f"{src}__{rel}__{dst}": nn.Parameter(torch.zeros(num_heads))
            for src, rel, dst in edge_types
        })

        self.a_linears = nn.ModuleDict({
            ntype: nn.Linear(out_dim, out_dim) for ntype in node_types
        })

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, node_features: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}

        for dst_type in self.node_types:
            if dst_type not in node_features:
                continue

            attn_outputs = []

            for src_type, rel_type, dt in self.edge_types:
                if dt != dst_type:
                    continue
                if src_type not in node_features:
                    continue

                edge_key = (src_type, rel_type, dst_type)
                if edge_key not in edge_index_dict:
                    continue

                edge_index = edge_index_dict[edge_key]
                src_nodes = edge_index[0]
                dst_nodes = edge_index[1]

                src_feat = node_features[src_type]
                dst_feat = node_features[dst_type]

                K = self.k_linears[src_type](src_feat).view(-1, self.num_heads, self.head_dim)
                Q = self.q_linears[dst_type](dst_feat).view(-1, self.num_heads, self.head_dim)
                V = self.v_linears[src_type](src_feat).view(-1, self.num_heads, self.head_dim)

                K_edges = K[src_nodes]
                Q_edges = Q[dst_nodes]
                V_edges = V[src_nodes]

                attn_scores = (Q_edges * K_edges).sum(dim=-1) / (self.head_dim ** 0.5)
                prior = self.priors[f"{src_type}__{rel_type}__{dst_type}"]
                attn_scores = attn_scores + prior.unsqueeze(0)

                attn_weights = self._softmax_by_dst(attn_scores, dst_nodes, dst_feat.size(0))
                attn_weights = self.dropout(attn_weights)

                messages = attn_weights.unsqueeze(-1) * V_edges

                msg_key = f"{src_type}__{rel_type}__{dst_type}"
                messages_flat = messages.view(-1, self.out_dim)
                messages_proj = self.msg_linears[msg_key](messages_flat)
                messages_proj = messages_proj.view(-1, self.num_heads, self.head_dim)

                aggregated = self._aggregate_by_dst(messages_proj, dst_nodes, dst_feat.size(0))
                attn_outputs.append(aggregated)

            if len(attn_outputs) == 0:
                output[dst_type] = node_features[dst_type]
                continue

            combined = torch.stack(attn_outputs, dim=0).sum(dim=0)
            combined = combined.view(-1, self.out_dim)

            transformed = self.a_linears[dst_type](combined)
            transformed = self.dropout(transformed)

            residual = node_features[dst_type]
            if residual.size(-1) != self.out_dim:
                residual = F.linear(residual, torch.eye(self.out_dim, residual.size(-1)))

            output[dst_type] = self.layer_norm(transformed + residual)

        return output

    def _softmax_by_dst(self, scores: torch.Tensor, dst_nodes: torch.Tensor,
                        num_dst: int) -> torch.Tensor:
        max_per_dst = torch.zeros(num_dst, scores.size(1), device=scores.device)
        idx = dst_nodes.unsqueeze(1).expand(-1, scores.size(1))
        max_per_dst.scatter_reduce_(0, idx, scores, reduce="amax", include_self=False)
        exp_scores = torch.exp(scores - max_per_dst[dst_nodes])
        sum_per_dst = torch.zeros(num_dst, scores.size(1), device=scores.device)
        sum_per_dst.scatter_add_(0, idx, exp_scores)
        return exp_scores / (sum_per_dst[dst_nodes] + 1e-8)

    def _aggregate_by_dst(self, messages: torch.Tensor, dst_nodes: torch.Tensor,
                          num_dst: int) -> torch.Tensor:
        out = torch.zeros(num_dst, messages.size(1), messages.size(2),
                         device=messages.device, dtype=messages.dtype)
        idx = dst_nodes.unsqueeze(1).unsqueeze(2).expand(-1, messages.size(1), messages.size(2))
        out.scatter_add_(0, idx, messages)
        return out


class HGT(nn.Module):
    """多层 HGT 网络"""

    def __init__(self, in_dims: Dict[str, int], hidden_dim: int, out_dim: int,
                 num_layers: int, num_heads: int,
                 node_types: List[str], edge_types: List[Tuple[str, str, str]],
                 dropout: float = 0.2):
        super().__init__()
        self.node_types = node_types

        self.input_projs = nn.ModuleDict({
            ntype: nn.Linear(in_dims.get(ntype, 128), hidden_dim)
            for ntype in node_types
        })

        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, hidden_dim, num_heads, node_types, edge_types, dropout)
            for _ in range(num_layers)
        ])

        self.output_projs = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim) for ntype in node_types
        })

    def forward(self, node_features: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = {}
        for ntype, feat in node_features.items():
            h[ntype] = F.relu(self.input_projs[ntype](feat))

        for layer in self.layers:
            h_new = layer(h, edge_index_dict)
            h = {k: h_new.get(k, h[k]) for k in h}

        out = {}
        for ntype, feat in h.items():
            out[ntype] = self.output_projs[ntype](feat)

        return out


class HGSampling:
    """
    异构 Mini-Batch 图采样

    核心思想: 为每种节点类型单独维护采样预算,
    保证子图中各类型节点数量平衡, 避免度数高的类型主导。
    """

    def __init__(self, num_nodes_per_type: Dict[str, int], num_hops: int = 2):
        self.num_nodes = num_nodes_per_type
        self.num_hops = num_hops

    def sample(self, seed_nodes: Dict[str, List[int]],
               edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
               node_counts: Dict[str, int]) -> Dict[str, Set[int]]:
        sampled = {ntype: set(nodes) for ntype, nodes in seed_nodes.items()}

        for _ in range(self.num_hops):
            new_nodes = defaultdict(set)

            for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
                if dst_type not in sampled:
                    continue

                dst_list = list(sampled[dst_type])
                dst_mask = torch.zeros(node_counts.get(dst_type, 0), dtype=torch.bool)
                for n in dst_list:
                    if n < len(dst_mask):
                        dst_mask[n] = True

                edges_to_dst = dst_mask[edge_index[1]]
                src_candidates = edge_index[0][edges_to_dst].tolist()

                budget = self.num_nodes.get(src_type, 10)
                if len(src_candidates) > budget:
                    src_candidates = random.sample(src_candidates, budget)

                new_nodes[src_type].update(src_candidates)

            for ntype, nodes in new_nodes.items():
                sampled.setdefault(ntype, set()).update(nodes)

        return sampled


def build_maternal_baby_graph() -> Tuple[Dict, Dict]:
    """
    构建母婴电商异构图 (纯字典格式)

    返回: (graph_data, metadata)
    graph_data: {
        "node_features": {node_type: [N, feat_dim]},
        "edge_index": {(src, rel, dst): [2, E]},
        "labels": {node_type: [N]},
        "masks": {node_type: {"train": [N], "val": [N], "test": [N]}}
    }
    """
    num_users, num_products = 100, 50
    num_reviews, num_attrs = 200, 30
    feat_dim = 128

    node_features = {
        "user": torch.randn(num_users, feat_dim),
        "product": torch.randn(num_products, feat_dim),
        "review": torch.randn(num_reviews, feat_dim),
        "attribute": torch.randn(num_attrs, feat_dim),
    }

    labels = {
        "product": torch.randint(0, 5, (num_products,))
    }

    torch.manual_seed(42)

    edge_index = {
        ("user", "purchased", "product"): torch.stack([
            torch.randint(0, num_users, (150,)),
            torch.randint(0, num_products, (150,))
        ]),
        ("user", "wrote", "review"): torch.stack([
            torch.randint(0, num_users, (200,)),
            torch.arange(num_reviews)
        ]),
        ("product", "has_review", "review"): torch.stack([
            torch.randint(0, num_products, (200,)),
            torch.arange(num_reviews)
        ]),
        ("product", "has_attribute", "attribute"): torch.stack([
            torch.randint(0, num_products, (80,)),
            torch.randint(0, num_attrs, (80,))
        ]),
        ("review", "mentions", "attribute"): torch.stack([
            torch.randint(0, num_reviews, (120,)),
            torch.randint(0, num_attrs, (120,))
        ]),
    }

    num_train = int(num_products * 0.6)
    num_val = int(num_products * 0.2)
    perm = torch.randperm(num_products)

    masks = {
        "product": {
            "train": torch.zeros(num_products, dtype=torch.bool),
            "val": torch.zeros(num_products, dtype=torch.bool),
            "test": torch.zeros(num_products, dtype=torch.bool),
        }
    }
    masks["product"]["train"][perm[:num_train]] = True
    masks["product"]["val"][perm[num_train:num_train + num_val]] = True
    masks["product"]["test"][perm[num_train + num_val:]] = True

    graph_data = {
        "node_features": node_features,
        "edge_index": edge_index,
        "labels": labels,
        "masks": masks,
    }

    metadata = {
        "node_types": ["user", "product", "review", "attribute"],
        "edge_types": list(edge_index.keys()),
        "num_nodes": {"user": num_users, "product": num_products,
                     "review": num_reviews, "attribute": num_attrs},
    }

    return graph_data, metadata


class ProductCategoryClassifier(nn.Module):
    """基于 HGT 的产品品类分类器"""

    def __init__(self, in_dims: Dict[str, int], hidden_dim: int, num_classes: int,
                 num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        node_types = ["user", "product", "review", "attribute"]
        edge_types = [
            ("user", "purchased", "product"),
            ("user", "wrote", "review"),
            ("product", "has_review", "review"),
            ("product", "has_attribute", "attribute"),
            ("review", "mentions", "attribute"),
        ]

        self.hgt = HGT(in_dims, hidden_dim, hidden_dim, num_layers, num_heads,
                       node_types, edge_types)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_features, edge_index_dict):
        h = self.hgt(node_features, edge_index_dict)
        return self.classifier(h["product"])


def train_hgt_classifier():
    """训练 HGT 产品分类器的完整流程"""
    print("=" * 70)
    print("HGT 异构图神经网络 — 母婴电商产品品类分类")
    print("=" * 70)

    print("\n[1] 构建母婴电商异构图...")
    graph, metadata = build_maternal_baby_graph()
    print(f"   节点: user={metadata['num_nodes']['user']}, "
          f"product={metadata['num_nodes']['product']}, "
          f"review={metadata['num_nodes']['review']}, "
          f"attribute={metadata['num_nodes']['attribute']}")
    print(f"   边类型: {len(metadata['edge_types'])}")

    print("\n[2] 初始化 HGT 分类器...")
    in_dims = {ntype: graph["node_features"][ntype].size(1)
               for ntype in metadata["node_types"]}
    model = ProductCategoryClassifier(in_dims, hidden_dim=128, num_classes=5,
                                      num_layers=2, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n[3] 训练模型 (100 epochs)...")
    best_val_acc = 0.0

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        logits = model(graph["node_features"], graph["edge_index"])
        train_mask = graph["masks"]["product"]["train"]
        loss = criterion(logits[train_mask], graph["labels"]["product"][train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph["node_features"], graph["edge_index"])
                pred = logits.argmax(dim=1)

                train_acc = (pred[train_mask] == graph["labels"]["product"][train_mask]).float().mean()
                val_mask = graph["masks"]["product"]["val"]
                val_acc = (pred[val_mask] == graph["labels"]["product"][val_mask]).float().mean()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc.item()

                print(f"   Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("\n[4] 测试集评估...")
    model.eval()
    with torch.no_grad():
        logits = model(graph["node_features"], graph["edge_index"])
        pred = logits.argmax(dim=1)
        test_mask = graph["masks"]["product"]["test"]
        test_acc = (pred[test_mask] == graph["labels"]["product"][test_mask]).float().mean()
        print(f"   测试准确率: {test_acc:.4f}")
        print(f"   最佳验证准确率: {best_val_acc:.4f}")

    print("\n[5] 异构表示分析...")
    with torch.no_grad():
        h = model.hgt(graph["node_features"], graph["edge_index"])
        for ntype, emb in h.items():
            print(f"   {ntype:12s} embedding shape: {tuple(emb.shape)}")

    print("\n" + "=" * 70)
    print("HGT 训练完成!")
    print("=" * 70)

    return model, graph


def demonstrate_cross_lingual_alignment():
    """演示跨语言商品属性对齐场景"""
    print("\n" + "=" * 70)
    print("场景演示: 跨语言商品属性对齐")
    print("=" * 70)

    print("""
    图结构:
        [product: 吸奶器] --has_attribute--> [attr_en: "portable"]
                                          --> [attr_zh: "便携"]
                                          --> [attr_en: "quiet"]
                                          --> [attr_zh: "静音"]

    HGT 机制:
        1. 英文属性 "portable" 和中文属性 "便携" 通过同一 product 节点聚合
        2. meta relation (attribute, has_attribute, product) 学习属性-产品交互模式
        3. 不同语言的属性节点在 product 表示空间中自然对齐

    业务价值:
        - 无需显式对齐词典, 图结构自动编码跨语言语义
        - 新语言属性可直接接入, 零样本扩展
    """)


def demonstrate_cold_start():
    """演示新品冷启动场景"""
    print("\n" + "=" * 70)
    print("场景演示: 新品冷启动表示推断")
    print("=" * 70)

    print("""
    新品 "Spectra S2" (无评论、无购买记录)
        |
    通过互补商品边连接到已有丰富数据的 "Spectra S1"
        |
    HGT 消息传递: S2 的表示 = f(S1的表示, 互补关系权重, 品牌共享)
        |
    推断出 S2 的品类、目标用户群、属性特征

    数学基础: 归纳式表示学习 (inductive representation learning)
     unseen 节点可通过邻居聚合获得有效表示, 不依赖训练时见过该节点。
    """)


if __name__ == "__main__":
    model, graph = train_hgt_classifier()
    demonstrate_cross_lingual_alignment()
    demonstrate_cold_start()

    print("\n" + "=" * 70)
    print("所有演示完成。生产环境建议:")
    print("  1. 使用真实 embedding (BERT/Word2Vec) 替代随机初始化")
    print("  2. 接入 pyHGT 官方实现处理 Web 规模图数据")
    print("  3. 使用 HGSampling 进行大规模 mini-batch 训练")
    print("  4. 结合 RTE 处理时序购买行为")
    print("=" * 70)
