"""
HGCN (Hyperbolic Graph Convolutional Networks) — 层级结构保持
基于论文: Chami et al. "Hyperbolic Graph Convolutional Neural Networks", NeurIPS 2019

核心能力:
1. 双曲空间嵌入 — 在 Poincare/Lorentz 模型中学习节点表示
2. 层级结构编码 — 天然适合树状/层次化图结构（品类树、组织架构）
3. 指数级容量 — 双曲空间比欧氏空间更能表达层次距离

母婴电商场景: 产品品类层次树、品牌-品类层次结构的表示学习
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict


class PoincareBall:
    """Poincare 球模型 — 双曲几何的实现"""

    def __init__(self, c=1.0):
        self.c = c

    def expmap0(self, v):
        """从切空间映射到双曲空间"""
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=1e-8)
        return torch.tanh(math.sqrt(self.c) * norm_v) * v / (math.sqrt(self.c) * norm_v)

    def logmap0(self, x):
        """从双曲空间映射到切空间"""
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        norm_x = torch.clamp(norm_x, min=1e-8)
        return torch.atanh(math.sqrt(self.c) * norm_x) * x / (math.sqrt(self.c) * norm_x)

    def mobius_add(self, x, y):
        """Möbius 加法"""
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        den = 1 + 2 * self.c * xy + self.c * x2 * y2
        return num / den

    def dist(self, x, y):
        """双曲距离"""
        mobius_diff = self.mobius_add(-x, y)
        norm = torch.norm(mobius_diff, dim=-1)
        return 2 / math.sqrt(self.c) * torch.atanh(math.sqrt(self.c) * norm.clamp(max=1 - 1e-7))


class HGCNLayer(nn.Module):
    """单层双曲图卷积"""

    def __init__(self, in_dim, out_dim, c=1.0, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c = c
        self.manifold = PoincareBall(c)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """x: [N, in_dim], adj: [N, N]"""
        h = self.linear(x)
        support = torch.mm(adj, h)
        output = support / (adj.sum(dim=1, keepdim=True) + 1e-8)
        output = self.dropout(output)
        output = self.manifold.expmap0(output)
        return output


class HGCN(nn.Module):
    """多层 HGCN 网络"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, c=1.0, dropout=0.2):
        super().__init__()
        self.manifold = PoincareBall(c)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HGCNLayer(hidden_dim, hidden_dim, c, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, adj)
            h = self.manifold.logmap0(h)
        output = self.output_proj(h)
        return output


def build_category_tree():
    """构建母婴电商产品品类层次树"""
    num_nodes = 11
    adj = torch.eye(num_nodes)
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8),
        (3, 9), (3, 10),
    ]
    for parent, child in edges:
        adj[parent, child] = 1
        adj[child, parent] = 1

    features = torch.eye(num_nodes)
    labels = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])

    metadata = {
        "node_names": ["母婴用品", "喂养用品", "洗护用品", "出行用品",
                      "吸奶器", "储奶袋", "温奶器",
                      "婴儿沐浴露", "婴儿洗发水",
                      "婴儿推车", "安全座椅"],
        "levels": {0: "根", 1: "一级品类", 2: "叶子"},
    }

    return features, adj, labels, metadata


class CategoryHierarchyClassifier(nn.Module):
    """基于 HGCN 的品类层次分类器"""

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.hgcn = HGCN(in_dim, hidden_dim, hidden_dim, num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        h = self.hgcn(x, adj)
        return self.classifier(h)


def demo_hgcn_category_tree():
    """演示 HGCN 在品类层次树上的应用"""
    print("=" * 70)
    print("HGCN — 双曲图卷积网络 (品类层次结构)")
    print("=" * 70)

    print("\n[1] 构建母婴电商品类层次树...")
    features, adj, labels, metadata = build_category_tree()
    print(f"   节点数: {features.size(0)}")
    print(f"   边数: {int((adj.sum() - adj.size(0)) / 2)}")
    for name, level in zip(metadata["node_names"], labels.tolist()):
        indent = "  " * level
        print(f"   {indent}{name} ({metadata['levels'][level]})")

    print("\n[2] 初始化 HGCN 分类器...")
    model = CategoryHierarchyClassifier(
        in_dim=features.size(1),
        hidden_dim=16,
        num_classes=3,
        num_layers=2
    )

    print("\n[3] 训练模型 (100 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
            print(f"   Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    print("\n[4] 双曲空间层次距离分析...")
    model.eval()
    with torch.no_grad():
        h = model.hgcn(features, adj)
        h_hyperbolic = PoincareBall(c=1.0).expmap0(h)
        manifold = PoincareBall(c=1.0)
        root = h_hyperbolic[0].unsqueeze(0)
        feeding = h_hyperbolic[1].unsqueeze(0)
        breast_pump = h_hyperbolic[4].unsqueeze(0)

        d_root_feeding = manifold.dist(root, feeding).item()
        d_root_pump = manifold.dist(root, breast_pump).item()
        d_feeding_pump = manifold.dist(feeding, breast_pump).item()

        print(f"   根 -> 一级品类距离: {d_root_feeding:.4f}")
        print(f"   根 -> 叶子距离: {d_root_pump:.4f}")
        print(f"   一级 -> 叶子距离: {d_feeding_pump:.4f}")
        print(f"   验证: 根->叶子 > 根->一级 {'OK' if d_root_pump > d_root_feeding else '需更多训练'}")

    print("\n[5] 跨品类相似度推理...")
    with torch.no_grad():
        pump = h_hyperbolic[4].unsqueeze(0)
        bag = h_hyperbolic[5].unsqueeze(0)
        stroller = h_hyperbolic[9].unsqueeze(0)
        d_pump_bag = manifold.dist(pump, bag).item()
        d_pump_stroller = manifold.dist(pump, stroller).item()
        print(f"   吸奶器 <-> 储奶袋 (同品类): {d_pump_bag:.4f}")
        print(f"   吸奶器 <-> 推车 (跨品类): {d_pump_stroller:.4f}")
        print(f"   同品类更近 {'OK' if d_pump_bag < d_pump_stroller else '需更多训练'}")

    print("\n" + "=" * 70)
    print("HGCN 演示完成!")
    print("=" * 70)


def demonstrate_hierarchical_benefit():
    """演示双曲空间对层次结构的优势"""
    print("\n" + "=" * 70)
    print("双曲空间 vs 欧氏空间 — 层次结构表示能力对比")
    print("=" * 70)

    print("""
    问题: 在欧氏空间中嵌入树状结构时，叶子节点数量呈指数增长，
          但空间体积只有多项式增长，导致无法保持树的度量。

    数学事实:
      - n 维欧氏空间: n 维球体积 ~ r^n (多项式)
      - n 维双曲空间: n 维球体积 ~ e^r (指数)

    结论:
      双曲空间天然适合表示树状/层次结构，
      因为体积指数增长可以容纳指数增长的叶子节点。

    母婴电商应用:
      品类树: 母婴 > 喂养 > 吸奶器 > [品牌A, 品牌B, 品牌C, ...]
      双曲嵌入后: 同类品牌距离近，跨品类距离远，层次距离可解释
    """)


if __name__ == "__main__":
    demo_hgcn_category_tree()
    demonstrate_hierarchical_benefit()

    print("\n生产环境建议:")
    print("  1. 使用官方 HGCN 实现 (github.com/HazyResearch/hgcn)")
    print("  2. 在真实产品品类树上验证层次距离的可解释性")
    print("  3. 结合 HGT 构建异构层次图 (品类树 + 产品图)")
    print("  4. 使用 Lorentz 模型替代 Poincare 模型（数值稳定性更好）")
