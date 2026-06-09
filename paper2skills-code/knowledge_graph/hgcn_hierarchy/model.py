"""
HGCN Hyperbolic Graph Convolutional Networks — 双曲层次嵌入
paper2skills-code: 08-知识图谱 | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field


@dataclass
class HierarchyNode:
    node_id: str
    node_type: str     # brand / category / product / attribute
    level: int         # 层级深度（0=根）
    parent_id: str = ""


@dataclass
class HyperbolicEmbedding:
    node_id: str
    poincare_coords: list[float]   # 庞加莱圆盘坐标（范数 < 1）
    norm: float = 0.0

    def __post_init__(self):
        self.norm = math.sqrt(sum(x**2 for x in self.poincare_coords))


def poincare_distance(u: list[float], v: list[float]) -> float:
    """庞加莱圆盘距离：层次越深的节点离原点越远"""
    norm_u = math.sqrt(sum(x**2 for x in u))
    norm_v = math.sqrt(sum(x**2 for x in v))
    diff_sq = sum((ui - vi)**2 for ui, vi in zip(u, v))
    denom = (1 - norm_u**2) * (1 - norm_v**2)
    if denom <= 0:
        return float("inf")
    return math.acosh(1 + 2 * diff_sq / denom)


def hierarchy_to_poincare(nodes: list[HierarchyNode],
                           dim: int = 2, seed: int = 42) -> dict[str, HyperbolicEmbedding]:
    """将层次结构映射到庞加莱圆盘（层级越深，范数越大）"""
    random.seed(seed)
    max_level = max(n.level for n in nodes) if nodes else 1
    embeddings = {}
    for node in nodes:
        radius = (node.level / max(max_level, 1)) * 0.9  # 归一化到 [0, 0.9]
        angle = random.uniform(0, 2 * math.pi)
        coords = [radius * math.cos(angle), radius * math.sin(angle)]
        if dim > 2:
            coords += [0.0] * (dim - 2)
        embeddings[node.node_id] = HyperbolicEmbedding(
            node_id=node.node_id,
            poincare_coords=coords,
        )
    return embeddings


class HGCNLayer:
    """双曲图卷积层（简化版：指数映射 + 切空间线性变换）"""

    def __init__(self, in_dim: int, out_dim: int, curvature: float = 1.0):
        self.curvature = curvature
        random.seed(42)
        self.W = [[random.gauss(0, 0.1) for _ in range(out_dim)] for _ in range(in_dim)]

    def mobius_add(self, x: list[float], y: list[float]) -> list[float]:
        """莫比乌斯加法（庞加莱圆盘加法）"""
        c = self.curvature
        x2 = sum(xi**2 for xi in x)
        y2 = sum(yi**2 for yi in y)
        xy = sum(xi*yi for xi, yi in zip(x, y))
        num = [(1 + 2*c*xy + c*y2)*xi + (1 - c*x2)*yi for xi, yi in zip(x, y)]
        denom = 1 + 2*c*xy + c**2*x2*y2
        return [n/denom for n in num]

    def forward(self, embeddings: list[list[float]],
                adj_matrix: list[list[float]]) -> list[list[float]]:
        """图卷积：聚合邻居信息"""
        n = len(embeddings)
        out = []
        for i in range(n):
            agg = [0.0] * len(embeddings[0])
            for j in range(n):
                if adj_matrix[i][j] > 0:
                    w = adj_matrix[i][j]
                    agg = [a + w * e for a, e in zip(agg, embeddings[j])]
            agg = [a / max(sum(adj_matrix[i]), 1e-6) for a in agg]
            transformed = [sum(agg[k] * self.W[k][d]
                               for k in range(min(len(agg), len(self.W))))
                           for d in range(len(self.W[0]))]
            norm = math.sqrt(sum(t**2 for t in transformed)) + 1e-9
            projected = [0.9 * t / norm for t in transformed]
            out.append(projected)
        return out


def run_hgcn_demo():
    nodes = [
        HierarchyNode("ROOT", "root", 0),
        HierarchyNode("CAT-BABY", "category", 1, "ROOT"),
        HierarchyNode("CAT-FORMULA", "category", 2, "CAT-BABY"),
        HierarchyNode("BRAND-HOLLE", "brand", 2, "CAT-BABY"),
        HierarchyNode("SKU-S1", "product", 3, "CAT-FORMULA"),
        HierarchyNode("SKU-S2", "product", 3, "CAT-FORMULA"),
        HierarchyNode("ATTR-ORGANIC", "attribute", 3, "CAT-FORMULA"),
    ]

    emb = hierarchy_to_poincare(nodes, dim=2)

    print("=== HGCN 双曲层次嵌入（母婴产品 KG）===")
    print(f"{'节点':<20} {'层级':>4} {'范数':>8} {'距离根节点':>10}")
    root_coords = emb["ROOT"].poincare_coords
    for n in nodes:
        e = emb[n.node_id]
        d = poincare_distance(root_coords, e.poincare_coords)
        print(f"  {n.node_id:<18} {n.level:>4} {e.norm:>8.4f} {d:>10.4f}")

    print("相关产品距离（庞加莱距离，越小越相关）:")
    d12 = poincare_distance(emb["SKU-S1"].poincare_coords, emb["SKU-S2"].poincare_coords)
    d1b = poincare_distance(emb["SKU-S1"].poincare_coords, emb["BRAND-HOLLE"].poincare_coords)
    print(f"  SKU-S1 <-> SKU-S2:    {d12:.4f}（同品类，相关度高）")
    print(f"  SKU-S1 <-> BRAND:     {d1b:.4f}（跨类型，距离较大）")
    print("✅ HGCN 演示完成")
if __name__ == "__main__":
    run_hgcn_demo()
