
"""
PC Algorithm for Causal Discovery
基于论文: Causation, Prediction, and Search (Spirtes et al., 2000)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class PCAlgorithm:
    """PC 算法因果发现"""

    def __init__(self, alpha: float = 0.05, max_cond_set: Optional[int] = None):
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.sepset: Dict[Tuple[str, str], Set[str]] = {}
        self.skeleton: Set[Tuple[str, str]] = set()
        self.directed_edges: Set[Tuple[str, str]] = set()
        self.undirected_edges: Set[Tuple[str, str]] = set()

    def _conditional_independence_test(self, data: pd.DataFrame, x: str, y: str, cond_set: List[str]):
        """条件独立性检验: 返回 (是否独立, p-value)"""
        if len(cond_set) == 0:
            corr, p_value = pearsonr(data[x], data[y])
            return p_value > self.alpha, p_value
        X_cond = data[cond_set].values
        reg_x = LinearRegression().fit(X_cond, data[x])
        reg_y = LinearRegression().fit(X_cond, data[y])
        resid_x = data[x] - reg_x.predict(X_cond)
        resid_y = data[y] - reg_y.predict(X_cond)
        corr, p_value = pearsonr(resid_x, resid_y)
        return p_value > self.alpha, p_value

    def _get_adjacent(self, node: str, skeleton: Set[Tuple[str, str]]) -> Set[str]:
        adj = set()
        for edge in skeleton:
            if node in edge:
                adj.add(edge[0] if edge[1] == node else edge[1])
        return adj

    def learn_skeleton(self, data: pd.DataFrame, var_names: List[str]) -> Set[Tuple[str, str]]:
        n_vars = len(var_names)
        skeleton = set()
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                skeleton.add((var_names[i], var_names[j]))
        max_k = self.max_cond_set or n_vars - 2
        self.sepset = {}
        for k in range(max_k + 1):
            edges_to_remove = []
            for edge in list(skeleton):
                x, y = edge
                adj_x = self._get_adjacent(x, skeleton) - {y}
                if len(adj_x) < k:
                    continue
                separated = False
                for cond_set in itertools.combinations(adj_x, k):
                    is_indep, p_val = self._conditional_independence_test(data, x, y, list(cond_set))
                    if is_indep:
                        edges_to_remove.append(edge)
                        self.sepset[(x, y)] = set(cond_set)
                        self.sepset[(y, x)] = set(cond_set)
                        separated = True
                        break
                if separated:
                    break
            for edge in edges_to_remove:
                if edge in skeleton:
                    skeleton.remove(edge)
            if not edges_to_remove:
                break
        self.skeleton = skeleton
        return skeleton

    def orient_v_structures(self, var_names: List[str]):
        self.directed_edges = set()
        self.undirected_edges = set(self.skeleton)
        for z in var_names:
            adj_z = self._get_adjacent(z, self.skeleton)
            for x, y in itertools.combinations(adj_z, 2):
                if (x, y) in self.skeleton or (y, x) in self.skeleton:
                    continue
                sepset_xy = self.sepset.get((x, y), set())
                if z not in sepset_xy:
                    self.directed_edges.add((x, z))
                    self.directed_edges.add((y, z))
                    for pair in [(x, z), (z, x), (y, z), (z, y)]:
                        if pair in self.undirected_edges:
                            self.undirected_edges.remove(pair)

    def apply_orientation_rules(self, var_names: List[str]):
        changed = True
        while changed:
            changed = False
            for x, y in list(self.directed_edges):
                adj_y = self._get_adjacent(y, self.undirected_edges | self.directed_edges)
                for z in adj_y - {x}:
                    if (y, z) in self.undirected_edges:
                        if (x, z) not in self.undirected_edges and (z, x) not in self.undirected_edges:
                            if (x, z) not in self.directed_edges and (z, x) not in self.directed_edges:
                                self.directed_edges.add((y, z))
                                if (y, z) in self.undirected_edges:
                                    self.undirected_edges.remove((y, z))
                                if (z, y) in self.undirected_edges:
                                    self.undirected_edges.remove((z, y))
                                changed = True
            for x, y in list(self.directed_edges):
                if (y, z) in self.directed_edges:
                    if (x, z) in self.undirected_edges:
                        self.directed_edges.add((x, z))
                        self.undirected_edges.remove((x, z))
                        changed = True

    def fit(self, data: pd.DataFrame) -> Dict:
        var_names = list(data.columns)
        self.learn_skeleton(data, var_names)
        self.orient_v_structures(var_names)
        self.apply_orientation_rules(var_names)
        return {
            "skeleton": self.skeleton,
            "directed_edges": self.directed_edges,
            "undirected_edges": self.undirected_edges,
            "sepset": self.sepset,
        }

    def to_cpdag_str(self) -> str:
        lines = ["CPDAG 因果结构:", "-" * 40]
        for src, dst in sorted(self.directed_edges):
            lines.append(f"  {src} --> {dst}  (定向)")
        for n1, n2 in sorted(self.undirected_edges):
            if (n1, n2) <= (n2, n1):
                lines.append(f"  {n1} --- {n2}  (未定向)")
        return "\n".join(lines)


def generate_mombaby_sales_data(n_weeks: int = 104, seed: int = 42):
    np.random.seed(seed)
    weeks = np.arange(n_weeks)
    season_index = 0.5 + 0.5 * np.sin(2 * np.pi * weeks / 52)
    competitor_price = 80 + 10 * np.sin(2 * np.pi * weeks / 26) + np.random.normal(0, 3, n_weeks)
    kol_collab = np.random.binomial(1, 0.15, n_weeks)
    ad_spend = 5000 + 50 * (100 - competitor_price) + 30 * season_index * 1000 + np.random.normal(0, 500, n_weeks)
    ad_spend = np.maximum(ad_spend, 1000)
    promotion_prob = 1 / (1 + np.exp(-(0.02 * (100 - competitor_price) + 0.5 * season_index - 1)))
    promotion_flag = np.random.binomial(1, promotion_prob)
    weekly_sales = (
        1000
        - 8 * (competitor_price - 80)
        + 0.3 * ad_spend
        + 500 * promotion_flag
        + 800 * kol_collab
        + 200 * season_index
        + np.random.normal(0, 100, n_weeks)
    )
    weekly_sales = np.maximum(weekly_sales, 100)
    df = pd.DataFrame({
        "competitor_price": competitor_price,
        "season_index": season_index,
        "ad_spend": ad_spend,
        "promotion_flag": promotion_flag,
        "kol_collab": kol_collab,
        "weekly_sales": weekly_sales,
    })
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("PC Algorithm: 因果发现 - 母婴电商销量驱动因素")
    print("=" * 60)
    print("\n[1] 生成模拟周级数据...")
    data = generate_mombaby_sales_data(n_weeks=104)
    print(f"    数据维度: {data.shape}")
    print(f"    变量: {list(data.columns)}")
    print("\n    数据预览:")
    print(data.head(3).to_string())
    print("\n[2] 运行 PC 算法 (alpha=0.05)...")
    pc = PCAlgorithm(alpha=0.05)
    result = pc.fit(data)
    print(f"\n    骨架边数: {len(result['skeleton'])}")
    print(f"    定向边数: {len(result['directed_edges'])}")
    print(f"    未定向边数: {len(result['undirected_edges'])}")
    print("\n[3] 因果结构发现结果:")
    print(pc.to_cpdag_str())
    print("\n[4] 已知因果结构验证:")
    known_edges = [
        ("competitor_price", "ad_spend"),
        ("competitor_price", "weekly_sales"),
        ("ad_spend", "weekly_sales"),
        ("promotion_flag", "weekly_sales"),
        ("kol_collab", "weekly_sales"),
        ("season_index", "ad_spend"),
    ]
    for src, dst in known_edges:
        found = (src, dst) in result["directed_edges"] or (dst, src) in result["directed_edges"]
        found |= (src, dst) in result["undirected_edges"] or (dst, src) in result["undirected_edges"]
        status = "发现" if found else "未发现"
        print(f"    {src} -> {dst}: {status}")
    print(f"\n{'=' * 60}")
    print("PC 算法测试完成")
    print(f"{'=' * 60}")
