"""
Skill-Supply-Chain-Causal-SCM-Attribution
Amazon Science 2025 | Pole, Rakshit et al.
End-to-End Causal Modeling for Supply Chain Attribution

核心框架：
  数据驱动因果发现（DAG）→ SCM 构建 + 严格验证
  → DoWhy GCM 执行根因归因 / 干预分析 / 反事实推理

应用场景：
  - 供应链异常根因归因（缺货 / 积压 / 交货延误）
  - 干预效果量化（"如果预测精度提升 X%，库存降低多少？"）
  - 反事实分析（"如果当时安全库存多 20%，缺货会减少多少？"）
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class SCMVariable:
    """供应链 SCM 变量节点定义"""
    name: str
    description: str
    var_type: str                        # "exogenous" 或 "endogenous"
    values: Optional[np.ndarray] = None  # 历史观测值


@dataclass
class AnomalyResult:
    """根因归因输出"""
    total_anomaly: float
    attributions: Dict[str, float]       # 各节点归因绝对值
    attribution_ratio: Dict[str, float]  # 各节点归因比例
    root_causes: List[str]               # 主要根因（比例 >= 20%）

    def __str__(self) -> str:
        lines = [f"异常总量: {self.total_anomaly:.4f}", "归因分解:"]
        for node, ratio in sorted(
            self.attribution_ratio.items(), key=lambda x: -x[1]
        ):
            mark = " ← 主要根因" if node in self.root_causes else ""
            lines.append(f"  {node}: {ratio:.1%}{mark}")
        return "\n".join(lines)


# ─── 因果 DAG ─────────────────────────────────────────────────────────────────

class CausalDAG:
    """
    有向无环图（DAG）：表示供应链变量间的因果结构

    支持手动构建（领域专家知识）和条件独立性 Falsification 验证。
    """

    def __init__(self):
        self._edges: List[Tuple[str, str]] = []
        self._nodes: set = set()

    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加因果边 from_node → to_node"""
        if from_node == to_node:
            raise ValueError(f"Self-loop not allowed: {from_node}")
        self._edges.append((from_node, to_node))
        self._nodes.add(from_node)
        self._nodes.add(to_node)

    @property
    def nodes(self) -> List[str]:
        return sorted(self._nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self._edges)

    def parents(self, node: str) -> List[str]:
        """返回 node 的所有父节点"""
        return [src for src, dst in self._edges if dst == node]

    def children(self, node: str) -> List[str]:
        """返回 node 的所有子节点"""
        return [dst for src, dst in self._edges if src == node]

    def descendants(self, node: str) -> List[str]:
        """BFS 返回 node 的所有后代节点"""
        visited, queue = set(), [node]
        while queue:
            current = queue.pop(0)
            for child in self.children(current):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return list(visited)

    def topological_sort(self) -> List[str]:
        """Kahn 算法拓扑排序，检测环路"""
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for _, dst in self._edges:
            in_degree[dst] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._nodes):
            raise ValueError("DAG 包含环路（Cycle detected）")
        return order

    def falsification_test(
        self,
        data: Dict[str, np.ndarray],
        alpha: float = 0.05,
    ) -> Dict[str, bool]:
        """
        DAG Falsification 检验（条件独立性）

        对每条边 A→B，控制 A 的父节点后检验 A 与 B 的偏相关显著性。
        不显著（p >= alpha）说明该边可能是虚假的。

        Returns:
            {边描述: True（通过）/ False（可疑）}
        """
        results = {}
        for src, dst in self._edges:
            if src not in data or dst not in data:
                continue
            x, y = data[src], data[dst]
            parents = self.parents(src)
            if parents and all(p in data for p in parents):
                Z = np.column_stack([data[p] for p in parents])
                x_res = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
                y_res = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
                _, p_val = stats.pearsonr(x_res, y_res)
            else:
                _, p_val = stats.pearsonr(x, y)
            results[f"{src}→{dst}"] = bool(p_val < alpha)
        return results


# ─── 结构因果模型 ──────────────────────────────────────────────────────────────

class StructuralCausalModel:
    """
    结构因果模型（SCM）

    结构方程：X_j = f_j(parents(X_j)) + ε_j
    其中 f_j 为线性回归（可扩展为非线性）。
    外生噪声 ε_j 捕捉未建模的随机性。

    三类因果查询：
    1. root_cause_attribution  - 根因归因（异常来自哪里？）
    2. intervention_analysis   - 干预效果（强制改变 X 后 Y 变化？）
    3. counterfactual          - 反事实推理（如果当时不同会怎样？）
    """

    def __init__(self, dag: CausalDAG):
        self.dag = dag
        self._coefs: Dict[str, np.ndarray] = {}
        self._residuals: Dict[str, np.ndarray] = {}
        self._data: Dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, data: Dict[str, np.ndarray]) -> "StructuralCausalModel":
        """
        拟合结构方程（OLS 线性回归）

        Args:
            data: {变量名: 时间序列观测值}

        Returns:
            self（链式调用）
        """
        self._data = {k: np.array(v, dtype=float) for k, v in data.items()}
        topo = self.dag.topological_sort()

        for node in topo:
            parents = self.dag.parents(node)
            if not parents:
                self._coefs[node] = np.array([])
                self._residuals[node] = (
                    self._data[node] - np.mean(self._data[node])
                )
            else:
                X = np.column_stack([self._data[p] for p in parents])
                X_aug = np.column_stack([np.ones(len(X)), X])
                y = self._data[node]
                coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                self._coefs[node] = coef
                self._residuals[node] = y - X_aug @ coef

        self._fitted = True
        return self

    def _forward_pass(
        self,
        overrides: Dict[str, np.ndarray],
        force_recompute_descendants: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        沿拓扑序前向传播，支持节点值覆盖。

        当 force_recompute_descendants=True 时（默认），
        被 override 节点的下游节点也会被重新计算（通过结构方程）。
        这保证根因归因的消融实验正确传播因果效应。
        """
        topo = self.dag.topological_sort()
        result: Dict[str, np.ndarray] = {}
        n = len(next(iter(overrides.values()))) if overrides else len(
            next(iter(self._data.values()))
        )

        for node in topo:
            parents = self.dag.parents(node)
            if node in overrides:
                result[node] = overrides[node]
            elif not parents or len(self._coefs.get(node, [])) == 0:
                result[node] = self._data[node][:n]
            else:
                X = np.column_stack([result[p][:n] for p in parents])
                X_aug = np.column_stack([np.ones(n), X])
                result[node] = X_aug @ self._coefs[node]
        return result

    def root_cause_attribution(
        self,
        anomalous_samples: Dict[str, np.ndarray],
        target_node: str,
    ) -> AnomalyResult:
        """
        根因归因：逐节点消融（ablation），量化各节点对异常的贡献

        方法：
          对每个候选根因节点 c，计算 "将 c 替换为正常基线值后，
          目标节点异常减少量" = attr_c
          归因比例 = |attr_c| / Σ|attr_i|

        Args:
            anomalous_samples: 异常时间段数据（与 _data 同结构）
            target_node: 要归因的目标变量（如"库存水位"）

        Returns:
            AnomalyResult
        """
        assert self._fitted, "请先调用 fit()"

        n = len(next(iter(anomalous_samples.values())))
        topo = self.dag.topological_sort()

        exogenous_nodes = [nd for nd in topo if not self.dag.parents(nd)]

        def forward_from_exo(exo_overrides: Dict[str, np.ndarray]) -> np.ndarray:
            """给定外生节点值，通过 SCM 前向传播计算所有内生节点"""
            result: Dict[str, np.ndarray] = {}
            for nd in topo:
                if nd in exo_overrides:
                    result[nd] = exo_overrides[nd]
                else:
                    parents = self.dag.parents(nd)
                    if not parents or len(self._coefs.get(nd, [])) == 0:
                        result[nd] = self._data[nd][:n]
                    else:
                        X = np.column_stack([result[p][:n] for p in parents])
                        X_aug = np.column_stack([np.ones(n), X])
                        result[nd] = X_aug @ self._coefs[nd]
            return result[target_node]

        anomalous_exo = {nd: anomalous_samples[nd][:n]
                         for nd in exogenous_nodes if nd in anomalous_samples}
        baseline_exo = {nd: self._data[nd][:n] for nd in exogenous_nodes}

        y_anomalous = forward_from_exo(anomalous_exo)
        y_baseline = forward_from_exo(baseline_exo)
        total_anomaly = float(np.mean(y_anomalous) - np.mean(y_baseline))

        attributions: Dict[str, float] = {}
        candidate_nodes = [nd for nd in exogenous_nodes if nd != target_node]

        for node in candidate_nodes:
            cf_exo = dict(anomalous_exo)
            cf_exo[node] = baseline_exo[node]
            y_cf = forward_from_exo(cf_exo)
            attributions[node] = float(np.mean(y_anomalous) - np.mean(y_cf))

        total_abs = sum(abs(v) for v in attributions.values()) or 1.0
        attribution_ratio = {k: abs(v) / total_abs for k, v in attributions.items()}
        root_causes = [k for k, r in attribution_ratio.items() if r >= 0.10]

        return AnomalyResult(
            total_anomaly=total_anomaly,
            attributions=attributions,
            attribution_ratio=attribution_ratio,
            root_causes=root_causes,
        )

    def intervention_analysis(
        self,
        do_variable: str,
        value: float,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        干预分析（do-calculus）：强制 do_variable = value

        切断 do_variable 与其父节点的连接（截断因果路径），
        沿 DAG 前向传播估计下游变量均值变化。

        Args:
            do_variable: 干预变量名
            value: 干预后取值（标量）
            n_samples: 样本数（默认使用训练集长度）

        Returns:
            {下游节点: 均值变化量}
        """
        assert self._fitted, "请先调用 fit()"
        n = n_samples or len(next(iter(self._data.values())))
        topo = self.dag.topological_sort()

        def _propagate(exo_vals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            result: Dict[str, np.ndarray] = {}
            for nd in topo:
                if nd in exo_vals:
                    result[nd] = exo_vals[nd]
                else:
                    parents = self.dag.parents(nd)
                    if not parents or len(self._coefs.get(nd, [])) == 0:
                        result[nd] = self._data[nd][:n]
                    else:
                        X = np.column_stack([result[p][:n] for p in parents])
                        X_aug = np.column_stack([np.ones(n), X])
                        result[nd] = X_aug @ self._coefs[nd]
            return result

        exo_nodes = {nd for nd in topo if not self.dag.parents(nd)}
        baseline_exo = {nd: self._data[nd][:n] for nd in exo_nodes}
        do_exo = dict(baseline_exo)
        if do_variable in exo_nodes:
            do_exo[do_variable] = np.full(n, float(value))

        result_do = _propagate(do_exo)
        baseline_mean = {k: float(np.mean(self._data[k][:n])) for k in topo}

        return {
            k: float(np.mean(result_do[k])) - baseline_mean[k]
            for k in topo
            if k != do_variable
        }

    def counterfactual(
        self,
        observed: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        反事实推理（Pearl 三步骤）

        1. Abduction: 从 observed 推断外生噪声 ε_j
        2. Action: 在 SCM 中施加干预（interventions）
        3. Prediction: 前向传播计算反事实结果

        Args:
            observed: {变量: 实际观测值（单个时间点）}
            interventions: {变量: 反事实假设值}

        Returns:
            {变量: 反事实结果}
        """
        assert self._fitted, "请先调用 fit()"
        topo = self.dag.topological_sort()

        noise: Dict[str, float] = {}
        for node in topo:
            obs_val = observed.get(node)
            if obs_val is None:
                noise[node] = 0.0
                continue
            parents = self.dag.parents(node)
            if not parents or len(self._coefs.get(node, [])) == 0:
                noise[node] = obs_val - float(np.mean(self._data[node]))
            else:
                pa_arr = np.array([[observed.get(p, float(np.mean(self._data[p])))
                                    for p in parents]])
                X_aug = np.column_stack([np.ones(1), pa_arr])
                predicted = float((X_aug @ self._coefs[node])[0])
                noise[node] = obs_val - predicted

        cf: Dict[str, float] = {}
        for node in topo:
            if node in interventions:
                cf[node] = float(interventions[node])
            else:
                parents = self.dag.parents(node)
                if not parents or len(self._coefs.get(node, [])) == 0:
                    cf[node] = float(np.mean(self._data[node])) + noise[node]
                else:
                    pa_arr = np.array([[cf.get(p, float(np.mean(self._data[p])))
                                        for p in parents]])
                    X_aug = np.column_stack([np.ones(1), pa_arr])
                    cf[node] = float((X_aug @ self._coefs[node])[0]) + noise[node]

        return cf


# ─── 场景构建器 ────────────────────────────────────────────────────────────────

def build_baby_formula_scm() -> Tuple[CausalDAG, StructuralCausalModel]:
    """
    母婴奶粉缺货归因 SCM

    因果结构：
      预测误差 ──→ 补货量 ──→ 库存水位
      安全库存参数 ─→ 补货量
      交货期延误天数 → 到货量 → 库存水位
    """
    dag = CausalDAG()
    dag.add_edge("预测误差", "补货量")
    dag.add_edge("安全库存参数", "补货量")
    dag.add_edge("交货期延误天数", "到货量")
    dag.add_edge("补货量", "库存水位")
    dag.add_edge("到货量", "库存水位")
    return dag, StructuralCausalModel(dag)


def simulate_supply_chain_data(
    n: int = 60, seed: int = 42
) -> Dict[str, np.ndarray]:
    """生成母婴奶粉供应链模拟数据"""
    rng = np.random.default_rng(seed)
    forecast_error = rng.normal(0, 0.15, n)
    safety_stock = rng.normal(50, 5, n)
    lead_time_delay = rng.exponential(0.5, n)
    reorder_qty = (
        100 + 200 * forecast_error + 0.8 * safety_stock
        + rng.normal(0, 5, n)
    )
    arrival_qty = 95 - 8 * lead_time_delay + rng.normal(0, 3, n)
    inventory_level = (
        0.6 * reorder_qty + 0.4 * arrival_qty - 60
        + rng.normal(0, 4, n)
    )
    return {
        "预测误差": forecast_error,
        "安全库存参数": safety_stock,
        "交货期延误天数": lead_time_delay,
        "补货量": reorder_qty,
        "到货量": arrival_qty,
        "库存水位": inventory_level,
    }


# ─── 测试 ──────────────────────────────────────────────────────────────────────

def run_test() -> None:
    print("=" * 60)
    print("Supply Chain Causal SCM Attribution 测试")
    print("=" * 60)

    dag, scm = build_baby_formula_scm()
    normal_data = simulate_supply_chain_data(n=60, seed=42)
    scm.fit(normal_data)

    topo = dag.topological_sort()
    print(f"\nDAG 拓扑序: {' → '.join(topo)}")

    anomalous = simulate_supply_chain_data(n=14, seed=100)
    anomalous["预测误差"] = anomalous["预测误差"] + 0.4
    anomalous["交货期延误天数"] = anomalous["交货期延误天数"] + 2.0

    print("\n[根因归因 - 库存水位异常]")
    result = scm.root_cause_attribution(anomalous, "库存水位")
    print(result)

    print("\n[干预分析 - do(预测误差=0)]")
    effects = scm.intervention_analysis("预测误差", 0.0)
    for node, eff in effects.items():
        print(f"  {node}: {eff:+.2f}")

    print("\n[反事实 - 如果安全库存提高 20%]")
    obs = {k: float(np.mean(v)) for k, v in anomalous.items()}
    cf = scm.counterfactual(
        observed=obs,
        interventions={"安全库存参数": obs["安全库存参数"] * 1.2},
    )
    print(f"  实际库存: {obs['库存水位']:.2f}")
    print(f"  反事实库存（安全库存+20%）: {cf['库存水位']:.2f}")

    print("\n[DAG Falsification]")
    falsify = dag.falsification_test(normal_data)
    for edge, passed in falsify.items():
        print(f"  {edge}: {'✓' if passed else '✗ 可疑'}")

    assert len(result.root_causes) >= 1, "应识别 ≥ 1 个根因"
    assert "库存水位" not in result.root_causes
    print("\n[✓] 所有断言通过")


if __name__ == "__main__":
    run_test()
