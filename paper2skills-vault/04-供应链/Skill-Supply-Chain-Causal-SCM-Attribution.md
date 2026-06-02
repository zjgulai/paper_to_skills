---
title: Supply Chain Causal SCM — 数据驱动供应链根因归因：DAG + DoWhy GCM
doc_type: knowledge
module: 04-供应链
topic: supply-chain-causal-scm-attribution
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: Supply Chain Causal SCM — 数据驱动供应链根因归因

> **领域**: 04-供应链 | **类型**: 综合萃取 | **来源**: Amazon Science 2025，Pole, Rakshit et al.

---

## ① 算法原理

**为什么传统归因不够**：瀑布式逻辑（"缺货→往上查库存→往上查采购"）本质上是相关性分析，无法区分"A 导致 B"与"C 同时导致 A 和 B"。实际供应链中，多因素常常通过间接路径（中介变量）影响结果——比如"需求预测误差→战术产能调整→Capped Out Hours（COH）"，传统归因会错误地把间接效应归给直接可见的变量。

**SCM 4 步框架**：
1. **因果发现（Causal Discovery）**：PC 算法 / LiNGAM 从观测数据自动学习变量间的因果方向，输出有向无环图（DAG）
2. **DAG 验证（Falsification）**：基于 d-separation 检验条件独立性，剔除虚假边；领域专家审查拓扑合理性
3. **SCM 构建（Structural Equations）**：为每条 DAG 边拟合结构方程 $X_j = f_j(\text{pa}(X_j), \varepsilon_j)$，捕捉量化因果强度
4. **因果查询（Causal Queries）**：三类查询
   - **根因归因**：DoWhy GCM 的 `attribute_anomalies()` 分解异常样本到各上游变量的贡献度
   - **干预分析**（do-calculus）：$P(Y | \text{do}(X=x))$，估计强制改变 X 后 Y 的变化
   - **反事实推理**：在已知结果前提下，假设某变量不同时的结果是多少

**根因归因算法（DoWhy GCM）**：对每个异常时间点，沿 DAG 拓扑序反向传播，计算每个节点的"异常得分"——若去除该节点的贡献后异常消失，则该节点是根因。输出各变量对异常的归因比例（可加性分解）。

**反事实量化**：利用 Pearl 的三步骤（Abduction→Action→Prediction）：先从观测数据推断外生噪声，再干预目标变量，最后用SCM前向传播预测反事实结果。

---

## ② 母婴出海应用案例

**场景一：母婴奶粉备货短缺根因归因**

- **业务问题**：母婴奶粉 SKU 出现缺货（库存降至 0），损失 GMV 约 8 万元/周。管理层需要快速定位：是需求预测误差（预测比实际低 40%）、交货期延误（供应商迟发货 3 天），还是安全库存设置过低？三个因素同时出现，传统瀑布分析无法量化各自责任；
- **数据要求**：30 天历史数据，每日变量：需求预测值、实际需求、采购触发量、供应商交货期、实际到货量、库存水位；共 6 个节点
- **SCM 做法**：PC 算法学习 DAG（需求预测误差→补货量→库存水位，交货期延误→到货量→库存水位，安全库存参数→补货触发点→补货量），拟合结构方程后对缺货时间点执行根因归因
- **预期产出**：输出各因素归因比例，如"需求预测误差贡献 55%，交货期延误贡献 35%，安全库存参数贡献 10%"
- **业务价值**：将缺货根因定位时间从 2-3 天（人工排查）缩短到 **2-3 小时**，针对性整改避免重复缺货

**场景二：WF-A 补货决策干预分析（"如果预测精度提升 10% 会怎样？"）**

- **业务问题**：WF-A 仓库补货计划师希望量化：如果把需求预测 MAPE 从 25% 降低到 15%，补货量应该如何调整？库存降低多少？这是一个反事实/干预问题，不是回归问题
- **数据要求**：历史 60 天补货决策链路数据（预测误差、补货量、库存日均水位、缺货率），需已有拟合好的 SCM
- **SCM 做法**：执行干预分析 `do(预测误差 = 原误差 × 0.6)` （模拟精度提升40%），通过 SCM 前向传播估计补货量和库存水位的变化；再用反事实推理量化已发生时间段的"假设场景"
- **预期产出**：预测精度提升 10% → 补货量减少约 8%（减少超订），库存日均水位降低 12%，缺货率从 3.2% 降至 1.8%；给出采购优化的优先级排序
- **业务价值**：量化预测精度投资回报，为数据团队预测模型优化提供 ROI 依据，避免"拍脑袋优化"

---

## ③ 代码模板

```python
"""
Supply Chain Causal SCM Attribution
Amazon Science 2025 | Pole, Rakshit et al.
端到端因果归因：DAG → SCM → 根因/干预/反事实
纯标准库 + numpy/scipy 实现（不依赖 DoWhy，原理实现）
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class SCMVariable:
    """供应链 SCM 变量节点"""
    name: str
    description: str
    var_type: str   # "exogenous"（外生）或 "endogenous"（内生）
    values: Optional[np.ndarray] = None  # 历史观测值


@dataclass
class AnomalyResult:
    """根因归因结果"""
    total_anomaly: float
    attributions: Dict[str, float]   # 各节点归因值
    attribution_ratio: Dict[str, float]  # 各节点归因比例
    root_causes: List[str]           # 主要根因（归因比例 > 20%）

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
    有向无环图（DAG）：供应链因果结构

    支持：
    - 手动构建（领域专家知识）
    - 简单数据驱动验证（条件独立性检验）
    """

    def __init__(self):
        self._edges: List[Tuple[str, str]] = []
        self._nodes: set = set()

    def add_edge(self, from_node: str, to_node: str) -> None:
        if from_node == to_node:
            raise ValueError("Self-loop not allowed in DAG")
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
        return [src for src, dst in self._edges if dst == node]

    def children(self, node: str) -> List[str]:
        return [dst for src, dst in self._edges if src == node]

    def topological_sort(self) -> List[str]:
        """Kahn 算法拓扑排序"""
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for _, dst in self._edges:
            in_degree[dst] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []
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
        基于条件独立性的 DAG 伪造检验（Falsification）

        对每条边 A→B，检验控制 A 的父节点后，A 与 B 的偏相关是否显著。
        若不显著，说明该边可能是虚假的。

        Returns:
            dict: {边描述: True（检验通过）/ False（可能虚假）}
        """
        results = {}
        for src, dst in self._edges:
            if src not in data or dst not in data:
                continue
            x = data[src]
            y = data[dst]
            parents = self.parents(src)
            if parents and all(p in data for p in parents):
                controls = np.column_stack([data[p] for p in parents])
                x_res = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
                y_res = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
                r, p_val = stats.pearsonr(x_res, y_res)
            else:
                r, p_val = stats.pearsonr(x, y)
            edge_key = f"{src}→{dst}"
            results[edge_key] = p_val < alpha
        return results


# ─── 结构因果模型 ──────────────────────────────────────────────────────────────

class StructuralCausalModel:
    """
    结构因果模型（SCM）

    为每个内生变量拟合结构方程：
      X_j = f_j(parents(X_j), ε_j)
    其中 f_j 为线性回归（可扩展为非线性）。

    支持：
    1. fit(data)                      - 拟合结构方程
    2. root_cause_attribution(...)    - 根因归因
    3. intervention_analysis(...)     - 干预分析
    4. counterfactual(...)            - 反事实推理
    """

    def __init__(self, dag: CausalDAG):
        self.dag = dag
        self._coefs: Dict[str, np.ndarray] = {}   # 结构方程系数
        self._residuals: Dict[str, np.ndarray] = {}  # 残差（外生噪声）
        self._data: Dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, data: Dict[str, np.ndarray]) -> "StructuralCausalModel":
        """
        拟合结构方程

        Args:
            data: {变量名: 时间序列数组} 字典

        Returns:
            self（支持链式调用）
        """
        self._data = {k: np.array(v) for k, v in data.items()}
        topo = self.dag.topological_sort()

        for node in topo:
            parents = self.dag.parents(node)
            if not parents:
                self._coefs[node] = np.array([])
                self._residuals[node] = self._data[node] - np.mean(self._data[node])
            else:
                X = np.column_stack([self._data[p] for p in parents])
                X_aug = np.column_stack([np.ones(len(X)), X])
                y = self._data[node]
                coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                self._coefs[node] = coef
                self._residuals[node] = y - X_aug @ coef

        self._fitted = True
        return self

    def _predict_node(
        self,
        node: str,
        data_override: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """给定数据（含干预覆盖），预测节点值"""
        parents = self.dag.parents(node)
        if not parents or len(self._coefs[node]) == 0:
            return data_override.get(node, self._data[node])
        X = np.column_stack([data_override[p] for p in parents])
        X_aug = np.column_stack([np.ones(len(X)), X])
        return X_aug @ self._coefs[node]

    def root_cause_attribution(
        self,
        anomalous_samples: Dict[str, np.ndarray],
        target_node: str,
    ) -> AnomalyResult:
        """
        根因归因：分解异常样本中各节点对目标节点的贡献

        方法：逐节点替换（one-at-a-time ablation）
        将每个节点替换为正常基线值，观察目标节点异常减少量

        Args:
            anomalous_samples: 异常时间段数据
            target_node: 目标变量（如库存水位）

        Returns:
            AnomalyResult 含各节点归因比例
        """
        assert self._fitted, "请先调用 fit()"

        baseline = self._data  # 正常基线
        topo = self.dag.topological_sort()

        def forward_pass(
            overrides: Dict[str, np.ndarray]
        ) -> np.ndarray:
            result = {}
            for node in topo:
                if node in overrides:
                    result[node] = overrides[node]
                else:
                    result[node] = self._predict_node(node, result)
            return result[target_node]

        y_anomalous = forward_pass(anomalous_samples)
        y_baseline = forward_pass(baseline)
        total_anomaly = float(np.mean(y_anomalous) - np.mean(y_baseline))

        attributions: Dict[str, float] = {}
        nodes = [n for n in topo if n != target_node]

        for node in nodes:
            override = dict(anomalous_samples)
            override[node] = baseline[node][:len(anomalous_samples[target_node])]
            y_counterfactual = forward_pass(override)
            attr = float(np.mean(y_anomalous) - np.mean(y_counterfactual))
            attributions[node] = attr

        total_attr = sum(abs(v) for v in attributions.values()) or 1.0
        attribution_ratio = {
            k: abs(v) / total_attr for k, v in attributions.items()
        }
        root_causes = [
            k for k, r in attribution_ratio.items() if r >= 0.20
        ]

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
    ) -> float:
        """
        干预分析（do-calculus）：P(target | do(do_variable = value))

        强制将 do_variable 设为 value（切断其父节点影响），
        沿 DAG 前向传播，估计对下游变量的因果效应

        Args:
            do_variable: 干预变量名
            value: 干预后的值（标量，广播到所有样本）
            n_samples: 样本数，默认使用训练数据长度

        Returns:
            dict: {下游节点名: 干预后均值变化}
        """
        assert self._fitted, "请先调用 fit()"
        n = n_samples or len(next(iter(self._data.values())))
        topo = self.dag.topological_sort()

        intervened_data = {k: v[:n].copy() for k, v in self._data.items()}
        intervened_data[do_variable] = np.full(n, value)

        result_do: Dict[str, np.ndarray] = {}
        for node in topo:
            if node == do_variable:
                result_do[node] = intervened_data[do_variable]
            else:
                parents = self.dag.parents(node)
                if not parents or len(self._coefs.get(node, [])) == 0:
                    result_do[node] = intervened_data[node]
                else:
                    X = np.column_stack([result_do[p] for p in parents])
                    X_aug = np.column_stack([np.ones(len(X)), X])
                    result_do[node] = X_aug @ self._coefs[node]

        baseline_mean = {k: float(np.mean(v[:n])) for k, v in self._data.items()}
        effect = {
            k: float(np.mean(result_do[k])) - baseline_mean[k]
            for k in topo
            if k != do_variable
        }
        return effect

    def counterfactual(
        self,
        observed: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        反事实推理（Pearl 三步骤）：
        给定已观测到的事实（observed），如果 interventions 中的变量不同，结果如何？

        1. Abduction：从 observed 推断外生噪声 ε
        2. Action：在 SCM 中施加干预
        3. Prediction：前向传播计算反事实结果

        Args:
            observed: {变量: 已观测值}
            interventions: {变量: 反事实干预值}

        Returns:
            dict: {变量: 反事实结果}
        """
        assert self._fitted, "请先调用 fit()"
        topo = self.dag.topological_sort()

        noise: Dict[str, float] = {}
        for node in topo:
            if node in observed:
                parents = self.dag.parents(node)
                if not parents or len(self._coefs.get(node, [])) == 0:
                    noise[node] = observed[node] - float(np.mean(self._data[node]))
                else:
                    pa_vals = np.array([[observed.get(p, float(np.mean(self._data[p])))
                                        for p in parents]])
                    X_aug = np.column_stack([np.ones(1), pa_vals])
                    predicted = float((X_aug @ self._coefs[node])[0])
                    noise[node] = observed[node] - predicted
            else:
                noise[node] = 0.0

        cf: Dict[str, float] = {}
        for node in topo:
            if node in interventions:
                cf[node] = interventions[node]
            else:
                parents = self.dag.parents(node)
                if not parents or len(self._coefs.get(node, [])) == 0:
                    cf[node] = float(np.mean(self._data[node])) + noise[node]
                else:
                    pa_vals = np.array([[cf.get(p, float(np.mean(self._data[p])))
                                        for p in parents]])
                    X_aug = np.column_stack([np.ones(1), pa_vals])
                    cf[node] = float((X_aug @ self._coefs[node])[0]) + noise[node]

        return cf


# ─── 供应链场景构建器 ──────────────────────────────────────────────────────────

def build_baby_formula_scm() -> Tuple[CausalDAG, StructuralCausalModel]:
    """
    母婴奶粉缺货归因 SCM

    因果结构：
      预测误差 → 补货量 → 库存水位
      交货期    → 到货量 → 库存水位
      安全库存  → 补货量
    """
    dag = CausalDAG()
    dag.add_edge("预测误差", "补货量")
    dag.add_edge("安全库存参数", "补货量")
    dag.add_edge("交货期延误天数", "到货量")
    dag.add_edge("补货量", "库存水位")
    dag.add_edge("到货量", "库存水位")

    scm = StructuralCausalModel(dag)
    return dag, scm


def simulate_supply_chain_data(n: int = 60, seed: int = 42) -> Dict[str, np.ndarray]:
    """生成母婴奶粉供应链模拟数据（60天）"""
    rng = np.random.default_rng(seed)

    forecast_error = rng.normal(0, 0.15, n)      # 预测误差（%）
    safety_stock = rng.normal(50, 5, n)           # 安全库存设置（件）
    lead_time_delay = rng.exponential(0.5, n)     # 交货期延误（天）

    reorder_qty = (
        100 + 200 * forecast_error + 0.8 * safety_stock
        + rng.normal(0, 5, n)
    )
    arrival_qty = (
        95 - 8 * lead_time_delay
        + rng.normal(0, 3, n)
    )
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

    topo = dag.topological_sort()
    print(f"\nDAG 拓扑序: {' → '.join(topo)}")
    print(f"DAG 边: {dag.edges}")

    normal_data = simulate_supply_chain_data(n=60, seed=42)
    scm.fit(normal_data)
    print("\n[SCM 已拟合]")
    for node, coef in scm._coefs.items():
        if len(coef) > 0:
            parents = dag.parents(node)
            print(f"  {node} = {coef[0]:.2f} + "
                  + " + ".join(f"{c:.3f}*{p}" for c, p in zip(coef[1:], parents)))

    anomalous_data = simulate_supply_chain_data(n=14, seed=100)
    anomalous_data["预测误差"] = anomalous_data["预测误差"] + 0.4  # 预测严重低估
    anomalous_data["交货期延误天数"] = anomalous_data["交货期延误天数"] + 2.0  # 供应商延迟

    print("\n[根因归因 - 库存水位异常]")
    result = scm.root_cause_attribution(anomalous_data, "库存水位")
    print(result)

    print("\n[干预分析 - 如果预测误差降至 0]")
    effects = scm.intervention_analysis("预测误差", 0.0)
    for node, effect in effects.items():
        print(f"  {node}: {effect:+.2f}")

    print("\n[反事实推理 - 如果安全库存提高 20%]")
    observed_snapshot = {k: float(np.mean(v)) for k, v in anomalous_data.items()}
    cf_result = scm.counterfactual(
        observed=observed_snapshot,
        interventions={"安全库存参数": observed_snapshot["安全库存参数"] * 1.2},
    )
    print(f"  实际库存水位: {observed_snapshot['库存水位']:.2f}")
    print(f"  反事实库存水位（安全库存+20%）: {cf_result['库存水位']:.2f}")
    print(f"  差异: {cf_result['库存水位'] - observed_snapshot['库存水位']:+.2f}")

    falsification = dag.falsification_test(normal_data)
    print("\n[DAG Falsification 检验]")
    for edge, passed in falsification.items():
        status = "✓ 通过" if passed else "✗ 可疑"
        print(f"  {edge}: {status}")

    assert len(result.root_causes) >= 1, "应识别至少 1 个根因"
    assert "库存水位" not in result.root_causes, "目标节点不应成为自身根因"
    print("\n[✓] 所有断言通过")


if __name__ == "__main__":
    run_test()
```

---

## ④ 技能关联

- **前置**：[[Skill-Causal-Discovery-PC-Algorithm]] / [[Skill-DML-Cohort-Causal-Effect]] / [[Skill-Demand-Forecasting-Supply-Chain]]
- **延伸**：[[Skill-AgentTrace-Causal-RCA]] / [[Skill-Causal-Time-Series-Forecasting-GCF]]
- **可组合**：[[Skill-Flowr-Supply-Chain-MAS]] / [[Skill-AIM-RM-LLM-Inventory-MAS-Memory]] / [[Skill-Inventory-Health-Aging-Attribution]]

---


- **跨域关联**：[[Skill-CausalRAG-Causal-Graph-Retrieval]] / [[Skill-Helicase-Supply-Chain-KG-MAS]]
- **跨域关联**：[[Skill-EventCast-LLM-Event-Forecasting]]
- **关联**：[[Skill-Multimodal-Table-Understanding]]
- **关联**：[[Skill-CSDM-Diffusion-ColdStart]]
- **关联**：[[Skill-AnchorCrafter-Virtual-Anchor-Demo]]
- **关联**：[[Skill-DS-DGA-GCN-Fake-Review-Group]]
- **关联**：[[Skill-AI-Brand-Storytelling]]

## ⑤ 商业价值

- **ROI**：供应链异常定位时间从 2-3 天→2-3 小时，量化干预效果避免无效行动；母婴旺季缺货损失每次约 5-10 万元 GMV，快速归因可减少 50%+ 的重复缺货事件
- **实施难度**：⭐⭐⭐⭐☆（需要因果发现 + SCM 建模知识；但供应链 DAG 结构相对清晰，可由领域专家直接构建）
- **优先级**：⭐⭐⭐⭐⭐（缺货/积压归因是供应链最高频的运营痛点，所有 SKU 类目均适用）
- **评估依据**：Amazon Science 案例验证 SCM 比传统瀑布分析提供更深入可行洞察；母婴品类季节性强、供应链链路短，SCM 节点数少（5-8个），实施复杂度可控
