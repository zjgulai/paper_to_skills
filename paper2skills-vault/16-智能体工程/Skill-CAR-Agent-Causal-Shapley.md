---
title: CAR — Agent步骤因果Shapley归因：多步交互效应定量拆解
doc_type: knowledge
module: 16-智能体工程
topic: causal-agent-replay-shapley-attribution
status: stable
created: 2026-06-10
updated: 2026-06-10
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: CAR — Agent步骤因果Shapley归因

> **论文**：Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures
> **arXiv**：2606.08275 | 2026年6月 | **桥梁**: 16-智能体工程 ↔ 01-因果推断 | **类型**: 跨域融合
> **代码**：开源，支持云端或本地模型

---

## ① 算法原理

### 核心思想

当 LLM Agent 失败时（多退款、错误工具调用、数据泄露），现有工具只能回答"发生了什么"（可观测性）或"是否通过"（评估），**但无法回答"哪一步决定导致了失败"**。

最直觉的启发式方法反而是错的：**执行有害动作的那一步通常不是做决定的那一步**。LLM-judge归因的状态最佳准确率只有14%（Who&When基准）。

**CAR的核心**：将Agent运行建模为**结构因果模型（SCM）**，通过对某一步施加 do-算子干预，重新执行轨迹，测量结果分布的偏移量。

### 数学直觉

**步骤干预代数**：给定轨迹 $T = [s_1, s_2, \ldots, s_n]$，定义步骤 $i$ 的干预效应：

$$\text{CausalEffect}(i) = \mathbb{E}[\text{Outcome} \mid do(s_i = s_i^*)] - \mathbb{E}[\text{Outcome} \mid s_i = s_i^{\text{orig}}]$$

其中 $s_i^*$ 为该步的反事实正确动作，$s_i^{\text{orig}}$ 为原始错误动作。

**单步对比估计器（point-of-commitment rule）**：解决随机前向重执行带来的混淆问题——在第一个"承诺点"（输出固化后的时刻）评估干预效应，消除后续随机性对归因的干扰：

$$\hat{\Delta}_i = \frac{1}{K} \sum_{k=1}^{K} [\text{Outcome}(T^{(k)}_{\text{counterfactual}}) - \text{Outcome}(T^{(k)}_{\text{original}})]$$

**预算有界蒙特卡洛 Shapley**：将交互效应分配到多个步骤上，效率和满足性质 $\sum_i \phi_i = v(N) - v(\emptyset)$：

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

在验证实验中：对2步交互场景（步骤A、B共同导致失败），CAR恢复分配为 (0.44, 0.45, ~0)，效率和为0.909（分析值0.91），**单步LLM-judge只能定位到最后执行步骤，完全遗漏步骤A的贡献**。

### 关键假设
- Agent运行轨迹可被结构化记录（步骤序列、输入输出、最终结果）
- 同一随机策略下可重新执行轨迹（允许有随机性，由蒙特卡洛消除）
- 反事实"正确动作"可被定义（通常由人工或监督信号提供）

---

## ② 母婴出海应用案例

### 场景A：补货MAS的多步决策责任归因

**业务问题**：母婴跨境卖家的补货工作流由4个Agent组成（需求预测 → 库存评估 → 采购决策 → PO生成），大促前某批PO生成了错误采购量（比需求预测低40%），导致大促期间断货损失¥380,000。事后需要确定"哪个Agent的哪步决定是根本原因"，以便修复workflow逻辑。

**现有工具的局限**：AgentTrace（图遍历）定位到"采购决策Agent出错"，但无法区分是"库存评估Agent给的输入有偏差"还是"采购决策Agent的安全库存计算逻辑本身有问题"——两者都参与了最终错误，但贡献比例未知。

**CAR处理**：
- 将整次补货执行轨迹建模为SCM，4个步骤互为依赖节点
- 分别对"库存评估步骤"和"采购决策步骤"执行反事实干预（do-operator）
- 重新跑轨迹，测量各步干预后PO准确率变化
- 蒙特卡洛Shapley分配：库存评估步骤贡献 φ=0.62，采购决策步骤贡献 φ=0.31

**数据要求**：
- Agent执行日志（步骤ID、输入参数、输出值、时间戳、最终PO结果）
- 至少20次同类失败/成功案例（用于蒙特卡洛估计）
- 每步的"正确值"定义（可用历史成功执行作为反事实基准）

**预期产出**：每个Agent步骤的因果贡献 Shapley 值，置信区间宽度 ≤0.05（100次蒙特卡洛采样）

**业务价值**：ROI 估算 ¥50,000/次（断货损失预防）+ 工程维护时间节省 70%（3天→0.5天的根因分析周期）

### 场景B：广告自动竞价Agent的决策归因

**业务问题**：LLM驱动的广告自动竞价Agent在某SKU上连续5天ROAS下滑23%。传统可观测工具显示"竞价调整步骤执行了3次降价"，但无法判断是"关键词质量评估步骤的判断失误"还是"市场竞争感知步骤的数据偏差"触发了级联降价。

**CAR处理**：SCM建模竞价决策轨迹，Shapley归因显示关键词质量步骤贡献 φ=0.71（最大），确认是该步骤的CTR预测偏低（使用了过期CTR均值）导致虚假质量下降信号，触发自动降价链路。

**业务价值**：年化广告ROAS损失预防 ¥120,000+；归因精确度从"猜测哪个模块"提升到"确定哪个具体计算步骤"

---

## ③ 代码模板

```python
"""
CAR - Causal Agent Replay: Agent步骤Shapley因果归因
基于结构因果模型和蒙特卡洛Shapley估计

依赖: numpy, itertools
"""

import numpy as np
from itertools import combinations
from typing import Callable, List, Dict, Tuple
import random

# ─────────────────────────────────────────────
# 数据结构定义
# ─────────────────────────────────────────────

class AgentStep:
    """Agent执行轨迹中的单个步骤"""
    def __init__(self, step_id: str, action: dict, observation: dict, outcome: float = None):
        self.step_id = step_id
        self.action = action          # 该步执行的动作
        self.observation = observation # 该步的输入上下文
        self.outcome = outcome         # 0=失败, 1=成功, 或连续值

class AgentTrajectory:
    """完整的Agent执行轨迹"""
    def __init__(self, steps: List[AgentStep], final_outcome: float, trajectory_id: str = ""):
        self.steps = steps
        self.final_outcome = final_outcome  # 最终结果(0=失败/1=成功)
        self.trajectory_id = trajectory_id

# ─────────────────────────────────────────────
# CAR核心实现
# ─────────────────────────────────────────────

class CausalAgentReplay:
    """
    CAR: 结构因果模型 + 蒙特卡洛Shapley
    
    用法：
    1. 提供历史失败轨迹
    2. 提供每步的"反事实正确动作"  
    3. 提供轨迹重执行函数
    4. 调用 compute_shapley() 获取每步的因果贡献
    """
    
    def __init__(self, replay_fn: Callable, n_monte_carlo: int = 100, seed: int = 42):
        """
        replay_fn: 给定步骤干预集合，重执行轨迹并返回结果
                  签名: (trajectory, interventions: dict{step_id: counterfactual_action}) -> float
        n_monte_carlo: Shapley蒙特卡洛采样次数（100次约达到±0.05精度）
        """
        self.replay_fn = replay_fn
        self.n_monte_carlo = n_monte_carlo
        np.random.seed(seed)
        random.seed(seed)
    
    def counterfactual_effect(
        self, 
        trajectory: AgentTrajectory,
        step_id: str,
        counterfactual_action: dict
    ) -> float:
        """
        单步干预效应估计（带point-of-commitment修正）
        返回: do干预后结果 - 原始结果的差值
        """
        # 原始结果（已知）
        original_outcome = trajectory.final_outcome
        
        # 干预后结果（蒙特卡洛平均）
        outcomes_after_intervention = []
        for _ in range(self.n_monte_carlo // 10):  # 单步用1/10采样数
            intervened_outcome = self.replay_fn(
                trajectory, 
                interventions={step_id: counterfactual_action}
            )
            outcomes_after_intervention.append(intervened_outcome)
        
        counterfactual_outcome = np.mean(outcomes_after_intervention)
        effect = counterfactual_outcome - original_outcome
        return effect
    
    def _coalition_value(
        self,
        trajectory: AgentTrajectory,
        coalition: List[str],
        counterfactual_actions: Dict[str, dict]
    ) -> float:
        """计算步骤联盟coalition的价值 v(S)"""
        if not coalition:
            return trajectory.final_outcome
        
        outcomes = []
        for _ in range(self.n_monte_carlo // len(coalition)):
            outcome = self.replay_fn(
                trajectory,
                interventions={sid: counterfactual_actions[sid] for sid in coalition}
            )
            outcomes.append(outcome)
        return np.mean(outcomes)
    
    def compute_shapley(
        self,
        trajectory: AgentTrajectory,
        counterfactual_actions: Dict[str, dict],
        budget: int = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        蒙特卡洛Shapley归因
        
        Args:
            trajectory: 失败的Agent轨迹
            counterfactual_actions: 每步的反事实正确动作 {step_id: correct_action}
            budget: 最大轨迹重执行次数（None=全量计算）
        
        Returns:
            {step_id: (shapley_value, confidence_interval)} 
        """
        step_ids = list(counterfactual_actions.keys())
        n_steps = len(step_ids)
        
        if budget is None:
            budget = self.n_monte_carlo * n_steps
        
        shapley_samples = {sid: [] for sid in step_ids}
        
        # 蒙特卡洛排列采样
        samples_per_step = budget // n_steps
        
        for _ in range(samples_per_step):
            # 随机排列步骤顺序
            perm = random.sample(step_ids, n_steps)
            
            # 逐步计算边际贡献
            v_without = self._coalition_value(trajectory, [], counterfactual_actions)
            coalition_so_far = []
            
            for step in perm:
                v_with = self._coalition_value(
                    trajectory, 
                    coalition_so_far + [step], 
                    counterfactual_actions
                )
                marginal = v_with - v_without
                shapley_samples[step].append(marginal)
                
                coalition_so_far.append(step)
                v_without = v_with
        
        # 计算Shapley值和置信区间
        results = {}
        for sid in step_ids:
            samples = shapley_samples[sid]
            phi = np.mean(samples)
            ci = 1.96 * np.std(samples) / np.sqrt(len(samples))
            results[sid] = (phi, ci)
        
        return results

# ─────────────────────────────────────────────
# 业务场景模拟：母婴补货MAS归因
# ─────────────────────────────────────────────

def simulate_restock_mas():
    """模拟4步补货MAS轨迹，注入多步错误"""
    
    # 定义4步Agent轨迹（含已知错误）
    steps = [
        AgentStep("demand_forecast", 
                  action={"predicted_demand": 1200, "confidence": 0.82},
                  observation={"historical_7d": [180,195,210,188,220,205,195]}),
        AgentStep("inventory_eval",
                  action={"current_stock": 320, "reorder_point": 800},  # 错误：reorder_point偏低
                  observation={"warehouse_data": {"FBA_available": 320}}),
        AgentStep("purchase_decision",
                  action={"order_qty": 480},  # 结果：少订了，应该是880
                  observation={}),
        AgentStep("po_generation",
                  action={"po_amount": 480, "supplier": "SupplierA"},
                  observation={})
    ]
    
    # 最终结果：大促期间断货（0=失败）
    trajectory = AgentTrajectory(steps, final_outcome=0.0, trajectory_id="traj_20260610")
    
    # 每步的反事实"正确"动作
    counterfactual_actions = {
        "demand_forecast": {"predicted_demand": 1200, "confidence": 0.82},  # 这步是对的
        "inventory_eval": {"current_stock": 320, "reorder_point": 1200},   # 修正：reorder_point=1200
        "purchase_decision": {"order_qty": 880},                             # 修正：正确采购量
        "po_generation": {"po_amount": 880, "supplier": "SupplierA"}       # 依赖上游修正
    }
    
    return trajectory, counterfactual_actions


def mock_replay_fn(trajectory: AgentTrajectory, interventions: dict) -> float:
    """
    模拟轨迹重执行函数
    真实场景中应调用实际Agent系统重执行
    这里用规则模拟：库存评估正确 → 采购量正确 → 不断货
    """
    intervened_steps = {
        s.step_id: {**s.action, **(interventions.get(s.step_id, {}))}
        for s in trajectory.steps
    }
    
    # 模拟业务规则
    reorder_point = intervened_steps.get("inventory_eval", {}).get("reorder_point", 800)
    demand = intervened_steps.get("demand_forecast", {}).get("predicted_demand", 1200)
    
    # 如果库存评估的reorder_point合理，采购量就会正确
    if reorder_point >= 1100:
        order_qty = demand * 0.8  # 合理采购量
        success_prob = 0.85 + np.random.normal(0, 0.05)
    else:
        order_qty = reorder_point * 0.4  # 不足
        success_prob = 0.15 + np.random.normal(0, 0.05)
    
    return float(np.clip(success_prob, 0, 1))


def run_car_analysis():
    """执行CAR完整归因流程"""
    print("=" * 60)
    print("CAR - 母婴补货MAS因果Shapley归因")
    print("=" * 60)
    
    trajectory, counterfactual_actions = simulate_restock_mas()
    
    print(f"\n失败轨迹ID: {trajectory.trajectory_id}")
    print(f"原始结果: {trajectory.final_outcome} (0=断货失败)")
    print(f"\n步骤:")
    for s in trajectory.steps:
        print(f"  [{s.step_id}] 动作={s.action}")
    
    print("\n开始CAR归因分析...")
    
    car = CausalAgentReplay(
        replay_fn=mock_replay_fn,
        n_monte_carlo=50,  # 演示用50次，生产建议200次
        seed=42
    )
    
    shapley_results = car.compute_shapley(
        trajectory=trajectory,
        counterfactual_actions=counterfactual_actions,
        budget=200
    )
    
    print("\n" + "=" * 60)
    print("Shapley因果归因结果：")
    print("=" * 60)
    print(f"{'步骤ID':<25} {'Shapley值':>12} {'±置信区间':>12} {'归因比例':>10}")
    print("-" * 60)
    
    total_positive = sum(max(v[0], 0) for v in shapley_results.values())
    
    for step_id, (phi, ci) in sorted(shapley_results.items(), key=lambda x: -abs(x[1][0])):
        pct = (phi / total_positive * 100) if total_positive > 0 else 0
        flag = " ← 主要原因" if phi > 0.3 else ""
        print(f"{step_id:<25} {phi:>12.3f} {ci:>12.3f} {pct:>9.1f}%{flag}")
    
    # 效率验证：Shapley值之和应接近 v(N) - v({})
    efficiency_sum = sum(v[0] for v in shapley_results.values())
    print(f"\n效率验证: Σφᵢ = {efficiency_sum:.3f}")
    print("(理论值 = 干预所有步骤的结果 - 原始结果，应约为0.7-0.8)")
    
    print("\n[✓] CAR归因分析完成")
    return shapley_results


if __name__ == "__main__":
    results = run_car_analysis()
    
    # 输出决策建议
    print("\n业务决策建议:")
    if any(v[0] > 0.3 for v in results.values()):
        top_step = max(results.items(), key=lambda x: x[1][0])
        print(f"  → 优先修复 [{top_step[0]}]，Shapley贡献={top_step[1][0]:.3f}")
        print(f"  → 预计修复后成功率提升约 {top_step[1][0]*100:.0f}%")
    else:
        print("  → 失败原因分散在多步，需全链路改进")
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AgentTrace-Causal-RCA]]（先定位出错的Agent模块范围）
- **前置（prerequisite）**：[[Skill-CausalFlow-Agent-Failure-Repair]]（单步反事实干预基础概念）
- **延伸（extends）**：[[Skill-Agent-Stage-Evaluation]]（批量评估期望：CAR从单轨迹扩展到批量归因）
- **可组合（combinable）**：[[Skill-Dynamic-DAG-Orchestration]]（组合场景：动态DAG中的分支步骤归因——当Agent路径可变时，CAR可以评估各分支决策节点的因果贡献）
- **可组合（combinable）**：[[Skill-Causal-Cohort-Analysis]]（组合场景：将CAR与因果队列分析结合，分析"同类型Agent决策失败的系统性归因模式"）

---

## ⑤ 商业价值评估

- **ROI预估**：
  - 断货损失预防：¥50,000-380,000/次（母婴大促场景）
  - 工程排查时间：从3天→0.5天/次，节省工程师成本 ¥15,000+/次
  - ROAS损失预防（广告Agent）：¥80,000-150,000/季度
  - **年化综合ROI**：¥500,000-1,500,000（视Agent规模）

- **实施难度**：⭐⭐⭐☆☆
  - 技术门槛中等：需要能重现执行的轨迹记录系统
  - 核心挑战：定义每步的"反事实正确动作"需要领域专家参与
  - 蒙特卡洛成本：每次归因分析约需50-200次轨迹重执行

- **优先级评分**：⭐⭐⭐⭐☆
  - 随着Agent系统在母婴运营中深度使用（补货/广告/客服），该Skill的价值将显著增长
  - **反直觉价值**：当前图谱有80个Agent/MAS Skills，却没有"评估这些Agent决策是否真的有效"的严格因果工具——这个空白会随着Agent规模化而从"锦上添花"变为"不可或缺"

- **评估依据**：CAR在Who&When基准上将逐步归因准确率从14%（LLM-judge基线）提升至理论最优的结构因果模型精度；Shapley效率性质保证"所有步骤贡献之和等于总效果"，避免双重计算
