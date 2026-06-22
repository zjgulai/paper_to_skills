# Skill Card: Multilevel Facility Location Optimization (多级设施选址优化)

> **论文来源**: arXiv: 2406.07382 | Multilevel Facility Location Optimization: A Novel Integer Programming Formulation and Approaches to Heuristic Solutions (2024-06, Last revised 2025-04)

roadmap_phase: phase1
---

## ① 算法原理

### 核心思想
多级设施选址问题（MFL）解决的核心问题是：**在一个从工厂到终端消费者的多层级供应链网络中，决定在哪里建哪类设施、各层级之间如何连通，使全链路固定成本和运输成本之和最小**。它比传统的"仓库-客户"两级模型更接近真实供应链。

本文的两大贡献：
1. **新型整数规划公式**：基于二次分配问题（QAP）变体重构模型，大幅减少决策变量数量，使精确求解器能处理更大规模问题。
2. **VND 启发式算法框架**：面对万级节点规模，设计了基于可变邻域下降（Variable Neighborhood Descent）的启发式算法簇（BVND / PVND / CVND / UVND），结合多起点和禁忌搜索，在合理时间内找到高质量解。

### 数学直觉

**目标函数（最小化总成本）**：

$$\min \sum_{l=1}^{L} \sum_{i \in F_l} f_i x_i + \sum_{l=1}^{L-1} \sum_{i \in F_l} \sum_{j \in F_{l+1}} c_{ij} \cdot v_{ij}$$

其中：
- $f_i$：开设设施 $i$ 的固定成本（一次性沉没成本）
- $x_i \in \{0,1\}$：是否开设设施 $i$
- $c_{ij}$：层 $l$ 设施 $i$ 到层 $l+1$ 设施 $j$ 的单位运输成本
- $v_{ij}$：通过边 $(i,j)$ 流转的物流量

**关键约束**：
- 容量约束：每个设施节点的吞吐量 $\leq$ 其处理能力上限 $U_i$
- 流量守恒：每个市场的需求必须被完全满足
- 开设约束：只有开设的设施才能承担物流分配

### 关键假设
- 网络层级结构固定（本文聚焦四级网络，可扩展）
- 各层级运输成本已知且为线性单价
- 设施容量上限已知
- 需求已知（确定性需求，随机需求为扩展方向）

---

## ② 出海供应链应用案例

### 场景一：出海品牌全球仓网拓扑规划

**业务问题**：
某中国出海品牌（消费电子或母婴用品）正在规划全球供应链网络。现有候选节点：2 个国内工厂、4 个国内/海外总仓候选位置、8 个海外分拨中心候选位置、覆盖 50 个终端市场。问题是：**应该开哪些仓？各仓如何互联？** 每个节点的开设都涉及高额固定租金合规成本，而物流方案又直接影响履约成本。

**数据要求**：
- 候选设施列表：各层级候选位置的固定年化成本（万元/年）和处理能力（件/月）
- 运输成本矩阵：各节点间单位货值的物流报价（万元/万件）
- 市场需求预测：各终端市场月均销量（件/月）

**预期产出**：
- 最优仓网拓扑：哪些工厂/总仓/分拨中心应该开设
- 物流分配方案：各层级之间的货量分配关系
- 全链路总成本：年化固定成本 + 运输成本

**业务价值**：
- 相比直觉决策，VND 优化方案通常可降低全链路成本 10-25%
- 以年物流费用 500 万为例，节省 50-125 万元/年
- 为仓网扩张/收缩决策提供量化依据，避免盲目建仓导致的固定成本陷阱

---

### 场景二：跨境多仓路由策略优化（四级 FLP 场景）

**业务问题**：
品牌在美国已有东岸和西岸两个区域总仓，但业务扩张后需决定：是否在欧洲、东南亚新增区域总仓；如果新增，下面应对应哪些分拨中心；国内哪个工厂/货源点对应供应哪个海外总仓。这是典型的四级网络重构决策。

**数据要求**：
- 现有仓网结构和成本数据
- 候选新增节点的建设/租赁预算（固定成本）
- 各市场的月销量预测和增长趋势
- 跨境物流报价表（货量分段定价转换为线性近似）

**预期产出**：
- 是否值得在欧洲/东南亚开仓的量化结论（开仓 vs 直发的成本对比）
- 推荐开仓方案下的全链路成本分解（固定成本占比 vs 运输成本占比）
- 敏感性分析：当需求增长 20%/50% 时，最优拓扑是否改变

**业务价值**：
- 为 2-5 年战略仓网规划提供可量化的决策依据
- 防止过早过度建仓（固定成本陷阱）或建仓不足（运输成本高企）
- 典型场景：节省的运输成本 > 新建仓固定成本的临界需求量（Break-even 分析）

---

## ③ 代码模板

代码路径：`paper2skills-code/04-供应链/multilevel_flp_2024/model.py`

```python
"""
快速上手示例 - 出海供应链四级 FLP 规划
完整代码见: paper2skills-code/04-供应链/multilevel_flp_2024/model.py
"""
import sys
sys.path.insert(0, "paper2skills-code/04-供应链/multilevel_flp_2024")

from model import (
    FacilityNode, MarketNode, MLFLPInstance,
    generate_random_instance, vnd_solve, print_solution_summary
)

# -------------------------------------------------------
# 1. 构建实例（或使用 generate_random_instance 生成测试数据）
# -------------------------------------------------------
instance = generate_random_instance(
    n_plants=2,       # 国内工厂数量
    n_warehouses=3,   # 候选总仓数量
    n_dcs=5,          # 候选分拨中心数量
    n_markets=20,     # 终端市场数量
    seed=2024,
)

# -------------------------------------------------------
# 2. VND 启发式求解
# -------------------------------------------------------
best_solution, cost_history = vnd_solve(
    instance,
    max_iterations=1000,
    seed=42,
    verbose=True,      # 打印迭代过程
)

# -------------------------------------------------------
# 3. 输出结果
# -------------------------------------------------------
print_solution_summary(best_solution, instance)

# 查看分配决策
print("\n仓网分配决策:")
print(f"  开放总仓: {best_solution.open_warehouses}")
print(f"  开放分拨中心: {best_solution.open_dcs}")
print(f"  DC -> Warehouse 映射: {best_solution.assign_wd}")

# 优化效益
initial_cost = cost_history[0]
final_cost = best_solution.total_cost
saving_pct = (initial_cost - final_cost) / initial_cost * 100
print(f"\n优化效益: {initial_cost:.2f} → {final_cost:.2f} 万元 (节省 {saving_pct:.1f}%)")
print("[✓] Multilevel_FLP 测试通过")
```

**核心数据输入格式**：

| 字段 | 说明 | 单位 |
|------|------|------|
| `fixed_cost` | 开设该设施的年化固定成本 | 万元/年 |
| `capacity` | 该设施的月处理能力上限 | 件/月 |
| `transport_cost[A][B]` | 从节点 A 到节点 B 的单位运输成本 | 万元/万件 |
| `demand` | 终端市场月均需求量 | 件/月 |

---

## ④ 技能关联

**前置技能**：
- [Skill-Multi-Echelon-Inventory](./[[Skill-Multi-Echelon-Inventory]].md)：多阶库存优化，理解库存成本结构和服务水平约束后，再看 FLP 的容量约束会更自然
- [Skill-Safety-Stock-Replenishment](./[[Skill-Safety-Stock-Replenishment]].md)：安全库存与补货策略，提供 FLP 中需求估算和波动建模的基础

**延伸技能**：
- [Skill-Two-Echelon-Inventory-DRL](./[[Skill-Two-Echelon-Inventory-DRL]].md)：网络拓扑确定后，用 DRL 动态调优各节点库存策略
- [Skill-Lead-Time-Distribution-Risk-GenQOT](./[[Skill-Lead-Time-Distribution-Risk-GenQOT]].md)：FLP 求解后，对提前期风险最高的路段做专项优化

**可组合**：
- 与 **Multi-Echelon-Inventory** 组合：FLP 确定"在哪建仓"，多阶库存确定"各仓备多少货"，形成完整的静态网络规划 + 动态库存决策闭环
- 与 **Demand-Forecasting-Supply-Chain** 组合：用需求预测结果替换 FLP 中的确定性需求输入，提升方案的前瞻性
- 与 **Gen-QOT（提前期风险）** 组合：在 FLP 的运输成本中嵌入提前期分布的期望延误惩罚，输出风险调整后的最优拓扑

---
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]
- **相关**：[[Skill-ROAS-Budget-Optimization]]

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 仓网优化减少冗余设施固定成本 + 降低运输成本；以年物流成本 1000 万的中型出海品牌为例，预计节省 100-250 万元/年（10-25%），一次性建模投入约 3-5 人周 |
| **实施难度** | ⭐⭐⭐☆☆（中等）：数据准备是主要难点（运输成本矩阵、精确容量数据），算法本身已封装 |
| **优先级评分** | ⭐⭐⭐⭐☆（高）：对面临仓网扩张决策或存在明显冗余仓的品牌，投入产出比极高 |
| **评估依据** | 仓网规划属于战略级一次性决策，建模质量直接影响未来 3-5 年的固定成本结构；相比运营层面的短期优化，ROI 周期更长但绝对收益更大；适合年物流成本 500 万以上、有多仓或扩张计划的品牌 |

**适用阶段标志**：
- ✓ 正在规划新市场进入（美国/欧洲/东南亚）
- ✓ 现有仓网有 3 个以上节点，怀疑存在冗余
- ✓ 面临仓网从 2 级升 3 级的扩张决策
- ✗ 仓网结构已固定且短期无调整计划（此时应转向动态库存优化）
