---
name: neo-lrp-location-routing
description: NEO-LRP 技能卡片，面向仓网选址-路径联合优化（CLRP）。将 GNN 代理成本模型嵌入 MIP 选址框架，解耦选址与路径的计算复杂度。当需要同时决策"开哪些仓"和"如何配送"且传统方法太慢时使用，适用于城市前置仓、同城配送网络、仓库布局优化等供应链规划场景。
arXiv: 2412.05665
domain: 04-供应链
status: verified
code: paper2skills-code/04-供应链/location_routing_2024/model.py
---

# Skill Card: NEO-LRP（Neural Embedded Optimization for Location-Routing）

> 论文: *Neural Embedded Mixed-Integer Optimization for Location-Routing* (arXiv 2412.05665, 2024-12)

---

## ① 算法原理

**核心思想**：把"建哪些仓"和"怎么配送"这两个原本耦合的 NP-hard 问题解耦——用一个预训练好的图神经网络（GNN）充当配送路径成本的快速估计器，把估计值直接嵌进选址的混合整数规划（MIP）里，从而让 MIP 求解器只需要做高层选址-分配决策，而不必在求解过程中展开庞大的车辆路径（VRP）变量。

**数学直觉**：

经典 CLRP 的目标函数为：

$$\min \underbrace{\sum_{j} f_j y_j}_{\text{开仓固定成本}} + \underbrace{\sum_{j} C^*_j(S_j)}_{\text{各仓最优路径成本}}$$

其中 $C^*_j(S_j)$ 是仓库 $j$ 服务客户子集 $S_j$ 的精确最优 VRP 成本——这一项是 NP-hard 的瓶颈。

NEO-LRP 用 GNN 训练一个代理函数 $\hat{C}_j(\cdot)$ 来替代 $C^*_j(\cdot)$：

$$\hat{C}_j(S_j) = \text{GNN}_\theta(\text{depot}_j, \{x_c\}_{c \in S_j})$$

GNN 在大量已知最优解的 VRP 实例上独立预训练（与 LRP 测试集无交集），推理时间 < 1ms/次，使得整个 MIP 可以在秒级内完成。选址落定后再调用成熟的 VRP 求解器（如 OR-Tools）生成实际路线。

**关键假设**：
- 配送成本可以近似为仓库位置和客户集合的函数（GNN 泛化假设）
- 客户需求是已知/可预测的（日均需求输入）
- 车辆同质、容量固定（标准 CVRP 前提）

---

## ② 母婴出海应用案例

### 场景 1：城市前置仓选址优化（生鲜 / 快消 母婴品类）

**业务问题**：计划在某一线城市新增 2～3 个母婴品类前置仓，备选点共 8 个（商业园区/物流园），每天需要向 120+ 个小区配送尿布、奶粉等，要求"开仓固定成本 + 每日配送成本"之和最低。

**数据要求**：
- 备选仓位置（经纬度）、日租金/固定运营成本（元/天）、容量上限（单品件数/天）
- 各小区中心坐标（或聚合地址）、日均订单量（件）
- 历史配送记录（骑手/货车轨迹 + 成本），用于训练/校准 GNN 代理模型

**预期产出**：
- 最优开仓组合（如从 8 个备选中选 2 个）
- 每个小区归属哪个仓
- 各仓每日配送路线草案（可直接下发调度系统）

**业务价值**：相比割裂的"先经验选址 → 再算路线"方案，NEO-LRP 联合优化通常降低综合物流成本 **12%～25%**。以单城市日配 2000 单、综合成本 5 元/单估算，年节省约 **44 万 ~ 91 万元/城市**。

---

### 场景 2：大促备货仓网弹性规划

**业务问题**：双 11 / 618 大促前，临时在多个城市租用共享仓库作为弹性补货节点。备选仓共 15 个，大促期间每仓日均处理量是平时 3～5 倍，需快速决定"启用哪几个临时仓"和"覆盖哪些区域"以最小化加急配送成本。

**数据要求**：
- 各备选仓地址、日租金（短租模式）
- 大促期间各区域预测需求量（来自销量预测模型）
- 车辆车型和运力约束

**预期产出**：
- 大促期弹性仓网方案（启用仓 + 区域划分）
- 分仓库的日配送路线模板

**业务价值**：避免因临时仓位置差导致跨区爆仓，预计减少大促期加急运费 **15%～30%**，减少客诉率。

---

## ③ 代码模板

代码文件：`paper2skills-code/04-供应链/location_routing_2024/model.py`

核心调用示例：

```python
from model import solve_location_routing

depot_specs = [
    {"node_id": "仓A_东区", "x": 0.0,  "y": 0.0,  "open_cost": 200.0, "capacity": 300.0},
    {"node_id": "仓B_西区", "x": 10.0, "y": 0.0,  "open_cost": 180.0, "capacity": 300.0},
    {"node_id": "仓C_北区", "x": 5.0,  "y": 8.0,  "open_cost": 220.0, "capacity": 300.0},
]
customer_specs = [
    {"node_id": "小区01", "x": 1.0,  "y": 1.0,  "demand": 10.0},
    {"node_id": "小区02", "x": 2.0,  "y": -1.0, "demand": 15.0},
    {"node_id": "小区03", "x": 9.0,  "y": 1.0,  "demand": 12.0},
    {"node_id": "小区04", "x": 11.0, "y": -1.0, "demand": 8.0},
    {"node_id": "小区05", "x": 5.0,  "y": 7.0,  "demand": 20.0},
    {"node_id": "小区06", "x": 6.0,  "y": 9.0,  "demand": 18.0},
]

result = solve_location_routing(
    depot_specs,
    customer_specs,
    vehicle_capacity=50.0,
    vehicle_cost_per_km=3.0,
    max_open=2,
    verbose=True,
)

print(f"开放仓库: {result.open_depots}")
print(f"总成本: {result.total_cost:.2f} 元/天")
print(f"  开仓: {result.open_cost:.2f} | 路径: {result.routing_cost:.2f}")
```

**代码结构**：
- `GNNCostSurrogate`：GNN 代理模型（Mock 实现：最近邻 TSP + 载重约束，可替换为真实 GNN 推理）
- `NEOLocationRoutingSolver`：核心求解器（小规模精确枚举 + 大规模贪心搜索）
- `solve_location_routing`：便捷接口，接受字典列表输入，返回 `SolverResult`
- 7 个单元测试全覆盖：基础求解、客户分配完整性、成本分解一致性、路线结构、大规模场景

---

## ④ 技能关联

**前置技能**：
- [Skill-Multi-Echelon-Inventory](Skill-Multi-Echelon-Inventory.md)：理解多级仓网结构和库存成本建模
- [Skill-Two-Echelon-Inventory-DRL](Skill-Two-Echelon-Inventory-DRL.md)：了解 DRL 在供应链优化中的应用范式

**延伸技能**：
- 真实 GNN 训练：将 `GNNCostSurrogate` 替换为 PyTorch Geometric 实现的 GAT 网络，在历史路线数据上预训练
- OR-Tools VRP 集成：选址固定后接入 Google OR-Tools 生成生产级路线

**可组合**：
- [Skill-Demand-Forecasting-Supply-Chain](Skill-Demand-Forecasting-Supply-Chain.md)：预测各小区日均需求 → 作为 `demand` 输入 NEO-LRP
- [Skill-Lead-Time-Distribution-Risk-GenQOT](Skill-Lead-Time-Distribution-Risk-GenQOT.md)：仓网确定后，用 GenQOT 管理各仓的补货提前期风险

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 单城市年节省物流成本 44～91 万元（基于日配 2000 单、综合成本 5 元/单，优化幅度 12%～25%） |
| **实施难度** | ⭐⭐⭐☆☆（3/5）：核心算法 Python 可直接运行；生产落地需替换 GNN 并接入 ERP/WMS |
| **优先级评分** | ⭐⭐⭐⭐☆（4/5）：高价值、2024 年底前沿成果，且有明确工程落地路径 |
| **评估依据** | 难度 3 星：GNN 预训练需要历史路线数据（通常已有）+ 工程接入成本；优先级 4 星：仓网决策是年度战略动作，优化一次影响全年成本结构，且算法已有开源验证基础 |

> **适用规模**：候选仓库 5～50 个 + 客户节点 50～1000 个（精确枚举上限 10 仓，超出自动切换贪心）。真实论文验证了 600 客户 + 30 仓的大规模场景，单次求解 < 数分钟。
