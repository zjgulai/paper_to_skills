# Skill Card: PPO-swap（图上设施选址强化学习）

roadmap_phase: phase1
---

## ① 算法原理

### 核心思想
PPO-swap 解决的是**在真实道路网络（加权图）上，如何快速决定把哪个仓库/站点搬去哪里，使全局配送成本最低**。传统 Gurobi 在大图上算不动（千节点场景需数小时），贪心启发式又容易陷入局部最优。PPO-swap 以"从初始布局出发、反复微调"取代"从零开始构建"，每一步只做一次**交换（Swap）**：关掉一个现有设施，在另一个节点重开，直到整体成本无法再降。

### 数学直觉

**P-median 目标**（成本越小越好）：

$$
\min \sum_{i \in V} d_i \cdot \min_{f \in F} w(i, f)
$$

其中 $d_i$ 是节点 $i$ 的客户需求，$w(i,f)$ 是客户 $i$ 到最近设施 $f$ 的道路距离，$F$ 是当前设施集合（$|F|=P$）。

**Swap 动作空间**（解耦设计，规模从 $O(P \times V)$ 降到 $O(P + V)$）：

$$
a = (\underbrace{\text{remove\_idx}}_{\text{移除哪个}},\ \underbrace{\text{add\_node}}_{\text{加在哪里}})
$$

**PPO 裁剪目标**（防止策略突变）：

$$
L^{\text{CLIP}} = \mathbb{E}_t \left[ \min\!\left(r_t A_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t\right) \right]
$$

其中 $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ 是新旧策略概率比，$A_t$ 是优势估计，$\varepsilon=0.2$。

**GNN 节点编码**（均值聚合，2 层）：

$$
h_i^{(l+1)} = \text{ReLU}\!\left(W \cdot \left[h_i^{(l)} \| \text{Mean}_{j \in \mathcal{N}(i)} h_j^{(l)}\right]\right)
$$

### 关键假设
1. **图结构稳定**：路网拓扑短期内不变（适合月级规划，不适合实时秒级路况）
2. **需求可量化**：每个节点有可测量的客户需求权重（如历史订单热力图）
3. **设施数量固定**：已知需要开多少个仓/站（P 值已知），解决的是"选哪里"而非"开多少个"
4. **泛化前提**：在小图（≤500 节点）训练后可零样本迁移到大图（≥1000 节点），但图的生成分布需要相似

---

## ② 母婴出海应用案例

### 场景1：快递站点季度微调（动态搬迁）

**业务问题**

某母婴出海公司在上海有 8 个自营快递服务站，每季度随城市开发会有 2-3 个站点的位置变得不再最优（如所在小区拆迁、周边大型社区新建）。重新规划 8 个站的全局布局，人工需要数周，算法（Gurobi）需要数小时，且无法实时响应季度需求热力图更新。

**数据要求**

| 字段 | 说明 | 来源 |
|------|------|------|
| `node_id` | 候选站点/小区节点编号 | 地图 API |
| `coord_x/y` | 节点坐标（WGS84） | 地图 API |
| `demand` | 过去 30 天订单量（权重） | OMS |
| `road_dist[i][j]` | 节点间道路距离矩阵 | 高德/百度路网 |
| `current_facility` | 当前站点节点 ID 列表 | 运营台账 |

**预期产出**
- 毫秒级输出：**哪个站搬到哪个位置**，附带搬迁后全局配送距离下降幅度
- 批量评估：对季度 10 个候选站点调整方案自动排优先级
- 基准对比报告：vs 现状、vs 随机重选

**业务价值**
- 每次季度微调可降低配送加权距离 10-15%，对应末端配送成本约节省 8-12%
- 决策时间从 2 周压缩到 1 天（算法跑完后人工审核），同等规模下比 Gurobi 快 **2000 倍以上**

### 场景2：应急物资库极速选址（动态重规划）

**业务问题**

母婴品类有应急备货需求（如奶粉断货风险期、台风路径下的海外仓调拨）。当某个港口/仓库突然不可用时，需要在剩余路网上快速重新计算 3-5 个临时周转仓位置，以最小化紧急调配的平均配送距离。Gurobi 等精确求解器在这种"道路部分中断 + 需求瞬间迁移"场景下根本来不及出结果。

**数据要求**
- 受损路网快照（剔除中断路段后的邻接矩阵）
- 紧急需求分布（各区域缺货量估算）
- 候选周转仓地址列表（租赁市场可用仓点）

**预期产出**
- 5 秒内给出临时仓选址方案（基于当前受损路网）
- 每 15 分钟可根据路况更新重新计算，支持动态调整
- 方案优先级排序（兼顾成本 + 仓库可用性约束）

**业务价值**
- 应急响应时间从"小时级"压缩到"分钟级"
- 临时仓选址质量（P-median 成本）比随机选址优化约 15-25%
- 支持高频重规划（每 15 分钟一次），传统方案无法实现

---

## ③ 代码模板

代码位置：`paper2skills-code/04-供应链/facility_location_rl_2025/model.py`

```python
"""
快速使用示例：PPO-swap 快递站点搬迁
"""
import sys
sys.path.insert(0, "paper2skills-code/04-供应链/facility_location_rl_2025")
from model import FacilityLocationGraph, FacilityLocationEnv, MockPPOSwapAgent

# ① 构建城市路网图（30 个候选位置，选 5 个站点）
graph = FacilityLocationGraph(n_nodes=30, n_facilities=5, seed=42)

# ② 初始化环境（物理启发式初始布局）
env = FacilityLocationEnv(graph, max_steps=30)
facilities, initial_cost = env.reset(init_method="physics")
print(f"初始成本: {initial_cost:.2f}，初始站点: {facilities}")

# ③ 使用 Mock Agent 运行优化（生产环境换成训练好的 PPOSwapTrainer）
agent = MockPPOSwapAgent(graph, n_candidates=10)

for step in range(30):
    remove_idx, add_node = agent.select_action(facilities)
    reward, done, info = env.step(remove_idx, add_node)
    facilities = env.facilities.copy()
    if info.get("valid") and info.get("delta_cost", 0) > 0:
        print(f"Step {step+1}: 将站点 {facilities[remove_idx]} → {add_node}，"
              f"成本下降 {info['delta_cost']:.2f}")
    if done:
        break

final_cost = graph.shortest_path_cost(facilities)
print(f"\n优化完成，成本从 {initial_cost:.2f} → {final_cost:.2f}，"
      f"下降 {(initial_cost - final_cost) / initial_cost:.1%}")
print(f"最优站点配置: {facilities}")
```

**核心 API 速查**

| 类/函数 | 作用 |
|---------|------|
| `FacilityLocationGraph(n_nodes, n_facilities)` | 构建城市路网图，支持自定义需求权重 |
| `FacilityLocationEnv.reset(init_method="physics")` | 物理启发式初始化，返回初始设施列表和成本 |
| `FacilityLocationEnv.step(remove_idx, add_node)` | 执行一次 Swap，返回奖励和成本变化 |
| `MockPPOSwapAgent.select_action(facilities)` | 贪心 Swap 选择（测试用） |
| `PPOSwapTrainer` | 完整 PPO-swap 训练器（生产用）|
| `FacilityLocationGraph.shortest_path_cost(facilities)` | 计算当前布局的 P-median 目标成本 |

---

## ④ 技能关联

### 前置技能
- [[Skill-Two-Echelon-Inventory-DRL]]：理解 MDP、PPO、奖励设计，与本技能共享 RL 底层框架
- [[Skill-Safety-Stock-Replenishment]]：理解设施选址与库存策略的上下游关系
- **图神经网络基础（HGT / HGCN）**：理解 GNN 节点聚合机制，有助于理解本算法的状态编码

### 延伸技能
- [[Skill-Multi-Echelon-Inventory]]：设施选址确定后，用多级库存优化各站点的补货策略
- [[Skill-Demand-Forecasting-Supply-Chain]]：为 PPO-swap 提供更准确的节点需求权重
- [[Skill-Lead-Time-Distribution-Risk-GenQOT]]：评估选址方案下的交货期风险

### 可组合
- **Two-Echelon-Inventory-DRL + PPO-swap**：先用 PPO-swap 决定开哪里，再用 Two-Echelon DRL 决定每个仓库存多少
- **Demand Forecasting + PPO-swap**：需求预测的季节性输出作为节点权重输入，实现"随需求热力图动态调仓"

---

## ⑤ 商业价值评估

### ROI 预估

| 指标 | 预估 | 量化依据 |
|------|------|---------|
| 末端配送成本 | 降低 8-12% | 加权配送距离下降 10-15%，成本弹性约 0.8 |
| 决策周期 | 从 2 周 → 1 天 | 算法替代人工重规划流程 |
| 应急响应速度 | 从小时级 → 分钟级 | 推理速度比 Gurobi 快 2000 倍 |
| 季度优化次数 | 4 次/年 → 12 次/年 | 低决策成本支持更高频调整 |

以日均 3000 单、每单末端成本 8 元估算：每年节省配送成本约 **70-100 万元**（中型母婴出海品牌），同时应急选址能力可有效降低断货损失。

### 实施难度
⭐⭐⭐☆☆ (3/5)
- 需要道路距离矩阵（高德/百度 API 可获取）
- 无需 GPU，CPU 上推理毫秒级
- 需要标注节点需求权重（订单热力图）
- 从 Mock Agent 测试到真实 PPO 训练需要 2-4 周工程投入

### 优先级评分
⭐⭐⭐⭐☆ (4/5)

### 评估依据
1. **高频需求**：母婴出海品牌面临城市快速扩张、配送站动态调整的强刚需，季度调仓是常规运营动作
2. **门槛适中**：不需要 GPU，只需订单数据 + 地图 API，技术栈对数据团队友好
3. **差异化能力**：能做应急秒级重规划，是传统工具无法替代的核心竞争力
4. **可扩展**：从单城市到多城市、从单品类到全品类，模型泛化能力强（论文验证的小图训练→大图推理）

---

## 参考资料
- Guo, W., Wang, R., Xu, Y., & Jin, Y. (2025). "Unified and Generalizable Reinforcement Learning for Facility Location Problems on Graphs." *The Web Conference 2025 (WWW 2025)*. 上海交通大学.
- 萃取来源：`paper2skills-vault/papers/04-供应链/facility_location_rl_2025/extract.md`
- 代码实现：`paper2skills-code/04-供应链/facility_location_rl_2025/model.py`
