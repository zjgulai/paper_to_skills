# Skill Card: AgentRouter — 知识图谱引导的多智能体路由器

roadmap_phase: phase2
---

## ① 算法原理

### 核心思想

**AgentRouter** 解决多智能体系统（MAS）中最头疼的调度问题：当一个复杂查询进来时，**如何决定把任务交给哪个 Agent**？传统路由依赖 LLM-as-judge 的 Prompt 规则，看不到查询背后隐藏的深层语义关系，经常把"技术故障导致退换货"错判为纯政策问题。

AgentRouter 的核心创新：**将 Agent Routing 转化为基于异构图的节点分类问题**。

1. **构建异构图**：将用户 Query、从查询中提取的领域实体（知识图谱节点）、所有可用 Agent，共同映射到一张异构图。节点之间通过领域重叠语义建边。
2. **GNN 消息聚合**：训练一个异构图神经网络（Hetero-GNN），在图上传播消息。GNN 能"看"到：这个 Query 涉及哪些领域实体，这些实体历史上与哪个 Agent 协作表现最好。
3. **软路由输出**：不做硬指派，而是输出各 Agent 的任务适应度分布（softmax），支持 Top-K 加权协作生成最终答案。

### 数学直觉

**异构消息传播**（每种元关系独立参数）：

```
消息: m_{s→t} = ReLU(W_{rel} · h_s + b_{rel})      # rel ∈ {query_entity, entity_agent, ...}
聚合: h'_t = ReLU(W_update · mean(m_{s→t}) + h_t)   # mean-pooling + 残差
评分: score_i = W_score · h'_{agent_i}               # Agent 适应度打分
路由: p_i = softmax(score_i)                          # 软路由概率分布
```

其中 `W_{rel}` 针对不同元关系 `(src_type, edge_type, dst_type)` 各自独立，这是异构 GNN 区别于同构 GNN 的核心差异。

### 关键假设

1. **领域重叠可建边**：Query 和 Agent 的领域标签能正确捕获语义关联（NER 提取质量要够）
2. **历史协作信号可学习**：积累一定量的 {Query, 路由决策, Agent 表现} 三元组用于训练
3. **异构图连通性**：Query → Entity → Agent 的路径存在，信息能传播到目标节点

---

## ② 母婴出海应用案例

### 场景一：跨品类复合型客诉分发

**业务问题**：

大促期间用户抛出跨领域复合型投诉："我上周买的 A 型号吸奶器，配的 B 充电线插上去闪红灯，而且你们退换货政策说 C 情况不让退，我这算吗？"。传统意图识别只抓住"退换货"关键词，直接转给政策 Agent，导致技术故障原因被忽略，回答残缺不全，用户满意度下降 23%。

**数据要求**：

| 数据类型 | 格式 | 说明 |
|---------|------|------|
| 产品知识图谱 | (产品, 配件, 故障类型) 三元组 | 各型号产品的硬件关系图 |
| 历史工单 + Agent 处理结果 | JSONL：{query, assigned_agent, csat_score} | 用于训练路由器 |
| Agent 能力标签 | 字典：{agent_name: [domain_tags]} | 技术排障/法务政策/订单/推荐 |
| 领域实体词典 | 关键词 → 领域映射 | "充电/闪灯" → product_tech |

**预期产出**：

```
输入查询: "A型号充电线闪红灯 + 退换货政策C情况"

知识图谱路由:
  检测领域: [product_tech, policy]
  GNN 聚合后适应度分布:
    技术排障Agent: 65%
    法务政策Agent: 28%
    订单物流Agent: 5%
    选品推荐Agent: 2%

Top-2 软路由输出:
  技术排障Agent (70%) + 法务政策Agent (30%) → 协作生成回答
```

**业务价值**：

- 大促高峰期每日 5 万条跨领域工单，正确路由率从 61% → 82%，每天减少约 10,500 条二次转单
- 单条转单处理成本约 5 元，节约运营成本 **5.25 万元/天**；年化 **1900 万元**
- 用户 CSAT 评分从 3.8 → 4.3（满分 5），复购意愿提升可观

### 场景二：选品导购 Agent 智能分流

**业务问题**：

母婴用户对产品的问题模糊多样："我宝宝3个月，想买个安静的吸奶器，有什么推荐？"——这里既包含商品推荐意图，也可能隐含产品参数比较诉求。传统路由容易均匀分流给所有 Agent，造成推荐 Agent 输出不够深度，参数对比缺失。

**数据要求**：

- 产品参数图谱（品类 × 功能 × 适龄标签）
- 用户历史浏览 / 购买行为图（用户 → 产品 → 属性三元组）
- 已标注的导购对话训练集（>3000 条）

**预期产出**：

```
输入: "3个月宝宝，静音吸奶器推荐"
路由: 选品推荐Agent (72%) + 产品参数Agent (28%)
输出: 按静音等级 + 适龄参数过滤的 Top-3 产品，附参数对比表
```

**业务价值**：

- 导购转化率从 8.3% → 12.1%，提升 46%
- 月导购对话量 8 万条 × 客单价 380 元 × 3.8% 增量转化 = **月增收 115 万元**

---

## ③ 代码模板

> 完整可运行代码见：`paper2skills-code/08-知识图谱/agentic_kg_2024/model.py`

```python
from paper2skills_code.knowledge_graph.agentic_kg_2024.model import (
    AgentRouter, AgentProfile
)
import numpy as np

# 1. 初始化路由器（feat_dim=8, hidden_dim=16, top_k=2）
router = AgentRouter(feat_dim=8, hidden_dim=16, top_k=2)

# 2. 注册业务 Agent
rng = np.random.default_rng(1)
router.register_agent(AgentProfile(
    name="技术排障Agent",
    domains=["product_tech", "product_info"],
    feature_vector=rng.uniform(-1, 1, 8),
))
router.register_agent(AgentProfile(
    name="法务政策Agent",
    domains=["policy"],
    feature_vector=rng.uniform(-1, 1, 8),
))

# 3. 添加领域知识图谱实体
router.add_knowledge_entity("充电故障知识库", ["product_tech"])
router.add_knowledge_entity("退换货政策图谱", ["policy", "order"])

# 4. 路由执行
result = router.route(
    "我的吸奶器充电线插上去闪红灯，而且你们的退换货政策说不让退，我这算吗？"
)

print(f"Top-K 路由: {result.top_k_agents}")
# 输出示例: [('技术排障Agent', 0.68), ('法务政策Agent', 0.32)]
print(f"路由原因: {result.routing_reason}")

# 5. 使用路由权重进行加权协作（示意）
for agent_name, weight in result.top_k_agents:
    response = call_agent(agent_name, result.query, weight=weight)
    # ... 加权聚合 response
```

**核心类说明**：

| 类/函数 | 职责 |
|---------|------|
| `HeterogeneousGraph` | 异构图数据结构（节点/边/邻接表）|
| `HeteroGNNLayer` | 单层异构 GNN，按元关系独立投影 + mean-pooling 聚合 |
| `AgentRouterGNN` | 两层 GNN + 分类头，输出 Agent 适应度分布 |
| `AgentRouter` | 业务封装：注册 Agent、添加知识实体、执行路由 |
| `RoutingResult` | 路由结果：概率字典 + Top-K 归一化权重 + 路由原因 |

---

## ④ 技能关联

**前置技能**：

- [[Skill-HGT-Heterogeneous-Graph-Transformer]] — 异构图 Transformer 原理，是 AgentRouter GNN 设计基础
- [[Skill-KGQA-Question-Answering]] — 知识图谱问答，理解 Query → KG 的连接逻辑
- [[Skill-KG-Auto-Construction-Agent-Driven]] — 知识图谱自动构建，为 AgentRouter 提供图谱数据

**延伸技能**：

- [[Skill-MAS-Orchestrator]] (10-MAS) — 在路由结果之上做 Agent 编排和任务规划
- [[Skill-Knowledge-Graph-for-Skills-Management]] — 将 Agent 能力本身用图谱管理，增强路由准确性

**可组合技能**：

- `AgentRouter` + [[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]] → 路由到正确 Agent 的同时提供 KG 增强的上下文
- `AgentRouter` + [[Skill-Hierarchical-Product-KG-Construction]] → 用层次化产品知识图谱作为 Router 的底座，覆盖更细粒度的产品领域划分
- `AgentRouter` + [[Skill-KG-Augmented-Recommendation-CoLaKG]] → 兼顾导购推荐意图的精准 Agent 分发

---
- **相关技能**：[[Skill-Agentic-SCKG-Risk]]
- **关联**：[[Skill-AI-Consumer-Wellbeing-Ethics]]
- **相关**：[[Skill-CDA-Privacy-Causal-Attribution]]
- **相关**：[[Skill-Dynamic-Pricing-Elasticity]]
- **相关**：[[Skill-AnchorCrafter-Virtual-Anchor-Demo]]

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 年节省运营成本 1900 万元（客服路由准确率 +21%）+ 导购月增收 115 万元 |
| **实施难度** | ⭐⭐⭐☆☆（3星）|
| **优先级评分** | ⭐⭐⭐⭐☆（4星）|

**评估依据**：

- **ROI 高**：路由准确率从 61% → 82% 对应真实可量化的转单成本节省（5元/条 × 10500条/天）
- **难度中等**：核心 GNN 代码已封装（见 model.py），主要工作在于：① 产品/政策知识图谱搭建（约 2 周）；② 历史工单标注训练集构建（约 3 周）；③ 上线 A/B 验证（约 2 周）
- **优先级高**：直接解决大促高峰的客服分发瓶颈，属于 **WF-C（客服工作流）的 P0 基础设施**
- **Gap 价值**：改变了知识图谱仅作"外部字典"查询的被动角色，**让 KG 成为整个多 Agent 团队的调度中枢**，属于图谱驱动智能体的核心能力跃升

**实施路线图**：

```
Week 1-2: 整理产品/政策知识图谱 → Neo4j 或 dict 形式导入
Week 3-5: 历史工单标注 (query, best_agent, csat) → 3000+ 条
Week 6:   训练 AgentRouterGNN（2层 Hetero-GNN，< 1h GPU 时间）
Week 7:   A/B 实验上线，对照组=纯关键词路由
Week 8:   评估 CSAT 和转单率，决定全量放量
```

---

*论文来源：AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering (arXiv: 2510.05445)*
*代码路径：`paper2skills-code/08-知识图谱/agentic_kg_2024/model.py`*
