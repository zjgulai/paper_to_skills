# Skill Card: Agentic SCKG Risk Analyzer — 供应链知识图谱智能风险分析框架

roadmap_phase: phase2
---

## ① 算法原理

### 核心思想

**Agentic SCKG Risk Analyzer** 解决出海品牌面临的生死级挑战：当全球某处发生黑天鹅事件（罢工/地震/制裁），你的旗舰产品会在**多少天后断供**？传统方案要么靠 ERP 人工逐层排查（耗时数周），要么让 LLM 在非结构化新闻里盲目搜索（漏掉隐藏链路）。

本框架的核心创新：**将网络科学与知识图谱的「二象性」结合**，构建端到端的供应链风险穿透系统。

1. **统一图表示**：将采购关系、BOM 物料清单、供应商层级、地理位置一次性映射为供应链知识图谱（SCKG）。节点=供应商/品牌方，边=供应关系（含 lead_time、依赖比例、年采购额）。

2. **中心度引导的图遍历**：风险事件触发时，用 **PageRank**（衡量经济影响力）和**介数中心度 Betweenness**（衡量瓶颈路径）的加权分作为导航指南针。图爬虫沿着"最具连带价值"的路径向下游传播，精准定位影响链。

3. **上下文外壳封装**：将冰冷的节点数字因子（违约概率、库存天数、lead_time）包裹在精巧的自然语言模版（Context Shell）中，使提取出的图结构能被 LLM 原生理解，直接输出"诊断报告 + 建议行动"。

### 数学直觉

**PageRank（行归一化，反映经济权重）**：

```
W[i,j] = annual_volume × dependency_ratio   (边权重)
W_norm = 行归一化(W)；悬空节点均匀跳转
PR_new = (1-d)/N + d × W_norm^T × PR        (d=0.85)
```

**级联风险得分**：

```
Risk = severity × Π(1 - 0.3 × slack_i) × (1 + mean(default_prob_i))
其中 slack_i = min(inventory_days_i / lead_time_i, 1)  # 库存缓冲率
```

风险沿路径逐节点累乘衰减，库存越充裕衰减越多；平均违约率叠加放大。

### 关键假设

1. **供应链图谱数据可获取**：需要 Tier 1~N 供应商数据和 BOM 关系（ERP/采购系统可导出）
2. **图的连通性**：风险源到目标品牌之间存在可通路径（即供应关系有完整录入）
3. **历史违约率可信**：各节点的 default_prob 由历史采购数据或信用评级估算

---

## ② 母婴出海应用案例

### 场景一：越南罢工 → 智能清洁家电断供推演（核心场景）

**业务问题**：

某国内出海智能家电品牌，V8 旗舰款吸尘器依赖一条跨越越南、韩国、深圳的四级供应链。某天新闻出现"越南工业区大规模罢工"，采购总监知道直接供应商里没有越南企业，但不知道越南工厂是否是自己三级供应商的材料独家来源。手动 ERP 排查可能要 2-3 周，到时候竞争对手早把备货扫空了。

**数据要求**：

| 数据类型 | 格式 | 说明 |
|---------|------|------|
| 供应商层级图 | 节点：{supplier_id, name, country, tier, default_prob, inventory_days} | ERP 或供应链系统导出 |
| 供应关系图 | 边：{src, dst, lead_time_days, dependency_ratio, annual_volume} | 采购订单/BOM 汇总 |
| 风险事件 | {event_type, location, affected_nodes, severity, description} | 新闻监控系统触发 |
| 备用供应商 | 列表：[{name, country, capacity}] | 供应商池管理系统 |

**预期产出**：

```
事件触发: "越南胡志明工业区大规模罢工（持续4周，严重程度80%）"

中心度引导图遍历结果:
  风险路径: 越南材料厂A → 韩国电机制造商B → 深圳马达组件C → 整机厂D → 品牌方
  传导时间: 45 天（lead_time 累加）
  当前库存: 30 天
  缺口: -15 天   ← 风险实锤

自动生成诊断报告:
  "越南罢工预计将在45天后造成V8马达断供。
   当前安全库存仅能支撑30天，存在15天缺口。
   建议立刻向备用材料商-泰国F追加采购，
   同时联系国内替代材料商G评估产能。"
```

**业务价值**：

- 断供预警从"出事后救火（2-3 周）"提前到"事件触发即秒级诊断（< 10 秒）"
- 旗舰 SKU 断货每天损失约 50-200 万元（依规模），提前 15 天响应直接避免断货损失
- 供应链弹性（Resilience）从"被动反应"升级为"主动链路穿透预警"

### 场景二：制裁/地缘风险下的多链路影响评估

**业务问题**：

出海硬件品牌的核心芯片来自德国 X 供应商，突然被美国实体清单制裁。采购团队需要在 24 小时内评估：① 影响哪些产品线？② 传导到自己需要多少天？③ 哪条备选链路风险最低？

**数据要求**：同场景一，增加"制裁清单"事件类型映射。

**预期产出**：

```
制裁事件: 德国芯片X被制裁（严重程度60%）
传播链1: 芯片X → 整机D → 品牌方  传导17天  风险30%
传播链2: 无直接其他芯片路径
建议: 立即评估备用芯片方案，启动国产替代论证流程
```

**业务价值**：

- 避免信息孤岛：跨部门（采购/研发/法务）统一看到同一张风险传播图
- 决策时效从 3 天压缩到 30 分钟，争取备货先手优势

---

## ③ 代码模板

> 完整可运行代码见：`paper2skills-code/08-知识图谱/supply_chain_kg_2025/model.py`

```python
from paper2skills_code._08_知识图谱.supply_chain_kg_2025.model import (
    AgenticSCKGRiskAnalyzer, SupplierNode, SupplyEdge, RiskEvent
)

# 1. 初始化分析器（指定品牌方节点 ID）
analyzer = AgenticSCKGRiskAnalyzer(brand_node_id="brand_001")

# 2. 构建供应链知识图谱
nodes = [
    SupplierNode("brand_001", "XX智能家电品牌", "中国", tier=0,
                 default_prob=0.01, inventory_days=30,
                 capacity_utilization=0.8, component_type="assembly"),
    SupplierNode("factory_d", "深圳整机厂D",   "中国", tier=1,
                 default_prob=0.03, inventory_days=20,
                 capacity_utilization=0.9, component_type="assembly"),
    SupplierNode("supplier_c", "马达供应商C",  "中国", tier=2,
                 default_prob=0.05, inventory_days=15,
                 capacity_utilization=0.85, component_type="motor"),
    SupplierNode("supplier_b", "韩国电机B",    "韩国", tier=3,
                 default_prob=0.04, inventory_days=10,
                 capacity_utilization=0.9, component_type="motor"),
    SupplierNode("factory_a", "越南材料厂A",   "越南", tier=4,
                 default_prob=0.08, inventory_days=5,
                 capacity_utilization=0.95, component_type="material"),
]
edges = [
    SupplyEdge("factory_a",  "supplier_b", lead_time_days=21, dependency_ratio=0.9,  annual_volume=500),
    SupplyEdge("supplier_b", "supplier_c", lead_time_days=14, dependency_ratio=0.75, annual_volume=800),
    SupplyEdge("supplier_c", "factory_d",  lead_time_days=7,  dependency_ratio=0.6,  annual_volume=1200),
    SupplyEdge("factory_d",  "brand_001",  lead_time_days=3,  dependency_ratio=1.0,  annual_volume=3000),
]
analyzer.build_kg(nodes, edges)

# 3. 触发风险事件分析
event = RiskEvent(
    event_id="evt_001", event_type="strike",
    location="越南胡志明工业区",
    affected_node_ids=["factory_a"],
    severity=0.8,
    description="越南胡志明工业区大规模罢工，预计持续4周",
)

chains, shells = analyzer.analyze_risk_event(
    event=event,
    brand_inventory_days=30,
    alternative_suppliers=["备用材料商-泰国F", "国内替代材料商G"],
    top_k_paths=3,
)

# 4. 输出诊断报告
for i, (chain, shell) in enumerate(zip(chains, shells)):
    print(f"\n【风险链 #{i+1}】级联风险: {chain.cascade_risk_score:.1%}")
    print(f"传播时间: {chain.total_lead_time_days} 天")
    print(shell)

# 5. 查看中心度摘要（节点重要性排名）
summary = analyzer.get_centrality_summary()
for nid, info in sorted(summary.items(), key=lambda x: -x[1]["pagerank"]):
    print(f"{info['name']:30s}  PR={info['pagerank']:.4f}  BT={info['betweenness']:.4f}")
```

**核心类说明**：

| 类/函数 | 职责 |
|---------|------|
| `SupplyChainKG` | 供应链知识图谱（节点/边/邻接表，支持上下游遍历） |
| `CentralityCalculator` | PageRank + 介数中心度计算（纯 numpy，无图库依赖） |
| `CentralityGuidedTraverser` | 中心度引导的 BFS 图遍历，提取 Top-K 风险传播链 |
| `ContextShellGenerator` | 将图结构数据包裹为 LLM 原生可读的自然语言模版 |
| `AgenticSCKGRiskAnalyzer` | 主框架：构建图谱 + 预计算中心度 + 分析风险事件 |
| `RiskPropagationChain` | 风险传播结果：路径段列表 + 传导时间 + 级联风险得分 |

---

## ④ 技能关联

**前置技能**：

- [[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]] — Graph RAG 基础，理解从图谱中提取上下文的原理
- [[Skill-HGT-Heterogeneous-Graph-Transformer]] — 异构图结构的核心概念（节点/边类型、图遍历）
- [[Skill-KG-Auto-Construction-Agent-Driven]] — 供应链知识图谱的自动构建与维护方法

**延伸技能**：

- [[Skill-AgentRouter-KG-Guided]] — 将风险预警系统的输出接入多 Agent 应急响应路由
- [[Skill-Hierarchical-Product-KG-Construction]] — 层次化产品图谱，可用于精细化 BOM 多级关系建模

**可组合技能**：

- `Agentic SCKG` + [[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]] → 图谱遍历找到路径后，用 GraphRAG 做深度语义增强检索，提升诊断报告质量
- `Agentic SCKG` + [[Skill-AgentRouter-KG-Guided]] → 风险传播链分析完毕后，智能路由到采购/法务/研发不同响应 Agent
- `Agentic SCKG` + 供应链库存 Skill（04-供应链域）→ 风险触发后自动计算最优紧急备货量，实现端到端的"感知-分析-决策"闭环

---
- **相关**：[[Skill-CausalRAG-Knowledge-Retrieval]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 旗舰 SKU 断货每天损失 50-200 万元；提前 15+ 天预警即可完全规避，年化价值 **数千万元级**（依品牌规模） |
| **实施难度** | ⭐⭐⭐☆☆（3星）|
| **优先级评分** | ⭐⭐⭐⭐⭐（5星）|

**评估依据**：

- **ROI 极高**：供应链断货是出海品牌生死级风险，一次黑天鹅事件的损失可覆盖系统建设成本数十倍，且本框架无需昂贵专用图数据库，仅依赖 Python 标准库 + numpy
- **难度中等**：核心算法已封装（见 model.py），主要实施成本在于①供应商数据治理（Tier 2+ 数据录入，约 4-8 周）；②新闻/风险事件监控接入（1-2 周）；③报告模版调优（1 周）
- **优先级最高**：属于 **WF-A（供应链工作流）的 P0 战略防御基础设施**，且当前知识图谱域 Skill 库中唯一覆盖"网络科学 × 风险传播"方向，填补关键缺口
- **技术壁垒**：将图论的中心度算法与 LLM Context Shell 结合，形成竞对难以快速复制的"链路穿透预警"护城河

**实施路线图**：

```
Week 1-2:  ERP 数据对齐 → 梳理 Tier 1~3 供应商节点 + 边关系
Week 3-4:  数据导入 SCKG，PageRank/中心度预计算，基础测试
Week 5:    接入新闻监控系统，自动触发风险事件
Week 6:    Context Shell 模版调优，对接 LLM 生成诊断报告
Week 7-8:  桌面推演（模拟 3 个历史黑天鹅场景），验证准确性
Week 9+:   生产上线，持续补录 Tier 4+ 供应商数据扩充图谱
```

---

*论文来源：Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis (arXiv: 2510.01115)*
*代码路径：`paper2skills-code/08-知识图谱/supply_chain_kg_2025/model.py`*
