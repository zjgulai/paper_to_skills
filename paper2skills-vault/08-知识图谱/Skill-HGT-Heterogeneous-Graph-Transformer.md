# Skill Card: HGT — 异构图 Transformer 表示学习

roadmap_phase: phase2
---

## ① 算法原理

### 核心思想

**HGT (Heterogeneous Graph Transformer)** 解决传统 GNN 无法处理异构图（节点和边有多种类型）的核心问题。传统 GNN 假设所有节点和边共享同一特征分布，这在电商场景（用户/产品/评论/属性共存）中完全不成立。

HGT 的核心创新：**用 meta relation (源节点类型, 边类型, 目标节点类型) 三元组来参数化注意力机制**，而非为每种边类型单独维护一套参数。这样实现了：

1. **参数共享**：相似 meta relation 共享参数，泛化到未见过关系
2. **分布自适应**：不同节点类型有独立的投影矩阵，自动适应特征分布差异
3. **隐式元路径学习**：不依赖人工设计的元路径，注意力权重自动发现重要关系链

### 数学直觉

**异构互注意力**：

对于边 e = (s, t)，其 meta relation 为 (tau(s), phi(e), tau(t))。HGT 将标准 Transformer 的 Q/K/V 投影分解为类型相关：

```
K = K-Linear_tau(s)(H^(l-1)[s])     # 源节点类型决定 Key 投影
Q = Q-Linear_tau(t)(H^(l-1)[t])     # 目标节点类型决定 Query 投影
V = V-Linear_tau(s)(H^(l-1)[s])     # 源节点类型决定 Value 投影
```

注意力得分加入 **meta relation 先验偏置**（可学习参数）：

```
Attention = softmax( (Q * K^T) / sqrt(d) + Prior_phi(e) )
```

**异构消息传递**：每种边类型有独立的消息投影矩阵，消息经过类型变换后再聚合。

**目标特定聚合**：聚合后的消息通过目标节点类型特定的线性层变换，最后残差连接 + LayerNorm。

**HGSampling**：异构 mini-batch 采样算法。为每种节点类型维护独立采样预算，保证子图中各类节点数量平衡，避免高热度类型（如用户）主导采样。

**相对时间编码 (RTE)**：用正弦/余弦函数编码边的时间戳差，处理动态图（如用户随时间的购买行为）。

### 关键假设

1. **Meta relation 能刻画异质性**：节点类型 + 边类型 + 节点类型足以区分不同交互模式
2. **参数共享可泛化**：相似 meta relation 共享大部分参数，仅需少量特定参数
3. **图连通性**：相关实体在图中存在路径连接（适用于电商场景的交互图）
4. **归纳式学习能力**：unseen 节点可通过邻居聚合获得有效表示（新品冷启动）

---

## ② 母婴出海应用案例

### 场景一：跨语言商品属性对齐与品类推断

**业务问题**：

母婴出海电商覆盖多语言市场（英文站、中文站、日文站）。同一产品在不同语言站的描述和评论存在大量语义等价但表述不同的属性（如英文 "portable" = 中文 "便携" = 日文 "持ち運び"）。传统方法需要人工维护多语言对齐词典，成本高且难以扩展。

**数据要求**：

- 多语言产品图：
  - 节点：product（产品）、attribute_en（英文属性）、attribute_zh（中文属性）、attribute_ja（日文属性）
  - 边：(product, has_attribute, attribute_*)、(attribute_*, synonym_of, attribute_*)
- 跨语言评论数据：各语言站的评论文本及提取的属性
- 产品基础信息：品牌、品类、价格带

**预期产出**：

```
输入: 日文站新品 "電動搾乳器"（无英文/中文数据）

HGT 推断过程:
  1. 通过品牌边连接到已有数据的同品牌产品
  2. 通过品类层次边连接到父品类 "搾乳器"
  3. 跨语言属性通过 product 节点隐式对齐

输出:
  - 推断品类: 吸奶器 / Breast Pump
  - 推断属性: 電動(electric), 静音(quiet), 便利(convenient)
  - 跨语言对齐: 電動=electric=电动, 静音=quiet=静音
```

**业务价值**：
- 新语言站点上线周期从 2-3 个月缩短至 2 周
- 无需人工维护多语言词典，节省翻译/标注成本 60%+
- 支持小语种市场（如阿拉伯语、泰语）快速扩展

---

### 场景二：新品冷启动表示推断与个性化推荐

**业务问题**：

母婴电商新品上架频繁（如新款吸奶器、新配方奶粉）。新品没有历史评论、购买记录，传统推荐系统无法为其生成有效表示，导致新品曝光不足、冷启动期过长。

**数据要求**：

- 产品关系图：
  - 互补商品（accessory）、替代商品（substitute）、同品牌（same_brand）、同品类（same_category）
- 用户-产品交互图：购买、浏览、收藏、评价
- 产品属性图：价格带、功能特征、适用人群

**预期产出**：

```
新品上架: "Spectra S2 便携吸奶器"
  - 无购买记录
  - 无评论数据
  - 仅有基础属性信息

HGT 冷启动推断:
  1. 通过 "same_brand" 边连接到 Spectra S1（已有丰富数据）
  2. 通过 "same_category" 边连接到同品类其他吸奶器
  3. 通过互补边连接到储奶袋、温奶器等配件

推断结果:
  - 产品表示向量 = 0.5*品牌邻居 + 0.3*品类邻居 + 0.2*属性自编码
  - 目标用户群: 上班族母乳妈妈（从同品牌用户推断）
  - 推荐搭配: 储奶袋（互补度 0.85）、温奶器（互补度 0.72）
  - 预估首月转化率: 3.2%（基于相似品品类比）
```

**业务价值**：
- 新品冷启动期从 30 天缩短至 7 天
- 首月转化率提升 40-60%（通过精准初始推荐）
- 长尾商品曝光量提升 2-3 倍

---

## ③ 代码模板

代码位置：`paper2skills-code/knowledge_graph/hgt_ecommerce/hgt_model.py`

核心组件：
- `HGTLayer`: 单层异构 Transformer（异构注意力 + 消息传递 + 目标聚合）
- `HGT`: 多层堆叠网络 + 输入/输出投影
- `HGSampling`: 异构 mini-batch 采样（类型平衡预算）
- `ProductCategoryClassifier`: 产品品类分类器示例
- `build_maternal_baby_hetero_graph`: 母婴电商异构图构建

运行方式：
```bash
cd paper2skills-code/knowledge_graph/hgt_ecommerce
pip install torch torch_geometric
python hgt_model.py
```

生产环境建议：
1. 接入官方 pyHGT 实现处理 Web 规模图（GitHub: acbull/pyHGT）
2. 使用预训练 embedding（BERT/Word2Vec）初始化节点特征
3. HGSampling + mini-batch 训练处理千万级节点
4. 结合 RTE 处理时序购买行为（季节性、促销效应）

---

## ④ 技能关联

### 前置技能
- **Graph Neural Networks (GNN)**：理解消息传递、图卷积基本概念
- **Transformer / Attention**：理解自注意力、多头注意力机制
- **图论基础**：节点、边、邻接矩阵、图遍历

### 延伸技能
- **超球面图神经网络 (HGCN)**：层次化结构编码（品类树、品牌层次）
- **动态图网络**：持续学习用户行为演化（RTE 扩展）
- **图对比学习**：自监督预训练提升表示质量
- **图神经架构搜索 (GNAS)**：自动优化异构网络结构

### 可组合技能
- **GraphRAG**：HGT 提供高质量实体表示，GraphRAG 基于表示做检索
- **推荐系统**：异构用户-产品表示用于协同过滤
- **知识图谱构建**：HGT 编码的节点表示用于实体链接和消歧
- **因果推断**：在异构图上建模干预效果（用户看到推荐后的购买决策）

---

- **前置（prerequisite）**：[[Skill-GNN-Foundations]]（图神经网络消息传递基础）
- **前置（prerequisite）**：[[Skill-HGCN-Hyperbolic-Graph-Convolutional-Networks]]（异构图卷积基础）
- **延伸（extends）**：[[Skill-KG-Powered-User-Profiling]]（HGT 学习的异构图表示驱动用户画像）
- **延伸（extends）**：[[Skill-Hierarchical-Product-KG-Construction]]（商品知识图谱构建后用 HGT 学习表示）
- **可组合（combinable）**：[[Skill-MAS-Collaborative-Recommendation]]（组合：HGT 学习异构商品-用户图 + MAS 多智能体协作推荐）

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 跨语言属性对齐 | 新站点上线成本降低 60%，翻译团队缩减 2-3 人 | 开发 4-6 周 | 12-18x |
| 新品冷启动 | 首月转化率提升 40-60%，长尾曝光提升 2-3 倍 | 开发 3-4 周 | 10-15x |
| 用户-产品表示优化 | 推荐点击率提升 15-25%，交叉销售率提升 20% | 开发 2-3 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：需要构建完整的异构图（多类型节点+边），数据工程量大
- 技术门槛：较高，需理解 meta relation、异构注意力、图采样
- 工程复杂度：中高，PyTorch Geometric 异构图 API 有学习曲线
- 维护成本：中，图结构变化需重新训练或增量更新

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- **业务价值高**：跨语言对齐和冷启动是出海电商核心痛点
- **技术前沿性**：HGT 是异构表示学习的 SOTA 基础，WWW 2020 后大量后续工作
- **可落地性强**：有成熟开源实现（pyHGT），可直接工程化
- **长期演进路径清晰**：可向动态图、超球面空间、对比学习方向持续迭代

### 评估依据
1. **HGT 在 OAG (179M nodes, 2B edges) 上验证**：比 SOTA 提升 9%-21%，Web 规模可扩展
2. **母婴电商天然适合异构图**：用户-产品-评论-属性四种类型天然构成异构结构
3. **与现有技能互补**：可与 GraphRAG、推荐系统、知识图谱构建形成技术栈闭环
4. **跨语言是出海刚需**：HGT 的归纳式学习能力直接解决多语言扩展问题

---

## 参考论文

1. **Heterogeneous Graph Transformer** (WWW 2020)
   - Hu, Z., Dong, Y., Wang, K., Sun, Y.
   - 核心贡献：meta relation 参数化的异构注意力 + HGSampling + RTE
   - 代码：https://github.com/acbull/pyHGT

2. **Inductive Representation Learning on Large Graphs** (NeurIPS 2017)
   - Hamilton, W., Ying, Z., Leskovec, J.
   - 核心贡献：GraphSAGE，归纳式学习的基础

3. **Modeling Relational Data with Graph Convolutional Networks** (ESWC 2018)
   - Schlichtkrull, M., et al.
   - 核心贡献：R-GCN，异构图卷积的先驱工作

---

## 开源资源

- **pyHGT (官方实现)**：https://github.com/acbull/pyHGT
- **PyTorch Geometric HeteroData**：https://pytorch-geometric.readthedocs.io/
- **DGL 异构图教程**：https://docs.dgl.ai/tutorials/blitz/index.html

---

## 学习路径

```
GNN 基础 (GCN, GraphSAGE)
    ↓
注意力机制 (GAT)
    ↓
异构 GNN (R-GCN, HAN)
    ↓
HGT (异构 Transformer)
    ↓
动态图 / 超球面图 / 图对比学习
```

---

## 与现有技能的关联

本技能是 **Knowledge Graph for Skills Management** 和 **GraphRAG** 的底层表示学习层：

```
Knowledge Graph 构建 (图结构)
        ↓
HGT 异构表示学习 (节点向量)
        ↓
GraphRAG 检索 (基于向量的图检索)
        ↓
推荐系统 (表示用于协同过滤)
```
