# 萃取记录: AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering

## 论文信息

- **arXiv ID**: 2510.05445
- **标题**: AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering
- **发布时间**: 2025 (预印版)
- **领域**: 16-智能体工程 / 08-知识图谱
- **研究方向**: 多智能体协作 (Multi-Agent Collaboration)、知识图谱引导 (KG-Guided)、智能体路由 (Agent Routing)、图神经网络 (GNN)

## 核心算法提炼

### 算法名称
AgentRouter (知识图谱引导的多智能体路由器)

### 核心思想
在多智能体系统（Multi-Agent Systems）中，不同的 Agent 拥有不同的能力（比如有的擅长数学计算，有的擅长信息检索，有的擅长逻辑推理）。当一个复杂的用户查询进来时，该怎么决定把任务分发给哪个 Agent？现有的路由策略多依赖基于规则的 LLM-as-judge，缺乏细粒度的上下文感知。

AgentRouter 的核心创新是**将 Agent Routing 转化为一个基于知识图谱的节点分类/链接预测问题**：
1. **构建异构图（Heterogeneous Graph）**：将当前的用户查询（Query）、从查询中抽取的背景知识实体（Contextual Entities）以及所有可用的智能体（Agents）共同映射到一张知识图谱中。节点间通过语义和关系边相连。
2. **GNN 任务感知路由**：训练一个异构图神经网络（Heterogeneous GNN），在图上进行信息聚合。图网络能“看”到：这个 Query 涉及到哪些领域的实体，而这些实体之前与哪个 Agent 协作得最好。
3. **软监督与集成**：不直接做硬指派（Hard assignment），而是输出一个各个 Agent 的任务适应度分布，使用软监督信号（基于各 Agent 历史在相似图结构上的表现）进行训练。最终结果可以是 Top-K Agent 的加权聚合。

### 为什么好用（优势）
1. **超越“纯大模型路由”的上下文穿透力**：纯文本的 LLM Router 看不到底层复杂关系；而结合了 KG 的 Router 可以通过实体跳跃（Multi-hop reasoning）发现“原来这个请求隐藏着化学领域的逻辑，应该调起化学 Agent”。
2. **自适应与动态学习**：随着系统历史数据的积累，Router 能动态学到哪些 Agent 在哪些特定知识图谱簇上表现更好，不断进化，而不是依赖写死的 Prompt 规则。

## 业务适配设计：基于知识图谱的智能客服与专家分发中枢

### 场景: 跨境电商 / 复杂 SaaS 的多模态 AI 专家客服分发
在大型出海品牌（如包含 3C、母婴、家居的多品类独立站）的售后中心，用户会抛出极其复杂的问题：“我上周买的 A 型号吸奶器，配的 B 充电线插上去闪红灯，而且你们的退换货政策说C情况不让退，我这算吗？”
- **痛点**：如果只用一个通用 LLM 处理，很容易产生幻觉或回答不专业。如果拆成多个特定领域的 Agent（产品技术 Agent、订单 Agent、法务政策 Agent），传统基于意图识别的 Router 常常分发错误（只看到了“退换货”，没看到底层的“技术故障导致”）。
- **AgentRouter 方案落地**：
  - 用户查询被瞬间拆解，连接到背后的产品知识图谱（A型号配件关系图）、订单图谱和售后政策图谱。
  - GNN Router 在这张异构图上跑一次，发现节点高度聚集在“产品硬件故障”和“特殊退换条例”周围。
  - 路由分布输出：70% 权重交给【高级技术排障 Agent】，30% 权重交给【法务政策 Agent】。
  - 两者协作生成最终回答给用户。
- **预期价值**：通过图谱增强分发，解决多 Agent 系统中最头疼的“请求调度”问题，使得大促期间海量跨界客诉的机器拦截率和准确率大幅跃升。

### 可以推荐给用户的理由
这是**Gap 2（图谱驱动的智能体）**的完美填补。它改变了知识图谱通常只作为“外部字典（RAG）”给 Agent 查单词的被动地位；在这里，**知识图谱变成了整个 Multi-Agent 团队的 CEO 和调度中枢**。它提供了一套基于深度学习（GNN）的硬核工程方法，把“智能体编排”从 prompt-engineering 时代推进到了 representation-learning 时代。
