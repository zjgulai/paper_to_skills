---
title: 22-数据采集工程 2026 萃取计划 — 新领域完整选题
doc_type: analysis
module: paper-选题
topic: data-collection-engineering-2026
status: active
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
---

# 22-数据采集工程 2026 萃取计划

## 背景

当前图谱 302 个 Skills 全部假设"数据已就绪"。母婴跨境电商的数据从哪来、如何采集、质量如何保证——是结构性空白。本计划新建 `22-数据采集工程/` 领域，填补**全部下游 Skill 的数据源前置**。

**定位**：Layer 1 基础层（所有分析 Skill 的上游前置），与 `09-DataAgent-LLM` 的关系：
- DataAgent = 分析**已有**数据
- 数据采集工程 = **获取**原始数据

---

## 领域选题全景（5 大方向，18 篇核心论文，全部 2026 年）

---

### 方向 A：智能网页爬取与变化检测

> 填补断链：`Skill-Competitor-Product-Intelligence`、`Skill-Listing-Quality-Scoring`、`Skill-Market-Signal-Realtime-Collection` 的数据源

#### A1 · `Skill-LLM-Focused-Web-Crawling`

**核心算法**：LLM/MLLM 引导的主题爬取，覆盖动态 JS 页面

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **Webscraper: MLLM for Index-Content Web Scraping** | [2603.29161](https://arxiv.org/abs/2603.29161) | 5 阶段 MLLM 提示流程自主导航动态页面，显著优于 Anthropic Computer Use 基线 |
| **Coverage-Aware Crawling via W→K→W Pipeline** | [2602.24262](https://arxiv.org/abs/2602.24262) | LLM 提取 KG 空洞反向驱动爬取，比基线少爬 32% 页面达更高实体发现率 |

**业务场景**：自动采集竞品 Amazon Listing（动态 JS 渲染）→ 驱动 `Skill-Competitive-Price-Monitoring`
**图谱 edges**：
- prerequisite → `Skill-Competitor-Product-Intelligence`（数据源）
- prerequisite → `Skill-Listing-Quality-Scoring`（数据源）
- combinable ↔ `Skill-GraphRAG-Knowledge-Enhanced`（KG 驱动爬取）

---

#### A2 · `Skill-Adaptive-Crawl-Scheduling`

**核心算法**：Sleeping Bandit + 噪声感知 Poisson 调度，最大化采集效率

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **SB-CLASSIFIER: Efficient Crawling via Sleeping Bandit** | [2602.11874](https://arxiv.org/abs/2602.11874) | 2200 万页实验：爬 20% 页面获得 90% 目标资源（EDBT 2026） |
| **Neural Prioritisation for Web Crawling** | [2506.16146](https://arxiv.org/abs/2506.16146) | LLM 质量评估器替代链接图优先级，HR 最高提升 +149% |
| **Noisy Change-Indicating Signals for Crawl Scheduling** | [2502.02430](https://arxiv.org/abs/2502.02430) | Poisson 更新 + 噪声信号，推导最优离散调度策略 |

**业务场景**：大促前自动加密采集热门 SKU 价格/BSR 变化频率，节省 70% 带宽配额
**图谱 edges**：
- extends → `Skill-LLM-Focused-Web-Crawling`（调度层）
- combinable ↔ `Skill-Market-Signal-Realtime-Collection`

---

#### A3 · `Skill-Web-Page-Change-Detection`

**核心算法**：VLM 视觉差异检测 + DOM MutationObserver 原子性保护

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **DiffSpot: VLMs for Web Interface Change Detection** | [2605.29615](https://arxiv.org/abs/2605.29615) | 4400 对 CSS 突变 benchmark，评估 13 个 VLM 的视觉级变化检测能力 |
| **Atomicity for Agents: Mitigating TOCTOU via DOM Observer** | [2603.00476](https://arxiv.org/abs/2603.00476) | DOM MutationObserver 执行前校验，TOCTOU 漏洞触发率降至 0% |

**业务场景**：监控竞品 Listing 页面价格/评价/图片变化，仅在"变化时"触发全量抓取
**图谱 edges**：
- prerequisite ← `Skill-Adaptive-Crawl-Scheduling`
- combinable ↔ `Skill-Listing-Quality-Scoring`

---

### 方向 B：文档智能与非结构化数据解析

> 填补断链：`Skill-Supplier-Capacity-Planning`、`Skill-Multi-SKU-Procurement-Budget-Allocation` 的数据输入

#### B1 · `Skill-Document-Intelligence-Parsing`

**核心算法**：Layout-aware LLM 解析 PDF/发票/报价单，统一图文输出

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **dots.mocr: Multimodal OCR Parse Anything** | [2603.13032](https://arxiv.org/abs/2603.13032) | 首个文本+图形统一结构化输出，3B 模型 olmOCR-Bench SOTA 83.9，开源 |
| **Qianfan-OCR: Layout-as-Thought Document Intelligence** | [2603.13398](https://arxiv.org/abs/2603.13398) | `<think>` token 驱动布局推理，OmniDocBench 端到端第一（93.12），表格 TEDS 91.02 |
| **MinerU-Popo: Cross-Page Table Recovery** | [2605.24973](https://arxiv.org/abs/2605.24973) | 跨页表格截断恢复专用后处理，TEDS 提升 ≥20%，RAG 延迟降低 70% |

**业务场景**：解析供应商 PDF 报价单 → 自动结构化 SKU/价格/MOQ → 输入 `Skill-Multi-SKU-Procurement`
**图谱 edges**：
- prerequisite → `Skill-Multi-SKU-Procurement-Budget-Allocation`（数据源）
- prerequisite → `Skill-Supplier-Capacity-Planning`（数据源）
- combinable ↔ `Skill-MAS-Dynamic-KG-Collaboration`

---

#### B2 · `Skill-Procurement-Email-Extraction`

**核心算法**：合同/邮件结构化提取 + MILP 合规验证

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **Contract2Plan: Contract-Grounded Procurement Planning** | [2601.06164](https://arxiv.org/abs/2601.06164) | 提取 MOQ/交期/价格梯度为带 evidence span 的类型化约束，MILP 合规验证+迭代修复 |
| **ProUIE: Progressive Universal Information Extraction** | [2604.10633](https://arxiv.org/abs/2604.10633) | 三阶段渐进学习+GRPO 奖励，36 个 NER/RE/EE 数据集一致超越微调基线 |
| **NLP-based Page Classification for Tender Documents** | [ACL PROPOR 2026](https://aclanthology.org/2026.propor-1.61.pdf) | XGBoost 页分类(F1=97.75%) + LLM 抽取，token 减少 64-88% |

**业务场景**：自动解析供应商报价邮件 → 提取 SKU/价格/交期 → 减少人工录入 4-6h/周
**图谱 edges**：
- extends → `Skill-Document-Intelligence-Parsing`
- prerequisite → `Skill-Dynamic-Lot-Sizing-MOQ`（MOQ 数据源）
- combinable ↔ `Skill-Cross-Org-Agent-Protocol`（跨组织数据交换）

---

### 方向 C：评论 / UGC 采集质量

> 填补断链：`Skill-AGRS-Aspect-Guided-Review-Summarization`、`Skill-Review-Pain-Point-Mining` 的输入噪声（30-40% 是垃圾）

#### C1 · `Skill-Review-Dedup-Quality-Filter`

**核心算法**：在线模糊去重 + 结构化聚合排序

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **FOLD: Fuzzy Online Deduplication for Large Evolving Datasets** | [2606.03001](https://arxiv.org/abs/2606.03001) | HNSW+bitmap，比 Milvus 吞吐量 2.09×，召回率 93-97%（2026-06） |
| **ResLPO: Listwise Preference Optimization for Review Ranking** | [2601.07449](https://arxiv.org/abs/2601.07449) | Baby_Products 等 Amazon 四品类，NDCG@10=0.931，50 条超长列表稳定 |
| **Scalable ABSA via LLM+ML for Cross-Platform Reviews** | [2602.21082](https://arxiv.org/abs/2602.21082) | ChatGPT 发现方面词 + 轻量分类器批量标注 470 万条跨平台评论 |

**业务场景**：每日从 Amazon/TikTok/独立站采集的原始评论 → 去重+质量打分 → 进入 VOC 分析流水线
**图谱 edges**：
- prerequisite → `Skill-AGRS-Aspect-Guided-Review-Summarization`（输入清洗）
- prerequisite → `Skill-Review-Pain-Point-Mining`（输入清洗）
- combinable ↔ `Skill-LACA-CrossLingual-ABSA`

---

#### C2 · `Skill-Fake-Review-Detection`

**核心算法**：图神经网络 + LLM 可解释欺诈检测，支持冷启动和语义伪装

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **JARVIS: Evidence-Grounded Deceptive Review Detection** | [2602.12941](https://arxiv.org/abs/2602.12941) | 多模态混合检索+异构证据图，JD.com 精度 0.988，人工审核减少 75% |
| **DS-DGA-GCN: Fake Reviewer Groups in Dynamic Networks** | [2603.08332](https://arxiv.org/abs/2603.08332) | 动态图注意力 GCN，Amazon+小红书双平台刷评群组检测 88-90% |
| **CAMERA: Unsupervised Graph Fraud vs Semantic Camouflage** | [2605.20032](https://arxiv.org/abs/2605.20032) | 无监督 ego-decoupled MoE，检测"刻意模仿正常用户"的进化型刷评 |

**业务场景**：过滤竞品刷评干扰（竞品通过刷低分评论影响 VOC 分析） → 保证数据质量
**图谱 edges**：
- extends → `Skill-Review-Dedup-Quality-Filter`（下游质量层）
- combinable ↔ `Skill-MAS-Dynamic-Trust`（可信数据源评估）
- combinable ↔ `Skill-Review-Fraud-Detection`（19-风控）

---

### 方向 D：点击流采集 / 身份解析

> 填补断链：`Skill-TRACE-Clickstream-Embedding`、`Skill-Trajectory-Pattern-Mining` 的输入数据管道

#### D1 · `Skill-Clickstream-Persona-Pipeline`

**核心算法**：原始点击流 → 离散 Persona Token，含身份拼接

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **SimPersona: Discrete Buyer Personas from Raw Clickstreams** | [2605.14205](https://arxiv.org/abs/2605.14205) | VQ-VAE 从点击流学习离散 persona token，837 万买家，78% 转化率对齐 |
| **BIP: Event Streams to Autonomous Insight** | [2604.22762](https://arxiv.org/abs/2604.22762) | 4 层架构（归一化→Markov→行为 KG→LLM 叙事），含匿名/认证 session identity stitching |
| **SimGym: Traffic-Grounded Browser Agents for A/B** | [2602.01443](https://arxiv.org/abs/2602.01443) | 生产点击流驱动 LLM agent 模拟购物会话，A/B 实验从数周压缩到 1 小时 |

**业务场景**：独立站/APP 原始行为日志 → 用户 persona 化 → 驱动 `Skill-Trajectory-Pattern-Mining`
**图谱 edges**：
- prerequisite → `Skill-TRACE-Clickstream-Embedding`（数据源）
- prerequisite → `Skill-Trajectory-Pattern-Mining`（数据源）
- combinable ↔ `Skill-User-Funnel-Analysis`

---

#### D2 · `Skill-Privacy-Safe-Identity-Resolution`

**核心算法**：跨设备/跨平台 ID 解析 + 隐私合规（GDPR/PIPL）

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **Sherpa.ai: Multi-Party Entity Alignment without Intersection Disclosure** | [2604.19219](https://arxiv.org/abs/2604.19219) | 多方 PSU 协议跨机构用户对齐，隐藏交集身份，适用垂直联邦学习 |
| **Quantizing Intent: Cross-Domain Semantic IDs** | [2606.01396](https://arxiv.org/abs/2606.01396) | 跨域行为 → 离散 token，冷启动用户 AUC +1.522%（工业广告验证） |
| **CAMP: Cross-Turn PII Protection in Multi-Turn Dialogs** | [2604.16521](https://arxiv.org/abs/2604.16521) | 跨轮次累积 PII 暴露形式化 + 会话级假名化，多轮合规 |

**业务场景**：跨平台（Amazon+TikTok+独立站）用户 ID 打通，满足 GDPR/PIPL → 驱动全链路归因
**图谱 edges**：
- prerequisite → `Skill-HGNN-Cross-Device-Matching`（数据源）
- prerequisite → `Skill-CDA-Cookieless-Attribution`（数据源）
- combinable ↔ `Skill-Identity-Fragmentation-Debiasing`

---

### 方向 E：数据质量 / 合成数据 / 血缘追踪

> 填补断链：所有下游 ML 模型的数据治理前置

#### E1 · `Skill-Ecommerce-Data-Quality-Assessment`

**核心算法**：商品 catalog 错误检测 + 缺失模态补全

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **Query-Guided Data Verification Error Mitigation (MESReduce)** | [2603.08612](https://arxiv.org/abs/2603.08612) | Maximal Error Score 量化标注错误影响，高效修复算法 |
| **Benchmarking MLLMs for Missing Modality Completion** | [2601.19750](https://arxiv.org/abs/2601.19750) | 首个电商缺失模态补全基准 MMPCBench，揭示 MLLM 细粒度对齐局限 |
| **macrOData: 2446 Datasets for Tabular Outlier Detection** | [2602.09329](https://arxiv.org/abs/2602.09329) | 超大规模异常检测基准，含电商领域，附 HuggingFace 排行榜 |

**业务场景**：商品 catalog 清洗（缺失图片/错误属性/异常价格）→ 保证下游推荐/搜索质量
**图谱 edges**：
- prerequisite → `Skill-Data-Drift-Detection`（数据质量→漂移检测链路）
- prerequisite → `Skill-Listing-Quality-Scoring`（数据质量前置）
- combinable ↔ `Skill-Model-Performance-Monitor`

---

#### E2 · `Skill-Synthetic-Data-Ecommerce`

**核心算法**：合成数据解决长尾/冷启动/跨域数据稀缺

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **Synthetic Data for Long-tail Knowledge-Intensive Queries** | [2602.23620](https://arxiv.org/abs/2602.23620) · SIGIR'26 | 多奖励 RL Query Rewriting，Query Goodrate +8.62pt |
| **Principled Synthetic Data for LLM Recommendation Scaling Laws** | [2602.07298](https://arxiv.org/abs/2602.07298) · ICML'26 | 首次在推荐场景验证 LLM 幂律，SasRec 召回率 +130% |
| **SCALR: Cross-Domain Event Synthetic Data** | [2606.00282](https://arxiv.org/abs/2606.00282) | 跨域事件迁移合成数据，工业 A/B CVR +0.14-0.24%，天然隐私保护 |

**业务场景**：新品上市无历史数据 → 合成数据驱动 `Skill-New-Product-Inventory-Coldstart` 和推荐冷启动
**图谱 edges**：
- combinable ↔ `Skill-New-Product-Inventory-Coldstart`（冷启动数据增强）
- combinable ↔ `Skill-Bass-Diffusion-New-Product-Forecasting`（新品数据补全）
- prerequisite → `Skill-Cold-Start-Product-Recommendation`

---

#### E3 · `Skill-Data-Provenance-Lineage`

**核心算法**：LLM 训练数据血缘追踪 + 多 Agent 演化图谱重建

| 论文 | arXiv | 核心贡献 |
|------|-------|---------|
| **Tracing the Roots: Multi-Agent Data Lineage Framework** | [2604.10480](https://arxiv.org/abs/2604.10480) | 自动化重建 430 数据集/971 继承边演化图，揭示 benchmark 污染传播路径 |
| **Data Provenance, Transparency and Traceability in LLMs (Survey)** | [2601.14311](https://arxiv.org/abs/2601.14311) | 首篇 95 论文 LLM 数据血缘三轴综述（来源·透明度·可追踪性） |
| **DEBUGLM: Traceable Training Data Provenance** | [2603.17884](https://arxiv.org/abs/2603.17884) | 训练嵌入 Provenance Tag，推理时不重训练即可定点溯源 |

**业务场景**：追踪商品推荐/风控模型的训练数据来源，满足 AI 法规审计要求
**图谱 edges**：
- combinable ↔ `Skill-Data-Drift-Detection`
- combinable ↔ `Skill-Model-Performance-Monitor`
- combinable ↔ `Skill-MAS-Testing-Verification`

---

#### E4 · `Skill-Privacy-Preserving-Federated-Collection`

**核心算法**：联邦学习 + 差分隐私跨平台数据采集

| 论文 | 链接 | 核心贡献 |
|------|------|---------|
| **Adaptive Weighted FL for Cross-Platform E-commerce** | [Informatica 2026](https://www.informatica.si/index.php/informatica/article/view/13208) | 动态多维权重聚合+分层差分隐私，模型反演攻击成功率压至 4.8% |
| **SF-UBM: Privacy-Preserving LLM Recommendation (Federated)** | [2604.14833](https://arxiv.org/abs/2604.14833) | 联邦语义加密跨域推荐，首解非重叠域 PPCDR 问题 |
| **MFG-RegretNet: Privacy Trading in Federated Learning** | [2603.28329](https://arxiv.org/abs/2603.28329) | 差分隐私预算商品化拍卖，复杂度 O(N²logN)→O(N) |

**业务场景**：跨平台（Amazon+TikTok+独立站）数据联合建模，GDPR/PIPL 合规下不共享原始数据
**图谱 edges**：
- combinable ↔ `Skill-Privacy-Safe-Identity-Resolution`
- combinable ↔ `Skill-Category-Compliance-Prescan`（数据合规层）
- prerequisite → `Skill-HGNN-Cross-Device-Matching`（联邦版跨设备）

---

## 萃取顺序（图谱路径排序）

按"填补最多下游 prerequisite 断链"原则排序：

```
Wave 1（最高价值，直接解锁下游 Skill）
  B1  Skill-Document-Intelligence-Parsing      → 解锁供应链 3 个 Skill
  C1  Skill-Review-Dedup-Quality-Filter        → 解锁 VOC 分析全链路
  A1  Skill-LLM-Focused-Web-Crawling           → 解锁竞品监控 2 个 Skill

Wave 2（填补重要数据管道）
  B2  Skill-Procurement-Email-Extraction       → B1 的延伸
  C2  Skill-Fake-Review-Detection              → C1 的下游质量层
  D1  Skill-Clickstream-Persona-Pipeline       → 解锁用户行为 2 个 Skill

Wave 3（规模化与合规）
  A2  Skill-Adaptive-Crawl-Scheduling          → A1 的调度层
  E1  Skill-Ecommerce-Data-Quality-Assessment  → 数据质量基础
  D2  Skill-Privacy-Safe-Identity-Resolution   → 跨平台合规层

Wave 4（前沿扩展）
  A3  Skill-Web-Page-Change-Detection          → 实时监控增量
  E2  Skill-Synthetic-Data-Ecommerce           → 冷启动数据增强
  E3  Skill-Data-Provenance-Lineage            → AI 法规合规
  E4  Skill-Privacy-Preserving-Federated-Collection → 联邦采集
```

---

## 图谱预期影响

| 指标 | 当前 | 目标 |
|------|------|------|
| 新领域 | — | 22-数据采集工程（新建） |
| 新 Skill 数 | 0 | 13 |
| 新建 prerequisite 边 | 0 | ~20条（解锁下游断链） |
| 解锁的下游 Skill 数 | — | ~15 个（供应链/VOC/用户分析） |

---

## 论文快速索引（arXiv ID）

| Skill | 主干论文 | 补充论文 |
|-------|---------|---------|
| A1 LLM-Focused-Web-Crawling | 2603.29161 · 2602.24262 | 2506.16146 |
| A2 Adaptive-Crawl-Scheduling | 2602.11874 · 2502.02430 | 2506.16146 |
| A3 Web-Page-Change-Detection | 2605.29615 · 2603.00476 | — |
| B1 Document-Intelligence-Parsing | 2603.13032 · 2603.13398 | 2605.24973 |
| B2 Procurement-Email-Extraction | 2601.06164 · 2604.10633 | PROPOR 2026 |
| C1 Review-Dedup-Quality-Filter | 2606.03001 · 2601.07449 | 2602.21082 |
| C2 Fake-Review-Detection | 2602.12941 · 2603.08332 | 2605.20032 |
| D1 Clickstream-Persona-Pipeline | 2605.14205 · 2604.22762 | 2602.01443 |
| D2 Privacy-Safe-Identity-Resolution | 2604.19219 · 2606.01396 | 2604.16521 |
| E1 Ecommerce-Data-Quality | 2603.08612 · 2601.19750 | 2602.09329 |
| E2 Synthetic-Data-Ecommerce | 2602.23620 · 2602.07298 | 2606.00282 |
| E3 Data-Provenance-Lineage | 2604.10480 · 2601.14311 | 2603.17884 |
| E4 Privacy-Preserving-Federated | Informatica·2604.14833 | 2603.28329 |
