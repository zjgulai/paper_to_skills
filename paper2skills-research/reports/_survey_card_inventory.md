# paper2skills-vault 146 张 Skill 卡片全量清点

> 生成方式：`python3` 批量解析 frontmatter + 抽取每卡「② 母婴出海应用案例 / 2. 业务应用」段；
> 业务场景一句话为人工阅读②段后压缩至 ≤30 汉字，未读到的字段记 `—`。

## 一、总量与域分布

- **Skill 卡片总数：146**（`paper2skills-vault/**/Skill-*.md`，16 个域目录；另存在 `00-项目管理/` 目录但**不含** Skill 卡）

| 域目录 | 卡片数 |
|---|---|
| `00-电商Agent` | 2 |
| `01-因果推断` | 6 |
| `02-A_B实验` | 6 |
| `03-时间序列` | 7 |
| `04-供应链` | 7 |
| `05-推荐系统` | 8 |
| `06-增长模型` | 11 |
| `07-NLP-VOC` | 42 |
| `08-知识图谱` | 9 |
| `09-DataAgent-LLM` | 6 |
| `10-MAS` | 14 |
| `12-ML基础` | 1 |
| `13-广告分析` | 5 |
| `14-用户分析` | 3 |
| `15-营销投放分析` | 2 |
| `16-智能体工程` | 17 |
| **合计** | **146** |

## 二、frontmatter 完整度（v2 口径）

| 字段 | 有值 | 缺失 | 覆盖率 |
|---|---|---|---|
| `paper_id`（arXiv/DOI） | 90 | 56 | 61.6% |
| `venue` | 33 | 113 | 22.6% |
| `venue_tier` | 25 | 121 | 17.1% |
| `evidence_grade` | 21 | 125 | 14.4% |
| `evidence_basis` | 127 | 19 | 87.0% |

- 五字段**全齐**的卡：**3 张**
- `evidence_basis: author-practice`（设计上无论文来源）：**48 张**
- `evidence_basis` 分布：`paper-verbatim` 69 / `author-practice` 48 / `（缺失）` 19 / `paper-traceable` 9 / `mixed` 1

> ⚠ 缺失率最高的三项是 `venue`（113 缺）、`venue_tier`（121 缺）、`evidence_grade`（125 缺）——
> 其中 `author-practice` 的 48 张按设计豁免，**真实欠账**约为 `venue` 65 / `venue_tier` 73 / `evidence_grade` 77。

## 三、代码块与能力域分布

- 含 ` ```python ` 代码块：**98/146**（67.1%）
- 不含代码块：**48 张**

### 16 个能力域标签分布（一卡可命中 1–2 个，故合计 > 146）

| 能力域 | 卡数 | 占比(以146为分母) |
|---|---|---|
| Agent/LLM工程 | 43 | 29.5% |
| 客服/VOC/体验 | 32 | 21.9% |
| 数据工程/口径/质量 | 23 | 15.8% |
| 用户/CRM/留存 | 23 | 15.8% |
| 广告/营销投放 | 16 | 11.0% |
| 因果推断/实验 | 15 | 10.3% |
| 推荐/排序 | 13 | 8.9% |
| 供应链/库存/履约 | 10 | 6.8% |
| 知识图谱/知识工程 | 10 | 6.8% |
| 搜索/流量/LISTING | 9 | 6.2% |
| 定价/促销 | 9 | 6.2% |
| 预测/Forecasting | 8 | 5.5% |
| 组织/流程/治理 | 6 | 4.1% |
| 内容/视觉/创意 | 3 | 2.1% |
| 财务/对账/成本 | 3 | 2.1% |
| 风控/合规/账号 | 2 | 1.4% |
| **合计命中** | **225** | — |

按规格给定顺序：

| # | 能力域 | 卡数 |
|---|---|---|
| 1 | 预测/Forecasting | 8 |
| 2 | 因果推断/实验 | 15 |
| 3 | 推荐/排序 | 13 |
| 4 | 定价/促销 | 9 |
| 5 | 广告/营销投放 | 16 |
| 6 | 供应链/库存/履约 | 10 |
| 7 | 用户/CRM/留存 | 23 |
| 8 | 内容/视觉/创意 | 3 |
| 9 | 搜索/流量/LISTING | 9 |
| 10 | 风控/合规/账号 | 2 |
| 11 | 客服/VOC/体验 | 32 |
| 12 | 财务/对账/成本 | 3 |
| 13 | 数据工程/口径/质量 | 23 |
| 14 | Agent/LLM工程 | 43 |
| 15 | 知识图谱/知识工程 | 10 |
| 16 | 组织/流程/治理 | 6 |

## 四、全量清单

| # | domain | card | paper_id | venue | venue_tier | evidence_grade | 业务场景一句话 | py | 能力域 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `00-电商Agent` | Agentic-Catalog-Enrichment | 2608.20844 | arXiv preprint | preprint | A | 把在售SKU月龄材质认证补进Listing结构化属性 | yes | 搜索/流量/LISTING / Agent/LLM工程 |
| 2 | `00-电商Agent` | Live-Catalog-Conversational-Rec | 2608.27006 | RecSys 2026 (Demo) | demo | A | 客服导购机器人目录增量对账，不再答已下架商品 | yes | 客服/VOC/体验 / 推荐/排序 |
| 3 | `01-因果推断` | Causal-Discovery-PC-Algorithm | — | — | — | — | 识别广告/价格/促销对周销量的因果驱动结构 | no | 因果推断/实验 |
| 4 | `01-因果推断` | DiD-Difference-in-Differences | — | — | — | — | 评估关税加征对跨境销量的政策冲击 | yes | 因果推断/实验 |
| 5 | `01-因果推断` | IV-Instrumental-Variables | — | — | — | — | 用工具变量估准价格弹性，指导定价 | yes | 定价/促销 / 因果推断/实验 |
| 6 | `01-因果推断` | Intelligent-Attribution-Causal-Forest | — | — | — | — | 多市场广告归因，避免把相关当因果 | yes | 广告/营销投放 / 因果推断/实验 |
| 7 | `01-因果推断` | Mediation-Causal-Mechanism-Analysis | — | — | — | — | 拆解推荐算法更新对转化提升的因果路径 | yes | 因果推断/实验 / 推荐/排序 |
| 8 | `01-因果推断` | Uplift-Modeling | — | — | — | — | 广告投放增量归因，优化人群与预算 | yes | 广告/营销投放 / 因果推断/实验 |
| 9 | `02-A_B实验` | AB-Experimental-Design | — | CIKM 2023 | CCF-B | — | 详情页主图A/B实验设计与显著性判定 | no | 因果推断/实验 |
| 10 | `02-A_B实验` | AB-Test-Result-Interpretation | — | — | — | — | 实验结果解读，判断改版是否真有效 | yes | 因果推断/实验 |
| 11 | `02-A_B实验` | Multi-Armed-Bandit | — | — | — | — | 广告素材多臂老虎机，自动择优投放 | yes | 广告/营销投放 / 因果推断/实验 |
| 12 | `02-A_B实验` | Persona-Based-AB-Simulation | 2609.01038 | EMNLP 2026 Industry Track | top | A | 旺季前详情页候选改版的事前粗筛 | yes | 因果推断/实验 / 内容/视觉/创意 |
| 13 | `02-A_B实验` | Power-Analysis-Sample-Size | — | — | — | — | 改版实验的样本量与检验力测算 | yes | 因果推断/实验 |
| 14 | `02-A_B实验` | Thompson-Sampling-MAB | — | — | — | — | 首页Banner用汤普森采样自动择优投放 | yes | 广告/营销投放 / 因果推断/实验 |
| 15 | `03-时间序列` | Decision-Conditioned-Forecasting | 2608.25871 | KDD 2026 | CCF-A | A | 黑五前六周定FBA备货量（三档排期） | yes | 预测/Forecasting / 供应链/库存/履约 |
| 16 | `03-时间序列` | Intelligent-Prediction-Doubly-Robust | — | — | — | — | 促销活动效果的双稳健智能预测 | yes | 定价/促销 / 预测/Forecasting |
| 17 | `03-时间序列` | Prophet-Forecasting | — | — | — | — | 黑五销量预测，提前12周向供应商备货 | yes | 预测/Forecasting / 供应链/库存/履约 |
| 18 | `03-时间序列` | Shipping-Cost-Estimation | 2607.16230 | arXiv preprint | preprint | A | 算分区加权运费，定包邮门槛与价格下限 | yes | 供应链/库存/履约 / 定价/促销 |
| 19 | `03-时间序列` | Temporal-Fusion-Transformer | — | — | — | — | 数百SKU未来7-30天销量预测 | no | 预测/Forecasting |
| 20 | `03-时间序列` | Time-Series-Anomaly-Detection | — | — | — | — | 日订单量异常监控与告警 | yes | 预测/Forecasting / 数据工程/口径/质量 |
| 21 | `03-时间序列` | Time-Series-Forecasting | — | — | — | — | 周销量预测，指导海外仓补货 | yes | 预测/Forecasting / 供应链/库存/履约 |
| 22 | `04-供应链` | Demand-Forecasting-Supply-Chain | — | — | — | — | 奶粉SKU-仓库组合的月度需求预测 | yes | 供应链/库存/履约 / 预测/Forecasting |
| 23 | `04-供应链` | Monodense-单品价格弹性估计 | 2603.29261 | — | — | — | 单品价格弹性估计支撑动态定价 | no | 定价/促销 |
| 24 | `04-供应链` | Multi-Echelon-Inventory | — | — | — | — | 国内仓-海外仓多级库存备货策略优化 | yes | 供应链/库存/履约 |
| 25 | `04-供应链` | Multi-Warehouse-Allocation-LLM | 2606.29366 | arXiv preprint | preprint | A | 头程到港后在FBA两仓与海外仓间分货 | yes | 供应链/库存/履约 |
| 26 | `04-供应链` | Safety-Stock-Replenishment | — | — | — | — | 奶粉SKU安全库存计算（海运6周交期） | yes | 供应链/库存/履约 |
| 27 | `04-供应链` | Supply-Network-Simulation | 2607.09745 | Winter Simulation Conference 2026 | CCF-B | A | 平台店与独立站双海外仓要不要互备代发 | yes | 供应链/库存/履约 |
| 28 | `04-供应链` | Two-Echelon-Inventory-DRL | 2204.09603 | AI4M 2023 (ECML PKDD Workshop) | workshop | C | 区域仓两级库存备货优化（DRL） | yes | 供应链/库存/履约 |
| 29 | `05-推荐系统` | Cold-Start-Meta-Learning-PAM | 2608.10240 | CIKM 2026 | CCF-B | A | 季度30+新品快速冷启动推荐 | yes | 推荐/排序 |
| 30 | `05-推荐系统` | Deep-Learning-Recommendation-HI | 2009.12969 | — | — | — | 首页个性化推荐去同质化，扶长尾新品 | yes | 推荐/排序 |
| 31 | `05-推荐系统` | Diversity-Reranking-SMMR | — | — | — | — | 首页猜你喜欢列表多样性重排 | yes | 推荐/排序 |
| 32 | `05-推荐系统` | Explainable-Recommendation | — | — | — | — | 给推荐结果加解释以提升用户信任 | yes | 推荐/排序 |
| 33 | `05-推荐系统` | Matrix-Factorization | — | — | — | — | 吸奶器配件复购「下次买什么」推荐 | yes | 推荐/排序 / 用户/CRM/留存 |
| 34 | `05-推荐系统` | NeuralNDCG-Learning-to-Rank | 2102.07831 | — | — | — | 站内搜索baby bottle结果排序优化 | yes | 搜索/流量/LISTING / 推荐/排序 |
| 35 | `05-推荐系统` | Semantic-ID-Retrieval-RPG | — | — | — | — | 多语言商品跨语言语义检索 | yes | 搜索/流量/LISTING |
| 36 | `05-推荐系统` | Session-Based-Recommendation-SR-GNN | 1811.00855 | — | — | — | 匿名用户会话内跨品类连带推荐 | yes | 推荐/排序 |
| 37 | `06-增长模型` | Cold-Start-Product-Recommendation | 2402.09176 | — | — | — | 零销量零评价新品吸奶器冷启动推荐 | yes | 推荐/排序 |
| 38 | `06-增长模型` | Customer-Churn-Prediction | — | — | — | — | 配件复购用户流失预警与名单 | yes | 用户/CRM/留存 |
| 39 | `06-增长模型` | Customer-Journey-Prototype | 2505.11086 | — | — | — | App/小程序/门店全渠道旅程与流失点诊断 | no | 用户/CRM/留存 |
| 40 | `06-增长模型` | DQN-Purchase-Prediction | 2506.17543 | — | — | — | 实时购买意向评分，选时机发券 | yes | 用户/CRM/留存 / 广告/营销投放 |
| 41 | `06-增长模型` | Deep-Learning-Churn-Prediction | — | — | — | — | 订阅制母婴用品盒流失预警 | no | 用户/CRM/留存 |
| 42 | `06-增长模型` | LTV-Prediction-ZILN | — | — | — | — | 新客LTV预测，指导广告出价与预算 | yes | 用户/CRM/留存 / 广告/营销投放 |
| 43 | `06-增长模型` | New-Product-Opportunity-Mining | 2405.19456 | — | — | — | 新品值不值得投、先投哪个市场 | yes | 预测/Forecasting |
| 44 | `06-增长模型` | RFM-Customer-Segmentation | — | — | — | — | 10万注册用户RFM分群做差异化推送 | yes | 用户/CRM/留存 |
| 45 | `06-增长模型` | Seasonal-Aligned-Churn-Label | 2608.18174 | arXiv preprint | preprint | A | 季度复购衰退预警清单重标（口径修正） | yes | 用户/CRM/留存 / 数据工程/口径/质量 |
| 46 | `06-增长模型` | Uplift-Churn-Prediction | 2312.07206 | — | — | — | 给高危流失用户精算发券（uplift分层） | no | 用户/CRM/留存 / 因果推断/实验 |
| 47 | `06-增长模型` | User-Lifecycle-STAN | 2306.12232 | — | — | — | 按生命周期阶段标签做差异化触达 | no | 用户/CRM/留存 |
| 48 | `07-NLP-VOC` | ABSA-BERT-MoE | — | — | — | — | 大规模评论实时方面情感分析降本 | yes | 客服/VOC/体验 |
| 49 | `07-NLP-VOC` | AGRS-属性引导评论摘要 | — | — | — | — | 消毒器双平台季度评论摘要报告 | yes | 客服/VOC/体验 |
| 50 | `07-NLP-VOC` | ALCHEmist-Weak-Supervision | 2407.11004 | NeurIPS 2024 | — | — | 新痛点标签的可审计标注规则快速生产 | yes | 数据工程/口径/质量 |
| 51 | `07-NLP-VOC` | Active-Learning-Annotation | — | — | — | — | 新标签冷启动的主动学习标注省钱 | no | 数据工程/口径/质量 |
| 52 | `07-NLP-VOC` | AdaNEN-Streaming-Classifier | 10.1145/3639054 | ACM TKDD 2024 | CCF-A | — | 季节性反馈漂移下分类器自动适配 | yes | 客服/VOC/体验 / 数据工程/口径/质量 |
| 53 | `07-NLP-VOC` | Aspect-Based-Sentiment-Analysis | — | — | — | — | 产品评论方面级情感洞察自动化 | no | 客服/VOC/体验 |
| 54 | `07-NLP-VOC` | AutoTag-SelfEvolving-Label-System | 2405.07195 | arXiv preprint (Amazon) | — | — | 多平台多语言评价标签统一追踪 | yes | 客服/VOC/体验 / 数据工程/口径/质量 |
| 55 | `07-NLP-VOC` | BERT-MoE高效方面情感分析 | 2602.12778 | arXiv preprint | — | — | 多语言评论情感分析的低成本方案 | yes | 客服/VOC/体验 |
| 56 | `07-NLP-VOC` | BERT-SRL-Event-Frame-Extraction | 1904.05255 | — | — | — | 从评论抽取谁-做了什么-针对什么事件帧 | no | 客服/VOC/体验 / 知识图谱/知识工程 |
| 57 | `07-NLP-VOC` | Behavioral-Intent-Tree-Parsing | 2408.05353 | arXiv preprint (Netflix) | — | — | 独立站行为路径拆解购买阶段意图 | yes | 用户/CRM/留存 |
| 58 | `07-NLP-VOC` | CSK-Customer-Sentiment-Clustering | — | — | — | — | 评论情感驱动用户分群，定触达动作 | no | 用户/CRM/留存 / 客服/VOC/体验 |
| 59 | `07-NLP-VOC` | CrossLingual-Semantic-Alignment | 2206.07587 | ACL 2023 | — | — | 多市场商品属性跨语言统一理解 | yes | 数据工程/口径/质量 / 搜索/流量/LISTING |
| 60 | `07-NLP-VOC` | CrossLingual-Sentiment-Transfer | 2508.09515 | ACL 2025 (底本未声明 venue) | — | — | 东南亚/中东无标注语种评论情感上线 | yes | 客服/VOC/体验 |
| 61 | `07-NLP-VOC` | Dialogue-to-Action-Graph | 2312.04668 | arXiv preprint (LG AI Research / University of Michigan) | — | — | 客服工单对话转行动图，标准化处理流程 | yes | 客服/VOC/体验 |
| 62 | `07-NLP-VOC` | GPLR-人群标签生成 | 2504.17304 | SIGIR 2025 | — | — | 把用户向量转成可执行营销人群标签 | yes | 用户/CRM/留存 / 广告/营销投放 |
| 63 | `07-NLP-VOC` | InstructUIE-Unified-Information-Extraction | 2304.08085 | — | — | — | 评论多任务统一抽取（实体/关系/事件） | no | 数据工程/口径/质量 |
| 64 | `07-NLP-VOC` | LLM-Personalized-Marketing-Copy-Generation | 2505.23809 | — | — | — | 同产品不同画像生成差异化详情页文案 | yes | 内容/视觉/创意 / 广告/营销投放 |
| 65 | `07-NLP-VOC` | MAA-行动建议生成 | — | — | — | — | 跨市场评论提炼差异化改进方向 | yes | 客服/VOC/体验 |
| 66 | `07-NLP-VOC` | MAS-Consumer-Behavior-Simulation | 2510.18155 | — | — | — | 黑五促销策略用虚拟消费者预演 | yes | Agent/LLM工程 / 定价/促销 |
| 67 | `07-NLP-VOC` | MAS-MARL-Dynamic-Pricing | 2507.02698 | — | — | — | 三市场动态定价（美英德差异） | yes | 定价/促销 |
| 68 | `07-NLP-VOC` | MAS-Multi-Objective-Recommendation | 2512.24325 | — | — | — | 首页推荐在CTR/利润/多样性间多目标权衡 | yes | 推荐/排序 / Agent/LLM工程 |
| 69 | `07-NLP-VOC` | MAS-VOC-Data-Analyst | 2402.01386 | — | — | — | 35万条Amazon评论自动主题分析 | yes | 客服/VOC/体验 / Agent/LLM工程 |
| 70 | `07-NLP-VOC` | NPS-Driver-Analysis | 2510.16551 | — | — | — | 新品NPS驱动诊断，定改进资源投向 | yes | 客服/VOC/体验 |
| 71 | `07-NLP-VOC` | OfflineRL-触达时机优化 | 2202.03867 | — | — | — | 新妈妈推送时机优化（离线RL） | yes | 用户/CRM/留存 / 广告/营销投放 |
| 72 | `07-NLP-VOC` | OpenWorld-Class-Incremental-Learning | — | ACL 2025 | CCF-A | — | 新品上市后标签体系增量扩展 | yes | 数据工程/口径/质量 |
| 73 | `07-NLP-VOC` | PERSONABOT-RAG用户画像生成 | 2505.17156 | — | — | — | 从用户评论生成个体画像做推荐 | no | 用户/CRM/留存 |
| 74 | `07-NLP-VOC` | Product-Attribute-Graph-Parsing | 2410.21237 | — | — | — | 竞品属性对比支撑产品定位决策 | yes | 知识图谱/知识工程 / 搜索/流量/LISTING |
| 75 | `07-NLP-VOC` | REVISION-无点击意图挖掘 | 2510.22739 | — | — | — | 无点击搜索查询的隐含意图拆解与承接 | no | 搜索/流量/LISTING / 用户/CRM/留存 |
| 76 | `07-NLP-VOC` | Review-Quality-Scoring | 2510.08081 | — | — | — | 采集评论的质量清洗与去噪 | yes | 数据工程/口径/质量 |
| 77 | `07-NLP-VOC` | Self-Improving-LLM-Agent-Pipeline | 2408.06292 | — | — | — | 商品文案按CTR自迭代优化 | yes | 内容/视觉/创意 / Agent/LLM工程 |
| 78 | `07-NLP-VOC` | Semantic-Blueprint-Compiler | 2307.09702 | — | — | — | 抽取结果Schema化编译以统一口径 | no | 数据工程/口径/质量 / 知识图谱/知识工程 |
| 79 | `07-NLP-VOC` | SoMeR-多视角用户表示 | 2405.05275 | — | — | — | 多源数据融合生成统一用户表示 | yes | 用户/CRM/留存 |
| 80 | `07-NLP-VOC` | Spiral-of-Silence-沉默少数派挖掘 | 2502.00952 | — | — | — | 好评掩盖下的少数派真实问题挖掘 | yes | 客服/VOC/体验 |
| 81 | `07-NLP-VOC` | StaR-观点语句排序 | — | — | — | — | 暖奶器跨市场观点提取与排序 | yes | 客服/VOC/体验 |
| 82 | `07-NLP-VOC` | TJAP-跨市场品类组合定价 | 2603.18114 | — | — | — | 德国站新品上市选品与组合定价 | yes | 定价/促销 |
| 83 | `07-NLP-VOC` | TSCAN-上下文感知挽回策略 | 2504.18881 | — | — | — | 流失挽回策略的上下文感知选择 | yes | 用户/CRM/留存 |
| 84 | `07-NLP-VOC` | TaxoAdapt-Taxonomy-Evolution | 2506.10737 | — | — | — | 季节新品驱动的标签体系自演化 | no | 数据工程/口径/质量 |
| 85 | `07-NLP-VOC` | TopicImpact-观点单元画像抽取 | 2507.13392 | — | — | — | 评论观点单元细粒度画像抽取 | no | 客服/VOC/体验 |
| 86 | `07-NLP-VOC` | VOC-Proxy-NPS-AIPL-统一萃取引擎 | — | — | — | — | 一条评论萃取AIPL等全部标签 | yes | 客服/VOC/体验 / 用户/CRM/留存 |
| 87 | `07-NLP-VOC` | VOC-Semantic-Blueprint | — | ACL 2023 | CCF-A | — | 评论结构化反馈支撑产品改进决策 | yes | 客服/VOC/体验 |
| 88 | `07-NLP-VOC` | iReFeed-需求优先级排序 | 2603.28677 | — | — | — | 季度产品功能需求优先级排序 | yes | 客服/VOC/体验 / 组织/流程/治理 |
| 89 | `07-NLP-VOC` | 大规模消费者评论方面情感分析 | — | — | — | — | 海量消费者评论方面情感自动分析 | no | 客服/VOC/体验 |
| 90 | `08-知识图谱` | Dense-Retrieval-Ecommerce-Semantic-Search | 2601.16492 | — | — | — | 母婴商品语义搜索取代关键词匹配 | yes | 搜索/流量/LISTING |
| 91 | `08-知识图谱` | GraphRAG-Knowledge-Enhanced-Retrieval | 2404.16130 | arXiv preprint | preprint | C | 客服问答用GraphRAG检索增强 | yes | 客服/VOC/体验 / 知识图谱/知识工程 |
| 92 | `08-知识图谱` | HGCN-Hyperbolic-Graph-Convolutional-Networks | — | — | — | — | 产品品类层次树嵌入（双曲空间） | no | 知识图谱/知识工程 / 推荐/排序 |
| 93 | `08-知识图谱` | HGT-Heterogeneous-Graph-Transformer | — | — | — | — | 跨语言商品属性对齐与品类推断 | no | 知识图谱/知识工程 / 搜索/流量/LISTING |
| 94 | `08-知识图谱` | KG-Auto-Construction-Agent-Driven | 2511.11017 | — | — | — | 从商品描述自动构建商品知识图谱 | yes | 知识图谱/知识工程 / Agent/LLM工程 |
| 95 | `08-知识图谱` | KG-Relation-Completion-CBLiP | — | — | — | — | 产品知识图谱关系补全 | yes | 知识图谱/知识工程 |
| 96 | `08-知识图谱` | KGQA-Question-Answering | — | — | — | — | 客服知识库图谱问答 | yes | 客服/VOC/体验 / 知识图谱/知识工程 |
| 97 | `08-知识图谱` | Knowledge-Graph-for-Skills-Management | — | — | — | — | 数据科学团队内部技能推荐（非电商） | yes | 组织/流程/治理 / 知识图谱/知识工程 |
| 98 | `08-知识图谱` | Multilingual-NER-Universal-v2 | — | — | — | — | 多语言评论实体抽取（美/德/日站） | yes | 客服/VOC/体验 / 数据工程/口径/质量 |
| 99 | `09-DataAgent-LLM` | Argos-Agentic-Anomaly-Detection | 2501.14170 | — | — | — | 四平台销量/广告ROI/退货率异常监控 | yes | 数据工程/口径/质量 / Agent/LLM工程 |
| 100 | `09-DataAgent-LLM` | Data-to-Dashboard-Multi-Agent-Visualization | 2505.23695 | — | — | — | 三平台周报仪表板自动生成 | yes | 数据工程/口径/质量 / Agent/LLM工程 |
| 101 | `09-DataAgent-LLM` | DeepAnalyze-Autonomous-Data-Science-Agent | 2510.16872 | — | — | — | 多平台销售数据自动分析报告 | yes | 数据工程/口径/质量 / Agent/LLM工程 |
| 102 | `09-DataAgent-LLM` | Root-Cause-Analysis-Agent | — | — | — | — | 转化率骤降的自动根因定位 | yes | 数据工程/口径/质量 / Agent/LLM工程 |
| 103 | `09-DataAgent-LLM` | SQL-Agent-Access-Control | 2607.22115 | arXiv preprint | preprint | A | 给LLM取数Agent发角色权限许可证 | yes | 风控/合规/账号 / Agent/LLM工程 |
| 104 | `09-DataAgent-LLM` | SQL-Agent-Text-to-SQL | — | — | — | — | 运营自助用自然语言取数 | yes | 数据工程/口径/质量 / Agent/LLM工程 |
| 105 | `10-MAS` | Agent-Memory-Learning | 2310.08560 | — | — | — | 长期用户对话Agent记住成长阶段偏好 | no | Agent/LLM工程 / 客服/VOC/体验 |
| 106 | `10-MAS` | AutoGen-Multi-Agent-Conversation | 2308.08155 | — | — | — | VOC分析多Agent协作流水线 | no | Agent/LLM工程 / 客服/VOC/体验 |
| 107 | `10-MAS` | CAMEL-Role-Playing-Agents | 2303.17760 | — | — | — | VOC评论分析的角色扮演协作 | no | Agent/LLM工程 / 客服/VOC/体验 |
| 108 | `10-MAS` | MAS-Orchestrator | — | — | — | — | 全品类VOC流水线编排（8并行+2串行） | no | Agent/LLM工程 / 客服/VOC/体验 |
| 109 | `10-MAS` | MetaGPT-SOP-Driven-Collaboration | 2308.00352 | — | — | — | VOC分析标准化SOP流水线 | no | Agent/LLM工程 / 组织/流程/治理 |
| 110 | `10-MAS` | Multi-Agent-Debate | 2305.19118 | — | — | — | 评论情感标注歧义的多Agent仲裁 | no | 数据工程/口径/质量 / Agent/LLM工程 |
| 111 | `10-MAS` | ReAct-Reasoning-Acting | 2210.03629 | — | — | — | 竞品情报多源信息收集Agent | no | Agent/LLM工程 |
| 112 | `10-MAS` | Reflexion-Self-Improvement | 2303.11366 | — | — | — | VOC打标新类型评论的自我改进 | no | 数据工程/口径/质量 / Agent/LLM工程 |
| 113 | `10-MAS` | Self-Improving-Agent-Feedback-Loop | 2303.17651 | — | — | — | VOC分析Agent的持续进化反馈环 | no | Agent/LLM工程 |
| 114 | `10-MAS` | Skill-Registry-Dynamic-Loading | — | — | — | — | 按任务动态匹配与加载技能 | no | Agent/LLM工程 |
| 115 | `10-MAS` | Subagent-Decomposition | — | — | — | — | 全品类VOC周报的Agent任务拆分 | no | Agent/LLM工程 / 客服/VOC/体验 |
| 116 | `10-MAS` | Tree-of-Thoughts-Planning | 2305.10601 | — | — | — | VOC标签体系设计的策略搜索 | no | Agent/LLM工程 / 数据工程/口径/质量 |
| 117 | `10-MAS` | Multi-Agent-Collaboration-Tax | 2608.22152 | EMNLP 2026 | top | A | 选品分析要不要拆成多agent的成本核算 | yes | Agent/LLM工程 / 组织/流程/治理 |
| 118 | `10-MAS` | Routed-Graph-Handoff | 2608.25277 | EMNLP 2026 | top | A | 选品→合规→定价三段委派交接 | yes | Agent/LLM工程 / 风控/合规/账号 |
| 119 | `12-ML基础` | Feature-Engineering | 2608.09162 | arXiv preprint | preprint | A | 流失预测的用户特征工程 | yes | 数据工程/口径/质量 / 用户/CRM/留存 |
| 120 | `13-广告分析` | Ad-Attribution-Modeling | — | — | — | — | 50万月广告预算在三渠道重分配 | yes | 广告/营销投放 |
| 121 | `13-广告分析` | Cannibalization-Corrected-Attribution | 2606.26690 | ADKDD 2026 | workshop | A | 校正品牌词广告抢自然搜索的功劳蚕食 | yes | 广告/营销投放 / 搜索/流量/LISTING |
| 122 | `13-广告分析` | Causal-Budget-Allocation | 2608.10182 | arXiv preprint | preprint | A | 站内广告位与站外种草的预算切分 | yes | 广告/营销投放 |
| 123 | `13-广告分析` | Funnel-Causal-Coupon-Allocation | 2608.11675 | CIKM 2026 | CCF-B | A | 旺季Coupon档位锁档与预算切分 | yes | 定价/促销 / 广告/营销投放 |
| 124 | `13-广告分析` | ROAS-Budget-Optimization | — | — | — | — | 按ROAS重分配三渠道预算 | yes | 广告/营销投放 |
| 125 | `14-用户分析` | Cohort-Retention-Analysis | — | — | — | — | 新客D7/D30留存诊断与归因 | yes | 用户/CRM/留存 |
| 126 | `14-用户分析` | Incrementality-Measurement | 2607.09608 | arXiv preprint | preprint | A | 跨平台增量测量，纠正独立站ROAS低估 | yes | 广告/营销投放 / 因果推断/实验 |
| 127 | `14-用户分析` | User-Funnel-Analysis | — | — | — | — | 详情页UV到支付的漏斗流失定位 | yes | 用户/CRM/留存 |
| 128 | `15-营销投放分析` | Marketing-Mix-Modeling | — | — | — | — | 年度600万广告预算MMM重规划 | yes | 广告/营销投放 |
| 129 | `15-营销投放分析` | Promotion-Effectiveness | — | — | — | — | 首单折扣券的真实增量效果评估 | yes | 定价/促销 / 因果推断/实验 |
| 130 | `16-智能体工程` | Active-Context-Pruning | 2601.07190 | — | — | — | 跨境客服长会话压缩以降本 | no | Agent/LLM工程 / 财务/对账/成本 |
| 131 | `16-智能体工程` | Agent-Stage-Evaluation | 2601.02752 | — | — | — | 客服Agent上线后能力体检定位短板 | no | Agent/LLM工程 |
| 132 | `16-智能体工程` | Agentic-Memory-Management | — | — | — | — | 母婴用户0-3岁LTM/STM记忆协同 | no | Agent/LLM工程 / 用户/CRM/留存 |
| 133 | `16-智能体工程` | Auto-Skill-Synthesis | 2604.08618 | SIGIR 2026 (Industry Track) | CCF-A | A | 从历史客服工单自动萃取多语言Skill库 | no | Agent/LLM工程 / 客服/VOC/体验 |
| 134 | `16-智能体工程` | Co-Evolutionary-Skill-Verification | 2604.01687 | — | — | — | 客服Skill自动萃取替代人工SOP撰写 | no | Agent/LLM工程 / 组织/流程/治理 |
| 135 | `16-智能体工程` | Context-Compression | 2510.00615 | — | — | — | 客服长对话VOC分析的上下文压缩 | no | Agent/LLM工程 / 客服/VOC/体验 |
| 136 | `16-智能体工程` | Long-Term-Preference-Memory | 2603.14864 | — | — | — | 长周期偏好建模支撑精准复购推荐 | no | Agent/LLM工程 / 推荐/排序 |
| 137 | `16-智能体工程` | MCP-A2A-Protocol-Stack | 2601.13671 | — | — | — | 跨境客服MAS的MCP+A2A协议栈 | no | Agent/LLM工程 |
| 138 | `16-智能体工程` | MCP-Tool-Use-Benchmark | 2512.24565 | — | — | — | 多个客服Agent的tool use能力评估 | no | Agent/LLM工程 |
| 139 | `16-智能体工程` | Memory-as-Action | 2510.12635 | — | — | — | 多目标客服Agent训练替代prompt方案 | no | Agent/LLM工程 |
| 140 | `16-智能体工程` | Open-Source-Tool-Use-Model | 2508.18255 | — | — | — | 客服Agent开源基座选型以替代闭源API | no | Agent/LLM工程 / 财务/对账/成本 |
| 141 | `16-智能体工程` | Orchestration-Trace-RL | 2605.02801 | — | — | — | 客服多agent编排器的RL训练 | no | Agent/LLM工程 |
| 142 | `16-智能体工程` | SLM-Tool-Calling-Optimization | 2512.15943 | — | — | — | 简单工单用SLM替代大模型降本 | yes | Agent/LLM工程 / 财务/对账/成本 |
| 143 | `16-智能体工程` | Skill-Lifecycle-Design | 2602.20867 | — | — | — | 用四元组契约重构本项目Skill库质量 | no | 组织/流程/治理 / Agent/LLM工程 |
| 144 | `16-智能体工程` | Stateful-Skill-Runtime | 2608.26263 | EMNLP | top | A | 售后长工单用状态机把prompt锁成常数 | yes | Agent/LLM工程 |
| 145 | `16-智能体工程` | Task-Adaptive-Topology | 2602.16873 | — | — | — | 3k+日工单自适应路由到合适agent | no | Agent/LLM工程 / 客服/VOC/体验 |
| 146 | `16-智能体工程` | Tool-Description-Audit | 2602.14878 | — | — | — | 内部MCP Tool描述质量审核 | no | Agent/LLM工程 / 数据工程/口径/质量 |

## 五、结构性异常与说明

### 5.1 frontmatter `module:` 与所在目录不一致

| # | 所在目录 | 卡片 | `module:` 实际值 |
|---|---|---|---|
| 34 | `05-推荐系统` | Skill-NeuralNDCG-Learning-to-Rank | `recommendation` |
| 99 | `09-DataAgent-LLM` | Skill-Argos-Agentic-Anomaly-Detection | `data-agent-llm` |
| 100 | `09-DataAgent-LLM` | Skill-Data-to-Dashboard-Multi-Agent-Visualization | `data-agent-llm` |
| 101 | `09-DataAgent-LLM` | Skill-DeepAnalyze-Autonomous-Data-Science-Agent | `data-agent-llm` |

> 共 4 张：`05-推荐系统` 有 1 张写成英文 `recommendation`；`09-DataAgent-LLM` 有 3 张写成 `data-agent-llm`（同域另 3 张写中文域名）——同域内命名不统一。

### 5.2 其他结构事实

- `07-NLP-VOC` 的 42 张卡**不在域根目录**，而在二级子目录 `07-NLP-VOC/00-知识库-Skill卡片/`（其余 15 个域均为域根平铺）。
- `paper2skills-vault/00-项目管理/` 存在但**不含任何** `Skill-*.md`，故实际参与清点的域为 16 个。
- 含 ` ```python ` 的 **98** 张中，`09-DataAgent-LLM/Skill-Argos-Agentic-Anomaly-Detection` 用的是 **4 反引号**围栏（````` ````python `````），按行首三反引号计数会漏掉它 —— 这解释了与仓库文档「97/146」的 1 张差异。

### 5.3 逐卡异常清单（18 条）

- **#7 `Mediation-Causal-Mechanism-Analysis`（01-因果推断）**：业务场景偏泛化（转化率提升归因），未落到母婴单品
- **#10 `AB-Test-Result-Interpretation`（02-A_B实验）**：业务场景为通用实验解读，无母婴具体设定
- **#13 `Power-Analysis-Sample-Size`（02-A_B实验）**：业务场景为通用样本量计算，无母婴具体设定
- **#19 `Temporal-Fusion-Transformer`（03-时间序列）**：业务场景泛化（“数百个SKU”），无母婴品类锚点
- **#43 `New-Product-Opportunity-Mining`（06-增长模型）**：场景为新品机会评估，非本域“增长模型”的核心动作，更接近选品/组合决策
- **#61 `Dialogue-to-Action-Graph`（07-NLP-VOC）**：工单流程，更接近客服/流程域；列入 NLP-VOC 域
- **#72 `OpenWorld-Class-Incremental-Learning`（07-NLP-VOC）**：无母婴场景，仅“新品上市后标签扩展”一句
- **#74 `Product-Attribute-Graph-Parsing`（07-NLP-VOC）**：竞品属性对比属竞争情报，与 07-NLP-VOC 域其余评论挖掘卡主题不同
- **#97 `Knowledge-Graph-for-Skills-Management`（08-知识图谱）**：⚠ 场景错位：落点是团队内部技能/学习路径推荐，而非对外经营决策（是本域唯一的“人”而非“货”场景）
- **#102 `Root-Cause-Analysis-Agent`（09-DataAgent-LLM）**：场景为通用“转化率骤降”，未锚定母婴/平台
- **#116 `Tree-of-Thoughts-Planning`（10-MAS）**：无母婴场景，为 VOC 标签体系设计的方法论演练
- **#121 `Cannibalization-Corrected-Attribution`（13-广告分析）**：正文无母婴关键词，场景为 Amazon 站内广告报表（通用）
- **#124 `ROAS-Budget-Optimization`（13-广告分析）**：场景为通用三渠道预算，无母婴锚点
- **#128 `Marketing-Mix-Modeling`（15-营销投放分析）**：场景为通用年度预算规划，无母婴锚点（母婴仅在背景出现1次）
- **#129 `Promotion-Effectiveness`（15-营销投放分析）**：场景为通用首单折扣评估，无母婴锚点
- **#138 `MCP-Tool-Use-Benchmark`（16-智能体工程）**：场景为通用客服Agent评测，母婴仅在背景出现1次
- **#143 `Skill-Lifecycle-Design`（16-智能体工程）**：⚠ 元卡：讲的是重构本项目自己的 Skill 库，不是母婴业务
- **#146 `Tool-Description-Audit`（16-智能体工程）**：场景为内部 MCP tool 质量审核，无母婴锚点
