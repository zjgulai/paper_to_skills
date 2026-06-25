# paper2skills 战略升级全量计划 — 2026

## 数据基线（2026-06-21）
- Skills: **874** | Domains: **25** | Edges: **16,107**
- D层决策型Skills: **~48个 (5%)** ← 核心结构缺陷
- Playbook引用率最低域: 12-ML基础(5%) / 10-MAS(8.8%) / 07-NLP-VOC(10%)
- Top Playbook Skills中缺D层对应: **12/20个 (60%)**
- 竞品无法复制算法护城河: **~15个**
- 商业化就绪度: **5.2/10**

## 核心升级目标
- Skills: 874 → **1100+**
- D层决策型Skills占比: 5% → **25%**
- Playbook引用率（全局）: 31% → **50%+**
- 算法护城河Skills: ~15 → **50+**
- 商业化就绪度: 5.2 → **7.5/10**

---

## Sprint A — 紧急窗口期（2周，2026-06-21 起）
> **时间窗口：CPSC eFiling 2026-07-08 强制执行**
> **市场价值：~15万中国母婴卖家，竞品零产品**

### A1：CPSC合规自动化 5个全新Skills（P0）

| Skill名 | 域 | 核心算法 | 业务价值 |
|---|---|---|---|
| Skill-CPSC-eFiling-Auto-Mapper | 21-合规决策 | NLP实体提取 + HTS码规则库匹配 | eFiling字段自动填充，节省20-40h/月 |
| Skill-HTS-Code-Risk-Classifier | 21-合规决策 | 多标签分类 + CPSC高风险清单匹配 | 上架前自动风险评级 |
| Skill-GCC-CPC-Document-Generator | 21-合规决策 | 模板填充 + 合规字段验证 | GCC/CPC文档5分钟生成 |
| Skill-Amazon-Compliance-Error-Resolver | 21-合规决策 | 错误码语义解析 + 修复建议生成 | 8572/8574/8591错误码自动处理 |
| Skill-GPSR-EU-Safety-Registry | 21-合规决策 | 欧盟GPSR规则图谱 + 合规检查 | 欧盟5995错误预防 |

### A2：发布「CPSC eFiling 72小时紧急自查手册」Playbook（P0）
- ID: `pb-cpsc-efiling-emergency`
- Steps: 产品风险扫描 → HTS码核查 → GCC/CPC文档补全 → eFiling提交清单
- 引用上述5个Skills
- 定位：2026-07-08前每个母婴卖家必须完成的清单

### A3：修复11个缺 roadmap_phase 的 Skills（P1）
缺失列表：
- Skill-QUBO-Ad-Budget-Allocation → phase2
- Skill-Epidemiological-Viral-Traffic-SIR → phase2
- Skill-AlphaFold-Bin-Packing → phase3
- Skill-Commodity-Futures-Cost-Baseline → phase1
- Skill-Navier-Stokes-Warehouse → phase3
- Skill-Continuous-NLP-SEO-Morphing → phase2
- Skill-Topological-Data-Analysis-Cross-Sell → phase2
- Skill-Counterfactual-Price-Elasticity → phase1
- Skill-GAN-Red-Team-Listing → phase2
- Skill-Crypto-Anomaly-Review-Fraud → phase2
- Skill-MonteCarlo-Tariff-Risk → phase1

**验收**：`grep -r "roadmap_phase" paper2skills-vault | wc -l` = 874

---

## Sprint B — 决策层重构（3周，结构性最高ROI）
> **核心目标：D层Skills 48 → 150，使Playbook从「手册」升级为「执行协议」**

### B1：为Top Playbook Skills补全D层决策触发器（12个，P0）

以下12个被Playbook高频引用但无D层对应的Skills，各补1个决策执行Skill：

| 分析型A层（已有） | 缺失D层决策器 | 决策逻辑 |
|---|---|---|
| Skill-Channel-Saturation-Curve(5x) | Skill-Channel-Budget-Reallocation-Trigger | 饱和度>80%→自动削减该渠道20%预算 |
| Skill-Price-Elasticity-Estimation(4x) | Skill-Elasticity-Based-Repricing-Gate | 弹性>1.5→触发降价5%测试 |
| Skill-Cohort-Retention-Analysis(3x) | Skill-Cohort-Churn-Intervention-Dispatcher | 30日留存<40%→触发挽回序列 |
| Skill-Lead-Time-Distribution-Risk-GenQOT(3x) | Skill-Lead-Time-Safety-Stock-Auto-Adjuster | 前置期P95>承诺×1.3→动态上调安全库存 |
| Skill-DARA-Agentic-MMM-Optimizer(3x) | Skill-MMM-Budget-Reallocation-Executor | MMM输出→预算调整API调用执行器 |
| Skill-Identified-Bayesian-MMM(3x) | Skill-Bayesian-MMM-Scenario-Action-Plan | 贝叶斯后验→生成Q+1预算分配方案 |
| Skill-Markdown-Optimization(3x) | Skill-Markdown-Schedule-Auto-Trigger | 库龄>45天+库存>目标×1.5→自动降价序列 |
| Skill-TikTok-Shop-Content-Attribution(2x) | Skill-Content-ROI-Budget-Shift-Trigger | 内容ROI<目标→自动减少该内容类型投放 |
| Skill-RFM-Customer-Segmentation(2x) | Skill-RFM-Segment-Campaign-Dispatcher | RFM分群→自动触发对应营销序列 |
| Skill-LTV-Prediction-ZILN(2x) | Skill-LTV-Acquisition-Budget-Gate | 预测LTV<CAC×3→停止该渠道新客投入 |
| Skill-Uplift-Churn-Prediction(2x) | Skill-Uplift-Intervention-Priority-Queue | Uplift分数→按ROI排序干预执行队列 |
| Skill-RFM-Customer-Segmentation(2x) → 补配 | Skill-High-Value-Customer-Alert-Action | RFM高价值客户流失风险→自动客服介入 |

### B2：低引用率域的业务转化专项（P1）

**07-NLP-VOC（引用率10% → 目标30%）**
NLP-VOC ↔ 增长模型桥梁只有2条——建5个桥接Skills：

| Skill名 | 桥接逻辑 |
|---|---|
| Skill-VOC-Churn-Signal-Extraction | 差评语义→流失预警信号 |
| Skill-Review-Sentiment-Growth-Trigger | 情感趋势跌破阈值→触发产品迭代流程 |
| Skill-VOC-New-Product-Gap-Scoring | 痛点频率×竞争密度→选品机会评分 |
| Skill-NPS-Proxy-Retention-Predictor | NPS代理指标→次月留存率预测 |
| Skill-Social-VOC-Viral-Potential-Score | 社媒UGC热度→爆品传播潜力评分 |

**10-MAS（引用率8.8% → 目标25%）**
MAS/智能体域场景抽象度最低(3.8/6)，补5个具体决策场景Skills：

| Skill名 | 业务锚点 |
|---|---|
| Skill-MAS-Inventory-Consensus-Action | 多仓库Agent协商补货分配决策 |
| Skill-MAS-Pricing-Coalition-Stability | 多SKU定价联合体的均衡维持 |
| Skill-MAS-Compliance-Multi-Market-Orchestrator | 多市场合规Agent并行执行协调 |
| Skill-MAS-Customer-Service-Escalation-Router | 多Agent客服升级路由决策 |
| Skill-MAS-Campaign-Budget-Negotiation | 广告/内容/搜索Agent预算博弈协商 |

**04-供应链（引用率21% → 目标40%，119个Skills利用率太低）**
119个Skills但只有25次引用，需补充「供应链决策桥接器」Playbook：
- 新Playbook: `pb-supply-chain-decision-bridge`（供应链信号→决策行动映射手册）
- 内容：每个供应链信号类型（缺货/过量/前置期异常/价格波动）→对应触发的决策Action

---

## Sprint C — 算法护城河建设（4周，长期竞争差异化）
> **核心目标：从「学术百科」到「竞品无法复制的决策武器库」**
> **从~15个护城河Skills扩展到50+个**

### C1：5大跨学科迁移算法 × 完整决策链路（P0）

每个方向：1篇顶刊 → 1个Skill → 1个专属Playbook → 关联1个Agent

**C1-1：Black-Scholes期权定价 → 新品上架时机决策**
- Skill: `Skill-Real-Options-Product-Launch-Timing`
- 核心：用实物期权理论计算「等待上架」的期权价值，量化最优上架时机
- 论文方向：`real options theory product launch timing e-commerce 2025`
- Playbook: `pb-new-product-launch-timing-options`
- 业务价值：新品首季备货准确率±15%→±8%，减少首批押注风险

**C1-2：SIR流行病学 → TikTok爆品传播预测**
- Skill: `Skill-SIR-Viral-Product-Adoption-Forecasting`（当前已有Epidemiological-SIR但未完整，需升级）
- 核心：通过发布后72小时传播参数拟合R₀，预测需求峰值时间和量级
- 论文方向：`SIR model product adoption social commerce TikTok 2025`
- Playbook: `pb-tiktok-viral-demand-prediction`
- 业务价值：爆单场景备货准确率提升35%，减少断货损失

**C1-3：CVaR金融风险度量 → 库存组合风险管理**
- Skill: `Skill-CVaR-Inventory-Risk-Portfolio`
- 核心：将Portfolio CVaR迁移到多SKU库存风险：计算整体库存组合的「极端滞销损失」分位数
- 论文方向：`CVaR portfolio optimization inventory management cross-border 2024`
- Playbook: 扩展到现有 `pb-inventory-festival` 的风险评估环节
- 业务价值：库存滞销率降低5-8%，年化释放现金20-40万

**C1-4：Stackelberg博弈论 → 竞品定价均衡求解**
- Skill: `Skill-Stackelberg-Equilibrium-Pricing`
- 核心：将Stackelberg领导者-追随者博弈迁移到「我方定价策略 × 竞品最优反应」的均衡计算
- 论文方向：`Stackelberg game pricing e-commerce marketplace competition 2024`
- Playbook: 扩展到现有 `pb-game-theory-pricing` 的博弈推演环节
- 业务价值：避免无效价格战，保护毛利率2-4pp

**C1-5：复杂网络中介中心性 → 供应链韧性瓶颈识别**
- Skill: `Skill-Supply-Chain-Betweenness-Centrality`
- 核心：将网络科学中介中心性迁移到供应链：识别「断供后影响最大的关键供应商节点」
- 论文方向：`supply chain network betweenness centrality resilience disruption risk 2024`
- Playbook: 新增到现有 `pb-supply-chain-intelligence` 的韧性评估环节
- 业务价值：提前识别单点依赖风险，关键供应商断货预警提前30天

### C2：完善现有护城河Skills的执行链路（P1）

以下「孤立的高价值Skills」已存在但无Playbook、无Agent入口：

| Skill | 问题 | 解决方案 |
|---|---|---|
| Skill-Counterfactual-Price-Elasticity | 缺roadmap_phase，无Playbook引用 | 补元数据+引入pb-game-theory-pricing |
| Skill-Epidemiological-Viral-Traffic-SIR | 缺roadmap_phase，Playbook引用=0 | 补元数据+引入pb-tiktok-shop |
| Skill-QUBO-Ad-Budget-Allocation | 缺roadmap_phase，无D层配对 | 补元数据+创建配对D层Skill |
| Skill-MonteCarlo-Tariff-Risk | 缺roadmap_phase，无Playbook引用 | 补元数据+引入pb-tariff-response |
| Skill-Navier-Stokes-Warehouse | 缺roadmap_phase，无D层配对 | 补元数据+创建配对D层Skill |

### C3：4个完全空白业务痛点补全（P1）

| 方向 | Skills数量 | 对应Playbook |
|---|---|---|
| 跨境汇率风险对冲 | +3个（FX套期保值/外汇敞口计算/汇率冲击预警） | 扩展pb-tariff-response |
| 差评管理与防御 | +3个（差评速率异常检测/Vine计划效果预测/差评根因溯源） | 新增pb-review-defense |
| 多账号关联风险 | +3个（账号指纹关联检测/IP行为聚类/操作时序异常检测） | 扩展pb-risk-defense |
| 母婴配方营养合规 | +2个（婴儿配方FDA营养标签验证/成分过敏原自动检测） | 扩展pb-compliance |

---

## Sprint D — 知识图谱精度升级（2周，质量债务清零）
> **核心目标：从「形式正确」到「语义精准」**

### D1：MAS/智能体工程域场景具体化（P1）

16-智能体工程(场景具体化3.6/6) + 10-MAS(3.8/6) = 最抽象的两个域
共选40个具体化度<3的Skills进行内容升级：
- 每个Skill的应用案例补充：具体品类（吸奶器/奶粉/婴儿车）+ 具体数字 + 具体平台
- 目标：平均分从3.7→5.0
- 执行：并行4个子任务，每任务处理10个Skills

### D2：NLP-VOC ↔ 增长模型桥梁修复（P0）

当前2条桥梁（全图最稀疏关键路径）：
- 在现有07-NLP-VOC的20个Skills中，为每个补充至少1条指向06-增长模型的`[[双括号]]`链接
- 在06-增长模型的45个Skills中，为「需求/增长预测类」补充指向07-NLP-VOC的链接
- 目标：NLP-VOC ↔ 增长模型桥梁从2条→15条

### D3：代码可执行性修复（P1）

根据审计报告，13-广告分析域的67%Skills无[✓]代码执行标记：
- 筛选所有无`[✓]`标记的Skills（估计~100个）
- 验证代码可运行性，补充末行测试输出
- 20-AI视频生成域的0% roadmap_phase单独处理

### D4：ps_override ROI精准化（P2）

当前ps_override 874条中约30%使用模糊ROI（「年化提升」「节省成本」）：
- 找出ROI无具体数字的条目（约260条）
- 批量升级为「年化节省X万元」或「提升Y%」的具体格式
- 重点：10-MAS / 16-智能体工程 / 11-AI人文 三个抽象度高的域

---

## Sprint E — 产品化闭环（3周，商业化就绪度5.2→7.5）
> **核心目标：从「知识库」到「可交付的决策产品」**

### E1：Agent ↔ Playbook 关联映射修复（P0）

当前19个Agents中只有约40%的Skill被Playbook引用，关联断层严重：
- 为每个Playbook的每个Step推荐对应Agent（生成映射表）
- 重点修复：`pb-new-product-launch`（无对应冷启动Agent）、`pb-inventory-festival`（无大促专用Agent）
- 新增2个Agent：「新品冷启动助手」「大促备货决策Agent」

### E2：反事实定价Solution补全（P1）

`sol-counterfactual-pricing` 当前core_skills仅3个，承诺的MAS博弈能力缺失：
- 补充7个核心Skills：MAS编排/博弈推演/反事实基线/Stackelberg均衡/价格战检测/利润保护门控/定价动作执行
- 使该Solution从「纸面架构」升级为「可执行方案」

### E3：新增2个高价值Solution（P1）

**sol-inventory-risk-management（库存风险管理架构）**
- 覆盖：CVaR风险度量 + 期权时机决策 + 韧性瓶颈识别 + 动态安全库存
- 关联：04-供应链 × 23-运营财务 × 18-物流履约
- ROI定位：年化减少库存滞销损失200-500万元

**sol-viral-growth-engine（病毒增长引擎架构）**
- 覆盖：SIR传播预测 + TikTok内容评分 + VOC驱动产品迭代 + 爆品备货协同
- 关联：07-NLP-VOC × 06-增长模型 × 20-AI视频生成 × 03-时间序列
- ROI定位：爆单场景GMV损失（断货）降低60%

### E4：3本新场景手册 Playbooks（P2）

| Playbook | 覆盖缺口 | 核心Skills |
|---|---|---|
| `pb-review-defense` | 差评管理完全空白 | 差评速率检测+根因溯源+Vine优化+差评防御策略 |
| `pb-fx-tariff-hedging` | 汇率/关税风险管理 | 外汇套期保值+关税分类优化+价格对冲策略 |
| `pb-supply-chain-decision-bridge` | 供应链Signal→Action断层 | 12类供应链信号对应决策触发矩阵 |

---

## 优先级总矩阵

```
┌─────────────────────────────────────────────────────────┐
│                  执行优先级 × 价值矩阵                    │
│                                                          │
│  高价值+快执行（本周必须）:                               │
│  ● Sprint A: CPSC合规5个Skills+Playbook（窗口期7天）      │
│  ● Sprint D2: NLP-VOC↔增长模型桥梁修复（半天）           │
│  ● Sprint A3: 修复11个无roadmap_phase（1小时）            │
│                                                          │
│  高价值+中执行（2-3周）:                                  │
│  ● Sprint B1: 12个D层决策触发器Skills                    │
│  ● Sprint C1-3,4,5: CVaR+Stackelberg+中介中心性         │
│  ● Sprint E1: Agent↔Playbook关联修复                     │
│                                                          │
│  高价值+慢执行（4-6周）:                                  │
│  ● Sprint C1-1,2: 期权定价+SIR传播算法                   │
│  ● Sprint B2: 低引用率域专项（NLP-VOC/MAS/供应链）        │
│  ● Sprint E3: 2个新Solution                              │
│                                                          │
│  中价值+快执行（1周内，质量债务）:                         │
│  ● Sprint D1: MAS/智能体工程具体化                        │
│  ● Sprint D3: 代码可执行性修复                            │
│  ● Sprint E2: 反事实定价Solution补全                      │
└─────────────────────────────────────────────────────────┘
```

---

## 数量目标追踪表

| Sprint | 新Skills | 新Playbooks | 新Solutions | D层Skill增量 |
|--------|---------|------------|------------|-------------|
| A | +5 | +1 | 0 | +0 |
| B | +22 | +1 | 0 | +12 |
| C | +13 | +3 | 0 | +5 |
| D | 0（升级） | 0 | 0 | +0 |
| E | +5 | +3 | +2 | +3 |
| **合计** | **+45** | **+8** | **+2** | **+20** |

**目标达成后**:
- Skills: 874 + 45 = **919**
- Playbooks: 25 + 8 = **33**
- Solutions: 10 + 2 = **12**
- D层Skills: 48 + 20 = **68 (7.4%)** ← 阶段性目标，长期目标25%需持续迭代

---

## 18个月愿景（可量化）

```
2026 Q3（本计划完成）:
  - Skills 919, D层7.4%, CPSC合规自动化上线, 护城河Skills ~35
  
2026 Q4（下一轮迭代）:
  - Skills 1000+, D层15%, 5大跨学科算法全部有Playbook
  - 发布「母婴跨境AI能力成熟度报告」（行业影响力建立）
  
2027 Q1（商业化启动）:
  - D层Skills 25%, Playbook引用率50%+
  - API化：Skill搜索API + Agent调用API发布
  - 3家GMV千万+卖家内嵌使用
  
2027 Q2（市场地位）:
  - Skills 1200+, 护城河Skills 50+
  - 2个「降维打击」案例（用金融/物理算法解决电商问题）公开发表
  - 「paper2skills方法论」成为母婴跨境算法决策行业参考
```
