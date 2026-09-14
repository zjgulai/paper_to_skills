# PHASE6 · S2 · 旧方案页收割（19 页 → 64 格体系）

> **本文件由 `paper2skills-research/scripts/harvest_legacy_solutions.py --run` 生成，手改会被 `--check` 判红。**
> 每个数字都由脚本现场算出；报告内嵌数字块必须等于**现场重算的真值**（判据 J6）——注意比对基准是现场真值，**不是产物里那一份**（产物本身可能已被改坏）。

<!-- LEGACY-SOLUTIONS-NUMBERS:BEGIN
{
  "anchor_verbatim_ok": 218,
  "boilerplate_common_lines": 201,
  "boilerplate_common_lines_unique": 149,
  "boilerplate_h2_titles": 2,
  "boilerplate_items_scoped": 0,
  "boilerplate_items_unscoped": 0,
  "chips_entries_joined": 192,
  "chips_entries_total": 194,
  "chips_unique_join_curated_146": 16,
  "chips_unique_joined": 179,
  "chips_unique_total": 181,
  "chips_unique_unjoined": 2,
  "chips_unjoined_list": [
    "Skill-Cross-Domain-Orthogonal-Signals",
    "Skill-Uplift-Cannibalization-Modeling"
  ],
  "flows_landed": 8,
  "harvest_entries_total": 218,
  "index_pages": 1,
  "judgements_declared": 15,
  "l3_landed": 87,
  "l3_total": 151,
  "layer_depth_declared_max": 7,
  "layer_depth_declared_min": 4,
  "layer_no_histogram": {
    "L1": 19,
    "L2": 19,
    "L3": 19,
    "L4": 19,
    "L5": 8,
    "L6": 7,
    "L7": 1
  },
  "layer_rows": 92,
  "layers_h2_mismatch_pages": 0,
  "m_cell_max_pages": 19,
  "m_cell_min_pages": 14,
  "m_cells_total": 24,
  "m_cells_with_any_page": 24,
  "mutations_declared": 17,
  "pages_without_any_join": 0,
  "roadmap_phases": 72,
  "roadmap_phases_with_labeled_entry": 0,
  "roadmap_phases_with_labeled_exit": 0,
  "roadmap_phases_with_outcome_field": 72,
  "roles_landed": 43,
  "solution_pages": 19,
  "source_files_scanned": 20,
  "tamper_samples_declared": 16,
  "trap_empty_section_pages": [
    "playbook/solutions/sol-counterfactual-pricing.html"
  ],
  "trap_pages": 18,
  "trap_rows": 54
}
LEGACY-SOLUTIONS-NUMBERS:END -->

## 1 结论

- 扫到 **20** 个 HTML（19 个方案页 + 1 个 index），收割 **218** 条 M 格素材：
  分层架构 **92** 条 / 路线图 **72** 条 / 架构陷阱 **54** 条。
- **丢弃**：19 页的「核心 Skill 索引」= **194** 条卡位（181 个唯一卡 id），显式登记在 `discarded.card_position_index`。
- **落格**（全派生）：**179/181** 个唯一卡 id（条目级 192/194）能 join 到产品侧 `classification.json`，覆盖 **87/151** 个 L3、**43** 个岗位、**8**/8 条 FLOW、**24/24** 个 M 格。
- ⚠️ **M 格覆盖 24/24 这个数字没有区分度**：每个 M 格被 14–19 个方案页可达。有区分度的是 **L3 层**（87/151）。

## 2 复跑

```bash
python3 paper2skills-research/scripts/harvest_legacy_solutions.py --run       # 重建产物 + 本报告
python3 paper2skills-research/scripts/harvest_legacy_solutions.py --check     # 现场重算比对（已有产物不算数）
python3 paper2skills-research/scripts/harvest_legacy_solutions.py --selftest  # 判据表 + 篡改样本 + 变异测试 + 端到端
```

退出码：**0 全过 / 1 判据红 / 2 输入没拿到（≠ 通过）/ 3 脚本内部错误（≠ 判红）**

## 3 判据表（15 条判据 · 16 份篡改样本 · 17 份变异体）

「打红样本」= 把产物改坏一处；「变异体」= 把该判据的守卫行改成 `if False:`。
**判定式是成对的**：原版对篡改样本必须 `exit 1`，变异体对**同一份**样本必须 `exit 0` ——
只有同时成立，才证明「这条判据在承重」，而不是「恰好被别的判据兜住了」。

| 判据 | 这条标定了什么 | 打红样本 | 变异体 | 期望 原版/变体 |
|---|---|---|---|---|
| `J0a` | 逐字内容漂移 | 改 traps[0].title 一个字 | M0a | 1 / 0 |
| `J0b` | 源指纹漂移 | 改 sources[0].sha256 | M0b | 1 / 0 |
| `J1` | 落格 join 完整性 | by_page[0].l3[0].role_id → AGT-999 | M1 | 1 / 0 |
| `J2` | 锚点合法性 | layers[0].line → 999999 | M2 | 1 / 0 |
| `J3` | 逐字回指 | layers[0].line_end → line（区间缩到一行） | M3 | 1 / 0 |
| `J4` | 丢弃登记 + 泄漏扫描 | ①往 harvest 注入 card_index 键；②把卡位标签塞进 layers[0].note（非文本载荷字段） | M4 | 1 / 0 |
| `J5` | 计数现算一致 | counts.chips_entries_total 194 → 193 | M5 | 1 / 0 |
| `J6` | 报告数字同步 | 改报告内嵌数字块 chips_entries_total | M6 | 1 / 0 |
| `J7` | 页面划分完整 | 从 by_page 删掉一页 | M7 | 1 / 0 |
| `J8` | 未 join 不静默 | 清空 landing.missing.unjoined_chips | M8 | 1 / 0 |
| `J9` | M 格自洽 | by_m_cell[0].n_pages → 99 | M9 | 1 / 0 |
| `J10` | 进入/退出条件现算 | counts.roadmap_phases_with_labeled_entry → 3 | M10 | 1 / 0 |
| `J11` | 样板不产生条目 | counts.boilerplate_items_scoped → 5 | M11 | 1 / 0 |
| `J12` | 层数与 h2 声明自洽 | pages[0].layers_declared_depth → 99 | M12 | 1 / 0 |
| `J13` | L3 落点自洽且满 151 | 从 by_l3 删掉一条 | M13 | 1 / 0 |
| `MS` | （仪器）解析作用域限定 | 不动产物，改脚本 | MS | 0 / 1（反向） |

另有 **6 条端到端用例**（`subprocess` 跑真 CLI + 夹具，**不 import 库函数**：干净夹具 `--run`/`--check` 各一条 + 四种「输入没拿到 ⇒ exit 2」）与 **3 条反向控制**（样板不产生条目 · 诱饵对照 · 进入/退出探测器能命中构造样本），全部由 `--selftest` 现场执行。
复跑 `--selftest` 的收尾会打印两行读数：`31 项用例 · 17/17 变异抓住` 与 `scripts/ 目录跑前跑后无新增文件`。

## 4 收割内容（每条都可逐字回指）

锚点四元组 = `source`（文件）+ `line`/`line_end`（行区间）+ `anchor`（可定位锚点）+ `verbatim`（逐字文本字段）。判据 `J0a` 保证内容与源逐字一致，`J2` 保证锚点合法。

### 4.1 分层架构（每层职责）

⚠️ **只有三元组**：层号 / 层名 / 描述。源 HTML 的 `.sd-layer` **没有边界字段、没有层间关系字段**（描述里 0 处边界语）。本报告不替它补。

| 页 | 层 | 层名 | 描述（逐字） | 锚点 |
|---|---|---|---|---|
| `sol-ads-organic-synergy.html` | L1 | 数据采集层 | 自然排名快照（每日）· 广告位置 & 出价 · 搜索量趋势 · CVR/CTR 历史 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-ads-organic-synergy.html` | L2 | 关键词分层层 | 核心词（Top 20）· 长尾词矩阵 · 品牌防守词 · 竞品拦截词 · SOV 计算 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-ads-organic-synergy.html` | L3 | 协同决策层 | 自然位置→广告出价映射函数 · 季节性调节 · 竞品动作响应 · 置信区间门控 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-ads-organic-synergy.html` | L4 | 预算优化层 | ROAS 预测模型 · 多目标线性规划 · 广告组预算再分配 · 归因窗口协调 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-ai-agent-ops-automation.html` | L1 | SOP 知识层 | 运营 SOP 文档化 · 案例库 · 决策规则树 · 异常处理手册 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-ai-agent-ops-automation.html` | L2 | 单 Agent 层 | 补货 Agent · 定价 Agent · Listing 优化 Agent · 客服分诊 Agent | L87 div.sd-layers > div.sd-layer[2] |
| `sol-ai-agent-ops-automation.html` | L3 | 工具调用层 | SP-API 数据获取 · 广告 API 调整 · 邮件发送 · ERP 写回 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-ai-agent-ops-automation.html` | L4 | 多 Agent 编排层 | Supervisor-Worker 架构 · DAG 执行计划 · 结果汇聚 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-ai-agent-ops-automation.html` | L5 | 置信度门控层 | 三档执行（全自动/人工确认/人工接管）· 高风险操作审批 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-ai-agent-ops-automation.html` | L6 | ROI 度量层 | Agent 成本追踪 · 决策效果归因 · ROI 仪表盘 | L115 div.sd-layers > div.sd-layer[6] |
| `sol-ai-engineering-acceleration.html` | L1 | 推理加速层 | Speculative Decoding草稿模型+目标模型协作，KV-Cache Prefix Sharing共享System Prompt | L80 div.sd-layers > div.sd-layer[1] |
| `sol-ai-engineering-acceleration.html` | L2 | 主动检索层 | Self-RAG按需触发检索，FLARE置信度驱动，比固定RAG减少40%冗余检索 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-ai-engineering-acceleration.html` | L3 | 少样本层 | TabPFN ICL 1秒建模，新品类50条样本AUC从0.62→0.82，无需训练周期 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-ai-engineering-acceleration.html` | L4 | 持续学习层 | EWC弹性权重约束，季节切换不遗忘旧知识，模型跨季节MAPE从28%降至11% | L101 div.sd-layers > div.sd-layer[4] |
| `sol-aigc-content-factory.html` | L1 | 素材基础层 | 商品白底图 · 品牌色板 · 语调档案 · 竞品参考图 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-aigc-content-factory.html` | L2 | 内容生成层 | 主图质量评分 · Stable Diffusion 场景化 · 多语言 Listing LLM · A+ 模板引擎 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-aigc-content-factory.html` | L3 | 内容审核层 | 合规词检测 · 品牌一致性检查 · 文化本地化审核 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-aigc-content-factory.html` | L4 | 测试分发层 | 多臂老虎机 A/B · 各平台规格适配 · 流量分配策略 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-aigc-content-factory.html` | L5 | 归因分析层 | 素材 → 点击 → 转化归因 · 内容特征 XGBoost 归因 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-aigc-content-factory.html` | L6 | 闭环学习层 | 胜出素材特征提炼 → 下一批生成参数更新 | L115 div.sd-layers > div.sd-layer[6] |
| `sol-causal-intelligence-pricing.html` | L1 | 去偏弹性层 | DML双重去偏，控制季节/促销混淆变量，获得真实价格因果效应 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-causal-intelligence-pricing.html` | L2 | 异质效应层 | X-Learner识别不同月龄段/消费层级用户的弹性差异，精准发券 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-causal-intelligence-pricing.html` | L3 | 市场评估层 | Augmented Synthetic Control评估新市场准入，合成对照组消除自然趋势 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-causal-intelligence-pricing.html` | L4 | 因果时序层 | CausalImpact量化促销活动的真实增量，识别需求借贷vs真实增长 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-cdp-lookalike-full-stack.html` | L1 | 身份统一层 | 哈希匹配 + 行为概率匹配 + GNN 跨设备匹配 → 统一 CDP 用户 ID | L80 div.sd-layers > div.sd-layer[1] |
| `sol-cdp-lookalike-full-stack.html` | L2 | 种子质量层 | 规则过滤 + 孤立森林异常检测 + 代理分类器净化 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-cdp-lookalike-full-stack.html` | L3 | Lookalike 建模层 | 双塔自建 / 图神经网络传播 / 联邦学习（隐私合规） | L94 div.sd-layers > div.sd-layer[3] |
| `sol-cdp-lookalike-full-stack.html` | L4 | 置信度校准层 | 密度估计 + Platt Scaling + Precision@K 扩展决策 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-counterfactual-pricing.html` | L1 | 信号解析层 | 正交数据获取：海关提单、社媒发帖加速度 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-counterfactual-pricing.html` | L2 | 因果推演层 | Do-Calculus 反事实基线，测算净增量 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-counterfactual-pricing.html` | L3 | 董事会博弈层 | 流量、供应链、财务 Agent 纳什均衡 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-counterfactual-pricing.html` | L4 | 门控执行层 | 三轨合规验证与 API 自动化执行 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-global-compliance-firewall.html` | L1 | 规则知识库 | Amazon ToS · FTC 禁用词 · CPSC 产品安全 · 各国 VAT 合规 · 平台规则变更 RSS | L80 div.sd-layers > div.sd-layer[1] |
| `sol-global-compliance-firewall.html` | L2 | 内容扫描层 | Listing 违规词检测 · 图片合规检查 · 医疗宣称识别 · 绝对化用语拦截 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-global-compliance-firewall.html` | L3 | 产品认证层 | 品类认证路径规划 · 证书有效期追踪 · SGS/BV/TÜV API 对接 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-global-compliance-firewall.html` | L4 | 账号监控层 | ODR/LR/VTR 实时追踪 · 异常信号预警 · 竞品恶意投诉识别 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-global-compliance-firewall.html` | L5 | 风险处置层 | 分级处置 SOP · POA 模板生成 · 申诉文档自动整理 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-global-compliance-firewall.html` | L6 | 护城河层 | 高门槛认证（CE/FDA 510k）建立竞争壁垒 · 合规知识库沉淀 | L115 div.sd-layers > div.sd-layer[6] |
| `sol-intelligent-pricing-revenue.html` | L1 | 数据采集层 | 竞品价格实时爬取 · 促销活动监控 · 平台费率变化追踪 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-intelligent-pricing-revenue.html` | L2 | 需求建模层 | 价格弹性回归 · 交叉弹性 · 季节性分解 · 节假日调整 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-intelligent-pricing-revenue.html` | L3 | 竞争分析层 | 竞品定价策略识别 · 跟价 vs 领价决策 · 品牌定价护城河 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-intelligent-pricing-revenue.html` | L4 | 调价执行层 | 规则引擎（floor/ceiling）· Repricing API · 促销协同 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-intelligent-pricing-revenue.html` | L5 | 财务归因层 | 定价 → 毛利归因 · P&L 仿真 · 现金流影响预测 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-intelligent-pricing-revenue.html` | L6 | 策略优化层 | 多目标优化（利润/份额/清仓）· 竞品博弈策略 | L115 div.sd-layers > div.sd-layer[6] |
| `sol-inventory-risk-management.html` | L1 | 风险感知层 | 多SKU历史需求数据 · 前置期分布建模 · 季节性分解 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-inventory-risk-management.html` | L2 | CVaR量化层 | Monte Carlo需求模拟 · 组合CVaR计算 · 边际风险贡献 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-inventory-risk-management.html` | L3 | 网络韧性层 | 供应链有向图 · 中介中心性 · 瓶颈节点识别 · 断供影响模拟 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-inventory-risk-management.html` | L4 | 决策执行层 | 安全库存动态公式 · P95触发补货 · 资金约束优化 · 执行回路 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-logistics-intelligence-esg.html` | L1 | 合规监控层 | IoT冷链温控全程监测，MKT计算，自动生成FDA/EU合规报告 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-logistics-intelligence-esg.html` | L2 | 碳足迹层 | GHG Protocol碳排放计算，ESG报告自动化，CBAM碳税合规 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-logistics-intelligence-esg.html` | L3 | 承运商优化层 | ML多目标承运商选择，帕累托最优成本-时效-碳排放组合 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-logistics-intelligence-esg.html` | L4 | 实时路由层 | 事件驱动动态重路由，突发堵塞/故障的实时路径重规划 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-logistics-intelligence-esg.html` | L5 | 知识图谱层 | 物流KG统一认证/关税/航线知识，3秒完成跨境发货决策查询 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-member-growth-system.html` | L1 | 体系设计层 | Markov 链 CLV 建模 → 最优等级数量和门槛 → 积分负债精算 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-member-growth-system.html` | L2 | 获客转化层 | 跨境注册漏斗分市场优化 → 早期 LTV 预测 → 动态激励分配 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-member-growth-system.html` | L3 | 生命周期管理层 | RL 序列干预 → 图神经网络流失预警 → 会员阶段触达策略 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-member-growth-system.html` | L4 | 裂变扩增层 | 裂变网络价值归因 → 最优激励额度 → 超级推荐者识别与激励 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-privacy-safe-advertising.html` | L1 | 身份隐私层 | 哈希 ID 匹配 + 差分隐私保护 + K-匿名性约束 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-privacy-safe-advertising.html` | L2 | 联邦建模层 | 垂直联邦学习 VFL + 知识蒸馏 + 非对齐用户处理 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-privacy-safe-advertising.html` | L3 | 洁净室协作层 | AWS Clean Rooms / 自建 DCR + SQL 受众分析 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-privacy-safe-advertising.html` | L4 | 因果归因层 | IPW 倒数概率加权 + 混淆去偏 + 真实 iROAS 计算 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-sc-tag-to-decision.html` | L1 | 数据基础层 | Golden Record · OKB Graph · Feature Store · Event Sourcing | L80 div.sd-layers > div.sd-layer[1] |
| `sol-sc-tag-to-decision.html` | L2 | 本体语义层 | ObjectTypes · LinkTypes · LLM自动构建 · Schema版本化 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-sc-tag-to-decision.html` | L3 | 标签工程层 | 静态标签 · 动态标签 · 预测标签 · 规则→ML→LLM→图传播 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-sc-tag-to-decision.html` | L4 | 信号分析层 | KPI监控 · 因果DAG根因归因 · GCF隐性需求 · 置信区间 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-sc-tag-to-decision.html` | L5 | 决策推理层 | What-if多情景 · MILP多目标规划 · 因果干预 · 置信门控 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-sc-tag-to-decision.html` | L6 | 行动执行层 | 全自动/半自动/人审三档 · MCP写回ERP/WMS · 多智能体共识 | L115 div.sd-layers > div.sd-layer[6] |
| `sol-sc-tag-to-decision.html` | L7 | 反馈学习层 | 数字孪生 · 审计追踪 · 决策结果闭环标签更新 | L122 div.sd-layers > div.sd-layer[7] |
| `sol-search-traffic-dominance.html` | L1 | 数据采集层 | 搜索量代理数据 · 竞品 Listing 抓取 · SP-API 搜索词报告 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-search-traffic-dominance.html` | L2 | 关键词分析层 | 蓝海缺口矩阵 · TF-IDF 竞品词挖掘 · 自相竞争检测 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-search-traffic-dominance.html` | L3 | Listing 优化层 | 语义相关性评分 · 关键词密度控制 · 多语言生成 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-search-traffic-dominance.html` | L4 | 排名追踪层 | 位置弹性建模 · CTR/CVR 归因 · 排名因子 SHAP 分析 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-search-traffic-dominance.html` | L5 | 广告协同层 | 有机+付费协同预算 · 飞轮溢出 ROI 追踪 · ACoS 优化 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-search-traffic-dominance.html` | L6 | 护城河层 | 高域名词独占策略 · 竞品词反防御 · 品牌词保护 | L115 div.sd-layers > div.sd-layer[6] |
| `sol-user-growth-repurchase-flywheel.html` | L1 | 用户数据层 | 订单历史 · 行为事件流 · 跨平台 ID 统一 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-user-growth-repurchase-flywheel.html` | L2 | 分层分析层 | RFM 分群 · Cohort 分析 · 生命周期阶段识别 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-user-growth-repurchase-flywheel.html` | L3 | 预测模型层 | 流失概率 · 复购时间窗 · LTV 预测 · 升级倾向 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-user-growth-repurchase-flywheel.html` | L4 | 触达策略层 | 个性化内容生成 · 多通道优先级 · 频次控制 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-user-growth-repurchase-flywheel.html` | L5 | 归因优化层 | 干预效果 DiD 归因 · Thompson Sampling · 策略演化 | L108 div.sd-layers > div.sd-layer[5] |
| `sol-user-growth-repurchase-flywheel.html` | L6 | 飞轮放大层 | 会员积分 ROI · 老带新裂变 · 私域社群运营 | L115 div.sd-layers > div.sd-layer[6] |
| `sol-video-commerce-intelligence.html` | L1 | 内容生产层 | Trend Agent选题→Script Agent脚本→Production Agent生成→Analytics Agent分析，3h→10min | L80 div.sd-layers > div.sd-layer[1] |
| `sol-video-commerce-intelligence.html` | L2 | 商品关联层 | CLIP多模态识别视频中商品，FAISS向量检索匹配SKU，自动生成购物链接 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-video-commerce-intelligence.html` | L3 | 质量对齐层 | RLHF奖励模型筛选高质量内容，从5个候选选最优，转化率+18% | L94 div.sd-layers > div.sd-layer[3] |
| `sol-video-commerce-intelligence.html` | L4 | 因果归因层 | Geo Holdout实验揭示平台归因虚高，获得真实增量ROI，优化预算分配 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-viral-growth-engine.html` | L1 | 信号采集层 | TikTok观看/点赞/转发数据 · UGC晒单识别 · VOC评论流式采集 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-viral-growth-engine.html` | L2 | 传播建模层 | SIR微分方程求解 · β/γ参数拟合 · R₀计算 · 置信区间构建 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-viral-growth-engine.html` | L3 | 需求预测层 | 传播曲线→需求曲线映射 · 市场规模约束 · 季节性叠加 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-viral-growth-engine.html` | L4 | 增长执行层 | 需求峰值触发补货 · 内容放大决策 · VOC反馈迭代 · 增长闭环 | L101 div.sd-layers > div.sd-layer[4] |
| `sol-voc-product-selection.html` | L1 | 数据采集层 | Amazon 评论爬取 · Reddit/X 社交 VOC · 品牌官网差评 · 异步去重队列 | L80 div.sd-layers > div.sd-layer[1] |
| `sol-voc-product-selection.html` | L2 | NLP 处理层 | 多语言情感分析 · ABSA 属性级情感 · BERTopic 主题聚类 · 事件框架提取 | L87 div.sd-layers > div.sd-layer[2] |
| `sol-voc-product-selection.html` | L3 | 信号量化层 | 痛点频次 × 情感强度 = 信号热度 · 竞争密度指数 · VOC 趋势斜率 | L94 div.sd-layers > div.sd-layer[3] |
| `sol-voc-product-selection.html` | L4 | 决策推理层 | 机会评分矩阵 · GO/NO-GO 阈值 · 首批 SKU 配置建议 · 三轨合规验证 | L101 div.sd-layers > div.sd-layer[4] |

### 4.2 路线图（阶段划分）

⚠️ **进入条件 / 退出条件在源里没有独立字段**：72 个 Phase 只有 阶段名 / 时长 / 动作 / 预期产出。带显式进入条件标记的 = **0** 条，显式退出条件标记 = **0** 条（这两个数是**算出来的**，不是「我没找」）。产物用 `outcome`（预期产出，72 条）如实承载退出证据。

| 页 | 序 | 阶段 | 时长 | 动作（逐字） | 预期产出（逐字） | 行 |
|---|---|---|---|---|---|---|
| `sol-ads-organic-synergy.html` | 1 | Phase 1 关键词矩阵 | 1-2 周 | 目标词分层（核心/长尾/防守）+ SOV 基线 | 流量图谱可见化 | L118 |
| `sol-ads-organic-synergy.html` | 2 | Phase 2 双轨监控 | 2-4 周 | 每日自然排名 + 广告位置 + 出价联动看板 | 响应时延从周→天 | L126 |
| `sol-ads-organic-synergy.html` | 3 | Phase 3 协同出价 | 1-2 月 | 自然排名↑→广告出价梯度下调算法 | ACoS 降低 15-20% | L134 |
| `sol-ads-organic-synergy.html` | 4 | Phase 4 预算智能分配 | 2-3 月 | ROAS 预测 + 多目标预算分配优化器 | 综合 ROAS 提升 30% | L142 |
| `sol-ai-agent-ops-automation.html` | 1 | Phase 0 SOP 梳理 | 1-2 周 | 识别高重复度/低判断门槛运营任务，建立 Agent 化优先级矩阵 | 明确 ROI 最高的 Agent 化方向 | L132 |
| `sol-ai-agent-ops-automation.html` | 2 | Phase 1 单 Agent 试点 | 3-8 周 | 补货 Agent / Listing 优化 Agent 各选一个上线 | 人力节省 40%，错误率 ↓60% | L140 |
| `sol-ai-agent-ops-automation.html` | 3 | Phase 2 多 Agent 协作 | 2-5 月 | 竞品情报 + 定价 + 广告 + 库存四 Agent 协同 | 运营效率 ↑200% | L148 |
| `sol-ai-agent-ops-automation.html` | 4 | Phase 3 自我优化 | 5-12 月 | Agent 从历史决策中学习，SOP 自动更新 | 决策质量持续提升 | L156 |
| `sol-ai-engineering-acceleration.html` | 1 | Phase 1 推理加速 | 1-2周 | vLLM+Speculative Decoding+KV-Cache | 延迟降低50%+ | L118 |
| `sol-ai-engineering-acceleration.html` | 2 | Phase 2 检索智能化 | 2-4周 | Self-RAG按需检索替换固定RAG | 准确率+10%，成本-40% | L126 |
| `sol-ai-engineering-acceleration.html` | 3 | Phase 3 少样本部署 | 1-2周 | TabPFN ICL新品类即时建模 | 新品类当天上线风控 | L134 |
| `sol-ai-engineering-acceleration.html` | 4 | Phase 4 持续学习 | 4-6周 | EWC持续学习防止模型退化 | 季节MAPE稳定在<12% | L142 |
| `sol-aigc-content-factory.html` | 1 | Phase 0 素材采集 | 1-2 周 | 商品原始图/视频采集 + 品牌调性提取 | 建立内容生产基准 | L132 |
| `sol-aigc-content-factory.html` | 2 | Phase 1 内容生成 | 3-6 周 | 主图质量评分 + 多语言 Listing + A+ 模板生成 | 内容成本 ↓50% | L140 |
| `sol-aigc-content-factory.html` | 3 | Phase 2 测试优化 | 2-4 月 | 多臂老虎机 A/B 测试 + CTR/CVR 追踪 | CTR 提升 15-25% | L148 |
| `sol-aigc-content-factory.html` | 4 | Phase 3 全自动化 | 4-8 月 | 内容生成 → 测试 → 胜出版本自动投放 | 运营人力 ↓60% | L156 |
| `sol-causal-intelligence-pricing.html` | 1 | Phase 1 弹性重建 | 2-3周 | DML去偏弹性估计替换OLS | 定价偏差归零 | L118 |
| `sol-causal-intelligence-pricing.html` | 2 | Phase 2 人群差异化 | 4-6周 | X-Learner异质弹性，用券替代全量降价 | 节省40%券成本 | L126 |
| `sol-causal-intelligence-pricing.html` | 3 | Phase 3 市场评估 | 2-3月 | ASCM评估新市场准入效果 | 年化减少错误扩张损失80万 | L134 |
| `sol-cdp-lookalike-full-stack.html` | 1 | Phase 1 身份统一 | 2-3 周 | L1 哈希匹配 + L2 行为概率 + L3 图结构 | 跨平台识别率 85%+ | L118 |
| `sol-cdp-lookalike-full-stack.html` | 2 | Phase 2 种子净化 | 1 周 | 孤立森林 + 代理分类器剔除刷单/员工内购 | 种子纯度 68%→94% | L126 |
| `sol-cdp-lookalike-full-stack.html` | 3 | Phase 3 Lookalike 扩展 | 2 周 | 双塔 / 图传播 / 联邦 Lookalike 三路并行 | ROAS 提升 40-80% | L134 |
| `sol-cdp-lookalike-full-stack.html` | 4 | Phase 4 置信校准 | 1 周 | Platt Scaling + Precision@K 决策扩展比例 | 避免盲目扩量损失 | L142 |
| `sol-counterfactual-pricing.html` | 1 | Phase 1 因果基线 | 1-2 月 | 计算历史真实 Uplift | 减少盲目降价 | L118 |
| `sol-counterfactual-pricing.html` | 2 | Phase 2 旁路博弈 | 3-4 月 | MAS Agent 辩论报告 | 决策视角升维 | L126 |
| `sol-counterfactual-pricing.html` | 3 | Phase 3 闭环门控 | 5-6 月 | 自动化调价与风险阻断 | 极致响应 | L134 |
| `sol-global-compliance-firewall.html` | 1 | Phase 0 合规基线 | 1-2 周 | 现有 Listing 全量合规扫描 + 问题分级 | 摸清风险存量 | L132 |
| `sol-global-compliance-firewall.html` | 2 | Phase 1 新品门控 | 3-6 周 | 上架前合规预扫描 + 证书要求清单 + 禁用词过滤 | 新品违规率 ↓90% | L140 |
| `sol-global-compliance-firewall.html` | 3 | Phase 2 动态监控 | 2-4 月 | 平台规则变更监控 + 竞品投诉预警 + 账号健康仪表盘 | 违规响应 3天→4小时 | L148 |
| `sol-global-compliance-firewall.html` | 4 | Phase 3 智能申诉 | 4-8 月 | 申诉 POA 自动生成 + 合规护城河构建 | 申诉成功率 ↑35% | L156 |
| `sol-intelligent-pricing-revenue.html` | 1 | Phase 0 数据基础 | 1-3 周 | 竞品价格爬取 + 成本结构录入 + 历史订单清洗 | 定价决策有数据支撑 | L132 |
| `sol-intelligent-pricing-revenue.html` | 2 | Phase 1 弹性建模 | 4-8 周 | 价格弹性估计 + 需求曲线拟合 + 竞品响应模型 | 找到利润最优价格点 | L140 |
| `sol-intelligent-pricing-revenue.html` | 3 | Phase 2 动态调价 | 2-4 月 | 规则引擎自动跟价 + 促销日历联动 + 库存价格联动 | 毛利率 ↑3-5pp | L148 |
| `sol-intelligent-pricing-revenue.html` | 4 | Phase 3 P&L 闭环 | 4-8 月 | 定价决策 → P&L 影响归因 → 定价策略自我修正 | 年化毛利增量 $5-20 万 | L156 |
| `sol-inventory-risk-management.html` | 1 | Phase 1 风险基线 | 1-2周 | CVaR库存风险建模+多SKU相关性矩阵 | 量化尾部风险，告别直觉备货 | L118 |
| `sol-inventory-risk-management.html` | 2 | Phase 2 韧性诊断 | 2-4周 | 供应链网络中介中心性+瓶颈识别 | 关键供应商断货预警提前30天 | L126 |
| `sol-inventory-risk-management.html` | 3 | Phase 3 动态决策 | 1-2月 | P95触发安全库存自动调整+补货执行 | 断货率降低25% | L134 |
| `sol-inventory-risk-management.html` | 4 | Phase 4 闭环优化 | 持续 | 大促CVaR压测+备货方案自动生成 | 大促备货准确率+15% | L142 |
| `sol-logistics-intelligence-esg.html` | 1 | Phase 1 合规基础 | 2-3周 | 冷链IoT监测+碳足迹核算 | FDA通关率提升 | L125 |
| `sol-logistics-intelligence-esg.html` | 2 | Phase 2 成本优化 | 4-6周 | ML承运商选择+碳最优路径 | 物流成本降低8% | L133 |
| `sol-logistics-intelligence-esg.html` | 3 | Phase 3 实时智能 | 2-3月 | 动态重路由+知识图谱决策 | 实时响应突发事件 | L141 |
| `sol-member-growth-system.html` | 1 | Phase 1 体系设计 | 1-2 周 | Markov CLV 优化等级结构 + 门槛 | CLV +12-18% | L118 |
| `sol-member-growth-system.html` | 2 | Phase 2 注册优化 | 2 周 | 分市场漏斗 + 早期 LTV 预测动态激励 | 注册转化 +10-15pp | L126 |
| `sol-member-growth-system.html` | 3 | Phase 3 生命周期运营 | 持续 | RL 序列干预 + 图预警 + 积分精算 | 流失率 -20% | L134 |
| `sol-member-growth-system.html` | 4 | Phase 4 裂变扩增 | 持续 | 裂变价值精算 + 分层激励 + 超级推荐者 | 获客成本 -40% | L142 |
| `sol-privacy-safe-advertising.html` | 1 | Phase 1 合规评估 | 1 周 | 数据流审计 + GDPR 风险识别 + 优先级排序 | 消除法律盲区 | L118 |
| `sol-privacy-safe-advertising.html` | 2 | Phase 2 联邦 Lookalike | 4-6 周 | VFL 跨平台建模，原始数据不出本地 | Lookalike 质量 +12% | L126 |
| `sol-privacy-safe-advertising.html` | 3 | Phase 3 洁净室协作 | 2-3 周 | AWS Clean Rooms 受众重叠分析 + 媒体规划 | 媒体决策准确率 +30% | L134 |
| `sol-privacy-safe-advertising.html` | 4 | Phase 4 归因去偏 | 2 周 | IPW 去混淆，识别真实 iROAS | 避免预算浪费 15-20 万 | L142 |
| `sol-sc-tag-to-decision.html` | 1 | Phase 0 地基 | 1-4 周 | SKU ID 统一 + 供应商本体 | 必须投入，不可跳过 | L139 |
| `sol-sc-tag-to-decision.html` | 2 | Phase 1 MVP | 5-12 周 | 库存健康标签 + 补货自动触发 | 补货延误 ↓30% | L147 |
| `sol-sc-tag-to-decision.html` | 3 | Phase 2 扩展 | 3-6 月 | ML预测标签 + 因果分析 + Agent | 决策准确率 ↑40% | L155 |
| `sol-sc-tag-to-decision.html` | 4 | Phase 3 智能化 | 6-12 月 | 全自动 Action + 闭环学习 | 运营工时 ↓60% | L163 |
| `sol-search-traffic-dominance.html` | 1 | Phase 0 关键词地图 | 1 周 | 蓝海关键词挖掘 + 竞争度分析 + 优先级排序 | 找到 ROI 最高的 20 个目标词 | L132 |
| `sol-search-traffic-dominance.html` | 2 | Phase 1 Listing 优化 | 2-4 周 | 语义相关性评分 + Title/Bullet 优化 + A+ Content 升级 | 搜索可见度 ↑40% | L140 |
| `sol-search-traffic-dominance.html` | 3 | Phase 2 排名突破 | 2-3 月 | 广告加速 + 飞轮效应追踪 + 位置弹性监控 | 核心词排名进入首页 | L148 |
| `sol-search-traffic-dominance.html` | 4 | Phase 3 流量护城河 | 3-6 月 | 关键词护城河建立 + 自相竞争消除 + 跨词协同 | 自然流量超越广告流量 | L156 |
| `sol-user-growth-repurchase-flywheel.html` | 1 | Phase 0 数据基础 | 1-2 周 | 历史订单清洗 + RFM 分层基线建立 | 用户价值可视化 | L132 |
| `sol-user-growth-repurchase-flywheel.html` | 2 | Phase 1 流失预警 | 3-6 周 | 流失预测模型上线 + 高危用户告警 | 复购率 ↑8% | L140 |
| `sol-user-growth-repurchase-flywheel.html` | 3 | Phase 2 精准触达 | 2-4 月 | 最优触达时机 + 多通道 A/B 优化 | LTV ↑20% | L148 |
| `sol-user-growth-repurchase-flywheel.html` | 4 | Phase 3 飞轮闭环 | 4-8 月 | 会员体系 + 自动化增长引擎 + 裂变机制 | CAC 摊薄 28% | L156 |
| `sol-video-commerce-intelligence.html` | 1 | Phase 1 内容自动化 | 2-4周 | MAS部署+AIGC内容矩阵 | 内容量1→3篇/天 | L118 |
| `sol-video-commerce-intelligence.html` | 2 | Phase 2 商品标签化 | 3-5周 | 多模态LLM视频商品识别 | 打标准确率88%，节省75%人工 | L126 |
| `sol-video-commerce-intelligence.html` | 3 | Phase 3 因果归因 | 6-8周 | Geo实验揭示平台归因偏差 | 年化节省40万元无效广告 | L134 |
| `sol-viral-growth-engine.html` | 1 | Phase 1 信号捕获 | 持续监控 | TikTok/社媒UGC病毒潜力评分+VOC早期信号提取 | 爆品识别提前4周 | L118 |
| `sol-viral-growth-engine.html` | 2 | Phase 2 传播预测 | 发布后72h | SIR参数拟合+R₀计算+需求峰值预测 | 峰值时间预测误差<2天 | L126 |
| `sol-viral-growth-engine.html` | 3 | Phase 3 备货响应 | 发布后7天 | 需求曲线→补货量计算→紧急补货触发 | 断货率降低50% | L134 |
| `sol-viral-growth-engine.html` | 4 | Phase 4 内容迭代 | 持续 | 差评信号→产品改进→新内容测试→传播扩大 | 产品迭代周期缩短60% | L142 |
| `sol-voc-product-selection.html` | 1 | Phase 1 信号采集 | 1-2 周 | 三层 VOC 数据接入（评论/社交/差评） | 覆盖 90% 消费者真实反馈 | L118 |
| `sol-voc-product-selection.html` | 2 | Phase 2 痛点挖掘 | 2-4 周 | 情感分析 + 主题聚类 + 频次评分 | 痛点识别准确率 >85% | L126 |
| `sol-voc-product-selection.html` | 3 | Phase 3 机会评分 | 1-2 月 | 需求强度 × 竞争密度 = 机会得分 | 选品命中率提升 40% | L134 |
| `sol-voc-product-selection.html` | 4 | Phase 4 需求预测 | 2-3 月 | 时序 + VOC 情绪信号融合预测 | 新品首季备货准确率 ±15% | L142 |

### 4.3 架构陷阱

⚠️ 源页面标题写「三大架构陷阱」但 `sol-counterfactual-pricing.html` 的陷阱段是**空的**：实收 54 = **18 页 × 3**，不是 19 × 3。不替它补。

| 页 | 序 | 陷阱 | 说明全文（逐字，含「为什么」与「怎么避免」） | 行 |
|---|---|---|---|---|
| `sol-ads-organic-synergy.html` | T1 | 归因冲突陷阱 | 广告和自然同时出现时，Last-click 会把转化归给广告，导致自然流量价值被低估 40-60%，需用 Data-Driven Attribution 修正 | L158 |
| `sol-ads-organic-synergy.html` | T2 | 出价下调时机陷阱 | 自然排名第 1 页不等于稳定，需连续 14 天 ≥ 第 5 位才可降广告出价，否则排名反弹后广告冷启动代价高 | L165 |
| `sol-ads-organic-synergy.html` | T3 | 关键词蚕食陷阱 | 广告词与自然词高度重叠时，同一词内部竞价抬高 CPC，需用 Keyword Cannibalization Detection 提前分离广告与自然词池 | L172 |
| `sol-ai-agent-ops-automation.html` | ① | 直接让 Agent 全自动执行高风险决策 | 未经验证的 Agent 自动提交大额 PO 或大幅调价，一个错误可能造成 $10 万+损失。必须先在沙箱中验证 3 个月，再逐步开放自动化权限。 | L172 |
| `sol-ai-agent-ops-automation.html` | ② | SOP 没有文档化就上 Agent | Agent 需要学习人类专家的经验。如果 SOP 只存在于老运营的脑子里，Agent 无法提炼，最终只是随机乱猜。必须先把 SOP 结构化文档化。 | L179 |
| `sol-ai-agent-ops-automation.html` | ③ | 用 Token 成本评估 Agent ROI | LLM Token 费用只是 Agent 总成本的 10-20%，大头是人工审核时间和错误决策成本。ROI 计算必须用「人工节省小时数×时薪 + 错误率降低×平均损失」来计算。 | L186 |
| `sol-ai-engineering-acceleration.html` | ① | 投机解码草稿模型与目标模型架构差异过大 | 草稿模型接受率<40%时投机解码反而更慢。必须用同家族模型（Llama-7B草稿→Llama-70B目标）并预测接受率。 | L158 |
| `sol-ai-engineering-acceleration.html` | ② | 直接用TabPFN处理>1000条样本的任务 | TabPFN在样本量>1000、特征>100时性能下降明显。必须先用样本量判断：<200→ICL，200-5k→TabPFN+XGB集成，>5k→传统ML。 | L165 |
| `sol-ai-engineering-acceleration.html` | ③ | EWC λ值设置过大冻结所有参数 | λ过大使模型完全无法学习新知识，λ过小遗忘旧知识。必须在验证集上调优λ，目标是新旧任务的性能均衡。 | L172 |
| `sol-aigc-content-factory.html` | ① | 先生成再合规审查 | AI 生成的内容含 'clinically tested' 等违禁词，上架后被 Amazon 下架或 FTC 罚款。必须在生成阶段嵌入合规过滤器。 | L172 |
| `sol-aigc-content-factory.html` | ② | 用同一套素材跨平台 | Amazon 主图要求白底、TikTok 要竖版短视频、Shopee 要本地化场景图。平台规格不匹配导致流量损失 30%+。 | L179 |
| `sol-aigc-content-factory.html` | ③ | A/B 测试样本量不足就结论 | 在统计显著性不足时停止测试，伪胜出版本实际 CTR 与对照组无差异。必须预设功效分析和最小样本量。 | L186 |
| `sol-causal-intelligence-pricing.html` | ① | 用OLS估计弹性直接指导降价 | 季节效应+促销混淆导致OLS弹性偏差50%以上，用-0.3的弹性做降价决策可能亏损。必须先用DML去偏。 | L150 |
| `sol-causal-intelligence-pricing.html` | ② | 全量降价而非精准发券 | 弹性分析显示老用户弹性-0.6（不敏感），新用户弹性-2.1（高敏感）。全量降价损失高弹性用户外的利润。 | L157 |
| `sol-causal-intelligence-pricing.html` | ③ | 不做安慰剂检验直接上合成控制 | ASCM有效性依赖捐赠市场的平行趋势假设，必须做安慰剂检验和敏感性分析验证结论可信度。 | L164 |
| `sol-cdp-lookalike-full-stack.html` | ① | 跳过种子净化直接建模 | 含 10-30% 噪声种子的 Lookalike 模型 ROAS 可能低于随机投放。种子净化是 Lookalike 效果的天花板，不是可选优化。 | L158 |
| `sol-cdp-lookalike-full-stack.html` | ② | 盲目扩展到 10% 受众 | 未校准的 Lookalike 分数在 Top 5%→10% 时精度骤降。必须用 Platt Scaling 校准后再决定扩展比例，否则 ROAS 崩盘。 | L165 |
| `sol-cdp-lookalike-full-stack.html` | ③ | 欧盟市场直接上传用户哈希 | GDPR Article 9 对儿童数据有特殊保护。欧盟市场必须用联邦 Lookalike 方案，数据不出本地。 | L172 |
| `sol-global-compliance-firewall.html` | ① | 只扫 Title 不扫 Bullet/A+ 内容 | Amazon 合规审查覆盖所有可见文本，包括 A+ Content、视频字幕、Q&A。只扫 Title 导致 70% 的违规漏网。 | L172 |
| `sol-global-compliance-firewall.html` | ② | 证书到期不追踪 | SGS/BV 检测报告通常 1-2 年有效期，过期后 Listing 可被投诉下架。必须建立证书到期日历和自动提醒。 | L179 |
| `sol-global-compliance-firewall.html` | ③ | 申诉模板千篇一律 | Amazon 审核员见过太多模板化 POA，识别率极高。申诉文件必须包含具体的违规修复证据截图和可量化的改进指标。 | L186 |
| `sol-intelligent-pricing-revenue.html` | ① | 只看竞品最低价跟价 | 最低价策略在非弹性品类会引发价格战，同时压缩自身利润率。必须先测定品类价格弹性，弹性 < 1 的品类不应激进跟价。 | L172 |
| `sol-intelligent-pricing-revenue.html` | ② | 调价不考虑库存水位 | 高库存时降价清仓 + 低库存时涨价保利润，两者必须联动。割裂运营导致滞销品越降越贱、爆款越涨越缺货。 | L179 |
| `sol-intelligent-pricing-revenue.html` | ③ | 促销期间不保护锚点价 | 频繁打折让用户的心理锚点价格永久降低，恢复原价后转化率暴跌。促销节奏和力度必须用 DiD 归因管控。 | L186 |
| `sol-inventory-risk-management.html` | T1 | CVaR参数估计陷阱 | 历史数据不足时（<24个月），CVaR会严重低估极端风险。必须用Bootstrap重采样+情景模拟扩增样本 | L158 |
| `sol-inventory-risk-management.html` | T2 | 相关性崩溃陷阱 | 大促期间SKU相关性显著升高（全线爆单），平时的相关矩阵在大促时失效，需单独建大促情景模型 | L165 |
| `sol-inventory-risk-management.html` | T3 | 网络拓扑过时陷阱 | 供应链网络中介中心性每季度重算一次，供应商变动后必须更新，否则韧性评估失真 | L172 |
| `sol-logistics-intelligence-esg.html` | ① | 用历史平均温度代替MKT评估冷链合规 | FDA要求使用Mean Kinetic Temperature(MKT)而非简单平均，直接平均会低估热暴露风险，导致合规报告被拒。 | L157 |
| `sol-logistics-intelligence-esg.html` | ② | 只计算运输排放忽略仓储和包装 | GHG Protocol Scope 3要求完整供应链碳足迹，仅计算运输可能遗漏40%以上的碳排放，导致CSRD报告不合规。 | L164 |
| `sol-logistics-intelligence-esg.html` | ③ | 动态重路由没有司机接受机制 | 欧洲劳工法规要求给司机提供拒绝临时路线调整的选项，强制自动调度可能引发法律纠纷。 | L171 |
| `sol-member-growth-system.html` | ① | 等级门槛拍脑袋设定 | 三级门槛 $50/$200/$500 是最常见的随意设定。Markov CLV 优化后，最优门槛可能完全不同（如 $80/$300），且不同用户类型最优门槛不同。 | L158 |
| `sol-member-growth-system.html` | ② | 所有用户同一干预策略 | 给高意图用户发 25% 折扣券是浪费。RL 序列干预证明：高意图用户只需内容触达，低意图用户才需折扣。统一策略比个性化策略多花 30-40%。 | L165 |
| `sol-member-growth-system.html` | ③ | 积分体系越来越成功越亏损 | 积分发放量快速增长但兑换率被低估，积分负债在资产负债表上悄悄累积。必须用 Beta-Binomial 精算兑换率，季度审查积分负债。 | L172 |
| `sol-privacy-safe-advertising.html` | ① | 用哈希邮箱上传就认为合规 | SHA-256 哈希可被逆向工程还原，且 GDPR 明确规定哈希数据仍属个人数据。欧盟市场必须用联邦学习，禁止任何形式的用户数据上传。 | L158 |
| `sol-privacy-safe-advertising.html` | ② | 相信平台报告的 ROAS | 平台 ROAS 通常高估 25-45%（混淆偏差）。按高估 ROAS 加大预算，实际是在「高意图人群」上浪费广告费。必须用 IPW 计算真实 iROAS 再做预算决策。 | L165 |
| `sol-privacy-safe-advertising.html` | ③ | 洁净室查询没有 K-匿名性约束 | 分组数量 < 50 的洁净室查询结果可反向推断个人信息。必须设置最小分组阈值，否则洁净室成为隐私泄露渠道。 | L172 |
| `sol-sc-tag-to-decision.html` | ① | 跳过实体 ID 统一直接上 ML | ERP/FBA/供应商三套编码不统一，所有分析建在沙上。Phase 2 强制补做，额外 3-6 个月。 | L179 |
| `sol-sc-tag-to-decision.html` | ② | 标签设计成「结论」而非「信号」 | `needs_replenishment` 是决策输出不是输入。标签只存储可测量事实状态，决策逻辑在 L5 动态计算。 | L186 |
| `sol-sc-tag-to-decision.html` | ③ | Action 全自动化没有置信度门控 | 置信度 55% 和 95% 的决策同等自动执行，一个错误 PO 锁定 10-50 万元资金。必须三档执行。 | L193 |
| `sol-search-traffic-dominance.html` | ① | 只优化主词忽略长尾 | 主词竞争激烈且排名见效慢（3-6个月）。长尾词（3-4词组合）竞争低且能立即带来转化流量。最优策略是80%精力投长尾、20%投主词，而非反过来。 | L172 |
| `sol-search-traffic-dominance.html` | ② | 用广告 ACOS 衡量广告对排名的贡献 | 广告最大价值不是直接 ROAS，而是通过销量速度提升自然排名（飞轮效应）。必须同时追踪「广告期间自然排名变化」，否则会因为短期 ACOS 高而错误砍掉高价值广告词。 | L179 |
| `sol-search-traffic-dominance.html` | ③ | 多 SKU 全部争同一组关键词 | 同店 SKU 自相竞争会导致两个 Listing 互相压低对方排名，平台算法也会困惑。必须先做关键词自相竞争检测，然后为每个 SKU 分配专属关键词矩阵。 | L186 |
| `sol-user-growth-repurchase-flywheel.html` | ① | 对所有用户发同一套挽回邮件 | 高价值沉睡用户和低价值流失用户需要完全不同的激励策略。同一套模板对高价值用户显得廉价，对低价值用户则浪费成本。必须先做 RFM 分层再差异化干预。 | L172 |
| `sol-user-growth-repurchase-flywheel.html` | ② | 复购提醒发得太频繁或太稀疏 | 消耗品类有自然消耗周期（奶粉 30 天，纸尿裤 15 天），在窗口期外发出的提醒打扰用户且转化率极低。必须用生存分析建模个体复购窗口。 | L179 |
| `sol-user-growth-repurchase-flywheel.html` | ③ | 会员积分设计成「成本中心」 | 积分发放率过高导致积分负债超过复购收益。必须用 DiD 净增量建模在推出积分体系前量化 ROI，设置积分有效期和上限。 | L186 |
| `sol-video-commerce-intelligence.html` | ① | 直接用平台Last-Click归因评估视频ROI | TikTok平台归因通常高估真实增量30%（最后点击≠真实原因）。必须做Geo Holdout实验获得真实增量。 | L150 |
| `sol-video-commerce-intelligence.html` | ② | 内容标签化后忽略质量控制 | LLM识别准确率88%≠100%，12%的错误关联会导致买家投诉。必须设置置信度阈值和人工抽检机制。 | L157 |
| `sol-video-commerce-intelligence.html` | ③ | MAS系统没有安全护栏 | 自动生成的内容可能包含误导性医疗建议（母婴场景高风险）。必须在发布前加入LLM-as-Judge质检步骤。 | L164 |
| `sol-viral-growth-engine.html` | T1 | R₀过拟合陷阱 | 72小时早期数据量太少，β/γ估计方差极大。必须设置R₀上限（如R₀<5）和下限（>0.1），防止离群点驱动的过拟合 | L158 |
| `sol-viral-growth-engine.html` | T2 | 传播饱和陷阱 | SIR模型假设人群均匀混合，但TikTok算法推送会造成「二次传播」，导致实际需求出现双峰。需在模型中加入推送放大因子 | L165 |
| `sol-viral-growth-engine.html` | T3 | 库存响应延迟陷阱 | 传播峰值预测再准，如果供应链前置期是30天，仍然赶不上3天后的需求峰。爆品备货的核心是「预测触发补货」要早于传播信号出现 | L172 |
| `sol-voc-product-selection.html` | T1 | VOC 滞后陷阱 | 评论反映的是 3-6 个月前的需求，需结合搜索趋势做前向校正，避免追逐过时痛点上新 | L158 |
| `sol-voc-product-selection.html` | T2 | 样本偏差陷阱 | 差评用户只占 5-10%，情感分布需加权，避免过度优化低频极端痛点而忽略沉默大多数 | L165 |
| `sol-voc-product-selection.html` | T3 | 语言墙陷阱 | 多市场 VOC 需语种对齐后聚类，直接合并英/德/日评论会稀释核心信号，导致选品偏向英语市场 | L172 |

## 5 丢弃登记

- **丢了什么**：19 个方案页的「核心 Skill 索引（N 个）」段 —— `.sd-skill-chip` 锚点构成的「方案页卡位 → 卡」映射表
- **多少条**：**194** 条卡位（181 个唯一卡 id），分布在 19 个方案页。
- **能 join 多少**：条目级 **192/194**；唯一级 **179/181**；其中能 join 到**精选线 146 张**的只有 **16** 个。
- **为什么丢**：
  - 它是**第四套平行分类**的产物：全 19 页 grep `AGT-0` / `SCN-0` 零命中，卡位不与任何岗位 / 责任 / 场景 / 格绑定。
  - 卡位映射是 2026-06 的历史快照，产品侧 `classification.json`（1338 张装线卡）已是卡片归属的权威事实源；收割卡位会造出**第二份卡片归属事实源**。
  - 本仓库纪律：**丢弃必须显式登记**，否则数字上看起来像「这 19 页本来就没那么多东西」。
- **丢弃但保留连接**：**丢弃的是内容，不是连接**：181 个唯一卡 id 被当作 join key（chip href → classification.json items[].id），这是旧方案页通往 64 格体系**唯一可派生**的连接；但它本身不作为素材条目登记。
  - join 到精选线 146 张的只有 **16** 个（16/181 = 8.8%）—— **这就是落格必须走装线卡而不是精选卡的原因**。
  - 未 join：`['Skill-Cross-Domain-Orthogonal-Signals', 'Skill-Uplift-Cannibalization-Modeling']`（**登记不猜**）。
- **反向控制**：判据 `J4` 扫 `harvest` 任意深度，卡位 id / 卡位标签 / 禁键名出现即判红（篡改样本见第 3 节）。

## 6 落格（全派生，未另造分类）

```
chip href → classification.json items[].id → items[].l3[] → capability-graph l3[].name → l3[].role_id → roles[].flows[] → cells[] where flow_id ∈ flows（M 格 = cell_kind "M"）
```

- 口径写死：STG 维度**不可从旧方案页派生**（源页面无任何阶段化字段）；产物里的 STG 是 M 格展开出来的，不是从旧页面判出来的。
- 未落格页 **0** 个（名单：`（无）`）——「一个卡 id 都 join 不到」的页**有素材但无落点**，登记不兜底，也不塞进「未分类」。

### 6.1 M 格供给（24 格）

⚠️ **读这张表要小心**：`涉及 L3 数` 是「所有可达该格的方案页的 L3 并集」，
所以它跟 `可达页数` 一样**没有区分度**（一张页贡献它全部 9–18 个 L3）。
有区分度的是 §6.2 的**逐 L3 页数**。这两列如实列出，是为了让「没有区分度」这件事本身可复核。

| 格 | 阶段 | 可达页数 | 涉及 L3 数 |
|---|---|---|---|
| `FLOW-01/STG-04` | 证据收集与诊断 | 19 | 87 |
| `FLOW-01/STG-05` | 方案与标准产物 | 19 | 87 |
| `FLOW-01/STG-08` | 结果核验、关闭与异步学习 | 19 | 87 |
| `FLOW-02/STG-04` | 证据收集与诊断 | 19 | 87 |
| `FLOW-02/STG-05` | 方案与标准产物 | 19 | 87 |
| `FLOW-02/STG-08` | 结果核验、关闭与异步学习 | 19 | 87 |
| `FLOW-03/STG-04` | 证据收集与诊断 | 14 | 74 |
| `FLOW-03/STG-05` | 方案与标准产物 | 14 | 74 |
| `FLOW-03/STG-08` | 结果核验、关闭与异步学习 | 14 | 74 |
| `FLOW-04/STG-04` | 证据收集与诊断 | 18 | 85 |
| `FLOW-04/STG-05` | 方案与标准产物 | 18 | 85 |
| `FLOW-04/STG-08` | 结果核验、关闭与异步学习 | 18 | 85 |
| `FLOW-05/STG-04` | 证据收集与诊断 | 16 | 77 |
| `FLOW-05/STG-05` | 方案与标准产物 | 16 | 77 |
| `FLOW-05/STG-08` | 结果核验、关闭与异步学习 | 16 | 77 |
| `FLOW-06/STG-04` | 证据收集与诊断 | 14 | 78 |
| `FLOW-06/STG-05` | 方案与标准产物 | 14 | 78 |
| `FLOW-06/STG-08` | 结果核验、关闭与异步学习 | 14 | 78 |
| `FLOW-07/STG-04` | 证据收集与诊断 | 17 | 87 |
| `FLOW-07/STG-05` | 方案与标准产物 | 17 | 87 |
| `FLOW-07/STG-08` | 结果核验、关闭与异步学习 | 17 | 87 |
| `FLOW-08/STG-04` | 证据收集与诊断 | 18 | 86 |
| `FLOW-08/STG-05` | 方案与标准产物 | 18 | 86 |
| `FLOW-08/STG-08` | 结果核验、关闭与异步学习 | 18 | 86 |

### 6.2 L3 供给（151 个，只列有页的）

| L3 | 岗位 | A/B/C | 页数 |
|---|---|---|---|
| GMV归因分析 | AGT-003 | A | 2 |
| Listing优化 | AGT-022 | A | 5 |
| Playbook评估 | AGT-048 | B | 3 |
| VOC编码 | AGT-006 | A | 2 |
| 业务工具实现 | AGT-047 | B | 5 |
| 主数据治理 | AGT-045 | A | 3 |
| 争议证据组织 | AGT-043 | B | 1 |
| 产品准入核对 | AGT-044 | B | 3 |
| 产品问答 | AGT-036 | B | 1 |
| 产品需求定义 | AGT-009 | B | 3 |
| 价格敏感性 | AGT-026 | A | 3 |
| 会员活动 | AGT-039 | B | 3 |
| 传播规划 | AGT-029 | B | 1 |
| 体验分析 | AGT-038 | B | 2 |
| 供应商评估 | AGT-014 | A | 1 |
| 供需协调 | AGT-016 | A | 1 |
| 依赖协调 | AGT-002 | A | 2 |
| 促销规划 | AGT-026 | A | 4 |
| 关务资料检查 | AGT-019 | B | 1 |
| 内容实验 | AGT-030 | A | 2 |
| 内容策划 | AGT-030 | B | 4 |
| 分群 | AGT-034 | A | 5 |
| 到货异常追踪 | AGT-019 | A | 2 |
| 合作复盘 | AGT-033 | B | 1 |
| 商品诊断 | AGT-022 | A | 4 |
| 因果局限审查 | AGT-035 | B | 3 |
| 增量分析 | AGT-035 | A | 8 |
| 复购实验 | AGT-034 | A | 3 |
| 安全事件处理 | AGT-050 | B | 1 |
| 实验设计 | AGT-035 | A | 1 |
| 宣称审查 | AGT-044 | B | 2 |
| 容量管理 | AGT-049 | A | 2 |
| 岗位能力分析 | AGT-004 | B | 1 |
| 差异追踪 | AGT-040 | A | 1 |
| 市场机会评估 | AGT-007 | B | 5 |
| 市场语境审查 | AGT-028 | B | 1 |
| 市场进入 | AGT-024 | B | 2 |
| 广告实验 | AGT-032 | A | 2 |
| 库存分层 | AGT-017 | A | 2 |
| 异常冻结与恢复 | AGT-002 | B | 1 |
| 情景模拟 | AGT-003 | A | 1 |
| 技能版本 | AGT-048 | B | 1 |
| 投放诊断 | AGT-032 | A | 7 |
| 抽样审计 | AGT-005 | A | 1 |
| 指标契约 | AGT-045 | A | 2 |
| 授权审查 | AGT-050 | B | 4 |
| 接口契约 | AGT-047 | B | 2 |
| 搜索意图分析 | AGT-022 | A | 2 |
| 数据管道 | AGT-046 | B | 4 |
| 数据质量 | AGT-046 | A | 3 |
| 本地化 | AGT-028 | B | 3 |
| 渠道经营分析 | AGT-021 | A | 1 |
| 漏斗诊断 | AGT-023 | A | 5 |
| 物流方案 | AGT-019 | A | 1 |
| 生命周期触达 | AGT-034 | A | 5 |
| 申诉材料准备 | AGT-027 | B | 1 |
| 知识产权检索 | AGT-043 | A | 1 |
| 知识溯源 | AGT-048 | A | 3 |
| 税务资料 | AGT-042 | C | 1 |
| 站点运营 | AGT-023 | B | 1 |
| 竞品研究 | AGT-007 | B | 3 |
| 素材版本管理 | AGT-031 | B | 2 |
| 组合取舍 | AGT-008 | A | 1 |
| 组合设计 | AGT-026 | A | 2 |
| 经济性分析 | AGT-041 | A | 4 |
| 联盟运营 | AGT-033 | B | 1 |
| 行动组合 | AGT-021 | A | 1 |
| 补货模拟 | AGT-016 | A | 6 |
| 规则监测 | AGT-027 | B | 1 |
| 视觉简报 | AGT-031 | C | 1 |
| 视频制作协作 | AGT-031 | C | 1 |
| 订单协调 | AGT-015 | A | 1 |
| 证据复核 | AGT-005 | B | 1 |
| 调拨清货建议 | AGT-017 | A | 2 |
| 账号商品映射 | AGT-045 | A | 1 |
| 账号诊断 | AGT-027 | A | 1 |
| 质量分析 | AGT-018 | A | 2 |
| 资源情景比较 | AGT-001 | A | 1 |
| 趋势监测 | AGT-007 | A | 2 |
| 转化优化 | AGT-023 | A | 3 |
| 运行监测 | AGT-049 | A | 2 |
| 采购比价 | AGT-015 | A | 2 |
| 阶段投资建议 | AGT-008 | B | 1 |
| 隐私需求分析 | AGT-044 | B | 3 |
| 集成验证 | AGT-047 | B | 1 |
| 需求预测 | AGT-016 | A | 7 |
| 预算分配 | AGT-032 | A | 8 |

**未被任何旧方案页触及的 L3：64 个**（这就是 S2 的真实供给图）。

## 7 源缺口登记（不编造）

| 缺口 | 证据 | 处置 |
|---|---|---|
| 路线图的「进入条件 / 退出条件」没有独立字段 | 72 个 Phase 只有 阶段名 / 时长 / 动作 / 预期产出 四个字段；带显式进入条件标记的 = 0 条、显式退出条件标记的 = 0 条 | 不编造。产物里用 `outcome`（预期产出）如实承载「退出证据」，并在报告中写死「进入条件复算不出来」。 |
| 分层架构的「边界 / 与其他层的关系」没有独立字段 | `.sd-layer` 只有 层号 / 层名 / 描述 三元组；描述里不含 边界/不得/禁止 一类边界语（实测 0 命中） | 不编造。只收三元组。 |
| 1 个方案页的「三大架构陷阱」段是空的 | sol-counterfactual-pricing.html 有 h2「三大架构陷阱」但 `.sd-trap` 计数 = 0 ⇒ 陷阱实收 54 = 18 页 × 3，而非 19 页 × 3 | 登记为空段，不替它补 3 条。 |

## 8 这一轮没做的事

| 族 | 条数 | 为什么没做 |
|---|---|---|
| hero 标题 / 副标题 / 分类 / 更新日 | 19 | 本任务三族之外，未收（已解析进 pages[] 元数据，但未作为素材条目登记） |
| ROI 横幅（`.sd-roi-banner`） | 19 | 同上；且其中的 ROI 数字无论文/业务出处，按 G2 口径不宜作 M 格素材 |
| 摘要段（`.sd-summary`） | 19 | 同上 |
| 核心 Skill 索引（`.sd-skill-chip`） | 194 | **显式丢弃**，见 discarded.card_position_index |

- **未改任何既有事实源**：`capability-graph.json` / `gap-ledger.json` / `contracts/**` / 任何 Skill 卡一个字都没动（本轮有并发写入者，动它们会造成写入冲突）。
- **未接入 `run_phase6_gates.py`**：新门禁没有挂到统一入口上（见第 9 节工单）。
- **未做语义对齐**：旧方案页与 8 条 FLOW 之间**没有**做人工语义映射 —— 那会造出第四套分类的第五个版本。
- **未处置 2 个 join 不到的卡 id**：只登记，未追查它们在装线语料里到底叫什么。
- **未收割 hero / ROI 横幅 / 摘要**（见本节表）。
- **未做「旧方案页 → 具体 STG」的判定**：源页面没有阶段化字段，任何把某个陷阱/某层挂到 STG-04 还是 STG-05 的动作都只能是新的语义判断 —— 本轮**只给到 FLOW 级闭包**。
- **未核验 `classification.json` 的 L3 归属本身对不对**：本轮把它当既有事实源只读 join，没有回头审它的 1327 条分类（那是 F5 的地盘）。

## 9 本轮实测撞出的**仪器缺陷**（判据抓到的，不是事后总结的）

这一节是本交付物最该留的部分：**每一条都是判据先报红，才被发现**。

| # | 缺陷 | 谁抓住的 | 后果（如果不修） |
|---|---|---|---|
| I1 | **`_pos` 是切片内下标，却拿去对全文算行号** | `J2` 首跑报 6 条 `[42,41]` 越界 | 218 条锚点的行号整体前移，最大那批落进**侧边栏**（39–42 行）—— 「把样板当正文」的坐标版；行号全错而条目内容全对，抽查极难发现 |
| I2 | **`strip_tags` 写成 `<[^>]+>`，而语料里有裸的小于号** | `J3` 首跑报 6 条「字段不在申报区间内」 | `接受率<40%时…`、`季节MAPE稳定在<12%` 被整段当标签吃掉。字段抽取侥幸没受影响（切片在 `</p>` 前截断），**行区间回查**却被截成半句 ⇒ 逐字回指这条判据会静默失效 |
| I3 | **锚点区间取「下一块起点 −1 / 文件末」** | 人工复查发现 `L102–L285` | 每族最后一条的区间一路铺到文件末，「可定位的锚点」名存实亡 |
| I4 | **`REPO` 由 `__file__` 推，而变异体跑在 tmp 里** | 变异矩阵首跑 12 项失败 | 变异体的源文件相对路径整体变形 ⇒ 「只差一行」的前提不成立，变异测试当场失去意义 |
| I5 | **多条判据抢同一份篡改样本** | 变异 M2/M5/M8/M10/M11 变体恒红 | 四组「判据 ∩ 判据」重叠（J0a∩J3、J5∩J6、J5∩J8、J2∩J3）与一条「夹具上空操作」的篡改样本 ⇒ 变异分数虚高而实际分不出谁在承重。**这正是台账 #25 的同族形态** |
| I6 | **`counts` 被当成报告基准** | 同上 | 报告数字块本应与**现场真值**比，与产物比会把「产物被改坏」错报成「报告过期」 |

## 10 下游工单

0. 🔴 **`paper2skills-research/data/legacy-solutions.json` 当前被 `.gitignore` 挡住**（规则 `.gitignore:93` 的 `paper2skills-research/data/**/*.json`）——**不补 `!` 例外，这份产物就不会进 commit**。本轮**没有改 `.gitignore`**：任务硬约束只许新增 3 个文件，且该文件正被并发写入者持续追加入库例外（`backlog-l3-map.json` / `backlog-routing.json` / `material-citations.json` / `search-queries.json` 都是同一形态），改它会造成写入冲突。
   - 建议加的行（理由同 `gap-ledger.json`：**它是一份账，不是中间态** —— 218 条素材各带逐字锚点，落格链路跨仓库依赖产品侧 `classification.json`）：
     ```gitignore
     # S2 旧方案页收割产物（PHASE6 S2）：它是一份账，不是中间态 ——
     # 逐条记录 19 个方案页的分层架构/路线图/陷阱及其行级锚点，落格走跨仓库 join。
     # 判据=`scripts/harvest_legacy_solutions.py --check`。
     !paper2skills-research/data/legacy-solutions.json
     ```
   - 核验命令：`git check-ignore -v paper2skills-research/data/legacy-solutions.json`（补前 exit 0 且有输出 = 被忽略；补后应无输出）。
1. **台账编号**：#54–#57 已由 S1-W/W4 预留（材料引文核验族），故本轮的仪器缺陷建议编号 **#58–#63**（对应 I1–I6）。本轮**未改台账文件**（并发写入者正在改契约与卡片 frontmatter）。
2. **把本脚本接进 `run_phase6_gates.py`**（本轮不许改共享文件，故留作工单）；接的时候注意：`run_phase6_gates.py` 的汇总纪律是**退出码不许合并成分数**，而本脚本有 0/1/2/3 四态，`2` 必须与 `1` 分开报。
3. **2 个 join 不到的卡 id**：`Skill-Cross-Domain-Orthogonal-Signals` / `Skill-Uplift-Cannibalization-Modeling` —— 在 1338 装线语料里查无此 id，需确认是历史重命名还是从未上线。
4. **`sol-counterfactual-pricing.html` 的空陷阱段**：源页面标题声称「三大架构陷阱」而内容为空，需决定是补写还是降标题。
5. **路线图缺进入/退出条件**（72 条 Phase 全缺）：若要它们成为 M 格素材，需要在源侧补字段，而不是在收割侧猜。

## 11 口径与已知残留

- **「L1–L6 架构」是个不准确的转述**：实测每页 **4–7** 层，最高到 **L7**；层号直方图 `{"L1": 19, "L2": 19, "L3": 19, "L4": 19, "L5": 8, "L6": 7, "L7": 1}`。任务书里的「L1–L6」应理解为「L1 起的分层架构」而非固定六层。
- **「194 卡位索引」复算成立**：实测 `.sd-skill-chip` 锚点 = **194** 条（181 唯一；192/179 可 join），与 194 一致。
- ⚠️ **两个分母不许混**：`entries`（194，含 13 个跨页重复）与 `unique`（181）是两个单位；首版把条目级 join 数除以唯一数打过一次「192/181」的假账 —— 现两者分列。
- **样板反向控制**：语料公共骨架 201 行 / 样板 h2 标题 2 个；限定作用域解析产出 **0** 条。⚠️ 这个 0 在**真语料**上是**结构性成立**的（样板里没有完整的「层号+层名+描述」三元组），所以它的可证伪性由 `--selftest` 的**构造样本**负责：把作用域限定去掉，构造样本立刻产出条目。
- **`--selftest` 不留临时文件**：全部写 `tempfile.mkdtemp()`，`finally` 删除，并断言 `scripts/` 目录跑前跑后无新增文件。
