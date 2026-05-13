---
name: phase5-bi-dashboard-spec
description: Phase 5 D11 T11.3 — 7 部门 BI 看板字段定义。当各部门接入 BI 看板（Superset / Metabase / 飞书文档）、需要明确 KPI 列表、数据源、标签映射、刷新频率与样例周报时使用。
doc_type: bi-spec
module: voc-nlp
phase: phase5
day: D11
status: stable
created: 2026-05-09
owner: voc-nlp
---

# Phase 5 BI 看板字段定义（7 部门 × 5 章节）

> 本文档规定 Phase 5 阶段 7 个一级部门看板的 **KPI 列表 / 数据源 / 标签映射 / 刷新频率 / 样例周报** 五个章节。
> 配套校验器 [`bi_spec_validator.py`](../02-脚本工具/06-诊断工具/bi_spec_validator.py) 强制 35 个断言全过。
> 关联：[决策 1 中闭环 ③→④](08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md)、[v4.0 字典](../04-输出结果/01-字典版本/tag_dictionary_v4.0.xlsx)。

## 一、共同前置

- **底层数据**：`research/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl`（364,569 条；含 labels / aspects / sentiment / persona / proxy_nps / brand_mentions）
- **字典版本**：`research/04-输出结果/01-字典版本/tag_dictionary_v4.0.xlsx`（267 通用标签 + 55 aspect Sheet）
- **MAA 策略输入**：[`maa_strategy_generator.py --dept <部门>`](../02-脚本工具/01-标签进化/maa_strategy_generator.py)
- **AGRS 摘要输入**：[`agrs_summarizer.py --filter-dept <部门>`](../02-脚本工具/01-标签进化/agrs_summarizer.py)
- **本期形态**：本期不上线可视化，仅产出 Markdown 周报；v5.0 接入 Superset / Metabase / 飞书

---

## 二、部门看板定义

### 全球客服中心

#### KPI列表

| KPI | 定义 | 单位 | 阈值（黄/红）|
|---|---|---|---|
| 客服响应快命中量 | TAG_L2_001 周新增量 | 条 | < 50 / < 20 |
| 一次解决率 | TAG_L2_002 命中量 / TAG_SRV_* 总命中量 | % | < 60 / < 40 |
| 负面工单率 | data_source=zendesk 中 sentiment_polarity < -0.3 占比 | % | > 25 / > 40 |
| 售后政策吐槽量 | TAG_L2_008 周新增量 | 条 | > 30 / > 60 |
| 说明书相关问题量 | TAG_L2_009 周新增量 | 条 | > 30 / > 60 |

#### 数据源

- 主源：`data_source=zendesk` 全部记录
- 辅助：`data_source=trustpilot` 中 `客服` / `service` / `support` 关键词命中
- 排除：内部测试工单（review_id 前缀 `internal_*`）

#### 标签映射

- 主责标签：v4.0 字典 `主责部门=全球客服中心` 共 37 个标签（TAG_L2_001~009、TAG_SRV_05~10、TAG_GEN_C001、TAG_ZEN_R001~009 等）
- AIPL 节点：以 P3（Purchase）+ L1-L4（Loyalty）为主，反映售后体验
- 情感映射：客服响应快/一次解决问题为正向；优化售后政策/说明书为负向

#### 刷新频率

- **日**（每日 02:00 自动重算最近 7 日窗口）
- 异常告警：负面工单率连续 3 日 > 40% 触发飞书 webhook

#### 样例周报

```markdown
## 全球客服中心周报（2026-W19）
- 一次解决率 67.2%（环比 +2.1pp，超阈值线 60%）
- 负面工单率 28.4%（黄色预警，已超 25%）
- TOP-3 投诉：售后政策模糊（+18%）/ 说明书缺失（+12%）/ 退款流程慢（+9%）
- 推荐行动：见 [maa_strategy_generator --dept customer_service Top-10](../04-输出结果/10-周报/2026-W19/全球客服中心.md)
- 代表评论：见 [agrs_summarizer --filter-dept 全球客服中心](../04-输出结果/10-周报/2026-W19/全球客服中心_AGRS.md)
```

---

### 产品中心

#### KPI列表

| KPI | 定义 | 单位 | 阈值（黄/红）|
|---|---|---|---|
| 产品质量缺陷 TOP 10 | 主责部门=产品中心的负向标签 Top 10 | 条 | TOP-1 > 1000 / > 5000 |
| ABSA 产品满意度 | aspects 中 `product quality` 主导情感 = positive 占比 | % | < 60 / < 40 |
| 按键故障命中量 | TAG_L1_005 周新增量 | 条 | > 50 / > 100 |
| 充电类故障 | TAG_L1_013/014 合计周新增量 | 条 | > 30 / > 60 |
| 配件破损率 | TAG_L1_031/139 命中量 / 总评论量 | % | > 0.3 / > 0.6 |

#### 数据源

- 主源：`data_source ∈ {amazon_competitor, momcozy, trustpilot}` 中带 product_line 的评论
- 辅助：zendesk 中含产品故障描述的工单
- 时间窗口：最近 30 日（产品迭代周期）

#### 标签映射

- 主责标签：v4.0 字典 `主责部门=产品中心` 共 62 个标签
- AIPL 节点：以 L0-L1（Lifestyle/Loyalty）为主，反映长期使用体感
- aspect 映射：10_Aspect库 中 category=`product_quality` / `usability` / `durability` 三类 25 行

#### 刷新频率

- **周**（每周一 03:00 重算）
- 季度对齐：每季度首周自动产出长期趋势图（Phase 6 接入）

#### 样例周报

```markdown
## 产品中心周报（2026-W19）
- TOP-1 缺陷：延迟（7,893 条，环比 +5.2%）
- 充电类故障合计 442 条（达红色阈值线 60，触发紧急排查）
- 推荐行动：见 [maa_strategy_generator --dept product_rd](../04-输出结果/10-周报/2026-W19/产品中心.md)
- SRAC TOP-3：质量感知 / 易用性 / 性能满意（spread = 5.30）
```

---

### 仓储物流部

#### KPI列表

| KPI | 定义 | 单位 | 阈值（黄/红）|
|---|---|---|---|
| 物流问题率 | 物流相关负向标签命中量 / 总评论量 | % | > 5 / > 10 |
| 配送咨询量 | zendesk 中含 `shipping` / `delivery` / `tracking` 的工单 | 条 | > 100 / > 300 |
| 延迟提及率 | TAG_L1_040 周命中量 / 总评论量 | % | > 1.5 / > 3 |
| 包装破损量 | TAG_L1_031（配件破损/断裂）weekly | 条 | > 50 / > 100 |
| 跨境清关申诉 | zendesk 中 `customs` / `clearance` 关键词命中量 | 条 | > 20 / > 50 |

#### 数据源

- 主源：`data_source=zendesk` + amazon 评论中 product_line 为空但含物流关键词
- 辅助：trustpilot 多语言评论（法/德/西，含配送相关词）

#### 标签映射

- 主责标签：v4.0 字典 `主责部门=仓储物流部` 共 39 个标签
- 关键 tag_id：TAG_L1_040（延迟）、TAG_L1_031（破损）、TAG_ZEN_R009（取消订单）等
- AIPL 节点：P3（Purchase）+ L4（Loyalty 因物流体验流失）

#### 刷新频率

- **日**（物流问题响应时效要求高）
- 异常告警：单日物流问题率 > 10% 触发飞书

#### 样例周报

```markdown
## 仓储物流部周报（2026-W19）
- 物流问题率 7.8%（黄色预警，已超 5%）
- 延迟相关 TAG_L1_040 周命中 7,893 条（环比 +12%）
- 重点市场：德国/法国（配送时效抱怨集中）
- 推荐行动：见 [maa_strategy_generator --dept logistics](../04-输出结果/10-周报/2026-W19/仓储物流部.md)
```

---

### 品牌市场中心

#### KPI列表

| KPI | 定义 | 单位 | 阈值（黄/红）|
|---|---|---|---|
| Proxy NPS | promoter 占比 - detractor 占比 | % | < 30 / < 10 |
| 品牌提及趋势 | brand_mentions 非空记录周环比 | % | < -5 / < -15 |
| 竞品对比量 | brand_comparison=true 的记录数 | 条 | < 50 / < 20 |
| TAG_GEN_016 强烈推荐 | 周命中量 | 条 | < 200 / < 100 |
| TAG_GEN_017 礼品购买意向 | 周命中量 | 条 | < 50 / < 20 |

#### 数据源

- 主源：`data_source ∈ {amazon_competitor, trustpilot}` 全部记录
- 辅助：reddit / momcozy 全量
- 维度切片：按 brand_mentions 解析竞品名（gerber / similac / fisher-price 等）

#### 标签映射

- 主责标签：v4.0 字典 `主责部门=品牌市场中心` 共 34 个标签
- 关键 tag_id：TAG_GEN_016 / TAG_GEN_017 / TAG_GEN_E001/E003（市场卖点）
- AIPL 节点：A（Awareness）+ I（Interest）+ P0-P2（Pre-purchase）

#### 刷新频率

- **周**（市场策略调整周节奏）
- 月度对齐：每月 1 日产出品牌健康度报告

#### 样例周报

```markdown
## 品牌市场中心周报（2026-W19）
- Proxy NPS 41.3%（健康，> 30 阈值线）
- 品牌提及环比 +3.2%（稳定）
- TOP 竞品对比：Momcozy vs Spectra（132 次）/ vs Medela（87 次）
- 推荐行动：见 [maa_strategy_generator --dept marketing](../04-输出结果/10-周报/2026-W19/品牌市场中心.md)
```

---

### 电商运营部

#### KPI列表

| KPI | 定义 | 单位 | 阈值（黄/红）|
|---|---|---|---|
| AIPL 漏斗转化率 | A → I → P → L 各节点占比 | % | L 占比 < 25 / < 15 |
| 购买体验正向率 | aipl_stage=P 中 sentiment_polarity > 0.3 占比 | % | < 70 / < 50 |
| 评论量周环比 | 总评论量周环比 | % | < -5 / < -15 |
| 5 星好评率 | rating=5 占比 | % | < 60 / < 40 |
| 退货关键词命中 | `return` / `refund` / `送回` 关键词周命中 | 条 | > 200 / > 500 |

#### 数据源

- 主源：`data_source ∈ {amazon_competitor, momcozy}` 全量评论
- 辅助：trustpilot 高质量评论（_quality_score > 80）

#### 标签映射

- 主责标签：v4.0 字典 `主责部门=电商运营部` 共 34 个标签
- AIPL 节点：覆盖 A/I/P0-P3/L1-L4 全链路
- 关键 aspect：price/value、convenience、purchase_intent

#### 刷新频率

- **周**（与电商促销节奏对齐）
- 大促期间：自动切换为 **日** 频率（618 / 黑五 / 双 11 / Prime Day）

#### 样例周报

```markdown
## 电商运营部周报（2026-W19）
- AIPL L 占比 28.4%（健康）
- 5 星好评率 68.2%（环比 -1.3pp，需关注）
- 退货关键词命中 234 条（黄色预警）
- 推荐行动：见 [maa_strategy_generator --dept ecommerce](../04-输出结果/10-周报/2026-W19/电商运营部.md)
```

---

### 品质管理中心

#### KPI列表

| KPI | 定义 | 单位 | 阈值（黄/红）|
|---|---|---|---|
| 8 缺陷聚类趋势 | 8 个核心缺陷标签合计命中量周趋势 | 条 | 周环比 > +5% / > +15% |
| 缺陷品类热力图 | product_line × negative_tag 矩阵命中量 | 条 | 单元格 > 100 / > 500 |
| 风险等级=高 标签 | 字典中 `风险等级=高` 的标签周命中 | 条 | > 20 / > 50 |
| 烧焦/冒烟告警 | TAG_L1_014 + 关键词组合 | 条 | ≥ 1 立即告警 |
| 故障全部品线分布 | 故障类标签按 product_line 分布 | 条 | 单品线 > 200 / > 500 |

#### 数据源

- 主源：`data_source ∈ {amazon_competitor, momcozy}` 中含故障描述
- 辅助：zendesk 工单含 `defect` / `broken` / `damaged` 关键词

#### 标签映射

- 主责标签：v4.0 字典 `主责部门=品质管理中心` 共 14 个标签
- 关键 tag_id：TAG_L1_080（自动亮灯）、TAG_L1_087（奶量检测不准）、TAG_L1_184（显示故障）
- AIPL 节点：L1-L4（长期使用暴露的品控问题）

#### 刷新频率

- **周**（与品控复盘会议同步）
- 紧急通道：高风险标签命中触发即时飞书告警

#### 样例周报

```markdown
## 品质管理中心周报（2026-W19）
- 8 缺陷聚类周命中合计 1,832 条（环比 +3.2%，绿色）
- 烧焦相关 TAG 命中 14 条（紧急复盘已启动）
- 缺陷品线 TOP-3：智能母婴电器 / 喂养电器 / 内衣服饰
- 推荐行动：见 [maa_strategy_generator --dept quality_control](../04-输出结果/10-周报/2026-W19/品质管理中心.md)
```

---

### 法务合规部

#### KPI列表

| KPI | 定义 | 单位 | 阈值（黄/红）|
|---|---|---|---|
| 安全/合规标签命中量 | 字典 `主责部门=法务合规部` 标签合计 | 条 | > 5 / > 15 |
| BPA / 化学成分提及 | 文本含 `BPA` / `phthalate` / `lead` 命中量 | 条 | > 10 / > 30 |
| FDA / CE 提及量 | 含监管机构名的评论数 | 条 | > 5 / > 20 |
| 召回关键词 | `recall` / `召回` 关键词命中 | 条 | ≥ 1 即报 |
| 合规风险等级 | 字典中 `风险等级` 字段合计 | 等级 | 出现 1 条高级别即报 |

#### 数据源

- 主源：全量 `data_source` 中含监管 / 安全关键词
- 辅助：zendesk 法规咨询工单
- 关键词列表：维护在 `02-脚本工具/01-标签进化/regulatory_keywords.txt`

#### 标签映射

- 主责标签：v4.0 字典 `主责部门=法务合规部` 共 8 个标签
- 关键词：BPA / lead / phthalate / FDA / CE / EN-71 / CPSC / recall
- AIPL 节点：跨阶段（合规问题不分 AIPL）

#### 刷新频率

- **月**（合规风险长周期）
- 强制告警：召回关键词命中即时飞书告警 + 邮件抄送法务合规部

#### 样例周报

```markdown
## 法务合规部月报（2026-05）
- 合规风险事件：0 起召回（健康）
- BPA / 化学成分提及 7 条（绿色）
- FDA 相关咨询 3 条（路由至客服 + 法务合规部）
- 推荐行动：见 [maa_strategy_generator --dept regulatory](../04-输出结果/10-周报/2026-W19/法务合规部.md)
```

---

## 三、上线节奏

| 阶段 | 时间 | 形态 | 工具 |
|---|---|---|---|
| Phase 5 D11（本期）| 2026-05-09 | Markdown 周报 | maa_strategy_generator + agrs_summarizer |
| Phase 5 D14（验收）| 2026-05-20 | 7 部门初稿周报落地 | 同上 + 飞书 webhook 推送 |
| Phase 6 启动 | 2026-06 | 看板原型（飞书文档/单页）| 接入 Superset / Metabase |
| v5.0 | 2026-Q3 | 多租户 + 实时刷新 | 接入数仓 + 即席查询 |

## 四、变更记录

| 日期 | 变更 |
|---|---|
| 2026-05-09 | T11.3 初稿落地，T11.3.5 校验器配套验证 |
