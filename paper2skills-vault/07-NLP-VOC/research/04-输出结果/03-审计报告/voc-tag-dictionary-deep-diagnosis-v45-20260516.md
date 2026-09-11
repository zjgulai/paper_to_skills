---
title: VOC 标签字段宽表 v4.5 深度诊断报告
doc_type: analysis
module: voc-nlp
topic: tag-dictionary-diagnosis
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: @Data
source: ai
---

# VOC 标签字段宽表 v4.5 深度诊断报告

> **诊断范围**：tag_dictionary_v4.5.xlsx（11 sheets / 645 行字典 + 402 行映射 + 55 行 Aspect）+ 项目路径 07-NLP-VOC 全量架构与数据资产
> **诊断样本**：phase6_d9_filtered.jsonl 随机 10,000 条 + 全量字典结构化解析
> **诊断目标**：为「标签字段宽表自动打标」和「打标后全维度分析洞察」扫清结构性障碍

---

## 一、执行摘要：10 个核心发现

| # | 问题域 | 严重程度 | 核心发现 | 对自动打标的影响 | 对全维度分析的影响 |
|---|--------|---------|---------|----------------|------------------|
| 1 | 字典结构 | **P0-阻塞** | AIPL 节点分布严重失衡（通用表无 A/I/P 节点），打标结果 L1 占 77.7% | LLM 闭集缺少 A/I/P 标签素材，AIPL 漏斗无法构建 | 用户旅程分析只能覆盖 Loyalty 阶段 |
| 2 | 字典质量 | **P0-阻塞** | 情感极性严重失衡：负向 78.6% / 正向 15.7% / 中性 4.9% | 自动打标情感方向系统性偏负，需校准层补偿 | 情感趋势分析失真，负面问题被放大 |
| 3 | 字典治理 | **P0-阻塞** | 191/267 标签（71.5%）审核状态未明确；93.6% 标签处于待优化状态 | 未审核标签进入自动打标 = 质量不可控 | BI 分析混入未经验证的标签，结论可信度低 |
| 4 | 字典一致性 | **P1-严重** | 11 种 Tag ID 命名前缀无统一规则；通用表/品线表/映射表三源不一致 | Tag ID 无法自解释，LLM prompt 中语义映射困难 | 标签归类、下钻、聚合逻辑复杂 |
| 5 | 映射关系 | **P1-严重** | 映射表支撑标签 100% 空值、比较对象 99.25% 空值；202 个标签无映射记录 | 标签关联网络缺失，LLM 无法利用上下文共现提升准确率 | 无法做标签关联分析、归因链分析 |
| 6 | 打标输出 | **P1-严重** | 45 个 orphan 标签（TAG_ALC_* 为主）不在字典中，占样本标签种类的 3.2% | 闭集约束被 ALCHEmist 弱监督引擎打破，字典 ground truth 地位被侵蚀 | BI 查询出现 N/A，dim_tag 覆盖率下降 |
| 7 | 打标输出 | **P1-严重** | n_tags 计数逻辑与 labels 数组长度不一致（35 条 n_tags=0 但有 labels） | 下游按 n_tags 过滤会漏掉有效标签 | BI 聚合统计（如人均标签数）失真 |
| 8 | 打标输出 | **P1-严重** | 标签字段格式不一致：Phase 4 标签 7 个字段 vs LLM 标签 4 个字段 | ETL 和 BI 层需要复杂的格式适配逻辑 | 同一标签在不同来源下字段缺失 |
| 9 | BI Schema | **P2-中等** | dim_tag 仅 10 列，字典 37 列中 27 列未入库（风险等级、优化优先级、策略包等） | 不直接阻塞打标，但限制分析维度 | 无法按风险等级、策略包、优化优先级做分析切片 |
| 10 | 数据质量 | **P2-中等** | product_line 缺失率 6.7%；sentiment_calibrated 类型混用（REAL vs STRING） | 品线维度分析样本不足 | BI 查询需额外处理类型转换 |

**总体判断**：当前标签字段宽表**不能直接用于全自动打标**。必须在自动打标前完成字典治理（P0 三项）和映射关系补全（P1），否则自动打标产出的数据质量将低于当前人工审核水平，全维度分析将建立在不可靠的标签地基上。

---

## 二、标签字典（tag_dictionary_v4.5.xlsx）深度诊断

### 2.1 Sheet 结构全景

| Sheet | 行数 | 列数 | 角色 | 诊断结论 |
|-------|------|------|------|---------|
| 00_字段说明 | 53 | 7 | 元信息 | 26.4% 字段元信息缺失 |
| 01_通用标签主表 | 267 | 37 | 核心字典 | AIPL 失衡、情感失衡、审核缺失 |
| 02_吸奶器 | 82 | 43 | 品线扩展 | 与通用表重叠 8 个，结构一致 |
| 03_内衣服饰 | 57 | 43 | 品线扩展 | 与通用表重叠 7 个 |
| 04_家居家纺 | 52 | 43 | 品线扩展 | 与通用表重叠 1 个 |
| 05_母婴综合护理 | 70 | 43 | 品线扩展 | 与通用表重叠 7 个 |
| 06_喂养电器 | 53 | 43 | 品线扩展 | 与通用表重叠 13 个 |
| 07_智能母婴电器 | 64 | 43 | 品线扩展 | 与通用表重叠 5 个 |
| 08_映射关系表 | 402 | 30 | 标签-业务映射 | 支撑标签/比较对象 100% 空值 |
| 09_存量标签归档 | 457 | 10 | 历史归档 | 正常 |
| 10_Aspect库 | 55 | 9 | ABSA 方面库 | 仅 general 品类，格式非结构化 |

**全量唯一 tag_id：604 个**（通用 267 + 品线 378 - 重叠 41）。

### 2.2 P0 问题：AIPL 节点分布失衡

**现象**：
- 通用标签主表中 AIPL 节点只有 L1/L2/L3/L4，无 A/I/P 节点
- 但映射关系表 402 行覆盖 A/I/P1/P2/L1/L2/L3 全部 7 个节点
- 打标结果中 AIPL 分布：L1 77.7%、L3 18.9%、L2 1.5%、P1 0.8%、A 0.7%、P2 0.3%、I 0.1%

**根因**：
1. A/I/P 阶段的标签分散在映射关系表和品线表中，未统一进入通用标签主表
2. LLM 闭集打标使用的字典是「通用标签主表 + 品线表合并后的闭集」，但 A/I/P 标签数量本身不足
3. 原始 VOC 数据以商品评论为主（Amazon 194K / Trustpilot 100K），天然偏向使用后的 Loyalty 阶段表达

**影响**：
- AIPL 漏斗分析只能从 L1 开始，Awareness-Interest-Purchase 三段缺失
- 用户旅程归因只能回答「产品好不好用」，无法回答「怎么知道我们的」「为什么选我们」「为什么买」
- 营销内容优化、广告投放归因、种草效果评估等业务场景无标签支撑

**优化方案**：

```yaml
方案: AIPL 标签补齐专项行动
优先级: P0
执行方: @Data + @EcomOps + @Creative
输入:
  - 映射关系表中 A/I/P 节点标签（约 80-100 个）
  - 品线表中 A/I/P 相关标签
  - 当前 364K 打标结果中 A/I/P 阶段实际出现的高频关键词
步骤:
  1. 提取映射表 + 品线表中所有 A/I/P/P1/P2 节点标签，去重后合并到通用标签主表
  2. 对合并后的标签进行审核：@EcomOps 确认业务口径，@Creative 确认内容场景匹配度
  3. 用当前 364K 数据做关键词反向验证：若某 A/I/P 标签在 364K 中零命中，标记为「冷标签」并降级优先级
  4. 补齐后通用表预计 320-340 行，A/I/P 占比目标 15-20%
  5. 更新 LLM system prompt 的闭集，确保 A/I/P 标签可被抽取
验收:
  - AIPL 7 节点全部在通用表中有 ≥5 个标签
  - 打标结果 A/I/P 合计占比 ≥10%（vs 当前 2.1%）
  - 审核状态 100% 明确
```

### 2.3 P0 问题：情感极性严重失衡

**现象**：
- 通用标签主表：负向 210 (78.6%)、正向 42 (15.7%)、中性 13 (4.9%)、neutral 1、positive 1
- 品线表情感分布与通用表基本一致
- 但打标结果情感极性：1.0（强正向）67.6%、0.8 13.7%、负向合计仅约 12%

**根因**：
1. 标签定义以「问题/痛点/缺陷」为主（物流时效 35 个、产品核心性能 39 个，多为负面表达）
2. 正向表达在评论中被归并为少数通用标签（如 TAG_GEN_003 舒适体验、TAG_GEN_016 强烈推荐）
3. 情感极性的标注基于「标签本身的语义方向」（如「物流慢」=负向），而非「评论整体情感」

**矛盾点**：标签字典负向占 78.6%，但评论整体以 4-5 星为主（Amazon 评论天然正向偏斜），导致「标签语义负向」与「评论整体正向」冲突。

**影响**：
- 自动打标时 LLM 倾向于匹配字典中数量更多的负向标签，造成假阴性（正向评论被错标为负向标签）
- 情感趋势分析显示「负面问题多」，但实际评论整体满意度高，结论与业务体感矛盾
- Proxy NPS 计算可能因标签情感方向偏差而失真

**优化方案**：

```yaml
方案: 情感极性再平衡 + 双轨情感标注
优先级: P0
执行方: @Data
步骤:
  1. 对 267 个通用标签逐一审核情感极性：
     - 区分「标签语义方向」（intrinsic polarity）和「评论上下文情感」（contextual sentiment）
     - 例：「物流时效」标签本身中性，但在「物流时效慢」中表达负向，在「物流时效快」中表达正向
  2. 引入「情感极性 → 情感表达」二维标注：
     - 极性：正向 / 中性 / 负向（标签语义基线）
     - 表达：正面表达 / 负面表达 / 中性陈述（上下文情感）
  3. 补齐正向标签：从 364K 数据中挖掘高频正向表达，新增 30-50 个正向标签
     - 目标：正向标签占比从 15.7% 提升到 35-40%
  4. 建立「情感对冲」校验规则：
     - 若一条评论 rating=5 且 sentiment_polarity=1.0，但打标结果全部为负向标签 → 自动触发人工复核
  5. 更新 LLM prompt：要求 LLM 同时输出标签语义极性和上下文情感表达
验收:
  - 正向标签占比 ≥35%
  - 5 星评论中负向标签占比 ≤20%（当前预估 >50%）
  - 情感校准后 sentiment_polarity 与标签情感方向一致性 ≥90%
```

### 2.4 P0 问题：审核状态大面积缺失

**现象**：
- 通用表 267 行中，审核状态明确的仅 76 行（已通过 56 + 已审核-自动填充 20）
- 191 行（71.5%）审核状态为空
- 优化优先级：P2 142 (53.2%)、P1 90 (33.7%)、P0 18 (6.7%)、已完成 5、已删除 4

**根因**：
1. v4.5 治理聚焦在「字段空值补齐」和「orphan 回灌」，未处理审核状态
2. 标签数量从 v3.5 的 643 压缩到 v4.5 的 604，压缩过程中审核状态丢失
3. LLM 自动填充的 230 处空值标记为「已审核-自动填充」，但 LLM 审核 ≠ 人工业务审核

**影响**：
- 未审核标签进入自动打标管道，相当于用未经验证的规则标记数据
- BI 分析时无法区分「已验证标签」和「待验证标签」，结论可信度不可量化
- 业务方（如 @EcomOps）无法判断哪些标签可以驱动行动，哪些标签需要观望

**优化方案**：

```yaml
方案: 标签审核状态清零 + 分级审核流程
优先级: P0
执行方: @Data + @EcomOps + 各品线负责人
步骤:
  1. 对 191 个未审核标签按以下规则自动初筛：
     - 若在 364K 打标结果中出现次数 ≥100 → 候选「已通过」
     - 若出现次数 10-100 → 候选「待审核」
     - 若出现次数 <10 或零命中 → 候选「已归档」
  2. @EcomOps 对候选「已通过」标签进行业务审核（确认标签口径、业务动作可执行）
  3. 各品线负责人对品线专属标签进行 domain 审核（确认品类特异性合理）
  4. 审核结果写入字典，状态只能是：已通过 / 待审核 / 已归档 / 已删除
  5. 自动打标管道增加「审核状态过滤门」：仅允许「已通过」标签进入闭集
     - 待审核标签进入「影子打标」模式（输出但不写入主表，供审核参考）
  6. 建立审核状态变更审计日志（who/when/why）
验收:
  - 通用表审核状态 100% 明确
  - 自动打标闭集中「已通过」标签占比 100%
  - 待审核标签有独立的 shadow 表存储
```

### 2.5 P1 问题：Tag ID 命名空间混乱

**现象**：
- 11 种前缀：TAG_L1(59)/TAG_P2(51)/TAG_GEN(30)/TAG_I(25)/TAG_L2(18)/TAG_P1(17)/TAG_ZEN(12)/TAG_A(10)/TAG_SRV(10)/TAG_L3(9)/TAG_DEF(8)
- 命名规则：TAG_{节点缩写}_{序号}，但「节点缩写」与「AIPL 节点」不完全对应
  - 例：TAG_GEN 表示「通用标签」，但分布在 L1/L2/L3 多个节点
  - TAG_ZEN 表示 Zendesk 专用，但 Zendesk 只是数据源，不是业务节点
  - TAG_SRV 表示客服相关，但客服是渠道不是 AIPL 节点

**根因**：
1. Tag ID 命名经历了多轮迭代（v3.5→v4.5），每轮新增前缀但旧前缀未清理
2. 命名同时承载了「AIPL 节点」「数据来源」「业务领域」三层语义，导致编码混杂

**影响**：
- LLM prompt 中无法通过 Tag ID 前缀快速理解标签语义，必须加载完整字典
- BI 分析时按前缀聚合无意义（如 TAG_GEN 跨越多个 AIPL 节点）
- 新增标签时命名规则不确定，容易制造新的混乱

**优化方案**：

```yaml
方案: Tag ID 命名空间规范化（v5.0）
优先级: P1
执行方: @Data + @AIArchitect
步骤:
  1. 定义新的命名规则（四层编码）：
     TAG_{AIPL节点}_{业务域}_{序号}
     例：TAG_L1_PRD_001 = L1 节点 + 产品域 + 001号
     业务域枚举：PRD(产品), LOG(物流), SRV(客服), BRN(品牌), MKT(营销), PRC(价格), SAF(安全)
  2. 建立旧 ID → 新 ID 的映射表（保留旧 ID 作为 alias，兼容历史数据）
  3. 更新所有打标输出中的 tag_id（通过 ETL 批量替换，不改原始 jsonl）
  4. 更新 dim_tag 表 schema，新增 tag_id_v5 列，旧 tag_id 列保留为 deprecated
  5. 更新 LLM system prompt，让 LLM 输出新 ID 格式
  6. 验证：全量 364K 数据中无重复新 ID、无 orphan 新 ID
验收:
  - 新 ID 100% 符合 {AIPL}_{业务域}_{序号} 规则
  - 旧 ID 映射覆盖率 100%
  - BI 查询支持新旧 ID 双轨（6 个月过渡期）
```

### 2.6 P1 问题：映射关系表结构性缺陷

**现象**：
- 402 行映射关系，支撑标签 100% 空值，比较对象 99.25% 空值
- 消费者习惯关键词/原话短语 17.2% 空值，英文关键词 2.5% 空值
- 202 个标签（604-402）无映射记录
- 是否进入正式标签库 100% 是，但该列 10 个 null

**根因**：
1. 映射关系表是 v3.6 阶段产物，当时只完成了 A 节点（Awareness）标签的完整映射
2. I/P/L 节点标签的映射关系未补全
3. 「支撑标签」「比较对象」是高级分析功能（标签关联网络、竞品对比），尚未实施

**影响**：
- LLM 打标时无法利用「主标签-支撑标签」关系做上下文增强
- 无法做「当用户提到 X 时，通常也会提到 Y」的关联分析
- 无法做竞品对比分析（比较对象缺失）
- 全维度分析中的「归因链路」和「问题传导」分析无数据支撑

**优化方案**：

```yaml
方案: 映射关系表补全 + 标签关联网络构建
优先级: P1
执行方: @Data
步骤:
  1. 为 202 个无映射标签补全映射关系（基于标签主题聚类 + 业务方确认）
  2. 支撑标签填充：
     - 从 364K 打标结果中计算标签共现矩阵（tag_i 和 tag_j 同时出现的 review 数）
     - 共现频次 TOP 3 的其他标签作为支撑标签候选
     - @EcomOps 确认业务合理性后写入映射表
  3. 比较对象填充：
     - 对「品牌提及」「竞品对比」类标签，明确比较对象（如 Momcozy vs Spectra vs Medela）
     - 对「价格价值感」类标签，比较对象设为「品类均价」或「竞品价格」
  4. 消费者习惯关键词补齐：
     - 从 364K 数据中挖掘每个标签对应的高频原话短语（TF-IDF top 5）
     - LLM 校验后写入映射表
  5. 映射表入库：
     - 新建 dim_tag_mapping Postgres 表，支撑 BI 关联查询
     - 或扩展 dim_tag 表，新增支撑标签数组列
验收:
  - 映射表覆盖率 100%（604/604 标签）
  - 支撑标签非空率 ≥80%
  - 比较对象非空率 ≥50%（仅对需要比较的标签）
  - 共现网络在 BI 中可查询
```

### 2.7 P2 问题：Aspect 库薄弱且格式非结构化

**现象**：
- 仅 55 行，category 只有 1 个值（general）
- 关联 tag_ids 格式为自由文本：`TAG_SRV_08:1.7|TAG_GEN_C001:1.2|...`
- 出现次数仅 14-28 次，样本量不足以支撑统计推断

**影响**：
- ABSA 方面抽取的 ground truth 不足，LLM 难以学习方面-观点关系
- 方面到标签的映射靠人工维护，无法自动进化
- 全维度分析中的「方面维度」分析受限

**优化方案**：

```yaml
方案: Aspect 库自动化扩展
优先级: P2
执行方: @Data
步骤:
  1. 用 LLM 对 364K 评论做方面抽取（prompt 要求输出 aspect + opinion + sentiment）
  2. 聚类抽取结果，合并同义 aspect（如 "shipping" / "delivery" / "物流" → 配送）
  3. 计算 aspect-tag 关联强度（条件概率 P(tag|aspect)），自动填充关联映射
  4. 按品类拆分 category（general → 吸奶器 / 内衣服饰 / 家居家纺 等）
  5. 出现次数通过全量数据自动统计，无需人工维护
验收:
  - Aspect 库 ≥200 行
  - category 覆盖全部 6 个品线
  - 关联 tag_ids 格式化为结构化数组
```

---

## 三、打标输出（phase6_d9_filtered.jsonl）深度诊断

### 3.1 P1 问题：45 个 Orphan 标签侵蚀字典 ground truth

**现象**：
- 10,000 样本中出现 45 个不在字典中的标签
- 主要是 TAG_ALC_*（ALCHEmist 弱监督引擎生成）：25 个
- 其他 orphan：20 个
- 出现频次总计 244 次（占样本标签总量的 0.8%）

**根因**：
1. D-01 决策「闭集为主」只对 LLM 引擎有效，ALCHEmist 作为 Phase 4 遗留的弱监督引擎不受闭集约束
2. ALCHEmist 按产品品类生成标签（如 TAG_ALC_PREGNANCY_PILLOW），这些标签未进入字典
3. 合并逻辑 `merge_phase4_phase5_llm.py` 未过滤字典外标签

**影响**：
- dim_tag 表查询时出现 N/A（orphan tag_id 无对应记录）
- BI 看板中「标签分布」图表出现未知标签，业务方无法解读
- 字典作为 ground truth 的权威性被侵蚀

**优化方案**：

```yaml
方案: Orphan 标签治理三步走
优先级: P1
执行方: @Data + @EcomOps
步骤:
  1. 对 45 个 orphan 标签进行分类：
     - 高频 orphan（出现 ≥10 次）：评估是否升级为正式标签
     - 低频 orphan（出现 <10 次）：标记为 deprecated，合并到最接近的正式标签
  2. 高频 orphan 的审核流程：
     - @EcomOps 确认业务价值
     - 若确认有价值 → 按 v5.0 命名规则新增正式标签，补全字典所有字段
     - 若确认无价值 → 建立「orphan → 正式标签」的映射规则（如 TAG_ALC_PREGNANCY_PILLOW → TAG_L1_PRD_xxx）
  3. 在打标管道增加「字典过滤门」：
     - phase5_unified_labeler.py 输出时，过滤所有 tag_id ∉ 字典闭集的标签
     - 被过滤的标签写入 orphan_log（供月度开集采样分析）
     - 月度 cron 分析 orphan_log，决定是否开集扩展字典
验收:
  - 打标结果中 orphan 标签占比 = 0%
  - orphan_log 每月产出一次分析报告
  - dim_tag 覆盖率 = 100%（所有 voc_label.tag_id 都有 dim_tag 对应记录）
```

### 3.2 P1 问题：n_tags 计数与 labels 数组长度不一致

**现象**：
- 10,000 样本中 35 条 n_tags=0（0.4%），但部分记录 labels 数组非空
- 样本中 n_tags 分布：0(35)/1(3108)/2(3107)/3(2060)/4(1017)/5(430)/6(169)/7(51)/8(18)/9(5)
- 平均 n_tags = 2.1

**根因**：
1. `n_tags` 是 Phase 4 规则引擎的计数，Phase 5/6 LLM 新增标签后未同步更新 n_tags
2. Method C 后处理过滤可能删除了标签但未更新 n_tags
3. 某些标签来源（如 alchemist）可能未计入 n_tags

**影响**：
- 下游按 n_tags=0 过滤会漏掉有效标签
- BI 统计「人均标签数」「标签覆盖率」时失真

**优化方案**：

```yaml
方案: n_tags 一致性修复
优先级: P1
执行方: @dev
步骤:
  1. 修复 phase6_d9_filtered.jsonl：遍历所有记录，将 n_tags 重置为 len(labels)
  2. 修复 unified_labeler.py：在输出前统一计算 n_tags = len(labels)
  3. 增加 schema validator 检查项：n_tags == len(labels)，不一致时 pipeline fail
  4. 重新跑 ETL → Superset，更新 BI 数据
验收:
  - n_tags == len(labels) 100% 一致
  - Schema validator 新增该检查项并通过
```

### 3.3 P1 问题：标签字段格式不一致

**现象**：
- Phase 4 标签格式：tag_id, tag_en, tag_cn, aipl_node, sentiment_preset, sentiment_calibrated, confidence
- Phase 5/6 LLM 标签格式：tag_id, confidence, evidence, source
- 同一标签在不同来源下字段缺失（如 LLM 标签缺少 tag_en/tag_cn/aipl_node）

**根因**：
1. Phase 4 和 Phase 5/6 由不同引擎生成，未统一输出 schema
2. LLM 输出只包含 tag_id 和 confidence，标签元信息由下游 lookup dim_tag 补齐
3. 但部分场景（如 alchemist 标签）既非 Phase 4 也非 Phase 5/6，格式自成一派

**影响**：
- ETL 需要复杂的分支逻辑处理不同格式
- BI 查询时某些标签缺少 tag_cn，显示为 tag_id（业务方不可读）
- 数据血缘追踪困难

**优化方案**：

```yaml
方案: 统一标签输出 schema（v5.0）
优先级: P1
执行方: @dev + @Data
schema:
  tag_id: str              # 标签唯一标识
  tag_en: str              # 英文标签名（lookup from dim_tag）
  tag_cn: str              # 中文标签名（lookup from dim_tag）
  aipl_node: str           # AIPL 节点（lookup from dim_tag）
  sentiment_preset: str    # 标签语义极性（lookup from dim_tag）
  sentiment_calibrated: float  # 上下文校准后的情感（-1~1）
  confidence: float        # 打标置信度（0~1）
  confidence_original: float   # 校准前置信度
  confidence_lift: float   # 校准提升量
  source: str              # 标签来源：phase4 / llm_v4flash / llm_kimi / alchemist / consensus
  evidence: str            # LLM 抽取的证据文本片段
  label_source_version: str   # 字典版本号（v4.5 / v5.0）
步骤:
  1. 修改 unified_labeler.py：无论哪个引擎生成标签，输出前统一 lookup dim_tag 补齐缺失字段
  2. 对历史数据（364K）批量修复：用 dim_tag 补齐缺失的 tag_en/tag_cn/aipl_node
  3. Schema validator 新增字段完整性检查：必填字段（tag_id/tag_cn/aipl_node/confidence/source）不得为空
验收:
  - 所有标签字段完整率 100%
  - BI 中无 tag_id 直接显示的情况（全部显示 tag_cn）
```

### 3.4 P2 问题：Product Line 缺失率 6.7%

**现象**：
- 10,000 样本中 669 条 product_line=None（6.7%）
- 缺失记录主要集中在 Amazon 数据源（asin 存在但 product_line 未映射）

**根因**：
1. ASIN → product_line 的映射表不完整
2. Amazon 竞品数据（amazon_competitor）中部分 ASIN 不在 Momcozy 产品目录中
3. ETL 时未做 ASIN 的模糊匹配（如变体 ASIN）

**影响**：
- 品线维度分析样本减少 6.7%
- 某些品线的指标可能因样本不足而波动

**优化方案**：

```yaml
方案: ASIN → Product Line 映射补全
优先级: P2
执行方: @Data
步骤:
  1. 提取所有 product_line=None 的 unique ASIN
  2. 用 Amazon SP-API 或爬虫获取这些 ASIN 的类目信息（category / browse node）
  3. 基于类目信息自动映射到 Momcozy 品线（规则引擎 + LLM 校验）
  4. 对无法映射的 ASIN，标记 product_line="other"（而非 None）
  5. 建立 ASIN 映射表的自动更新机制（新 ASIN 进入时自动分类）
验收:
  - product_line 缺失率 <1%
  - 无 None 值，全部有明确品线归属
```

---

## 四、BI Schema 深度诊断

### 4.1 dim_tag 字段覆盖不足

**现象**：
- dim_tag 表仅 10 列：tag_id, tag_cn, tag_en, aipl_node, polarity, dept_owner, biz_action, strategy_pkg, is_general, audit_status
- 字典 37 列中 27 列未入库，包括：风险等级、优化优先级、问题诊断、品类特异性指数、故事线关联、Proxy NPS 贡献等

**影响**：
- BI 分析维度受限，无法按风险等级切片、无法按优化优先级排序、无法按故事线聚合
- 全维度分析沦为「标签频次统计」，缺乏业务语义深度

**优化方案**：

```yaml
方案: dim_tag 全字段入库
优先级: P2
执行方: @dev
步骤:
  1. 扩展 dim_tag 表 schema，新增字段：
     - risk_level TEXT
     - optimization_priority TEXT
     - issue_diagnosis TEXT
     - category_specificity_index REAL
     - storyline_link TEXT
     - proxy_nps_contribution TEXT
     - product_line TEXT
     - collab_dept TEXT
     - metric_direction TEXT
     - default_priority TEXT
  2. 修改 ETL 脚本 etl_to_postgres.py，全量导入字典字段
  3. 修改 Superset dataset 定义，暴露新增字段
  4. 新增 BI 图表：
     - 风险等级分布热力图
     - 优化优先级瀑布图
     - 故事线关联桑基图
验收:
  - dim_tag 字段数 ≥25
  - BI 中新增 3 个分析维度图表
```

### 4.2 缺乏标签关系表

**现象**：
- 无 tag-tag 共现关系表
- 无 tag-aspect 映射表（ Aspect 库未入库）
- 无 tag-review 的权重/证据表（voc_label 缺少 evidence 列）

**影响**：
- 无法做「标签关联分析」（提到 A 的用户也提到 B 的概率）
- 无法做「归因链路分析」（从 Awareness → Interest → Purchase → Loyalty 的标签流转）
- 无法做「证据溯源」（为什么这条评论被打上这个标签）

**优化方案**：

```yaml
方案: 新增标签关系表族
优先级: P2
执行方: @dev + @Data
表设计:
  dim_tag_cooccurrence:
    - tag_id_a, tag_id_a_cn
    - tag_id_b, tag_id_b_cn
    - cooccurrence_count
    - conditional_prob_a_given_b
    - mutual_information
    - product_line
    - updated_at
  
  dim_tag_aspect:
    - aspect_id
    - aspect_cn
    - tag_id
    - tag_cn
    - association_strength
    - sample_count
  
  voc_label_evidence:
    - review_id
    - tag_id
    - evidence_text
    - evidence_position  # 证据在评论中的位置（start, end）
    - confidence
    - source
步骤:
  1. 从 364K 数据计算 tag-tag 共现矩阵，写入 dim_tag_cooccurrence
  2. 从 Aspect 库和 LLM 抽取结果构建 dim_tag_aspect
  3. 修改打标管道，将 LLM 的 evidence 字段写入 voc_label_evidence
  4. 在 Superset 中新增「标签关联网络」和「证据溯源」图表
验收:
  - 共现表覆盖全部 604 标签的两两组合（或对角线稀疏矩阵）
  - BI 中可查询「与 X 标签最常共现的 Top 10 标签」
  - 点击 BI 中的标签可查看原始证据文本
```

---

## 五、项目架构诊断

### 5.1 闭集约束执行不一致

**现象**：
- D-01 决策要求「闭集为主 + 月度开集采样 5%」，LLM 确实 0 非法 tag_id
- 但 ALCHEmist 弱监督引擎生成 TAG_ALC_* 系列，不受闭集约束
- 规则引擎（Phase 4）生成的 brand_label、negative_defect、general_tag 也不校验字典

**根因**：
- 闭集约束只在 LLM 层实现（Pydantic 后置校验），未在整个 unified_labeler 中统一执行
- Phase 4 规则引擎是「免费打底」层，为兼容未做硬约束

**影响**：
- 字典 ground truth 地位被架空，实际运行的是「字典 + orphan 混合集」
- 字典治理的成果（v4.5 清理 56 行脏数据）被 orphan 标签抵消

**优化方案**：

```yaml
方案: 全管道闭集硬约束
优先级: P1
执行方: @dev
步骤:
  1. 在 unified_labeler.py 的 merge 阶段增加「全局闭集过滤」：
     - 无论标签来自 phase4 / alchemist / llm / 任何来源
     - 输出前统一校验 tag_id ∈ dim_tag.tag_id
     - 非法标签写入 orphan_log，不进入主输出
  2. 对 Phase 4 规则引擎的 brand_label / negative_defect / general_tag 源：
     - 维护这些规则到字典 tag_id 的映射表
     - 规则输出时通过映射表转换为字典内的正式 tag_id
     - 无法映射的规则输出标记为 deprecated，逐步淘汰
  3. 对 ALCHEmist：
     - 限制 ALCHEmist 只生成品类 aspect（如 pregnancy_pillow）
     - aspect 必须经过「aspect → 正式 tag_id」映射后才能进入标签输出
     - ALCHEmist 输出改为写入 aspect 中间表，不作为最终标签
验收:
  - unified_labeler 输出中非法 tag_id 数量 = 0
  - orphan_log 每月增长量 = 0（除非开集扩展）
```

### 5.2 数据血缘与版本追踪缺失

**现象**：
- 打标结果中有 label_sources 数组（如 ["phase4", "llm_v4flash"]），但无字典版本号
- dim_tag 表无 loaded_at 以外的版本信息
- 无法回答「这条评论的标签是用 v4.1 还是 v4.5 字典打的」

**影响**：
- 字典升级后，历史打标结果与当前字典可能不一致
- BI 分析跨时间对比时，标签口径变化导致结论不可比
- 无法做 A/B 测试（用不同字典版本打标同批数据）

**优化方案**：

```yaml
方案: 数据血缘 + 字典版本追踪
优先级: P2
执行方: @dev
步骤:
  1. 在 voc_review 表新增 dict_version 列（如 "v4.5"）
  2. 在 voc_label 表新增 dict_version 列
  3. 每次字典升级后，记录版本变更日志（变更标签清单、变更原因、变更人）
  4. 支持「字典版本对比」功能：同一批数据用 v4.5 和 v5.0 分别打标，diff 分析标签变化
  5. BI 筛选器增加「字典版本」维度，支持跨版本对比
验收:
  - 每条 review/label 都有 dict_version
  - 字典版本变更日志可查询
  - BI 支持按字典版本切片
```

---

## 六、自动打标方案设计（下一步目标）

基于以上诊断，自动打标不能直接在 v4.5 上启动。必须先完成「字典治理 Phase 8」，再进入「自动打标 Phase 9」。

### 6.1 Phase 8：字典治理（预计 2 周）

| 周 | 任务 | 负责 | 输入 | 输出 |
|---|------|------|------|------|
| W1 | AIPL 标签补齐 + 情感极性再平衡 + 审核状态清零 | @Data + @EcomOps | v4.5 字典 + 364K 打标结果 + 映射表 | v5.0-rc 字典（通用表 320+ 行，审核状态 100%） |
| W1 | Tag ID 命名规范化 + 映射关系补全 | @Data + @AIArchitect | v5.0-rc 字典 + 共现矩阵 | v5.0 字典 + 新 ID 映射表 + 完整映射关系表 |
| W2 | Aspect 库扩展 + 全字段入库 + 关系表构建 | @dev + @Data | v5.0 字典 + 364K 数据 | 扩展 dim_tag + dim_tag_cooccurrence + dim_tag_aspect |
| W2 | 全管道闭集硬约束 + n_tags 修复 + schema 统一 | @dev | unified_labeler.py + schema validator | 零 orphan 输出 + 统一 schema + 100% 字段完整 |
| W2 | 全量重打 + 质量验证 | @Data + @QA | 364K 评论 + v5.0 字典 + 新管道 | phase8_v50_full_labeled.jsonl + Gate 报告 |

### 6.2 Phase 9：自动打标引擎（预计 2 周）

**架构设计**：

```
输入层：新评论（Amazon/Trustpilot/Zendesk/Reddit/客服工单）
  ↓
预处理：语种识别 → 质量评分 → 去重 → ASIN → Product Line 映射
  ↓
自动打标引擎（并行）：
  ├── 规则引擎（Phase 4 遗留，快速打底）
  ├── LLM 闭集打标（DeepSeek-V4-Flash 主 + Kimi-K2.6 兜底）
  └── ABSA 方面抽取（Aspect 库匹配）
  ↓
合并层：unified_labeler（v5.0）→ 闭集过滤 → 置信度校准 → 冲突消解
  ↓
后处理：Method C 过滤 → n_tags 校验 → schema 校验 → orphan 检测
  ↓
输出层：voc_review + voc_label + voc_label_evidence
  ↓
BI 层：Superset 实时刷新（增量 ETL）
```

**关键设计决策**：

1. **增量 vs 全量**：
   - 自动打标采用增量模式（只处理新评论）
   - 字典升级时触发全量重打（用 v5.0 重打历史 364K）
   - 全量重打通过 Temporal Workflow 编排，分批执行

2. **置信度阈值动态调整**：
   - 不同品线设置不同阈值（吸奶器阈值可高于家居家纺，因吸奶器评论更长）
   - 不同 AIPL 节点设置不同阈值（L1 使用体验标签阈值可低于 A 品牌认知标签）
   - 阈值由 @Data 基于历史准确率数据建模，每月 review

3. **人机协同审核**：
   - 高置信度（≥0.9）标签自动通过
   - 中置信度（0.7-0.9）标签进入「影子模式」（写入但默认不展示）
   - 低置信度（<0.7）标签进入人工审核队列
   - 审核结果反馈到 LLM prompt（few-shot learning）

4. **实时性目标**：
   - Amazon 评论：T+1 自动打标（每日增量）
   - Trustpilot：T+1 自动打标
   - Zendesk 工单：T+0（工单关闭后即时触发）
   - Reddit：T+7（低频，周度批量）

### 6.3 Phase 10：全维度分析洞察（预计 3 周）

基于自动打标产出，构建全维度分析能力：

| 分析维度 | 指标/图表 | 业务价值 | 依赖 |
|---------|----------|---------|------|
| AIPL 漏斗 | Awareness→Interest→Purchase→Loyalty 转化率 | 用户旅程诊断 | AIPL 标签补齐 |
| 情感趋势 | 正向/负向/中性标签占比时序 | 品牌健康度监测 | 情感极性再平衡 |
| 标签关联 | 共现网络 + 关联规则 | 问题传导分析 | 映射关系补全 + 共现表 |
| 品线对比 | 各品线标签分布雷达图 | 品类差异化洞察 | product_line 零缺失 |
| 竞品对比 | Momcozy vs 竞品标签对比 | 竞争定位分析 | 比较对象填充 |
| 归因链路 | 从广告 → 内容 → 购买 → 复购的标签流转 | 营销效果归因 | AIPL 标签 + 故事线关联 |
| 预警监测 | 负向标签突增自动告警 | 危机预警 | 审核状态 100% + 风险等级 |
| 证据溯源 | 点击标签查看原始评论证据 | 结论可解释 | voc_label_evidence 表 |
| 画像洞察 | 55 画像标签 × 行为标签交叉分析 | 精准运营 | 画像标签准确率验证 |
| 预测分析 | 基于标签序列预测复购/流失 | 用户生命周期管理 | 时间序列标签数据积累 |

---

## 七、总结与行动清单

### 7.1 阻塞项清单（必须先完成）

| # | 阻塞项 | 优先级 | 负责 | 预计时间 | 解锁能力 |
|---|--------|--------|------|---------|---------|
| 1 | AIPL 标签补齐（通用表覆盖 A/I/P 节点） | P0 | @Data + @EcomOps | 3 天 | AIPL 漏斗分析 |
| 2 | 情感极性再平衡（正向标签 ≥35%） | P0 | @Data | 3 天 | 可信情感趋势分析 |
| 3 | 审核状态清零（100% 明确） | P0 | @Data + @EcomOps | 5 天 | 自动打标质量可控 |
| 4 | 全管道闭集硬约束（零 orphan） | P1 | @dev | 2 天 | 字典 ground truth 恢复 |
| 5 | n_tags 一致性修复 | P1 | @dev | 1 天 | BI 统计准确 |
| 6 | 统一标签输出 schema | P1 | @dev + @Data | 2 天 | ETL 简化 + BI 字段完整 |
| 7 | 映射关系补全（支撑标签/比较对象） | P1 | @Data | 5 天 | 标签关联分析 |
| 8 | Tag ID 命名规范化 | P1 | @Data + @AIArchitect | 3 天 | 标签体系可扩展 |

### 7.2 总时间估算

- **字典治理 Phase 8**：2 周（P0+P1 全部完成）
- **自动打标 Phase 9**：2 周（引擎开发 + 全量重打验证）
- **全维度分析 Phase 10**：3 周（BI 图表开发 + 分析模板沉淀）
- **总计**：7 周（约 2 个月）

### 7.3 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| A/I/P 标签在评论中天然稀疏，补齐后覆盖率仍低 | 高 | AIPL 漏斗分析失真 | 引入非评论数据源（广告点击、社媒互动、搜索行为）补充 A/I/P 标签 |
| 情感极性再平衡需要新增大量正向标签，导致标签膨胀 | 中 | 闭集变大，LLM 准确率下降 | 正向标签控制在 35-40%，优先复用现有正向表达，不追求 1:1 对称 |
| 全量重打 364K 数据成本过高 | 中 | 项目延期 | 采用增量验证：先用 5K 子集验证 v5.0 字典质量，再决定是否全量重打 |
| @EcomOps 审核资源不足 | 中 | 审核状态清零延期 | 建立「业务方审核 SOP + 审核模板」，降低单次审核时间到 5 分钟/标签 |
| Superset BI 性能瓶颈（364K + 标签膨胀后记录数增长） | 低 | BI 查询变慢 | 预先评估：当前 364K × 平均 2.1 标签 = 765K voc_label 行，扩容后预计 <2M，PostgreSQL 可支撑 |

---

> **报告交付状态**：已完成，等待 @Cindy 拆解为具体任务，@bestore-pray 确认优先级和资源投入。
