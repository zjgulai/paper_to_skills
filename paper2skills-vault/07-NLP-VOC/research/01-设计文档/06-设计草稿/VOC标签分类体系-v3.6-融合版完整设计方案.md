# VOC 标签分类体系 v3.6 融合版 — 完整设计方案

> 基座：v3.5 标签表（503 标签，10 Sheet）
> 蓝图：v0.2 方案（56 标签，动态规则层）
> 目标：一次性解决标签分类体系的四大核心问题
> 日期：2026-04-23

---

## 执行摘要

本文档解决四个核心问题：

| # | 问题 | 根因 | v3.6 解决方案 |
|---|------|------|--------------|
| 1 | AIPL 极度偏斜 L1 | 品线表数据源单一（仅售后评论/工单）+ A/I/P 标签被困在通用表 | 数据源分层模型 + 标签下沉策略 + 动态锚定规则 |
| 2 | 17 低分 + 84 待审核 + 20 缺失字段 | 标签质量缺乏自动化监控 + 算法发现标签未审核 | 逐个修复 + 审核策略 + 自动化监控机制 |
| 3 | 无多语言映射 | 仅英文关键词 | 分级覆盖策略（P0全量/P1高频/P2中频/P3模板化） |
| 4 | 正向标签严重不足 | 品线表负向占 83-87% | 通用正向 40 个 + 品线正向 60+ 个 |

**核心设计原则**：v3.5 的 503 标签为**基座不动**，叠加 v0.2 的**动态规则层**，形成六层架构的 v3.6。

---

## 第一部分：AIPL 偏斜问题深度剖析与解决方案

### 1.1 根因分析（三层）

#### 根因 Layer 1：数据源与 AIPL 的天然绑定

```
数据源类型                    天然产生的 AIPL 标签
─────────────────────────────────────────────────────────
售后评论 + 客服工单    →      L1（首购使用）+ L2（持续使用）
订单数据 + 物流跟踪    →      P2（购买决策）
客服售前 + 问答社区    →      P1（评估）+ I（兴趣）
社媒广告 + 品牌搜索    →      A（认知）+ I（兴趣）
测评内容 + KOL 内容    →      I（兴趣）+ P1（评估）
Trustpilot + 推荐评论  →      L3（推荐）
```

**v3.5 品线表的数据源分析**：

| 品线 | 主要 VOC 载体 | 占比 | 产生的 AIPL |
|------|-------------|------|------------|
| 吸奶器 | 电商评论+社媒+客服工单 | 94% | L1/L2 |
| 内衣服饰 | 电商评论+社媒+客服工单 | 96% | L1/L2 |
| 家居家纺 | 电商评论+社媒+客服工单 | 93% | L1/L2 |
| 母婴综合护理 | 电商评论+社媒+客服工单 | 93% | L1/L2 |
| 喂养电器 | 电商评论+社媒+客服工单 | 100% | L1/L2 |
| 智能母婴电器 | 电商评论+社媒+客服工单 | 98% | L1/L2 |

**结论**：品线表的 VOC 载体几乎 100% 来自**售后反馈**，天然只能产生 L1/L2 标签。

#### 根因 Layer 2：标签设计方法论的分裂

```
┌─────────────────────────────────────────────────────────────┐
│                    v3.5 标签设计方法论                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  通用表（191标签）                                            │
│  ├── A/I/P1 标签：47个，来源="新建"，适用品线="通用"            │
│  │   → 由市场/品牌团队设计，关注营销漏斗                        │
│  ├── L1/L2/L3 标签：69+18+12=99个，来源混合                   │
│  │   → 跨品线通用售后标签                                      │
│  └── P2 标签：39个，来源="存量清洗"                            │
│      → 来自订单/物流系统                                       │
│                                                             │
│  品线表（312标签）                                            │
│  ├── 来源="存量清洗"：124个（65%）                            │
│  │   → 从历史业务分类清洗而来，历史分类全部来自售后场景          │
│  └── 来源="新建"：少量                                         │
│      → 补充的品线专属售后标签                                  │
│                                                             │
│  问题：A/I/P1 标签在通用表，品线表看不到                         │
│       品线表只有 L1/L2，AIPL 漏斗是"空心"的                    │
└─────────────────────────────────────────────────────────────┘
```

**关键洞察**：
- 通用表的 A/I/P1 标签（47 个）全部被标记为"适用产品品线=通用"
- 这意味着设计时的假设是："品牌陌生""参数说明清楚"等标签跨品线通用，不需要在每个品线表中重复
- 但后果是：**按品线做 AIPL 漏斗分析时，A/I/P 阶段完全空白**

#### 根因 Layer 3：组织架构的壁垒

| 标签类型 | 维护团队 | 关注阶段 | 数据源 | 所在表格 |
|----------|----------|----------|--------|----------|
| 品牌认知类 | 品牌市场中心/品牌部 | A | 社媒/广告 | 通用表 |
| 信息查询类 | 电商运营部 | I/P1 | 评论/Q&A | 通用表 |
| 产品问题类 | 产品中心/品线 | L1 | 评论/工单 | 品线表 |
| 售后问题类 | 全球客服与体验中心 | L2 | 工单 | 品线表 |
| 物流问题类 | 物流运营部 | P2 | 订单系统 | 通用表 |

**结论**：不同团队维护不同阶段的标签，标签分散在不同表格中，没有按"用户旅程"统一组织。

### 1.2 解决方案一：数据源分层模型（Data Source Layering）

**核心思路**：为每个 AIPL 阶段配置专属的数据源，打破"品线表 = 售后评论"的单一绑定。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    数据源分层模型（v3.6 新增）                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  A 认知阶段          I 兴趣阶段           P1 评估阶段                │
│  ┌──────────┐      ┌──────────┐       ┌──────────┐                │
│  │社媒帖子   │      │电商Q&A    │       │客服售前对话│                │
│  │广告评论   │      │测评内容   │       │问答社区   │                │
│  │品牌搜索词 │      │Reddit讨论 │       │对比测评   │                │
│  │KOL内容   │      │YouTube评测│       │产品页浏览 │                │
│  └────┬─────┘      └────┬─────┘       └────┬─────┘                │
│       │                 │                  │                        │
│       └─────────────────┼──────────────────┘                        │
│                         ▼                                           │
│              ┌─────────────────────┐                                │
│              │   A/I/P1 标签萃取    │                                │
│              │   （新增数据源接入）   │                                │
│              └─────────────────────┘                                │
│                         │                                           │
│  P2 决策阶段            ▼              L1 首购阶段    L2 持续阶段   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐   ┌──────────┐  │
│  │订单数据   │    │A/I/P1萃取结果│    │电商评论   │   │客服工单   │  │
│  │退货留言   │ →  │+ P2订单标签  │ →  │社媒帖子   │   │追评      │  │
│  │物流跟踪   │    │+ L1售后标签  │    │客服工单   │   │复购行为   │  │
│  └──────────┘    └──────────────┘    └──────────┘   └──────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**具体实施**：

| AIPL 阶段 | 新增数据源 | 当前状态 | v3.6 行动 |
|-----------|-----------|----------|----------|
| A（认知） | TikTok/Instagram 品牌内容评论、Google 品牌搜索词、Facebook 广告互动 | ❌ 未接入 | 接入社媒 listening 平台 |
| I（兴趣） | Amazon Q&A、 Reddit r/breastfeeding、 YouTube 测评评论区、小红书种草笔记 | ⚠️ 部分接入（测评内容） | 全面接入，按品线配置 subreddit/话题 |
| P1（评估） | 客服售前对话、 产品页热力图（停留区域）、 对比测评内容 | ⚠️ 部分接入（客服售前） | 接入客服售前系统，配置售前标签 |
| P2（决策） | 订单系统（取消原因）、 物流跟踪数据、 退货留言 | ⚠️ 已接入（退货/物流） | 已覆盖，优化即可 |
| L1（首购） | 电商评论、 社媒帖子、 客服工单 | ✅ 已接入 | 已覆盖，优化即可 |
| L2（持续） | 客服工单、 追评、 复购行为数据 | ⚠️ 部分接入 | 接入复购行为数据 |
| L3（推荐） | Trustpilot、 推荐评论（"recommended to"）、 社媒分享（@品牌） | ⚠️ 部分接入 | 接入社媒分享监听 |

### 1.3 解决方案二：标签下沉策略（Tag Sink Strategy）

**核心思路**：将通用表的 A/I/P 标签下沉到各品线表，并针对品线做**个性化适配**。

**不是简单复制，而是"下沉+适配"**：

```
通用表标签 "品牌陌生"
    ↓ 下沉到吸奶器品线
    → "品牌陌生-吸奶器"：用户在搜索吸奶器时首次接触 Momcozy 品牌
    → 适用画像：first_time_parent, research_driven
    → 竞品提及：Spectra/Medela/Willow/Elvie
    → 载体：Google 搜索词、Amazon 搜索词
    
    ↓ 下沉到内衣服饰品线  
    → "品牌陌生-文胸"：用户在搜索哺乳文胸时首次接触 Momcozy 品牌
    → 适用画像：first_time_parent, price_sensitive
    → 竞品提及：Kindred Bravely/Hofish
    → 载体：Google 搜索词、Amazon 搜索词
```

**各品线需要下沉的 A/I/P 标签清单**：

#### 吸奶器品线（当前 L1=85%，目标 L1=50%）

| AIPL | 下沉标签 | 品线适配 | 新增数据源 |
|------|----------|----------|-----------|
| A | 品牌陌生 | +竞品提及 Spectra/Medela/Willow/Elvie | Google/Amazon 搜索词 |
| A | 内容打中痛点 | +"背奶""静音""穿戴"等吸奶器场景痛点 | TikTok/小红书内容 |
| A | 真实妈妈种草 | +吸奶器使用场景的妈妈推荐 | 社媒帖子 |
| I | 参数说明清楚 | +吸力参数/噪音分贝/续航时间等 | Amazon Q&A |
| I | 评价真实可参考 | +吸奶器专属评价维度 | Amazon 评论 |
| I | 实测证据充分 | +吸力测试视频/分贝测试 | YouTube 测评 |
| P1 | 核心卖点清晰 | +"穿戴式""静音""免手扶"等卖点 | 产品页浏览数据 |
| P1 | 对比信息充分 | +Spectra vs Momcozy 对比 | Reddit/测评内容 |
| P1 | 认证背书可信 | +FDA/FCC/医疗级认证 | 产品页/客服售前 |
| P1 | 物超所值 | +"比Willow便宜一半" | 评论/Reddit |

**吸奶器品线 AIPL 目标分布**：

| AIPL | 当前 | 目标 | 变化 |
|------|------|------|------|
| A | 0% | 8% | +6个标签 |
| I | 0% | 15% | +11个标签 |
| P1 | 8% | 12% | +3个标签 |
| P2 | 6% | 8% | +0个（已有） |
| **L1** | **85%** | **45%** | **-16个（相对比例下降）** |
| L2 | 1% | 7% | +4个标签 |
| L3 | 0% | 5% | +3个标签 |

#### 内衣服饰品线（当前 L1=81%，目标 L1=50%）

| AIPL | 下沉标签 | 品线适配 |
|------|----------|----------|
| A | 品牌陌生 | +竞品提及 Kindred Bravely/Hofily |
| A | 内容打中痛点 | +"单手穿脱""哺乳便利""孕期尺码变化" |
| I | 参数说明清楚 | +尺码表、面料成分、支撑力说明 |
| I | 评价真实可参考 | +文胸专属评价（尺码/支撑/穿脱） |
| P1 | 核心卖点清晰 | +"单手开合""无痕设计""孕期+哺乳期两用" |
| P1 | 认证背书可信 | +OEKO-TEX 认证 |
| P1 | 物超所值 | +"比 maternity store 便宜" |

#### 喂养电器品线（当前 L1=96%，目标 L1=50%）

| AIPL | 下沉标签 | 品线适配 |
|------|----------|----------|
| A | 品牌陌生 | +暖奶器/消毒器/调奶器场景 |
| I | 参数说明清楚 | +温度范围/容量/消毒方式/材质 |
| I | 评价真实可参考 | +加热均匀性/操作便捷性评价 |
| P1 | 核心卖点清晰 | +"精准温控""UV消毒""一键操作" |
| P1 | 认证背书可信 | +食品安全认证/医疗器械认证 |

#### 智能母婴电器品线（当前 L1=100%，目标 L1=45%）

| AIPL | 下沉标签 | 品线适配 |
|------|----------|----------|
| A | 品牌陌生 | +婴儿监视器品类品牌认知 |
| A | 广告过度打扰 | +App 推送过多 |
| I | 参数说明清楚 | +分辨率/夜视距离/连接方式 |
| I | 评价真实可参考 | +画质/稳定性/隐私评价 |
| P1 | 核心卖点清晰 | +" encrypted""no monthly fee""local storage" |
| P1 | 认证背书可信 | +数据安全认证/隐私合规 |
| P1 | 第三方测评真实 | +TechCrunch/ parenting magazine 测评 |

### 1.4 解决方案三：AIPL 动态锚定规则

**核心思路**：同一标签在不同上下文中的 AIPL 节点不同。

**v3.6 动态锚定规则库（Top 20 高频标签）**：

```yaml
规则库版本: v3.6

规则1:
  标签: 物流延迟
  默认AIPL: L1
  调整规则:
    - 条件: source_type == "order_data" AND 订单状态 == "已取消"
      AIPL: P2
      权重倍率: 1.5
      理由: 取消订单时的物流延迟 → 购买决策失败
    - 条件: source_type == "review" AND text 包含 "before baby arrived"
      AIPL: L1
      权重倍率: 1.4
      理由: 预产期前未送达 → 情感强度高
    - 条件: source_type == "review" AND text 包含 "compared with"
      AIPL: I
      权重倍率: 1.2
      理由: 对比时提及物流 → 影响兴趣阶段

规则2:
  标签: 尺码偏小
  默认AIPL: L1
  调整规则:
    - 条件: source_type == "return_note" AND 退货原因 == "尺码不符"
      AIPL: L1
      权重倍率: 1.5
      理由: 已退货 → 高业务影响
    - 条件: source_type == "review" AND text 包含 "ordered" AND text 包含 "size up"
      AIPL: P1
      权重倍率: 1.2
      理由: 购买前已发现尺码问题 → 评估阶段
    - 条件: source_type == "qa" OR source_type == "pre_sale"
      AIPL: I
      权重倍率: 1.0
      理由: Q&A 中的尺码询问 → 兴趣阶段

规则3:
  标签: 吸力不足
  默认AIPL: L1
  调整规则:
    - 条件: source_type == "review" AND text 包含 "compared with"
      AIPL: I
      权重倍率: 1.3
      理由: 对比竞品时提及吸力 → 影响兴趣阶段
    - 条件: source_type == "review" AND text 包含 "would not recommend"
      AIPL: L3
      权重倍率: 1.2
      理由: 因吸力问题不推荐 → 影响推荐阶段
    - 条件: source_type == "return_note"
      AIPL: L1
      权重倍率: 1.5
      理由: 退货原因 → 高业务影响

规则4:
  标签: 噪音太大
  默认AIPL: L1
  调整规则:
    - 条件: text 包含 "night" OR text 包含 "sleeping"
      AIPL: L1
      权重倍率: 1.5
      理由: 夜间噪音 → 最高情感强度场景
    - 条件: text 包含 "office" OR text 包含 "work"
      AIPL: L1
      权重倍率: 1.3
      理由: 工作场景噪音 → 高情感强度场景
    - 条件: text 包含 "compared with" AND text 包含 "quiet"
      AIPL: I
      权重倍率: 1.2
      理由: 对比静音竞品 → 影响兴趣阶段

规则5:
  标签: 价格过高
  默认AIPL: I
  调整规则:
    - 条件: text 包含 "after buying" OR text 包含 "not worth"
      AIPL: L1
      权重倍率: 1.2
      理由: 购买后觉得不值 → 使用阶段
    - 条件: text 包含 "compared with"
      AIPL: I
      权重倍率: 1.3
      理由: 对比时提及价格 → 兴趣阶段
    - 条件: source_type == "pre_sale" AND text 包含 "discount"
      AIPL: P1
      权重倍率: 1.0
      理由: 售前询问折扣 → 评估阶段
```

### 1.5 各品线 AIPL 目标分布

| 品线 | A | I | P1 | P2 | L1 | L2 | L3 | 新增标签数 |
|------|---|---|----|----|----|----|----|----------|
| 吸奶器 | 8% | 15% | 12% | 8% | **45%** | 7% | 5% | +27 |
| 内衣服饰 | 6% | 12% | 10% | 12% | **50%** | 5% | 5% | +20 |
| 家居家纺 | 4% | 10% | 8% | 6% | **55%** | 10% | 7% | +18 |
| 母婴综合护理 | 3% | 8% | 7% | 5% | **60%** | 10% | 7% | +15 |
| 喂养电器 | 5% | 10% | 10% | 8% | **50%** | 12% | 5% | +20 |
| 智能母婴电器 | 7% | 15% | 13% | 5% | **45%** | 10% | 5% | +30 |

**关键变化**：L1 占比从 85-100% 降至 45-60%，A/I/P 阶段从 0-15% 提升至 20-35%。

---

## 第二部分：全量 503 标签修复方案

### 2.1 17 个低分标签逐个修复

| # | 标签ID | 标签名 | 评分 | 核心问题 | 修复方案 | 修复后评分预估 |
|---|--------|--------|------|----------|----------|--------------|
| 1 | TAG_L1_035 | 烧焦 | 39.5 | Jaccard=1.00 与"充电口烧焦"完全混淆；异常共现 5 对 | **拆分为 2 个标签**：①"产品过热/烧焦（安全）"——关键词：overheated, burning smell, scorched, melted；②"充电口烧焦（使用）"——关键词：charging port burned, charger melted, port damaged。两个标签添加互斥规则。 | 70+ |
| 2 | TAG_P2_034 | 物流跟踪 | 39.8 | 覆盖率 16.75%；关键词混淆 33 对；内外覆盖率差异大 | **拆分为 3 个标签**：①"物流跟踪信息缺失"——no tracking, tracking not provided；②"物流跟踪更新滞后"——tracking not updated, no movement；③"物流跟踪信息错误"——wrong tracking number, tracking shows delivered but not received。限制每个子标签覆盖率 <5%。 | 65+ |
| 3 | TAG_P2_035 | 发货和运输时长 | 43.1 | 覆盖率 22.37%；定义过于宽泛 | **拆分为 3 个标签**：①"发货延迟"——not shipped, processing too long；②"运输时长超预期"——shipping took longer than expected, delivery slower than promised；③"配送窗口不明确"——no delivery window, unclear when arriving。添加时间量化条件（如"超过承诺时间 X 天"）。 | 65+ |
| 4 | TAG_P2_015 | Cancel未明确原因 | 44.0 | 样本不足 0.03%；关键词混淆 33 对 | **重新定义**：从"取消原因未知"改为"系统取消/未说明原因"，关键词聚焦：system cancelled, order cancelled without reason, no explanation for cancellation。添加排除规则（排除已明确原因的取消）。 | 60+ |
| 5 | TAG_P2_005 | 退货-收到礼物 | 46.2 | 覆盖率 14.73%；置信度不稳定 | **细化为 2 个标签**：①"退货-作为礼物收到"——received as gift, was a gift, didn't order；②"退货-礼物不合适"——gift didn't fit, wrong size for gift recipient。添加送礼场景关键词（gift, present, recipient）。 | 65+ |
| 6 | TAG_L1_082 | 使用后无法重连 | 46.6 | 样本不足 0.06%；关键词混淆 22 对 | **扩大样本**：增加关键词：won't reconnect after use, disconnects after using, have to re-pair every time, loses connection after session。扩展到吸奶器和智能母婴电器两个品线。 | 60+ |
| 7 | TAG_L1_031 | 配件破损/断裂 | 46.9 | Jaccard=1.00 与"配件破损/断裂-安全带"混淆 | **按品线拆分**：吸奶器→"配件破损/断裂-吸奶器"（导管/阀门/隔膜）；内衣服饰→"配件破损/断裂-服饰"（挂钩/扣件）；智能母婴电器→"配件破损/断裂-监视器"（支架/电源线）。各品线独立关键词。 | 70+ |
| 8 | TAG_L1_014 | 充电口烧焦 | 47.0 | 置信度不稳定；Jaccard=1.00 与"烧焦"混淆 | 与#1协同修复，重新定义边界：充电口烧焦仅限 charging port, USB port, charging cable 相关。添加部位限定词。 | 70+ |
| 9 | TAG_P2_028 | 运输时间太长 | 47.1 | 覆盖率 17.42% | **与"发货和运输时长"合并处理**：将"运输时间太长"作为"运输时长超预期"的子标签。统一关键词：shipping too long, delivery too slow, took forever to arrive。添加量化阈值（如"超过承诺时间"）。 | 65+ |
| 10 | TAG_P2_019 | 取消部分产品 | 47.5 | 覆盖率 17.86% | **细化为 2 个标签**：①"取消部分产品-缺货"——cancelled part of order due to out of stock；②"取消部分产品-用户主动"——cancelled part of order by choice。添加原因区分关键词。 | 65+ |
| 11 | TAG_P2_030 | 丢包/到达未收到 | 48.1 | 覆盖率 12.07% | **拆分为 2 个标签**：①"包裹丢失"——package lost, never arrived, missing package；②"显示送达但未收到"——marked delivered but not received, delivered to wrong address。区分承运人责任和投递错误。 | 65+ |
| 12 | TAG_P2_023 | 发货延迟 | 48.1 | 置信度不稳定（标准差 0.21） | **稳定置信度**：添加更多上下文关键词：order still processing, hasn't shipped yet, waiting to ship, fulfillment delay。排除关键词：shipping delayed（归属"运输时长超预期"）。 | 65+ |
| 13 | TAG_L1_039 | 烧焦-非安规问题 | 48.2 | 低置信度（均值 0.79）；关键词混淆 | **与"烧焦"合并**：将"烧焦-非安规问题"和"烧焦"合并为"产品过热/烧焦"，不再区分是否安规问题（安规问题由安全标签体系独立处理）。 | 70+ |
| 14 | TAG_L1_203 | 侧倾提醒 | 49.4 | 覆盖率 0.35%；样本不足 | **扩大样本或降级**：若扩大关键词后仍无法达到 1% 覆盖率，降级为"子标签"（归属"产品核心性能"主题），不作为独立一级标签。 | 55+ 或 删除 |
| 15 | TAG_P2_012 | 取消-收到礼物 | 49.5 | 覆盖率 11.16% | **与"退货-收到礼物"统一处理**：将"取消-收到礼物"和"退货-收到礼物"合并为"收到礼物（非本人购买）"，区分取消和退货两个动作。 | 65+ |
| 16 | TAG_L1_077 | 无法记录早于当前时间的奶量 | 49.5 | 覆盖率 12.50%；过于具体 | **抽象化为"App时间记录错误"**：扩大适用范围：wrong time recorded, time stamp incorrect, logged wrong time, app shows wrong time。不再限定"早于当前时间"。 | 65+ |
| 17 | TAG_P2_026 | 空包裹 | 49.5 | 覆盖率 0.18%；样本不足 | **扩大样本**：增加关键词：empty box, nothing inside, package was empty, missing contents, only packaging。跨品线通用（不仅限于物流）。 | 60+ |

### 2.2 84 个"自动发现-待审核"标签审核策略

**分布**：
- 智能母婴电器：27 个（63%）
- 母婴综合护理：19 个（33%）
- 家居家纺：11 个（24%）
- 喂养电器：7 个（15%）

**审核决策树**：

```
自动发现标签
    ↓
是否是产品类别关键词？（如"吸鼻器""推车""餐椅"）
    ├── 是 → 转为"品类识别标签"（不进入 VOC 情感标签库）
    │         → 用于产品分类，不用于情感分析
    └── 否 → 是否有明确的情感方向？
                ├── 是 → 进入正式标签库
                │         → 补充：AIPL节点、情感极性、标签主题、关键词
                └── 否 → 需要业务审核
                            → 审核通过：进入正式标签库
                            → 审核不通过：删除或归档
```

**智能母婴电器 27 个待审核标签处理建议**：

| 标签名 | 类型判断 | 处理建议 |
|--------|----------|----------|
| nasal_aspirator（吸鼻器） | 品类关键词 | 转为品类识别标签，不进入 VOC 标签库 |
| baby_stroller（推车） | 品类关键词 | 同上 |
| air_purifier（空气净化器） | 品类关键词 | 同上 |
| wooden_high（成长型餐椅） | 品类关键词 | 同上 |
| food_grade（食品级） | 属性关键词 | 转为"材质安全-食品级"标签，情感=正向 |
| grade_tray（托盘等级） |  unclear | 业务审核 |
| adjustable_ergonomic（可调节人体工学） | 属性关键词 | 转为"设计-人体工学"标签，情感=正向 |
| natural_wood（天然木材） | 属性关键词 | 转为"材质-天然木材"标签，情感=正向 |
| extendable_upf（可扩展 UPF） | 属性关键词 | 转为"设计-防晒扩展"标签，情感=正向 |
| upf_canopy（UPF 遮阳篷） | 属性关键词 | 转为"设计-防晒"标签，情感=正向 |
| stroller_convertible（推车可转换） | 属性关键词 | 转为"设计-可转换"标签，情感=正向 |
| para_beb（婴儿用） | 品类关键词 | 转为品类识别标签 |
| electric_nasal（电动吸鼻器） | 品类关键词 | 转为品类识别标签 |
| baby_nail（婴儿指甲） | 品类关键词 | 转为品类识别标签 |
| breast_milk（母乳） | 通用词 | 删除（过于宽泛，会与大量评论共现） |

### 2.3 20 个缺失字段补全

**缺失分布**：
- 默认优先级：20 个"【待填写】"
- 主责部门：20 个"【待填写】"

**补全策略**：

| 标签主题 | 默认优先级 | 主责部门 | 协同部门 |
|----------|-----------|----------|----------|
| 品牌初始印象 | P2 | 品牌市场中心 | 品牌运营部 |
| 需求内容匹配 | P2 | 品牌市场中心 | 内容运营部 |
| 广告真实性感知 | P1 | 品牌市场中心 | 法务部 |
| 种草内容可信度 | P2 | 品牌市场中心 | KOL运营部 |
| 达人推荐可信度 | P2 | 品牌市场中心 | KOL运营部 |
| 信息获取便捷性 | P2 | 电商运营部 | 产品中心/品线 |
| 参数规格透明度 | P1 | 产品中心/品线 | 电商运营部 |
| 评价参考价值 | P2 | 电商运营部 | 全球客服与体验中心 |
| 价格透明度 | P2 | 电商运营部 | 财务部 |
| 证据内容充分度 | P2 | 品牌市场中心 | 产品品牌市场中心 |
| 跨平台信息一致性 | P2 | 品牌运营部 | 品牌市场中心 |
| 核心功能认知准确度 | P1 | 产品中心/品线 | 产品品牌市场中心 |
| 竞品对比支持 | P2 | 产品品牌市场中心 | 竞品分析组 |
| 安全/合规认证可信度 | P0 | 质量与法规部 | 产品中心/品线 |
| 第三方测评可信度 | P2 | 品牌市场中心 | PR部 |
| 价格价值感 | P2 | 产品品牌市场中心 | 财务部 |
| 售后响应速度 | P1 | 全球客服与体验中心 | 运营部 |
| 问题解决效率 | P1 | 全球客服与体验中心 | 产品中心/品线 |
| 客服专业度 | P2 | 全球客服与体验中心 | 培训部 |
| 质量投诉闭环 | P0 | 全球客服与体验中心 | 品控部 |

### 2.4 关键词混淆修复策略

**问题类型**：大量标签之间存在 Jaccard > 0.3 的关键词重叠。

**修复策略**：

1. **互斥词表建立**：为每对高混淆标签建立互斥关键词。
   - 例："烧焦" vs "充电口烧焦" → 互斥词：charging port, USB, cable

2. **上下文窗口限定**：通过前后文限定词区分。
   - 例："物流跟踪" → 前文有"no""can't find" → "物流跟踪信息缺失"
   - 例："物流跟踪" → 前文有"not updated""stuck" → "物流跟踪更新滞后"

3. **品线前缀限定**：在关键词中添加品线限定。
   - 例："配件破损" → 吸奶器品线：flange, valve, tube, diaphragm
   - 例："配件破损" → 监视器品线：mount, cable, adapter, stand

4. **自动化监控**：建立 Jaccard 系数自动计算流水线。
   - 每周计算全部标签对的 Jaccard
   - Jaccard > 0.3 的自动告警
   - Jaccard > 0.5 的自动冻结标签

### 2.5 自动化质量监控机制（v3.6 新增）

```python
# 伪代码：标签质量监控流水线
class TagQualityMonitor:
    def weekly_check(self, tags: List[Tag]) -> QualityReport:
        report = QualityReport()
        
        for tag in tags:
            # 1. 覆盖率检查
            coverage = self.calculate_coverage(tag)
            if coverage > 10:
                report.add_issue(tag, "覆盖率过高", coverage)
            elif coverage < 0.1:
                report.add_issue(tag, "覆盖率过低", coverage)
            
            # 2. 关键词混淆检查
            for other_tag in tags:
                if tag.id == other_tag.id:
                    continue
                jaccard = self.calculate_jaccard(tag.keywords, other_tag.keywords)
                if jaccard > 0.5:
                    report.add_issue(tag, f"与{other_tag.id}严重混淆", jaccard)
                elif jaccard > 0.3:
                    report.add_issue(tag, f"与{other_tag.id}轻度混淆", jaccard)
            
            # 3. 内外覆盖率差异检查
            internal_coverage = self.calculate_internal_coverage(tag)
            external_coverage = self.calculate_external_coverage(tag)
            if internal_coverage > external_coverage * 5:
                report.add_issue(tag, "内外覆盖率差异大", 
                    f"{internal_coverage}% vs {external_coverage}%")
            
            # 4. 置信度稳定性检查
            confidence_std = self.calculate_confidence_std(tag)
            if confidence_std > 0.1:
                report.add_issue(tag, "置信度不稳定", confidence_std)
            
            # 5. 异常共现检查
            abnormal_cooccurrence = self.check_cooccurrence(tag)
            if abnormal_cooccurrence > 0:
                report.add_issue(tag, "异常共现", abnormal_cooccurrence)
        
        return report
    
    def auto_action(self, report: QualityReport):
        for issue in report.issues:
            if issue.severity == "极高风险":
                self.freeze_tag(issue.tag)  # 自动冻结
                self.notify_owner(issue.tag, "紧急修复")
            elif issue.severity == "高风险":
                self.flag_tag(issue.tag)  # 标记待修复
                self.add_to_backlog(issue.tag)
```

---

## 第三部分：全量 503 标签多语言映射方案

### 3.1 分级覆盖策略

**不是全部 503 标签都补充 4 语言——按优先级分级**：

| 优先级 | 标签类型 | 标签数量 | 语言覆盖 | 实施方式 |
|--------|----------|----------|----------|----------|
| **P0** | 安全相关 + P0 优先级 | ~20 | 英/西/法/德 + 荷兰/意大利 | 人工精译 |
| **P1** | 高频业务标签（Top 100） | 100 | 英/西/法/德 | 人工翻译 + 业务校对 |
| **P2** | 中频标签（Top 101-250） | 150 | 英/西/法 | AI 翻译 + 人工抽检 |
| **P3** | 低频标签（251-503） | 253 | 英文 + 西语模板 | AI 批量翻译 |

### 3.2 语言特性差异与关键词设计

**英语（美国/加拿大/英国）**：
- 美国： Moms 常用"pump""nursing bra""diaper"
- 英国：Mums 常用"breast pump""maternity bra""nappy"
- 加拿大：混合美英用词

**西班牙语（美国拉美/西班牙）**：
- 拉美："sacaleches"（吸奶器）"sostén de lactancia"（哺乳文胸）
- 西班牙："extractor de leche""sujetador de lactancia"
- 差异：同一产品在不同西语市场用词不同

**法语（法国/加拿大魁北克）**：
- 法国："tire-lait""soutien-gorge d'allaitement"
- 魁北克："extracteur de lait""soutien-gorge d'allaitement"（与法国基本一致）

**德语（德国）**：
- "Milchpumpe""Still-BH"
- 特点：复合词多，用户倾向使用完整词而非缩写
- 文化特性：对噪音容忍度低，"zu laut"提及率高 35%

### 3.3 高频标签多语言映射示例（Top 20）

| 中文标签 | 英文 | 西班牙文 | 法文 | 德文 |
|----------|------|----------|------|------|
| 吸力不足 | weak suction, not enough suction | succión débil, no succiona bien | succion faible, pas assez puissante | schwacher Sog, nicht stark genug |
| 噪音太大 | too loud, noisy, wakes baby | demasiado ruidoso, despierta al bebé | trop bruyant, réveille bébé | zu laut, weckt das Baby |
| 尺码偏小 | too small, runs small, tight | demasiado pequeño, talla pequeña, apretado | trop petit, taille petit, serré | zu klein, fällt klein aus, eng |
| 物流延迟 | late delivery, shipping delayed | entrega tardía, envío retrasado | livraison tardive, expédition retardée | verspätete Lieferung, Versand verzögert |
| 价格过高 | too expensive, overpriced | demasiado caro, sobreprecio | trop cher, surfait | zu teuer, überbewertet |
| 推荐意愿 | recommend, would suggest | recomendar, lo recomiendo | recommander, je recommande | empfehlen, ich empfehle |
| 面料不舒适 | itchy, rough, irritating | pica, áspero, irrita | démange, rugueux, irrite | juckt, kratzt, unangenehm |
| 温度不准 | temperature inaccurate, too hot | temperatura incorrecta, demasiado caliente | température inexacte, trop chaud | Temperatur ungenau, zu heiß |
| 连接不稳定 | keeps disconnecting, drops | se desconecta, conexión inestable | se déconnecte, connexion instable | verliert Verbindung, Verbindung bricht |
| 隐私担忧 | privacy concerns, don't trust | preocupación privacidad, no confío | préoccupation vie privée, ne confie pas | Datenschutzbedenken, vertraue nicht |
| 充电口烧焦 | charging port burned, melted | puerto de carga quemado, derretido | port de charge brûlé, fondu | Ladeanschluss verbrannt, geschmolzen |
| 漏奶 | leaks, leaking, spills | gotea, pierde leche, se derrama | fuit, perd du lait, dégouline | undicht, leckt, Milch überall |
| 配件破损 | broke, cracked, wore out | se rompió, agrietado, desgastado | cassé, fissuré, usé | gebrochen, gerissen, verschlissen |
| 续航不足 | battery dies, doesn't last | batería dura poco, se descarga rápido | batterie faible, ne dure pas | Akku hält nicht, schnell leer |
| 洗后变形 | shrunk, stretched out, lost shape | encogió, perdió forma, deformado | a rétréci, a perdu sa forme | eingegangen, Form verloren |
| 单手操作困难 | hard to open, need two hands | difícil de abrir, necesito dos manos | difficile à ouvrir, besoin de deux mains | schwer zu öffnen, zwei Hände nötig |
| 支撑力不足 | no support, sagging, flimsy | sin soporte, no sostiene, se cae | pas de soutien, trop mou | keine Stütze, zu weich |
| 加热不均匀 | uneven heating, hot spots | calentamiento desigual, puntos calientes | chauffage inégal, points chauds | ungleichmäßige Erwärmung |
| 安装复杂 | complicated, too many buttons | complicado, muchos botones | compliqué, trop de boutons | kompliziert, zu viele Knöpfe |
| 烧焦 | burned, scorched, melting smell | quemado, chamuscado, olor a quemado | brûlé, odeur de brûlé | verbrannt, verbrannter Geruch |

### 3.4 实施流程

**Phase 1（Week 1-2）：P0 安全标签**
- 20 个安全相关标签
- 4 语言人工精译
- 业务 + 法务 + 本地化团队三方校对

**Phase 2（Week 3-5）：P1 高频标签**
- 100 个高频标签
- AI 翻译（DeepL/Google Translate API）
- 业务团队逐条校对
- 重点校对西班牙文（拉美 vs 西班牙差异）

**Phase 3（Week 6-8）：P2 中频标签**
- 150 个中频标签
- AI 翻译
- 抽样 20% 人工校对

**Phase 4（Week 9-10）：P3 低频标签**
- 253 个低频标签
- AI 批量翻译
- 建立"翻译模板库"（如"XX 破损"→"XX broke/cracked"的模板）

---

## 第四部分：正向标签体系设计

### 4.1 设计原则

**不是简单增加"好评"标签，而是建立"正向体验→Promoter 驱动"的完整链路**：

```
正向标签命中
    ↓
Proxy NPS = Promoter 驱动
    ↓
归因分析：哪个正向标签是 Promoter 的主要驱动因素？
    ↓
策略输出：强化该正向体验，放大口碑传播
```

### 4.2 通用正向标签优化（v3.5 已有 40 个，优化后 50 个）

**v3.5 现有正向标签分析**：
- A 阶段：5 个（第一印象专业、内容打中痛点、真实妈妈种草、达人实测可信...）
- I 阶段：7 个（参数说明清楚、评价真实可参考、到手价清晰...）
- P1 阶段：5 个（核心卖点清晰、对比信息充分、认证背书可信、物超所值...）
- L1 阶段：6 个（易用性、舒适体验、质量感知、外观设计、耐用性、性能满意）
- L2 阶段：6 个（客服响应快、一次解决问题、退款换新顺畅...）
- L3 阶段：9 个（强烈推荐、会再次购买、社群互动活跃...）
- P2 阶段：1 个（配送满意）

**问题**：L1 阶段仅 6 个正向标签，但 L1 是标签最多的阶段（69 个负向 vs 6 个正向 = 11:1）

**优化方案**：

| AIPL | 现有正向标签数 | 新增 | 目标总数 | 关键新增标签 |
|------|--------------|------|----------|-------------|
| A | 5 | +3 | 8 | 品牌熟悉、广告精准、KOL可信 |
| I | 7 | +5 | 12 | 信息易找、评价详细、视频测评有用 |
| P1 | 5 | +5 | 10 | 尺码表准确、退换政策清晰、质保可信 |
| P2 | 1 | +4 | 5 | 发货快、包装完好、物流更新及时 |
| **L1** | **6** | **+14** | **20** | **静音好评、吸力强劲、穿戴舒适、温度精准...** |
| L2 | 6 | +4 | 10 | 客服态度好、问题解决快、补偿满意 |
| L3 | 9 | +3 | 12 | 推荐给朋友、社交分享、品牌忠诚 |

### 4.3 品线正向标签设计（按 4 大品线）

#### 吸奶器品线正向标签（15 个）

| # | 标签名 | AIPL | 关键词 | 业务价值 |
|---|--------|------|--------|----------|
| 1 | 吸力强劲 | L1 | strong suction, powerful, empties well | 核心功能满意 |
| 2 | 吸力模式丰富 | L1 | multiple modes, customizable, great settings | 差异化卖点 |
| 3 | 静音好评 | L1 | quiet, silent, discreet, whisper quiet | 夜间/工作场景 |
| 4 | 穿戴舒适 | L1 | comfortable, can't feel it, fits well | 长时间穿戴 |
| 5 | 隐蔽性好 | L1 | discreet, invisible under clothes, no one knows | 外出/工作场景 |
| 6 | 续航持久 | L1 | long battery life, lasts all day, great battery | 外出场景 |
| 7 | 充电快速 | L1 | charges quickly, fast charging, quick recharge | 使用便利 |
| 8 | 清洗方便 | L1 | easy to clean, few parts, dishwasher safe | 高频痛点反向 |
| 9 | 配件耐用 | L1 | durable parts, long lasting, quality accessories | 降低复购成本 |
| 10 | 便携性好 | L1 | portable, lightweight, easy to carry | 外出/旅行场景 |
| 11 | 不漏奶 | L1 | no leaks, secure seal, stays dry | 核心痛点反向 |
| 12 | App功能好用 | L1 | great app, love the tracking, useful features | 智能化卖点 |
| 13 | 法兰尺寸合适 | L1 | perfect fit, right size, comfortable flange | 尺码痛点反向 |
| 14 | 性价比高 | P1 | great value, worth every penny, affordable | 价格敏感用户 |
| 15 | 比竞品好 | I | better than Spectra, prefer over Medela | 竞品转化 |

#### 内衣服饰品线正向标签（12 个）

| # | 标签名 | AIPL | 关键词 |
|---|--------|------|--------|
| 1 | 尺码准确 | L1 | true to size, accurate sizing, fits perfectly |
| 2 | 面料柔软 | L1 | soft fabric, comfortable material, gentle on skin |
| 3 | 透气性好 | L1 | breathable, cool, doesn't trap heat |
| 4 | 弹性佳 | L1 | stretchy, good elasticity, accommodates growth |
| 5 | 单手操作顺畅 | L1 | easy one-handed, simple clasp, quick to open |
| 6 | 支撑力好 | L1 | great support, lifts well, holds everything |
| 7 | 洗后不变形 | L1 | holds shape after wash, doesn't shrink, color stays |
| 8 | 外观满意 | L1 | cute design, pretty color, doesn't look like grandma |
| 9 | 哺乳开口设计好 | L1 | convenient nursing access, easy to feed |
| 10 | 孕期哺乳期两用 | P1 | works for pregnancy and nursing, grows with me |
| 11 | 无痕设计 | L1 | seamless, smooth under clothes, no lines |
| 12 | 肩带舒适 | L1 | comfortable straps, doesn't dig in, wide straps |

#### 喂养电器品线正向标签（10 个）

| # | 标签名 | AIPL | 关键词 |
|---|--------|------|--------|
| 1 | 温度精准 | L1 | accurate temperature, perfect warmth, just right |
| 2 | 加热均匀 | L1 | heats evenly, no hot spots, consistent temperature |
| 3 | 加热速度快 | L1 | heats quickly, fast warm up, no waiting |
| 4 | 操作简单 | L1 | easy to use, intuitive, one button |
| 5 | 容量足够 | L1 | fits multiple bottles, large capacity, enough space |
| 6 | 材质安全无异味 | L1 | no plastic smell, safe materials, BPA free |
| 7 | 清洗方便 | L1 | easy to clean, descale reminder, wide opening |
| 8 | 夜间使用方便 | L1 | easy in the dark, backlit display, quiet beep |
| 9 | 消毒彻底 | L1 | thoroughly sterilized, kills germs, peace of mind |
| 10 | 省电 | L1 | energy efficient, low power, auto shut off |

#### 智能母婴电器品线正向标签（10 个）

| # | 标签名 | AIPL | 关键词 |
|---|--------|------|--------|
| 1 | 画质清晰 | L1 | clear picture, sharp image, HD quality |
| 2 | 夜视效果好 | L1 | great night vision, can see clearly in dark |
| 3 | 连接稳定 | L1 | reliable connection, never drops, always connected |
| 4 | 延迟低 | L1 | minimal lag, real-time, no delay |
| 5 | App体验好 | L1 | great app, user-friendly, feature-rich |
| 6 | 安装简单 | L1 | easy setup, quick install, clear instructions |
| 7 | 隐私保护好 | P1 | secure, encrypted, local storage, no cloud |
| 8 | 双向通话清晰 | L1 | clear two-way audio, can soothe baby remotely |
| 9 | 移动侦测灵敏 | L1 | sensitive motion detection, accurate alerts |
| 10 | 多设备管理方便 | L1 | easy to manage multiple cameras, split screen |

### 4.4 正向标签的 AIPL 分布与业务价值

**目标分布**：

| AIPL | 负向标签 | 正向标签 | 中性标签 | 正向占比目标 |
|------|---------|---------|---------|-------------|
| A | 5 | 8 | 0 | 62% |
| I | 18 | 12 | 0 | 40% |
| P1 | 13 | 10 | 0 | 43% |
| P2 | 38 | 5 | 0 | 12% |
| **L1** | **69** | **20** | **0** | **22%** |
| L2 | 12 | 10 | 0 | 45% |
| L3 | 3 | 12 | 0 | 80% |

**关键洞察**：
- L3（推荐阶段）正向占比目标 80%——符合逻辑，推荐阶段以正面反馈为主
- L1（首购使用）正向占比目标 22%——仍偏低，但较现状（6/69=8.7%）大幅提升
- P2（决策阶段）正向占比 12%——物流/订单问题以负面为主，正向少是合理的

---

## 第五部分：v3.6 融合版完整架构

### 5.1 六层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         v3.6 融合版六层架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 5: 质量评估与监控层（v3.5 成熟体系 + v3.6 自动化增强）                  │
│  ├── 合理性评分（64.8±11.9 基线）                                            │
│  ├── 风险等级（高/中/低/极高）                                               │
│  ├── 问题诊断（覆盖率/混淆/共现/置信度）                                      │
│  ├── 优化优先级（P0/P1/P2）                                                  │
│  └── 自动化监控（周度 Jaccard 扫描 + 覆盖率告警 + 异常冻结）                   │
│                                                                             │
│  Layer 4: 动态规则层（v0.2 核心能力，叠加到 v3.5）                            │
│  ├── AIPL 动态锚定（默认节点 + 上下文调整规则）                               │
│  ├── 渠道权重矩阵（4路 × 6品线差异化）                                       │
│  ├── 画像推导引擎（标签命中 → 原子加分 → 推导画像）                           │
│  ├── 安全标签 P0 强制升级（品线 × 市场监管）                                  │
│  └── 优先级升级规则（聚合条件 → 自动升级）                                    │
│                                                                             │
│  Layer 3: 业务闭环层（v3.5 增强）                                            │
│  ├── 策略包（品牌认知校准包 / 核心体验改良包 / 人群内容匹配包...）              │
│  ├── 主责部门 + 协同部门                                                     │
│  ├── 默认优先级 + 动态升级规则                                               │
│  ├── 故事线关联（品牌破冰效率 / 核心体验问题 / Gap聚类...）                    │
│  ├── 业务动作模板（"XX部：围绕XX主题做专项优化和闭环"）                        │
│  └── Proxy NPS 贡献（Promoter驱动 / Detractor驱动 / 中性）                   │
│                                                                             │
│  Layer 2: 多锚点层（v0.2 核心 + v3.5 扩展）                                  │
│  ├── AIPL 节点（静态，v3.5 已有）                                           │
│  ├── 竞品维度（品牌提及 / 对比 / 转换）                                       │
│  ├── 时间/生命周期维度（孕晚期/月子期/哺乳期...）                             │
│  ├── 多语言语义映射（英/西/法/德，分级覆盖）                                  │
│  └── 正向/负向情感极性（v3.5 已有，v3.6 增强正向）                            │
│                                                                             │
│  Layer 1: 内容分类层（v3.5 基座，503标签）                                   │
│  ├── 通用标签（191个，A/I/P1/P2/L1/L2/L3 全阶段）                            │
│  ├── 品线标签（312个，6品线）                                                │
│  ├── 细粒度标签（保留 v3.5 粒度）                                            │
│  ├── 正向标签（通用50 + 品线47 = 97个）                                      │
│  ├── 负向标签（v3.5 已有）                                                   │
│  ├── 映射关系（392行，主标签→支撑标签→比较对象）                             │
│  └── 存量归档（457个，版本管理）                                             │
│                                                                             │
│  Layer 0: 数据源层（v3.6 新增）                                              │
│  ├── A阶段数据源：TikTok/Instagram/Google搜索/Facebook广告                     │
│  ├── I阶段数据源：Amazon Q&A/Reddit/YouTube测评/小红书                         │
│  ├── P1阶段数据源：客服售前/产品页热力图/对比测评                              │
│  ├── P2阶段数据源：订单系统/物流跟踪/退货留言                                  │
│  ├── L1阶段数据源：电商评论/社媒/客服工单（已有）                              │
│  ├── L2阶段数据源：客服工单/追评/复购行为（新增复购数据）                       │
│  └── L3阶段数据源：Trustpilot/推荐评论/社媒分享（新增分享监听）                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 v3.6 标签字典模板

```yaml
# v3.6 标签字典模板（融合版）

标签ID: {TAG_[AIPL]_[序号]}
标签名称: {中文名}
标签英文: {英文名}

# === Layer 0: 内容分类 ===
所属品线: [通用/吸奶器/内衣服饰/...]
标签主题: {主题名}
情感极性: {正向/负向/中性}
是否通用标签: {是/否}
来源类型: {存量清洗/新建/逆向分析/自动发现}

# === Layer 1: 多锚点 ===
AIPL节点: {默认节点}
AIPL动态规则:
  - 条件: {逻辑表达式}
    调整后节点: {节点}
    权重倍率: {float}
    理由: {说明}

竞品维度:
  品牌提及: {品牌列表}
  对比模式: {是否触发竞品对比分析}

生命周期关联:
  高关联阶段: [孕晚期/月子期/哺乳期早期/...]
  时间衰减: {规则}

多语言语义映射:
  英文: [关键词列表]
  西班牙文: [关键词列表]
  法文: [关键词列表]
  德文: [关键词列表]

# === Layer 2: 业务闭环 ===
策略包: {策略包名}
主责部门: {部门名}
协同部门: [{部门列表}]
默认优先级: {P0/P1/P2/P3}
优先级升级规则:
  - 条件: {聚合条件}
    升级: {目标优先级}
    附加动作: {自动化动作}

故事线关联: {故事线主题}
业务动作: {具体动作描述}
ProxyNPS贡献: {Promoter驱动/Detractor驱动/中性}

# === Layer 3: 画像推导 ===
适用用户画像: [{画像列表}]
画像推导信号:
  - 条件: {命中此标签 + 其他条件}
    原子标签加分: {标签名: 权重}
    推导画像加分: {画像名: 权重}

# === Layer 4: 渠道权重 ===
适用VOC载体: [{载体列表}]
渠道权重:
  return_note: {float}
  ticket: {float}
  review: {float}
  trustpilot: {float}

# === Layer 5: 质量评估 ===
合理性评分: {分数}
风险等级: {极高/高/中/低}
问题诊断: {诊断结果}
优化优先级: {P0/P1/P2}
可识别性: {高/中/低}
归因清晰度: {高/中/低}
故事线支撑度: {高/中/低}
业务动作明确度: {高/中/低}

# === 安全标签专用 ===
安全等级: {P0紧急/P1高危/P2关注/P3记录}
监管标准映射:
  美国: {CPSC/CPSIA/ASTM}
  欧盟: {CE/EN71/REACH}
  英国: {UKCA/UK REACH}
  加拿大: {CCPSA}
  德国: {LFGB/BDSG}
  法国/西班牙: {GDPR/REACH}

版本: v3.6
更新日期: 2026-04-23
```

### 5.3 实施路径（10周）

#### Week 1-2：P0 紧急修复

| 天数 | 任务 | 产出 |
|------|------|------|
| D1-2 | 修复 2 个极高风险标签（烧焦/物流跟踪） | 修复后的标签定义 |
| D3-5 | 审核智能母婴电器 27 个待审核标签 | 审核结果（转正/转品类标签/删除） |
| D6-8 | 补全 20 个缺失字段（优先级+部门） | 完整字段表 |
| D9-10 | 为 Top 20 高频标签添加 AIPL 动态锚定规则 | 动态规则库 v1.0 |

#### Week 3-4：AIPL 标签下沉

| 天数 | 任务 | 产出 |
|------|------|------|
| D11-13 | 通用表 A 标签下沉到 6 品线 | 各品线 A 标签（~30个） |
| D14-16 | 通用表 I 标签下沉到 6 品线 | 各品线 I 标签（~60个） |
| D17-18 | 通用表 P1 标签下沉到 6 品线 | 各品线 P1 标签（~30个） |
| D19-20 | 验证各品线 AIPL 分布 | AIPL 分布报告 |

#### Week 5-6：正向标签 + 渠道权重

| 天数 | 任务 | 产出 |
|------|------|------|
| D21-23 | 设计 47 个品线正向标签 | 品线正向标签表 |
| D24-25 | 为全部 503 标签添加渠道权重字段 | 渠道权重配置文件 |
| D26-28 | 设计渠道权重矩阵（6品线×4数据源） | 权重矩阵表 |
| D29-30 | 验证渠道权重合理性 | 权重验证报告 |

#### Week 7-8：多语言映射

| 天数 | 任务 | 产出 |
|------|------|------|
| D31-35 | P0+P1 标签（120个）4语言翻译 | 多语言关键词表 v1.0 |
| D36-38 | P2 标签（150个）AI翻译+抽检 | 多语言关键词表 v2.0 |
| D39-40 | P3 标签（253个）模板化批量翻译 | 多语言关键词表 v3.0 |

#### Week 9-10：画像推导 + 质量监控 + 整合测试

| 天数 | 任务 | 产出 |
|------|------|------|
| D41-43 | 设计画像推导规则（Top 50 标签） | 画像推导规则库 |
| D44-45 | 建立自动化质量监控流水线 | 监控面板 |
| D46-48 | 端到端测试（吸奶器品线试点） | 测试报告 |
| D49-50 | 全量标签验证 + 修复 | v3.6 正式版 |

---

## 附录：v3.5 → v3.6 变更清单

| 变更项 | v3.5 状态 | v3.6 变更 | 影响范围 |
|--------|----------|----------|----------|
| AIPL 动态锚定 | 无 | 新增 Top 20 标签动态规则 | 20 标签 |
| 渠道权重字段 | 无 | 新增 4 路权重字段 | 503 标签 |
| 画像推导规则 | 无 | 新增推导信号 | 50 标签 |
| A/I/P 标签下沉 | 通用表 only | 下沉到 6 品线 | ~120 标签 |
| 正向标签补充 | 40 个 | 新增 57 个品线正向 | 57 标签 |
| 多语言映射 | 英文 only | 英/西/法/德（分级） | 503 标签 |
| 安全标签细化 | 分散在各主题 | 独立维度 + 监管映射 | 20+ 标签 |
| 低分标签修复 | 17 个低分 | 逐个修复 | 17 标签 |
| 待审核标签处理 | 84 个待审核 | 审核转正/删除 | 84 标签 |
| 缺失字段补全 | 20 个缺失 | 补全 | 20 标签 |
| 自动化质量监控 | 无 | 新增周度监控 | 503 标签 |
| 数据源分层 | 无 | 新增 A/I/P 数据源接入 | 系统架构 |

---

**文档版本**: v3.6  
**创建日期**: 2026-04-23  
**覆盖标签**: 503（v3.5 基座）+ ~180（新增/优化）  
**覆盖品线**: 6 个  
**覆盖市场**: 美/西/加/德/英/法  
**实施周期**: 10 周
