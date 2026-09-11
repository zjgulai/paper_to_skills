---
title: VOC标签体系Phase4执行计划 — 覆盖率85%+与品牌关键词库构建
doc_type: workflow
module: voc-nlp
topic: tag-evolution-phase4
status: review
created: 2026-04-28
updated: 2026-04-28
owner: self
source: human+ai
---

# VOC 标签体系 Phase 4 执行计划

> **基线状态**: v3.7 已完成，覆盖率 78.97% (287,864 / 364,569)，569 标签
> **目标状态**: 覆盖率 ≥85%，字典扩展至 600+ 标签
> **执行周期**: 3 周（2026-04-28 ~ 2026-05-19）
> **沿用工作流**: Phase 1~3 完整工作流（统一加载 → 逆向分析 → 覆盖率提升）

---

## 一、执行摘要

本次 Phase 4 聚焦三个核心任务，均沿用已验证的 Phase 1~3 工作流：

| 任务 | 问题 | 策略 | 预期覆盖率提升 |
|------|------|------|---------------|
| **T1** | Trustpilot 竞品品牌 VOC 被误判为零标签 | 构建品牌关键词库，竞品 VOC 保留并打品牌维度标签 | +2~3% |
| **T2** | Momcozy 3,697 条零标签为负面缺陷描述超出现有字典 | 零标签文本聚类 → 候选标签过滤 → ALCHEmist Label Function → 字典扩展 | +4~6% |
| **T3** | 通用标签仅有正面维度，德/法情感词不足，Zendesk 极简文本无标签 | 添加 8 组通用负面标签 + 德/法负面情感词 + Zendesk 极简规则 | +2~3% |
| **合计** | | | **78.97% → 87~90%** |

---

## 二、任务 T1：品牌关键词库构建与竞品 VOC 保留策略

### 2.1 问题重定义

**原始判断（已修正）**: Trustpilot 30,441 条零标签中"绝大部分为非 Momcozy 品牌评论"→ 误判为需要剔除。

**正确判断**: 竞品品牌（Willow, Medela, Philips Avent 等）的 VOC 只要属于我司品类范围，就是行业情报数据来源，**必须保留**并打上品牌维度标签。零标签的原因是品牌维度缺失导致品线专属标签无法匹配，而非数据质量问题。

### 2.2 品牌关键词库设计

**品牌分层结构**:

```python
BRAND_KEYWORD_LIBRARY = {
    # 自有品牌
    "own_brand": {
        "Momcozy": {
            "keywords": ["momcozy", "mom cozy", "momcozy.com"],
            "variants": ["momcozy", "mom cozy"],
            "priority": 1,
        },
    },
    # 直接竞品（吸奶器/哺乳内衣核心竞品）
    "direct_competitor": {
        "Spectra": {
            "keywords": ["spectra", "spectra s1", "spectra s2", "spectra pump"],
            "category": "吸奶器",
        },
        "Medela": {
            "keywords": ["medela", "medela pump", "medela freestyle", "medela symphony"],
            "category": "吸奶器",
        },
        "Willow": {
            "keywords": ["willow", "willow pump", "willow go", "willow 3.0"],
            "category": "吸奶器",
        },
        "Elvie": {
            "keywords": ["elvie", "elvie pump", "elvie stride"],
            "category": "吸奶器",
        },
        "Lansinoh": {
            "keywords": ["lansinoh", "lansinoh pump", "lansinoh breastfeeding"],
            "category": "吸奶器/哺乳内衣",
        },
        "Haakaa": {
            "keywords": ["haakaa", "haakaa pump", "haakaa collector"],
            "category": "吸奶器配件",
        },
        "Philips Avent": {
            "keywords": ["philips avent", "avent", "avent pump", "avent sterilizer"],
            "category": "吸奶器/喂养电器",
        },
    },
    # 间接竞品（母婴综合品类）
    "indirect_competitor": {
        "Comfelie": {
            "keywords": ["comfelie", "comfelie bra", "comfelie nursing"],
            "category": "内衣服饰",
        },
        "Kindred Bravely": {
            "keywords": ["kindred bravely", "kindred bravely bra"],
            "category": "内衣服饰",
        },
        "BabyBjorn": {
            "keywords": ["babybjorn", "baby bjorn", "babybjorn carrier"],
            "category": "母婴综合护理",
        },
        "Ergobaby": {
            "keywords": ["ergobaby", "ergo baby", "ergobaby carrier", "ergo 360"],
            "category": "母婴综合护理",
        },
        "Dr. Brown's": {
            "keywords": ["dr. brown", "dr browns", "dr brown's bottle"],
            "category": "喂养电器",
        },
        "Tommee Tippee": {
            "keywords": ["tommee tippee", "tommee tippee bottle", "tommee tippee sterilizer"],
            "category": "喂养电器",
        },
        "Nanit": {
            "keywords": ["nanit", "nanit monitor", "nanit camera"],
            "category": "智能母婴电器",
        },
        "Owlet": {
            "keywords": ["owlet", "owlet sock", "owlet monitor", "owlet dream"],
            "category": "智能母婴电器",
        },
    },
}
```

**品牌识别规则设计**:

| 规则 | 触发条件 | 品牌标签 | 置信度 |
|------|---------|---------|--------|
| R1 | 标题/正文包含品牌名（含大小写变体） | 对应品牌 | 0.95 |
| R2 | 标题/正文包含品牌名 + 产品型号 | 对应品牌 | 0.99 |
| R3 | 多条品牌名同时出现（"vs" / "compared to"） | 多品牌标签 | 0.90 |
| R4 | 无品牌名但包含通用产品词 | "unidentified_brand" | 0.50 |

### 2.3 竞品 VOC 标签策略

| 数据层 | 品牌识别结果 | 通用标签 | 品线/品类标签 | 品牌维度标签 | 用途 |
|--------|-------------|---------|-------------|-------------|------|
| Momcozy 官方 | Momcozy | 209 个 | 360 个品线专属 | Momcozy | 业务决策、产品改进 |
| 直接竞品 | Spectra/Medela/Willow/... | 209 个 | 通用品线标签（按品类推断） | 对应品牌 | 竞品情报、功能对比 |
| 间接竞品 | Comfelie/BabyBjorn/... | 209 个 | 通用品线标签 | 对应品牌 | 市场趋势、品类扩展 |
| 无法识别 | unidentified | 209 个 | 无 | 无 | 丢弃或人工审核 |

**关键区别**：竞品评论不打 Momcozy 品线专属标签（如"吸力不足-Momcozy M5"），但打通用品线标签（如"吸力不足-吸奶器通用"）。

### 2.4 执行步骤

| Step | 动作 | 复用脚本 | 产出 |
|------|------|---------|------|
| T1.1 | 品牌关键词库构建（上表 JSON） | 新建 `brand_keyword_library.py` | `04-输出结果/08-辅助数据/brand_keyword_library.json` |
| T1.2 | 品牌识别 Label Function 开发 | 复用 `alchemist_label_generator.py` 模式 | `02-脚本工具/04-数据处理/brand_label_functions.py` |
| T1.3 | Trustpilot 全量数据品牌打标 | 新建 `apply_brand_labels.py` | 品牌分层结果 JSONL |
| T1.4 | 竞品评论品线推断（基于产品关键词） | 复用 `infer_product_line.py` | 品线推断结果 |
| T1.5 | 竞品评论重新打标（通用标签 + 品线通用标签） | 复用 `general_tag_labeler.py` + `incremental_labeling.py` | 打标后 JSONL |
| T1.6 | 覆盖率验证 | 复用 `quick_coverage_test.py` | 分层覆盖率报告 |

### 2.5 验收标准

- 品牌识别准确率 ≥95%（200 条人工校验）
- Momcozy 品牌评论覆盖率 ≥85%
- 竞品品牌评论通用标签覆盖率 ≥70%
- 品牌维度标签零遗漏

---

## 三、任务 T2：Momcozy 零标签负面缺陷字典扩展

### 3.1 问题定位

3,697 条零标签的核心根因：**负面缺陷描述超出 v3.7 字典范围**。

数据证据（来自 `tag_gap_analysis/gap_analysis.json`）：

| 品线 | 高频 novel words | 当前字典缺口 |
|------|-----------------|-------------|
| 吸奶器 | suction, flange, letdown, clog, valve, diaphragm, backflow | 无"堵奶/回流""法兰不适配""阀片老化"标签 |
| 哺乳内衣 | uniboob, velcro, strap, support, clip, band, underwire | 无"支撑不足""扣件难操作""钢圈不适"标签 |
| 消毒器 | descale, residue, dry, heating element, mold, limescale | 无"水垢积累""残留不干""发霉"标签 |
| 孕妇枕 | velcro, stuffing, itchy, tip over, hot, flat | 无"填充物移位""面料刺痒""不稳倾倒"标签 |
| 温控类 | inaccurate, fluctuate, calibration, thermometer | 无"温度不准""波动大"标签 |
| 智能设备 | disconnect, app crash, firmware, bluetooth, pairing | 无"断连""App崩溃""固件"标签 |

### 3.2 沿用工作流：Phase 2 逆向分析

**Step T2.1: 零标签提取**

- 脚本：复用 `zero_label_extractor.py`
- 输入：`phase3_p3_labeled.jsonl` 中 `labels == []` 的记录
- 过滤：仅保留 Momcozy 自有数据（排除竞品）
- 输出：`phase4_zero_label_samples.json` + `phase4_zero_label_by_category.json`

**Step T2.2: 缺口检测（文本聚类）**

- 脚本：复用 `gap_detector.py`，增强聚类模块
- 方法：TF-IDF + KMeans（k=15~20）对零标签文本聚类
- 输入：3,697 条零标签文本
- 输出：主题聚类报告 + 各主题高频词

**Step T2.3~T2.4: 候选标签过滤**

- 脚本：复用 `candidate_tag_filter.py`
- 过滤规则：
  1. 聚类主题词频 ≥10（确保有足够样本支撑）
  2. 与现有标签 Jaccard < 0.3（避免重复）
  3. 业务相关性评分 ≥3/5（由规则引擎判断）
- 预期产出：30~40 个候选负面标签

**Step T2.5: Active-Learning 质量把关**

- 脚本：复用 `active_learning_audit.py`
- 审核维度：
  - 标签定义是否清晰
  - 关键词边界是否明确
  - 与现有标签是否互斥
- 阈值：自动通过率 ≥95%，剩余人工审核

**Step T2.6: ALCHEmist Label Function 生成**

- 脚本：复用 `alchemist_label_generator.py`
- 输入：审核通过的候选标签
- 输出：Python Label Functions（预期 30~40 个，~600 行）
- 设计原则：
  - 带否定词检测（"not working" vs "working well"）
  - 整词边界匹配
  - 情感极性强制为 negative

**Step T2.7: 标签字典更新（v3.8）**

- 脚本：复用 `tag_dictionary_updater.py`
- 新增标签字段：
  - tag_id: 按现有编码规则（TAG_L1_XXX / TAG_P2_XXX 等）
  - tag_en / tag_cn
  - aipl: 根据缺陷类型推断（L1 产品使用 / L2 售后 / P2 物流）
  - sentiment: negative（强制）
  - category: 品线归属
  - keywords: Label Function 中的触发词
  - 策略包 / 主责部门 / 优先级 / 原子指标：复用 v3.7 字段补全规则

**Step T2.8~T2.9: 字段补全与验证**

- 脚本：复用 `v37_field_injector.py` + `dictionary_validator.py`
- 补全：策略包、主责部门、优先级、原子指标
- 验证：全字段 0 空值、标签 ID 唯一性、AIPL 有效性

### 3.3 新增标签预览（基于数据分析）

| 品线 | 新增负面标签 | 触发关键词示例 | 预期命中数 |
|------|------------|---------------|-----------|
| **吸奶器** | 堵奶/回流 | clog, clogged duct, milk flowing back, backflow | ~180 |
| | 法兰不适配 | flange too small/large, wrong flange size, shield doesn't fit | ~120 |
| | 阀片/隔膜老化 | valve worn out, diaphragm torn, valve not sealing | ~80 |
| | 吸力突然下降 | suction dropped, suddenly weak, lost suction | ~90 |
| **哺乳内衣** | 支撑不足 | no support, sagging, not supportive, breasts unsupported | ~150 |
| | 扣件难操作 | clip hard to open, clasp difficult, one-handed clip | ~60 |
| | 钢圈不适 | underwire digging, wire poking, underwire painful | ~40 |
| | 面料过敏 | rash from fabric, allergic reaction, skin irritation | ~30 |
| **消毒器** | 水垢积累 | limescale buildup, descaling needed, white residue | ~100 |
| | 残留不干 | not dry, still wet, water spots, moisture left | ~80 |
| | 发霉 | mold in tank, moldy smell, black spots | ~30 |
| **孕妇枕** | 填充物移位 | stuffing shifted, filling moved, flat in spots | ~60 |
| | 面料刺痒 | itchy fabric, scratchy material, irritating texture | ~40 |
| | 不稳倾倒 | tips over, not stable, falls off bed, slippery | ~30 |
| | 散热差 | too hot, sleeps hot, traps heat, sweaty | ~50 |
| **智能设备** | 断连 | disconnects, loses connection, drops bluetooth, won't stay paired | ~70 |
| | App崩溃 | app crashes, app freezes, app not responding | ~40 |
| | 固件问题 | firmware issue, update failed, firmware bug | ~20 |
| **通用** | 噪音问题 | too loud, noisy motor, whirring sound, grinding noise | ~100 |
| | 充电问题 | won't charge, charging issue, battery dies fast | ~80 |
| | 异味 | chemical smell, plastic odor, weird smell, stinks | ~60 |

**预期新增标签总数**：~25 个负面标签，覆盖核心缺口。

### 3.4 验收标准

- 新增负面标签 25±5 个
- Active-Learning 自动通过率 ≥95%
- 新增标签在 200 条测试样本中精确率 ≥80%
- Momcozy 自有数据覆盖率从 81.3% 提升至 ≥87%

---

## 四、任务 T3：通用负面标签 + 德/法情感词 + Zendesk 极简规则

### 4.1 通用负面标签扩展

**当前状态**（`general_tag_labeler.py`）：仅有 8 个正面通用标签。

**扩展方案**：为每个正面标签添加对应的负面标签，形成正反对称体系。

```python
# 新增通用负面标签定义
GENERAL_NEGATIVE_TAGS = [
    {
        "tag_id": "TAG_GEN_N001",
        "tag_en": "difficult_to_use",
        "tag_cn": "使用困难",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "difficult to use", "hard to use", "complicated", "confusing",
            "not intuitive", "steep learning curve", "frustrating to use",
            "difficult to figure out", "hard to operate", "not user friendly",
            # German
            "schwierig zu bedienen", "kompliziert", "nicht intuitiv",
            # French
            "difficile à utiliser", "compliqué", "pas intuitif",
        ],
        "counterpart": "TAG_GEN_E001",  # 与 ease_of_use 互斥
    },
    {
        "tag_id": "TAG_GEN_N002",
        "tag_en": "uncomfortable",
        "tag_cn": "不舒适",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "uncomfortable", "painful", "hurts", "rough", "scratchy",
            "digs in", "pinches", "too tight", "squeezes", "chafes",
            # German
            "unbequem", "schmerzhaft", "zu eng",
            # French
            "inconfortable", "douloureux", "trop serré",
        ],
        "counterpart": "TAG_GEN_E002",
    },
    {
        "tag_id": "TAG_GEN_N003",
        "tag_en": "poor_quality",
        "tag_cn": "质量差",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "poor quality", "bad quality", "cheap", "flimsy", "cheaply made",
            "poor construction", "falls apart", "not well made", "shoddy",
            # German
            "schlechte qualität", "billig", "mindere qualität",
            # French
            "mauvaise qualité", "pas cher", "qualité médiocre",
        ],
        "counterpart": "TAG_GEN_E003",
    },
    {
        "tag_id": "TAG_GEN_N004",
        "tag_en": "poor_design",
        "tag_cn": "设计差",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "poor design", "bad design", "ugly", "cheap looking", "dated",
            "awkward design", "poorly designed", "doesn't look good",
        ],
        "counterpart": "TAG_GEN_E004",
    },
    {
        "tag_id": "TAG_GEN_N005",
        "tag_en": "not_durable",
        "tag_cn": "不耐用",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "broke quickly", "fell apart", "didn't last", "not durable",
            "poor durability", "wore out fast", "lasted only", "cheap material",
            # German
            "schnell kaputt", "hält nicht lange", "nicht haltbar",
            # French
            "s'est cassé vite", "pas durable", "ne dure pas",
        ],
        "counterpart": "TAG_GEN_E005",
    },
    {
        "tag_id": "TAG_GEN_N006",
        "tag_en": "poor_performance",
        "tag_cn": "性能差",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "poor performance", "weak", "ineffective", "doesn't work well",
            "not powerful enough", "underperforms", "disappointing performance",
        ],
        "counterpart": "TAG_GEN_E006",
    },
    {
        "tag_id": "TAG_GEN_N007",
        "tag_en": "poor_value",
        "tag_cn": "性价比差",
        "aipl": "P1",
        "sentiment_base": "negative",
        "keywords": [
            "overpriced", "not worth", "waste of money", "too expensive",
            "poor value", "not worth the price", "overpriced for what it is",
            # German
            "zu teuer", "nicht sein geld wert", "schlechtes preis-leistungs-verhältnis",
            # French
            "trop cher", "pas worth", "mauvais rapport qualité prix",
        ],
        "counterpart": "TAG_GEN_E007",
    },
    {
        "tag_id": "TAG_GEN_N008",
        "tag_en": "size_fit_negative",
        "tag_cn": "尺码不合",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "too tight", "too loose", "doesn't fit", "runs small", "runs large",
            "wrong size", "size off", "sizing is wrong", "fits poorly",
            # German
            "zu klein", "zu groß", "passt nicht", "fällt klein aus",
            # French
            "trop petit", "trop grand", "ne correspond pas", "taille incorrecte",
        ],
        "counterpart": "TAG_GEN_E008",
    },
]
```

**互斥规则**：
- 正面标签与对应负面标签互斥（同一条文本不能同时命中 ease_of_use 和 difficult_to_use）
- 若同时命中，取否定词检测结果为准（"not easy" → difficult_to_use）

### 4.2 德/法情感词补全

**当前德语关键词**（v3.7）：
```
正面: sehr zufrieden, sehr gut, alles bestens, top qualität, gute qualität
```

**需补充德语负面关键词**：
```
nicht zufrieden, unzufrieden, enttäuscht, sehr enttäuscht
schlecht, sehr schlecht, schlechte qualität, minderwertig
defekt, kaputt, gebrochen, funktioniert nicht
zu teuer, nicht sein geld wert, überteuert
nicht empfehlenswert, würde nicht wieder kaufen
```

**当前法语关键词**（v3.7）：
```
正面: très satisfait, super produit, bonne qualité, excellente qualité
```

**需补充法语负面关键词**：
```
pas satisfait, insatisfait, très déçu, déçu
mauvaise qualité, qualité médiocre, fragile
cassé, ne fonctionne pas, défectueux
trop cher, pas worth, mauvais rapport qualité prix
ne recommande pas, ne rachèterai pas
```

**实施方式**：直接扩展 `general_tag_labeler.py` 中各标签的 keywords 列表，无需新建标签。

### 4.3 Zendesk 极简规则

**问题**：Zendesk 工单有大量极短文本（<20 字符），现有 10 个服务标签无法覆盖。

**数据**：`zendesk_momcozy_report.json` 显示 Zendesk 总计 48,005 条采样，零标签率约 18%。

**极简规则设计**（文本长度 <30 字符）：

| 规则 ID | 触发条件（正则/关键词） | 标签 | AIPL | 置信度 | 示例 |
|---------|----------------------|------|------|--------|------|
| ZM_001 | `r'refund\|money back\|get my money'` | Refund_Request | L2 | 0.95 | "I want a refund" |
| ZM_002 | `r'return\|send back\|send it back'` | Return_Request | L1 | 0.95 | "Need to return" |
| ZM_003 | `r'broken\|not working\|stopped working\|defective'` | Warranty_Claim | L2 | 0.90 | "Pump broken" |
| ZM_004 | `r'wrong (item\|size\|product)\|doesn.t fit'` | Return_Request | L1 | 0.90 | "Wrong size" |
| ZM_005 | `r'cancel\|stop (order\|shipment)\|don.t want'` | Order_Cancelled | P2 | 0.85 | "Cancel order" |
| ZM_006 | `r'where\|track\|status\|when (will\|arrive\|deliver)'` | Shipping_Inquiry | P2 | 0.85 | "Where is my order" |
| ZM_007 | `r'thank\|thanks\|appreciate\|great (service\|help)'` | General_Feedback | L3 | 0.90 | "Thanks for help" |
| ZM_008 | `r'disappointed\|unhappy\|frustrated\|terrible (service\|experience)'` | Complaint | L2 | 0.85 | "Very disappointed" |
| ZM_009 | `r'exchange\|swap\|replace (with\|for)'` | Exchange_Request | L2 | 0.90 | "Want to exchange" |
| ZM_010 | `r'missing\|didn.t receive\|not in (package\|box)\|empty'` | Missing_Item | P2 | 0.90 | "Missing parts" |
| ZM_011 | `r'how (to\|do)\|instructions\|manual\|use\|setup'` | Product_Inquiry | I | 0.85 | "How to clean" |
| ZM_012 | `r'discount\|coupon\|code\|promo\|sale\|cheaper'` | Discount_Inquiry | P1 | 0.80 | "Any discount?" |

**规则优先级**：
1. 长度 <15 字符：优先匹配极简规则
2. 长度 15~30 字符：极简规则 + 现有 Zendesk 服务标签同时匹配
3. 长度 >30 字符：仅使用现有 Zendesk 服务标签

### 4.4 执行步骤

| Step | 动作 | 复用/新建 | 产出 |
|------|------|----------|------|
| T3.1 | `general_tag_labeler.py` 扩展：添加 8 个负面标签 | 修改现有文件 | 更新后的 `general_tag_labeler.py` |
| T3.2 | 德/法负面情感词注入各标签 keywords | 修改现有文件 | 多语言关键词扩展 |
| T3.3 | Zendesk 极简规则开发（12 条） | 新建 `zendesk_minimal_labeler.py` | 极简规则模块 |
| T3.4 | Zendesk 数据长度分析 | 新建脚本 | 长度分布报告 |
| T3.5 | 端到端测试：负面标签精确率 | 复用测试框架 | 精确率报告 |
| T3.6 | 覆盖率验证 | 复用 `quick_coverage_test.py` | 覆盖率增量报告 |

### 4.5 验收标准

- 8 个通用负面标签在 500 条测试样本中精确率 ≥80%
- 德/法负面情感词覆盖率在非英语评论中 ≥60%
- Zendesk 极简规则在 <30 字符工单中覆盖率 ≥70%
- 整体覆盖率提升 +2~3%

---

## 五、三任务整合路线图

### Week 1（04-28 ~ 05-05）: T1 品牌关键词库 + T3 通用负面标签

```
Day 1-2: T1.1 品牌关键词库构建
Day 3-4: T1.2 品牌识别 Label Function 开发
Day 5:   T1.3 Trustpilot 品牌打标
Day 1-3: T3.1 通用负面标签开发（并行）
Day 4-5: T3.2 德/法情感词注入（并行）
Weekend: T1.4 品线推断 + T1.5 重新打标
```

### Week 2（05-05 ~ 05-12）: T2 负面缺陷字典扩展

```
Day 1:   T2.1 零标签提取（Momcozy 自有数据）
Day 2-3: T2.2 缺口检测（TF-IDF + KMeans 聚类）
Day 4:   T2.3~T2.4 候选标签过滤
Day 5:   T2.5 Active-Learning 审核
Weekend: T2.6 ALCHEmist Label Function 生成
```

### Week 3（05-12 ~ 05-19）: T2 字典更新 + T3 Zendesk 极简规则 + 整合验证

```
Day 1-2: T2.7 标签字典更新 → v3.8
Day 3:   T2.8~T2.9 字段补全 + 验证
Day 1-2: T3.3 Zendesk 极简规则开发（并行）
Day 3-4: T3.4 Zendesk 长度分析 + T3.5 测试（并行）
Day 5:   全量数据重新打标 + 最终覆盖率验证
Weekend: v3.8 最终审计报告
```

### 关键里程碑

| 日期 | 里程碑 | 验收标准 |
|------|--------|---------|
| 05-05 | T1 完成 | 品牌分层覆盖率报告，Momcozy 层 ≥85% |
| 05-05 | T3.1~T3.2 完成 | 通用负面标签精确率 ≥80% |
| 05-12 | T2.6 完成 | 25±5 个负面 Label Functions 生成 |
| 05-19 | Phase 4 完成 | 整体覆盖率 ≥85%，v3.8 字典交付 |

---

## 六、验收标准总表

| 维度 | 指标 | 目标值 | 验证方式 |
|------|------|--------|---------|
| **覆盖率** | 整体覆盖率 | ≥85% | `quick_coverage_test.py` |
| | Momcozy 自有数据覆盖率 | ≥87% | 品类级覆盖率报告 |
| | Trustpilot 品牌分层覆盖率 | Momcozy层≥85%，竞品层≥70% | 分层覆盖率报告 |
| | Zendesk 极简文本覆盖率 | <30字符工单≥70% | 长度分层报告 |
| **标签质量** | 新增标签精确率 | ≥80% | 200条人工校验 |
| | Active-Learning 自动通过率 | ≥95% | `active_learning_audit.py` |
| | 标签冲突率 | <5% | `tag_diagnostics.py` |
| **字典完整性** | 全字段空值 | 0 | `dictionary_validator.py` |
| | 标签 ID 唯一性 | 100% | `dictionary_validator.py` |
| | 德/法多语言覆盖率 | ≥60% | 非英语样本测试 |

---

## 七、资源需求

| 资源 | 数量 | 说明 |
|------|------|------|
| 人工标注 | ~500 条 | 品牌识别 200 + 负面标签精确率 200 + 德/法验证 100 |
| LLM API | ~$15 | 候选标签生成 + 字典字段补全 |
| 计算 | 现有环境 | 复用现有脚本，无额外基础设施 |
| 存储 | <1GB | 新增打标结果 JSONL |

---

## 八、风险登记

| 风险 ID | 风险描述 | 等级 | 缓解措施 |
|---------|---------|------|---------|
| R001 | 竞品品牌名与通用词冲突（如 "Willow" 也是植物名） | 中 | 增加上下文校验（"willow pump" vs "willow tree"） |
| R002 | 负面标签与现有正面标签互斥规则误触发 | 中 | 强制否定词检测优先，人工校验 200 条边界案例 |
| R003 | Zendesk 极简规则过度匹配（如 "return" 匹配 "return customer"） | 低 | 添加排除模式（"return customer" → 不打 Return_Request） |
| R004 | 德/法情感词在混合语言评论中误匹配 | 低 | 语言检测前置，仅对德/法评论应用对应关键词 |
| R005 | 新增负面标签导致整体情感分布偏移 | 低 | 对比 v3.7 情感分布基线，偏差 >10% 时告警 |

---

## 九、交付物清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 品牌关键词库 | `04-输出结果/08-辅助数据/brand_keyword_library.json` | 自有品牌 + 竞品品牌关键词 |
| 品牌识别 Label Functions | `02-脚本工具/04-数据处理/brand_label_functions.py` | 品牌维度标签规则 |
| 更新后的通用标签器 | `02-脚本工具/01-标签进化/general_tag_labeler.py` | 含 8 个负面标签 + 德/法扩展 |
| Zendesk 极简规则 | `02-脚本工具/05-NPS管道/zendesk_minimal_labeler.py` | 12 条极简规则 |
| 新增 ALCHEmist Label Functions | `04-输出结果/alchemist_label_functions_v38.py` | ~25 个负面缺陷标签 |
| 标签字典 v3.8 | `04-输出结果/01-字典版本/tag_dictionary_v3.8.xlsx` | 600+ 标签 |
| 最终打标结果 | `04-输出结果/unified_labeling/phase4_final_labeled.jsonl` | 全量重新打标 |
| Phase 4 审计报告 | `04-输出结果/03-审计报告/phase4_final_audit_report.md` | 覆盖率验证 + 质量评估 |

---

## 十、下一步行动

1. **立即启动 T1.1**：品牌关键词库构建（0.5 天，零成本）
2. **同步启动 T3.1**：`general_tag_labeler.py` 负面标签扩展（1 天，零成本）
3. **关键依赖确认**：Amazon 竞品原始数据中是否保留 ASIN 类目信息？影响 T1.4 品线推断的准确率
