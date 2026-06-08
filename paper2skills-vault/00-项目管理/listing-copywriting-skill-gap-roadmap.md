---
name: listing-copywriting-skill-gap-roadmap
description: Listing 文案生成技能缺口分析与论文萃取选题清单。用于「路径B补白」Sprint，填补跨境电商 Stage 3 内容制作的文案生成空白。当下一轮论文萃取 Sprint 启动时使用。
---

# Listing 文案生成 Skill 缺口 & 萃取选题清单

> **背景**：Playbook 现有 10 个内容制作 skill，全部是视频生成。文字文案（标题/Bullet/描述/ST 关键词）= 0 个 skill，而这是跨境卖家**每周必做**的高频刚需。`Skill-Listing-Quality-Scoring`（质量评分诊断）已有，但文案自动生成本身的 skill 完全缺失。

---

## 当前状态

| 已有 Skill | 能力范围 | 缺口 |
|---|---|---|
| `Skill-Listing-Quality-Scoring` | 评分诊断：告诉你哪里差（KDD 2023 Amazon，CTR +10.26%）| 不能生成文案，只能评分 |
| `Skill-AI-Brand-Storytelling` | 品牌故事生成 | 不针对 Amazon Listing 格式 |
| `Skill-Negative-Keyword-Safe-Guard` | 关键词否定过滤 | 不是文案生成 |

**缺口一句话**：我们能告诉卖家「你的标题有问题」，但还不能帮他们「写一个更好的标题」。

---

## 萃取优先级清单（P0 → P2）

### P0 — 立即萃取（最高频刚需，有成熟论文）

#### 选题 1：Amazon Listing 文案自动生成
- **业务问题**：上新 SKU 需要写标题（200字符）+ 5条 Bullet + 商品描述（2000字），人工撰写 1 个 SKU 约需 2-3 小时
- **论文方向**：
  - `APGT: Attribute-Guided Product Title Generation` (WWW 2023 / KDD 2022)
  - `Multimodal Product Description Generation` — 输入商品图片+属性→输出描述
  - Amazon 内部论文 `PLM for E-commerce Product Description` (ACL 2022)
- **预期 Skill 输出**：给定商品属性（品类/材质/功能/目标用户），自动生成符合 Amazon 格式要求的完整 Listing 文案
- **ROI 估算**：一个运营 10 SKU/月的卖家，节省 20-30 小时人工 = 约 5-8 万/年人力成本

#### 选题 2：SEO 关键词挖掘与 Listing 植入
- **业务问题**：不知道哪些关键词能带来流量，ST 关键词填充随意，标题关键词布局不合理
- **论文方向**：
  - `E-commerce Search Term Optimization` (SIGIR 2023)
  - `Keyword-Guided Listing Optimization` — 从竞品 Listing 挖掘高价值关键词
  - Amazon 搜索广告中的关键词相关性建模
- **预期 Skill 输出**：给定品类和竞品 ASIN，输出推荐关键词列表（按搜索量/竞争度/相关性排序）+ 在 Listing 中的最优布局建议
- **ROI 估算**：关键词优化带来自然排名提升，保守估计 CVR +15% = 月增 GMV 约 30 万（按 200 万月 GMV）

---

### P1 — 次优先萃取（有成熟方案，价值高）

#### 选题 3：A+ Content / EBC 结构化生成
- **业务问题**：A+ Content 是 Amazon 认证品牌专属功能，能提升转化率 5-10%，但大多数卖家不会充分利用
- **论文方向**：
  - `Structured Visual-Text Content Generation for E-commerce` (MM 2023)
  - 多模态 A+ 内容生成：输入产品图+卖点→输出模块化 A+ 布局建议
- **预期 Skill 输出**：给定品类卖点和产品图，自动生成 A+ Content 各模块文案（品牌故事/技术规格/使用场景/对比图）

#### 选题 4：跨市场 Listing 本地化
- **业务问题**：直接把英文 Listing 机翻为德语/日语，文化语气不对，转化率比英语市场低 25-35%
- **论文方向**：
  - `Cultural-Aware E-commerce Text Localization` — 不只翻译，还需文化适配
  - `Cross-Lingual Product Title Transfer with Cultural Adaptation`
- **预期 Skill 输出**：给定英文 Listing，输出文化适配的德语/日语版本（不是字面翻译，而是语气/表达习惯适配）

---

### P2 — 后续萃取（有价值但非紧迫）

#### 选题 5：竞品 Listing 差距分析自动化
- **业务问题**：想知道「我的 Listing 和竞品相比差在哪里」，但人工对比耗时
- **论文方向**：
  - 结合 `Skill-Listing-Quality-Scoring`（已有）做竞品 Listing 评分对比
  - `Comparative Text Analysis for Product Listings`

#### 选题 6：Listing 合规内容检查（违规声明自动检测）
- **业务问题**：ChatGPT 生成的文案可能含违规声明（"clinically proven" / "FDA approved"），导致 listing 下架
- **注意**：`Skill-Compliance-Scored-Guardrail-Orchestration`（已有）部分覆盖，需要 Amazon Listing 特化版本

---

## 与现有 Skill 的集成路径

完整「Listing 优化闭环」（萃取后可实现）：

```
竞品关键词挖掘（选题2）
    ↓
Listing 文案生成（选题1）
    ↓
质量评分诊断（Skill-Listing-Quality-Scoring ← 已有）
    ↓
合规检查（Skill-Compliance-Scored-Guardrail-Orchestration ← 已有）
    ↓
A+ Content 生成（选题3）
    ↓
多市场本地化（选题4）
    ↓
效果追踪（A/B测试 ← Skill-Switchback / Skill-AB-Experimental-Design 已有）
```

萃取 P0 两个选题后，即可构建「手册 14：Listing 全链路优化手册」。

---

## 推荐论文搜索策略

搜索词（ArXiv / ACL / KDD / WWW / SIGIR）：

```
"product title generation" e-commerce
"listing optimization" amazon NLP
"attribute-to-text generation" product description
"e-commerce copywriting" large language model
"SEO keyword" listing generation
"cross-lingual product description" localization
```

重点关注以下机构的相关论文：
- Amazon Research（有大量电商特化 NLP 工作）
- Alibaba DAMO（多语言电商文案）
- JD.com Research（商品描述生成）
- Walmart Labs（Listing 优化实践）
