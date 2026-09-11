# 03-数据资产: 数据层

本目录存放 VOC 数据资产。

**注意**: 大型数据文件（>10MB）已加入 `.gitignore`，不纳入 git 版本控制。本地文件保留，可通过元数据清单了解数据内容。

---

## 子目录

| 子目录 | 内容 | 规模 | 状态 |
|--------|------|------|------|
| `momcozy/` | Momcozy 自有 VOC 数据（Zendesk + Trustpilot） | 355,697 条 | 已采样 |
| `third-party/reddit/` | Reddit 社区采样数据 | ~4k 条 | 已采样 |
| `third-party/trustpilot/` | Trustpilot 评论数据 | ~2.6k 条 | 已采样 |
| `amazon-competitor/` | 竞品 Amazon VOC 数据（去重后） | ~165 ASIN | 已采样 |
| `产品主数据/` | **产品主数据 + enriched 版本** | 431 SPU | 已补全 |
| `原种子标签/` | V3.0 标签字典模板 + 映射关系 | — | 已复用 |
| `高质量数据源/` | 采样后的均衡数据集（Amazon/Trustpilot/Reddit/Zendesk） | — | 已生成 |

---

## 产品主数据（Phase 3 P1 关键产出）

| 文件 | 说明 | 更新内容 |
|------|------|----------|
| `产品主数据_VOC维度关联.xlsx` | 原始产品主数据 | 431 SPU，52个空值英文名 |
| `产品主数据_VOC维度关联_enriched.xlsx` | 补全英文名 + 新增"产品型号"列 | 52空值补全，255条分列 |
| `产品主数据_VOC维度关联_v3.5.xlsx` | **最新版：补全147条品线映射** | 147 SPU 新增品线 |
| `product_spoken_keywords.json` | 消费者口语表达变体词库 | 255 SKU × 5 变体 = 1,189 条 |

---

## 多语言关键词映射（Phase 3 P2 产出）

| 文件 | 说明 |
|------|------|
| `multilingual_keyword_map.json` | 法语45个 + 德语54个关键词映射 |

---

## 数据清单

详见 `../01-设计文档/data-inventory/数据资产盘点与缺口分析.md`
