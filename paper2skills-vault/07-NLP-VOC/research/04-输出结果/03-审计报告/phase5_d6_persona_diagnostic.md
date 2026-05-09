---
name: phase5-d6-persona-diagnostic
description: Phase 5 D6 画像标签诊断报告。涵盖 55 标签整体渗透率、各维度/数据源/品线渗透率热力表、死标签清单。当评估 D6 画像标签器效果、定位冷门标签、为字典进化 v4.1 选择候选时使用。
title: Phase 5 D6 画像标签诊断报告
doc_type: audit
module: voc-nlp
topic: persona-diagnostic
status: final
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# Phase 5 D6 画像标签诊断报告

**总记录数**：5000
**至少命中 1 个画像标签**：3696 / 5000 = **73.92%**（红线 ≥ 60%）
**命中至少 1 次的标签**：**54 / 55**（红线 ≥ 45）
**平均每条命中**：1.70 个标签

**D6 场景 2 判定**：渗透率 ✅  ｜  标签覆盖 ✅

## 一、各维度命中分布

| 维度 | 命中记录数 | 记录占比 | 覆盖标签数 | 总标签数 |
|---|---:|---:|---:|---:|
| WHO | 439 | 8.78% | 12 | 13 |
| WHY | 250 | 5.00% | 6 | 6 |
| WHAT | 1713 | 34.26% | 10 | 10 |
| WHEN | 467 | 9.34% | 6 | 6 |
| HOW | 658 | 13.16% | 10 | 10 |
| EMOTION | 1091 | 21.82% | 7 | 7 |
| LANGUAGE | 2064 | 41.28% | 3 | 3 |

## 二、数据源 × 维度 渗透率热力表

| data_source (n) | WHO | WHY | WHAT | WHEN | HOW | EMOTION | LANGUAGE | 至少 1 个 |
|---|---|---|---|---|---|---|---|---|
| amazon_competitor (2670) | 11.3% | 6.6% | 47.5% | 13.2% | 16.1% | 26.9% | 27.2% | 74.1% |
| trustpilot (1369) | 3.7% | 1.9% | 7.8% | 3.4% | 7.9% | 11.2% | 69.8% | 78.7% |
| zendesk (647) | 7.3% | 2.5% | 28.6% | 4.9% | 9.4% | 19.0% | 42.7% | 62.1% |
| momcozy (270) | 13.7% | 11.5% | 51.1% | 13.0% | 20.4% | 33.0% | 35.6% | 78.5% |
| reddit (44) | 4.5% | 0.0% | 36.4% | 2.3% | 9.1% | 18.2% | 25.0% | 59.1% |

## 三、品线 × 维度 渗透率热力表

| product_line (n) | WHO | WHY | WHAT | WHEN | HOW | EMOTION | LANGUAGE |
|---|---|---|---|---|---|---|---|
| 内衣服饰 (762) | 11.9% | 9.6% | 45.9% | 7.2% | 19.3% | 29.3% | 41.1% |
| 吸奶器 (663) | 9.8% | 4.8% | 47.1% | 12.4% | 13.6% | 28.8% | 43.3% |
| 智能母婴电器 (488) | 12.5% | 2.9% | 59.0% | 22.1% | 21.3% | 34.0% | 41.4% |
| 家居家纺 (381) | 18.9% | 13.4% | 48.8% | 19.4% | 17.6% | 29.4% | 26.0% |
| 喂养电器 (267) | 8.6% | 6.7% | 43.4% | 12.0% | 15.7% | 27.0% | 40.1% |
| 母婴综合护理 (112) | 6.2% | 2.7% | 33.9% | 17.9% | 14.3% | 32.1% | 32.1% |

## 四、Top-20 标签命中

| 排名 | tag_id | tag_en | dimension | 命中数 | 占比 |
|---:|---|---|---|---:|---:|
| 1 | P-L2-55 | language_french | LANGUAGE | 1036 | 20.72% |
| 2 | P-L2-53 | language_english | LANGUAGE | 943 | 18.86% |
| 3 | P-L2-47 | care_driven | EMOTION | 722 | 14.44% |
| 4 | P-L2-54 | language_german | LANGUAGE | 712 | 14.24% |
| 5 | P-L2-27 | comfort_focused | WHAT | 631 | 12.62% |
| 6 | P-L2-21 | portable_seeker | WHAT | 498 | 9.96% |
| 7 | P-L2-39 | price_sensitive | HOW | 418 | 8.36% |
| 8 | P-L2-20 | quiet_seeker | WHAT | 244 | 4.88% |
| 9 | P-L2-28 | quality_focused | WHAT | 223 | 4.46% |
| 10 | P-L2-25 | smart_tech_seeker | WHAT | 214 | 4.28% |
| 11 | P-L2-24 | easy_clean_seeker | WHAT | 176 | 3.52% |
| 12 | P-L2-23 | high_capacity_seeker | WHAT | 165 | 3.30% |
| 13 | P-L2-49 | hopeful | EMOTION | 150 | 3.00% |
| 14 | P-L2-16 | solve_pain_discomfort | WHY | 146 | 2.92% |
| 15 | P-L2-33 | home_user | WHEN | 135 | 2.70% |
| 16 | P-L2-22 | hands_free_seeker | WHAT | 133 | 2.66% |
| 17 | P-L2-34 | outdoor_user | WHEN | 130 | 2.60% |
| 18 | P-L2-32 | travel_user | WHEN | 122 | 2.44% |
| 19 | P-L2-06 | prenatal | WHO | 121 | 2.42% |
| 20 | P-L2-29 | convenience_focused | WHAT | 121 | 2.42% |

## 五、死标签（本批 0 命中）

| tag_id | tag_en | dimension | 首批关键词 | 建议 |
|---|---|---|---|---|
| P-L2-10 | extended_nursing | WHO | extended breastfeeding, nursing beyond, over 1 year | 采样稀缺，D7/D9 开集阶段可挖更多关键词或 LLM 兜底 |

## 六、观察结论

✅ **D6 场景 2 全通过**：渗透率与标签覆盖双过红线，画像标签器可进 D7 集成。
