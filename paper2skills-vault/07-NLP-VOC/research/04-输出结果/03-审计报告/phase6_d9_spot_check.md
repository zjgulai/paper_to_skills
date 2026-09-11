---
name: phase6-d7-llm-spot-check
description: Phase 6 D7 #3 LLM 输出抽样质量评估 — 用 Kimi 独立判定 F4/F3 LLM 重打的 tag 是否被原文支撑。当审计 trustpilot 多语言 / zendesk / amazon 三批 LLM 输出可信度时使用。
date: 2026-05-10
phase: phase6
day: D7
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D7 #3 LLM 输出抽样质量评估

- 每源抽样：50 records
- 判官：Kimi (独立第二意见，避免与 DeepSeek 同模型偏见)
- 时间：2026-05-10T15:27:52

## 一、Per-source Precision

| 数据源 | samples | tags | accepted | rejected | judge_failed | **precision** | 评级 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| trustpilot (D4 F4) | 50 | 92 | 52 | 40 | 0 | **0.565** | 🔴 差 |
| zendesk (D5 F3) | 50 | 95 | 53 | 42 | 0 | **0.558** | 🔴 差 |
| amazon_competitor (D5 F3) | 50 | 85 | 55 | 30 | 0 | **0.647** | 🔴 差 |

> 阈值：precision ≥ 0.85 视为可信；< 0.85 标记质量风险

## 二、Rejected 样本（每源最多 10 条）

### trustpilot (D4 F4)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| amz_B07Q3YX5YN_167801 | TAG_L1_153 (/) | no evidence or context provided for this tag | Offers no holding power whatsoever. If you're hoping for something to help hold your tummy and midriff rolls in, as comm |
| amz_B07Q3YX5YN_167801 | TAG_L2_006 (不会再次购买/won't buy again) | no statement about future purchase intent | Offers no holding power whatsoever. If you're hoping for something to help hold your tummy and midriff rolls in, as comm |
| amz_B0C7T7KQHL_68194 | TAG_GEN_007 (/) | no mention of compression or shaping function | Perfect for what I needed. I needed something to wear under this mesh top and hold me is. It did just that. I could have |
| amz_B0C7T7KQHL_68194 | TAG_GEN_003 (/) | no reference to tummy control or waist slimming | Perfect for what I needed. I needed something to wear under this mesh top and hold me is. It did just that. I could have |
| amz_B0CV4KX7C5_16157 | TAG_GEN_003 (/) | no mention of safety concerns or issues | We have one of the other ergobaby carriers that isn't a hip seat. We also have a separate hip seat carrier. It was nice  |
| amz_B0CV4KX7C5_16157 | TAG_GEN_002 (/) | no mention of difficulty using or learning curve | We have one of the other ergobaby carriers that isn't a hip seat. We also have a separate hip seat carrier. It was nice  |
| amz_B0CV4KX7C5_16157 | TAG_GEN_004 (/) | no mention of cleaning or maintenance | We have one of the other ergobaby carriers that isn't a hip seat. We also have a separate hip seat carrier. It was nice  |
| amz_B0C5B1SLFL_148656 | TAG_I_004 (评价真实可参考/helpful authentic reviews) | no mention of other reviews or their authenticity | This is similar to the MAM 0-3 and 0-6m pacifiers (the only ones that don't fall out of my daughter's mouth) in its nipp |
| amz_B0FDQRVMYQ_62882 | TAG_GEN_008 (/) | no mention of temperature control or adjustable heat settings | It’s large enough to heat a full bottle of milk or water for multiple feedings, yet the unit itself remains sleek and po |
| zd_972121_37389 | TAG_SRV_09 (产品咨询/使用指导/Product_Inquiry) | no review text provided to check for any product usage/installation/compatibility question | Conversation with Web User 697395f1e23232da844d5de0 |

### zendesk (D5 F3)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| tp_668921b8b8f14d1afd86d282_37069 | TAG_GEN_014 (/) | no mention of any technical issue or need for troubleshooting | Excellent Customer Service The customer service rep I spoke with was a delight! |
| amz_B089HQRW9C_48819 | TAG_GEN_003 (/) | no mention of product durability or longevity beyond adjustability | I purchased these for my 3.5 month old son so we could attend the Christmas parade without worrying about loud noises. I |
| zd_1128902_41598 | TAG_SRV_09 (产品咨询/使用指导/Product_Inquiry) | no product-related question or usage/installation topic mentioned | Conversation with Web User 69c320c7db51c9424ca6a60c |
| zd_1128902_41598 | TAG_ZEN_R012 (一般咨询/general_inquiry) | no actual inquiry text provided to confirm a general question | Conversation with Web User 69c320c7db51c9424ca6a60c |
| mz_R3UTVHHXM0S2M9_9217 | TAG_GEN_E005 (耐用性/durability) | no mention of how long the tablets or their effect last | These washing blocks do a great job at cleaning bottles and pump parts — they dissolve well and leave everything fresh w |
| mz_R3UTVHHXM0S2M9_9217 | TAG_GEN_S001 (工作场景/work_scenario) | review is about home baby-bottle washing, not a work scenario | These washing blocks do a great job at cleaning bottles and pump parts — they dissolve well and leave everything fresh w |
| amz_B0B1GZLP2P_116315 | TAG_GEN_003 (/) | no mention of product quality issues, only sizing/fit concern | I was so excited to try it once it was here. I got a XL thinking it will fit but it didn't. I couldn't even get it up to |
| amz_B0DHV475TK_43301 | TAG_GEN_S001 (工作场景/work_scenario) | no mention of any work or job-related usage. | This is a dupe of the momcozy bottle washer that works just as great! Better than the baby breeza bottle washer also bec |
| amz_B0DD3R48H2_28980 | TAG_GEN_007 (/) | no specific attribute mentioned to classify | This is really a Shapewear as it’s not stretchy but it stretches enough so that you can look snatched. It truly is a Sha |
| amz_B0DD3R48H2_28980 | TAG_L1_162 (尺码小/size runs small) | review says it's true to size, not small | This is really a Shapewear as it’s not stretchy but it stretches enough so that you can look snatched. It truly is a Sha |

### amazon_competitor (D5 F3)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| amz_B0BTBKMG25_41278 | TAG_GEN_007 (/) | no mention of any assembly issues or difficulty | Love the crib! It's well constructed and feels safe for our little one. Like many others have lamented in having trouble |
| amz_B0D3RX764R_127636 | TAG_GEN_008 (/) | no mention of fragrance or scent at all | Love it.  Texture is good. Absorbency is great. Love that it doesnt have bad chemicals and ingredients on the product. W |
| amz_B09YV1K1BB_128637 | TAG_GEN_A002 (材质提及/material_mention) | no mention of any material or fabric composition | They produce wonderful, clean air and with the little scent cloth you can have your home smelling fantastic and have cle |
| tp_68a5b3029f8d68cc127d9575_31969 | TAG_SRV_03 (物流/配送/Shipping_and_Delivery) | only mentions absence of tracking, not a query about it | Vertrauenswürdiges Produkt Das Produkt sieht hochwertig aus und dazu gibt gute Einleitungen. Die Firma hat Erfahrung und |
| amz_B0DRCVX8H3_177868 | TAG_I_003 (参数不透明/unclear specs) | no mention of missing or unclear specs | Product runs very small up a size ! |
| amz_B0DCJRZ7W5_86851 | TAG_GEN_N002 (不舒适/uncomfortable) | review explicitly says "without being uncomfortable" | I was surprised how good this shaping thong is! It gives a nice lift and smooth shape without being uncomfortable. The q |
| amz_B00LZKC1S8_161594 | TAG_GEN_017 (/) | no mention of softness or texture in the review | As a gift for my friend, her baby loved it very much |
| amz_B087TM6NDK_84920 | TAG_GEN_003 (/) | no mention of any odor or smell in the review | Very soft and comfortable. A little challenging to put on, but that is mostly due to the shape of the pillow. Has a larg |
| mz_R1ZJBP02T66C4T_80 | TAG_P1_015 (产品价格/price concern) | no mention of price, cost, or value-for-money concerns | I'm a side sleeper in her 4th pregnancy. I have the huge pregnancy pillow, the one that feels like adding an extra adult |
| mz_R1ZJBP02T66C4T_80 | TAG_P1_020 (/) | tag label is empty and review gives no matching theme | I'm a side sleeper in her 4th pregnancy. I have the huge pregnancy pillow, the one that feels like adding an extra adult |

## 三、综合评估

- **整体 precision** = 160/272 = **0.588**
- 阈值：≥ 0.85 视为通过
- 判定：🔴 **不达标** — 需重打或调整 prompt
