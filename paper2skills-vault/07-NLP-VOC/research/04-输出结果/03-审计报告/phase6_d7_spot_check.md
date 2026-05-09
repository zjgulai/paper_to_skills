---
name: phase6-d7-llm-spot-check
description: Phase 6 D7 #3 LLM 输出抽样质量评估 — 用 Kimi 独立判定 F4/F3 LLM 重打的 tag 是否被原文支撑。当审计 trustpilot 多语言 / zendesk / amazon 三批 LLM 输出可信度时使用。
date: 2026-05-09
phase: phase6
day: D7
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D7 #3 LLM 输出抽样质量评估

- 每源抽样：50 records
- 判官：Kimi (独立第二意见，避免与 DeepSeek 同模型偏见)
- 时间：2026-05-09T21:31:11

## 一、Per-source Precision

| 数据源 | samples | tags | accepted | rejected | judge_failed | **precision** | 评级 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| trustpilot (D4 F4) | 50 | 67 | 47 | 20 | 0 | **0.701** | 🔴 差 |
| zendesk (D5 F3) | 50 | 61 | 43 | 18 | 0 | **0.705** | 🔴 差 |
| amazon_competitor (D5 F3) | 50 | 63 | 32 | 31 | 0 | **0.508** | 🔴 差 |

> 阈值：precision ≥ 0.85 视为可信；< 0.85 标记质量风险

## 二、Rejected 样本（每源最多 10 条）

### trustpilot (D4 F4)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| tp_6818531a4f68b9a812e4b999_38178 | TAG_I_001 (信息难找/hard to find information) | review complains about slow shipping, not about difficulty finding product information | Prima Produkte Bestellten Artikel haben uns gut gefallen. Einzig schlechte war der Versand. Hier sollte überlegt werden  |
| tp_5c49dcec97afa106306ff2e6_57048 | TAG_P1_009 (物超所值/worth the price) | no mention of price or value | Die Produkte haben eine tolle Qualität… Die Produkte haben eine tolle Qualität und die Lieferung war sehr schnell. Ich h |
| tp_692dec4dd08d271d55284841_42770 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | no mention of a return or refund, only complaint about poor sizing | Très bon rapport qualité prix j'habille… Très bon rapport qualité prix j'habille mes jumeaux depuis quelque temps chez o |
| tp_5aa02e0fd5a57009786c266c_17122 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | review complains about fit/appearance but never mentions returning the item | Dress was terrible made me look frumpy Dress was terrible made me look frumpy. I would advise you buy a size smaller at  |
| tp_64258b6a1f08d0d08492e7e6_17242 | TAG_L2_002 (一次解决问题/issue solved right away) | no mention of any issue being solved | Best Service Quick, considerate, kind, and so helpful! Most pleasant customer service experience I have ever had. |
| tp_6320831d6a3e1ed2c3d19b85_56213 | TAG_L2_008 (优化售后政策/advance after sales policy) | no mention of any after-sales policy, only long wait for spare part | Lo peor de lo peor Lo peor de lo peor,nunca más compraré algo de esta marca son los más incompetentes del mundo casi dos |
| tp_6892ca6b2a1d5b60b61d069b_97980 | TAG_L2_003 (客服专业有耐心/knowledgeable support) | review text is only two repeated words with no mention of support staff or professionalism | The helpfulness The helpfulness |
| tp_67d2a676cf9ce6a97b24dacc_56592 | TAG_L2_002 (一次解决问题/issue solved right away) | no evidence the out-of-stock issue was fully solved in one go, only that communication was quick | Entrega rapida, buen servicio al cliente, productos de calidad No habia stock de uno de los productos que habia pedido p |
| tp_643e8c6e734f97e68fa05036_6729 | TAG_P2_001 (错件/漏件/多件/wrong/missing/extra parts) | no mention of wrong, missing or extra parts, only refusal to replace one of two pieces | It wasn’t an easy fix It wasn’t an easy fix. They gave me a hard time and said they would replace one piece and not the  |
| tp_67b13d9b11b1fd911a80b489_83874 | TAG_P1_001 (核心卖点清晰/clear value proposition) | no mention of any specific value proposition or core function—only generic praise | Une marque référente dans le secteur… Une marque référente dans le secteur des vêtements d'allaitement, adaptée aux mama |

### zendesk (D5 F3)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| zd_1004415_8848 | TAG_I_001 (信息难找/hard to find information) | review is about missing protection plan, not difficulty finding information | Hello my name is Dakota I placed and order for the bottle washer and I added the protection for the extra 5$ however I d |
| zd_1134462_13270 | TAG_I_001 (信息难找/hard to find information) | review complains about delayed shipment, not about difficulty finding product or order information | Hey I been waiting on my wife nursing bras that I got for her and they haven't came in yet and still saying the order is |
| zd_1009611_44169 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | no mention of product failing to meet expectations, only asking how to return and status of refund | Hello I would like to return this item. Please inform me how to get that completed.   Thank you ====================== T |
| zd_1079900_20182 | TAG_P2_007 (修改产品SKU/modify product) | review asks to cancel/replace order, not to modify the product SKU itself | Hi there -  I just placed an order and am looking to cancel to get a different item instead. Order number is: JJ1525834  |
| zd_1112507_13697 | TAG_I_001 (信息难找/hard to find information) | review complains about missing shipment, not about difficulty finding information | Sehr geehrte Damen und Herren,   vor 11 Tagen habe ich etwas bei Ihnen bestellt, jedoch bisher keine Versandbestätigung  |
| zd_986727_38752 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | customer wants to return because they bought another item elsewhere, not because the product failed to meet expectations | This is a follow-up to your previous request #960636 "Re: Commande #JJ1414134 con..."  Bonjour,  Vu que votre commande a |
| zd_986734_33117 | TAG_P2_004 (退货-物流发货延迟/return shipping delays) | customer complains about non-arrival, not about return shipping delays | Sehrgehhrte Damen und Herren, 14.01.2026 hab ich eine Momvozy Elektrische Babynagelfeile bestellt und meine Bestellungen |
| zd_1000998_19558 | TAG_L1_040 (延迟/general core product performance issue) | no mention of delay; issue is zero suction, not delayed performance | Hello,  We purchased the M5 wearable breast pump and there is no suction. We verified the seals and battery level and ev |
| zd_919631_2866 | TAG_P2_006 (Return未明确原因/return not specified) | customer clearly states reason: friend already owns one | Hello,   I would like to return an item. Purchased for a friend but she already has one. Package is unopened.   Thanks,  |
| zd_1096093_5661 | TAG_P2_001 (错件/漏件/多件/wrong/missing/extra parts) | customer only asks about missing delivery date, not about wrong/missing/extra parts | Sehr geehrte Damen und Herren,   ich habe am 04.03. zwei Produkte bestellt. Die Milchpumpe kam bereits am 06.03. bei mir |

### amazon_competitor (D5 F3)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| amz_B0DMFKC6F8_107407 | TAG_P1_001 (核心卖点清晰/clear value proposition) | review only mentions fit and smooth look, no indication user grasped any brand-level value proposition | Fit just right under my clothes and left a smooth look on my tummy. |
| amz_B0CSFGZCM7_190885 | TAG_P1_009 (物超所值/worth the price) | no mention of price or value in the review | The best pillow ever |
| amz_B09C8LN5RW_164505 | TAG_I_003 (参数不透明/unclear specs) | review only says 'I bought it xs and it didn't fit me at all' with no mention of specs or unclear parameters | La verdad la compre xs y no me soco nada |
| amz_B09N73WTJT_68003 | TAG_P1_009 (物超所值/worth the price) | no mention of price or value-for-money | I went back and forth on buying a pregnancy pillow, and I am SO glad I did!!!! I am 17 weeks pregnant and could not get  |
| amz_B07KF71M2X_159649 | TAG_P1_002 (卖点理解偏差/feature misunderstanding) | no sign user misunderstood the feature, only that it failed to work | This did not control my stomach.Instead of one big tummy, I had 3 rolls. |
| amz_B07KF71M2X_159649 | TAG_L2_006 (不会再次购买/won't buy again) | review never states or implies they won't repurchase | This did not control my stomach.Instead of one big tummy, I had 3 rolls. |
| amz_B0B2Z7YL5P_185038 | TAG_P1_010 (性价比差/poor value for money) | no mention of price or value in the review | i have not found any pregnant pillow that actually work |
| amz_B0CLWXDQ2G_170715 | TAG_L2_005 (会再次购买/would buy again) | no explicit statement of repurchase intent | They loved it. Apparently is doubles as storage during the newborn stage. |
| amz_B07QXL254R_133507 | TAG_P1_001 (核心卖点清晰/clear value proposition) | review only praises weight, colors, quality; never mentions any brand-level selling point or core function | Las sonajas son ligeras en cuanto al peso por lo que a mi bebé le resulto muy comodo el movimiento al colocarlas, los co |
| amz_B0C1Y84QPK_180439 | TAG_P1_001 (核心卖点清晰/clear value proposition) | review only praises shaping and confidence, never states what the product's core selling point actually is | Es increíble con molda tu silueta , te sientes mas confiada y linda, |

## 三、综合评估

- **整体 precision** = 122/191 = **0.639**
- 阈值：≥ 0.85 视为通过
- 判定：🔴 **不达标** — 需重打或调整 prompt
