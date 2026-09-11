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
- 时间：2026-05-09T22:39:38

## 一、Per-source Precision

| 数据源 | samples | tags | accepted | rejected | judge_failed | **precision** | 评级 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| trustpilot (D4 F4) | 50 | 57 | 51 | 6 | 0 | **0.895** | 🟢 良 |
| zendesk (D5 F3) | 50 | 54 | 46 | 8 | 0 | **0.852** | 🟢 良 |
| amazon_competitor (D5 F3) | 50 | 54 | 49 | 5 | 0 | **0.907** | 🟢 优 |

> 阈值：precision ≥ 0.85 视为可信；< 0.85 标记质量风险

## 二、Rejected 样本（每源最多 10 条）

### trustpilot (D4 F4)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| tp_6725093c6f42c60311860cf9_33654 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | no mention of returning the TV, only waiting for an exchange | J’ai acheté un téléviseur il y a quatre… J’ai acheté un téléviseur il y a quatre mois, il tombe en panne. Je contacte le |
| tp_60a40e51f9f4870b70078ef7_13531 | TAG_L2_002 (一次解决问题/issue solved right away) | replacement was shipped quickly but required multiple troubleshooting attempts and a long Q&A before resolution, so not  | Pump shutting down intermittently My pump began shutting down immediately when I turned it on and hit the start button.  |
| tp_6495f6721dbe1dee06f9ef43_51995 | TAG_L2_002 (一次解决问题/issue solved right away) | no mention of the address mistake being fixed | I made a mistake about shipping address… I made a mistake about shipping address but customer care team (Kathlene) was r |
| tp_69307ef666464f29f45e52ac_33162 | TAG_I_001 (信息难找/hard to find information) | no mention of difficulty finding product info, only stock-outs | Mauvaise magasin ⭐☆☆☆☆ – Très déçu de mon expérience chez Orchestra Je suis vraiment déçu par ma visite en magasin. L’ac |
| tp_686a32e218a3a2f06dcd47aa_47790 | TAG_P1_001 (核心卖点清晰/clear value proposition) | review never states what the product's core selling point is, only personal pros/cons | Mamabelly Auf Grund meiner Schwangerschaft brauchte ich eine Stütze in der Nacht und konnte mich mit einem Stillkissen n |
| tp_63a0b2674d0773066a1ee3c7_1942 | TAG_P2_001 (错件/漏件/多件/wrong/missing/extra parts) | review complains about missing half of the shipment, not about wrong/missing/extra parts inside a package. | Avoid Avoid Avoid Avoid, Avoid, Avoid! Made an order nearly a month ago now and still haven't received half of it. For u |

### zendesk (D5 F3)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| zd_1098394_15240 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | customer only asks how to exchange size, never mentions returning because item didn't meet expectations | Bonjour,  Je viens de recevoir ma commande mais la taille c’est pas bon, comment je peux faire une échange de taille ?   |
| zd_1029848_19446 | TAG_I_001 (信息难找/hard to find information) | review complains about missing delivery, not about difficulty finding information | Hello,    I purchased the breast pump on the 11th February but hasn’t received  it yet and last tracking info is also fr |
| zd_928291_45856 | TAG_I_001 (信息难找/hard to find information) | review complains about missing shipping update, not difficulty finding product info | Hello  I just would like to ask for an update with regards to my purchase last December 16, 2025, I ordered a Sterilizer |
| zd_1038561_17667 | TAG_I_001 (信息难找/hard to find information) | review complains about forced app install for tracking, not difficulty finding product info | Conversation with Sevi 🧚‍♀️ ====================== Dear team,     I am writing regarding order JJ1448082 which still has |
| zd_981453_45614 | TAG_P2_001 (错件/漏件/多件/wrong/missing/extra parts) | customer only reports wrong size, not a missing/extra part. | Hello,   I have this item I purchased and noticed it is not 50x50, and is 45x45. May I please have a replacement? |
| zd_952054_39918 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | customer wants cancellation/refund for non-shipment, not because product failed to meet expectations | Please cancel my order and issue a refund. It has been almost a month ordered and no shipping details, and I am unable t |
| zd_994635_34779 | TAG_L1_074 (退货-不符预期/return not meeting expectations) | customer wants to cancel for slow shipping, not because item fails expectations | Hello,  Please cancel and refund order JJ1429662. This order was placed on January 23 and has not shipped, I no longer h |
| zd_993429_19492 | TAG_L1_040 (延迟/general core product performance issue) | no mention of delay or latency, only charging failure | Order number :  114-6576496-4034634  My pump battery is not charging on one of my pumps. I’m wondering if I can get a ne |

### amazon_competitor (D5 F3)

| review_id | tag_id (cn/en) | Kimi 判官 reason | text 截取 |
|---|---|---|---|
| amz_B0C5B1SLFL_175491 | TAG_P1_001 (核心卖点清晰/clear value proposition) | no mention of any specific feature or selling point—only that baby accepts it | The ONLY pacifier my baby will take |
| amz_B07WCBD6YC_181463 | TAG_L1_031 (配件破损/断裂/parts crack or break) | bristles falling out is not the same as parts cracking or breaking | My son didn’t use it too much, but the bristles are already came out. The quality is not good even for such a famous bra |
| amz_B00MYRXSE2_141128 | TAG_L2_006 (不会再次购买/won't buy again) | user only says switching to UV sterilizer, never states won't buy this product again | The product manual requires distilled water which is a pain because you have to refill every use - or every other use. I |
| amz_B09Q86424N_178821 | TAG_P1_001 (核心卖点清晰/clear value proposition) | no mention of brand or core selling point, only basic function noted | Can warm up both bottles. It warms quicker than expected. |
| amz_B0D9YB1F2M_15662 | TAG_L2_002 (一次解决问题/issue solved right away) | took an update weeks later, not solved in the first contact | I contacted the seller because the humidifier started to turn off by itself, every 4 or 5 hours, he told me that it is n |

## 三、综合评估

- **整体 precision** = 146/165 = **0.885**
- 阈值：≥ 0.85 视为通过
- 判定：✅ **通过** — F4/F3 LLM 输出整体质量可信
