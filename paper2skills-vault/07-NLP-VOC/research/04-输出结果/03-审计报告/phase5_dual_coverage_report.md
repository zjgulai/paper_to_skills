---
name: phase5-d10-dual-coverage-report
description: Phase 5 D10 双覆盖率审计报告（原始 / 广义 / 业务有效）。当审计 Phase 5 整体覆盖率达成、与 Phase 4 对比、查看排除分布与抽样时使用。
computed_at: 2026-05-09T09:30:42
doc_type: audit-report
module: voc-nlp
status: stable
---

# Phase 5 D10 双覆盖率审计报告

**输入**：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl`
**总样本**：364,569
**计算时间**：2026-05-09T09:30:42

## 一、核心指标

| 指标 | 分子 | 分母 | 覆盖率 | Phase 4 基线 | Δ |
|---|---:|---:|---:|---:|---:|
| 原始覆盖率 | 277,471 | 364,569 | **76.11%** | 82.58% | -6.47pp |
| 广义覆盖率（含品牌命中）| 277,488 | 364,569 | **76.11%** | — | — |
| 业务有效覆盖率 | 277,471 | 310,092 | **89.48%** | — | — |

> 业务有效分母 = 总数 - 排除（54,477 条，占比 14.94%）

## 二、排除桶分布

| 桶 | 定义 | 样本数 | 占总比 |
|---|---|---:|---:|
| `too_short` | text.strip() < 30 字符 | 3,744 | 1.03% |
| `off_category` | product_line 空 且 零标签 且 无品牌 mentions | 50,560 | 13.87% |
| `generic_only` | 零标签 且 去停用词后有效 token < 3 | 173 | 0.05% |

## 三、按数据源拆解

| data_source | 总数 | with_label | 原始覆盖率 | brand_only | 排除合计 | 业务有效覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| amazon_competitor | 194,734 | 164,042 | 84.24% | 14 | 18,694 | 93.18% |
| trustpilot | 99,853 | 71,684 | 71.79% | 0 | 24,005 | 94.51% |
| zendesk | 47,204 | 24,060 | 50.97% | 2 | 9,685 | 64.13% |
| momcozy | 19,808 | 15,609 | 78.80% | 1 | 1,591 | 85.68% |
| reddit | 2,970 | 2,076 | 69.90% | 0 | 502 | 84.12% |

## 四、排除追踪抽样（每桶最多 100 条，供人工 spot check）

### 4.1 `too_short` （text.strip() < 30 字符）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| tp_672e89de6cff0176fca52b43_98156 | trustpilot | — | 0 | Timely arrived Timely arrived |
| amz_B0F58BHLF5_190580 | amazon_competitor | — | 0 | Sharp image.Easy setup. |
| zd_973249_34908 | zendesk | — | 0 | Sent from my iPhone |
| amz_B08M965VSL_193448 | amazon_competitor | — | 0 | One of the best . |
| amz_B0C73ZRQPX_125616 | amazon_competitor | — | 0 | Work very well, I recommend 👍 |
| amz_B0DT48R3BR_184789 | amazon_competitor | — | 0 | Seems to be doing the job! |
| amz_B0CBT9R4FM_126055 | amazon_competitor | — | 0 | Nice fit and I wanted pockets |
| amz_B0CNGK191C_185263 | amazon_competitor | — | 0 | Prácticos , buena calidad |
| amz_B07QL4B6S5_174791 | amazon_competitor | — | 0 | It Fits better than expected. |
| amz_B07WCBD6YC_193461 | amazon_competitor | — | 0 | Daughter loves it |
| tp_5f15f0593f06f20614358554_98178 | trustpilot | — | 0 | Gutes Angebot Gutes Angebot |
| amz_B0BL8Q6SPW_188305 | amazon_competitor | — | 0 | So soothing for my baby. |
| amz_B07NSNGSCV_195533 | amazon_competitor | — | 0 | It's very supportive |
| amz_B0BHQQ81HK_193206 | amazon_competitor | — | 0 | My wife loves this! |
| zd_965321_33122 | zendesk | — | 0 | Conversation with Marinita 💛 |
| amz_B0CGQNN29F_195185 | amazon_competitor | — | 0 | Didn’t do much for me |
| mz_R803QATIJ1QAY_19676 | momcozy | 智能母婴电器 | 0 | Doesn’t stay charged |
| amz_B0DC6JN2XC_184628 | amazon_competitor | — | 0 | Clasps in front break easily. |
| zd_1113399_21593 | zendesk | 吸奶器 | 0 | Conversation with Leah ✨💖 |
| amz_B08QMSNGKS_184792 | amazon_competitor | — | 0 | Excelente, es transparente |
| amz_B07KF61W2N_173846 | amazon_competitor | — | 0 | didn’t work for me at all |
| amz_B0C47L679F_140162 | amazon_competitor | — | 0 | Wife loves the feel and fit. |
| amz_B09F7ZHPKP_191787 | amazon_competitor | — | 0 | Very nice baby loved bed |
| amz_B0BQ6X1CCS_190840 | amazon_competitor | — | 0 | Really good and covers |
| amz_B0CQD8WDYF_186306 | amazon_competitor | — | 0 | Excelente recomendada..!!🥰🥰 |
| amz_B08KHW6C49_195214 | amazon_competitor | — | 0 | Don't  clean very well |
| mz_R2VJBYUNMKNJ70_19773 | momcozy | — | 0 | Not only saline anymore. |
| amz_B0D7VVDPQX_178699 | amazon_competitor | — | 0 | Very useful during pregnancy. |
| amz_B0C1Y9BQBV_195252 | amazon_competitor | — | 0 | Sólido, buena calidad |
| tp_685271f32b2aac0fc362948d_98316 | trustpilot | — | 0 | Envoi rapide Envoi rapide |
| zd_1114578_32863 | zendesk | — | 0 | Conversation with Cláudia |
| amz_B08LR2TTP8_194222 | amazon_competitor | — | 0 | She loves it |
| amz_B014A7MB4G_192518 | amazon_competitor | — | 0 | Drawers are not solid |
| zd_1123053_38975 | zendesk | 吸奶器 | 0 | Conversation with Nooshin |
| amz_B09DY7NQGJ_191200 | amazon_competitor | — | 0 | My son loves them |
| amz_B0CBR8CT7P_194971 | amazon_competitor | — | 0 | No es lo que yo esperaba |
| amz_B09C8N75VM_179464 | amazon_competitor | — | 0 | Muy cómoda, buen material |
| zd_945971_36546 | zendesk | 吸奶器 | 0 | Conversation with Aleksandra |
| amz_B0CG5C9KRS_189688 | amazon_competitor | — | 0 | Snatch my body together |
| amz_B0BFX3XJM3_195155 | amazon_competitor | — | 0 | Im snatched asf oh my |
| amz_B07QRRM748_166109 | amazon_competitor | — | 0 | Muy buena calidad me encantó |
| amz_B09M98HYQX_194354 | amazon_competitor | — | 0 | good product |
| amz_B0D5CRDY5G_194366 | amazon_competitor | — | 0 | good workout |
| amz_B0CBRTKMNC_195098 | amazon_competitor | — | 0 | It did not work for me. |
| amz_B08LR1Y16Y_195585 | amazon_competitor | — | 0 | Excelente muy bonita |
| tp_672014ff0138bedb1110d6ad_83550 | trustpilot | — | 0 | Good product Good product |
| amz_B0F91D1ZVP_184775 | amazon_competitor | — | 0 | Excelente calidad y precio |
| amz_B0F73SZP81_189211 | amazon_competitor | — | 0 | Love. Worth your money. |
| zd_1021013_46136 | zendesk | — | 0 | Conversation with AdaUndis |
| amz_B00QGFO76G_195268 | amazon_competitor | — | 0 | The waist rolls down. |
| amz_B07PXTQWBX_172621 | amazon_competitor | — | 0 | My wife loves these tops. |
| amz_B00LZKC1S8_184606 | amazon_competitor | — | 0 | Bought for a gift - adorable! |
| amz_B0D3CY9X4J_192365 | amazon_competitor | — | 0 | not fit for large women |
| tp_60b53e94f9f4870a94b23894_98110 | trustpilot | — | 0 | Mi hija lo ama Mi hija lo ama |
| tp_69283883b0a7df9523af5976_71423 | trustpilot | — | 0 | Good service Good service |
| mz_R24I9U8NBOE6R0_19241 | momcozy | — | 0 | Really works and stays cool 😎 |
| zd_1089235_43596 | zendesk | — | 0 | Conversation with Xuechun Bai |
| mz_R3GK8ZLO3CBH64_19230 | momcozy | — | 0 | Excelente opción de repuestos |
| zd_1100496_38977 | zendesk | 吸奶器 | 0 | Conversation with Suleyme |
| amz_B0B8TMQDN3_180643 | amazon_competitor | — | 0 | used this product for a baby. |
| amz_B0C5HNF93L_193287 | amazon_competitor | — | 0 | Daughter loves it! |
| zd_1025275_24614 | zendesk | 吸奶器 | 0 | Conversation with Katchi_sipi |
| amz_B0CR134MHF_193575 | amazon_competitor | — | 0 | Fiancée Loves It |
| amz_B0DPSJYSGF_178817 | amazon_competitor | — | 0 | A bit tight but they'll do |
| zd_1097968_21576 | zendesk | 吸奶器 | 0 | Conversation with Houriri |
| amz_B0DD7V3FHP_195063 | amazon_competitor | — | 0 | Compacto y fácil de usar |
| amz_B0B4RLDN8D_185187 | amazon_competitor | — | 0 | Cumplió mis expectativas. |
| amz_B0C4NZHX61_195332 | amazon_competitor | — | 0 | Funciona de maravilla |
| rd_747964_4909 | reddit | — | 0 | Conversation with Sara Mohsen |
| amz_B0BPD54HQ6_190830 | amazon_competitor | — | 0 | My daughter  loves it |
| zd_1105463_24625 | zendesk | 吸奶器 | 0 | Conversation with Jennette |
| zd_1034952_29193 | zendesk | — | 0 | Quando arriva il mio ordine? |
| amz_B00CWXWXZ8_195425 | amazon_competitor | — | 0 | Nickel comme produit |
| amz_B0BS94NLY6_178448 | amazon_competitor | — | 0 | Flat not like picture flabby |
| zd_1134141_38979 | zendesk | 吸奶器 | 0 | Sent from my iPhone |
| tp_675758026e46b695eed6e566_98192 | trustpilot | — | 0 | Kein Komentar Kein Komentar |
| amz_B09XJJTJR9_172462 | amazon_competitor | — | 0 | My little one loves these. |
| amz_B0CBS1RSBD_171826 | amazon_competitor | — | 0 | Can see stright through these |
| amz_B0DK9Y8HQ4_195101 | amazon_competitor | — | 0 | Bought as a shower gift |
| amz_B0C5B7XTP8_195485 | amazon_competitor | — | 0 | Me gustaron bastante |
| amz_B08QMSG4YJ_189775 | amazon_competitor | — | 0 | One of my must haves. |
| amz_B0DXQ3MKQ2_195039 | amazon_competitor | — | 0 | Doesn't come with manual |
| amz_B0C1Y777FK_185095 | amazon_competitor | — | 0 | Me encantó súper recomendad |
| amz_B0CTQQ61D7_192628 | amazon_competitor | — | 0 | lower cut then shown |
| amz_B07S82FTD5_184495 | amazon_competitor | — | 0 | Muy buena calidad y cómoda 👌 |
| amz_B0C7T8LZDT_194864 | amazon_competitor | — | 0 | I plan on buyer g more. |
| amz_B00LZKC1S8_192245 | amazon_competitor | — | 0 | Baby loves it. |
| amz_B0C1SG13F2_194970 | amazon_competitor | — | 0 | Way to firm and awkward. |
| amz_B0BRQ5R2JM_195341 | amazon_competitor | — | 0 | kept coming unplugged! |
| amz_B07KF71M2X_171703 | amazon_competitor | — | 0 | Very nice stays up no rolling |
| tp_6563af89833b86a875deee39_94691 | trustpilot | — | 0 | Faire prices. Faire prices. |
| amz_B0DKPHQ5B5_194285 | amazon_competitor | — | 0 | She loves it |
| amz_B0FRS9BC2P_194224 | amazon_competitor | — | 0 | Good product |
| amz_B0D35KRVL8_185132 | amazon_competitor | — | 0 | Je regrette pas mon achat |
| amz_B0CPDXGW25_195140 | amazon_competitor | — | 0 | Excelente recomendado |
| zd_967277_24921 | zendesk | 吸奶器 | 0 | Conversation with aliyah k💋 |
| mz_R1VSBO8N6E44AY_17934 | momcozy | — | 0 | Very thin! Not my preference! |
| amz_B00TU03HU6_194942 | amazon_competitor | — | 0 | As advertised. No issues |
| amz_B0BHQM5WJR_194045 | amazon_competitor | — | 0 | Good support. |
| amz_B0CD9J5JL9_184971 | amazon_competitor | — | 0 | This item runs very small. |

### 4.2 `off_category` （product_line 空 且 零标签 且 无品牌 mentions）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| tp_6399a646d075435bd8db6c3f_10381 | trustpilot | — | 0 | Charged for nothing! The website couldn’t process my order as my card and delivery address are diffe |
| tp_680335085ff53402f8a79cdd_44228 | trustpilot | — | 0 | Bestellung wurde sehr schnell… Die Bestellung wurde sehr schnell bearbeitet und der Artikel wurde zü |
| tp_6546d90823f45d1a5e894dc7_66614 | trustpilot | — | 0 | search bar Why is there no search bar, it would have made the experience much easier and I would hav |
| tp_630e10a16a3e1ed2c3c4b2c5_84443 | trustpilot | — | 0 | Le site plante Le site plante, je n’arrive pas à aller dans les catégories le menu disparaît quand j |
| tp_61e17249a16c1e751f691b08_68727 | trustpilot | — | 0 | Die Bänder sind ganz gut. Die Bänder sind ganz gut. Leider nicht ganz so elastisch, wie man es von s |
| mz_R22XWR0D7X1R8X_15539 | momcozy | — | 0 | Why did you pick this product vs others?:OBSESSED is an understatement. It’s saving our backs big ti |
| zd_966249_42946 | zendesk | — | 0 | Hi    I made this order  JJ1399550 on 6th Jan but have still not received any part of it and the cov |
| amz_B0CTQQ61D7_160191 | amazon_competitor | — | 0 | I’m between S and M size, bought M because the reviews were saying it runs on a small side, don’t fe |
| tp_692058f55a55f2af632d8cc2_63018 | trustpilot | — | 0 | Ik heb een gepersonaliseerde badjas… Ik heb een gepersonaliseerde badjas besteld om cadeau te geven. |
| amz_B0BSSZW9H8_146695 | amazon_competitor | — | 0 | Nothing at all like the pictures. Very short in the torso, and does NOT hide everything like the pro |
| amz_B09FL7LV9W_136901 | amazon_competitor | — | 0 | My wife has loved it since I got it for her |
| tp_651d2993782780742be585c6_41021 | trustpilot | — | 0 | Richtig tolle Etiketten und Armbänder Richtig tolle Etiketten und Armbänder. Sicherlich kommt noch e |
| amz_B08GHZW3GL_154831 | amazon_competitor | — | 0 | My wife loved her gift she sleeps like a baby no tossing and turning on her side just |
| tp_64033f5a9b64b1bdaf67ef60_41209 | trustpilot | — | 0 | Très bonne expérience Très bonne expérience, vendeuse souriante, disponible, prévenante . Je me suis |
| amz_B0CLCG93B5_136719 | amazon_competitor | — | 0 | Granddaughter loves it! She think she has something she doesn't suppose to have😊 |
| mz_RTV3KMU60B8OU_18014 | momcozy | — | 0 | Non ne vale la pena. Scarso in qualità e pressoché inutile... |
| zd_1106078_5664 | zendesk | — | 0 | Bonjour, je n’ai toujours pas reçu ma commande passé le 3 mars cela est-il normal ?  J’aimerais savo |
| tp_640a3c74e27b60c7362ff721_54002 | trustpilot | — | 0 | Meine erste Bestellung bei Hans Natur - fabelhaft Die Beschreibung der angebotenen Produkte ist sehr |
| tp_61d5a783e1196e51d75a4c84_61832 | trustpilot | — | 0 | Just as the website states! Very simple/easy ordering process. Shipment was QUICK, which was extreme |
| tp_609a2d9df9f4870b48c45d2a_14344 | trustpilot | — | 0 | Schon die 2. Bestellung. Tolle Brotdosen! Wir haben nun schon zum zweiten Mal bei Gutmarkiert bestel |
| tp_694c66cfaa453dbf083db62b_54105 | trustpilot | — | 0 | Ich bestelle jedes Jahr einen Kalender… Ich bestelle jedes Jahr einen Kalender bei euch und bin einf |
| tp_657b1518db234014765f6f7e_96707 | trustpilot | — | 0 | Einfache Benutzerführung Einfache Benutzerführung |
| tp_68a04d8ec823ae5686a9b310_79213 | trustpilot | — | 0 | Sehr schöner BH Sehr schöner BH. Tolles Material. Perfekte Passform. |
| amz_B0BMFQFZSH_54501 | amazon_competitor | — | 0 | This is the worst product of any kind that I have ever used. None of the locking mechanisms would en |
| amz_B08YWHN79N_72702 | amazon_competitor | — | 0 | I was very hopeful after reading the reviews but I guess I  know why the price is so low. 3 assembly |
| tp_65833fc42ffce8b320c0a29e_65415 | trustpilot | — | 0 | Fast and easy service! Fast and easy service! |
| tp_693d653b3abe4c4e57f8642c_72603 | trustpilot | — | 0 | Les photos sont magnifiques Les photos sont magnifiques Petit bémol je n’avais pas vu qu’il fallait  |
| tp_67a9fcaf8bd7538700a9d67d_82319 | trustpilot | — | 0 | Die Lieferung kam wie beschrieben an Die Lieferung kam wie beschrieben an. Es gab kurze Verzögerunge |
| amz_B0DKFG7LSY_154151 | amazon_competitor | — | 0 | Fit is good ,  Comes with extra hook band. |
| zd_1079728_2201 | zendesk | — | 0 | Conversation with Web User 69abde9771c0e9847a1b90d5 |
| amz_B07WDHNKKB_160249 | amazon_competitor | — | 0 | N'est pas tout à fait idéal pour les redressements assis les pieds trop rapprochés pour le soutien.  |
| zd_921965_40063 | zendesk | — | 0 | Please cancel this order I want to order the air 1 instead   Korie |
| tp_6821e98349bd5829e7a19fc0_81390 | trustpilot | — | 0 | Meilleure marque d'allaitement EVER Une marque d'allaitement avec des vêtements splendides. C'est si |
| zd_1103224_42481 | zendesk | — | 0 | Hi,  I'm looking to find out when my order will be dispatched, I ordered this 10 days ago now and st |
| amz_B0CND44J4C_166858 | amazon_competitor | — | 0 | Engaging and fun way to strengthen the pelic floor muscles |
| zd_938131_45746 | zendesk | — | 0 | Bonjour, J'aimerais savoir s'il est possible d'acheter des pièces de rechange pour le ventilateur mo |
| tp_65db3806bc7838f2382536cf_91324 | trustpilot | — | 0 | Correspond parfaitement à mes attentes Correspond parfaitement à mes attentes |
| amz_B07MS8QP2J_122797 | amazon_competitor | — | 0 | I didn’t know that I had ordered enough for an entire daycare center.  Packaging very inconvenient a |
| amz_B09QY2H8SG_147512 | amazon_competitor | — | 0 | Won’t connect to my Wi-Fi I am not able to use it at all |
| tp_5f3062b01a5a6902684659eb_49991 | trustpilot | — | 0 | Excellent Service rapide et efficace. Prix ultra compétitifs. |
| tp_60b3df0cf9f4870b7012a76b_56291 | trustpilot | — | 0 | Reibungslos und schnell Wir haben zum ersten Mal hier bestellt und es lief alles reibungslos. Der Ve |
| amz_B08QR6V8WR_144923 | amazon_competitor | — | 0 | Los use para mi hija y son lo máximo y los compro para regalo porque se que lo van  a usar en los be |
| tp_68640b009be574a582e61033_74951 | trustpilot | — | 0 | Très contente de l'achat des tututes… Très contente de l'achat des tututes pour mon fils elles sont  |
| amz_B09C8N75VM_183501 | amazon_competitor | — | 0 | La rebaja la grasa y sudor del cuero |
| amz_B096LRSJCS_146820 | amazon_competitor | — | 0 | Excelente el ajuste,cómodo y buen material. Ahora solo sirve para lactar o no para extraer con la má |
| tp_694416712fb07fa6450afba3_97250 | trustpilot | — | 0 | Très bonne expérience Très bonne expérience |
| zd_1038793_32335 | zendesk | — | 0 | Conversation with Web User 69970047ab4511aab96b05bb |
| tp_6925f486732786bbc3cad091_66764 | trustpilot | — | 0 | Die Bestellung und Lieferung war… Die Bestellung und Lieferung war einwandfrei! |
| tp_5ffde483755dc1032c13fc63_64900 | trustpilot | — | 0 | Bestpreis Bestpreis für Lego-Displaybox, problemloser und einfacher Einkauf! Bin begeistert! |
| tp_68b474d0a846c794808164bd_72272 | trustpilot | — | 0 | Très belle emballage soigné et envoi… Très belle emballage soigné et envoi rapide. Les produits sont |
| tp_66a3bf747a27e04d5ea976fe_88597 | trustpilot | — | 0 | Articles de qualité qui correspondent… Articles de qualité qui correspondent bien à mes attentes |
| tp_6761f9050e5b0db7c7d5602c_43468 | trustpilot | — | 0 | God kvalitet, lagom starka magneter Jag har använt mitt armband i några veckor nu, och varit väldigt |
| amz_B07S736G1H_183918 | amazon_competitor | — | 0 | Linda y con espectacular compresión |
| amz_B07ZLSQXCS_154254 | amazon_competitor | — | 0 | I like how it looks specially with jeans. I will buy more colors. |
| amz_B07Q3XFZC8_148029 | amazon_competitor | — | 0 | There is no specific "tummy control". The entire cami is constricting, so it also flattens your brea |
| zd_1125749_6213 | zendesk | — | 0 | Hallo . Ich warte schon seit einigen Tagen auf meine Bestellung mit der Bestellnummer JJ1557854. Kön |
| amz_B0DRCXZM4F_194766 | amazon_competitor | — | 0 | La mejor faja q e probado y q vale la pena comprar sin mirar su precio la verdad me encantó |
| amz_B0CKYFBH55_117905 | amazon_competitor | — | 0 | Ease of use warmth is a plus it does what you set it on it makes life easier for me also value for m |
| amz_B07QPD1PLD_174614 | amazon_competitor | — | 0 | Tela suave se ajusta cómodamente al cuerpo. Ideal para usar con pantalones blancos o vestidos porque |
| amz_B0FM7DSD4Z_161417 | amazon_competitor | — | 0 | Très bon babyphone, l’image est ultra propre avec la caméra 2K et l’écran partagé permet de surveill |
| zd_934056_28751 | zendesk | — | 0 | Good day,  I want to enquiry about my order that I made on the 19th of December. I haven't received  |
| amz_B09HJDX5S6_66923 | amazon_competitor | — | 0 | This product serves its function and has saved up a ton of time. Only CON: Although there are multip |
| zd_1024657_32370 | zendesk | — | 0 | Conversation with Web User 698f6962e2eb8e850ee8454f |
| zd_1123419_33089 | zendesk | — | 0 | Conversation with Web User 69c0fddfdee1314b55a4427c |
| tp_69bc0dd57c81d39267b3c0c8_57951 | trustpilot | — | 0 | Aquawasserbahn Die Auswahl und der Bestellvorgang ist sehr kundenfreundlich. Die Bestellung wurde ze |
| tp_63d6a06e9b64b1bdaf468f9e_64662 | trustpilot | — | 0 | gute Darstellung der Ware gute Darstellung der Ware unkomplizierte Bestellung und Registrierung |
| zd_1088605_41408 | zendesk | — | 0 | Conversation with Web User 69b04261de1c608150030de8 |
| tp_661d201477db8bce2528b7e3_75897 | trustpilot | — | 0 | J'ai pris un abonnement annuel que j'ai… J'ai pris un abonnement annuel que j'ai payé 30€ et comme j |
| tp_5deb6c95c8454506301abfff_24686 | trustpilot | — | 0 | Nachdem ich nach meiner Bestellung hier… Nachdem ich nach meiner Bestellung hier die Bewertungen gel |
| tp_5e578d353c93ae0864a8c353_53068 | trustpilot | — | 0 | Alles wunderbar, danke! Alles hat wunderbar geklappt, danke! Die Karten sehen aus wie am Bildschirm  |
| amz_B0CC5BYVDL_167436 | amazon_competitor | — | 0 | La tela es muy suave, son cómodas y ligeras y amoldan excelente, de hecho no necesito usar sostén co |
| tp_5c8ba44197afa10a488bf36d_65153 | trustpilot | — | 0 | Hervorragend von A bis Z Informationen zu Produkten bei Rückfragen schnell und umfassend; Lieferung  |
| amz_B00EANJZEK_143978 | amazon_competitor | — | 0 | Easy assembly for our new grandson.  Will convert as he gets older to a toddler bed. |
| tp_6625301641e627ea28119483_35747 | trustpilot | — | 0 | Goed product, snelle levering Duidelijke bestel informatie, goed kwaliteit materiaal, snelle leverin |
| amz_B07FRYYZ3N_161534 | amazon_competitor | — | 0 | Muy suaves y perfectas para la piel delicada del bebéEstas toallitas húmedas Huggies Natural Care ha |
| tp_69b6ebb516605b75de811397_72927 | trustpilot | — | 0 | My problem was handled professionally… My problem was handled professionally and expeditiously and c |
| tp_6855882f0a64136ecdae4783_95335 | trustpilot | — | 0 | Schnell und unkompliziert Schnell und unkompliziert |
| amz_B0DPQ44BJL_194254 | amazon_competitor | — | 0 | My Pulmonary doctor and the FDA do NOT recommend using this for sanitizing my C -pap equipment. |
| amz_B0DR1ZZH49_135024 | amazon_competitor | — | 0 | Honestly the first item I have purchased on AMAZON that exceeded the product specifications.  I am l |
| tp_68f49f04a580af8a7bd44ad8_90198 | trustpilot | — | 0 | Cadeau de naissance bien arrivé dans… Cadeau de naissance bien arrivé dans les délais |
| tp_689758a3ee898ecf4ea1b5ea_75966 | trustpilot | — | 0 | Je suis ravie de mes vêtements Daronnes… Je suis ravie de mes vêtements Daronnes et aussi des chouch |
| tp_5f931a28798e6f04a41be1f6_72529 | trustpilot | — | 0 | Tres decevant Apres des heures de travail passees a choisir notre faire part et apres avoir commandé |
| tp_673235aa92e5a5ecacafe290_89887 | trustpilot | — | 0 | It delivered fast emailed during the… It delivered fast emailed during the shortage |
| tp_662a83d54f2ae05dea180ce9_96692 | trustpilot | — | 0 | Geht schnell und einfach Geht schnell und einfach |
| amz_B09TR3XX3V_173702 | amazon_competitor | — | 0 | My newborn needed these for the Fourth of July I got them just in time when my kid of a husband lit  |
| mz_R731WVGUBVSYB_17843 | momcozy | — | 0 | Tout est bien pensé pour les nouvelles mamans souhaitant se préparer pour l’hôpital. |
| tp_6176859e234f1b1fe563ee38_58091 | trustpilot | — | 0 | Bin zufrieden Bin zufrieden, hält am besten auf glatten Flächen. Ein Stern Abzug gibt es weil manchm |
| amz_B0BFKNQS6S_165138 | amazon_competitor | — | 0 | It looks very small but stretches to fit. |
| tp_5f88bcf8798e6f0bcc46b375_66853 | trustpilot | — | 0 | Schnelle und unkomplizierte Bestellung Schnelle und unkomplizierte Bestellung |
| tp_5cbe0f07a8436908c40d5645_89522 | trustpilot | — | 0 | Ottimo ed economico Prodotto come da descrizione. Ottimo servizio. Comunicazione perfetta. |
| tp_67cfe8106a149284f9d6397d_67519 | trustpilot | — | 0 | I haven't got my package yet I haven't got my package yet. How much longer should I wait |
| zd_1139984_2107 | zendesk | — | 0 | Conversation with Web User 69c9810c26a76090644987f5 |
| zd_1130786_5593 | zendesk | — | 0 | Conversation with Web User 69c412102923e1f5f6b1ae18 |
| tp_68d3a709318bb2dc388ba0db_41395 | trustpilot | — | 0 | pas de sav J’ai acheté une caméra de la marque Béaba et elle ne fonctionne plus malgré après un an d |
| tp_639cd3a4d075435bd8de8d9d_91021 | trustpilot | — | 0 | Tolle Qualität Schnelle unkomplizierte Lieferung, Ware in toller Bio Qualität |
| tp_66f2a2f91357856fd6c3060b_81735 | trustpilot | — | 0 | Zitterpartie,lange Lieferzeit,Termin… Zitterpartie,lange Lieferzeit,Termin wäre zu spät gewesen, auf |
| tp_5bc6e43d9d378004f01ec907_72021 | trustpilot | — | 0 | Impeccable service re iron Had an issue with iron. Sent an email to main address on google. Response |
| tp_68f1192df6720747c3c91283_82465 | trustpilot | — | 0 | Marque d'allaitement chouchoute Marque d'allaitement chouchoute , avec un vrai état d'esprit famille |
| zd_1072176_42324 | zendesk | — | 0 | Hello, I received notification that my order was delivered on Monday. However, I have not received i |
| tp_5e175b2fc845450914b49797_40149 | trustpilot | — | 0 | Listen to all these reviews Listen to all these reviews, they echo my experience. Wish I had checked |

### 4.3 `generic_only` （零标签 且 去停用词后有效 token < 3）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| amz_B08QW89LFS_110886 | amazon_competitor | 喂养电器 | 0 | The only pacifier both my sons loved and used! |
| amz_B08352W1T8_117191 | amazon_competitor | 家居家纺 | 0 | I absolutely love my pillow!¡! |
| amz_B00PH7DSOQ_120091 | amazon_competitor | 智能母婴电器 | 0 | Used it for my massage clients, and they loved it! |
| amz_B0B7MFS2W4_132743 | amazon_competitor | 家居家纺 | 0 | It’s the best stroller I really like it |
| amz_B09FPLGYF2_134653 | amazon_competitor | 智能母婴电器 | 0 | It fits everywherePracticalLightweightComfortable |
| amz_B095W2VKYW_135754 | amazon_competitor | 家居家纺 | 0 | In love with my maternity pillow. Will recommend |
| zd_1121578_36556 | zendesk | 吸奶器 | 0 | Conversation with Patrick1010!! |
| zd_943560_21599 | zendesk | 吸奶器 | 0 | 2026/11/14にAMAZONで購入しました。 注文番号: 503-7687813-4250268  1回使用しましたが、2回目からモニターのボタンが正常に作動せず、電動部分の隙間から母乳が |
| amz_B0DJVFX2KY_141278 | amazon_competitor | 智能母婴电器 | 0 | This humidifier its really good. I recommend as very good one. |
| zd_927691_27105 | zendesk | 吸奶器 | 0 | Conversation with AmandaRoeske |
| amz_B0057R47NW_151516 | amazon_competitor | 内衣服饰 | 0 | Product came dirty and used. Would like a replacement. |
| amz_B0BZJ82N7W_151987 | amazon_competitor | 家居家纺 | 0 | I like them but they don’t have pockets as described |
| amz_B0CR35MZ98_152088 | amazon_competitor | 内衣服饰 | 0 | Best bra from all of those that i have |
| zd_1135887_28589 | zendesk | 吸奶器 | 0 | Can I cancel my order thank you |
| amz_B089QJ6X3X_154166 | amazon_competitor | 家居家纺 | 0 | We love our wagon! It definitely gets used! |
| amz_B07NNVPMFG_154468 | amazon_competitor | 内衣服饰 | 0 | I liked the shapewear, wears very nice! 👍 |
| amz_B00AWMP880_156620 | amazon_competitor | 内衣服饰 | 0 | Loved it as soon as it was unwrapped |
| amz_B00A282SGO_157836 | amazon_competitor | 喂养电器 | 0 | These are the only pacifiers I can get my baby to use. |
| amz_B01AUQD9LU_159462 | amazon_competitor | 喂养电器 | 0 | Only pacifiers my two youngest would use. |
| mz_R17JX8BPZ2USQ2_19163 | momcozy | 吸奶器 | 0 | This did not come with the duck valves. |
| zd_1072141_27088 | zendesk | 吸奶器 | 0 | Conversation with MélodieSabra |
| zd_1137196_27137 | zendesk | 吸奶器 | 0 | الصفحه دي بتقول انها توكيل ممكوزي في مصر ؟؟  هل ده حقيقي ؟؟ |
| amz_B0DXTH8BW7_162502 | amazon_competitor | 喂养电器 | 0 | Great wipe warmer! 10/10 recommend |
| amz_B01MYGNGKK_164187 | amazon_competitor | 智能母婴电器 | 0 | I really like the functions of this humidifier. |
| amz_B08YKVJTDH_164644 | amazon_competitor | 母婴综合护理 | 0 | They are very good, more wet than other wipes |
| mz_R2CLIRR23JHTF4_19420 | momcozy | 吸奶器 | 0 | 手ざわりのよいのハードケースは中の搾乳機をしっかりと保護、専用バッグなのでぴったりと収まります。開閉はスムーズでカーブ部分もストレスなくファスナーが滑ります。トレイや仕切りが設けられているのでしっくり |
| mz_R1OGOFWR2JYVKQ_19424 | momcozy | 吸奶器 | 0 | 旅行、帰省などで搾乳器の持ち運びとして活用するために注文しました。機械が守れるので故障する心配なく助かります。ただちょっと大きすぎますね。。もう少しコンパクトだと嬉しいです。ですが他のパーツや充電器も |
| zd_983755_24617 | zendesk | 吸奶器 | 0 | Conversation with KaiaDeLaRosa |
| zd_1026539_21502 | zendesk | 吸奶器 | 0 | Conversation with N Λ T Λ L ⋮ Λ |
| zd_1122199_27072 | zendesk | 吸奶器 | 0 | Conversation with RaeganMueller |
| zd_1008659_38965 | zendesk | 吸奶器 | 0 | Conversation with MustafAsiye🤍 |
| amz_B08JYJKZNL_170265 | amazon_competitor | 家居家纺 | 0 | Good product, everything as described 👍 |
| amz_B0DTHSRTD4_170340 | amazon_competitor | 智能母婴电器 | 0 | Very good purifier for a 10x12 room. |
| amz_B09XJJFL4P_170823 | amazon_competitor | 喂养电器 | 0 | This is the only pacifier he loves |
| mz_R21C9UMPTDWO7X_19016 | momcozy | 喂养电器 | 0 | Every parents should have this |
| zd_930820_27093 | zendesk | 吸奶器 | 0 | Conversation with Ayşe Kocaman |
| zd_1129054_28672 | zendesk | 吸奶器 | 0 | Conversation with CassidyLasick |
| amz_B0CLWXDQ2G_171626 | amazon_competitor | 家居家纺 | 0 | My granddaughter loved the crib. |
| amz_B08PMJBN33_171660 | amazon_competitor | 家居家纺 | 0 | My granddaughter loved the crib |
| amz_B09C8MNW9S_173299 | amazon_competitor | 内衣服饰 | 0 | This lacked compression.  I will not re purchase. |
| zd_986769_28533 | zendesk | 吸奶器 | 0 | Conversation with VictoriaCarman |
| zd_973345_27075 | zendesk | 吸奶器 | 0 | Conversation with IngzelleFergie |
| amz_B0B6Q7RZ9N_175861 | amazon_competitor | 家居家纺 | 0 | My daughter did not like this stroller at all |
| zd_933297_28531 | zendesk | 吸奶器 | 0 | Conversation with KylieMccomber |
| mz_R3TZWUG1YN7VMQ_18855 | momcozy | 智能母婴电器 | 0 | Massager used when you've got pain |
| amz_B0995JN3NJ_176752 | amazon_competitor | 家居家纺 | 0 | This product is nothing like described |
| amz_B07WPC88Y9_176965 | amazon_competitor | 吸奶器 | 0 | This was really helpful when I was pumping! |
| zd_992356_27079 | zendesk | 吸奶器 | 0 | Conversation with NelidaGonzalez |
| amz_B077GJLGC4_178217 | amazon_competitor | 母婴综合护理 | 0 | these are the only wipes we use! |
| amz_B07FT2H5J8_179267 | amazon_competitor | 内衣服饰 | 0 | I will not be wearing it as it really does nothing. |
| amz_B09K39Y11L_179803 | amazon_competitor | 家居家纺 | 0 | She liked the pillow. Now to get her pregnant... |
| zd_1039366_27076 | zendesk | 吸奶器 | 0 | Conversation with JessicaNalerio |
| amz_B0BL8M2Z1Q_181970 | amazon_competitor | 母婴综合护理 | 0 | It’s a wipe, it did what it was supposed to do. |
| amz_B09MCK1HQN_182195 | amazon_competitor | 家居家纺 | 0 | This pillow came with bedbugs. DO NOT PURCHASE! |
| amz_B09FPLGYF2_182390 | amazon_competitor | 家居家纺 | 0 | It’s not as small as they describe it to be. |
| mz_R15HG3K8WN5UKE_19478 | momcozy | 吸奶器 | 0 | 専用ケースで、授乳部品の隙間が無く丁度よく使えました。出先など、ほこりの心配も少ない点が良かったです。引き続き耐久性を確認していきます。 |
| amz_B0FMDMHD5G_183003 | amazon_competitor | 内衣服饰 | 0 | Very tight and definitely as shapewear |
| amz_B00P7FJM48_183185 | amazon_competitor | 吸奶器 | 0 | This is a good pumping bra! !!!! |
| mz_R12EY0V8BG9NLA_19621 | momcozy | 吸奶器 | 0 | 純正品なのでサイズピッタリでした。なんの問題も無く使用出来ています。持ち運びには必須アイテムです |
| mz_RGYGQ2IDL95PL_19439 | momcozy | 吸奶器 | 0 | 育児バッグの中でもかなり優秀な部類でした。まず、ハードシェルなので中身をしっかり守ってくれる安心感があり、搾乳機を2台入れても余裕のある設計。取り外し可能なトレイが想像以上に便利で、整理整頓がラクにで |
| amz_B08DYTD65X_183664 | amazon_competitor | 家居家纺 | 0 | These pacifiers are as described! |
| amz_B0D9M88VTX_183731 | amazon_competitor | 家居家纺 | 0 | It's exactly how it was described. |
| zd_1136437_21504 | zendesk | 吸奶器 | 0 | Conversation with JennanGatewood |
| amz_B07Y3Z86DB_183901 | amazon_competitor | 内衣服饰 | 0 | More compression would be better. |
| zd_927745_38952 | zendesk | 吸奶器 | 0 | Conversation with BrookeMcClain |
| amz_B0CR133K23_184313 | amazon_competitor | 家居家纺 | 0 | It’s not as fluffy as described |
| mz_R12WAJ0TO461RW_18269 | momcozy | 吸奶器 | 0 | Exactly what I was looking for |
| amz_B095HLPGTV_185991 | amazon_competitor | 吸奶器 | 0 | Did not work for pumping at all. |
| amz_B0BQ6X1CCS_192845 | amazon_competitor | 内衣服饰 | 0 | It’s nice but my daughter don’t wears it |
| mz_R2YHYZRE1UU67D_19392 | momcozy | 吸奶器 | 0 | 約9000円と高いだけあり、造りとしては非常にしっかりしています。搾乳機が2つ入る構造をしていて、取っ手を持つことで楽に携帯可能なデザイン。またその際には製品ロゴが見えるようになっています。キャリーケ |
| tp_682772be79f9307e628e8cca_72195 | trustpilot | 智能母婴电器 | 0 | Really good product Love how light the stroller is |
| rd_R3GFY0R4DMBPQN_945 | reddit | 内衣服饰 | 0 | Just like the other brands I would definitely not recommend |
| zd_969273_21598 | zendesk | 吸奶器 | 0 | もう片方と部品を一つずつ変えてみたり、違うサイズのニップルで試してみても吸引せず  モーターを変えたら吸引できたので、片方のモーターの不具合だと思います。  同一商品との交換をお願いします。 色が他の |
| zd_1004592_1214 | zendesk | 吸奶器 | 0 | Conversation with RosalindaOlivares |
| zd_949266_1218 | zendesk | 吸奶器 | 0 | Conversation with MelissaRinehart |
| zd_977241_36557 | zendesk | 吸奶器 | 0 | Conversation with Patrick1010!! |
| zd_1086926_1224 | zendesk | 吸奶器 | 0 | Conversation with KaylahBarriger |
| zd_918399_27059 | zendesk | 吸奶器 | 0 | Conversation with CatherineStewart |
| zd_1054162_1228 | zendesk | 吸奶器 | 0 | Conversation with JasmineHughes |
| zd_1119520_24592 | zendesk | 吸奶器 | 0 | Why is the item not shipped yet? |
| zd_991647_24597 | zendesk | 吸奶器 | 0 | My order number is .   JJ1443656 |
| zd_1040055_27061 | zendesk | 吸奶器 | 0 | Conversation with AllyFriedenberger |
| zd_1121465_27112 | zendesk | 吸奶器 | 0 | Conversation with BrigitteDery |
| zd_1080110_28525 | zendesk | 吸奶器 | 0 | Conversation with BriannaMitchell |
| zd_927521_39601 | zendesk | 吸奶器 | 0 | JJ1385202 Has this order being sent yet? |
| zd_931743_10867 | zendesk | 喂养电器 | 0 | I was wondering when will my order be shipped?  #JJ1393147 |
| zd_946595_24621 | zendesk | 吸奶器 | 0 | Conversation with Monica Muñiz |
| zd_1081322_12538 | zendesk | 喂养电器 | 0 | Conversation with NataliaMadrigal |
| zd_1067569_12544 | zendesk | 喂养电器 | 0 | Conversation with CadieCampbell |
| zd_1067762_12550 | zendesk | 喂养电器 | 0 | Conversation with Sinderella 🤍 |
| zd_1132267_14373 | zendesk | 喂养电器 | 0 | Conversation with ChanellCericeee |
| zd_1023795_15566 | zendesk | 智能母婴电器 | 0 | Conversation with @karenagred ❤ |
| zd_1087047_16544 | zendesk | 内衣服饰 | 0 | Conversation with brandihutchings |
| zd_1138847_16550 | zendesk | 内衣服饰 | 0 | Conversation with JewelOlbrantz |
| zd_985678_19447 | zendesk | 吸奶器 | 0 | Hi my order number is JJ1434973 , when I get delivery? |
| zd_1086717_20529 | zendesk | 吸奶器 | 0 | Hi I have a problem with my pumps |
| zd_942991_21457 | zendesk | 吸奶器 | 0 | Conversation with Sophieeeee😇🤪 |
| zd_1018395_21485 | zendesk | 吸奶器 | 0 | Conversation with George...😎... 🌊 |
| zd_1081770_21494 | zendesk | 吸奶器 | 0 | Conversation with melaniefouquet98 |
| zd_1093158_21501 | zendesk | 吸奶器 | 0 | Conversation with د. فهد الشرهان |

## 五、阈值判定（决策 4）

| 指标 | 阈值 | 实测 | 结果 |
|---|---:|---:|:---:|
| 原始覆盖率 | 88.00% | 76.11% | 🔴 FAIL |
| 业务有效覆盖率 | 94.00% | 89.48% | 🔴 FAIL |

## 六、与 Phase 4 对比

- Phase 4 5K 子集原始覆盖率：82.58%
- Phase 5 全量原始覆盖率：76.11% （-6.47pp）
- Phase 5 全量业务有效覆盖率：89.48%
