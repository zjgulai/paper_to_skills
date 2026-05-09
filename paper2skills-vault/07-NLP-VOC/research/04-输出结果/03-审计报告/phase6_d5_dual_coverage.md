---
name: phase5-d10-dual-coverage-report
description: Phase 5 D10 双覆盖率审计报告（原始 / 广义 / 业务有效）。当审计 Phase 5 整体覆盖率达成、与 Phase 4 对比、查看排除分布与抽样时使用。
computed_at: 2026-05-09T20:17:05
doc_type: audit-report
module: voc-nlp
status: stable
---

# Phase 5 D10 双覆盖率审计报告

**输入**：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d5_final.jsonl`
**总样本**：364,569
**计算时间**：2026-05-09T20:17:05

## 一、核心指标

| 指标 | 分子 | 分母 | 覆盖率 | Phase 4 基线 | Δ |
|---|---:|---:|---:|---:|---:|
| 原始覆盖率 | 325,811 | 364,569 | **89.37%** | 82.58% | +6.79pp |
| 广义覆盖率（含品牌命中）| 325,814 | 364,569 | **89.37%** | — | — |
| 业务有效覆盖率 | 325,811 | 338,946 | **96.12%** | — | — |

> 业务有效分母 = 总数 - 排除（25,623 条，占比 7.03%）

## 二、排除桶分布

| 桶 | 定义 | 样本数 | 占总比 |
|---|---|---:|---:|
| `too_short` | text.strip() < 30 字符 | 2,611 | 0.72% |
| `off_category` | product_line 空 且 零标签 且 无品牌 mentions | 22,890 | 6.28% |
| `generic_only` | 零标签 且 去停用词后有效 token < 3 | 122 | 0.03% |

## 三、按数据源拆解

| data_source | 总数 | with_label | 原始覆盖率 | brand_only | 排除合计 | 业务有效覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| amazon_competitor | 194,734 | 185,522 | 95.27% | 2 | 6,694 | 98.66% |
| trustpilot | 99,853 | 88,801 | 88.93% | 0 | 9,944 | 98.77% |
| zendesk | 47,204 | 33,803 | 71.61% | 0 | 6,892 | 83.85% |
| momcozy | 19,808 | 15,609 | 78.80% | 1 | 1,591 | 85.68% |
| reddit | 2,970 | 2,076 | 69.90% | 0 | 502 | 84.12% |

## 四、排除追踪抽样（每桶最多 100 条，供人工 spot check）

### 4.1 `too_short` （text.strip() < 30 字符）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| tp_647afb12706f837cb1f70d40_93889 | trustpilot | — | 0 | ottimo sono molto soddisfatta |
| zd_946172_34453 | zendesk | — | 0 | Conversation with Ruba Anis |
| mz_RBSW5IF7B4A75_19792 | momcozy | — | 0 | Si le gustaron a mi bb |
| amz_B00LZKC1S8_184606 | amazon_competitor | — | 0 | Bought for a gift - adorable! |
| amz_B0CYZPJMJR_190929 | amazon_competitor | — | 0 | Really good product👍 |
| amz_B0C9HBKQ5D_189649 | amazon_competitor | — | 0 | Humidifies my room fast! |
| amz_B0CSM8S3RS_191740 | amazon_competitor | — | 0 | Quality set. |
| amz_B0DHDBK5BD_189666 | amazon_competitor | — | 0 | Hoping it ease my pain. |
| amz_B0CCKSWX3V_192124 | amazon_competitor | — | 0 | My baby loves it! |
| amz_B0CYPL1HSX_192481 | amazon_competitor | — | 0 | Color & fit just fine. |
| amz_B0D7LWD6LS_190548 | amazon_competitor | — | 0 | It good for her bottle's |
| zd_1047860_47543 | zendesk | — | 0 | Conversation with Minahil |
| amz_B098F8GMM1_192125 | amazon_competitor | — | 0 | Baby loved these! |
| tp_61e812a9a16c1e751f6dfaa3_99829 | trustpilot | — | 0 | Perfekt Perfekt, danke |
| amz_B07R1YM6QF_193578 | amazon_competitor | — | 0 | Nice shower gift |
| mz_R39HYHNQ7EXSLA_19605 | momcozy | 母婴综合护理 | 0 | Softest diapers ! |
| amz_B07Q3YF34Z_193759 | amazon_competitor | — | 0 | Hard to put on! |
| tp_640076c19b64b1bdaf6678e3_99838 | trustpilot | — | 0 | Parfait 👍🏼 Parfait 👍🏼 |
| amz_B0DD3R1X5Y_194125 | amazon_competitor | — | 0 | good purchase |
| tp_64f171b51b23248e1dbc1b7a_98060 | trustpilot | — | 0 | Tolle Qualität Tolle Qualität |
| amz_B0CY2BSHTM_192645 | amazon_competitor | — | 0 | Nunca me calento |
| amz_B09Q86424N_193494 | amazon_competitor | — | 0 | Very good product |
| amz_B09MWJB3LF_184448 | amazon_competitor | — | 0 | Para los más pequeños de cada |
| amz_B00FNZQHJA_195088 | amazon_competitor | — | 0 | La mejor sin duda alguna |
| amz_B0DPSJYSGF_178817 | amazon_competitor | — | 0 | A bit tight but they'll do |
| mz_R1WYACN9Z29ZL4_19245 | momcozy | — | 0 | Never worked from day one. |
| zd_1084865_1240 | zendesk | 吸奶器 | 0 | Conversation with AydenMalloy |
| zd_1098615_38966 | zendesk | 吸奶器 | 0 | Conversation with Alice A. |
| zd_951591_12549 | zendesk | 喂养电器 | 0 | Conversation with Faith Wade |
| zd_1035662_4718 | zendesk | — | 0 | Conversation with Chloé 🌺 |
| amz_B0DJ1YNC1H_193773 | amazon_competitor | — | 0 | Very satisfied! |
| amz_B0CBN4Z6FH_185249 | amazon_competitor | — | 0 | Excited with the product. |
| amz_B08YF5KTSY_172539 | amazon_competitor | — | 0 | Helping good every position |
| zd_1127630_16553 | zendesk | — | 0 | Conversation with Suman Bains |
| zd_1083665_28553 | zendesk | 吸奶器 | 0 | Conversation with earnnie |
| amz_B0C23TCPK7_184996 | amazon_competitor | — | 0 | Crystal clear instructions. |
| amz_B091F8PF5Q_190594 | amazon_competitor | — | 0 | A good gift to purchase |
| amz_B0CKMWWYJT_195526 | amazon_competitor | — | 0 | Runs very very small |
| tp_5b3bf3b66d33bc0c94b0a018_99837 | trustpilot | — | 0 | Perfekt Perfekt, danke |
| amz_B0DHV475TK_193721 | amazon_competitor | — | 0 | Very very nice |
| mz_R10QHKGVF7AASM_19758 | momcozy | — | 0 | Bay loves! |
| amz_B0DMNL5C2W_158164 | amazon_competitor | 家居家纺 | 0 | So in love with this carrier! |
| amz_B0DC5GGJRP_172307 | amazon_competitor | — | 0 | Nice! Just what we needed! |
| tp_60560518f85d750bf4fa1d44_99821 | trustpilot | — | 0 | 4D scan Very welcoming |
| amz_B07DWLTGSF_194954 | amazon_competitor | — | 0 | Hermosa, le doy in 10/10 |
| zd_1074241_21552 | zendesk | 吸奶器 | 0 | ﻿ Von meinem iPhone gesendet |
| amz_B07YTFZB26_195471 | amazon_competitor | — | 0 | Make your body right |
| tp_680f5763e0ea666e6c534abd_99465 | trustpilot | — | 0 | ❤️❤️❤️❤️ ❤️❤️❤️❤️❤️❤️❤️❤️❤️❤️ |
| tp_662beeb0c40ad53b1ed365ac_83027 | trustpilot | — | 0 | Best formula! Best formula! |
| zd_950320_21594 | zendesk | 吸奶器 | 0 | Conversation with Simran🌼 |
| zd_1058049_11853 | zendesk | — | 0 | Conversation with Jessica |
| amz_B0CMJ1GKXM_184982 | amazon_competitor | — | 0 | Este producto es muy útil. |
| amz_B09XJKHCXZ_172342 | amazon_competitor | — | 0 | Good for gifts to new moms |
| zd_1027984_33210 | zendesk | — | 0 | Conversation with Victoria W |
| amz_B08F99YLND_184625 | amazon_competitor | — | 0 | Fácil de montar y muy cómodo |
| amz_B07FRYYZ3N_192021 | amazon_competitor | — | 0 | Nice for a baby shower |
| amz_B0058O755G_171964 | amazon_competitor | — | 0 | Best for babies and toddlers! |
| tp_6328bc856a3e1ed2c3d70f69_82633 | trustpilot | — | 0 | Quick and easy Quick and easy |
| zd_974619_29522 | zendesk | — | 0 | Conversation with Gaurav Dang |
| amz_B00EVIS0CW_159304 | amazon_competitor | — | 0 | Standard item & good to use |
| amz_B0BWSB23HC_184604 | amazon_competitor | — | 0 | Excelente lo que yo esperaba |
| amz_B07YTFZB26_192553 | amazon_competitor | — | 0 | It keeps me snatch |
| amz_B07P7DZTYR_171730 | amazon_competitor | 家居家纺 | 0 | I’m in love with this pillow |
| amz_B0CG5D62DL_184740 | amazon_competitor | — | 0 | Muy fácil de poner y suave … |
| amz_B09LVPQ4XF_194364 | amazon_competitor | — | 0 | Good product |
| tp_67a9f549125b24fdc1a4c60b_99651 | trustpilot | — | 0 | good good product |
| zd_1080574_44654 | zendesk | — | 0 | Conversation with Dhruv 🙂‍↔️ |
| amz_B08JYJKZNL_172190 | amazon_competitor | 家居家纺 | 0 | Good product as described. |
| amz_B0BQL6CB5B_193965 | amazon_competitor | — | 0 | It works good |
| mz_R3VOKDZ4AL6WS7_14617 | momcozy | 吸奶器 | 0 | Happy I could find this size |
| amz_B07RK763T2_194915 | amazon_competitor | — | 0 | Me gusto mucho , comoda |
| mz_R39Z9JFU357F2O_18628 | momcozy | 喂养电器 | 0 | My baby loves her teether🥰 |
| amz_B07XMJ6547_184822 | amazon_competitor | 家居家纺 | 0 | It's a stroller and carrier |
| amz_B0CCKSWX3V_175212 | amazon_competitor | — | 0 | My baby girl loves these! |
| mz_R19ZUPGTJS7CAF_19780 | momcozy | — | 0 | Es un producto muy bueno |
| amz_B0FJFNDKL3_166067 | amazon_competitor | — | 0 | Me encantó me queda ajustable |
| amz_B0BYVHKZLZ_195066 | amazon_competitor | — | 0 | Le gustó mucho a mi bebé |
| amz_B0D3372YK1_194979 | amazon_competitor | 家居家纺 | 0 | Items just as described |
| amz_B0D8VM1HLF_186654 | amazon_competitor | — | 0 | Size and quality good |
| amz_B0C3RGKNPQ_186165 | amazon_competitor | — | 0 | Helps with messes for baby. |
| amz_B09NYKBP6K_194225 | amazon_competitor | — | 0 | Best leggins |
| mz_R3LBUPJMK49UZV_19228 | momcozy | 吸奶器 | 0 | Didnt clean milk residue well |
| amz_B092JKR5Y6_184832 | amazon_competitor | 吸奶器 | 0 | Ideal si se tiene extractor |
| amz_B0CR4YFTPZ_194528 | amazon_competitor | — | 0 | Good stuff |
| zd_941290_4706 | zendesk | — | 0 | Conversation with LoriMiller |
| amz_B09XJKHCXZ_189516 | amazon_competitor | — | 0 | My son LOVES these!! |
| amz_B0CQJT4P7H_194128 | amazon_competitor | — | 0 | Good product. |
| zd_954880_15243 | zendesk | — | 0 | Conversation with Serafina |
| mz_R2LFCH3IRZQTBA_19599 | momcozy | 母婴综合护理 | 0 | The best wipes ever! |
| zd_1083364_4716 | zendesk | — | 0 | Conversation with Alberto C |
| amz_B0BTYVZWF9_190418 | amazon_competitor | — | 0 | Finish is nice and clean |
| zd_1122710_24637 | zendesk | 吸奶器 | 0 | Conversation with Marilyn |
| zd_923442_32862 | zendesk | — | 0 | Conversation with Krystle |
| zd_1111751_47615 | zendesk | — | 0 | Sent from my iPhone |
| amz_B0DBP99TZQ_184835 | amazon_competitor | — | 0 | Did what needed to be done. |
| zd_926081_32851 | zendesk | — | 0 | IMG_4894.jpeg IMG_4873.mp4 |
| amz_B0779Z53SD_166978 | amazon_competitor | — | 0 | Ok but don’t like the closure |
| amz_B07NNVPMFG_167172 | amazon_competitor | — | 0 | Fits just like the add says. |
| zd_1109776_21561 | zendesk | 吸奶器 | 0 | Conversation with Leighoni |
| amz_B0CQJTYZL1_185019 | amazon_competitor | — | 0 | Definitely seemed to work! |

### 4.2 `off_category` （product_line 空 且 零标签 且 无品牌 mentions）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| zd_1036220_13734 | zendesk | — | 0 | This is a follow-up to your previous request #1015306 "Bestellung JJ1441160"  Hallo,   Ich möchte di |
| zd_1080994_45430 | zendesk | — | 0 | Conversation with Web User 69acac60736ff9437913e342 |
| amz_B09NL59JCG_173269 | amazon_competitor | — | 0 | Bueno y a tiempo. Me gusto mucho |
| zd_1010844_32407 | zendesk | — | 0 | Conversation with Web User 69895c928aed5fd5179733fc |
| amz_B01MA265ZY_173650 | amazon_competitor | — | 0 | These are exactly what I was looking for. |
| rd_R1YZD7FUPYZPDD_5267 | reddit | — | 0 | A la segunda vez de utilizarlo,ya no succionaba. |
| zd_1095000_28808 | zendesk | — | 0 | Conversation with Web User 69b2cae0a35094bdcf8cf9f0 |
| tp_5f848f28798e6f0bcc42d946_43471 | trustpilot | — | 0 | Gute Artikel Gute Artikel, aber Versand etwas langsam. Sollte man auf jedenfall bedenken und lieber  |
| zd_1065995_13724 | zendesk | — | 0 | Hi, I need to cancel this order, made the purchase by accident. Ordee# JJ1510796  Thank you, Susan W |
| amz_B0G6S6B196_105796 | amazon_competitor | — | 0 | This lamp became common in my life, the skin, circulation and inflammation have thanked it for its a |
| tp_5a7460ee6116dd0e9c217441_88438 | trustpilot | — | 0 | RAPIDE et sérieux je ne suis jamais … RAPIDE et sérieux je ne suis jamais déçue par Beaba ... |
| zd_1086086_43061 | zendesk | — | 0 | Conversation with Web User 69af340b9caa106daaf067eb |
| tp_5a4b2c60a5b3290f746323f7_27217 | trustpilot | — | 0 | Excellent quick service and lovely … Excellent quick service and lovely material Was a gift to my si |
| zd_1041375_43104 | zendesk | — | 0 | Conversation with Web User 69982eb41dff7cd437cb0518 |
| amz_B0DHGZ8PD6_178693 | amazon_competitor | — | 0 | La mejor solución para mi bebe |
| amz_B0DLLTJFJL_147474 | amazon_competitor | — | 0 | This has made a drastic improvement on how fast my husband and I fall asleep and stay asleep. He men |
| tp_67334093b002c10ee8b31025_87958 | trustpilot | — | 0 | Lange Lieferzeit und mit PayPal konnte… Lange Lieferzeit und mit PayPal konnte nicht gezahlt werden. |
| mz_R22XWR0D7X1R8X_15539 | momcozy | — | 0 | Why did you pick this product vs others?:OBSESSED is an understatement. It’s saving our backs big ti |
| zd_1123172_16490 | zendesk | — | 0 | Conversation with Web User 69c0dc557caa293f3d325e75 |
| zd_1107436_14466 | zendesk | — | 0 | Hi,  I haven't received any tracking information yet. When is my order expected to arrive?  Thanks i |
| tp_636bece3b84cc27618e71a81_53060 | trustpilot | — | 0 | Die bestellten Karten sind einfach… Die bestellten Karten sind einfach Klasse. Leider ist die Bearbe |
| zd_1139213_46210 | zendesk | — | 0 | Can you please cancel this order? It doesn’t appear to have shipped yet Thank you Dorie Brown Sent f |
| tp_5a4a35dea5b3290f746301b9_80374 | trustpilot | — | 0 | service rapide et efficace service rapide et efficace |
| amz_B075KCNKVT_81811 | amazon_competitor | — | 0 | Al principio pensé que no me iba a funcionar. Luego de 8 horas en el trabajo (de escritorio, mayorme |
| amz_B00MYRXSE2_108765 | amazon_competitor | — | 0 | Para mí es súper práctica deja los teteros bien secos y eso me encanta no necesito ninguna más con e |
| tp_619ef014bd63dda482642ad2_75408 | trustpilot | — | 0 | One month gone and I still have not… One month gone and I still have not received it. |
| mz_R107ELQ98R2A94_19177 | momcozy | — | 0 | Mejores productos post parto natural |
| tp_5a0a9c7d1ea9160b44fddf81_76339 | trustpilot | — | 0 | Site agréable et facile d'utilisation Site agréable et facile d'utilisation. Envoi rapide et soigné |
| zd_971196_3143 | zendesk | — | 0 | Hallo  Ich habe die Walsche Ware schon zurück geschickt und wollte wissen wann meine richtige Ware a |
| amz_B09QKR3PHV_124891 | amazon_competitor | — | 0 | Been using it for a month now, it’s really good. Haven’t had any issues. Makes my night so easier. |
| tp_5e37e2913c93ae04c0d914a4_43657 | trustpilot | — | 0 | Le produit est conforme au descriptif Le produit est conforme au descriptif, la rapidité de livraiso |
| mz_R12V3KX6LETYWG_18858 | momcozy | — | 0 | It really helped my baby when he was teething. |
| zd_1031999_34197 | zendesk | — | 0 | Hey sorry i put the wrong delivery address on here i was wondering if i could change it before it is |
| zd_957499_45505 | zendesk | — | 0 | Conversation with WhatsApp User 33767867255 |
| tp_661e82137d752ff980dd49ee_68636 | trustpilot | — | 0 | Schönste Auswahl für Babys und Kleinkinder Es macht einfach Spaß durch den Shop zu stöbern: geschmac |
| zd_956674_47206 | zendesk | — | 0 | Conversation with Web User 696b4bea42966220d9e2886d |
| tp_69770b9ba02a595c36009751_78574 | trustpilot | — | 0 | Commande rapide et livraison soignée Commande rapide, livraison soignée |
| tp_6069e792f85d75087047c319_46199 | trustpilot | — | 0 | The reception staff were really nice The reception staff were really nice. Lovely experience 😊 |
| amz_B09FL82SRC_182319 | amazon_competitor | — | 0 | Muy buena calidad muy fuerte y es bien grande |
| mz_R1SJX7MM947WDG_16348 | momcozy | — | 0 | It is not compatible with other lids, so I can't use it with other ones. |
| zd_952894_41362 | zendesk | — | 0 | You guys shipped it to my old address I don’t even know how it filled in that way I put in my addres |
| tp_6983678686cc20dd718dd90b_47853 | trustpilot | — | 0 | Har ALDRIG oplevet så dårlig service Har ALDRIG oplevet så dårlig service. Købte et Philips TV der f |
| amz_B073YC2F4Q_148081 | amazon_competitor | — | 0 | Maravilloso me encantó todo más el color |
| tp_68ab40b9d70a6ab0bd3a131f_93840 | trustpilot | — | 0 | Parfait Parfait la tétine est un peu grosse mais sinon nickel |
| amz_B07FCDY93D_138114 | amazon_competitor | — | 0 | Bought this for a gift. They had nothing but good things to say about it. |
| amz_B075KBXFBK_110435 | amazon_competitor | — | 0 | Fits very nice, I got a size medium. |
| mz_R31IHF5K3W3TZ1_13924 | momcozy | — | 0 | This is not as thick as I would like it's very thin material. So it gets wetter quickly |
| zd_1041152_32326 | zendesk | — | 0 | Conversation with Web User 6998059382a7464b197fc4a4 |
| amz_B09F7ZHPKP_177679 | amazon_competitor | — | 0 | De buena calidad y tal como está en las fotos, recomendable y portátil |
| mz_RWK8AW7AXTMTB_18452 | momcozy | — | 0 | Enfin une gaine suffisamment "grande" ! C'est à dire qu'elle monte assez haut et descend assez bas,  |
| tp_6718aaa48aeab4969a1005cf_87789 | trustpilot | — | 0 | Schöne und hochwertige Produkte und… Schöne und hochwertige Produkte und schneller zuverlässiger Ver |
| zd_1117530_44879 | zendesk | — | 0 | Conversation with Web User 69bd2e714596e7f1800b083c |
| tp_63a083f12338b6d417a09ecf_94462 | trustpilot | — | 0 | Commande rapide et efficace. Commande rapide et efficace. |
| tp_6945e053a5f349ec3f3604d5_23810 | trustpilot | — | 0 | smooth delivery service and excellent… smooth delivery service and excellent product quality |
| tp_5eb991f325e5d20a88928896_97067 | trustpilot | — | 0 | Parfait Parfait ! Commande passée rapidement |
| amz_B0DLLTJFJL_128054 | amazon_competitor | — | 0 | I got the cocoa color and it was like shown in the pictures! I also liked the sounds, they still wok |
| zd_987464_5410 | zendesk | — | 0 | Conversation with Web User 697c1e83a34a543448aaf385 ====================== Hello Dear,  I just found |
| amz_B0866RV9SM_163208 | amazon_competitor | — | 0 | My baby is 7 months and just got teeth. He doesn’t mind the taste and is very interested in holding  |
| tp_698227a63bfae00e45e7613a_80188 | trustpilot | — | 0 | The delivery was really fast The delivery was really fast |
| tp_6502b2ddbc4f2027d5e60688_45763 | trustpilot | — | 0 | A good product keeps planet from… A good product keeps planet from pollution. Service very quick. |
| tp_65af6c8242e610b1cdec7902_84333 | trustpilot | — | 0 | Commande simple et facile pour une… Commande simple et facile pour une liste de naissance. Envoi rap |
| amz_B07YNY75LX_100824 | amazon_competitor | — | 0 | Consider  your waist b/c the high waist might not be for you. You might can use the short size  the  |
| mz_R3NFVGN6QN1MBJ_19087 | momcozy | — | 0 | Not as simple as instructions make it out to be |
| zd_1122357_14075 | zendesk | — | 0 | Conversation with Web User 69c05fcb74ff1a20e5f9bfce |
| tp_6027a3b2679d9708b4d4dbf3_61573 | trustpilot | — | 0 | Nous souhaitions commander la poussette… Nous souhaitions commander la poussette Melio chez Cybex. N |
| amz_B08687T6WR_116969 | amazon_competitor | — | 0 | It's only been a week but so far so good! Will check back in 3-6 months |
| tp_65e3159e605d52b3aa132ea4_37644 | trustpilot | — | 0 | Le colis ne m'a jamais été livré ! Comment puis-je noter un colis qui ne m'a jamais été livré ? Le c |
| amz_B0DKFG7LSY_154151 | amazon_competitor | — | 0 | Fit is good ,  Comes with extra hook band. |
| zd_1080171_28756 | zendesk | — | 0 | Hello,   I couldn’t find an option to add a gift note to my order number: JJ1526927.  I would like t |
| mz_RE0S1FNABSP6G_19027 | momcozy | — | 0 | Check before buying. The app does not work worldwide. |
| tp_5ceeda70a84369094cfa9655_69956 | trustpilot | — | 0 | Bestens geklappt Bestens geklappt, schnell und problemlos. |
| amz_B003P9WSVC_110692 | amazon_competitor | — | 0 | Someone got this for us as a gift but as most people do when they give gifts, there only concern is  |
| amz_B07DWLTGSF_184319 | amazon_competitor | — | 0 | Excelente buenisimas si funciona |
| zd_1038753_34851 | zendesk | — | 0 | Conversation with Web User 6996f9b2b281defdc673b6b3 |
| amz_B0D6VX1HYZ_113506 | amazon_competitor | — | 0 | My daughter loves these! The different sizes work so well for her and they are the easiest to clean! |
| mz_R1VQE3UMVAWFMQ_17424 | momcozy | — | 0 | My baby loves very good product, excellent material. |
| zd_1071327_3171 | zendesk | — | 0 | Hallo, ich wollte fragen, wie der Bestellstatus vo meinem Babyphone aussieht.  Die Bestellnummer; JJ |
| zd_974036_4497 | zendesk | — | 0 | Conversation with Web User 6974d84d51041a526d1f743f |
| amz_B00JV27MF4_88046 | amazon_competitor | — | 0 | These Boveda 62% RH Two-Way Humidity Control Packs are for 1oz. These are size 8. They come in a pac |
| amz_B075KBXFBK_183071 | amazon_competitor | — | 0 | Really does work. Boosts your confidence |
| tp_65a15393acf7be15eb404ad1_86926 | trustpilot | — | 0 | Bon achat correspond aux attentes de ma… Bon achat correspond aux attentes de ma fille les bottes ta |
| tp_695c15456813a0d65201d027_95203 | trustpilot | — | 0 | Très bien pour les petits Très bien pour les petits |
| tp_638bd7eb0b7fc02715d31529_96668 | trustpilot | — | 0 | Alles wunderbar gelaufen Alles wunderbar gelaufen |
| tp_59aa9bd9f05e68079cfc167e_88429 | trustpilot | — | 0 | Produit livré rapidement et conforme à … Produit livré rapidement et conforme à la commande ! |
| tp_65b8eee3e3363f03ff3253a0_78663 | trustpilot | — | 0 | Usually ships the next day after… Usually ships the next day after ordering. This time it took 3 day |
| tp_6617fe31ab9a104259759630_89814 | trustpilot | — | 0 | Ottima qualità e modelli perfetti per… Ottima qualità e modelli perfetti per bambini |
| tp_6697870eab925d502bebc11a_41921 | trustpilot | — | 0 | Nous sommes extrêmement déçu Nous sommes extrêmement déçu, nous avons reçu un parc avec deux élément |
| tp_6362f136252cba2c02cd63d9_77280 | trustpilot | — | 0 | Dommage la livraison en magasin n'est… Dommage, la livraison en magasin n'est pas proposée |
| zd_1001813_40538 | zendesk | — | 0 | Conversation with Web User 698416da409456670f8450c5 |
| zd_999953_41502 | zendesk | — | 0 | Conversation with Melissa Gonzalez |
| amz_B0B1H8V941_73430 | amazon_competitor | — | 0 | I like it but I’m not as in love with it as I was hoping to be. Maybe I should have bought a smaller |
| zd_1032004_46857 | zendesk | — | 0 | Hi there,   Can you please help with my order as tracking has never updated further than a label has |
| zd_953555_32221 | zendesk | — | 0 | Conversation with Web User 696997ee7c2c448c31683931 |
| zd_1131659_43209 | zendesk | — | 0 | Conversation with Web User 69c45cb3e79181ba22f4e15d |
| zd_1072401_45424 | zendesk | — | 0 | Conversation with Web User 69a871ad7da644c9c8b49bcd |
| tp_6759db6f90597f28b46e1ce2_78896 | trustpilot | — | 0 | Supers produits Supers produits, site très joli et rapidité de livraison |
| zd_1131718_5407 | zendesk | — | 0 | Bonjour je vient vers vous car j’ai fait une commande de plusieur article au nom de BOUKIMI AMINA ma |
| tp_639cb552d075435bd8de6deb_88779 | trustpilot | — | 0 | So wie es Freude macht. So wie es Freude macht. |
| zd_1110698_46395 | zendesk | — | 0 | Conversation with Web User 69ba09cfd054c8a9c03ee099 |
| zd_979640_41953 | zendesk | — | 0 | Conversation with Web User 697828ca10f28e4e07f1b1fd |

### 4.3 `generic_only` （零标签 且 去停用词后有效 token < 3）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| amz_B08352W1T8_117191 | amazon_competitor | 家居家纺 | 0 | I absolutely love my pillow!¡! |
| amz_B00PH7DSOQ_120091 | amazon_competitor | 智能母婴电器 | 0 | Used it for my massage clients, and they loved it! |
| amz_B0B7MFS2W4_132743 | amazon_competitor | 家居家纺 | 0 | It’s the best stroller I really like it |
| amz_B09XJJB478_140899 | amazon_competitor | 喂养电器 | 0 | The only pacifier my grandson will use. |
| amz_B07NNVPMFG_154468 | amazon_competitor | 内衣服饰 | 0 | I liked the shapewear, wears very nice! 👍 |
| amz_B00AWMP880_156620 | amazon_competitor | 内衣服饰 | 0 | Loved it as soon as it was unwrapped |
| mz_R2SRHOGC9JO6A6_18219 | momcozy | 吸奶器 | 0 | Bought for my pump. No complaints! |
| amz_B0007CQ726_160896 | amazon_competitor | 吸奶器 | 0 | I use this to lubricate my pump |
| amz_B09538RHV4_165428 | amazon_competitor | 吸奶器 | 0 | I really like it to use the pump with. |
| amz_B07RGPYHJJ_169583 | amazon_competitor | 喂养电器 | 0 | My two daughters have loved these pacifier. |
| mz_R2ZYS2LCE07FKU_16780 | momcozy | 内衣服饰 | 0 | My friend loved this bra that I bought her |
| amz_B0DTHSRTD4_170340 | amazon_competitor | 智能母婴电器 | 0 | Very good purifier for a 10x12 room. |
| amz_B0CFDZ3DG9_170949 | amazon_competitor | 内衣服饰 | 0 | Very good compression and support. |
| amz_B09NHVF9QJ_171314 | amazon_competitor | 家居家纺 | 0 | good transaction, as described. |
| amz_B0CLWXDQ2G_171626 | amazon_competitor | 家居家纺 | 0 | My granddaughter loved the crib. |
| amz_B08PMJBN33_171660 | amazon_competitor | 家居家纺 | 0 | My granddaughter loved the crib |
| amz_B07FRYYZ3N_174067 | amazon_competitor | 母婴综合护理 | 0 | I don't use any other type of wipes. |
| mz_R39MSLVTEG31ZB_19401 | momcozy | 吸奶器 | 0 | 2台の搾乳機が収納可能なハードシェルバッグです。同メーカーの装着型搾乳機2セットが固定して入るようになっています。バッグとしては、大きさの割には少し重ために感じますが、内側外側共に防水素材が使われてい |
| amz_B09K39Y11L_179803 | amazon_competitor | 家居家纺 | 0 | She liked the pillow. Now to get her pregnant... |
| amz_B0BL8M2Z1Q_181970 | amazon_competitor | 母婴综合护理 | 0 | It’s a wipe, it did what it was supposed to do. |
| amz_B0FMDMHD5G_183003 | amazon_competitor | 内衣服饰 | 0 | Very tight and definitely as shapewear |
| amz_B00P7FJM48_183185 | amazon_competitor | 吸奶器 | 0 | This is a good pumping bra! !!!! |
| amz_B08DYTD65X_183664 | amazon_competitor | 家居家纺 | 0 | These pacifiers are as described! |
| amz_B0D9M88VTX_183731 | amazon_competitor | 家居家纺 | 0 | It's exactly how it was described. |
| tp_682772be79f9307e628e8cca_72195 | trustpilot | 智能母婴电器 | 0 | Really good product Love how light the stroller is |
| rd_R3GFY0R4DMBPQN_945 | reddit | 内衣服饰 | 0 | Just like the other brands I would definitely not recommend |
| mz_R2CLIRR23JHTF4_19420 | momcozy | 吸奶器 | 0 | 手ざわりのよいのハードケースは中の搾乳機をしっかりと保護、専用バッグなのでぴったりと収まります。開閉はスムーズでカーブ部分もストレスなくファスナーが滑ります。トレイや仕切りが設けられているのでしっくり |
| zd_1004592_1214 | zendesk | 吸奶器 | 0 | Conversation with RosalindaOlivares |
| mz_R29V6K414IR7ND_19440 | momcozy | 吸奶器 | 0 | 高品質なシリコン素材で作られており、BPAフリーで安心して使用できます。ダブルシールフランジがしっかりと密閉し、効率的な搾乳をサポートします。簡単に取り付けられ、洗浄も容易です。1パックで提供されてい |
| zd_1120908_1219 | zendesk | 吸奶器 | 0 | Conversation with ElizabethTaylor |
| zd_1086926_1224 | zendesk | 吸奶器 | 0 | Conversation with KaylahBarriger |
| zd_1044502_1226 | zendesk | 吸奶器 | 0 | Conversation with JessicaLambert |
| zd_1054162_1228 | zendesk | 吸奶器 | 0 | Conversation with JasmineHughes |
| zd_1003882_1229 | zendesk | 吸奶器 | 0 | Conversation with JazminSevilla |
| zd_1121958_1235 | zendesk | 吸奶器 | 0 | Conversation with BhavikaSethi |
| zd_922464_1243 | zendesk | 吸奶器 | 0 | Conversation with RebekahPaine |
| zd_954725_1409 | zendesk | 吸奶器 | 0 | Conversation with KalissiaClark |
| zd_1089536_10150 | zendesk | 喂养电器 | 0 | When I will receive my order my order number is JJ1532515 |
| zd_987921_10649 | zendesk | 喂养电器 | 0 | สวัสดีครับ  ผมขอรบกวนติดตามสถานะคำสั่งซื้อ เนื่องจากขณะนี้คำสั่งซื้อของผมเกินระยะเวลาดำเนินการ/จัดส่ |
| mz_R3TZWUG1YN7VMQ_18855 | momcozy | 智能母婴电器 | 0 | Massager used when you've got pain |
| zd_1100441_12310 | zendesk | 喂养电器 | 0 | hi,I would like to know when my order will be shipped? |
| mz_RGYGQ2IDL95PL_19439 | momcozy | 吸奶器 | 0 | 育児バッグの中でもかなり優秀な部類でした。まず、ハードシェルなので中身をしっかり守ってくれる安心感があり、搾乳機を2台入れても余裕のある設計。取り外し可能なトレイが想像以上に便利で、整理整頓がラクにで |
| zd_1067569_12544 | zendesk | 喂养电器 | 0 | Conversation with CadieCampbell |
| mz_R8E2DLJ2IQRIZ_19433 | momcozy | 吸奶器 | 0 | 搾乳機をしっかり守って持ち運ぶ事に重きを置いている為、形状が大きく、かさばります。その分、防水性は非常に高く、中を開けると仕切りが多く整理しやすいです。互換性が高いので他の製品でも使えると思います。追 |
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
| zd_1026539_21502 | zendesk | 吸奶器 | 0 | Conversation with N Λ T Λ L ⋮ Λ |
| mz_R2CR04F45V7I2I_19467 | momcozy | 吸奶器 | 0 | 専用設計なので、搾乳機がケースの中で暴れず、収まりが非常に良いです。ですが、安い搾乳機と引けを取らないくらいの価格帯でケースのみなので、購入をしにくいのは難点。 |
| zd_1028758_21532 | zendesk | 吸奶器 | 0 | Conversation with Jo Jo Lovell |
| zd_1119520_24592 | zendesk | 吸奶器 | 0 | Why is the item not shipped yet? |
| mz_R1XWJRWB7XIIOK_19635 | momcozy | 吸奶器 | 0 | 純正品のためサイズが合っていて、問題なく使用できました。特に不満はありません。 |
| zd_983755_24617 | zendesk | 吸奶器 | 0 | Conversation with KaiaDeLaRosa |
| zd_946595_24621 | zendesk | 吸奶器 | 0 | Conversation with Monica Muñiz |
| zd_1144877_24779 | zendesk | 吸奶器 | 0 | Conversation with AllisonVance |
| zd_1019606_27033 | zendesk | 吸奶器 | 0 | Conversation with MorganHarrelson |
| zd_1026992_27047 | zendesk | 吸奶器 | 0 | My order says it was delivered and it’s not |
| mz_R1OGOFWR2JYVKQ_19424 | momcozy | 吸奶器 | 0 | 旅行、帰省などで搾乳器の持ち運びとして活用するために注文しました。機械が守れるので故障する心配なく助かります。ただちょっと大きすぎますね。。もう少しコンパクトだと嬉しいです。ですが他のパーツや充電器も |
| zd_1014071_27056 | zendesk | 吸奶器 | 0 | Conversation with KristenAlexander |
| zd_918399_27059 | zendesk | 吸奶器 | 0 | Conversation with CatherineStewart |
| zd_1040055_27061 | zendesk | 吸奶器 | 0 | Conversation with AllyFriedenberger |
| zd_1122199_27072 | zendesk | 吸奶器 | 0 | Conversation with RaeganMueller |
| mz_R17JX8BPZ2USQ2_19163 | momcozy | 吸奶器 | 0 | This did not come with the duck valves. |
| zd_1039366_27076 | zendesk | 吸奶器 | 0 | Conversation with JessicaNalerio |
| zd_992356_27079 | zendesk | 吸奶器 | 0 | Conversation with NelidaGonzalez |
| zd_1025068_27080 | zendesk | 吸奶器 | 0 | Conversation with Yasmin 🥰🫶🏻😇🙏🏻 |
| mz_R1Z9UY70KLKXHF_19389 | momcozy | 吸奶器 | 0 | 搾乳機の専用ケースだけあって収まりがよく、破損など絶対なさそうです。子どもが入院してる時に搾乳が必要なことがあり、こういったケースがあるのは心強いです。高級感あるこういうケースから出し入れする時はテン |
| zd_932042_39666 | zendesk | 吸奶器 | 0 | Hi I would like to to cancel my order   JJ1402104  Please |
| mz_R37ZO13P17LEC1_19425 | momcozy | 吸奶器 | 0 | 搾乳機は使っていないので小さめのカップなどの洗浄に使っていますがやはり用途が違いサイズが短いため洗いにくいのが本音です。品質的には持ち手などはしっかりしているもののスポンジ部分はもう一つかな…本来の使 |
| zd_930820_27093 | zendesk | 吸奶器 | 0 | Conversation with Ayşe Kocaman |
| zd_1130148_27096 | zendesk | 吸奶器 | 0 | Conversation with AmandaForney |
| zd_1102395_27097 | zendesk | 吸奶器 | 0 | Conversation with CristinaDiaz |
| zd_927691_27105 | zendesk | 吸奶器 | 0 | Conversation with AmandaRoeske |
| mz_R15HG3K8WN5UKE_19478 | momcozy | 吸奶器 | 0 | 専用ケースで、授乳部品の隙間が無く丁度よく使えました。出先など、ほこりの心配も少ない点が良かったです。引き続き耐久性を確認していきます。 |
| zd_1121465_27112 | zendesk | 吸奶器 | 0 | Conversation with BrigitteDery |
| zd_973678_27136 | zendesk | 吸奶器 | 0 | الاسم:- احمد محمد العطار  الشارع:- 932/ابن سيرين (منطقة 34)  رقم المبني:- 150 المدينه:- مدينة خليفة  |
| zd_1137196_27137 | zendesk | 吸奶器 | 0 | الصفحه دي بتقول انها توكيل ممكوزي في مصر ؟؟  هل ده حقيقي ؟؟ |
| zd_1080110_28525 | zendesk | 吸奶器 | 0 | Conversation with BriannaMitchell |
| mz_R12WAJ0TO461RW_18269 | momcozy | 吸奶器 | 0 | Exactly what I was looking for |
| zd_986769_28533 | zendesk | 吸奶器 | 0 | Conversation with VictoriaCarman |
| zd_1055792_28535 | zendesk | 吸奶器 | 0 | Conversation with SelenaDagher |
| zd_1086070_28540 | zendesk | 吸奶器 | 0 | Conversation with SelenaDagher |
| zd_1135887_28589 | zendesk | 吸奶器 | 0 | Can I cancel my order thank you |
| zd_1129054_28672 | zendesk | 吸奶器 | 0 | Conversation with CassidyLasick |
| zd_1125668_35832 | zendesk | 吸奶器 | 0 | Hi,  I would like to cancel my order. My order number is  JJ1567311 |
| zd_920160_36533 | zendesk | 吸奶器 | 0 | Abigail S. DiSalvo (203)-885-9313 |
| zd_1121578_36556 | zendesk | 吸奶器 | 0 | Conversation with Patrick1010!! |
| mz_R21C9UMPTDWO7X_19016 | momcozy | 喂养电器 | 0 | Every parents should have this |
| zd_966862_37019 | zendesk | 吸奶器 | 0 | Hi, I want the refund.  order 1409307 |
| zd_1054287_37048 | zendesk | 吸奶器 | 0 | Conversation with DestinyMelton |
| zd_927745_38952 | zendesk | 吸奶器 | 0 | Conversation with BrookeMcClain |
| zd_927521_39601 | zendesk | 吸奶器 | 0 | JJ1385202 Has this order being sent yet? |
| zd_1021491_39362 | zendesk | 吸奶器 | 0 | Conversation with JessicaBennett |

## 五、阈值判定（决策 4）

| 指标 | 阈值 | 实测 | 结果 |
|---|---:|---:|:---:|
| 原始覆盖率 | 88.00% | 89.37% | 🟢 PASS |
| 业务有效覆盖率 | 94.00% | 96.12% | 🟢 PASS |

## 六、与 Phase 4 对比

- Phase 4 5K 子集原始覆盖率：82.58%
- Phase 5 全量原始覆盖率：89.37% （+6.79pp）
- Phase 5 全量业务有效覆盖率：96.12%
