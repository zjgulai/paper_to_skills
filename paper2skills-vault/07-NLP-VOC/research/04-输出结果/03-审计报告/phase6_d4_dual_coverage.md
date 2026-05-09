---
name: phase5-d10-dual-coverage-report
description: Phase 5 D10 双覆盖率审计报告（原始 / 广义 / 业务有效）。当审计 Phase 5 整体覆盖率达成、与 Phase 4 对比、查看排除分布与抽样时使用。
computed_at: 2026-05-09T19:07:07
doc_type: audit-report
module: voc-nlp
status: stable
---

# Phase 5 D10 双覆盖率审计报告

**输入**：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d4_merged.jsonl`
**总样本**：364,569
**计算时间**：2026-05-09T19:07:07

## 一、核心指标

| 指标 | 分子 | 分母 | 覆盖率 | Phase 4 基线 | Δ |
|---|---:|---:|---:|---:|---:|
| 原始覆盖率 | 294,588 | 364,569 | **80.80%** | 82.58% | -1.78pp |
| 广义覆盖率（含品牌命中）| 294,605 | 364,569 | **80.81%** | — | — |
| 业务有效覆盖率 | 294,588 | 324,153 | **90.88%** | — | — |

> 业务有效分母 = 总数 - 排除（40,416 条，占比 11.09%）

## 二、排除桶分布

| 桶 | 定义 | 样本数 | 占总比 |
|---|---|---:|---:|
| `too_short` | text.strip() < 30 字符 | 3,694 | 1.01% |
| `off_category` | product_line 空 且 零标签 且 无品牌 mentions | 36,549 | 10.03% |
| `generic_only` | 零标签 且 去停用词后有效 token < 3 | 173 | 0.05% |

## 三、按数据源拆解

| data_source | 总数 | with_label | 原始覆盖率 | brand_only | 排除合计 | 业务有效覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| amazon_competitor | 194,734 | 164,042 | 84.24% | 14 | 18,694 | 93.18% |
| trustpilot | 99,853 | 88,801 | 88.93% | 0 | 9,944 | 98.77% |
| zendesk | 47,204 | 24,060 | 50.97% | 2 | 9,685 | 64.13% |
| momcozy | 19,808 | 15,609 | 78.80% | 1 | 1,591 | 85.68% |
| reddit | 2,970 | 2,076 | 69.90% | 0 | 502 | 84.12% |

## 四、排除追踪抽样（每桶最多 100 条，供人工 spot check）

### 4.1 `too_short` （text.strip() < 30 字符）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| amz_B0C1NSFSSS_191501 | amazon_competitor | — | 0 | Working so good |
| amz_B0F58BHLF5_190580 | amazon_competitor | — | 0 | Sharp image.Easy setup. |
| amz_B0DLGDRRXY_172709 | amazon_competitor | — | 0 | Very difficult to set up… |
| zd_1085421_34905 | zendesk | — | 0 | Conversation with Gelamming |
| amz_B0C73ZRQPX_125616 | amazon_competitor | — | 0 | Work very well, I recommend 👍 |
| zd_1111751_47615 | zendesk | — | 0 | Sent from my iPhone |
| amz_B0CBT9R4FM_126055 | amazon_competitor | — | 0 | Nice fit and I wanted pockets |
| amz_B0CNGK191C_185263 | amazon_competitor | — | 0 | Prácticos , buena calidad |
| amz_B07QL4B6S5_174791 | amazon_competitor | — | 0 | It Fits better than expected. |
| amz_B07WCBD6YC_193461 | amazon_competitor | — | 0 | Daughter loves it |
| amz_B00JF3RYPM_131166 | amazon_competitor | — | 0 | I like it I use for my self |
| zd_1025676_1259 | zendesk | 吸奶器 | 0 | Conversation with Nikilia |
| amz_B07NSNGSCV_195533 | amazon_competitor | — | 0 | It's very supportive |
| tp_686abff1896045a2a7b891cd_98276 | trustpilot | — | 0 | Envoie rapide Envoie rapide |
| amz_B0F4KD57W5_162768 | amazon_competitor | 吸奶器 | 0 | keeps my breast milk cold! |
| amz_B0CGQNN29F_195185 | amazon_competitor | — | 0 | Didn’t do much for me |
| amz_B08L68WR4B_139664 | amazon_competitor | 喂养电器 | 0 | Best to recover my nipples |
| zd_927910_28545 | zendesk | 吸奶器 | 0 | Conversation with Deepanshu |
| amz_B07Q51PPF1_184819 | amazon_competitor | — | 0 | Rolls up does not stay flat |
| amz_B08QMSNGKS_184792 | amazon_competitor | — | 0 | Excelente, es transparente |
| amz_B07KF61W2N_173846 | amazon_competitor | — | 0 | didn’t work for me at all |
| amz_B0C47L679F_140162 | amazon_competitor | — | 0 | Wife loves the feel and fit. |
| amz_B09F7ZHPKP_191787 | amazon_competitor | — | 0 | Very nice baby loved bed |
| amz_B0BQ6X1CCS_190840 | amazon_competitor | — | 0 | Really good and covers |
| tp_65761963c31e38045a916f37_99805 | trustpilot | — | 0 | Alles prima Alles prima |
| amz_B08KHW6C49_195214 | amazon_competitor | — | 0 | Don't  clean very well |
| zd_968816_45519 | zendesk | — | 0 | Conversation with Fatiha 🌺 |
| amz_B0D7VVDPQX_178699 | amazon_competitor | — | 0 | Very useful during pregnancy. |
| amz_B0C1Y9BQBV_195252 | amazon_competitor | — | 0 | Sólido, buena calidad |
| amz_B086P8NJF4_185084 | amazon_competitor | — | 0 | Fue un regalo no ha quejas |
| amz_B0DHH25XM1_190542 | amazon_competitor | — | 0 | Nephew loved these toys |
| amz_B08LR2TTP8_194222 | amazon_competitor | — | 0 | She loves it |
| amz_B014A7MB4G_192518 | amazon_competitor | — | 0 | Drawers are not solid |
| mz_R2UHTRWMIIN12F_19786 | momcozy | — | 0 | Muy buenas buen tamaño |
| amz_B09DY7NQGJ_191200 | amazon_competitor | — | 0 | My son loves them |
| mz_R2ZLCM08WEDNUK_18379 | momcozy | — | 0 | My daughter-in-law loves it |
| amz_B09C8N75VM_179464 | amazon_competitor | — | 0 | Muy cómoda, buen material |
| amz_B0DDS1PLCD_184641 | amazon_competitor | — | 0 | Didn’t realize it was a thong |
| amz_B0CG5C9KRS_189688 | amazon_competitor | — | 0 | Snatch my body together |
| mz_R2X81BFWBRM5MP_19665 | momcozy | — | 0 | Excelente y práctico uso |
| amz_B07QRRM748_166109 | amazon_competitor | — | 0 | Muy buena calidad me encantó |
| amz_B09M98HYQX_194354 | amazon_competitor | — | 0 | good product |
| amz_B0D5CRDY5G_194366 | amazon_competitor | — | 0 | good workout |
| amz_B0CBRTKMNC_195098 | amazon_competitor | — | 0 | It did not work for me. |
| tp_629a5511d5573095630a763d_98227 | trustpilot | — | 0 | Very pleased! Very pleased! |
| amz_B0BK9F3XP9_184447 | amazon_competitor | — | 0 | Muy buenas cómodas y bonitas |
| zd_976531_1262 | zendesk | 吸奶器 | 0 | Conversation with Carson |
| amz_B0F73SZP81_189211 | amazon_competitor | — | 0 | Love. Worth your money. |
| amz_B07NZPT3TR_193915 | amazon_competitor | — | 0 | Easy and light |
| tp_64733136706f837cb1f33751_98275 | trustpilot | — | 0 | Alles perfekt Alles perfekt |
| zd_1041465_1501 | zendesk | — | 0 | Conversation with Mimina 🥀 |
| zd_1006597_1397 | zendesk | 吸奶器 | 0 | Conversation with BabesSapon |
| amz_B0D3CY9X4J_192365 | amazon_competitor | — | 0 | not fit for large women |
| amz_B0DT48R3BR_195387 | amazon_competitor | — | 0 | Trust Amazon & Dreo. |
| amz_B0BYZ13585_174801 | amazon_competitor | 吸奶器 | 0 | ITS THE BEST PUMPING BRA EVER |
| zd_1075193_32794 | zendesk | — | 0 | Conversation with Marisol |
| amz_B0BLVHBLRP_193468 | amazon_competitor | — | 0 | Very good product |
| zd_1092703_12551 | zendesk | 喂养电器 | 0 | Conversation with Emanuel |
| amz_B0BG98BPW6_194605 | amazon_competitor | — | 0 | Loved this |
| amz_B0B8TMQDN3_180643 | amazon_competitor | — | 0 | used this product for a baby. |
| tp_644dfdcd734f97e68fa7f04b_99743 | trustpilot | — | 0 | 29042023 Parfait intacte |
| amz_B0CQD95SFC_190446 | amazon_competitor | 内衣服饰 | 0 | Really good compression |
| amz_B0CR134MHF_193575 | amazon_competitor | — | 0 | Fiancée Loves It |
| zd_1109469_47785 | zendesk | — | 0 | Conversation with M. Othman |
| amz_B07Q3ZSB4Y_158016 | amazon_competitor | — | 0 | Too short for my long torso |
| tp_67d592bdc20ec0e976b88592_98345 | trustpilot | — | 0 | Impeccable ! Impeccable ! |
| zd_1059164_6258 | zendesk | — | 0 | Conversation with Ahmad ~ |
| amz_B0C4NZHX61_195332 | amazon_competitor | — | 0 | Funciona de maravilla |
| mz_R3DN7FSBERYQC3_18889 | momcozy | 吸奶器 | 0 | Thought they would be bigger |
| mz_R2J9OHKMC4G472_19244 | momcozy | — | 0 | Suave con la boca del bebé |
| amz_B0CRBLKKF7_190781 | amazon_competitor | 家居家纺 | 0 | The best for the gym! |
| amz_B0D6KGJ1V6_194739 | amazon_competitor | — | 0 | es una faja moldeadora |
| amz_B00CWXWXZ8_195425 | amazon_competitor | — | 0 | Nickel comme produit |
| amz_B0BS94NLY6_178448 | amazon_competitor | — | 0 | Flat not like picture flabby |
| amz_B07QVLTZ5N_162984 | amazon_competitor | — | 0 | Baby loves it gets so excited |
| amz_B085ZJLP3W_178895 | amazon_competitor | — | 0 | We used and had no problems |
| amz_B09XJJTJR9_172462 | amazon_competitor | — | 0 | My little one loves these. |
| zd_1115796_12768 | zendesk | — | 0 | Conversation with Hillary |
| amz_B0DK9Y8HQ4_195101 | amazon_competitor | — | 0 | Bought as a shower gift |
| amz_B0C5B7XTP8_195485 | amazon_competitor | — | 0 | Me gustaron bastante |
| tp_5edfd01c7dd7530690e49b32_98148 | trustpilot | — | 0 | alles geklappt alles geklappt |
| amz_B0DXQ3MKQ2_195039 | amazon_competitor | — | 0 | Doesn't come with manual |
| zd_1142102_33661 | zendesk | — | 0 | Conversation with علي Ali |
| amz_B0CTQQ61D7_192628 | amazon_competitor | — | 0 | lower cut then shown |
| amz_B07S82FTD5_184495 | amazon_competitor | — | 0 | Muy buena calidad y cómoda 👌 |
| amz_B0C7T8LZDT_194864 | amazon_competitor | — | 0 | I plan on buyer g more. |
| mz_R1QLDC3GFLG40E_19262 | momcozy | 吸奶器 | 0 | Modèle parfait et d'origine |
| amz_B0C1SG13F2_194970 | amazon_competitor | — | 0 | Way to firm and awkward. |
| zd_1101061_27099 | zendesk | 吸奶器 | 0 | Conversation with AliciaSmits |
| zd_1039172_27135 | zendesk | 吸奶器 | 0 | Conversation with Roxanna |
| amz_B0C3DJZ3YW_174912 | amazon_competitor | 吸奶器 | 0 | The best pumping bra. Ever. |
| amz_B0DKPHQ5B5_194285 | amazon_competitor | — | 0 | She loves it |
| amz_B0FRS9BC2P_194224 | amazon_competitor | — | 0 | Good product |
| amz_B0D35KRVL8_185132 | amazon_competitor | — | 0 | Je regrette pas mon achat |
| amz_B0CPDXGW25_195140 | amazon_competitor | — | 0 | Excelente recomendado |
| amz_B0B8HDP3QP_190513 | amazon_competitor | — | 0 | Very good, confortable. |
| tp_68ed05807bb9430638c86690_83056 | trustpilot | — | 0 | Good bumper Decent bumper . |
| amz_B00TU03HU6_194942 | amazon_competitor | — | 0 | As advertised. No issues |
| zd_1058023_21521 | zendesk | 吸奶器 | 0 | Conversation with Matan Naim |
| zd_1086702_33929 | zendesk | — | 0 | Conversation with Domi V🌸 |

### 4.2 `off_category` （product_line 空 且 零标签 且 无品牌 mentions）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| tp_6868a6d84702142dace3a4ef_25720 | trustpilot | — | 0 | Excellent costumer service Excellent costumer service. My toddlers loved kendamil organic formula. A |
| zd_937598_42567 | zendesk | — | 0 | Conversation with Web User 69613eeb963e4f4d562dd8a0 |
| zd_1038658_209 | zendesk | — | 0 | Sehr geehrte Damen und Herren,   ich habe bei Ihnen eine Babyschaukel bestellt. Die Schaukel ist wir |
| tp_5ed12c9525e5d20a88a2a766_53504 | trustpilot | — | 0 | Le produit est conforme et correspond à… Le produit est conforme et correspond à nos attentes. Par c |
| zd_950525_3015 | zendesk | — | 0 | Conversation with Web User 6968217a71daa8bfa98897f2 |
| amz_B00BVUQEWG_150658 | amazon_competitor | — | 0 | Our baby takes these binkies so easily, and they glow the whole night long which makes it easy for u |
| tp_5ed69e2825e5d20a88a5d370_46422 | trustpilot | — | 0 | S’abstenir absolument !! Commande passer le 07 mai aucun article reçu tout est en préparation depuis |
| amz_B0CTQQ61D7_160191 | amazon_competitor | — | 0 | I’m between S and M size, bought M because the reviews were saying it runs on a small side, don’t fe |
| tp_690e40672c7d9e3fdcfbcade_97786 | trustpilot | — | 0 | Très beau Très beau Très pratique |
| amz_B0BSSZW9H8_146695 | amazon_competitor | — | 0 | Nothing at all like the pictures. Very short in the torso, and does NOT hide everything like the pro |
| amz_B09FL7LV9W_136901 | amazon_competitor | — | 0 | My wife has loved it since I got it for her |
| tp_62b4d388ba5bb6ef04217d21_70709 | trustpilot | — | 0 | Le site est claire et facile de… Le site est claire et facile de navigation entre les différentes ca |
| amz_B08GHZW3GL_154831 | amazon_competitor | — | 0 | My wife loved her gift she sleeps like a baby no tossing and turning on her side just |
| tp_68136f0ecf4e0731ab4162ee_71083 | trustpilot | — | 0 | Rapide et efficace La qualité est au rdv comme toujours. La livraison ultra rapide c’est génial surt |
| amz_B0CLCG93B5_136719 | amazon_competitor | — | 0 | Granddaughter loves it! She think she has something she doesn't suppose to have😊 |
| zd_1046349_16471 | zendesk | — | 0 | Conversation with Web User 699b5be946fea87d9a7e6a1e |
| amz_B014A7MAYC_178152 | amazon_competitor | — | 0 | Todo viene al 100 buen material |
| tp_69b68da9d8bbe34c91af1b65_88630 | trustpilot | — | 0 | magasin propre bien ordonné ce qui… magasin propre bien ordonné ce qui facilite les recherches |
| tp_6806398784a276ef6e83a64b_96786 | trustpilot | — | 0 | Taille très bien Taille très bien, belle couleur |
| tp_694a5afea96b1dbe6fd0fd6e_32893 | trustpilot | — | 0 | Exzellenter Kundenservice Der Kundenservice bei der Kartenmacherei ist exzellent und reagiert sehr s |
| tp_63541da58056669a30a873fc_88763 | trustpilot | — | 0 | Einfach Einfach, schnell und unkompliziert. Macht Spaß! Große Auswahl und die Qualität passt! |
| zd_955903_45342 | zendesk | — | 0 | Bonjour,    J’ai passé une commande le 9 janvier 2026 et à ce jour, je n’ai aucune nouvelle de ma co |
| zd_1083596_32553 | zendesk | — | 0 | Conversation with Web User 69ae3f134f1a3037b0a61262 |
| amz_B0BMFQFZSH_54501 | amazon_competitor | — | 0 | This is the worst product of any kind that I have ever used. None of the locking mechanisms would en |
| amz_B08YWHN79N_72702 | amazon_competitor | — | 0 | I was very hopeful after reading the reviews but I guess I  know why the price is so low. 3 assembly |
| rd_739506_3068 | reddit | — | 0 | Call from: +1 (970) 275-9840 Call to: +1 (619) 848-0676 Time of call: October 1, 2025 at 5:05:46 PM  |
| zd_1075204_11817 | zendesk | — | 0 | Conversation with Web User 69a9ac15d120f56fc502904e |
| amz_B0C9GTH2R1_160307 | amazon_competitor | — | 0 | What song doesn't really sterilize and doesn't really dry you have to keep putting a water so many t |
| amz_B0DKFG7LSY_154151 | amazon_competitor | — | 0 | Fit is good ,  Comes with extra hook band. |
| zd_980560_29167 | zendesk | — | 0 | Conversation with Web User 6978be428d1777bc9d222f89 |
| amz_B07WDHNKKB_160249 | amazon_competitor | — | 0 | N'est pas tout à fait idéal pour les redressements assis les pieds trop rapprochés pour le soutien.  |
| tp_5ee125d47dd75307088005f6_87108 | trustpilot | — | 0 | Pour rentrer la date de naissance quand… Pour rentrer la date de naissance quand on est né en 87 com |
| zd_983282_34252 | zendesk | — | 0 | Hi I haven’t received my order i paid insurance as well idk how I can be helped  Sent from my iPhone |
| mz_R1J69J1P0XDNBG_17866 | momcozy | — | 0 | Angenehme Creme mit zitronigem Duft.Ich nutze die Creme abends, nachdem ich mir den Bauch mit einer  |
| amz_B0CND44J4C_166858 | amazon_competitor | — | 0 | Engaging and fun way to strengthen the pelic floor muscles |
| zd_1094717_31870 | zendesk | — | 0 | Sehr geehrte Damen und Herren,    mein Name ist Claudia Bonanno. Am 10.03. habe ich das Produkt „Tra |
| mz_RH00LQI09VJ9P_15135 | momcozy | — | 0 | No me gustó, me pareció muy costoso todo. Y innecesario |
| amz_B07MS8QP2J_122797 | amazon_competitor | — | 0 | I didn’t know that I had ordered enough for an entire daycare center.  Packaging very inconvenient a |
| amz_B09QY2H8SG_147512 | amazon_competitor | — | 0 | Won’t connect to my Wi-Fi I am not able to use it at all |
| tp_5d7a93b53585c706fc11669a_83177 | trustpilot | — | 0 | die Lieferung ist sehr schnell und…. die Lieferung ist sehr schnell und unkompliziert. Ich bin überr |
| tp_6954d4e2fcc2740e8008f037_91219 | trustpilot | — | 0 | Beaucoup de produits a petits prix Beaucoup de produits a petits prix. Du choix |
| amz_B08QR6V8WR_144923 | amazon_competitor | — | 0 | Los use para mi hija y son lo máximo y los compro para regalo porque se que lo van  a usar en los be |
| zd_1105618_15031 | zendesk | — | 0 | Conversation with Web User 69b81fada30708375e769432 |
| amz_B09C8N75VM_183501 | amazon_competitor | — | 0 | La rebaja la grasa y sudor del cuero |
| amz_B096LRSJCS_146820 | amazon_competitor | — | 0 | Excelente el ajuste,cómodo y buen material. Ahora solo sirve para lactar o no para extraer con la má |
| amz_B078C7CGCL_74103 | amazon_competitor | — | 0 | Mi experiencia:Compré estos audífonos con cancelación de ruido para proteger los oídos de mi bebé en |
| tp_668c98feacd156c4fb7056b3_81220 | trustpilot | — | 0 | Easy ordering Easy ordering. Fast shipment. |
| tp_5f742add798e6f09602344fb_52555 | trustpilot | — | 0 | Malhonnête, passez votre chemin ! Malhonnête ! j'ai passé une commande le 2 Août, sur le suivi clien |
| rd_RPY5OMHAUE6IE_2058 | reddit | — | 0 | Bastante decepcionada con el producto. Lo compré el 14 de enero y a día de hoy 2 de febrero ya no fu |
| zd_1012648_6203 | zendesk | — | 0 | Conversation with Web User 698a22d276a5da15cc912953 |
| zd_1086107_46919 | zendesk | — | 0 | Bonjour,  j’ai passé commande il y a une semaine et elle est toujours en attente de préparation.  J’ |
| tp_6863ed5116f1e5edaa309b4f_74567 | trustpilot | — | 0 | Reklamere for hurtig forsendelse Reklamere for hurtig forsendelse, de sender ikke hurtigt, og svarer |
| amz_B07S736G1H_183918 | amazon_competitor | — | 0 | Linda y con espectacular compresión |
| amz_B07ZLSQXCS_154254 | amazon_competitor | — | 0 | I like how it looks specially with jeans. I will buy more colors. |
| amz_B07Q3XFZC8_148029 | amazon_competitor | — | 0 | There is no specific "tummy control". The entire cami is constricting, so it also flattens your brea |
| amz_B09MWJB3LF_88045 | amazon_competitor | — | 0 | Purchased to have on hand for new grand son to come visit. Folds up for easy storage. Seems to works |
| amz_B0DRCXZM4F_194766 | amazon_competitor | — | 0 | La mejor faja q e probado y q vale la pena comprar sin mirar su precio la verdad me encantó |
| amz_B0CKYFBH55_117905 | amazon_competitor | — | 0 | Ease of use warmth is a plus it does what you set it on it makes life easier for me also value for m |
| amz_B07QPD1PLD_174614 | amazon_competitor | — | 0 | Tela suave se ajusta cómodamente al cuerpo. Ideal para usar con pantalones blancos o vestidos porque |
| amz_B0FM7DSD4Z_161417 | amazon_competitor | — | 0 | Très bon babyphone, l’image est ultra propre avec la caméra 2K et l’écran partagé permet de surveill |
| amz_B08GHYDS7Z_176504 | amazon_competitor | — | 0 | My daughter needed it .she sleeps with it all the time.. it's supports her whole body |
| amz_B09HJDX5S6_66923 | amazon_competitor | — | 0 | This product serves its function and has saved up a ton of time. Only CON: Although there are multip |
| tp_5e35330e3c93ae04c0d80396_87616 | trustpilot | — | 0 | Site qui beug impossible de passer une… Site qui beug impossible de passer une commande depuis 1 sem |
| amz_B09ZD6LK9K_100766 | amazon_competitor | — | 0 | I only use this periodically and get so frustrated when I do because I always forget the magic steps |
| tp_6887c083938dda96575526da_92752 | trustpilot | — | 0 | I have to pay for return I have to pay for return. Thats ridiculous. |
| rd_R15X1WIK7I73HN_1294 | reddit | — | 0 | Ils sont très bien mais il est important de savoir qu’ils ne sont pas fait pour un usage exclusif, j |
| zd_1062157_14751 | zendesk | — | 0 | Bonjour,  J’ai commander un porte bébé Move2fit mais je n’avais pas remarqué qu’il été grand.  Possi |
| zd_1078167_16433 | zendesk | — | 0 | Conversation with Web User 69ab00a238f0f4072aed5e67 |
| tp_67a77a145a4ad73fc7aa804a_46279 | trustpilot | — | 0 | Love all my things Love all my things. ♥️ |
| tp_5d1f5ebe42fa560424877b92_87577 | trustpilot | — | 0 | L'article que je veux est en rupture de… L'article que je veux est en rupture de stock depuis 6 mois |
| amz_B0CC5BYVDL_167436 | amazon_competitor | — | 0 | La tela es muy suave, son cómodas y ligeras y amoldan excelente, de hecho no necesito usar sostén co |
| rd_753441_2196 | reddit | — | 0 | Call from: +1 (815) 354-1415 Call to: +1 (619) 848-0676 Time of call: October 8, 2025 at 5:42:11 PM  |
| zd_949100_40633 | zendesk | — | 0 | Conversation with Web User 696796dd2339e9ac7fb46fa2 |
| tp_67617baad1e27dda187fea59_62098 | trustpilot | — | 0 | Choix, prix, rapidité de la livraison ++ Le site offre un large choix d'articles, à des prix intéres |
| amz_B07FRYYZ3N_161534 | amazon_competitor | — | 0 | Muy suaves y perfectas para la piel delicada del bebéEstas toallitas húmedas Huggies Natural Care ha |
| zd_1078308_12729 | zendesk | — | 0 | Conversation with Web User 69ab0c0ba9faf8c4855cf941 |
| mz_R2UH7H1ADJPP04_17770 | momcozy | — | 0 | It did not fit on my Maxi Cosi set |
| amz_B0DPQ44BJL_194254 | amazon_competitor | — | 0 | My Pulmonary doctor and the FDA do NOT recommend using this for sanitizing my C -pap equipment. |
| amz_B0DR1ZZH49_135024 | amazon_competitor | — | 0 | Honestly the first item I have purchased on AMAZON that exceeded the product specifications.  I am l |
| zd_969622_47883 | zendesk | — | 0 | Good morning,  Can you please cancel my order JJ1417997? This was ordered on 1/16 and has not yet be |
| zd_1125858_16463 | zendesk | — | 0 | Conversation with Web User 69c1db114a12ddb6873b462a |
| zd_1045510_11791 | zendesk | — | 0 | Conversation with Web User 699ae293fbc06a57495f4aee |
| tp_67b08c323f7896127c68cdd3_93994 | trustpilot | — | 0 | Fast and the order was correct Fast and the order was correct |
| amz_B073WJDQMN_178410 | amazon_competitor | — | 0 | BUY THE LARGER UNIT, IT LASTS LONGER AND FILTERS MUCH BETTER. |
| amz_B09TR3XX3V_173702 | amazon_competitor | — | 0 | My newborn needed these for the Fourth of July I got them just in time when my kid of a husband lit  |
| mz_RHYGMCXRLUGJV_17632 | momcozy | — | 0 | No me gustó no hace nada, no hidradrata, no cura, no hace nada, también tiene un olor neutro medio f |
| tp_66c71a21ea4377e6dd92a4b7_92823 | trustpilot | — | 0 | Sehr schöner Shop Sehr schöner Shop, unkompliziert und zuverlässig |
| amz_B0BFKNQS6S_165138 | amazon_competitor | — | 0 | It looks very small but stretches to fit. |
| tp_6059fab8f85d7507e41e869c_66346 | trustpilot | — | 0 | I sent the dress as a gift I sent the dress as a gift and the recipient loved it! |
| zd_1084460_47379 | zendesk | — | 0 | Bonjour, j’ai commandée le 5 mars l’écharpe porte bebe sur votre site sauf qu’à ce jour je n’ai plus |
| zd_918250_2070 | zendesk | — | 0 | Bonjour je viens de faire une commande je me suis trompée sur l'adresse c'est 108 avenue du Maréchal |
| rd_R22GKUPK3A9JGS_5554 | reddit | — | 0 | Battery doesn’t last as long as they claim |
| tp_6968b4a903c2b031c5457633_51554 | trustpilot | — | 0 | Still waiting on the item I ordered in… Still waiting on the item I ordered in November (now late Ja |
| tp_666833b0722454b5b38efdf8_71439 | trustpilot | — | 0 | We love Jovi goat formula! We love Jovi goat formula! |
| mz_R2CX7K0THKU8XR_13875 | momcozy | — | 0 | They are ok but huge would be nice if they had smaller size.I’m a B cup.. |
| zd_1082655_34427 | zendesk | — | 0 | Witam, złożyłam u Państwa ZAMÓWIENIE #JJ1532259, lecz przez przypadek nie podałam pełnego adresu dos |
| zd_1139392_6096 | zendesk | — | 0 | Hola ,  mi aspiradora nasal para bebé si enciende pero no aspira , me pueden ayudar que puedo hacer  |
| zd_1015050_37160 | zendesk | — | 0 | Hello,  I’m following up for clarity on the processing timeline.  I was advised that this order woul |
| tp_65942d57eda687522e189c9b_76886 | trustpilot | — | 0 | Parfait Parfait ! Heureuse de porter la marque et d’allaiter mon bébé partout tout le temps |
| tp_6255b64bc7628b203baa977c_69125 | trustpilot | — | 0 | Achat d’un lit mezzanine pour ma fille… Achat d’un lit mezzanine pour ma fille de 10 ans, je suis ra |

### 4.3 `generic_only` （零标签 且 去停用词后有效 token < 3）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| amz_B08QW89LFS_110886 | amazon_competitor | 喂养电器 | 0 | The only pacifier both my sons loved and used! |
| zd_1135887_28589 | zendesk | 吸奶器 | 0 | Can I cancel my order thank you |
| mz_R17JX8BPZ2USQ2_19163 | momcozy | 吸奶器 | 0 | This did not come with the duck valves. |
| amz_B0B7MFS2W4_132743 | amazon_competitor | 家居家纺 | 0 | It’s the best stroller I really like it |
| amz_B09FPLGYF2_134653 | amazon_competitor | 智能母婴电器 | 0 | It fits everywherePracticalLightweightComfortable |
| zd_946595_24621 | zendesk | 吸奶器 | 0 | Conversation with Monica Muñiz |
| mz_R12EY0V8BG9NLA_19621 | momcozy | 吸奶器 | 0 | 純正品なのでサイズピッタリでした。なんの問題も無く使用出来ています。持ち運びには必須アイテムです |
| amz_B09XJJB478_140899 | amazon_competitor | 喂养电器 | 0 | The only pacifier my grandson will use. |
| amz_B0DJVFX2KY_141278 | amazon_competitor | 智能母婴电器 | 0 | This humidifier its really good. I recommend as very good one. |
| amz_B07TDN9MKC_149031 | amazon_competitor | 喂养电器 | 0 | All of my babies have loved these pacifiers! |
| amz_B0057R47NW_151516 | amazon_competitor | 内衣服饰 | 0 | Product came dirty and used. Would like a replacement. |
| amz_B0BZJ82N7W_151987 | amazon_competitor | 家居家纺 | 0 | I like them but they don’t have pockets as described |
| amz_B0CR35MZ98_152088 | amazon_competitor | 内衣服饰 | 0 | Best bra from all of those that i have |
| amz_B07M98L1GK_152773 | amazon_competitor | 家居家纺 | 0 | Bed was nice,but it did not come with mattress. |
| amz_B089QJ6X3X_154166 | amazon_competitor | 家居家纺 | 0 | We love our wagon! It definitely gets used! |
| mz_RGYGQ2IDL95PL_19439 | momcozy | 吸奶器 | 0 | 育児バッグの中でもかなり優秀な部類でした。まず、ハードシェルなので中身をしっかり守ってくれる安心感があり、搾乳機を2台入れても余裕のある設計。取り外し可能なトレイが想像以上に便利で、整理整頓がラクにで |
| zd_948756_21601 | zendesk | 吸奶器 | 0 | 使い始めてまだ１ヶ月くらいですが 片方が急に動かなくなりました。 交換をお願いしたいです   iPhoneから送信 |
| amz_B00A282SGO_157836 | amazon_competitor | 喂养电器 | 0 | These are the only pacifiers I can get my baby to use. |
| zd_986769_28533 | zendesk | 吸奶器 | 0 | Conversation with VictoriaCarman |
| mz_R2CR04F45V7I2I_19467 | momcozy | 吸奶器 | 0 | 専用設計なので、搾乳機がケースの中で暴れず、収まりが非常に良いです。ですが、安い搾乳機と引けを取らないくらいの価格帯でケースのみなので、購入をしにくいのは難点。 |
| zd_930820_27093 | zendesk | 吸奶器 | 0 | Conversation with Ayşe Kocaman |
| amz_B07CYRYX3R_161857 | amazon_competitor | 喂养电器 | 0 | Only pacifiers my baby will use! |
| amz_B0DXTH8BW7_162502 | amazon_competitor | 喂养电器 | 0 | Great wipe warmer! 10/10 recommend |
| amz_B01MYGNGKK_164187 | amazon_competitor | 智能母婴电器 | 0 | I really like the functions of this humidifier. |
| zd_920160_36533 | zendesk | 吸奶器 | 0 | Abigail S. DiSalvo (203)-885-9313 |
| amz_B09538RHV4_165428 | amazon_competitor | 吸奶器 | 0 | I really like it to use the pump with. |
| zd_983755_24617 | zendesk | 吸奶器 | 0 | Conversation with KaiaDeLaRosa |
| amz_B09QHW8CKK_169524 | amazon_competitor | 家居家纺 | 0 | It would be good if the pillow is firmer. |
| amz_B07RGPYHJJ_169583 | amazon_competitor | 喂养电器 | 0 | My two daughters have loved these pacifier. |
| mz_R2YHYZRE1UU67D_19392 | momcozy | 吸奶器 | 0 | 約9000円と高いだけあり、造りとしては非常にしっかりしています。搾乳機が2つ入る構造をしていて、取っ手を持つことで楽に携帯可能なデザイン。またその際には製品ロゴが見えるようになっています。キャリーケ |
| amz_B0BHT3DQ8D_170215 | amazon_competitor | 智能母婴电器 | 0 | Love it! This is a great air purifier. |
| amz_B08JYJKZNL_170265 | amazon_competitor | 家居家纺 | 0 | Good product, everything as described 👍 |
| amz_B0DTHSRTD4_170340 | amazon_competitor | 智能母婴电器 | 0 | Very good purifier for a 10x12 room. |
| zd_1055792_28535 | zendesk | 吸奶器 | 0 | Conversation with SelenaDagher |
| amz_B08PVXSH5N_170892 | amazon_competitor | 家居家纺 | 0 | Very nice stroller and works good |
| zd_933297_28531 | zendesk | 吸奶器 | 0 | Conversation with KylieMccomber |
| mz_R8E2DLJ2IQRIZ_19433 | momcozy | 吸奶器 | 0 | 搾乳機をしっかり守って持ち運ぶ事に重きを置いている為、形状が大きく、かさばります。その分、防水性は非常に高く、中を開けると仕切りが多く整理しやすいです。互換性が高いので他の製品でも使えると思います。追 |
| zd_930730_27084 | zendesk | 吸奶器 | 0 | Hi when will this order arrive? |
| amz_B08PMJBN33_171660 | amazon_competitor | 家居家纺 | 0 | My granddaughter loved the crib |
| amz_B09C8MNW9S_173299 | amazon_competitor | 内衣服饰 | 0 | This lacked compression.  I will not re purchase. |
| zd_1026992_27047 | zendesk | 吸奶器 | 0 | My order says it was delivered and it’s not |
| mz_R1OGOFWR2JYVKQ_19424 | momcozy | 吸奶器 | 0 | 旅行、帰省などで搾乳器の持ち運びとして活用するために注文しました。機械が守れるので故障する心配なく助かります。ただちょっと大きすぎますね。。もう少しコンパクトだと嬉しいです。ですが他のパーツや充電器も |
| zd_1028758_21532 | zendesk | 吸奶器 | 0 | Conversation with Jo Jo Lovell |
| amz_B08QR6V8WR_176019 | amazon_competitor | 母婴综合护理 | 0 | These are the only wipes we use at our house |
| amz_B01LY80MNC_176097 | amazon_competitor | 智能母婴电器 | 0 | I Really Like The Sturdiness Of This Bassinet |
| amz_B0995JN3NJ_176752 | amazon_competitor | 家居家纺 | 0 | This product is nothing like described |
| amz_B07WPC88Y9_176965 | amazon_competitor | 吸奶器 | 0 | This was really helpful when I was pumping! |
| amz_B0BJJXJSCN_178066 | amazon_competitor | 家居家纺 | 0 | We use this more than a stroller |
| zd_969273_21598 | zendesk | 吸奶器 | 0 | もう片方と部品を一つずつ変えてみたり、違うサイズのニップルで試してみても吸引せず  モーターを変えたら吸引できたので、片方のモーターの不具合だと思います。  同一商品との交換をお願いします。 色が他の |
| zd_1014071_27056 | zendesk | 吸奶器 | 0 | Conversation with KristenAlexander |
| amz_B09K39Y11L_179803 | amazon_competitor | 家居家纺 | 0 | She liked the pillow. Now to get her pregnant... |
| amz_B00BVUQEWG_180457 | amazon_competitor | 喂养电器 | 0 | My baby only use these pacifiers. |
| zd_1122199_27072 | zendesk | 吸奶器 | 0 | Conversation with RaeganMueller |
| zd_977241_36557 | zendesk | 吸奶器 | 0 | Conversation with Patrick1010!! |
| zd_1020398_27090 | zendesk | 吸奶器 | 0 | Conversation with SergiCardoza |
| zd_1137196_27137 | zendesk | 吸奶器 | 0 | الصفحه دي بتقول انها توكيل ممكوزي في مصر ؟؟  هل ده حقيقي ؟؟ |
| zd_992356_27079 | zendesk | 吸奶器 | 0 | Conversation with NelidaGonzalez |
| amz_B00P7FJM48_183185 | amazon_competitor | 吸奶器 | 0 | This is a good pumping bra! !!!! |
| amz_B09539M877_183510 | amazon_competitor | 吸奶器 | 0 | Definitely buy it if you plan to pump |
| amz_B0B8SGV9L4_183517 | amazon_competitor | 喂养电器 | 0 | Only received 5 of 6 of the pacifiers |
| zd_1040055_27061 | zendesk | 吸奶器 | 0 | Conversation with AllyFriedenberger |
| zd_1026539_21502 | zendesk | 吸奶器 | 0 | Conversation with N Λ T Λ L ⋮ Λ |
| zd_1019606_27033 | zendesk | 吸奶器 | 0 | Conversation with MorganHarrelson |
| amz_B07Y3Z86DB_183901 | amazon_competitor | 内衣服饰 | 0 | More compression would be better. |
| amz_B0BMFXSDRN_184295 | amazon_competitor | 内衣服饰 | 0 | Very thin, no compression at all |
| zd_1025068_27080 | zendesk | 吸奶器 | 0 | Conversation with Yasmin 🥰🫶🏻😇🙏🏻 |
| amz_B07QKG7BD5_184660 | amazon_competitor | 家居家纺 | 0 | They were exactly as described |
| zd_966862_37019 | zendesk | 吸奶器 | 0 | Hi, I want the refund.  order 1409307 |
| zd_1080110_28525 | zendesk | 吸奶器 | 0 | Conversation with BriannaMitchell |
| amz_B0BB2CD4RN_195159 | amazon_competitor | 家居家纺 | 0 | Was not good for my infants teeth |
| tp_682772be79f9307e628e8cca_72195 | trustpilot | 智能母婴电器 | 0 | Really good product Love how light the stroller is |
| zd_932042_39666 | zendesk | 吸奶器 | 0 | Hi I would like to to cancel my order   JJ1402104  Please |
| zd_929510_1200 | zendesk | 吸奶器 | 0 | Conversation with JessicaThompson |
| zd_1129054_28672 | zendesk | 吸奶器 | 0 | Conversation with CassidyLasick |
| zd_949266_1218 | zendesk | 吸奶器 | 0 | Conversation with MelissaRinehart |
| zd_1072141_27088 | zendesk | 吸奶器 | 0 | Conversation with MélodieSabra |
| zd_1086926_1224 | zendesk | 吸奶器 | 0 | Conversation with KaylahBarriger |
| zd_1086070_28540 | zendesk | 吸奶器 | 0 | Conversation with SelenaDagher |
| zd_918399_27059 | zendesk | 吸奶器 | 0 | Conversation with CatherineStewart |
| zd_1003882_1229 | zendesk | 吸奶器 | 0 | Conversation with JazminSevilla |
| mz_R37ZO13P17LEC1_19425 | momcozy | 吸奶器 | 0 | 搾乳機は使っていないので小さめのカップなどの洗浄に使っていますがやはり用途が違いサイズが短いため洗いにくいのが本音です。品質的には持ち手などはしっかりしているもののスポンジ部分はもう一つかな…本来の使 |
| zd_922464_1243 | zendesk | 吸奶器 | 0 | Conversation with RebekahPaine |
| zd_954725_1409 | zendesk | 吸奶器 | 0 | Conversation with KalissiaClark |
| zd_1089536_10150 | zendesk | 喂养电器 | 0 | When I will receive my order my order number is JJ1532515 |
| zd_927745_38952 | zendesk | 吸奶器 | 0 | Conversation with BrookeMcClain |
| zd_931743_10867 | zendesk | 喂养电器 | 0 | I was wondering when will my order be shipped?  #JJ1393147 |
| zd_1100441_12310 | zendesk | 喂养电器 | 0 | hi,I would like to know when my order will be shipped? |
| zd_991647_24597 | zendesk | 吸奶器 | 0 | My order number is .   JJ1443656 |
| zd_1067569_12544 | zendesk | 喂养电器 | 0 | Conversation with CadieCampbell |
| zd_1067762_12550 | zendesk | 喂养电器 | 0 | Conversation with Sinderella 🤍 |
| zd_1132267_14373 | zendesk | 喂养电器 | 0 | Conversation with ChanellCericeee |
| zd_1023795_15566 | zendesk | 智能母婴电器 | 0 | Conversation with @karenagred ❤ |
| zd_1087047_16544 | zendesk | 内衣服饰 | 0 | Conversation with brandihutchings |
| zd_1138847_16550 | zendesk | 内衣服饰 | 0 | Conversation with JewelOlbrantz |
| mz_R2ZYS2LCE07FKU_16780 | momcozy | 内衣服饰 | 0 | My friend loved this bra that I bought her |
| zd_1086717_20529 | zendesk | 吸奶器 | 0 | Hi I have a problem with my pumps |
| zd_960883_27052 | zendesk | 吸奶器 | 0 | Conversation with StefanieTousignant |
| zd_1018395_21485 | zendesk | 吸奶器 | 0 | Conversation with George...😎... 🌊 |
| zd_1081770_21494 | zendesk | 吸奶器 | 0 | Conversation with melaniefouquet98 |
| zd_1093158_21501 | zendesk | 吸奶器 | 0 | Conversation with د. فهد الشرهان |

## 五、阈值判定（决策 4）

| 指标 | 阈值 | 实测 | 结果 |
|---|---:|---:|:---:|
| 原始覆盖率 | 88.00% | 80.80% | 🔴 FAIL |
| 业务有效覆盖率 | 94.00% | 90.88% | 🔴 FAIL |

## 六、与 Phase 4 对比

- Phase 4 5K 子集原始覆盖率：82.58%
- Phase 5 全量原始覆盖率：80.80% （-1.78pp）
- Phase 5 全量业务有效覆盖率：90.88%
