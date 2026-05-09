---
name: phase5-d10-dual-coverage-report
description: Phase 5 D10 双覆盖率审计报告（原始 / 广义 / 业务有效）。当审计 Phase 5 整体覆盖率达成、与 Phase 4 对比、查看排除分布与抽样时使用。
computed_at: 2026-05-09T22:41:43
doc_type: audit-report
module: voc-nlp
status: stable
---

# Phase 5 D10 双覆盖率审计报告

**输入**：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d8_final.jsonl`
**总样本**：364,569
**计算时间**：2026-05-09T22:41:43

## 一、核心指标

| 指标 | 分子 | 分母 | 覆盖率 | Phase 4 基线 | Δ |
|---|---:|---:|---:|---:|---:|
| 原始覆盖率 | 299,918 | 364,569 | **82.27%** | 82.58% | -0.31pp |
| 广义覆盖率（含品牌命中）| 299,924 | 364,569 | **82.27%** | — | — |
| 业务有效覆盖率 | 299,918 | 322,236 | **93.07%** | — | — |

> 业务有效分母 = 总数 - 排除（42,333 条，占比 11.61%）

## 二、排除桶分布

| 桶 | 定义 | 样本数 | 占总比 |
|---|---|---:|---:|
| `too_short` | text.strip() < 30 字符 | 3,519 | 0.97% |
| `off_category` | product_line 空 且 零标签 且 无品牌 mentions | 38,658 | 10.60% |
| `generic_only` | 零标签 且 去停用词后有效 token < 3 | 156 | 0.04% |

## 三、按数据源拆解

| data_source | 总数 | with_label | 原始覆盖率 | brand_only | 排除合计 | 业务有效覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| amazon_competitor | 194,734 | 172,670 | 88.67% | 5 | 14,325 | 95.71% |
| trustpilot | 99,853 | 80,008 | 80.13% | 0 | 17,336 | 96.96% |
| zendesk | 47,204 | 29,555 | 62.61% | 0 | 8,579 | 76.52% |
| momcozy | 19,808 | 15,609 | 78.80% | 1 | 1,591 | 85.68% |
| reddit | 2,970 | 2,076 | 69.90% | 0 | 502 | 84.12% |

## 四、排除追踪抽样（每桶最多 100 条，供人工 spot check）

### 4.1 `too_short` （text.strip() < 30 字符）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| mz_R3LYAX05UGTE27_19243 | momcozy | — | 0 | Me alivio rápido los pezones |
| zd_1059620_2379 | zendesk | — | 0 | Conversation with Angela 🍭 |
| amz_B0CBRWH758_192956 | amazon_competitor | — | 0 | No me gustó |
| amz_B0C6Q4JXKT_184780 | amazon_competitor | — | 0 | Worked for about 4 months. |
| amz_B07QXL254R_171771 | amazon_competitor | — | 0 | My granddaughter loved these |
| zd_1027266_416 | zendesk | — | 0 | Sent from my iPhone |
| amz_B0CBT9R4FM_126055 | amazon_competitor | — | 0 | Nice fit and I wanted pockets |
| amz_B0834M6TRN_189890 | amazon_competitor | — | 0 | Excelente me encanta |
| amz_B0873B22Q3_190485 | amazon_competitor | 内衣服饰 | 0 | By far my favorite bra!! |
| amz_B0CQKZ14BG_192027 | amazon_competitor | — | 0 | My grand-baby loves it |
| mz_RKLB8NFCDPMB4_19269 | momcozy | — | 0 | My little one lives these |
| tp_6197f60e8931a3308730ac02_71707 | trustpilot | — | 0 | Good service Good service |
| amz_B08KHW6C49_193831 | amazon_competitor | 母婴综合护理 | 0 | Horrible wipes |
| amz_B0C691MHD6_194026 | amazon_competitor | — | 0 | They loved it |
| zd_978734_6251 | zendesk | — | 0 | Conversation with Ahmed Hamid |
| amz_B0C27L4JW6_190447 | amazon_competitor | — | 0 | My daughter loves them! |
| amz_B08QRL9YMY_184895 | amazon_competitor | — | 0 | Same as title hereinabove! |
| amz_B0C1Y84QPK_193366 | amazon_competitor | — | 0 | Good back support |
| amz_B0BFNF82MG_194363 | amazon_competitor | — | 0 | thanqyu good |
| zd_1066756_36539 | zendesk | 吸奶器 | 0 | Conversation with Tanz Cheema |
| amz_B0CYZPJMJR_190929 | amazon_competitor | — | 0 | Really good product👍 |
| zd_1092847_146 | zendesk | — | 0 | Sent from my iPhone |
| amz_B0F62NYXVR_194316 | amazon_competitor | — | 0 | She loves it |
| amz_B0DHH2G85H_190652 | amazon_competitor | — | 0 | Very nice & works good |
| amz_B07VPKTRZ7_185157 | amazon_competitor | — | 0 | Bon produit très pratique |
| amz_B07KF71M2X_140602 | amazon_competitor | — | 0 | Size up, smooth fit and airy |
| amz_B073JYK4JT_192907 | amazon_competitor | — | 0 | Love iiiit!!! ❤️❤️❤️ |
| zd_1006954_38976 | zendesk | 吸奶器 | 0 | Conversation with Divya M |
| amz_B0779Z53SD_141400 | amazon_competitor | — | 0 | Very good size and quality |
| amz_B07FT1VMB4_189947 | amazon_competitor | — | 0 | Size was the right fit |
| amz_B0CGV466MS_179182 | amazon_competitor | 内衣服饰 | 0 | the legs rollup like panty |
| zd_1054461_1261 | zendesk | 吸奶器 | 0 | help me with my pumps |
| amz_B01FCDWZFC_184404 | amazon_competitor | — | 0 | Muy bueno, excelente producto |
| amz_B0BLVHBLRP_192513 | amazon_competitor | — | 0 | just what i wanted |
| zd_930446_38968 | zendesk | 吸奶器 | 0 | Conversation with Rinieshah |
| amz_B0B42BRHZ9_180748 | amazon_competitor | — | 0 | These work as they should. |
| zd_1109469_47785 | zendesk | — | 0 | Conversation with M. Othman |
| amz_B0BFNH9694_181654 | amazon_competitor | — | 0 | Buy yours before it sells out |
| zd_1056592_21583 | zendesk | 吸奶器 | 0 | Conversation with Sara GF |
| amz_B07W961P88_184519 | amazon_competitor | — | 0 | Relief from alot of back pain |
| amz_B0FH7N8PT6_188763 | amazon_competitor | — | 0 | Working good so far. |
| amz_B00FNZQHJA_195088 | amazon_competitor | — | 0 | La mejor sin duda alguna |
| amz_B0C9ZS159B_193194 | amazon_competitor | — | 0 | My wife loved this. |
| amz_B0D942JHP9_192008 | amazon_competitor | — | 0 | Our baby loves these! |
| amz_B0FRS9BC2P_194224 | amazon_competitor | — | 0 | Good product |
| amz_B0DNYK6V1B_185139 | amazon_competitor | — | 0 | Funktioniert einwandfrei! |
| amz_B0CW5B8PKS_189664 | amazon_competitor | — | 0 | Muy buenas , me encantan |
| amz_B0DXQ3MKQ2_191011 | amazon_competitor | — | 0 | it works really good |
| amz_B07WG7H7VT_172370 | amazon_competitor | 家居家纺 | 0 | Im in love with this pillow |
| zd_1117531_4714 | zendesk | — | 0 | Conversation with LisaKameo |
| amz_B07KF61W2N_195285 | amazon_competitor | — | 0 | Didn’t meet standards |
| amz_B0DFWZKKRG_171377 | amazon_competitor | — | 0 | It was giving me back pain |
| mz_R29C20WJ53NQ9F_19267 | momcozy | — | 0 | Muy buena quedé encantada |
| zd_1050938_24920 | zendesk | 吸奶器 | 0 | Conversation with Pablo Nogue |
| tp_65a95d5b7d03dcab845bda87_98114 | trustpilot | — | 0 | 5 stars ⭐️ 😊😊😊 5 stars ⭐️ 😊😊😊 |
| amz_B0DCBDR1Q8_159491 | amazon_competitor | — | 0 | Quality and just plain nice |
| amz_B0D3LRXBBJ_193735 | amazon_competitor | — | 0 | Nice headphones |
| amz_B07W961P88_194304 | amazon_competitor | — | 0 | Good product |
| amz_B014A7MB4G_192518 | amazon_competitor | — | 0 | Drawers are not solid |
| amz_B0CBRTNV7T_172504 | amazon_competitor | — | 0 | definitely snatched me up!! |
| amz_B0C7W6WVMN_194623 | amazon_competitor | — | 0 | Works good |
| amz_B0DF2CGMSC_195092 | amazon_competitor | — | 0 | Reduced a lot of burden. |
| amz_B00BPE9M5E_195266 | amazon_competitor | 家居家纺 | 0 | Exactly as described. |
| amz_B099NCDLT3_195591 | amazon_competitor | — | 0 | Needs more stuffing. |
| amz_B0D963T56C_190943 | amazon_competitor | — | 0 | My daughter loves it |
| amz_B000KK2C6A_193532 | amazon_competitor | — | 0 | Loved! This bed ! |
| tp_67dd2ffb3fca746281886b3e_93951 | trustpilot | — | 0 | Quick service. Quick service. |
| amz_B0BHBSGS89_166093 | amazon_competitor | — | 0 | It helps with my breathing. |
| zd_1065812_33126 | zendesk | — | 0 | Conversation with Nick James |
| amz_B0DHGP2JQ8_194321 | amazon_competitor | — | 0 | Easy set upj |
| zd_1075193_32794 | zendesk | — | 0 | Conversation with Marisol |
| amz_B0B42H2HGF_185009 | amazon_competitor | — | 0 | Es fuerte y fácil de armar |
| zd_970460_41401 | zendesk | — | 0 | When will my order arrive ? |
| amz_B07Y43SHHN_192656 | amazon_competitor | — | 0 | Rides up my ass. |
| amz_B0BFX2ZKLP_195060 | amazon_competitor | — | 0 | It’s just not snatching |
| amz_B0B4PVWK6T_192050 | amazon_competitor | — | 0 | my baby hated these🥲 |
| amz_B0B8HDNSHB_194758 | amazon_competitor | — | 0 | Better than expected |
| tp_66ae59ffa02aba860609b05a_99481 | trustpilot | — | 0 | ............. ............. |
| amz_B0B4BP458Z_189559 | amazon_competitor | — | 0 | 100% recommend! |
| zd_1134141_38979 | zendesk | 吸奶器 | 0 | Sent from my iPhone |
| amz_B0C5B1C743_192057 | amazon_competitor | — | 0 | Worked good for baby |
| amz_B0FP2Y2LJN_172419 | amazon_competitor | — | 0 | Makes cleaning bottles easy |
| amz_B0D8W1SF8S_166347 | amazon_competitor | — | 0 | Es un buen materia me encantó |
| zd_1017439_21529 | zendesk | 吸奶器 | 0 | Conversation with Sue Howell |
| amz_B09GJKZTWM_193332 | amazon_competitor | 母婴综合护理 | 0 | Really good wipes |
| zd_1124379_43126 | zendesk | — | 0 | I have not received my order |
| zd_1055426_39710 | zendesk | 吸奶器 | 0 | Conversation with GraceMarien |
| amz_B0CBT8V717_195178 | amazon_competitor | — | 0 | It’s not the long one. |
| tp_69cfbf37cfef094f9bfb1986_98311 | trustpilot | — | 0 | Jamais déçue Jamais déçue |
| amz_B086P8NJF4_184649 | amazon_competitor | — | 0 | Takes a long time to get warm |
| amz_B07CYVCQBC_180668 | amazon_competitor | — | 0 | The baby likes these most. |
| amz_B09WY63268_172658 | amazon_competitor | — | 0 | It seems to do a good job |
| zd_1048263_43487 | zendesk | — | 0 | Conversation with Dilpreet |
| amz_B0BS94NLY6_184545 | amazon_competitor | — | 0 | Very heavy and unmanageable. |
| amz_B07KF61W2N_173846 | amazon_competitor | — | 0 | didn’t work for me at all |
| amz_B09NJNPRC7_184968 | amazon_competitor | — | 0 | No they we not tight enough |
| amz_B07CZTLXG2_192133 | amazon_competitor | — | 0 | My baby loves them |
| zd_1012418_41508 | zendesk | — | 0 | Conversation with Sanata Dama |
| amz_B00LZKC1S8_192245 | amazon_competitor | — | 0 | Baby loves it. |
| amz_B0FJXZXCVL_187711 | amazon_competitor | — | 0 | It fits me very well. |

### 4.2 `off_category` （product_line 空 且 零标签 且 无品牌 mentions）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| tp_5ac4c6ce6d33bc08ec884f42_58216 | trustpilot | — | 0 | Pas de problème sur les produits… Pas de problème sur les produits commandés en revanche il serait b |
| zd_953125_32223 | zendesk | — | 0 | Conversation with Web User 69695e95493735f106e4415b |
| tp_633c02c14a5fff53cad5ea0d_94638 | trustpilot | — | 0 | Équipe Formidable Très bonne expérience dans la boutique |
| zd_1071363_32474 | zendesk | — | 0 | Conversation with Web User 69a8053e56359ebae79be087 |
| tp_5aae29a3d5a5700a48a7e3fa_96260 | trustpilot | — | 0 | Rapidité d'expédition et produit… Rapidité d'expédition et produit conforme à ce que je voulais. Plu |
| tp_611f8a786e38167b798bb8de_44790 | trustpilot | — | 0 | Praktisch und beständig Praktisch, schöne Motive, Formen und Farben. Halten sogar der Spülmaschine s |
| tp_69119b1047cfb1174278dcac_62942 | trustpilot | — | 0 | J'ai beaucoup apprécié la qualité et la… J'ai beaucoup apprécié la qualité et la rapidité de mes inv |
| amz_B0DDKNFVCC_171005 | amazon_competitor | — | 0 | Calidad excelente Tiene Una compresión muy buena en el vientre no se marca ✔️ Me gustó mucho q Tiene |
| tp_5b825cad8c83fd0b58d069b2_92095 | trustpilot | — | 0 | Der Versand hat extrem lange gedauert Der Versand hat extrem lange gedauert |
| amz_B08MDW2H86_158580 | amazon_competitor | — | 0 | I'm usually a XL I sized down to a medium and it did what he needed to do |
| amz_B00LZKC1S8_149983 | amazon_competitor | — | 0 | Baby loves.  Didn't like the orange piece around the neck of the elephant.  Didn't do well when clea |
| tp_64cf78585edd9094e48777bc_73966 | trustpilot | — | 0 | Übersichtlichkeit Die Seite sollte für mich übersichtlich sein, nicht zu umständlich und das ist hie |
| amz_B0C696JYCL_166941 | amazon_competitor | — | 0 | Long-term use, there is no adverse reaction, very reassuring. |
| tp_5ad76f1d6d33bc0a240059c4_74127 | trustpilot | — | 0 | Nickel Produit qu’il me fallait. Réception très rapide |
| amz_B0D12T33P7_149816 | amazon_competitor | — | 0 | i Accidentally fell asleep with it on and it wasn’t bothering me at all best faja ever |
| tp_6396c4130b7fc02715dc01cc_69158 | trustpilot | — | 0 | facilité commande Paiement facile. Adresse facturation/livraison différente possible Aurez-vous un j |
| amz_B09LG36SMN_178502 | amazon_competitor | — | 0 | Me encantó, y de buen material |
| tp_62e7afdd4c35e69ec51fc98f_84866 | trustpilot | — | 0 | Toujours ravie des produits proposés Toujours ravie des produits proposés, aussi bien par la qualité |
| tp_633e63a43d107cfdfcd437f1_91281 | trustpilot | — | 0 | Had a lot of trouble with checking out Had a lot of trouble with checking out |
| tp_62626d897b2618e9fea2afe2_58778 | trustpilot | — | 0 | Bonjour Nouvelle cliente avril 2022 j… Bonjour Nouvelle cliente avril 2022 j ai commandé et reçu mes |
| tp_61e6978ca16c1e751f6c9f07_84962 | trustpilot | — | 0 | Artikel wie beschrieben und die… Artikel wie beschrieben und die Lieferung erfolgte auch schnell obw |
| mz_R2V64XKLZ9Z3WY_17052 | momcozy | — | 0 | Der Ventilator ist aufladbar (über USB) und somit für unterwegs praktisch. Er muss während des Betri |
| zd_950212_12712 | zendesk | — | 0 | Conversation with Web User 69680044e705877de86363e1 |
| amz_B0C2V642R8_63227 | amazon_competitor | — | 0 | This dream egg has many sounds. It also has an on off button and memory.  When you find the sound th |
| amz_B0BYNDL3SW_83458 | amazon_competitor | — | 0 | a little bigger then i was expecting but my son still loves it! |
| tp_65d379dab3a84d7111e8b260_93887 | trustpilot | — | 0 | Gute Beschreibung der Produkte Gute Beschreibung der Produkte |
| tp_653a1913c152425d264c2df4_99170 | trustpilot | — | 0 | Erschwerte Bestellung sowie online als auch telefonisch Es ist mir online nicht gelungen, eine neue  |
| zd_1102351_16449 | zendesk | — | 0 | Conversation with Web User 69b6b2aaa00a09a579d8cf85 |
| tp_614dfcae6223e22118b9dbef_48754 | trustpilot | — | 0 | Einfach wunderschön Einfach wunderschön, sowohl der neue Kinderwagen als auch der Wagensack. Tolle Q |
| zd_1103024_4600 | zendesk | — | 0 | Bonjour je vous es commander une caméra je n’arrive pas à connecter deux telephone déçu j’aimerai sa |
| amz_B081HRXPPG_171078 | amazon_competitor | — | 0 | Mat is very hard. Hard enough to be concerned about baby hitting head |
| tp_67e93a5df2d939a092dec842_83687 | trustpilot | — | 0 | Allaitementfriendly Vêtements de qualité fabriqués dans l'U.E. Motifs des tissus originaux et slogan |
| zd_1122633_15327 | zendesk | — | 0 | Hi, it has been 10 days since I made my order with no update. My order number is #JJ1546795.   Can y |
| zd_1058241_45476 | zendesk | — | 0 | Conversation with Web User 69a0df2eeda5ce970076b322 |
| amz_B08S2RXKHJ_178153 | amazon_competitor | — | 0 | Buen material excelente producto |
| zd_1104656_5578 | zendesk | — | 0 | Conversation with Web User 69b7d664e94493d407376e99 |
| zd_1036824_44258 | zendesk | — | 0 | Conversation with Web User 69960dd94447bdf14142e11e |
| amz_B07SRVB6T6_136632 | amazon_competitor | — | 0 | Wheels wonky af. They splay outwards, don’t turn, and make it annoying to use. Boo. |
| amz_B093GS5723_159997 | amazon_competitor | — | 0 | I always get compliments and get asked where I got this everywhere I go! |
| tp_661f7dd4fdc07b2a131bc2e9_80945 | trustpilot | — | 0 | qualitatives Spielzeug bestelle zum ersten Mal |
| tp_69766bb02a44a117a152fd1c_87104 | trustpilot | — | 0 | Email alert about sale and discount… Email alert about sale and discount code automatically applied  |
| amz_B0D3S481VH_172967 | amazon_competitor | — | 0 | My preschool class was waaaaaaay too boogery this winter.  I bought this and told all the parents ab |
| amz_B09V5PVTGF_117038 | amazon_competitor | — | 0 | I just received it today, so I haven't used it much.  I'm excited to start seeing the benefits! |
| tp_5d5aa5fff01869069cd00247_24005 | trustpilot | — | 0 | Good service Good service, could be slightly faster on delivery but not a deal breaker |
| amz_B0CG5BSMVD_158774 | amazon_competitor | — | 0 | Excelente comodidad al 100, buena tela invisible totalmente. Normalmente uso talla M, compré la faja |
| mz_R1IX4OADJUREOE_18133 | momcozy | — | 0 | Muy pesada, muy grande la verdad muy mala experiencia. Mis 1.56cm y la verdad se me hizo muy muy inc |
| tp_6677e67a9a886f0077eb5cf0_79889 | trustpilot | — | 0 | J'adore aller dans ce magasin J'adore aller dans ce magasin, quand on y rentre c'est très agréable g |
| tp_68131ddd37731762af011c74_94748 | trustpilot | — | 0 | Orchestra hornu Ravi de retrouver un magasin orchestra |
| tp_659a6b69c69aec31224a7439_93540 | trustpilot | — | 0 | Prompte Lieferung, Ware ident mit Abbildung Prompte Lieferung, |
| tp_6093e1eef9f4870a6ca78770_98783 | trustpilot | — | 0 | My purchased item from Philips did not… My purchased item from Philips did not arrive. I went to the |
| zd_934776_41005 | zendesk | — | 0 | Conversation with Web User 695fe22e406cd5de42e009aa |
| tp_692f2b267e2f9d256580752a_75908 | trustpilot | — | 0 | magasin Anglet Grand magasin très agréable , personnel très gentil et à l écoute . Je vous le recomm |
| tp_522dde1c00006400025880aa_27356 | trustpilot | — | 0 | Where is my stuff? Well I'm still waiting for my products so I don't think I'm in a position to rate |
| amz_B0035ER8GM_166550 | amazon_competitor | — | 0 | Preciosa cuña. Facil de armar. No incluye colchón ni accesorios |
| tp_644fa06b746ca7dfea13fb4e_35355 | trustpilot | — | 0 | Cliente depuis 2019 Je suis une de vos clientes depuis 2019. Je trouves que la qualité est de moins  |
| amz_B09RGRVM9S_100482 | amazon_competitor | — | 0 | These are huge! Such a bit different between size 2. My son hates them |
| tp_6832dc07e7dfaf0e8f80cdda_51651 | trustpilot | — | 0 | Absolut empfehlenswert Beratung, Freundlichkeit, Hilfsbereitschaft. Es war einfach ein perfekter run |
| amz_B07FCDY93D_132033 | amazon_competitor | — | 0 | I feel like I am squishing my grandbaby when I make it tight enough to feel secure. |
| amz_B0DF6NRKP7_158826 | amazon_competitor | — | 0 | Luego de usarlo periódicamente en el plazo de un mes. Se ha vuelto como loco el humificador al cambi |
| amz_B00FNZQHJA_172159 | amazon_competitor | — | 0 | I originally used it for chapped lips! Better than aquaphor! |
| amz_B01898CQIS_179330 | amazon_competitor | — | 0 | Nach 4 mal benutzen funktioniert er schon nicht mehr |
| amz_B08BN8BYR2_76212 | amazon_competitor | — | 0 | ¡Producto perfecto y súper funcional para mi bebé!<br>Compré este centro para bebé de Pamo Babe y ha |
| tp_5e1484fbc845450914b2f31a_84074 | trustpilot | — | 0 | Ich bin enttäuscht von den Möbeln Ich bin enttäuscht von den Möbeln, überall ist der Lack abgeplatzt |
| mz_R1ZAKKTFOB8F70_15407 | momcozy | — | 0 | After Less than a week of using it, I have put in the microwave for 60 seconds (the instructions say |
| tp_59b5106cd2c87507849193b4_88477 | trustpilot | — | 0 | rapide et conforme à la demande. rapide et conforme à la demande. je conseil à tout le monde. |
| tp_682cb0f32bb52589ef1100b5_93448 | trustpilot | — | 0 | Parfait Produit conforme à mes attentes et reçu très rapidement ! |
| rd_REAZ7WMAR558_5265 | reddit | — | 0 | This product did not last and no remedy was offered |
| zd_1090893_2326 | zendesk | — | 0 | Hi there,   Please can you cancel our order and issue a refund?   Thanks Sally |
| tp_68cc3d1171cc5e928e5e4666_62872 | trustpilot | — | 0 | Tonies vorbestellt und Lieferung zum… Tonies vorbestellt und Lieferung zum Erscheinungstag. Vielen D |
| tp_68ab52deb59edaabf3460e3c_83980 | trustpilot | — | 0 | Tout s’est très bien passé sur… Tout s’est très bien passé sur faireparterie! Leur site internet est |
| amz_B0D37VQF9P_178630 | amazon_competitor | — | 0 | Works very well to purify the air and keep environment fresh |
| tp_5f0090943f06f20658d212df_93658 | trustpilot | — | 0 | Site clair & facile à utiliser . Site clair & facile à utiliser . |
| zd_1082908_31624 | zendesk | — | 0 | Good night,  I made a mistake in my delivery address. Could you please correct it and provide the co |
| tp_69691b82dffe692dae9ebc93_69555 | trustpilot | — | 0 | Très bonne boutique et bons articles… Très bonne boutique et bons articles dans l’ensemble. Peut etr |
| amz_B08N7T58YY_162356 | amazon_competitor | — | 0 | Me gusta mucho este cochecito, solo que hubiera deseado que fuera liviano es un poco pesado pero en  |
| tp_619a86348931a330873220fd_99422 | trustpilot | — | 0 | Artikel häufig nicht lieferbar Leider sind schon zum 3. Mal die Artikel, die ich bestellt habe, hint |
| mz_R1BGBB875YY3XN_12704 | momcozy | — | 0 | Very bad buy this one for my wife she has so much complain about it..it’s make her inching not value |
| tp_5c62b1e397afa10b687aaf2f_50380 | trustpilot | — | 0 | Große Auswahl an Karten Große Auswahl an Karten, schnelle Bearbeitung. Es wird auch nach Bestellung  |
| amz_B07RK763T2_148222 | amazon_competitor | — | 0 | Proporciona una compresión moderada que mejora la postura y puede ayudar a reducir molestias en la z |
| zd_934199_43089 | zendesk | — | 0 | Conversation with Web User 695fae49d89890d8a3e4d597 |
| amz_B08BF3N47B_119966 | amazon_competitor | — | 0 | I got this from my registry and now that i finally put it together and went to use it, i see the man |
| tp_69d0c62a4a361efdac623e79_99107 | trustpilot | — | 0 | Très contente !!! Très contente !!! |
| zd_1060099_42652 | zendesk | — | 0 | Dzień dobry,  Kupiłem dwa razy ten sam produkt. Proszę o odwołania zamówienia JJ1503150 oraz zwrot ś |
| mz_R3MPEW7OJXQZGR_17001 | momcozy | — | 0 | Heated one pad up for my first usage and the package immediately got damaged. |
| amz_B0DJ8X5F83_184307 | amazon_competitor | — | 0 | Comodas y el tamaño es correcto |
| zd_1127859_45388 | zendesk | — | 0 | Conversation with Web User 69c2beac0d69284eb0cda68e ====================== Conversation with Web Use |
| tp_690af77795f9f9fdeff8ca15_88525 | trustpilot | — | 0 | Commande arrivée très rapidement et… Commande arrivée très rapidement et produits de qualité |
| amz_B0B9RHZ2KL_183696 | amazon_competitor | — | 0 | Lo compre para un regalo. Especial |
| tp_67ee4bf3a3de395b390fdebd_94827 | trustpilot | — | 0 | Alles pünktlich Alles pünktlich, Artikel funktioniert. |
| zd_1121968_42286 | zendesk | — | 0 | Bonjour,on vient de faire une commande chez vous le 19 mars mais, quand on regarde sur l'application |
| tp_5f0b1d343f06f20a9c9b8564_95315 | trustpilot | — | 0 | Schnell und unkompliziert Schnell und unkompliziert |
| tp_619f58e48931a33087363a5f_94454 | trustpilot | — | 0 | Humidificateur et purificateur Satisfaite de mon achat. |
| tp_672f790691ae1200bf00f92c_64949 | trustpilot | — | 0 | Quick and easy shipping! Quick and easy shipping! |
| tp_695af4016f8b3b9786a49c3c_74254 | trustpilot | — | 0 | Expérience convaincante Beaucoup de choix à des prix corrects, surtout avec la carte Orchestra 👍et p |
| zd_940968_43950 | zendesk | — | 0 | Conversation with Web User 6963b2ff1ac1acfb5fccb7a5 |
| zd_1125282_15682 | zendesk | — | 0 | Hello, I ordered some items from mom Cosy and still haven’t received them. The order number is  #JJ1 |
| tp_635477d98056669a30a8a8d7_98553 | trustpilot | — | 0 | Die reinste Katastrophe!!! Ich muss mich den negativen Bewertungen hier anschließen und waren vor Be |
| zd_1100168_16509 | zendesk | — | 0 | Hallo, Leider warte ich immernoch auf eine Versandbestätigung von der Bestellung vom 8.3.  Wann wird |
| tp_63680f57252cba2c02d0cae6_77224 | trustpilot | — | 0 | Très bon site Très bon site Propose une variété de produits de bon goût Prix raisonnables |
| tp_6214a70bba48cd57a29f2949_73218 | trustpilot | — | 0 | Difficile de se faire rembourser Remboursement effectué près d'un mois après le retour, mais sans ré |

### 4.3 `generic_only` （零标签 且 去停用词后有效 token < 3）— 抽样 100 条

| review_id | data_source | product_line | n_tags | text_preview |
|---|---|---|---:|---|
| amz_B08352W1T8_117191 | amazon_competitor | 家居家纺 | 0 | I absolutely love my pillow!¡! |
| zd_1091562_38978 | zendesk | 吸奶器 | 0 | Εχω θέμα με το θηλαστρο V1pro που είχα πάρει ενώ έχω γάλα κανονικά κ όταν θηλάζω τρέχουν γάλατα από  |
| amz_B0B7MFS2W4_132743 | amazon_competitor | 家居家纺 | 0 | It’s the best stroller I really like it |
| amz_B09FPLGYF2_134653 | amazon_competitor | 智能母婴电器 | 0 | It fits everywherePracticalLightweightComfortable |
| zd_927691_27105 | zendesk | 吸奶器 | 0 | Conversation with AmandaRoeske |
| amz_B004GUU802_135874 | amazon_competitor | 家居家纺 | 0 | It's a very nice crib. I really like it |
| amz_B09XJJB478_140899 | amazon_competitor | 喂养电器 | 0 | The only pacifier my grandson will use. |
| amz_B0DJVFX2KY_141278 | amazon_competitor | 智能母婴电器 | 0 | This humidifier its really good. I recommend as very good one. |
| amz_B07TDN9MKC_149031 | amazon_competitor | 喂养电器 | 0 | All of my babies have loved these pacifiers! |
| mz_R1Z9UY70KLKXHF_19389 | momcozy | 吸奶器 | 0 | 搾乳機の専用ケースだけあって収まりがよく、破損など絶対なさそうです。子どもが入院してる時に搾乳が必要なことがあり、こういったケースがあるのは心強いです。高級感あるこういうケースから出し入れする時はテン |
| zd_1021491_39362 | zendesk | 吸奶器 | 0 | Conversation with JessicaBennett |
| mz_R39MSLVTEG31ZB_19401 | momcozy | 吸奶器 | 0 | 2台の搾乳機が収納可能なハードシェルバッグです。同メーカーの装着型搾乳機2セットが固定して入るようになっています。バッグとしては、大きさの割には少し重ために感じますが、内側外側共に防水素材が使われてい |
| amz_B089QJ6X3X_154166 | amazon_competitor | 家居家纺 | 0 | We love our wagon! It definitely gets used! |
| zd_1086070_28540 | zendesk | 吸奶器 | 0 | Conversation with SelenaDagher |
| zd_1122199_27072 | zendesk | 吸奶器 | 0 | Conversation with RaeganMueller |
| zd_1020398_27090 | zendesk | 吸奶器 | 0 | Conversation with SergiCardoza |
| amz_B01AUQD9LU_159462 | amazon_competitor | 喂养电器 | 0 | Only pacifiers my two youngest would use. |
| amz_B0DSGDGH42_160811 | amazon_competitor | 智能母婴电器 | 0 | Very pleased with this purifier.I recommend! |
| zd_1055792_28535 | zendesk | 吸奶器 | 0 | Conversation with SelenaDagher |
| zd_986769_28533 | zendesk | 吸奶器 | 0 | Conversation with VictoriaCarman |
| zd_932042_39666 | zendesk | 吸奶器 | 0 | Hi I would like to to cancel my order   JJ1402104  Please |
| zd_1040055_27061 | zendesk | 吸奶器 | 0 | Conversation with AllyFriedenberger |
| amz_B08YKVJTDH_164644 | amazon_competitor | 母婴综合护理 | 0 | They are very good, more wet than other wipes |
| amz_B09538RHV4_165428 | amazon_competitor | 吸奶器 | 0 | I really like it to use the pump with. |
| zd_930730_27084 | zendesk | 吸奶器 | 0 | Hi when will this order arrive? |
| amz_B0BHT3DQ8D_170215 | amazon_competitor | 智能母婴电器 | 0 | Love it! This is a great air purifier. |
| amz_B08JYJKZNL_170265 | amazon_competitor | 家居家纺 | 0 | Good product, everything as described 👍 |
| amz_B0DTHSRTD4_170340 | amazon_competitor | 智能母婴电器 | 0 | Very good purifier for a 10x12 room. |
| zd_1039366_27076 | zendesk | 吸奶器 | 0 | Conversation with JessicaNalerio |
| amz_B08PVXSH5N_170892 | amazon_competitor | 家居家纺 | 0 | Very nice stroller and works good |
| mz_R3TZWUG1YN7VMQ_18855 | momcozy | 智能母婴电器 | 0 | Massager used when you've got pain |
| zd_1121465_27112 | zendesk | 吸奶器 | 0 | Conversation with BrigitteDery |
| mz_R12WAJ0TO461RW_18269 | momcozy | 吸奶器 | 0 | Exactly what I was looking for |
| amz_B08PMJBN33_171660 | amazon_competitor | 家居家纺 | 0 | My granddaughter loved the crib |
| amz_B07FRYYZ3N_174067 | amazon_competitor | 母婴综合护理 | 0 | I don't use any other type of wipes. |
| zd_1137196_27137 | zendesk | 吸奶器 | 0 | الصفحه دي بتقول انها توكيل ممكوزي في مصر ؟؟  هل ده حقيقي ؟؟ |
| amz_B0B6Q7RZ9N_175861 | amazon_competitor | 家居家纺 | 0 | My daughter did not like this stroller at all |
| amz_B08QR6V8WR_176019 | amazon_competitor | 母婴综合护理 | 0 | These are the only wipes we use at our house |
| amz_B01LY80MNC_176097 | amazon_competitor | 智能母婴电器 | 0 | I Really Like The Sturdiness Of This Bassinet |
| zd_1004703_27107 | zendesk | 吸奶器 | 0 | Conversation with KylerHandley |
| mz_R2CR04F45V7I2I_19467 | momcozy | 吸奶器 | 0 | 専用設計なので、搾乳機がケースの中で暴れず、収まりが非常に良いです。ですが、安い搾乳機と引けを取らないくらいの価格帯でケースのみなので、購入をしにくいのは難点。 |
| amz_B077GJLGC4_178217 | amazon_competitor | 母婴综合护理 | 0 | these are the only wipes we use! |
| amz_B07FT2H5J8_179267 | amazon_competitor | 内衣服饰 | 0 | I will not be wearing it as it really does nothing. |
| amz_B09K39Y11L_179803 | amazon_competitor | 家居家纺 | 0 | She liked the pillow. Now to get her pregnant... |
| amz_B00BVUQEWG_180457 | amazon_competitor | 喂养电器 | 0 | My baby only use these pacifiers. |
| amz_B0BL8M2Z1Q_181970 | amazon_competitor | 母婴综合护理 | 0 | It’s a wipe, it did what it was supposed to do. |
| mz_R12EY0V8BG9NLA_19621 | momcozy | 吸奶器 | 0 | 純正品なのでサイズピッタリでした。なんの問題も無く使用出来ています。持ち運びには必須アイテムです |
| amz_B07KF75Q7J_182967 | amazon_competitor | 家居家纺 | 0 | This definitely supports as described. |
| zd_920160_36533 | zendesk | 吸奶器 | 0 | Abigail S. DiSalvo (203)-885-9313 |
| zd_973345_27075 | zendesk | 吸奶器 | 0 | Conversation with IngzelleFergie |
| amz_B09539M877_183510 | amazon_competitor | 吸奶器 | 0 | Definitely buy it if you plan to pump |
| mz_R21C9UMPTDWO7X_19016 | momcozy | 喂养电器 | 0 | Every parents should have this |
| amz_B0BJ59XZBK_183832 | amazon_competitor | 喂养电器 | 0 | These are our favorite pacifiers. |
| zd_1129054_28672 | zendesk | 吸奶器 | 0 | Conversation with CassidyLasick |
| amz_B0CR133K23_184313 | amazon_competitor | 家居家纺 | 0 | It’s not as fluffy as described |
| amz_B07QKG7BD5_184660 | amazon_competitor | 家居家纺 | 0 | They were exactly as described |
| amz_B095HLPGTV_185991 | amazon_competitor | 吸奶器 | 0 | Did not work for pumping at all. |
| mz_R37ZO13P17LEC1_19425 | momcozy | 吸奶器 | 0 | 搾乳機は使っていないので小さめのカップなどの洗浄に使っていますがやはり用途が違いサイズが短いため洗いにくいのが本音です。品質的には持ち手などはしっかりしているもののスポンジ部分はもう一つかな…本来の使 |
| amz_B0BB2CD4RN_195159 | amazon_competitor | 家居家纺 | 0 | Was not good for my infants teeth |
| tp_682772be79f9307e628e8cca_72195 | trustpilot | 智能母婴电器 | 0 | Really good product Love how light the stroller is |
| zd_1130148_27096 | zendesk | 吸奶器 | 0 | Conversation with AmandaForney |
| zd_929510_1200 | zendesk | 吸奶器 | 0 | Conversation with JessicaThompson |
| zd_1004592_1214 | zendesk | 吸奶器 | 0 | Conversation with RosalindaOlivares |
| zd_949266_1218 | zendesk | 吸奶器 | 0 | Conversation with MelissaRinehart |
| zd_1120908_1219 | zendesk | 吸奶器 | 0 | Conversation with ElizabethTaylor |
| zd_1086926_1224 | zendesk | 吸奶器 | 0 | Conversation with KaylahBarriger |
| zd_1054287_37048 | zendesk | 吸奶器 | 0 | Conversation with DestinyMelton |
| zd_1054162_1228 | zendesk | 吸奶器 | 0 | Conversation with JasmineHughes |
| mz_R15HG3K8WN5UKE_19478 | momcozy | 吸奶器 | 0 | 専用ケースで、授乳部品の隙間が無く丁度よく使えました。出先など、ほこりの心配も少ない点が良かったです。引き続き耐久性を確認していきます。 |
| zd_1121958_1235 | zendesk | 吸奶器 | 0 | Conversation with BhavikaSethi |
| zd_922464_1243 | zendesk | 吸奶器 | 0 | Conversation with RebekahPaine |
| zd_954725_1409 | zendesk | 吸奶器 | 0 | Conversation with KalissiaClark |
| zd_1089536_10150 | zendesk | 喂养电器 | 0 | When I will receive my order my order number is JJ1532515 |
| zd_987921_10649 | zendesk | 喂养电器 | 0 | สวัสดีครับ  ผมขอรบกวนติดตามสถานะคำสั่งซื้อ เนื่องจากขณะนี้คำสั่งซื้อของผมเกินระยะเวลาดำเนินการ/จัดส่ |
| zd_931743_10867 | zendesk | 喂养电器 | 0 | I was wondering when will my order be shipped?  #JJ1393147 |
| zd_1100441_12310 | zendesk | 喂养电器 | 0 | hi,I would like to know when my order will be shipped? |
| zd_1008659_38965 | zendesk | 吸奶器 | 0 | Conversation with MustafAsiye🤍 |
| zd_1080110_28525 | zendesk | 吸奶器 | 0 | Conversation with BriannaMitchell |
| mz_R1OGOFWR2JYVKQ_19424 | momcozy | 吸奶器 | 0 | 旅行、帰省などで搾乳器の持ち運びとして活用するために注文しました。機械が守れるので故障する心配なく助かります。ただちょっと大きすぎますね。。もう少しコンパクトだと嬉しいです。ですが他のパーツや充電器も |
| zd_927745_38952 | zendesk | 吸奶器 | 0 | Conversation with BrookeMcClain |
| zd_933297_28531 | zendesk | 吸奶器 | 0 | Conversation with KylieMccomber |
| zd_1087047_16544 | zendesk | 内衣服饰 | 0 | Conversation with brandihutchings |
| zd_1138847_16550 | zendesk | 内衣服饰 | 0 | Conversation with JewelOlbrantz |
| zd_1086717_20529 | zendesk | 吸奶器 | 0 | Hi I have a problem with my pumps |
| zd_942991_21457 | zendesk | 吸奶器 | 0 | Conversation with Sophieeeee😇🤪 |
| zd_1018395_21485 | zendesk | 吸奶器 | 0 | Conversation with George...😎... 🌊 |
| zd_927521_39601 | zendesk | 吸奶器 | 0 | JJ1385202 Has this order being sent yet? |
| zd_1093158_21501 | zendesk | 吸奶器 | 0 | Conversation with د. فهد الشرهان |
| zd_1025068_27080 | zendesk | 吸奶器 | 0 | Conversation with Yasmin 🥰🫶🏻😇🙏🏻 |
| zd_918399_27059 | zendesk | 吸奶器 | 0 | Conversation with CatherineStewart |
| zd_1028758_21532 | zendesk | 吸奶器 | 0 | Conversation with Jo Jo Lovell |
| zd_1119520_24592 | zendesk | 吸奶器 | 0 | Why is the item not shipped yet? |
| zd_991647_24597 | zendesk | 吸奶器 | 0 | My order number is .   JJ1443656 |
| mz_R8E2DLJ2IQRIZ_19433 | momcozy | 吸奶器 | 0 | 搾乳機をしっかり守って持ち運ぶ事に重きを置いている為、形状が大きく、かさばります。その分、防水性は非常に高く、中を開けると仕切りが多く整理しやすいです。互換性が高いので他の製品でも使えると思います。追 |
| zd_946595_24621 | zendesk | 吸奶器 | 0 | Conversation with Monica Muñiz |
| zd_1144877_24779 | zendesk | 吸奶器 | 0 | Conversation with AllisonVance |
| zd_1019606_27033 | zendesk | 吸奶器 | 0 | Conversation with MorganHarrelson |
| zd_1026992_27047 | zendesk | 吸奶器 | 0 | My order says it was delivered and it’s not |
| zd_977241_36557 | zendesk | 吸奶器 | 0 | Conversation with Patrick1010!! |
| mz_R29V6K414IR7ND_19440 | momcozy | 吸奶器 | 0 | 高品質なシリコン素材で作られており、BPAフリーで安心して使用できます。ダブルシールフランジがしっかりと密閉し、効率的な搾乳をサポートします。簡単に取り付けられ、洗浄も容易です。1パックで提供されてい |

## 五、阈值判定（决策 4）

| 指标 | 阈值 | 实测 | 结果 |
|---|---:|---:|:---:|
| 原始覆盖率 | 88.00% | 82.27% | 🔴 FAIL |
| 业务有效覆盖率 | 94.00% | 93.07% | 🔴 FAIL |

## 六、与 Phase 4 对比

- Phase 4 5K 子集原始覆盖率：82.58%
- Phase 5 全量原始覆盖率：82.27% （-0.31pp）
- Phase 5 全量业务有效覆盖率：93.07%
