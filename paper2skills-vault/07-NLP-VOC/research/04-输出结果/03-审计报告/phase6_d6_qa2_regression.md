# Phase 5 D3 三方评估报告

## 1. 总览

| 系统 | n_eval | tag macro-F1 | tag weighted-F1 | mean Jaccard | sentiment κ | NPS κ |
|---|---:|---:|---:|---:|---:|---:|
| Phase 4 (rule + ALCHEmist) | 63 | 0.000 | 0.000 | 0.190 | 0.253 | 0.325 |
| Phase 5 D2 (DeepSeek-V4-Flash) | 63 | 0.000 | 0.000 | 0.000 | 1.000 | 0.325 |


## 2. Phase 4 (rule + ALCHEmist) 详细指标

- 评估样本: 63（缺失 0）
- tag 全集: 41 个

**Sentiment 混淆矩阵** (n=60)

| true \ pred | positive | neutral | negative |
|---|---|---|---|
| **positive** | 26 | 12 | 1 |
| **neutral** | 0 | 1 | 0 |
| **negative** | 5 | 10 | 5 |

**Proxy NPS 混淆矩阵** (n=60)

| true \ pred | promoter | passive | detractor |
|---|---|---|---|
| **promoter** | 23 | 15 | 1 |
| **passive** | 0 | 1 | 0 |
| **detractor** | 2 | 9 | 9 |

**Top-30 高频 tag 表现（按 support 降序）**

| tag_id | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| BRAND_Elvie | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Kindred_Bravely | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Momcozy | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Spectra | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Tommee_Tippee | 0 | 0.00 | 0.00 | 0.00 |
| TAG_ALC_BREAST_PUMP | 0 | 0.00 | 0.00 | 0.00 |
| TAG_ALC_VIELEN_DANK | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N005 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N006 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N008 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_002 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_003 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_004 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_008 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_011 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_014 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_015 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_016 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_A001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_A002 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_C001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_D001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_E001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_E003 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_E004 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_N008 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_P001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_S004 | 0 | 0.00 | 0.00 | 0.00 |

## 2. Phase 5 D2 (DeepSeek-V4-Flash) 详细指标

- 评估样本: 63（缺失 0）
- tag 全集: 68 个

**Sentiment 混淆矩阵** (n=14)

| true \ pred | positive | neutral | negative |
|---|---|---|---|
| **positive** | 7 | 0 | 0 |
| **neutral** | 0 | 0 | 0 |
| **negative** | 0 | 0 | 7 |

**Proxy NPS 混淆矩阵** (n=60)

| true \ pred | promoter | passive | detractor |
|---|---|---|---|
| **promoter** | 23 | 15 | 1 |
| **passive** | 0 | 1 | 0 |
| **detractor** | 2 | 9 | 9 |

**Top-30 高频 tag 表现（按 support 降序）**

| tag_id | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| BRAND_Elvie | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Kindred_Bravely | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Momcozy | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Spectra | 0 | 0.00 | 0.00 | 0.00 |
| BRAND_Tommee_Tippee | 0 | 0.00 | 0.00 | 0.00 |
| TAG_ALC_BREAST_PUMP | 0 | 0.00 | 0.00 | 0.00 |
| TAG_ALC_VIELEN_DANK | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N005 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N006 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N007 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_DEF_N008 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_002 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_003 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_004 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_008 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_011 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_014 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_015 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_016 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_A001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_A002 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_C001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_D001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_E001 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_E003 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_E004 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_E007 | 0 | 0.00 | 0.00 | 0.00 |
| TAG_GEN_N008 | 0 | 0.00 | 0.00 | 0.00 |