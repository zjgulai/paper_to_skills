---
name: phase5-d3-handoff-manual
description: Phase 5 D3 金标标注 + 三方评估交接手册（方案 A，双 LLM 共识 + 人工仲裁）。提供 1.5-2h 人工仲裁的工作流，避免 LLM-as-judge 偏见的同时把人工工作量从 4-6h 压到 1/3。当开始 D3 金标标注或使用 evaluation_suite 时使用。
---

# Phase 5 D3 交接手册 — 双 LLM 共识 + 人工仲裁（方案 A）

**日期**：2026-05-07
**阶段**：Phase 5 D3（金标 500 + 三方评估）
**状态**：✅ Kimi 共识已跑完，66.4% 自动落盘，仅 **168 条**需要 ~1.5-2h 人工仲裁

---

## 一、为什么走方案 A（关键决策记录）

### 不能用单 LLM 当金标的方法论问题
- LLM 当评判 LLM 的"裁判"会触发 **self-preference bias**（学术界有记录）
- 同模型双跑 κ ≈ 1.0 但跟真实标签可能 κ < 0.5 — 评估失真
- 对外报"准确率 X%" 时，单 LLM 金标不可信

### 方案 A 的正确性
- **DeepSeek-V4-Flash + Kimi-K2.6** 是异构家族（不同训练数据 / 不同架构）
- 两者**独立同时正确**的概率 ≫ 同时错的概率（独立性假设）
- 取**交集**作为高置信度金标 = 类似集成学习的"硬投票"
- 人工只需仲裁分歧样本，避免 4-6h 重复劳动

### 实证数据（在我们 500 条上）
| 共识规则 | 自动落盘率 | 人工工作量 |
|---|---:|---|
| 严格共识（Jaccard ≥ 0.6 + sent + NPS）| 12.4% | ~5h（不可用）|
| **软共识（≥1 共享 tag + sent + NPS）** | **66.4%** | **~1.5-2h** ✅ |
| 仅 sent + NPS | 78% | 但标签信息不可信 |

---

## 二、当前状态

| 项 | 数值 |
|---|---:|
| 总金标样本 | 500 |
| ✅ 自动落盘（双 LLM 共识）| **332**（66.4%）|
| 🔶 需人工仲裁 | **168**（33.6%）|
| 平均每条仲裁耗时 | 30-45 秒 |
| 预计完成时间 | **1.5-2 小时** |

**金标文件路径**：[`golden_set_500_consensus.jsonl`](../../03-数据资产/golden_set_500_consensus.jsonl)

---

## 三、人工仲裁操作命令

### 启动仲裁（只看 168 条分歧）

```bash
cd /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎

python3 human_annotation_cli.py \
  --input /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/golden_set_500_consensus.jsonl \
  --only-disagreement
```

### 分歧时 CLI 会展示什么

```
[12/500]  review_id=amz_B0BYVHKZLZ_xxxxx
source=amazon_competitor   lang=🇬🇧 EN   rating=5
--------------------------
评论原文 ...
--------------------------
Phase 4 labels (2):
  [P4] TAG_GEN_X001        ...
DeepSeek pred (3): overall=positive  nps=promoter
  [1] TAG_GEN_E003        Quality praise  conf=0.95
  [2] TAG_GEN_E002        Comfort         conf=0.88
  [3] TAG_GEN_E008        Value for money conf=0.81
Kimi pred (3): overall=positive  nps=passive   ← NPS 不一致！
  [k1] TAG_GEN_E003       Quality praise  conf=0.91
  [k2] TAG_L2_005         Brand trust     conf=0.85
  [k3] TAG_GEN_E007       Recommendation  conf=0.79
⚠ Disagreement: nps_diff:promoter!=passive
--------------------------
```

### 你只需做 3 个判断

1. **标签**：通常按 `Enter` 或 `1,3` 接受 LLM 预测 — 因为分歧多在排序，不在主标签
2. **sentiment**：`p`/`n`/`x`
3. **NPS**：`m`/`p`/`d` — 这是分歧最多发的字段

### 快捷键

| 键 | 含义 |
|---|---|
| `Enter` | 接受 DeepSeek Top-3 |
| `1,3` | 仅采纳 DeepSeek 第 1、3 条 |
| `k1,k2` | 采纳 Kimi 第 1、2 条（注：当前 CLI 只支持 DeepSeek 索引；可手写 tag_id）|
| `TAG_X001` | 手写 tag_id |
| `s` / `b` / `q` | 跳过 / 回退 / 退出（自动保存）|

---

## 四、分歧画像（你心里有底）

| 分歧原因 | 数量 | 应对 |
|---|---:|---|
| no_shared_tags | 138 | 看原文判断哪个 LLM 的标签更对，可能两边都部分对 → 选 1-2 个真实命中的 |
| sentiment_diff | 30 | 通常一边判 positive 一边 neutral；自己看原文决断 |
| nps_diff | 29 | DS 倾向 promoter，KM 倾向 passive — 看是否真的"主动推荐" |
| both_zero_label | 14 | 短文本/纯品牌名，可填空 + sentiment 估个 neutral |
| kimi_missing | 2 | API 失败，参考 DS 即可 |

---

## 五、自一致性检查（仲裁后第 2 天，可选）

如果你担心仲裁质量，从 168 条仲裁结果里抽 30 条第 2 天再标一遍：

```bash
cd /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产
python3 -c "
import json, random
recs=[json.loads(l) for l in open('golden_set_500_consensus.jsonl')]
human_done=[r for r in recs if r.get('golden_source')=='human']
random.seed(20260508)
pick=random.sample(human_done, min(30, len(human_done)))
with open('golden_set_30_redo.jsonl','w') as f:
    for r in pick:
        r2={**r,'golden_labels':[],'golden_overall_sentiment':None,'golden_proxy_nps':None,'golden_notes':'','golden_source':'needs_human'}
        f.write(json.dumps(r2, ensure_ascii=False)+'\n')
print('Wrote 30 blanks')
"

python3 /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/human_annotation_cli.py \
  --input golden_set_30_redo.jsonl

python3 /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/evaluation_suite.py self-consistency \
  golden_set_500_consensus.jsonl golden_set_30_redo.jsonl \
  --json-out /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d3_self_consistency.json
```

---

## 六、三方评估（仲裁完成后）

```bash
python3 /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/evaluation_suite.py three-way \
  --golden /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/golden_set_500_consensus.jsonl \
  --pred-p4 /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/test_set_5k_stratified.jsonl \
  --pred-llm /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/test_set_5k_p5_llm.jsonl \
  --report /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_eval_v0.md \
  --json-out /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_eval_v0.json
```

### 进阶：分别评估"共识金标 vs 人工金标"

```bash
# 仅在共识金标上跑（332 条）
python3 evaluation_suite.py three-way \
  --golden .../golden_set_500_consensus.jsonl \
  --pred-p4 .../test_set_5k_stratified.jsonl \
  --pred-llm .../test_set_5k_p5_llm.jsonl \
  --golden-source-filter consensus_llm \
  --report .../phase5_eval_consensus.md

# 仅在人工金标上跑（168 条 — 难样本）
python3 evaluation_suite.py three-way \
  --golden .../golden_set_500_consensus.jsonl \
  --pred-p4 ... --pred-llm ... \
  --golden-source-filter human \
  --report .../phase5_eval_human.md
```

> **解读**：人工金标上的 F1 通常会比共识金标低 10-15pp（因为这部分是模型分歧的难样本）— 这是**真实的难度信号**，比单一指标更有诊断价值。

---

## 七、工具清单（更新）

| 工具 | 路径 | 用途 |
|---|---|---|
| Golden 抽样器 | [`golden_set_sampler.py`](../../02-脚本工具/07-LLM引擎/golden_set_sampler.py) | 500 条分层抽样（已跑）|
| Kimi 第二意见 | `llm_labeler.py --vendor kimi` | 用 Kimi 跑 500 条（已跑）|
| 共识合并器 | [`consensus_prefill.py`](../../02-脚本工具/07-LLM引擎/consensus_prefill.py) | 软/严两种共识规则 |
| 人工仲裁 CLI | [`human_annotation_cli.py`](../../02-脚本工具/07-LLM引擎/human_annotation_cli.py) | `--only-disagreement` 仅看 168 条 |
| 三方评估器 | [`evaluation_suite.py`](../../02-脚本工具/07-LLM引擎/evaluation_suite.py) | `--golden-source-filter` 分层评估 |

---

## 八、字段契约（含方案 A 新增）

```jsonc
{
  "review_id":      "...",
  "data_source":    "amazon_competitor",
  "language":       "en",
  "rating":         5,
  "text":           "...",

  "phase4_labels":  [...],

  // DeepSeek 第一意见（D2 输出）
  "llm_pred":              [...],
  "llm_overall_sentiment": "positive",
  "llm_proxy_nps":         "promoter",

  // Kimi 第二意见（方案 A 新增）
  "kimi_pred":              [...],
  "kimi_overall_sentiment": "positive",
  "kimi_proxy_nps":         "passive",

  // 共识结果
  "golden_labels":             [{"tag_id":"...","tag_en":"..."}],
  "golden_overall_sentiment":  "positive",
  "golden_proxy_nps":          "promoter",
  "golden_source":             "consensus_llm" | "human",
  "consensus_meta":            {"jaccard":0.5, "shared_tags":[...], "mode":"soft"},
  "disagreement_reason":       "" | "nps_diff:promoter!=passive",
  "golden_notes":              ""
}
```

---

## 九、风险记录

| 风险 | 应对 |
|---|---|
| 共识金标里两个 LLM "都犯一样的错" | 332 条共识 + 168 条人工 → 评估时分开看 F1，差异 > 15pp 时怀疑共识金标 |
| 软规则 ≥1 shared tag 太松 | 已经只取 intersection 作为 golden_labels，没有任何"妥协"的标签进入金标 |
| 168 条难样本人工标错 | 第 2 天可选 30 条 self-consistency 检查 |

---

## 十、一行总结

> 用 Kimi 当 DeepSeek 的"独立第二意见"，软共识规则把 66.4% 的金标自动化落盘，**人工只需 1.5-2h 仲裁 168 条难样本**，既避免了 LLM-as-judge 偏见，又把工作量压到原方案的 1/3。
