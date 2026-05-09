# D9 T9.3 候选标签三过滤报告

**输入**: 163 原始候选  ｜  **输出**: 2 通过

| 过滤器 | 阈值 | 拒绝 | 保留 |
|---|---|---:|---:|
| F1 频率 | support >= 10 | 130 | 33 |
| F2 Jaccard 去重 | distance < 0.3 | 15 | 18 |
| F3 LLM 业务相关性 | score >= 3/5 | 16 | 2 |

**最终通过**: **2**  目标范围 [20, 40]

## 通过候选标签（按 support 降序）

| tag_en | category | support | Jaccard 最近 | LLM | reason |
|---|---|---:|---|---:|---|
| `schnelle_lieferung` | unknown | 375 | schnelle lieferung; schneller  | 3 | German for 'fast delivery' - relevant for shipping feedback but not product-spec |
| `super_easy` | unknown | 198 | easy to clean aspirator | 5 | Positive sentiment about ease of use, directly actionable. |

## F3 LLM 拒绝（top 30 by lowest score）

| tag_en | support | LLM | reason |
|---|---:|---:|---|
| `call_call` | 350 | 1 | Noisy metadata from call logs, not actionable. |
| `call_october` | 174 | 1 | Timestamp-related noise, not relevant. |
| `conversation_user` | 15 | 1 | Generic conversation metadata, not actionable. |
| `conversation_user` | 112 | 1 | Generic conversation metadata, not actionable. |
| `sent_iphone` | 104 | 1 | Device metadata, not relevant. |
| `conversation_user` | 22 | 1 | Generic conversation metadata, not actionable. |
| `sent_iphone` | 19 | 1 | Device metadata, not relevant. |
| `conversation_user` | 14 | 1 | Generic conversation metadata, not actionable. |
| `utm_source` | 11 | 1 | UTM tracking parameter, irrelevant. |
| `utm_medium` | 11 | 1 | UTM tracking parameter, irrelevant. |
| `sent_iphone` | 39 | 1 | Device metadata, not relevant. |
| `hope_having` | 38 | 1 | Unclear phrase, likely noisy. |
| `having_thinking` | 38 | 1 | Unclear phrase, likely noisy. |
| `any_questions` | 16 | 2 | Generic phrase indicating questions, marginally useful for identifying inquiries. |
| `vielen_dank` | 14 | 2 | German 'many thanks' - positive but generic, limited actionability. |
| `family_kind` | 39 | 2 | Vague phrase 'family kind' possibly indicating sentiment but unclear. |
