---
name: phase6-d1-v41-dict-fix
description: Phase 6 D1 F7 字典字段修复报告 — aspect→tag 映射 + aspect_cn 中文 + 优化建议补齐。当验证 v4.0 → v4.1 字段质量修复效果、查看 LLM 调用成本时使用。
date: 2026-05-09
phase: phase6
day: D1
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D1 F7 字典字段修复报告（v4.0 → v4.1）

- **基础字典**：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/tag_dictionary_v4.0.xlsx`
- **产出字典**：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx`
- **运行时间**：2026-05-09T16:51:59

## 一、修复统计

| 子任务 | 实施 | 数量 |
|---|---|---:|
| F7.1 aspect→tag_id 映射（规则打分）| 填入 | 40 |
| F7.1 aspect→tag_id 映射 | 无匹配（记 <NO_MATCH>）| 15 |
| F7.2 aspect_cn 中文补齐（LLM）| 成功填入 | 55 |
| F7.3 优化建议补齐（LLM）| 成功填入 | 58 |
| LLM 调用次数 | — | 4 |
| LLM 失败次数 | — | 0 |
