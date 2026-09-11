# 04-输出结果: 输出层

本目录存放当前有效的打标输出、诊断报告、审计文件和标签字典。

---

## 目录结构

| 子目录 | 内容 |
|--------|------|
| `01-字典版本/` | 标签字典 v3.5 ~ v3.7 |
| `02-历史字典/` | 标签字典 v3.4 draft/filled（Phase 2 历史版本） |
| `03-审计报告/` | Markdown 审计报告（v3.6/v3.7/Phase 3 总结） |
| `04-审计数据/` | JSON 审计数据（Phase 1~3 各阶段） |
| `05-设计方案/` | YAML 设计方案配置（Week 1-10） |
| `06-诊断报告/` | 标签诊断、覆盖率测试、改进计划 |
| `07-采样报告/` | Reddit/Trustpilot/Zendesk 采样报告 |
| `08-辅助数据/` | 多语言关键词映射、Zendesk 服务标签 |
| `labeling-latest/` | 最新打标输出（按数据源分目录） |
| `tag_gap_analysis/` | 标签缺口分析结果 |
| `unified_labeling/` | 统一打标结果（Phase 1~3 JSONL） |

---

## 最新打标结果

| 文件 | 记录数 | 覆盖率 | 说明 |
|------|--------|--------|------|
| `unified_labeling/phase3_p3_labeled.jsonl` | 364,569 | **78.97%** | **最终完整打标结果** |
| `unified_labeling/phase3_p2_labeled.jsonl` | 364,569 | 51.53% | P2 完成后结果 |
| `unified_labeling/phase3_p1_labeled.jsonl` | 364,569 | 49.28% | P1 完成后结果 |

### 按数据源覆盖率（最终）

| 数据源 | 总数 | 有标签 | 覆盖率 |
|--------|------|--------|--------|
| Amazon竞品 | 194,734 | 159,791 | **82.1%** |
| Momcozy自有 | 19,808 | 16,111 | **81.3%** |
| Zendesk | 47,204 | 40,793 | **86.4%** |
| Trustpilot | 99,853 | 69,412 | **69.5%** |
| Reddit | 2,970 | 1,757 | **59.2%** |

---

## 标签字典

| 版本 | 文件 | 标签总数 | 说明 |
|------|------|----------|------|
| v3.4 draft | `02-历史字典/tag_dictionary_v3.4_draft.xlsx` | 483 | Phase 2 草稿 |
| v3.4 filled | `02-历史字典/tag_dictionary_v3.4_filled.xlsx` | 483 | Phase 2 填充完成 |
| v3.5 final | `01-字典版本/tag_dictionary_v3.5_final.xlsx` | 503 | Phase 3 完成版（+20通用标签） |
| v3.6 final | `01-字典版本/tag_dictionary_v3.6_final.xlsx` | 587 | 字段补全版 |
| **v3.7** | **`01-字典版本/tag_dictionary_v3.7.xlsx`** | **569** | **最终版：噪声清理+全字段0空值** |

### 标签分布（v3.7）

| Sheet | 标签数 | 说明 |
|-------|--------|------|
| 01_通用标签主表 | 209 | 含20个通用体验标签 |
| 02_吸奶器 | 82 | |
| 03_内衣服饰 | 57 | |
| 04_家居家纺 | 52 | |
| 05_母婴综合护理 | 70 | |
| 06_喂养电器 | 53 | |
| 07_智能母婴电器 | 64 | |

---

## 审计报告

### v3.7 标签字典升级

| 报告 | 文件 | 关键数据 |
|------|------|----------|
| **v3.7 最终审计** | `03-审计报告/v3.7_final_audit_report.md` | 18个噪声标签清理，全字段补全，覆盖率99.85% |
| **v3.6 字段补全审计** | `03-审计报告/auto_fill_v36_audit_report.md` | 94个标签4字段补全，规则引擎+LLM推理 |

### Phase 3 覆盖率提升专项

| 报告 | 文件 | 关键数据 |
|------|------|----------|
| 最终综合报告 | `03-审计报告/phase3_final_summary.md` | 完整Phase 1~3总结 |
| P3 审计 | `04-审计数据/phase3_p3_audit.json` | 通用标签部署：100,024条新打标 |
| P2 审计 | `04-审计数据/phase3_p2_audit.json` | ALCHEmist：8,216条新打标 |
| P1 审计 | `04-审计数据/phase3_p1_audit.json` | 品线推断：27,358条新打标 |

### Phase 1~2 标签字典重构

| 报告 | 文件 | 关键数据 |
|------|------|----------|
| Phase 2.9 字典验证 | `04-审计数据/phase2_9_audit.json` | 483标签，0错误，0警告 |
| Phase 2.8 字段填充 | `04-审计数据/phase2_8_audit.json` | 匹配填充率 90%+ |
| Phase 2.6 ALCHEmist | `04-审计数据/phase2_6_audit.json` | 74个函数生成 |
| Phase 2.5 AL审核 | `04-审计数据/phase2_5_audit.json` | 1个待人工审核 |
| Phase 2.3~2.4 过滤 | `04-审计数据/phase2_3_4_audit.json` | 74个候选标签 |
| Phase 2.2 缺口检测 | `04-审计数据/phase2_2_audit.json` | 112个品类缺口 |
| Phase 2.1 零标签 | `04-审计数据/phase2_1_audit.json` | 211,919条零标签 |
| Phase 1.5 品线推断 | `unified_labeling/phase1_5_audit.json` | 覆盖率41.9% |
| Phase 1.3 萃取打标 | `unified_labeling/phase1_3_audit.json` | 152,650条有标签 |
| Phase 1.2 质量筛选 | `unified_labeling/phase1_2_audit.json` | 高质量率统计 |
| Phase 1.1 统一格式 | `unified_labeling/phase1_1_audit.json` | 364,569条统一 |

---

## 设计方案（YAML）

| 文件 | 内容 |
|------|------|
| `05-设计方案/week1-2_P0_fix_17_low_score_tags.yaml` | Week 1-2: 17个低分标签修复 |
| `05-设计方案/week1-2_P0_fix_20_missing_fields.yaml` | Week 1-2: 20个缺失字段补全 |
| `05-设计方案/week1-2_P0_aip1_dynamic_anchor_rules.yaml` | Week 1-2: AIPL 动态锚定规则 |
| `05-设计方案/week3-4_AIPL_tag_sink_to_6_categories.yaml` | Week 3-4: AIPL 标签下沉到6品线 |
| `05-设计方案/week5-6_positive_tags_and_channel_weights.yaml` | Week 5-6: 正向标签+渠道权重 |
| `05-设计方案/week7-8_multilingual_mapping_503_tags.yaml` | Week 7-8: 503标签多语言映射 |
| `05-设计方案/week9-10_persona_inference_quality_monitor_integration_test.yaml` | Week 9-10: 画像推导+质量监控+测试 |

---

## Label Functions

| 文件 | 标签数 | 说明 |
|------|--------|------|
| `alchemist_label_functions.py` | 74 | ALCHEmist 候选标签标注规则 |
| `zendesk_service_label_functions.py` | 10 | Zendesk 售中/售后 AIPL 标签 |
| `02-脚本工具/01-标签进化/general_tag_labeler.py` | 20 | 通用情感/体验/属性标签（含多语言） |

---

## 诊断报告

| 文件 | 说明 |
|------|------|
| `06-诊断报告/tag_diagnostic_report.json` | 标签诊断报告 |
| `06-诊断报告/tag_diagnostic_details.json` | 标签诊断详情 |
| `06-诊断报告/tag_improvement_plan.json` | 标签改进计划 |
| `06-诊断报告/v3.7_coverage_test_report.json` | v3.7 覆盖度测试报告 |

---

## 采样报告

| 文件 | 说明 |
|------|------|
| `07-采样报告/reddit_report.json` | Reddit 采样报告 |
| `07-采样报告/trustpilot_report.json` | Trustpilot 采样报告 |
| `07-采样报告/zendesk_momcozy_report.json` | Zendesk 采样报告 |
| `07-采样报告/report.json` | 综合报告 |

---

## 历史版本

历史打标输出已归档至 `../00-归档资料/labeling-outputs/`
