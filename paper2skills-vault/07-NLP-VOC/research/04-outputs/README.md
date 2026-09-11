# 04-outputs: 输出层

本目录存放当前有效的打标输出、诊断报告、审计文件和标签字典。

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
| v3.4 draft | `tag_dictionary_v3.4_draft.xlsx` | 483 | Phase 2 草稿 |
| v3.4 filled | `tag_dictionary_v3.4_filled.xlsx` | 483 | Phase 2 填充完成 |
| **v3.5 final** | **`tag_dictionary_v3.5_final.xlsx`** | **503** | **Phase 3 最终版（+20通用标签）** |

### 标签分布（v3.5）

| Sheet | 标签数 | 新增 | 说明 |
|-------|--------|------|------|
| 01_通用标签主表 | 191 | **+20** | 含Phase 3新增通用体验标签 |
| 02_吸奶器 | 72 | +4 | |
| 03_内衣服饰 | 47 | +6 | |
| 04_家居家纺 | 46 | +11 | |
| 05_母婴综合护理 | 58 | +19 | |
| 06_喂养电器 | 46 | +7 | |
| 07_智能母婴电器 | 43 | +27 | |

---

## 审计报告

### Phase 3 覆盖率提升专项

| 报告 | 文件 | 关键数据 |
|------|------|----------|
| **最终综合报告** | `phase3_final_summary.md` | 完整Phase 1~3总结 |
| P3 审计 | `phase3_p3_audit.json` | 通用标签部署：100,024条新打标 |
| P2 审计 | `phase3_p2_audit.json` | ALCHEmist：8,216条新打标 |
| P1 审计 | `phase3_p1_audit.json` | 品线推断：27,358条新打标 |

### Phase 1~2 标签字典重构

| 报告 | 文件 | 关键数据 |
|------|------|----------|
| Phase 2.9 字典验证 | `phase2_9_audit.json` | 483标签，0错误，0警告 |
| Phase 2.8 字段填充 | `phase2_8_audit.json` | 匹配填充率 90%+ |
| Phase 2.6 ALCHEmist | `phase2_6_audit.json` | 74个函数生成 |
| Phase 2.5 AL审核 | `phase2_5_audit.json` | 1个待人工审核 |
| Phase 2.3~2.4 过滤 | `phase2_3_4_audit.json` | 74个候选标签 |
| Phase 2.2 缺口检测 | `phase2_2_audit.json` | 112个品类缺口 |
| Phase 2.1 零标签 | `phase2_1_audit.json` | 211,919条零标签 |
| Phase 1.5 品线推断 | `unified_labeling/phase1_5_audit.json` | 覆盖率41.9% |
| Phase 1.3 萃取打标 | `unified_labeling/phase1_3_audit.json` | 152,650条有标签 |
| Phase 1.2 质量筛选 | `unified_labeling/phase1_2_audit.json` | 高质量率统计 |
| Phase 1.1 统一格式 | `unified_labeling/phase1_1_audit.json` | 364,569条统一 |

---

## Label Functions

| 文件 | 标签数 | 说明 |
|------|--------|------|
| `alchemist_label_functions.py` | 74 | ALCHEmist 候选标签标注规则 |
| `zendesk_service_label_functions.py` | 10 | Zendesk 售中/售后 AIPL 标签 |
| `general_tag_labeler.py` | 20 | 通用情感/体验/属性标签（含多语言） |

---

## 历史诊断报告

| 文件 | 说明 |
|------|------|
| `tag_diagnostic_report.json` | 标签诊断报告 |
| `tag_diagnostic_details.json` | 标签诊断详情 |
| `tag_improvement_plan.json` | 标签改进计划 |
| `reddit_report.json` | Reddit 采样报告 |
| `trustpilot_report.json` | Trustpilot 采样报告 |
| `zendesk_momcozy_report.json` | Zendesk 采样报告 |
| `report.json` | 综合报告 |

---

## 历史版本

历史打标输出已归档至 `../00-archived/labeling-outputs/`
