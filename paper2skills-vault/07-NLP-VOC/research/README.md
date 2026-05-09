# VOC 统一萃取引擎与标签字典重构 — 研究目录

> **当前状态**: Phase 1~3 已完成，标签字典 v3.7 已发布
> **最终覆盖率**: 78.97% (287,864 / 364,569)
> **标签总数**: 569 (通用209 + 品线专属360)
> **更新日期**: 2026-04-24

---

## 目录结构

```
research/
├── 00-归档资料/            # 历史版本、中间产物和重复文件归档
├── 01-设计文档/            # 设计文档、调研报告、数据资产盘点
├── 02-脚本工具/            # 业务适配脚本（数据采样、打标入口、诊断工具）
│   ├── tag-evolution/      # 标签进化工作流核心脚本（Phase 1~3）
│   ├── data-sampling/      # Reddit/Trustpilot/Zendesk 数据采样
│   ├── batch-labeling/     # 批量打标入口脚本
│   ├── nps-pipeline/       # NPS校准、看板生成流程
│   ├── diagnostics/        # 标签诊断、覆盖率测试
│   └── data-processing/    # 标签词典扩展、品线矩阵生成
├── 03-数据资产/            # 数据资产（产品主数据、原始VOC、关键词库）
├── 04-输出结果/            # 打标输出、审计报告、标签字典
└── .gitignore              # 忽略临时文件和大数据文件
```

---

## 工作流总览（3 Phase）

### Phase 1: 数据统一加载与萃取打标

| Step | 脚本 | 产出 | 关键指标 |
|------|------|------|----------|
| 1.1 统一输入格式 | `tag-evolution/unify_voc_input.py` | 统一 VOCRecord JSONL | 364,569 条 |
| 1.2 质量筛选 | `tag-evolution/quality_filter.py` | 高质量子集 + 质量分 | 均值 94.3 |
| 1.3 萃取引擎打标 | `tag-evolution/transcribe_v33_labels.py` + `incremental_labeling.py` | v3.3 转录 + 增量打标 | 152,650 条有标签 |
| 1.4~1.5 品线推断与输出 | `tag-evolution/infer_product_line.py` | 品线推断 + 统一输出 | 覆盖率 41.9% |

### Phase 2: 逆向分析与标签字典更新

| Step | 脚本 | 产出 | 关键指标 |
|------|------|------|----------|
| 2.1 零标签提取 | `tag-evolution/zero_label_extractor.py` | 零标签样本 + 品类分布 | 211,919 条零标签 |
| 2.2 缺口检测 | `tag-evolution/gap_detector.py` | 112 个品类缺口 | 候选标签池 |
| 2.3~2.4 候选标签过滤 | `tag-evolution/candidate_tag_filter.py` | 过滤后候选标签 | 74 个 |
| 2.5 Active-Learning 审核 | `tag-evolution/active_learning_audit.py` | 人工审核清单 | 1 个待审核 |
| 2.6 ALCHEmist Label Function | `tag-evolution/alchemist_label_generator.py` | 74 个 Python 标注规则 | 可审计 |
| 2.7 字典更新 | `tag-evolution/tag_dictionary_updater.py` | 标签字典草稿 | 483 标签 |
| 2.8 V3.0 字段补充 | `tag-evolution/v3_field_mapper.py` | 填充后字典 | 匹配 90%+ |
| 2.9 字典验证 | `tag-evolution/dictionary_validator.py` | 验证报告 | 通过 |

### Phase 3: 覆盖率提升 + 字典进化（v3.5 → v3.7）

| 子阶段 | 措施 | 覆盖率提升 | 关键产出 |
|--------|------|-----------|----------|
| P1 | 产品主数据补全 + Amazon竞品品线推断 | +3.8% | 147 SPU 品线映射 |
| P2 | ALCHEmist 74标签 + Zendesk售中售后10标签 + 多语言 | +5.83% | 84 个 Label Functions |
| **P3** | **通用情感/体验/属性标签（20个）+ 德/法/英多语言** | **+27.44%** | `general_tag_labeler.py` |
| **v3.6** | **94个标签字段自动补全（策略包/主责部门/优先级/原子指标）** | — | 规则引擎 + LLM 推理 |
| **v3.7** | **噪声标签清理（18个）+ VOC数据驱动字段补全 + 全字段0空值** | — | `tag_dictionary_v3.7.xlsx` |
| **最终** | | **78.97%** | `tag_dictionary_v3.7.xlsx` (569标签) |

---

## 核心交付物索引

### 标签字典

| 版本 | 路径 | 说明 |
|------|------|------|
| v3.4 | `04-输出结果/02-历史字典/tag_dictionary_v3.4_filled.xlsx` | Phase 2 完成版（483标签） |
| v3.5 | `04-输出结果/01-字典版本/tag_dictionary_v3.5_final.xlsx` | Phase 3 完成版（503标签） |
| v3.6 | `04-输出结果/01-字典版本/tag_dictionary_v3.6_final.xlsx` | 字段补全版（587标签） |
| **v3.7** | **`04-输出结果/01-字典版本/tag_dictionary_v3.7.xlsx`** | **最终版（569标签，全字段0空值）** |

### 最终打标结果

| 文件 | 记录数 | 说明 |
|------|--------|------|
| `04-输出结果/unified_labeling/phase3_p3_labeled.jsonl` | 364,569 | 最终完整打标结果 |

### 审计报告

| 报告 | 路径 |
|------|------|
| v3.7 最终审计 | `04-输出结果/03-审计报告/v3.7_final_audit_report.md` |
| v3.6 字段补全审计 | `04-输出结果/03-审计报告/auto_fill_v36_audit_report.md` |
| Phase 3 最终总结 | `04-输出结果/03-审计报告/phase3_final_summary.md` |

---

## 各子目录说明

- [00-归档资料/README.md](00-归档资料/README.md) — 归档层说明
- [01-设计文档/README.md](01-设计文档/README.md) — 设计文档索引
- [02-脚本工具/README.md](02-脚本工具/README.md) — 脚本层说明
- [02-脚本工具/tag-evolution/README.md](02-脚本工具/tag-evolution/README.md) — 标签进化完整工作流
- [03-数据资产/README.md](03-数据资产/README.md) — 数据资产清单
- [04-输出结果/README.md](04-输出结果/README.md) — 输出层索引
