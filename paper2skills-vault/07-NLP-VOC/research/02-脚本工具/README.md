# 02-脚本工具: 脚本层

本目录存放业务适配脚本、入口脚本和诊断工具。

**注意**: 核心算法实现位于 `paper2skills-code/nlp_voc/` 目录，本目录仅存放：
- 数据采样/入口调用脚本
- 业务特定的适配脚本
- 诊断/验证工具脚本
- 标签进化工作流脚本

---

## 子目录

| 子目录 | 内容 | 状态 |
|--------|------|------|
| `01-标签进化/` | **标签进化工作流（Phase 1~3）** — 统一萃取、逆向分析、覆盖率提升 | 已完成 |
| `02-数据采样/` | Reddit/Trustpilot/Zendesk 数据采样脚本 | 已完成 |
| `03-批量打标/` | 批量打标入口脚本 | 已完成 |
| `05-NPS管道/` | NPS 校准、看板生成流程脚本 | 已完成 |
| `06-诊断工具/` | 标签诊断、覆盖率测试、快速验证脚本 | 已完成 |
| `04-数据处理/` | 标签词典扩展、品线矩阵生成、关键词挖掘 | 已完成 |

---

## 01-标签进化/ 子目录（核心）

标签进化工作流是 VOC 萃取体系的核心迭代引擎，覆盖数据统一 → 萃取打标 → 逆向分析 → 字典更新 → 覆盖率提升的完整闭环。

详见 [01-标签进化/README.md](01-标签进化/README.md)。

### 关键脚本速查

| 脚本 | Phase | 功能 |
|------|-------|------|
| `unify_voc_input.py` | 1.1 | 统一各数据源到 VOCRecord 格式 |
| `quality_filter.py` | 1.2 | ReviewQualityScorer 质量筛选 |
| `transcribe_v33_labels.py` | 1.3 | v3.3 打标结果转录 |
| `incremental_labeling.py` | 1.3 | 增量关键词打标 |
| `infer_product_line.py` | 1.4~1.5 | 品线/品类推断 |
| `zero_label_extractor.py` | 2.1 | 零标签 VOC 提取 |
| `gap_detector.py` | 2.2 | 缺口检测 |
| `candidate_tag_filter.py` | 2.3~2.4 | 候选标签过滤与合并 |
| `active_learning_audit.py` | 2.5 | Active-Learning 质量把关 |
| `alchemist_label_generator.py` | 2.6 | ALCHEmist Label Function 生成 |
| `tag_dictionary_updater.py` | 2.7 | 标签字典自动更新 |
| `v3_field_mapper.py` | 2.8 | V3.0 增量字段补充 |
| `dictionary_validator.py` | 2.9 | 标签字典结构验证 |
| `final_audit.py` | — | 最终综合审计 |
| `final_summary.py` | — | 最终综合报告生成 |
| `general_tag_labeler.py` | 3.P3 | 通用情感/体验/属性标签打标器（20标签，多语言） |

---

## 外部依赖

所有脚本通过 `sys.path.insert` 引用：

```python
sys.path.insert(0, "../../../paper2skills-code/nlp_voc/proxy_nps_aipl_workflow")
```

依赖的核心模块：
- `proxy_nps_aipl_workflow/unified_label_extraction.py` — 萃取引擎
- `review_quality_scoring/` — 质量评分
- `alchemist_weak_supervision/` — ALCHEmist Label Function
