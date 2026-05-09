# VOC 萃取覆盖率提升最终报告

## 执行概览

| 指标 | 数值 |
|------|------|
| 总 VOC 记录 | 364,569 |
| 起始覆盖率 | 41.9% (2024-04-22) |
| **最终覆盖率** | **78.97%** |
| **总提升** | **+37.07%** |
| 零标签记录 | 76,705 (21.0%) |

---

## 各阶段贡献

```
41.9% ── 原始标签 (v3.3 + 增量)
       + 9.63% ── Phase 3 P1+P2 (品线推断 + ALCHEmist 74标签 + Zendesk售中售后 + 多语言)
51.53% ── Phase 3 P2 完成
       + 27.44% ── Phase 3 P3 (通用情感/体验/属性标签 + 多语言关键词)
78.97% ── Phase 3 P3 完成
```

| 阶段 | 措施 | 覆盖率提升 | 关键产出 |
|------|------|-----------|---------|
| P1 | Momcozy品线映射补全 + Amazon竞品关键词推断 | +3.8% | 147条SPU品线映射 |
| P2 | ALCHEmist 74标签 + Zendesk售中售后10标签 + 多语言关键词 | +5.83% | 74个Label Functions |
| **P3** | **通用情感/体验/属性标签 + 德/法/英多语言情感关键词** | **+27.44%** | **18个通用标签** |

---

## 按数据源覆盖率

| 数据源 | 总数 | 有标签 | 覆盖率 | 关键标签 |
|--------|------|--------|--------|---------|
| Amazon竞品 | 194,734 | 159,791 | **82.1%** | general_positive, comfort_experience, strong_recommendation |
| Momcozy自有 | 19,808 | 16,111 | **81.3%** | comfort_experience, strong_recommendation, ease_of_use |
| Zendesk | 47,204 | 40,793 | **86.4%** | 售中/售后AIPL标签, service_satisfaction |
| Trustpilot | 99,853 | 69,412 | **69.5%** | 德/法情感关键词, general_positive, quality_perception |
| Reddit | 2,970 | 1,757 | **59.2%** | strong_recommendation, performance_satisfaction |

---

## Phase 3 P3 通用标签体系（18个标签）

### 情感/体验维度（8个）

| 标签ID | 英文 | 中文 | AIPL | 匹配数 |
|--------|------|------|------|--------|
| TAG_GEN_E001 | ease_of_use | 易用性 | L1 | 11,125 |
| TAG_GEN_E002 | comfort_experience | 舒适体验 | L1 | 5,234 |
| TAG_GEN_E003 | quality_perception | 质量感知 | L1 | 16,741 |
| TAG_GEN_E004 | appearance_design | 外观设计 | L1 | 5,684 |
| TAG_GEN_E005 | durability | 耐用性 | L1 | 2,160 |
| TAG_GEN_E006 | performance_satisfaction | 性能满意 | L1 | 2,596 |
| TAG_GEN_E007 | price_value_positive | 性价比正面 | P1 | 1,937 |
| TAG_GEN_E008 | size_fit_positive | 尺码合身 | L1 | 1,447 |

### 场景维度（4个）

| 标签ID | 英文 | 中文 | AIPL | 匹配数 |
|--------|------|------|------|--------|
| TAG_GEN_S001 | work_scenario | 工作场景 | L1 | 1,288 |
| TAG_GEN_S002 | night_use_scenario | 夜间使用 | L1 | 7,251 |
| TAG_GEN_S003 | travel_scenario | 旅行场景 | L1 | 2,951 |
| TAG_GEN_S004 | strong_recommendation | 强烈推荐 | L3 | 25,544 |

### 产品属性维度（5个）

| 标签ID | 英文 | 中文 | AIPL | 匹配数 |
|--------|------|------|------|--------|
| TAG_GEN_A001 | color_mention | 颜色提及 | L1 | 5,238 |
| TAG_GEN_A002 | material_mention | 材质提及 | L1 | 5,768 |
| TAG_GEN_A003 | wireless_handsfree | 无线免提 | L1 | 1,125 |
| TAG_GEN_A004 | portable_lightweight | 便携轻量 | L1 | 1,163 |
| TAG_GEN_A005 | quiet_operation | 静音运行 | L1 | 566 |

### 服务/配送维度（3个）

| 标签ID | 英文 | 中文 | AIPL | 匹配数 |
|--------|------|------|------|--------|
| TAG_GEN_D001 | delivery_satisfaction | 配送满意 | P2 | 10,066 |
| TAG_GEN_C001 | service_satisfaction | 服务满意 | L3 | 6,807 |
| TAG_GEN_P001 | general_positive | 总体正面 | L3 | 51,967 |

---

## 关键突破：多语言情感关键词

Trustpilot 零标签中 **~65% 为德语/法语评论**，原有英文关键词完全无法匹配。

**德语关键词映射示例**：
- `sehr zufrieden` / `sehr gut` / `alles bestens` → quality_perception
- `schnelle lieferung` / `schneller versand` → delivery_satisfaction
- `gerne wieder` / `immer wieder` → strong_recommendation
- `guter service` / `freundlich` → service_satisfaction

**法语关键词映射示例**：
- `très satisfait` / `bonne qualité` → quality_perception
- `livraison rapide` / `reçu rapidement` → delivery_satisfaction
- `je recommande` → strong_recommendation
- `bon service` / `service client` → service_satisfaction

多语言支持为 Trustpilot 单独贡献 **~30,000** 条新标签，覆盖率从 26.5% 提升至 69.5%。

---

## 质量保障措施

1. **整词边界匹配**：单字符关键词（如 red, tan, top, super）使用 `\bword\b` 正则匹配，避免 "ordered" 误匹配 "red"
2. **否定词检测**：匹配词前25字符内检测 not/no/never/n't 等否定词，自动反转情感极性
3. **否定短语过滤**："not easy", "doesn't fit", "wouldn't recommend" 等直接过滤
4. **零标签优先**：通用标签只应用于已有零标签记录，避免与现有标签冲突

---

## 剩余 76,705 条零标签分析

| 数据源 | 零标签数 | 占比 | 特征 |
|--------|---------|------|------|
| Amazon竞品 | 34,943 | 45.6% | 混合情感、短文本、产品型号提及 |
| Trustpilot | 30,441 | 39.7% | 大量非Momcozy品牌评论（Willow、Philips等） |
| Zendesk | 6,411 | 8.4% | 极短文本（"取消订单"、"不想要了"）、图片附件 |
| Momcozy | 3,697 | 4.8% | 负面缺陷描述（现有标签未覆盖的细分问题） |
| Reddit | 1,213 | 1.6% | 对比讨论、技术问题 |

**进一步突破路径**（如需从 79% → 85%+）：
- 添加通用负面情感标签（size_fit_negative、performance_negative）
- 补全更多德/法情感词（bien、gut、einfach）
- 针对Zendesk短文本设计极简标签规则
- 扩展产品缺陷关键词库

---

## 产出文件

| 文件 | 说明 |
|------|------|
| `04-outputs/unified_labeling/phase3_p3_labeled.jsonl` | 最终打标结果（364,569条） |
| `04-outputs/phase3_p3_audit.json` | P3 审计报告 |
| `02-scripts/tag-evolution/general_tag_labeler.py` | 通用标签Label Function（18个标签，含多语言） |
