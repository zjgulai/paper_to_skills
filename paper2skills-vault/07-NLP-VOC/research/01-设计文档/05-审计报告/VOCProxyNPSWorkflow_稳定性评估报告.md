# VOCProxyNPSWorkflow 稳定性深度评估报告

> **评估日期**: 2026-04-22
> **评估范围**: `paper2skills-code/nlp_voc/proxy_nps_aipl_workflow/` 全套代码
> **数据规模**: 4 个高质量数据源合计 354,218 条 VOC 待打标

---

## 一、工作流架构总览

```
VOCProxyNPSWorkflow
├── Phase 1: 标签字典加载 (TagSeedDictionary)
├── Phase 2: 质量筛选 (ReviewQualityPipeline, 可选)
├── Phase 3: 统一标签萃取 (UnifiedLabelingPipeline)
│   ├── VOCLabelExtractor.extract() — 6维度并行
│   │   ├── _match_aipl_tags()      — 376标签关键词匹配
│   │   ├── _derive_aipl_stage()    — AIPL主阶段推导
│   │   ├── PersonaTagMatcher.match() — 55原子画像标签
│   │   ├── SentimentCalibrator.calibrate() — 预定义+ABSA校准
│   │   ├── BrandDetector.detect()   — 品牌提及检测
│   │   └── ProxyNPSCalculator.calculate() — NPS贡献计算
│   └── _aggregate_business_meta()   — 业务元数据聚合
└── Phase 4: 看板生成 (DashboardGenerator)
    ├── Proxy NPS 多维度拆分
    ├── AIPL 旅程漏斗
    ├── 驱动分析
    ├── 画像洞察
    └── 品牌分析
```

---

## 二、各组件稳定性评估

### 2.1 TagSeedDictionary（标签字典）⚠️ **阻塞**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 数据结构 | ✅ | `dict[str, TagSeed]` + 多索引，设计合理 |
| from_csv | ⚠️ | 只支持 CSV，实际标签字典是 xlsx 多 sheet |
| from_xlsx | ❌ | **不存在**，需新增 |
| 字段映射 | ⚠️ | 部分字段名需确认映射关系 |
| filter_by_line | ✅ | 通用标签 + 品线专属标签逻辑正确 |
| 实际标签数 | ⚠️ | 喂养电器 V3.2 final = **352 条**（179通用+163子表+5消毒器+5暖奶器），非 376 条 |
| 多品线支持 | ❌ | 缺少内衣服饰等品线标签字典 |

**阻塞详情**:
- 代码 `TagSeedDictionary.from_csv()` 期望 CSV 单文件格式
- 实际标签字典在 `/Users/pray/project/sgcs/.../SGCS_VOC标签字典_喂养电器V3.2_final.xlsx`，含 4 个 sheet
- 字段名映射大部分匹配，但 `主责部门` 在 xlsx 中有 `业务动作/责任部门` 和 `主责部门` 两个列

### 2.2 VOCLabelExtractor（核心萃取引擎）✅ **稳定**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| extract() 流程 | ✅ | 9 步顺序清晰，无循环依赖 |
| _match_aipl_tags() | ✅ | 多标签共存，支持否定词检测 |
| NEGATION_WORDS | ✅ | 已添加，但**仅对 L3 推荐标签生效** |
| _derive_aipl_stage() | ✅ | 按置信度加权投票 |
| _aggregate_business_meta() | ⚠️ | 取第一个非空值，非最高优先级排序 |

**已知问题**:
- 否定词检测仅作用于 `aipl_node == "L3"`（推荐意愿标签），其他节点标签（如 "噪音大" 在 "not noisy" 语境下）**未被否定检测覆盖**
- `_aggregate_business_meta` 中优先级聚合逻辑：遍历标签时取第一个非空值，不是真正的"取最高优先级"

### 2.3 SentimentCalibrator（情感校准器）✅ **稳定**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| ABSA 实现 | ✅ | **简化版基于词汇统计**，无外部模型依赖 |
| 预定义 vs ABSA 冲突检测 | ✅ | 冲突标记 + 保守策略（取预定义方向×0.5） |
| 程度副词 | ✅ | 12 个 intensifiers 覆盖 |
| 否定词 | ✅ | 12 个 negators 覆盖 |
| 星级评分融合 | ✅ | ABSA(0.7) + Rating(0.3) |

**说明**: ABSA 是简化规则版，非深度学习模型。优点是零依赖、可解释；缺点是精度有限（无上下文语义理解）。对 35 万条数据来说，这是合理取舍。

### 2.4 PersonaTagMatcher（画像标签匹配）⚠️ **部分可用**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| DEFAULT_ATOMIC_PERSONA_TAGS | ⚠️ | 只有 **15 条硬编码示例**，远少于 55 条 |
| match() 逻辑 | ✅ | 关键词命中 + 权重计分 |
| derive_business_persona() | ✅ | 共现模式计分，取最高 |
| 从外部加载 | ❌ | 无 CSV/Excel 加载方法 |

### 2.5 BrandDetector（品牌检测）✅ **稳定**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Momcozy 检测 | ✅ | "momcozy" 关键词匹配 |
| 竞品检测 | ✅ | Spectra/Medela/Willow/Elvie/Lansinoh |
| 品牌对比 | ✅ | "compared to", "better than" 等对比词 |

### 2.6 ProxyNPSCalculator（NPS 计算）✅ **稳定**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 多标签场景 | ✅ | 推荐意愿标签优先级最高 |
| 星级评分兜底 | ✅ | 无标签时按评分判定 |
| batch 计算 | ✅ | 聚合统计 + 百分比 |

### 2.7 DashboardGenerator（看板生成）✅ **稳定**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Proxy NPS 多维度 | ✅ | overall + by_product_line + by_persona + by_platform |
| AIPL 漏斗 | ✅ | 7 节点计数 + top themes |
| 驱动分析 | ✅ | detractor/promoter themes |
| 画像洞察 | ✅ | 渗透率 + 主题 + 情感 + NPS |
| 品牌分析 | ✅ | 提及率 + 对比率 |
| tag_coverage | ✅ | 覆盖率统计 |

### 2.8 VOCProxyNPSWorkflow（工作流编排）✅ **稳定**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 四阶段流程 | ✅ | 字典→质量→萃取→看板 |
| ReviewQuality 懒加载 | ⚠️ | sys.path 硬编码路径，耦合度高 |
| 测试结果 | ✅ | 4 条 demo 数据全部通过 |
| WorkflowResults.save() | ✅ | extractions.json + dashboard.json + summary.json |

---

## 三、接入 35.4 万条真实数据前的必要修复

### 🔴 阻塞项（必须修复）

| # | 问题 | 影响 | 修复方案 | 预估工时 |
|---|------|------|---------|---------|
| 1 | `TagSeedDictionary` 不支持 xlsx 多 sheet | 无法加载真实 352 条标签 | 新增 `from_xlsx()` 方法，合并所有 sheet | 30min |
| 2 | 字段名映射不完整 | 部分业务元数据为空 | 对照 xlsx 列名修正 `from_xlsx` 字段映射 | 20min |
| 3 | 标签总数 352 ≠ 376 | 文档声称 376 条但实际 352 条 | 确认其他品线标签字典位置，或更新文档 | 需确认 |
| 4 | 55 原子画像标签只有 15 条硬编码 | 画像推导覆盖不全 | 补充完整 55 条，或从 xlsx/CSV 加载 | 1-2h |

### 🟡 优化项（建议修复）

| # | 问题 | 影响 | 修复方案 |
|---|------|------|---------|
| 5 | 否定词检测仅对 L3 生效 | "not noisy" 可能误匹配 "噪音大" | 扩展否定检测到所有节点标签，或至少 I/L1/L2 节点 |
| 6 | `_aggregate_business_meta` 优先级逻辑 | 可能取到非最高优先级标签的元数据 | 按 P0>P1>P2>P3 排序后取第一个 |
| 7 | ReviewQualityPipeline 路径硬编码 | 模块耦合，路径变更即崩溃 | 改为构造函数注入或配置化 |
| 8 | `from_csv` 中 `applicable_line` 解析 | 通用标签的 `applicable_line=[]` 在 filter_by_line 中被正确包含，但需验证 | 添加单元测试 |

### 🟢 已知限制（可接受）

| # | 限制 | 说明 |
|---|------|------|
| 9 | ABSA 是简化版 | 基于词汇统计，无深度学习。对 35 万条数据足够，但长文本/复杂语境可能不准 |
| 10 | 多语言混合 | Zendesk 工单含英文/德文/西班牙文，ABSA 词库只有英文。非英文数据情感校准可能偏差 |
| 11 | 标签关键词覆盖 | 352 条标签的关键词可能无法覆盖所有 VOC 表达，需逆向完善 |

---

## 四、修复执行计划

```
Phase A: 标签字典接入（阻塞项 1-3）— 1h
  ├── A1: 新增 TagSeedDictionary.from_xlsx() 支持多 sheet
  ├── A2: 完善字段名映射（对照 xlsx 列名）
  ├── A3: 合并所有 sheet 生成统一标签列表
  └── A4: 运行 100 条小规模测试验证加载正确性

Phase B: 画像标签补全（阻塞项 4）— 1-2h
  ├── B1: 补充完整 55 原子画像标签到代码
  └── B2: 或实现从 CSV/Excel 加载

Phase C: 优化项修复 — 1h
  ├── C1: 扩展否定词检测到所有节点（或至少 I/L1/L2）
  ├── C2: 修复 _aggregate_business_meta 优先级排序
  └── C3: ReviewQualityPipeline 注入式重构

Phase D: 大规模测试 — 2h
  ├── D1: 1000 条 Amazon VOC 测试打标
  ├── D2: 检查标签覆盖率、冲突标记率
  ├── D3: 抽样人工复核标签准确性
  └── D4: 根据结果逆向完善标签关键词
```

---

## 五、测试验证策略

### 5.1 小规模测试（Phase A 后）

```python
# 测试标签字典加载
from proxy_nps_aipl_workflow import TagSeedDictionary

tag_dict = TagSeedDictionary.from_xlsx(
    "/path/to/SGCS_VOC标签字典_喂养电器V3.2_final.xlsx"
)
print(tag_dict.summary())
# 期望: total_tags ~352, by_aipl 各节点均有分布

# 测试单条萃取
from proxy_nps_aipl_workflow import VOCLabelExtractor, VOCRecord

extractor = VOCLabelExtractor(tag_dict=tag_dict)
voc = VOCRecord(
    review_id="TEST001",
    text="The suction is weak and it's too noisy at night.",
    source_type="review",
    platform="amazon",
    spu_code="SPU001",
    product_line="breast_pump",
    category="wearable_pump",
    rating=2.0,
)
result = extractor.extract(voc)
print(result.aipl_tags)  # 期望命中 [suction_weak, noise_loud] 等
print(result.sentiment_calibration)  # 期望 "calibrated"
```

### 5.2 中等规模测试（Phase C 后）

- 取 1,000 条 Amazon 高质量评论
- 运行完整工作流
- 检查：
  - 标签覆盖率（期望 >60% VOC 至少命中 1 个标签）
  - 冲突标记率（期望 <5%）
  - 每标签平均命中 VOC 数（检查长尾分布）
  - AIPL 漏斗分布是否合理

### 5.3 大规模测试（Phase D 后）

- 分批处理 20 万条 Amazon + 10 万条 Trustpilot
- 每批 5,000-10,000 条，避免内存溢出
- 输出中间结果，支持断点续跑

---

## 六、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 标签关键词覆盖不足导致大量 VOC 无标签 | 高 | 指标失真 | Phase D 逆向完善，人工补标采样校准 |
| 非英文数据（德/西/法）情感校准偏差 | 中 | 多语言场景 NPS 不准 | 标记 language 字段，非英文数据降低情感权重 |
| 35 万条处理时间过长 | 中 | 迭代周期长 | 分批次处理，每批持久化中间结果 |
| 标签字典字段名变更导致加载失败 | 低 | 代码崩溃 | 字段名映射配置文件化，加 fallback |
| 多标签冲突导致业务解读混乱 | 中 | 决策偏差 | 完善冲突消解规则文档，增加人工复核队列 |

---

## 七、结论与建议

### 7.1 总体评估

**工作流架构稳定性: 7.5/10**

- ✅ 核心萃取引擎设计合理，6 维度并行，测试通过
- ✅ 情感校准、品牌检测、NPS 计算均为纯规则方法，零外部依赖
- ⚠️ **标签字典加载是最大阻塞项** — 需新增 xlsx 多 sheet 支持
- ⚠️ 画像标签和否定词检测覆盖不完整
- ❌ 缺少多品线标签字典（仅喂养电器 352 条）

### 7.2 是否可以接入真实数据？

**建议: 先完成 Phase A（标签字典接入），再接入真实数据。**

理由:
1. 当前 demo 标签只有 12 条，真实标签 352 条，关键词密度和覆盖范围完全不同
2. 字段名映射未验证，直接跑 35 万条可能大量元数据为空
3. 修复 Phase A 仅需约 1 小时，阻塞风险低

### 7.3 执行顺序建议

```
立即执行: Phase A（标签字典接入） → 1h
  ↓
并行执行: Phase B（画像标签补全）+ Phase C（优化项） → 2h
  ↓
验证执行: Phase D（1000条测试 → 人工复核 → 逆向完善） → 2-4h
  ↓
大规模执行: 35.4万条分批打标 → 视数据量而定
```

---

*报告版本: 2026-04-22*
*关联文件: `unified_label_extraction.py`, `workflow.py`, `__init__.py`*
*标签字典源: `/Users/pray/project/sgcs/.../SGCS_VOC标签字典_喂养电器V3.2_final.xlsx`*
