# VOC Proxy NPS 跨源偏差校准方法论

## 1. 问题定义

当 VOC 数据来自多个异构源（Amazon 评论、Reddit 讨论、Trustpilot 评分、Zendesk 工单），各源存在**样本选择偏差**：

- Reddit 用户主动吐槽（~51.8% 负面）→ 天然 NPS 偏低
- Trustpilot 用户主动好评（~62.2% 推荐）→ 天然 NPS 偏高
- Zendesk 全是投诉工单 → 100% 负面
- Amazon 评论分布相对中性 → 作为全局基准

**后果**：直接聚合各源 NPS 会得到一个有偏的全局指标，不可比、不可信。

## 2. 核心思想

**校准 = 减去数据源的系统性偏差，使各源 NPS 对齐到同一基准。**

类比：不同温度计有自己的零点偏移。直接读数不可比，先减去零点偏移，再比较。

## 3. 三层校准体系

```
Layer 1: 原始指标（Raw NPS）
    ↓ 减去 source_bias
Layer 2: 数据源校准指标（Source-Calibrated NPS）
    ↓ 减去 category_bias
Layer 3: 品类相对指标（Category-Relative NPS）
```

### 3.1 Layer 1 — 原始指标

保留各数据源的原始 NPS，确保数据透明。任何分析都可回溯到原始值。

公式：
```
raw_nps = promoter_pct - detractor_pct
```

### 3.2 Layer 2 — 数据源校准

**核心公式：**
```
calibrated_nps = raw_nps - source_bias
where source_bias = source_baseline_nps - global_baseline_nps
```

**步骤：**
1. 选定全局基准源（建议：样本量最大、情绪分布最中性的源）
2. 计算各源的原始 NPS 作为其情绪基线
3. 各源偏差 = 该源基线 - 全局基准基线
4. 每个观测 NPS 减去对应 source_bias

**示例：**
| 数据源 | 原始 NPS | source_bias | 校准后 |
|--------|---------|-------------|--------|
| Amazon | 21.7 | 0.0 | 21.7 |
| Reddit | -2.3 | -24.0 | 21.7 |
| Trustpilot | 44.2 | +22.6 | 21.6 |
| Zendesk | 7.6 | -14.0 | 21.6 |

### 3.3 Layer 3 — 品类标准化

不同品类天然情绪差异大（如消毒器 NPS 81.3 vs 磨甲器可能极低），跨品类直接对比不公平。

**公式：**
```
category_relative_nps = calibrated_nps - category_avg_nps
```

其中 `category_avg_nps` 为该品类在所有已校准数据上的平均 NPS。

**解读：**
- 正值 = 该源在该品类上表现优于品类平均水平
- 负值 = 表现低于品类平均水平

## 4. 完整工作流

```
Step 1: 数据准备
    ├── 各数据源必须提供：总记录数、推荐者%、贬损者%
    └── 验证数据完整性（样本量 > 1000）

Step 2: 选定全局基准
    ├── 候选源评估：样本量、情绪分布中性度、业务代表性
    └── 推荐：Amazon（电商评论，分布相对均衡，样本最大）

Step 3: 计算源偏差
    ├── 对每个源：source_nps = promoter_pct - detractor_pct
    ├── source_bias = source_nps - global_baseline_nps
    └── 记录偏差方向和幅度

Step 4: 执行校准
    ├── 对每个观测：calibrated = raw - source_bias
    └── 输出 Layer 2 指标

Step 5: 品类基线计算（可选，需要品类标签）
    ├── 按品类分组，计算各品类的平均 NPS
    ├── 过滤样本量 < 10 的品类（不稳定）
    └── 输出品类基线表

Step 6: 品类标准化（可选）
    ├── 对每个 (源, 品类) 组合：relative = calibrated - category_avg
    └── 输出 Layer 3 指标

Step 7: 自证审计（强制）
    ├── 检查 1: 偏差是否在合理范围 (-50, +50)
    ├── 检查 2: 校准后 NPS 是否在理论范围 (-100, 100)
    ├── 检查 3: 校准变化幅度是否合理 (< 50)
    ├── 检查 4: 基准源自洽（基准源校准后不变）
    └── 检查 5: 样本量充足性 (> 1000)

Step 8: 输出报告
    ├── JSON 校准报告（完整元数据 + 审计结果）
    ├── CSV 三层指标看板（BI 可用）
    └── Markdown 摘要（人可读）
```

## 5. 自证审计清单

| 检查项 | 通过标准 | 失败处理 |
|--------|---------|---------|
| 偏差合理性 | \|bias\| < 50 | 拒绝该校准，复核数据源定义 |
| NPS 范围 | -100 ≤ calibrated ≤ 100 | 检查原始数据是否有异常值 |
| 校准变化 | \|shift\| < 50 | 同上 |
| 基准自洽 | 基准源校准后变化 < 0.01 | 检查偏差计算逻辑 |
| 样本充足 | 样本量 ≥ 1000 | 标记为不稳定，降低权重 |

**审计状态规则：**
- 有任何 issue → FAIL，必须修复后才能使用
- 无 issue 有 warning → PASS，但需人工确认 warning
- 全通过 → PASS，可直接使用

## 6. 代码快速接入

```python
from nps_calibration import NPSCalibrator, NPSAuditor, CategoryStandardizer, NPSDashboard

# Step 1: 初始化校准器（使用默认基线或自定义）
calibrator = NPSCalibrator()

# Step 2: 单条校准
result = calibrator.calibrate("reddit", raw_nps=-2.3)
print(f"校准后: {result.calibrated_nps:.1f}")

# Step 3: DataFrame 批量校准
df_calibrated = calibrator.calibrate_dataframe(df, source_col="source", nps_col="nps")

# Step 4: 品类标准化
standardizer = CategoryStandardizer(category_stats={"消毒器": {"avg_nps": 81.3, "sample_size": 155}})

# Step 5: 生成三层看板
dashboard = NPSDashboard(calibrator, standardizer)
df_dashboard = dashboard.generate(records)

# Step 6: 自证审计
auditor = NPSAuditor(calibrator)
audit = auditor.audit(calibrated_results)
assert audit["status"] == "PASS"
```

## 7. 关键假设与边界

**假设：**
1. 各数据源的系统性偏差在短期内稳定（不随时间剧烈变化）
2. 全局基准源的选择具有业务合理性
3. 品类标签的准确性和一致性

**不适用场景：**
- 单数据源分析（无跨源可比需求）
- 数据源样本量 < 1000（基线估计不稳定）
- 数据源定义频繁变化（偏差会漂移）

**维护频率：**
- 源基线：每季度复核一次
- 品类基线：每月更新一次（新数据进入）
- 全局基准：仅在新增/移除主要数据源时重新评估

## 8. 输出物清单

| 输出物 | 格式 | 用途 |
|--------|------|------|
| 校准报告 | JSON | 存档、程序化消费 |
| 三层看板 | CSV | BI 导入、Tableau/PowerBI |
| 摘要文档 | Markdown | 人读、汇报 |
| 审计日志 | JSON 内嵌 | 质量追溯 |

## 9. 复制到新项目的检查清单

- [ ] 收集所有数据源的 (总记录数, 推荐者%, 贬损者%)
- [ ] 选定全局基准源（建议样本最大 + 情绪最中性）
- [ ] 确认各数据源命名一致（大小写、空格等）
- [ ] 准备品类标签（如需 Layer 3）
- [ ] 运行校准引擎
- [ ] 审计必须 PASS
- [ ] 将校准后 NPS 接入下游指标 pipeline
- [ ] **新增数据源**: 若数据源不在默认基线中，收集至少 1000 条后计算其 (promoter_pct, detractor_pct)，调用 `NPSCalibrator(baselines=custom_baselines)` 注入
- [ ] **基线维护**: 建立 cron/定时任务，每月更新品类基线，每季度复核源基线
