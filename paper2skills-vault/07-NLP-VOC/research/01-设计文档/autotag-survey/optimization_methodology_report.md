# VOC 品线分类优化方法论报告

## 一、优化概览

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| 总记录数 | 354,218 | 354,218 | — |
| other 率 | 66.0% | 59.4% | **-6.6pp** |
| 覆盖品线数 | 26 | 26 | 不变 |
| 迭代轮次 | — | 2 轮关键词扩展 | — |

---

## 二、五级分类架构

```
Level 1: Service 分流（物流/客服/退货关键词）→ customer_service
Level 2: 强特征词（高区分度关键词）→ 直接命中
Level 3: 关键词(stemming) + 上下文权重 → 加权投票
Level 4: 品牌推断（独家品牌直接命中 / 非独家品牌验证上下文）
Level 5: 标签辅助推断（AIPL 标签共现模式）→ 兜底
```

**核心设计原则**：
- 越靠前的 level 置信度越高，命中即返回
- 每个 level 的阈值独立可调
- 支持词干化（`simple_stem`）和连字符变体（`hands free` ↔ `hands-free`）

---

## 三、关键词优化迭代工作流

### 3.1 迭代闭环

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ① 全量分类运行  │ ──→ │  ② 提取 other   │ ──→ │  ③ LLM采样分类  │
│   (规则引擎)     │     │  (未命中文本)    │     │  (2000条样本)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ↑                                               │
         │                                               ↓
         │                                        ┌─────────────────┐
         │                                        │  ④ 分析 LLM     │
         │                                        │   分类结果       │
         │                                        │  (关键词提取)    │
         │                                        └─────────────────┘
         │                                               │
         │                                               ↓
         │                                        ┌─────────────────┐
         └────────────────────────────────────────│  ⑤ 更新规则     │
                                                  │  (关键词/阈值)   │
                                                  └─────────────────┘
```

### 3.2 每轮迭代的关键动作

| 步骤 | 动作 | 工具/方法 | 产出 |
|------|------|----------|------|
| ① | 全量跑规则分类 | `generate_product_line_spu_matrix.py` | other 率基线 |
| ② | 提取 other 文本 | 筛选 `product_line == "other"` | 待分析语料 |
| ③ | LLM 采样分类 | Claude API, batch_size=30 | 2000 条分类结果 |
| ④ | 关键词模式挖掘 | 高频词提取 + 人工审核 | 候选关键词列表 |
| ⑤ | 更新规则并重跑 | 修改 `PRODUCT_LINE_RULES`/`CONTEXT_RULES` | 新的 other 率 |

### 3.3 本轮两轮迭代的具体改进

**第一轮（基线 → 初步优化）**：
- 从 other 中提取高频实词，发现 `body shaper`、`tummy control` 等 postpartum_recovery 强信号
- 发现 `biberon`（法语奶瓶）、`dresses` 等跨语言/跨品类关键词

**第二轮（精细调整）**：
- 新增 **EXCLUSIVE_BRANDS** 机制：单一产品品牌（elvie, willow, nanit 等）直接命中，无需上下文验证
- 修复品牌推断 bug：`ctx_words[:5]` → `ctx_words`（原只查前5个上下文词）
- 阈值微调：wearable_breast_pump 从 5 提到 7（减少误匹配），pregnancy_pillow/postpartum_recovery 从 5 降到 4
- 多语言物流词扩展（德/法/意/西）

---

## 四、自证机制设计

### 4.1 数据完整性自证

```python
# 核心断言：输入输出记录数必须一致
total_input = sum(1 for ... in all_input_files)
total_output = len(df_matrix)
assert total_input == total_output, f"数据不一致: {total_input} vs {total_output}"

# 品线分布变化自证
for line in all_lines:
    before = baseline_counts.get(line, 0)
    after = new_counts.get(line, 0)
    delta = after - before
    if abs(delta) > 1000:
        print(f"  ⚠️ {line}: {before:,} → {after:,} (Δ{delta:+,.0f})")
```

### 4.2 规则合理性自证

- **LLM 误判发现**：Baby earmuffs（婴儿耳罩）被 LLM 误判为 sound_machine，验证规则保守策略的正确性
- **边界案例验证**：`dress` 一词可能指孕妇装（pregnancy_pillow）或普通衣物，通过上下文词（`tummy control`, `compression`）辅助区分

---

## 五、可复用工作流模板

### 5.1 快速复用步骤

```bash
# Step 1: 运行全量分类
python generate_product_line_spu_matrix.py

# Step 2: 提取 other 样本供 LLM 分类
python extract_other_samples.py --input labeling_output_v3.3 --n 2000

# Step 3: LLM 分类（外部执行，使用 Claude API）
# 结果保存为 llm_classified_2k.json

# Step 4: 从 LLM 分类结果提取关键词模式
python analyze_llm_patterns.py --input llm_classified_2k.json

# Step 5: 更新规则并重跑
# 编辑 generate_product_line_spu_matrix.py 中的规则
python generate_product_line_spu_matrix.py

# Step 6: 自证
python self_check.py --before baseline.csv --after new.csv
```

### 5.2 规则扩展原则

| 场景 | 处理方式 | 示例 |
|------|---------|------|
| 跨语言同义词 | 加入关键词列表 | `biberon` (法) → `baby_bottle` |
| 复合概念 | 加入强特征词 | `body shaper` → `postpartum_recovery` |
| 单一品牌 | 加入 EXCLUSIVE_BRANDS | `elvie` → `wearable_breast_pump` |
| 易混淆词 | 提高阈值或加排除词 | `dress` 需配合 `tummy` 才命中 |
| 多语言服务词 | 扩展 SERVICE_KEYWORDS | `gut geklappt` (德) → `customer_service` |

### 5.3 阈值调整指南

```python
# 默认阈值
default_threshold = 5

# 提高阈值（减少误匹配）
- wearable_breast_pump: 5 → 7  # 与 breast_pump 易混淆

# 降低阈值（增加召回）
- pregnancy_pillow: 5 → 4      # 独特场景，误匹配风险低
- postpartum_recovery: 5 → 4   # 新扩充关键词，需要更低门槛
```

---

## 六、经验总结

### 6.1 有效策略

1. **LLM 采样 → 关键词反哺**：LLM 在 2000 样本上的分类能力比规则更灵活，但存在误判。最佳用法是提取 LLM 的高置信度分类中的关键词模式，而非直接用 LLM 替代规则。
2. **独家品牌机制**：对于只生产单一品类的小众品牌，直接绑定品线可大幅提升召回。
3. **词干化 + 连字符变体**：显著提升关键词覆盖率，成本几乎为零。

### 6.2 踩坑记录

1. **品牌推断范围过窄**：`ctx_words[:5]` 只查前 5 个上下文词，导致排在后面的词（如 `elvie` 在第 9 位）永远匹配不到。
2. **阈值一刀切**：默认阈值 5 对高混淆品类（wearable vs 常规 breast_pump）不够严格，对新扩充品类（postpartum_recovery）又太严格。
3. **LLM 过度泛化**：LLM 会将 "noise-canceling headphones for babies" 归类为 sound_machine，因为两者都与 "noise" 相关。

### 6.3 下一步优化方向

1. **零标签文本挖掘**：对剩余 59.4% other 文本做主题聚类，发现新关键词
2. **品线内子品类细分**：如将 breast_pump 细分为 "单边/双边/穿戴式"
3. **多语言关键词扩展**：当前仅覆盖英/德/法/意/西，需扩展至更多欧洲/拉美语种
4. **动态阈值学习**：根据历史正确率自动调整各品线阈值

---

*报告生成时间: 2026-04-22*
*适用版本: VOC 标签系统 V3.3.1*
