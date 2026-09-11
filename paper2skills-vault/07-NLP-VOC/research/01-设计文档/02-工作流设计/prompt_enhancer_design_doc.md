---
title: prompt_enhancer.py 设计文档
doc_type: architecture
module: voc-nlp
 topic: prompt-enrichment
status: draft
created: 2026-05-16
updated: 2026-05-16
owner: @AIArchitect
source: ai
---

# prompt_enhancer.py 设计文档

## 1. 背景与问题

### 1.1 当前状态

`tag_dict_loader.py` → `build_compact_prompt()` 仅消费字典 4/37 个字段（10.8%）：

| 消费字段 | 用途 |
|---------|------|
| `标签ID` | TAG_L1_001 |
| `VOC标签（英文）` | comfortable |
| `情感极性` | → `+` / `-` / `·` |
| `AIPL节点` | L1 |

生成的 compact prompt 示例：
```
TAG_L1_001|comfortable+|L1
```

### 1.2 核心问题

- LLM 只能靠英文缩写猜标签边界，不知道排除口径
- 无 seed words 辅助匹配用户表达变体
- 无口语化原话匹配能力
- Method C 9 个高风险标签误判的根因之一：信息不足

### 1.3 量化影响

- Prompt 字段利用率：**10.8%**（4/37）
- 当前 precision：**0.896**（LLM 盲打条件下）
- 预估 enriched prompt 后 precision：**0.93+**（+3~5pp）
- 预估实现成本：**低**（不改模型、不改架构，只改 prompt 生成逻辑）

---

## 2. 设计目标

| 目标 | 指标 | 验收标准 |
|------|------|---------|
| 字段覆盖率 | ≥80% | 消费字段数 / 字典总字段数 |
| Precision 不 regress | ≥0.896 | 5K 子集 A/B 测试 |
| Token 预算合规 | ≤15K tokens | DeepSeek V4-Flash context 128K，余量充足 |
| 分级加载 | 按标签质量动态调整 | 高优先级标签信息完整，低优先级可降级 |

---

## 3. 系统接口

### 3.1 输入

```python
@dataclass
class PromptEnhancerInput:
    dict_path: Path                    # v4.5+ 字典 Excel 路径
    output_format: str = "compact"     # "compact" | "enriched" | "tiered"
    tier_config: TierConfig = None     # 分级加载配置
    max_tokens: int = 15000            # Token 预算上限
    product_line_filter: str = None    # 按品线过滤（如 "吸奶器"）
    audit_status_filter: list = None   # 按审核状态过滤
```

### 3.2 输出

```python
@dataclass
class PromptEnhancerOutput:
    system_prompt: str                 # 可直接喂给 LLM 的 system prompt
    metadata: PromptMetadata           # 统计信息
    rejected_tags: list[TagReject]     # 被降级/拒绝的标签清单

@dataclass
class PromptMetadata:
    total_tags: int
    included_tags: int
    field_coverage: float              # 字段覆盖率
    estimated_tokens: int
    tier_distribution: dict            # 各分级标签数量
```

---

## 4. 核心组件

### 4.1 字段选择器（Field Selector）

**消费字段清单（15 个，覆盖率 40.5%）**：

| 优先级 | 字段名 | 对 LLM 价值 | 消费方式 |
|--------|--------|------------|---------|
| P0 | `标签ID` | 必需 | `TAG_L1_001` |
| P0 | `VOC标签（英文）` | 必需 | `comfortable` |
| P0 | `情感极性` | 必需 | `+` / `-` / `·` |
| P0 | `AIPL节点` | 必需 | `L1` |
| P0 | `标签定义` | **核心**——边界口径 | 完整纳入 |
| P0 | `英文关键词/典型表达` | **核心**——seed words | `seed:soft,gentle` |
| P0 | `消费者习惯关键词/原话短语` | **核心**——口语匹配 | `phrases:"太软了","不够支撑"` |
| P1 | `标签主题` | 高——Aspect 聚合 | `theme:使用舒适度` |
| P1 | `适用产品品线` | 中——路由过滤 | `products:吸奶器,喂养电器` |
| P1 | `适用VOC载体` | 中——语境差异 | `channels:评论,客服工单` |
| P1 | `策略包` | 中——业务权重 | `strategy:产品改进` |
| P2 | `业务动作/责任部门` | 低——下游分析 | 可选纳入 |
| P2 | `故事线关联` | 低——归因用 | 可选纳入 |
| P2 | `MetricDirection` | 低——校验用 | 可选纳入 |
| P2 | `Proxy NPS贡献` | 低——分析用 | 可选纳入 |

**不消费字段（22 个）**：
- 治理字段（合理性评分/风险等级/问题诊断/优化建议/优化优先级）→ 用于分级，不进入 prompt
- v3.6 遗留字段（渠道权重/版本/状态/安全等级）→ 已废弃
- 审核状态 → 用于分级过滤
- 来源类型/适用用户画像/对应原子指标 → 当前对 LLM 价值有限

### 4.2 Prompt 格式化器（Prompt Formatter）

**旧格式（compact）**：
```
TAG_L1_001|comfortable+|L1
```

**新格式（enriched）**：
```
TAG_L1_001|comfortable+|L1|theme:使用舒适度|def:用户对产品核心功能满足感的表达，排除外观/包装/价格因素|seed:soft,gentle,effective|phrases:"太舒服了","支撑很好","用着顺手"|products:吸奶器,喂养电器|channels:评论,客服工单
```

**格式规范**：
- 每标签 1 行，字段用 `|` 分隔
- 子字段用 `key:value` 格式，多值用 `,` 分隔
- 中文内容用 UTF-8，长度截断至 80 字符（含 `...`）
- 空字段省略（不输出 `key:`）

### 4.3 分级加载器（Tiered Loader）

**分级规则**：

| 分级 | 条件 | Prompt 内容 | 占比目标 |
|------|------|------------|---------|
| Tier 1（核心） | 审核状态=已审核 AND 合理性评分≥60 | 全字段（15 个） | ~30% |
| Tier 2（标准） | 审核状态=已审核 AND 合理性评分<60 | 核心字段（7 个） | ~40% |
| Tier 3（精简） | 审核状态=待审核 OR 审核状态为空 | 必需字段（4 个） | ~20% |
| Tier 4（排除） | 审核状态=废弃 OR 风险等级=高 | 不纳入 prompt | ~10% |

**动态降级策略**：
- 当总 token 数接近预算上限时，从 Tier 2 开始逐批降级为 Tier 3
- 降级优先级：先降级 `适用产品品线` → `适用VOC载体` → `策略包` → `标签主题`
- 保留底线：任何标签至少保留 `标签ID` + `英文` + `情感` + `AIPL`

### 4.4 Token 预算管理器（Token Budget Manager）

**估算公式**：
```python
def estimate_tokens(prompt_text: str) -> int:
    """中文字符 ≈ 1 token，英文单词 ≈ 0.75 token，标点 ≈ 0.5 token"""
    cn_chars = sum(1 for c in prompt_text if '一' <= c <= '鿿')
    en_words = len(re.findall(r'[a-zA-Z]+', prompt_text))
    punctuation = len(re.findall(r'[|,;:"\'\-\_\+]', prompt_text))
    return int(cn_chars + en_words * 0.75 + punctuation * 0.5)
```

**预算分配**：
- System prompt 总预算：15,000 tokens
- Schema instruction（固定）：~800 tokens
- Tag 列表（动态）：~14,000 tokens / 645 标签 ≈ 21 tokens/标签
- Enriched 格式预估：15-25 tokens/标签（取决于字段填充率）
- 结论：645 标签全量 enriched 约 12,000-14,000 tokens，在预算内

---

## 5. 与现有流水线集成

### 5.1 替换点

```python
# 旧调用（llm_labeler.py 第 57 行）
def get_system_prompt() -> str:
    return build_compact_prompt() + "\n\n" + SCHEMA_INSTRUCTION

# 新调用
def get_system_prompt() -> str:
    from prompt_enhancer import PromptEnhancer
    enhancer = PromptEnhancer(dict_path=DICT_PATH_V45)
    result = enhancer.generate(
        output_format="tiered",
        tier_config=DEFAULT_TIER_CONFIG,
        max_tokens=15000
    )
    return result.system_prompt + "\n\n" + SCHEMA_INSTRUCTION
```

### 5.2 向后兼容

- `build_compact_prompt()` 保留为 legacy API
- `tag_dict_loader.py` 增加 `--prompt-version` 参数：
  - `v1` = compact（旧格式，兼容现有流水线）
  - `v2` = enriched（新格式）
  - `v3` = tiered（分级加载）

### 5.3 版本控制

```python
DICT_PATHS = {
    "v3.9": Path("...tag_dictionary_v3.9.xlsx"),
    "v4.1": Path("...tag_dictionary_v4.1.xlsx"),
    "v4.5": Path("...tag_dictionary_v4.5.xlsx"),
    "v4.6": Path("...tag_dictionary_v4.6.xlsx"),  # Phase 1 产出
}
```

---

## 6. 测试策略（与 @QA 验收清单对齐）

### 6.1 Schema 契约测试

```python
def test_field_coverage():
    enhancer = PromptEnhancer(dict_path=V45_PATH)
    result = enhancer.generate(output_format="enriched")
    assert result.metadata.field_coverage >= 0.80, \
        f"Coverage {result.metadata.field_coverage} < 80%"
```

### 6.2 回归测试（A/B）

```python
def test_precision_no_regress():
    # 旧 prompt baseline
    old_prompt = build_compact_prompt()
    old_result = run_llm_labeler(prompt=old_prompt, dataset=GOLDEN_5K)
    old_precision = evaluate(old_result, golden=GOLDEN_HUMAN149)

    # 新 prompt
    new_prompt = PromptEnhancer(V45_PATH).generate().system_prompt
    new_result = run_llm_labeler(prompt=new_prompt, dataset=GOLDEN_5K)
    new_precision = evaluate(new_result, golden=GOLDEN_HUMAN149)

    assert new_precision >= old_precision, \
        f"Precision regressed: {new_precision} < {old_precision}"
```

### 6.3 Token 预算测试

```python
def test_token_budget():
    enhancer = PromptEnhancer(V45_PATH)
    result = enhancer.generate(max_tokens=15000)
    assert result.metadata.estimated_tokens <= 15000
```

### 6.4 分级加载测试

```python
def test_tiered_loading():
    enhancer = PromptEnhancer(V45_PATH)
    result = enhancer.generate(output_format="tiered")

    # Tier 1 标签信息完整度
    tier1_tags = [t for t in result.tags if t.tier == 1]
    for tag in tier1_tags:
        assert "def:" in tag.prompt_line  # 包含定义
        assert "seed:" in tag.prompt_line  # 包含关键词

    # Tier 3 标签至少保留 4 个必需字段
    tier3_tags = [t for t in result.tags if t.tier == 3]
    for tag in tier3_tags:
        fields = tag.prompt_line.split("|")
        assert len(fields) >= 4
```

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Enriched prompt 超过 token 预算 | 中 | LLM 截断，尾部标签丢失 | 分级加载 + 动态降级 |
| 标签定义过长导致 prompt 膨胀 | 中 | 单标签 token 超标 | 定义截断至 80 字符 |
| 新增字段引入噪声，precision 下降 | 低 | 比 compact 还差 | A/B 测试，旧 prompt 为 baseline，regress 则回滚 |
| 字典 v4.5 → v4.6 升级导致字段变更 | 中 | prompt_enhancer 需适配 | 字段选择器用配置驱动，非硬编码 |
| 分级规则误杀高价值标签 | 低 | 关键标签被降级或排除 | 分级规则可配置，人工审核白名单 |

---

## 8. 实现计划

### D1（今天）
- [ ] `prompt_enhancer.py` 骨架代码（字段选择器 + 格式化器）
- [ ] 旧/新 prompt 格式对比输出
- [ ] 单标签 enriched 格式样例

### D2
- [ ] 分级加载器实现
- [ ] Token 预算管理器实现
- [ ] 与 `tag_dict_loader.py` 集成

### D3
- [ ] 5K 子集 A/B 测试跑通
- [ ] Precision 对比报告
- [ ] QA 验收

---

## 9. 附录

### 9.1 单标签 enriched 样例对比

**旧（compact）**：
```
TAG_L1_001|comfortable+|L1
```

**新（enriched，Tier 1）**：
```
TAG_L1_001|comfortable+|L1|theme:使用舒适度|def:用户对产品核心功能满足感的表达，排除外观包装价格因素|seed:soft,gentle,effective|phrases:"太舒服了","支撑很好","用着顺手"|products:吸奶器,喂养电器|channels:评论,客服工单
```

**新（enriched，Tier 3）**：
```
TAG_L1_001|comfortable+|L1
```
（与旧格式相同，因为审核状态为空，降级为精简版）

### 9.2 预估 Token 变化

| 格式 | 645 标签总 tokens | 单标签平均 |
|------|------------------|-----------|
| compact（旧） | ~7,000 | ~10.8 |
| enriched（全量） | ~14,000 | ~21.7 |
| tiered（混合） | ~12,000 | ~18.6 |

---

> **状态**：设计文档初稿，待 @Dev 评审 + @QA 验收清单确认后进入实现。
