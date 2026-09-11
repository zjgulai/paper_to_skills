# Skill Card: 产品属性图谱解析
# Product Attribute Graph Parsing

**论文来源**: Hierarchical Knowledge Graph Construction from Images for Scalable E-Commerce (GenAIRec 2024, arXiv:2410.21237)
**理论基础**: Schema-Guided Generation + Hierarchical Expansion + Regex-Constrained JSON
**适用领域**: NLP-VOC / 电商产品信息抽取 / 竞品分析

---

## ① 算法原理

电商产品信息的核心矛盾：自由文本描述（非结构化）→ 可计算的结构化属性（层次化树/图）。

本文提出**三阶段 Schema-Guided KG 构建**框架：

1. **Graph Initialization**：定义产品属性 Schema（属性名、数据类型、可选值、单位）
   - 例：`{name: "噪音水平", type: "float", unit: "dB", choices: null}`

2. **Cycle of Enrollment**（四步循环）：
   - **Extracting**：VLM 从图片提取信息，或 LLM 从文本描述提取
   - **Formatting & Inferring**：SGLang 做 regex-constrained generation，输出严格符合 Schema 的 JSON
   - **Hierarchy Expansion**：在产品和抽象类目间插入中间实体（如 "Dark Chocolate Bar" → "Chocolate" → "Food and Beverage"）
   - **Graph Pruning**：合并相似实体，去重

3. **Inventory Usage**：KG 用于检索、推荐、产品分析

**数学直觉**：将产品描述映射为一个带约束的结构化生成问题。Schema 定义了"合法输出空间"，regex-constrained generation 保证输出永不越界。层次化扩展将扁平属性提升为可导航的语义树。

---

## ② 母婴出海应用案例

### 案例 A：吸奶器竞品属性对比

**场景**：对比 Momcozy 与 Spectra/Medela 的关键属性差异，支撑产品定位决策。

**输入**：
- Momcozy S12 Pro: "9 suction levels, 45dB, medical-grade silicone, 230g, 1200mAh"
- Spectra S1: "12 suction levels, 50dB, hospital-grade, 1.1kg, no battery"

**输出（属性图谱对比）**：
```json
{
  "product_a": "Momcozy S12 Pro",
  "product_b": "Spectra S1",
  "different_attributes": [
    {"attribute": "吸力档位", "Momcozy": "9档", "Spectra": "12档"},
    {"attribute": "噪音水平", "Momcozy": "45 dB", "Spectra": "50 dB"},
    {"attribute": "重量", "Momcozy": "230 g", "Spectra": "1100 g"},
    {"attribute": "便携性", "Momcozy": "高", "Spectra": "低"}
  ]
}
```

**业务价值**：
- 产品团队可自动生成"竞品属性差异矩阵"
- 营销团队可提炼差异化卖点（如 Momcozy 的静音+便携 vs Spectra 的大吸力）
- 选品团队可快速评估新品在属性空间中的定位

**数据需求**：
- 产品描述文本（Title + Content）
- 品类 Schema（可复用代码模板中的预定义 Schema）

### 案例 B：跨市场属性缺失检测

**场景**：检测同一产品在不同市场的属性描述完整性。

**输入**：
- Momcozy S12 US Amazon: 完整描述（含噪音、材质、电池）
- Momcozy S12 日本乐天: 仅有基础描述（缺少噪音、电池信息）

**输出**：
- 日本市场属性完整度评分：40%（vs US 市场 90%）
- 缺失属性列表：["噪音水平", "电池容量", "智能功能"]

**业务价值**：
- 自动识别不同市场的产品信息缺口
- 指导运营团队补充缺失信息，提升转化率

---

## ③ 代码模板

**核心文件**: `paper2skills-code/nlp_voc/product_attribute_graph_parsing/model.py`

```python
from product_attribute_graph_parsing import ProductAttributeGraphParser

# 初始化解析器（使用默认吸奶器 Schema）
parser = ProductAttributeGraphParser()

# 单条解析
graph = parser.parse(
    product_name="Momcozy S12 Pro",
    description="9 suction levels with hospital-grade suction. "
                "Ultra-quiet at 45dB. Medical-grade silicone. "
                "Lightweight at 230g. Battery 1200mAh.",
    category_hint="吸奶器"
)
print(graph.to_dict())

# 批量解析
graphs = parser.parse_batch([
    ("Product-A", "description-a", "吸奶器"),
    ("Product-B", "description-b", "纸尿裤"),
])

# 竞品对比
from product_attribute_graph_parsing import compare_graphs
result = compare_graphs(graph_a, graph_b)
```

**预定义 Schema**:
```python
BREAST_PUMP_SCHEMA = [
    SchemaProperty("吸力档位", "choices", choices=["3档", "5档", "9档", "12档"]),
    SchemaProperty("噪音水平", "float", unit="dB"),
    SchemaProperty("材质", "choices", choices=["医用硅胶", "PP", "PPSU"]),
    SchemaProperty("电池容量", "float", unit="mAh"),
    SchemaProperty("重量", "float", unit="g"),
]

DIAPER_SCHEMA = [
    SchemaProperty("尺码", "choices", choices=["NB", "S", "M", "L", "XL"]),
    SchemaProperty("吸收量", "float", unit="mL"),
    SchemaProperty("厚度", "choices", choices=["超薄", "薄", "中等", "厚"]),
]
```

**运行测试**:
```bash
cd paper2skills-code/nlp_voc/product_attribute_graph_parsing
python3 model.py
```

**生产环境替换**:
- 规则基线 → LLM (GPT-4o/Claude) + SGLang regex-constrained generation
- 增加 VLM 多模态输入（产品图片 + 文本描述）
- 接入产品数据库做属性一致性校验

---

## ④ 技能关联

### 前置技能
- **Skill-ABSA-BERT-MoE** — 提供方面级情感分析，识别产品描述中的关键属性维度
- **Skill-VOC-Semantic-Blueprint** — 提供评论结构化抽取，可作为属性图谱的用户反馈输入
- **Skill-TaxoAdapt-Taxonomy-Evolution** — 提供品类 taxonomy，支撑 Schema 的动态更新

### 延伸技能
- **Skill-Kano-需求分类与优先级** — 属性图谱的 (属性, 用户情感) 可直接输入 Kano 分类
- **Skill-iReFeed-需求优先级排序** — 属性重要度排序驱动产品路线图
- **Skill-TJAP-跨市场品类组合定价** — 属性差异支撑跨市场定价策略

### 可组合
- **属性图谱 + VOC 语义蓝图**: 产品官方属性 + 用户感知的属性差异 = 完整产品画像
- **属性图谱 + CSK 聚类**: 按关键属性聚类用户，发现细分人群
- **属性图谱 + TJAP**: 属性差异驱动跨市场定价差异化

---

## ⑤ 商业价值评估

### ROI 预估

| 指标 | 现状（人工整理） | 实施后（自动抽取） | 节省/提升 |
|------|---------------|------------------|----------|
| 单品属性整理时间 | 15-30 分钟 | 2-5 秒 | **95%↓** |
| 1000 SKU 属性库建设 | 2-3 周（2人） | 2-4 小时 | **95%↓** |
| 竞品属性对比报告 | 1-2 天/份 | 实时生成 | **即时** |
| 属性完整度（跨市场） | 约 60% | 可提升至 90%+ | **+50%↑** |

**年化价值**: ~80 万人民币/年（按 5000 SKU × 15 分钟节省 × 运营人力成本）

### 实施难度
⭐⭐⭐☆☆（3/5星）
- 规则基线版：1 天可上线（已有代码模板 + 预定义 Schema）
- LLM 增强版：接入 GPT-4o + SGLang，+3-5 天
- 多品类 Schema 扩展：每个新品类需定义 Schema，约 2-4 小时/品类

### 优先级评分
⭐⭐⭐⭐☆（4/5星）
- 直接支撑选品、竞品分析、产品定位三大业务场景
- 与 VOC 语义蓝图形成"官方属性 + 用户感知"的完整闭环
- 论文方法成熟，工程可行性高

**综合评分: 8/10**
