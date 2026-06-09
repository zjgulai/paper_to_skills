---
title: 层级商品知识图谱自动构建（图片→KG）
doc_type: knowledge
module: 08-知识图谱
topic: hierarchical-product-kg-construction
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
paper: arXiv:2410.21237
roadmap_phase: phase2
---

# Skill: Hierarchical Product KG Construction — 图片驱动的层级商品知识图谱自动构建

> 论文:**Hierarchical Knowledge Graph Construction from Images for Scalable E-Commerce** (Yang et al., GenAIECom@CIKM 2024) · arXiv:2410.21237

---

## ① 算法原理

### 核心思想

零样本下用商品图片自动构建跨语种属性知识图谱:**Schema 先行 → VLM 多轮萃取 → LLM 约束推理 → 层级扩展 → 程序化去重**。建库成本与 SKU 数量线性解耦,无需人工标注模板。

### 数学直觉

**两阶段建图流程**:
- **Stage 1 - Schema 初始化**:对每个属性 $x$ 用自回归方式预测最适配数据类型
  $$t' = \arg\max_{t' \in \{int, float, str, choices\}} P(t \mid x)$$
  避免 LLM 自由发挥产生结构噪音。
- **Stage 2 - 每件商品 4 步**:VLM 多轮萃取非结构化描述 → SGLang 正则约束生成合法 JSON → 层级扩展(在叶节点和根品类间迭代插入中间节点) → 程序化去重(归一化大小写/词序)。

**评估指标**:
- 类别属性:Accuracy(常规分类准确率)
- 数值属性:$\text{Acc}@\theta = \mathbb{1}\left[\frac{|v_{\text{pred}} - v_{\text{gt}}|}{v_{\text{gt}}} \leq \theta\right]$,$\theta=0.05$ 即误差≤5% 视为正确。

### 关键假设

1. **图片信息充分**:商品图片包含足够的语言无关属性信号(品类、颜色、材质、形状)
2. **Schema 可枚举**:目标属性可用 `int / float / str / choices` 四种类型表达,不强制开放式生成
3. **LLM 推理可信**:对图片不可见的属性(如品牌→品类归属),Llama3.1-70B 级别模型的 CoT 推理可靠

### 关键效果数字(论文 Table 1)

| 方法 | Category Acc | Weight Acc@0.05 |
|---|---|---|
| 零样本 baseline | 0.00% | 9.78% |
| + Schema | 62.86% | 16.30% |
| **完整方法** | **97.14%** | **73.91%** |

---

## ② 母婴出海应用案例

### 场景一:Amazon US/EU 母婴 SKU 多语种属性冷启动

- **业务问题**:出海卖家上架 1 万件母婴新品到 Amazon US,需逐 SKU 填写品类路径(`Baby > Feeding > Baby Formula > Infant Formula 0-6m`)、净重(g)、包装材质、适用月龄等 10+ 个强制属性。人工填写需 2-3 人月,且因中英描述不一致导致 listing 违规下架风险。
- **数据要求**:仅需厂商提供的商品主图(JPG/PNG ≥ 448×448)和可选的中文描述
- **预期产出**:对每件 SKU 自动输出 Amazon Flat File 兼容 JSON,含完整品类路径 + 物理属性 + 适用人群
- **业务价值**:1 万 SKU 的 listing 准备从 2-3 人月压缩到 8 小时(GPU 推理),按品类合规率 97% 计算,违规下架损失从 5-8% 降至 0.5% 以下,单卖家年节省 listing 维护成本约 50-80 万元

### 场景二:东南亚跨语种母婴选品图谱

- **业务问题**:某母婴出海选品平台需对马来西亚/印尼市场 5 万件 SKU 建可检索属性图谱。商品多为中国制造,只有中文描述+图片,但目标市场用 Bahasa/英文搜索。传统做法需要 3 套语种各建一遍图谱。
- **数据要求**:商品主图(语言无关)+ 顶层品类配置文件(英文/中文/Bahasa 三语对照表)
- **预期产出**:统一的层级 KG,中间节点天然可挂载多语种别名(`安抚奶嘴 / pacifier / dot bayi`),支持任意语种检索
- **业务价值**:单一图谱多语种复用,建图成本从 30 人月降到 5 GPU 天;因多语种检索召回率提升 25-40%,选品平台 GMV 增量可量化为单月 300-500 万元(以中型选品平台 5 千万月 GMV 计)

---

## ③ 代码模板

```python
"""
Hierarchical Product KG Construction — 论文 arXiv:2410.21237 最小骨架实现

依赖:
    pip install sglang[all] transformers pillow

注意:
    论文无公开代码,以下骨架按论文 §3-4 描述还原。
    生产环境替换 mock VLM/LLM 为 InternVL2-8B + Llama3.1-70B 或同等 API。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


BABY_ECOM_SCHEMA: Dict[str, object] = {
    "product_name": "str",
    "category": {
        "type": "choices",
        "options": [
            "Infant Formula", "Baby Food", "Baby Bottle", "Pacifier",
            "Diaper", "Baby Clothing", "Stroller", "Baby Carrier",
            "Baby Wipes", "Baby Skincare", "Others",
        ],
    },
    "brand": "str",
    "primary_color": {
        "type": "choices",
        "options": ["White", "Pink", "Blue", "Green", "Yellow", "Purple", "Others"],
    },
    "package_material": {
        "type": "choices",
        "options": ["Plastic", "Metal", "Cardboard", "Glass", "Fabric", "Others"],
    },
    "weight_kg": "float",
    "age_range": {
        "type": "choices",
        "options": ["0-6m", "6-12m", "1-3y", "3-6y", "All ages"],
    },
}


@dataclass
class ProductKGNode:
    properties: Dict[str, object]
    category_hierarchy: List[str]


def _mock_vlm_extract(image_path: str, schema: Dict[str, object]) -> str:
    desc_parts = []
    fname = image_path.lower()
    if "aptamil" in fname or "formula" in fname:
        desc_parts.append("infant formula in metallic cylindrical can")
    if "pacifier" in fname or "dot" in fname:
        desc_parts.append("silicone pacifier in pink/blue plastic packaging")
    if "diaper" in fname:
        desc_parts.append("disposable baby diaper in cardboard box")
    return " | ".join(desc_parts) or "generic baby product packaging"


def _mock_llm_infer(description: str, schema: Dict[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {"product_name": "Auto-Detected Product"}
    desc = description.lower()
    if "formula" in desc:
        output["category"] = "Infant Formula"
        output["package_material"] = "Metal"
        output["weight_kg"] = 0.8
        output["age_range"] = "0-6m"
    elif "pacifier" in desc:
        output["category"] = "Pacifier"
        output["package_material"] = "Plastic"
        output["primary_color"] = "Pink"
        output["age_range"] = "0-6m"
        output["weight_kg"] = 0.05
    elif "diaper" in desc:
        output["category"] = "Diaper"
        output["package_material"] = "Cardboard"
        output["weight_kg"] = 1.5
    else:
        output["category"] = "Others"
        output["weight_kg"] = 0.5
    return output


def regex_constrained_json(raw_text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def hierarchical_expand(
    product_name: str,
    leaf_category: str,
    parent_lookup: Optional[Callable[[str], str]] = None,
    max_levels: int = 4,
) -> List[str]:
    default_taxonomy = {
        "Infant Formula": "Baby Food",
        "Baby Food": "Baby Products",
        "Baby Products": "Mother & Baby",
        "Pacifier": "Baby Feeding",
        "Baby Feeding": "Baby Products",
        "Diaper": "Baby Care",
        "Baby Care": "Baby Products",
    }
    parent_lookup = parent_lookup or (lambda x: default_taxonomy.get(x, ""))

    chain: List[str] = [product_name, leaf_category]
    current = leaf_category
    for _ in range(max_levels):
        parent = parent_lookup(current)
        if not parent or parent == current:
            break
        chain.append(parent)
        current = parent
    return chain


def prune_graph(nodes: List[str]) -> List[str]:
    seen = set()
    pruned: List[str] = []
    for node in nodes:
        key = frozenset(node.lower().split())
        if key not in seen:
            seen.add(key)
            pruned.append(node)
    return pruned


def build_product_kg_node(
    image_path: str,
    schema: Optional[Dict[str, object]] = None,
    vlm_extract: Optional[Callable[[str, Dict[str, object]], str]] = None,
    llm_infer: Optional[Callable[[str, Dict[str, object]], Dict[str, object]]] = None,
) -> ProductKGNode:
    schema = schema or BABY_ECOM_SCHEMA
    vlm_extract = vlm_extract or _mock_vlm_extract
    llm_infer = llm_infer or _mock_llm_infer

    description = vlm_extract(image_path, schema)
    properties = llm_infer(description, schema)

    category = str(properties.get("category", "Others"))
    product_name = str(properties.get("product_name", image_path))
    hierarchy = hierarchical_expand(product_name, category)
    properties["category_hierarchy"] = prune_graph(hierarchy)

    return ProductKGNode(
        properties=properties,
        category_hierarchy=properties["category_hierarchy"],
    )


def evaluate_attribute_accuracy(
    predictions: List[Dict[str, object]],
    ground_truths: List[Dict[str, object]],
    threshold: float = 0.05,
) -> Dict[str, float]:
    metrics = {"category_acc": 0.0, "weight_acc": 0.0}
    n = len(predictions)
    if n == 0:
        return metrics

    cat_hits = sum(
        1 for p, g in zip(predictions, ground_truths)
        if p.get("category") == g.get("category")
    )

    weight_hits = 0
    weight_total = 0
    for p, g in zip(predictions, ground_truths):
        v_p = p.get("weight_kg")
        v_g = g.get("weight_kg")
        if isinstance(v_p, (int, float)) and isinstance(v_g, (int, float)) and v_g != 0:
            weight_total += 1
            err = abs(v_p - v_g) / abs(v_g)
            if err <= threshold:
                weight_hits += 1

    metrics["category_acc"] = cat_hits / n
    metrics["weight_acc"] = weight_hits / weight_total if weight_total else 0.0
    return metrics


def main() -> None:
    print("=" * 60)
    print("Hierarchical Product KG Construction — Demo")
    print("=" * 60)

    sample_skus = [
        "/data/products/aptamil_organic_stage1.jpg",
        "/data/products/pigeon_silicone_pacifier_pink.jpg",
        "/data/products/huggies_diaper_size3.jpg",
    ]

    nodes = [build_product_kg_node(p) for p in sample_skus]
    for sku, node in zip(sample_skus, nodes):
        print(f"\nSKU: {sku}")
        print(json.dumps(node.properties, indent=2, ensure_ascii=False))

    ground_truths = [
        {"category": "Infant Formula", "weight_kg": 0.85},
        {"category": "Pacifier", "weight_kg": 0.05},
        {"category": "Diaper", "weight_kg": 1.4},
    ]
    predictions = [n.properties for n in nodes]
    metrics = evaluate_attribute_accuracy(predictions, ground_truths)
    print("\n" + "=" * 60)
    print(f"Category Acc: {metrics['category_acc']*100:.2f}%")
    print(f"Weight Acc@0.05: {metrics['weight_acc']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

### 前置技能
- [Skill-Multilingual-NER-Universal-v2](./[[Skill-Multilingual-NER-Universal-v2]].md) — 多语 NER 是文本属性提取的方法学基础,本 Skill 用 VLM 替代 NER 处理图片但共享"实体抽取"思路
- [Skill-Knowledge-Graph-for-Skills-Management](./[[Skill-Knowledge-Graph-for-Skills-Management]].md) — 理解 KG schema 设计是本 Skill 的方法学前置

### 延伸技能
- [Skill-KG-Relation-Completion-CBLiP](./[[Skill-KG-Relation-Completion-CBLiP]].md) — 本 Skill 构建初步 KG 后,用关系补全完善边
- [Skill-GraphRAG-Knowledge-Enhanced-Retrieval](./[[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]].md) — 构建好的产品 KG 直接用于 RAG 检索增强

### 可组合
- [Skill-Cold-Start-Product-Recommendation](../06-增长模型/[[Skill-Cold-Start-Product-Recommendation]].md) — 新 SKU 自动建图后立刻进入冷启动推荐管线,形成端到端冷启动闭环
- [Skill-Dense-Retrieval-Ecommerce-Semantic-Search](./[[Skill-Dense-Retrieval-Ecommerce-Semantic-Search]].md) — KG 节点的多语种别名为稠密检索提供天然查询扩展

---

## ⑤ 商业价值评估

### ROI 预估

**场景 A(Amazon US/EU 母婴 SKU 冷启动)**:
- 节省 listing 准备人力:1 万 SKU × 3 人月 × 月薪 1.5 万 = **45 万元/年/卖家**
- 减少违规下架损失:违规率 5-8% → 0.5%,按平均 GMV 100 万元/月计 = **节省 50-80 万元/年**
- 总收益:**单卖家年增 95-125 万元**,投入 GPU 推理成本约 2-5 万元
- **ROI ≈ 30-60 倍**

**场景 B(东南亚跨语种选品图谱)**:
- 多语种建图复用:30 人月 → 5 GPU 天 = 节省人工成本 **~100 万元**
- 多语种检索召回提升 25-40%:中型平台月 GMV 5000 万 × 0.5%(召回提升的转化贡献) = **月增 25 万元**
- 年化收益:**~400 万元**

### 实施难度:⭐⭐⭐☆☆ (3/5)

- 主要难点:VLM 与 LLM 的 GPU 部署成本(InternVL2-8B 需 24GB+ 显存,Llama3.1-70B 需 80GB+)
- 易处:Schema 设计 + SGLang 约束生成是声明式的,无需训练数据
- 论文实验仅 105 张图片,大规模(10万+ SKU)的稳定性需要工程团队补充测试

### 优先级评分:⭐⭐⭐⭐☆ (4/5)

**评估依据**:
1. **业务相关度高**:直接服务于母婴出海跨境电商的 listing 准备核心痛点
2. **方法新颖度高**:Schema-guided + 层级扩展的组合在 2024 年是较新的范式,孤立工作较多但端到端管线完整
3. **可立即落地**:虽然官方无开源代码,但论文提供了完整的 prompt 设计和系统架构,工程实现路径清晰
4. **填补图谱关键缺口**:本 Skill 同时连接 08-知识图谱、05-推荐系统(冷启动)、09-DataAgent-LLM(多模态 Agent)三个领域,是高价值跨域桥梁

**风险**:
- 论文实验规模偏小(105 张图),生产化需要扩展验证
- 对 VLM 描述质量敏感,低分辨率/光线差的图片效果下降
- 无公开代码,实现细节(尤其 SGLang 正则约束的精确写法)需自行调试
