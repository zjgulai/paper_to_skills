---
title: AI Agent 驱动的电商知识图谱自动构建
doc_type: knowledge
module: 08-知识图谱
topic: knowledge-graph-auto-construction
status: stable
created: 2026-05-01
updated: 2026-05-01
owner: self
source: human+ai
---

# Skill Card: AI Agent 驱动的电商知识图谱自动构建

## ① 算法原理

### 核心思想

传统知识图谱构建依赖人工定义 Schema 和编写抽取规则，成本高、扩展性差。**AI Agent 驱动的 KG 自动构建** 将全流程拆解为三个由 LLM Agent 协作完成的阶段，从非结构化产品描述中自动产出结构化知识图谱，无需预定义 Schema 或人工规则。

三阶段流水线：

1. **Ontology Creation & Expansion（本体创建与扩展）**
   - 从产品描述语料中采样代表性样本
   - LLM Agent 提取产品类、属性、关系，组织为 RDF/Turtle 格式
   - 迭代扩展：持续送入新样本，Agent 自动发现新类/属性并扩展 Schema
   - 直到每轮新增元素显著衰减（plateau），平衡覆盖率与可管理性

2. **Ontology Refinement（本体精炼）**
   - 将完整本体作为输入，LLM Agent 执行零样本精炼
   - 建议修订：拆分复合属性、合并冗余类、明确域/范围约束
   - 消除歧义、提升通用性、增强跨品类适应能力

3. **Knowledge Graph Population（图谱填充）**
   - 基于精炼后的本体，逐产品描述生成实例级 RDF 三元组
   - Agent 严格按照本体约束映射属性值，避免幻觉
   - 仅当描述中明确包含值时才生成三元组

### 数学直觉

**本体覆盖度评分**：

$$\text{Coverage}(O, D) = \frac{|\{p \in \text{Properties}(O) : \exists d \in D, \text{extracts}(d, p)\}|}{|\text{Properties}(O)|}$$

其中 $O$ 为本体，$D$ 为产品描述集合。论文实验中达到 **97.1%**。

**迭代扩展收敛条件**：

$$\Delta_t = \frac{|\text{NewElements}_t|}{|\text{TotalElements}_{t-1}|} < \epsilon \quad (\epsilon = 0.05)$$

第 $t$ 轮迭代中，新增元素占比低于阈值时停止扩展。

**三元组生成约束**：

对于产品描述 $d$ 和本体属性 $p$，仅当 $d$ 中显式包含 $p$ 的值时生成三元组：

$$(s, p, o) \in \text{KG} \iff \exists \text{span} \in d : \text{matches}(\text{span}, p)$$

这避免了 LLM 的推理幻觉，确保数据忠实于原文。

### 关键假设

- **产品描述质量**：输入文本需包含足够的结构化信息（规格、功能、适用场景）
- **LLM 一致性**：同一模型在多次调用中对相同描述产生一致输出
- **品类内相似性**：同品类产品共享大部分属性和关系
- **可验证性**：生成的三元组应与原文可对应，便于人工抽检

---

## ② 母婴出海应用案例

### 场景一：从 Amazon 商品描述自动构建母婴商品知识图谱

**业务问题**：
母婴出海电商管理数千个 SKU，商品信息分散在 Amazon、Shopify、供应商提供的不同格式描述中。人工整理"商品-属性-关系"三元组成本极高，且新品上架频繁，知识图谱难以跟上更新速度。

**数据要求**：
- 商品描述文本（标题 + Bullet Points + 描述段落）
- 品类分类信息（如"吸奶器"、"储奶袋"、"温奶器"）
- 可选：产品规格表（结构化补充）

**预期产出**：
- 自动生成的商品本体 Schema：
  ```
  类: Product, Brand, Category, Feature, Material, AgeGroup
  属性: hasBrand, hasPrice, hasMaterial, compatibleWith, recommendedFor
  关系: complementaryTo, alternativeTo, upgradePath
  ```
- 实例级三元组：
  ```
  Spectra_S1 rdf:type BreastPump
  Spectra_S1 hasBrand Spectra
  Spectra_S1 hasFeature "双边电动"
  Spectra_S1 hasFeature "内置电池"
  Spectra_S1 complementaryTo Lansinoh_StorageBags
  Spectra_S1 alternativeTo Medela_Pump
  ```

**业务价值**：
- 新商品上架时，知识图谱自动扩展，无需人工维护
- 为 GraphRAG 提供持续更新的结构化知识源
- 支持跨品类关联发现（如"吸奶器→储奶袋→温奶器"购买链路）

### 场景二：从用户评论中抽取结构化信息扩展图谱

**业务问题**：
用户评论中包含大量产品使用场景、配件搭配、问题反馈等结构化信息，但传统 NLP 方法只能做情感分析，无法将评论中的实体关系提取到知识图谱中。

**数据要求**：
- Amazon/Reddit 用户评论文本
- 已有的商品知识图谱（由场景一构建）
- 评论元数据：评分、 verified purchase 标识

**预期产出**：
- 从评论中提取的扩展三元组：
  ```
  User_1234 rdf:type NewMom
  User_1234 purchased Spectra_S1
  User_1234 mentioned "漏奶问题"
  Spectra_S1 hasIssue "配件密封圈老化"  [from review analysis]
  Spectra_S1 compatibleWith "21mm法兰"    [from review mention]
  ```
- 问题聚类：自动发现高频产品问题，反馈给产品团队

**业务价值**：
- 将 VOC（Voice of Customer）数据自动转化为结构化知识
- 发现官方描述中未覆盖的产品特性与问题
- 与 Kano 需求分类技能联动：评论中提取的需求自动进入 Kano 分析

---

## ③ 代码模板

```python
"""
AI Agent 驱动的电商知识图谱自动构建系统
基于 arXiv:2511.11017 (Peshevski et al., 2025)

功能：
1. 本体创建与扩展（Ontology Creation & Expansion）
2. 本体精炼（Ontology Refinement）
3. 知识图谱填充（Knowledge Graph Population）

Author: paper2skills
Date: 2026-05-01
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import random


# ============================================================
# 数据模型
# ============================================================

@dataclass
class ProductDescription:
    """产品描述"""
    product_id: str
    title: str
    bullets: List[str]
    description: str
    category: str

    def full_text(self) -> str:
        return f"{self.title}\n" + "\n".join(self.bullets) + f"\n{self.description}"


@dataclass
class OntologyClass:
    """本体类"""
    name: str
    comment: str = ""
    parent: Optional[str] = None


@dataclass
class OntologyProperty:
    """本体属性"""
    name: str
    domain: str
    range_type: str  # "string", "integer", "float", "ClassName"
    comment: str = ""
    is_object_property: bool = False


@dataclass
class Ontology:
    """本体定义"""
    classes: List[OntologyClass] = field(default_factory=list)
    properties: List[OntologyProperty] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "classes": [asdict(c) for c in self.classes],
            "properties": [asdict(p) for p in self.properties]
        }


@dataclass
class Triple:
    """RDF 三元组"""
    subject: str
    predicate: str
    object: str
    subject_type: str = ""
    object_type: str = ""

    def to_turtle(self) -> str:
        return f"<:{self.subject}> <:{self.predicate}> <:{self.object}> ."


# ============================================================
# LLM Agent 接口（模拟/真实）
# ============================================================

class LLMAgent:
    """
    LLM Agent 基类。

    实际使用时替换为真实 LLM API 调用（OpenAI, Claude, 本地模型等）。
    这里使用基于规则的模拟实现，用于演示和测试。
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def _mock_extract_ontology(self, descriptions: List[ProductDescription]) -> Ontology:
        """模拟从描述中提取本体"""
        # 基于母婴商品常见属性构建模拟本体
        classes = [
            OntologyClass("Product", "通用商品类"),
            OntologyClass("BreastPump", "吸奶器", "Product"),
            OntologyClass("StorageBag", "储奶袋", "Product"),
            OntologyClass("Bottle", "奶瓶", "Product"),
            OntologyClass("Brand", "品牌"),
            OntologyClass("Feature", "产品特性"),
            OntologyClass("Material", "材质"),
        ]

        properties = [
            OntologyProperty("hasBrand", "Product", "Brand", "商品品牌", True),
            OntologyProperty("hasPrice", "Product", "float", "商品价格"),
            OntologyProperty("hasFeature", "Product", "string", "产品特性"),
            OntologyProperty("hasMaterial", "Product", "Material", "产品材质", True),
            OntologyProperty("compatibleWith", "Product", "Product", "兼容配件", True),
            OntologyProperty("alternativeTo", "Product", "Product", "替代品", True),
            OntologyProperty("recommendedFor", "Product", "string", "适用人群"),
        ]

        # 根据实际描述动态扩展
        all_text = " ".join([d.full_text().lower() for d in descriptions])

        if "baby bottle" in all_text or "奶瓶" in all_text:
            classes.append(OntologyClass("BabyBottle", "婴儿奶瓶", "Bottle"))
            properties.append(OntologyProperty("hasCapacity", "BabyBottle", "integer", "容量(ml)"))
            properties.append(OntologyProperty("isAntiColic", "BabyBottle", "boolean", "是否防胀气"))

        if "nipple" in all_text or "奶嘴" in all_text:
            classes.append(OntologyClass("Nipple", "奶嘴", "Product"))
            properties.append(OntologyProperty("nippleSize", "Nipple", "string", "奶嘴尺寸"))

        if "warmer" in all_text or "温奶" in all_text:
            classes.append(OntologyClass("BottleWarmer", "温奶器", "Product"))
            properties.append(OntologyProperty("heatingMode", "BottleWarmer", "string", "加热方式"))

        return Ontology(classes=classes, properties=properties)

    def _mock_refine_ontology(self, ontology: Ontology) -> Ontology:
        """模拟本体精炼"""
        refined = Ontology(
            classes=[OntologyClass(**asdict(c)) for c in ontology.classes],
            properties=[OntologyProperty(**asdict(p)) for p in ontology.properties]
        )

        # 合并冗余属性：hasCapacity 和 capacity 统一
        seen_names = set()
        deduped = []
        for p in refined.properties:
            if p.name not in seen_names:
                seen_names.add(p.name)
                deduped.append(p)
        refined.properties = deduped

        # 拆分复合属性
        refined.properties.append(
            OntologyProperty("hasColor", "Product", "string", "产品颜色")
        )

        # 添加通用性注释
        for c in refined.classes:
            if not c.comment:
                c.comment = f"{c.name} 类商品"

        return refined

    def _mock_populate_kg(self, desc: ProductDescription, ontology: Ontology) -> List[Triple]:
        """模拟从单个描述填充三元组"""
        triples = []
        text = desc.full_text().lower()
        product_uri = re.sub(r'[^\w]', '_', desc.product_id)

        # 确定产品类型（优先级关键字匹配，避免简单子串误匹配）
        product_type = "Product"
        type_priority = [
            ("BreastPump", ["breast pump", "breastpump", "吸奶器"]),
            ("BabyBottle", ["baby bottle", "babybottle", "婴儿奶瓶"]),
            ("BottleWarmer", ["bottle warmer", "warmer", "温奶器"]),
            ("StorageBag", ["storage bag", "milk bag", "储奶袋"]),
            ("Bottle", ["feeding bottle", "奶瓶"]),
            ("Nipple", ["nipple", "奶嘴"]),
        ]
        for cls_name, keywords in type_priority:
            if any(kw in text for kw in keywords):
                product_type = cls_name
                break

        # rdf:type
        triples.append(Triple(product_uri, "rdf:type", product_type, product_type))

        # 属性提取（基于关键词匹配模拟 LLM 抽取）
        # 品牌
        brand_patterns = [
            r'(spectra|medela|lansinoh|avent|dr\.?\s*brown|philips)',
        ]
        for pattern in brand_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                brand = match.group(1).replace(" ", "_")
                triples.append(Triple(product_uri, "hasBrand", brand, product_type, "Brand"))
                break

        # 价格
        price_match = re.search(r'\$(\d+(?:\.\d+)?)', desc.full_text())
        if price_match:
            triples.append(Triple(product_uri, "hasPrice", price_match.group(1), product_type, "float"))

        # 特性（从 bullet points 提取）
        for bullet in desc.bullets:
            bullet_lower = bullet.lower()
            if any(kw in bullet_lower for kw in ['electric', '电动', 'battery', '电池']):
                triples.append(Triple(product_uri, "hasFeature", "电动", product_type, "string"))
            if any(kw in bullet_lower for kw in ['double', '双边', 'dual']):
                triples.append(Triple(product_uri, "hasFeature", "双边", product_type, "string"))
            if any(kw in bullet_lower for kw in ['silent', '静音', 'quiet']):
                triples.append(Triple(product_uri, "hasFeature", "静音", product_type, "string"))
            if any(kw in bullet_lower for kw in ['anti-colic', '防胀气', 'vent']):
                triples.append(Triple(product_uri, "isAntiColic", "true", product_type, "boolean"))

        # 适用人群
        if any(kw in text for kw in ['newborn', '新生儿', '0-3']):
            triples.append(Triple(product_uri, "recommendedFor", "新生儿", product_type, "string"))
        elif any(kw in text for kw in ['new mom', '新手妈妈', 'first time']):
            triples.append(Triple(product_uri, "recommendedFor", "新手妈妈", product_type, "string"))

        # 兼容性（基于品类推断）
        if product_type == "BreastPump":
            triples.append(Triple(product_uri, "compatibleWith", "StorageBag", product_type, "Product"))
            triples.append(Triple(product_uri, "compatibleWith", "BottleWarmer", product_type, "Product"))

        return triples

    def extract_ontology(self, descriptions: List[ProductDescription]) -> Ontology:
        """本体创建与扩展"""
        if self.use_mock:
            return self._mock_extract_ontology(descriptions)
        # TODO: 替换为真实 LLM API 调用
        raise NotImplementedError("真实 LLM 调用需配置 API key")

    def refine_ontology(self, ontology: Ontology) -> Ontology:
        """本体精炼"""
        if self.use_mock:
            return self._mock_refine_ontology(ontology)
        # TODO: 替换为真实 LLM API 调用
        raise NotImplementedError("真实 LLM 调用需配置 API key")

    def populate_kg(self, description: ProductDescription, ontology: Ontology) -> List[Triple]:
        """知识图谱填充"""
        if self.use_mock:
            return self._mock_populate_kg(description, ontology)
        # TODO: 替换为真实 LLM API 调用
        raise NotImplementedError("真实 LLM 调用需配置 API key")


# ============================================================
# 主框架：三阶段流水线
# ============================================================

class KGAutoConstructionFramework:
    """
    AI Agent 驱动的知识图谱自动构建框架
    """

    def __init__(self, llm_agent: Optional[LLMAgent] = None):
        self.agent = llm_agent or LLMAgent(use_mock=True)
        self.ontology: Optional[Ontology] = None
        self.knowledge_graph: List[Triple] = []
        self.iteration_log: List[Dict] = []

    def stage1_create_ontology(self, sample_descriptions: List[ProductDescription],
                                max_iterations: int = 5,
                                convergence_threshold: float = 0.05) -> Ontology:
        """
        阶段1：本体创建与迭代扩展

        Args:
            sample_descriptions: 代表性产品描述样本
            max_iterations: 最大迭代轮数
            convergence_threshold: 收敛阈值（新增元素占比低于此值时停止）

        Returns:
            最终本体定义
        """
        print("[Stage 1] Ontology Creation & Expansion")

        # 初始本体
        ontology = self.agent.extract_ontology(sample_descriptions)
        initial_size = len(ontology.classes) + len(ontology.properties)

        print(f"  Initial ontology: {len(ontology.classes)} classes, {len(ontology.properties)} properties")

        # 迭代扩展
        for i in range(max_iterations):
            # 模拟：每轮送入新样本，发现新元素
            # 实际实现中，这里应该分批送入不同的产品描述
            expanded = self.agent.extract_ontology(sample_descriptions)
            new_size = len(expanded.classes) + len(expanded.properties)

            delta = (new_size - initial_size) / max(initial_size, 1)
            self.iteration_log.append({
                "stage": "expansion",
                "iteration": i + 1,
                "new_elements": new_size - initial_size,
                "delta_ratio": delta
            })

            print(f"  Iteration {i+1}: {len(expanded.classes)} classes, {len(expanded.properties)} properties "
                  f"(delta: {delta:.3f})")

            if delta < convergence_threshold:
                print(f"  Converged at iteration {i+1}")
                ontology = expanded
                break

            ontology = expanded
            initial_size = new_size

        self.ontology = ontology
        return ontology

    def stage2_refine_ontology(self) -> Ontology:
        """
        阶段2：本体精炼

        Returns:
            精炼后的本体
        """
        if not self.ontology:
            raise ValueError("Must run stage1_create_ontology first")

        print("\n[Stage 2] Ontology Refinement")
        before = len(self.ontology.properties)

        refined = self.agent.refine_ontology(self.ontology)
        after = len(refined.properties)

        print(f"  Before: {len(self.ontology.classes)} classes, {before} properties")
        print(f"  After:  {len(refined.classes)} classes, {after} properties")
        print(f"  Changes: property delta = {after - before}")

        self.ontology = refined
        return refined

    def stage3_populate_kg(self, descriptions: List[ProductDescription]) -> List[Triple]:
        """
        阶段3：知识图谱填充

        Args:
            descriptions: 所有产品描述

        Returns:
            生成的三元组列表
        """
        if not self.ontology:
            raise ValueError("Must run stage2_refine_ontology first")

        print(f"\n[Stage 3] Knowledge Graph Population")
        print(f"  Processing {len(descriptions)} product descriptions...")

        all_triples = []
        success_count = 0

        for desc in descriptions:
            try:
                triples = self.agent.populate_kg(desc, self.ontology)
                all_triples.extend(triples)
                success_count += 1
            except Exception as e:
                print(f"  Warning: failed to process {desc.product_id}: {e}")

        self.knowledge_graph = all_triples

        # 统计
        unique_predicates = set(t.predicate for t in all_triples)
        unique_subjects = set(t.subject for t in all_triples)

        print(f"  Success: {success_count}/{len(descriptions)} ({success_count/len(descriptions)*100:.1f}%)")
        print(f"  Total triples: {len(all_triples)}")
        print(f"  Unique subjects: {len(unique_subjects)}")
        print(f"  Unique predicates: {len(unique_predicates)}")

        # 计算属性覆盖率
        populated_props = set(t.predicate for t in all_triples
                              if t.predicate not in ["rdf:type"])
        total_props = len(self.ontology.properties)
        coverage = len(populated_props) / max(total_props, 1)
        print(f"  Property coverage: {coverage*100:.1f}%")

        return all_triples

    def run_pipeline(self, descriptions: List[ProductDescription]) -> Dict:
        """
        运行完整的三阶段流水线

        Returns:
            {
                'ontology': 最终本体,
                'triples': 所有三元组,
                'statistics': 统计信息
            }
        """
        print("=" * 70)
        print("AI Agent 驱动的知识图谱自动构建")
        print("=" * 70)

        # Stage 1
        self.stage1_create_ontology(descriptions)

        # Stage 2
        self.stage2_refine_ontology()

        # Stage 3
        triples = self.stage3_populate_kg(descriptions)

        stats = {
            "total_descriptions": len(descriptions),
            "ontology_classes": len(self.ontology.classes),
            "ontology_properties": len(self.ontology.properties),
            "total_triples": len(triples),
            "unique_subjects": len(set(t.subject for t in triples)),
        }

        return {
            "ontology": self.ontology.to_dict(),
            "triples": [asdict(t) for t in triples],
            "statistics": stats,
            "iteration_log": self.iteration_log
        }

    def export_to_turtle(self, filepath: str):
        """导出为 Turtle 格式"""
        lines = []
        lines.append("@prefix : <http://paper2skills.com/maternal-baby#> .")
        lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        lines.append("")

        # 本体定义
        if self.ontology:
            for cls in self.ontology.classes:
                lines.append(f":{cls.name} rdf:type rdfs:Class ;")
                lines.append(f"    rdfs:comment \"{cls.comment}\" .")
                lines.append("")

            for prop in self.ontology.properties:
                prop_type = "owl:ObjectProperty" if prop.is_object_property else "owl:DatatypeProperty"
                lines.append(f":{prop.name} rdf:type {prop_type} ;")
                lines.append(f"    rdfs:domain :{prop.domain} ;")
                lines.append(f"    rdfs:range :{prop.range_type} ;")
                lines.append(f"    rdfs:comment \"{prop.comment}\" .")
                lines.append("")

        # 实例数据
        for triple in self.knowledge_graph:
            lines.append(triple.to_turtle())

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\nExported to: {filepath}")


# ============================================================
# 测试数据：母婴出海商品
# ============================================================

def create_test_data() -> List[ProductDescription]:
    """创建母婴商品测试数据"""
    return [
        ProductDescription(
            product_id="spectra-s1",
            title="Spectra S1 Plus Electric Breast Pump",
            bullets=[
                "Hospital grade double electric breast pump",
                "Built-in rechargeable battery for portability",
                "Ultra-quiet motor (45dB) for discreet pumping",
                "Adjustable suction levels 1-12 with massage mode",
                "Compatible with all Spectra bottles and accessories",
            ],
            description=("The Spectra S1 Plus is a hospital-grade double electric breast pump "
                        "designed for new moms who need efficient and comfortable milk expression. "
                        "Its built-in battery allows for portable use anywhere. The quiet motor "
                        "ensures discreet pumping sessions at home or work."),
            category="BreastPump"
        ),
        ProductDescription(
            product_id="medela-pump",
            title="Medela Pump In Style with MaxFlow",
            bullets=[
                "Double electric breast pump with 2-Phase Expression",
                "Hospital performance in a personal-use pump",
                "Compact and lightweight design",
                "Includes cooler bag, bottles, and breast shields",
            ],
            description=("Medela Pump In Style delivers hospital-performance pumping with "
                        "MaxFlow technology. The 2-Phase Expression technology mimics baby's "
                        "natural nursing rhythm for more milk in less time."),
            category="BreastPump"
        ),
        ProductDescription(
            product_id="lansinoh-bags",
            title="Lansinoh Breastmilk Storage Bags, 100 Count",
            bullets=[
                "Pre-sterilized and BPA/BPS free",
                "Double zipper seal prevents leaks",
                "Lay flat for efficient storage and quick thawing",
                "Compatible with all major breast pump brands",
            ],
            description=("Lansinoh Breastmilk Storage Bags are designed for safe and convenient "
                        "breast milk storage. The double zipper seal ensures no leaks during "
                        "storage or transport. Pre-sterilized for immediate use."),
            category="StorageBag"
        ),
        ProductDescription(
            product_id="avent-warmer",
            title="Philips Avent Fast Baby Bottle Warmer",
            bullets=[
                "Warms milk in just 3 minutes",
                "Gentle defrost setting for frozen breast milk",
                "Compatible with all bottle sizes and brands",
                "Automatic shut-off for safety",
            ],
            description=("Philips Avent Fast Bottle Warmer gently and evenly warms breast milk "
                        "or formula in just 3 minutes. The smart temperature control prevents "
                        "hot spots and preserves nutrients."),
            category="BottleWarmer"
        ),
        ProductDescription(
            product_id="dr-brown-bottle",
            title="Dr. Brown's Options+ Wide-Neck Baby Bottle",
            bullets=[
                "Anti-colic vent system reduces gas and spit-up",
                "Wide-neck design for easy cleaning",
                "BPA-free glass and silicone materials",
                "Includes Level 1 slow flow nipple for newborns",
            ],
            description=("Dr. Brown's Options+ bottles feature a patented vent system that "
                        "reduces colic, gas, and spit-up. The wide-neck design makes filling "
                        "and cleaning easy. Perfect for newborns 0-3 months."),
            category="BabyBottle"
        ),
    ]


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：演示完整的 KG 自动构建流程"""
    print("=" * 70)
    print("母婴出海 - AI Agent 驱动的知识图谱自动构建系统")
    print("=" * 70)

    # 1. 准备测试数据
    print("\n[1] 加载母婴商品测试数据...")
    descriptions = create_test_data()
    print(f"   商品数量: {len(descriptions)}")
    for d in descriptions:
        print(f"   - {d.product_id}: {d.title}")

    # 2. 初始化框架
    print("\n[2] 初始化 KG 自动构建框架...")
    agent = LLMAgent(use_mock=True)  # 使用模拟 LLM，实际使用时设为 False
    framework = KGAutoConstructionFramework(agent)

    # 3. 运行三阶段流水线
    print("\n[3] 运行三阶段流水线...")
    result = framework.run_pipeline(descriptions)

    # 4. 输出结果
    print("\n[4] 输出结果...")
    print(f"\n   本体统计:")
    print(f"   - 类数量: {result['statistics']['ontology_classes']}")
    print(f"   - 属性数量: {result['statistics']['ontology_properties']}")

    print(f"\n   知识图谱统计:")
    print(f"   - 三元组总数: {result['statistics']['total_triples']}")
    print(f"   - 唯一主体: {result['statistics']['unique_subjects']}")

    # 5. 展示部分三元组
    print(f"\n[5] 示例三元组:")
    for t in result['triples'][:8]:
        print(f"   {t['subject']} --{t['predicate']}--> {t['object']}")

    # 6. 导出
    print("\n[6] 导出知识图谱...")
    framework.export_to_turtle("/tmp/maternal_baby_kg.ttl")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)

    return result


if __name__ == '__main__':
    result = main()
```

---

## ④ 技能关联

### 前置技能
- **Knowledge Graph for Skills Management**：理解图结构、三元组、RDF 等基本概念
- **LLM API 调用基础**：了解如何调用 OpenAI/Claude 等 LLM API 进行结构化输出
- **Prompt Engineering**：掌握 few-shot、CoT 等 prompt 设计方法

### 延伸技能
- **GraphRAG**：本技能构建的 KG 是 GraphRAG 的输入数据源
- **KARMA（多 Agent KG 富化）**：在此基础上增加多 Agent 协作的图谱精炼和扩展
- **时序知识图谱**：支持商品信息和用户行为的时序演化
- **跨语言 KG 对齐**：将中文社媒 KG 与英文电商 KG 对齐

### 可组合技能
- **GraphRAG**：构建 KG → GraphRAG 检索 → 智能问答
- **VOC Semantic Blueprint**：从 VOC 提取的结构化数据（属性、关系）可直接注入 KG
- **Kano 需求分类**：KG 中抽取的产品特性可作为 Kano 分析的输入
- **SoMeR 多视角表示**：用户画像数据可与商品 KG 融合为异构图
- **TopicImpact**：评论中的观点单元可映射为 KG 中的问题/特性节点

---


- **可组合**：[[Skill-KGQA-Question-Answering]] / [[Skill-KG-Augmented-Recommendation-CoLaKG]]

- **可组合**：[[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]] / [[Skill-HGT-Heterogeneous-Graph-Transformer]]

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 商品知识图谱自动构建 | 新品上架时自动扩展图谱，减少 80% 人工标注成本 | 开发 2-3 周 | 12-18x |
| VOC 数据结构化 | 评论/客服数据自动转化为 KG，支持多维度分析 | 开发 1-2 周 | 10-15x |
| GraphRAG 知识库维护 | 持续自动更新，无需专职人员维护 | 集成 1 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要商品描述文本数据，门槛较低
- 技术门槛：中等，核心依赖 LLM API 调用和 prompt 设计
- 工程复杂度：中，三阶段流水线需要模块化设计
- 维护成本：低，新商品自动扩展，人工仅需抽检

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **战略价值极高**：是 08-知识图谱领域的核心缺口，直接影响 GraphRAG 的可用性
- **业务匹配度完美**：电商产品 KG 构建是母婴出海的刚需场景
- **技术可落地性强**：基于成熟 LLM API，代码模板可直接适配
- **与现有体系衔接自然**：上游连接 VOC 技能，下游连接 GraphRAG

### 评估依据
1. **论文实验验证充分**：在 291 个真实空调产品上达到 97% 处理成功率和 97.1% 属性覆盖率
2. **母婴电商场景天然适配**：商品描述结构规范、品类内属性相似，适合自动化抽取
3. **填补核心缺口**：08-知识图谱从 2 个技能扩展到 3 个，构建链路更完整
4. **与 MAS 技能形成协同**：Agent 驱动的架构与项目已有 MAS 体系一致

---

## 参考论文

1. **AI Agent-Driven Framework for Automated Product Knowledge Graph Construction in E-Commerce** (2025)
   - arXiv:2511.11017
   - 核心贡献：三阶段 Agent 驱动框架，从电商产品描述自动构建 KG
   - 实验：291 个空调产品，97% 成功率，97.1% 属性覆盖率

2. **Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction** (EMNLP 2024)
   - arXiv:2404.03868
   - 核心贡献：三阶段流水线（抽取 → 定义 → 规范化），方法论支撑

3. **iText2KG: Incremental Knowledge Graphs Construction using Large Language Models** (WISE 2024)
   - 核心贡献：增量式 KG 构建，支持持续更新

4. **KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment** (2025)
   - arXiv:2502.06472
   - 核心贡献：多 Agent KG 富化，可作为本技能的延伸方向

---

## 开源资源

- **RDFLib**: https://rdflib.readthedocs.io/ - Python RDF 处理库
- **LangChain LLMGraphTransformer**: https://python.langchain.com/ - 内置 KG 抽取工具
- **Neo4j LLM Knowledge Graph Builder**: https://neo4j.com/labs/llm-graph-builder/ - 可视化 KG 构建

---

## 技能演进路径

```
Round 1: 基础框架（当前）
  - 三阶段流水线：创建 → 精炼 → 填充
  - Mock LLM 实现，便于离线测试

Round 2: 真实 LLM 集成
  - 替换 Mock 为真实 LLM API（GPT-4 / Claude）
  - 添加 prompt 缓存和错误重试机制
  - 支持流式增量构建

Round 3: 多模态扩展
  - 整合商品图片信息（视觉属性提取）
  - 支持规格表 PDF 的结构化解析

Round 4: 持续学习与进化
  - 与 AutoTag 标签系统联动：新标签自动触发 KG 扩展
  - 与用户反馈闭环：错误三元组自动修正
```
