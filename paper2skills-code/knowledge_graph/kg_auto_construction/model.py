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

        # 确定产品类型 - 按优先级匹配，避免误判
        # 更具体的品类关键词优先于宽泛关键词
        product_type = "Product"
        type_priority = [
            ("BreastPump", ["breast pump", "breastpump", "吸奶器"]),
            ("BabyBottle", ["baby bottle", "babybottle", "婴儿奶瓶"]),
            ("BottleWarmer", ["bottle warmer", "warmer", "温奶器"]),
            ("StorageBag", ["storage bag", "milk bag", "储奶袋"]),
            ("Bottle", ["feeding bottle", "奶瓶"]),  # 单独匹配避免与 baby bottle 冲突
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

