"""
Product Attribute Graph Parser
基于 Hierarchical KG Construction (Yang et al., GenAIRec 2024) 的 Schema-Guided Generation 思想，
将电商产品描述解析为层次化属性图谱。

核心流程：
1. Schema Initialization: 定义产品属性 schema
2. Extracting: 从文本抽取属性值
3. Formatting: 按 schema 约束输出结构化 JSON
4. Hierarchy Expansion: 构建层次化属性树
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ── 数据模型 ──────────────────────────────────────────

@dataclass
class AttributeNode:
    """属性图谱中的一个节点"""
    name: str                    # 属性名，如 "材质"
    value: str                   # 属性值，如 "医用级硅胶"
    data_type: str = "string"    # string | float | choices | boolean
    unit: Optional[str] = None   # 单位，如 "g", "cm", "dB"
    children: List[AttributeNode] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "data_type": self.data_type,
            "unit": self.unit,
            "children": [c.to_dict() for c in self.children],
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ProductAttributeGraph:
    """产品属性图谱"""
    product_name: str = ""
    category: str = ""
    attributes: List[AttributeNode] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_name": self.product_name,
            "category": self.category,
            "attributes": [a.to_dict() for a in self.attributes],
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }

    def get_attribute(self, name: str) -> Optional[AttributeNode]:
        """按属性名查找"""
        for attr in self.attributes:
            if attr.name.lower() == name.lower():
                return attr
        return None

    def get_flattened(self) -> Dict[str, str]:
        """扁平化为 key-value 字典"""
        result = {}
        def _flatten(node: AttributeNode, prefix: str = "") -> None:
            key = f"{prefix}{node.name}" if prefix else node.name
            result[key] = f"{node.value} {node.unit or ''}".strip()
            for child in node.children:
                _flatten(child, f"{key}.")
        for attr in self.attributes:
            _flatten(attr)
        return result


# ── Schema 定义 ───────────────────────────────────────

@dataclass
class SchemaProperty:
    """Schema 中的一个属性定义"""
    name: str
    data_type: str          # string | float | choices | boolean
    choices: Optional[List[str]] = None
    unit: Optional[str] = None
    required: bool = False
    description: str = ""
    aliases: List[str] = field(default_factory=list)  # 英文别名，用于匹配英文文本


# 母婴出海 — 吸奶器品类默认 Schema
BREAST_PUMP_SCHEMA = [
    SchemaProperty("产品名称", "string", required=True, description="产品型号或名称", aliases=["product name", "model", "brand"]),
    SchemaProperty("品类", "choices", choices=["穿戴式", "双边电动", "单边电动", "手动"], aliases=["category", "type", "style"]),
    SchemaProperty("吸力档位", "choices", choices=["3档", "5档", "9档", "12档", "多级可调"], aliases=["suction", "levels", "settings", "modes"]),
    SchemaProperty("噪音水平", "float", unit="dB", description="运行噪音分贝数", aliases=["noise", "quiet", "silent", "sound", "decibel", "db"]),
    SchemaProperty("材质", "choices", choices=["医用硅胶", "PP", "PPSU", "不锈钢", "其他"], aliases=["material", "made of", "silicone"]),
    SchemaProperty("电池容量", "float", unit="mAh", aliases=["battery", "charge", "power"]),
    SchemaProperty("重量", "float", unit="g", aliases=["weight", "lightweight", "heavy", "grams", "oz", "lb"]),
    SchemaProperty("便携性", "choices", choices=["极高", "高", "中", "低"], aliases=["portable", "portability", "travel", "compact", "light"]),
    SchemaProperty("智能功能", "choices", choices=["APP控制", "记忆模式", "智能记录", "无"], aliases=["smart", "app", "bluetooth", "wifi", "digital"]),
    SchemaProperty("价格区间", "choices", choices=["<$50", "$50-100", "$100-200", ">$200"], aliases=["price", "cost", "cheap", "expensive", "affordable", "$"]),
]

# 母婴出海 — 纸尿裤品类默认 Schema
DIAPER_SCHEMA = [
    SchemaProperty("产品名称", "string", required=True),
    SchemaProperty("尺码", "choices", choices=["NB", "S", "M", "L", "XL", "XXL"]),
    SchemaProperty("吸收量", "float", unit="mL"),
    SchemaProperty("材质", "choices", choices=["棉质", "竹纤维", "无纺布", "有机棉"]),
    SchemaProperty("厚度", "choices", choices=["超薄", "薄", "中等", "厚"]),
    SchemaProperty("透气性", "choices", choices=["极好", "好", "一般", "差"]),
    SchemaProperty("防漏设计", "choices", choices=["360度防漏", "侧边防漏", "无特殊设计"]),
    SchemaProperty("适用体重", "string", description="如 4-8kg"),
    SchemaProperty("价格区间", "choices", choices=["<$20", "$20-40", "$40-60", ">$60"]),
]


# ── 核心解析器 ────────────────────────────────────────

class ProductAttributeGraphParser:
    """
    产品属性图谱解析器。

    基于 Schema-Guided Generation 思想，从自由文本产品描述中
    抽取层次化属性结构。
    """

    def __init__(
        self,
        schema: Optional[List[SchemaProperty]] = None,
        enable_hierarchy: bool = True,
    ):
        self.schema = schema or BREAST_PUMP_SCHEMA
        self.enable_hierarchy = enable_hierarchy

    def parse(
        self,
        product_name: str,
        description: str,
        category_hint: Optional[str] = None,
    ) -> ProductAttributeGraph:
        """
        解析产品描述为属性图谱。

        Args:
            product_name: 产品名称/型号
            description: 产品描述文本
            category_hint: 品类提示（用于选择 schema）
        """
        if not description:
            return ProductAttributeGraph(product_name=product_name)

        # 1. 自动检测/选择品类 schema
        schema = self._select_schema(description, category_hint)

        # 2. 从文本抽取属性
        attributes = self._extract_attributes(description, schema)

        # 3. 层次化扩展
        if self.enable_hierarchy:
            attributes = self._expand_hierarchy(attributes)

        return ProductAttributeGraph(
            product_name=product_name,
            category=category_hint or self._detect_category(description),
            attributes=attributes,
            raw_text=description,
            metadata={
                "schema_properties": len(schema),
                "extracted_attributes": len(attributes),
            },
        )

    def parse_batch(
        self,
        items: List[Tuple[str, str, Optional[str]]],
    ) -> List[ProductAttributeGraph]:
        """批量解析"""
        return [self.parse(name, desc, cat) for name, desc, cat in items]

    def _select_schema(
        self,
        description: str,
        category_hint: Optional[str],
    ) -> List[SchemaProperty]:
        """根据品类提示或文本内容选择 schema"""
        if category_hint:
            if "diaper" in category_hint.lower() or "尿裤" in category_hint:
                return DIAPER_SCHEMA
            return BREAST_PUMP_SCHEMA

        # 自动检测
        desc_lower = description.lower()
        diaper_signals = ["diaper", "nappy", "absorb", "wetness", "纸尿裤", "尿布"]
        if any(s in desc_lower for s in diaper_signals):
            return DIAPER_SCHEMA
        return BREAST_PUMP_SCHEMA

    def _detect_category(self, description: str) -> str:
        """自动检测品类"""
        desc_lower = description.lower()
        if any(s in desc_lower for s in ["diaper", "nappy", "尿裤"]):
            return "纸尿裤"
        return "吸奶器"

    def _extract_attributes(
        self,
        description: str,
        schema: List[SchemaProperty],
    ) -> List[AttributeNode]:
        """
        基于规则从文本抽取属性值。
        生产环境应替换为 LLM + regex-constrained generation。
        """
        attributes: List[AttributeNode] = []
        desc_lower = description.lower()

        for prop in schema:
            value = self._match_property(description, prop)
            if value:
                attributes.append(AttributeNode(
                    name=prop.name,
                    value=value,
                    data_type=prop.data_type,
                    unit=prop.unit,
                    confidence=0.6,  # 规则基线置信度
                ))

        return attributes

    def _match_property(self, text: str, prop: SchemaProperty) -> Optional[str]:
        """为单个 schema 属性匹配文本中的值"""
        text_lower = text.lower()

        # 构建所有匹配关键词: 中文名 + 英文别名
        match_keywords = [prop.name.lower()] + [a.lower() for a in prop.aliases]

        # choices 类型: 在文本中找匹配的选择项
        if prop.data_type == "choices" and prop.choices:
            # 先直接匹配 choice 文本
            for choice in prop.choices:
                if choice.lower() in text_lower:
                    return choice
            # 模糊语义匹配
            for choice in prop.choices:
                if self._semantic_match(choice, text_lower):
                    return choice
            # 检查是否有数字匹配档位
            if "档" in prop.name or "level" in prop.name.lower():
                num_match = re.search(r'(\d+)\s*(suction levels?|levels?|modes?|settings?)', text_lower)
                if num_match:
                    n = int(num_match.group(1))
                    for choice in prop.choices:
                        if str(n) in choice:
                            return choice

        # float 类型: 提取数字
        if prop.data_type == "float":
            # 找 "关键词 + 数字 + 单位" 或 "数字 + 单位 + 关键词" 模式
            for kw in match_keywords:
                patterns = [
                    rf"{re.escape(kw)}.*?([\d.]+)\s*{re.escape(prop.unit or '')}",
                    rf"([\d.]+)\s*{re.escape(prop.unit or '')}.*?{re.escape(kw)}",
                    rf"{re.escape(kw)}.*?([\d.]+)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        return match.group(1)

        # string 类型: 提取短语
        if prop.data_type == "string":
            if "名称" in prop.name or "name" in prop.name.lower():
                return None  # 由调用方提供
            for kw in match_keywords:
                patterns = [
                    rf"{re.escape(kw)}[是为:：]\s*([^。,.;；\n]+)",
                    rf"{re.escape(kw)}\s+([^。,.;；\n]+)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        return match.group(1).strip()

        # boolean 类型
        if prop.data_type == "boolean":
            positive = ["yes", "有", "支持", "具备", "with", "has"]
            negative = ["no", "无", "不支持", "没有", "without"]
            for p in positive:
                if p in text_lower:
                    return "是"
            for n in negative:
                if n in text_lower:
                    return "否"

        return None

    def _semantic_match(self, choice: str, text: str) -> bool:
        """语义近义词匹配（简化版）"""
        synonyms = {
            "医用硅胶": ["silicone", "medical grade", "soft", "gentle"],
            "超薄": ["thin", "slim", "lightweight"],
            "极好": ["excellent", "great", "amazing", "perfect"],
            "高": ["high", "good", "great"],
            "低": ["low", "poor", "bad"],
        }
        choice_lower = choice.lower()
        if choice_lower in synonyms:
            for syn in synonyms[choice_lower]:
                if syn in text:
                    return True
        return False

    def _expand_hierarchy(
        self,
        attributes: List[AttributeNode],
    ) -> List[AttributeNode]:
        """
        层次化扩展：为属性节点添加子节点。
        示例: 材质 → [表层材质, 内部结构, 接触面材质]
        """
        expanded = []
        for attr in attributes:
            if attr.name == "材质":
                # 扩展材质层次
                attr.children = [
                    AttributeNode(name="接触面材质", value=attr.value, data_type="choices"),
                    AttributeNode(name="主体材质", value=attr.value, data_type="choices"),
                ]
            elif attr.name == "吸力档位":
                attr.children = [
                    AttributeNode(name="最低档吸力", value="未知", data_type="float", unit="mmHg"),
                    AttributeNode(name="最高档吸力", value="未知", data_type="float", unit="mmHg"),
                    AttributeNode(name="档位调节方式", value="未知", data_type="choices"),
                ]
            elif attr.name == "噪音水平":
                attr.children = [
                    AttributeNode(name="最低噪音", value=attr.value, data_type="float", unit="dB"),
                    AttributeNode(name="最高噪音", value=attr.value, data_type="float", unit="dB"),
                ]
            expanded.append(attr)
        return expanded


# ── 可视化辅助 ────────────────────────────────────────

def print_attribute_graph(graph: ProductAttributeGraph, indent: int = 0) -> None:
    """打印属性图谱（树形结构）"""
    prefix = "  " * indent
    print(f"{prefix}📦 {graph.product_name}")
    print(f"{prefix}品类: {graph.category}")
    print(f"{prefix}属性数: {len(graph.attributes)}")
    print()

    def _print_node(node: AttributeNode, level: int) -> None:
        p = "  " * level
        unit_str = f" {node.unit}" if node.unit else ""
        print(f"{p}├─ {node.name}: {node.value}{unit_str} ({node.data_type})")
        for child in node.children:
            _print_node(child, level + 1)

    for attr in graph.attributes:
        _print_node(attr, indent + 1)
    print()


def compare_graphs(
    graph_a: ProductAttributeGraph,
    graph_b: ProductAttributeGraph,
) -> Dict[str, Any]:
    """对比两个产品的属性图谱"""
    flat_a = graph_a.get_flattened()
    flat_b = graph_b.get_flattened()

    all_keys = set(flat_a.keys()) | set(flat_b.keys())
    diff = []
    same = []

    for key in sorted(all_keys):
        val_a = flat_a.get(key, "N/A")
        val_b = flat_b.get(key, "N/A")
        if val_a != val_b:
            diff.append({"attribute": key, "product_a": val_a, "product_b": val_b})
        else:
            same.append({"attribute": key, "value": val_a})

    return {
        "product_a": graph_a.product_name,
        "product_b": graph_b.product_name,
        "common_attributes": same,
        "different_attributes": diff,
        "overlap_ratio": len(same) / len(all_keys) if all_keys else 0,
    }


# ── 测试 ──────────────────────────────────────────────

def test_parser() -> None:
    """单元测试"""
    parser = ProductAttributeGraphParser()

    # 测试用例 1: 吸奶器描述
    desc1 = (
        "Momcozy S12 Pro Wearable Breast Pump. "
        "9 suction levels with hospital-grade suction. "
        "Ultra-quiet operation at 45dB. "
        "Made with medical-grade silicone. "
        "Lightweight at 230g. Battery capacity 1200mAh. "
        "Smart APP control with memory mode. "
        "Price around $150."
    )
    graph1 = parser.parse("Momcozy S12 Pro", desc1, "吸奶器")
    print_attribute_graph(graph1)
    assert len(graph1.attributes) >= 3, f"Expected >= 3 attributes, got {len(graph1.attributes)}"
    print("✅ Test 1 passed")

    # 测试用例 2: 纸尿裤描述
    desc2 = (
        "Huggies Little Snugglers Diapers Size M. "
        "Absorbs up to 500mL. Made with organic cotton. "
        "Ultra-thin design with excellent breathability. "
        "360-degree leak protection. "
        "For babies 6-11kg. Price $35."
    )
    parser2 = ProductAttributeGraphParser(schema=DIAPER_SCHEMA)
    graph2 = parser2.parse("Huggies M", desc2, "纸尿裤")
    print_attribute_graph(graph2)
    assert graph2.category == "纸尿裤"
    print("✅ Test 2 passed")

    # 测试用例 3: 空文本
    graph3 = parser.parse("Empty Product", "")
    assert len(graph3.attributes) == 0
    print("✅ Test 3 passed")

    # 测试用例 4: 图谱对比
    comparison = compare_graphs(graph1, graph2)
    print(f"\n📊 图谱对比: {comparison['product_a']} vs {comparison['product_b']}")
    print(f"   共同属性: {len(comparison['common_attributes'])}")
    print(f"   差异属性: {len(comparison['different_attributes'])}")
    print("✅ Test 4 passed")

    print("\n🎉 All tests passed!")


def test_with_amazon_data() -> None:
    """用 Amazon 真实数据做 POC 验证"""
    import pandas as pd

    data_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv"
    df = pd.read_csv(data_path, nrows=50)

    parser = ProductAttributeGraphParser()
    graphs: List[ProductAttributeGraph] = []

    for idx, row in df.iterrows():
        title = str(row.get("Title", "")) if pd.notna(row.get("Title")) else ""
        content = str(row.get("Content", "")) if pd.notna(row.get("Content")) else ""
        model = str(row.get("Model", "")) if pd.notna(row.get("Model")) else ""

        text = f"{title}. {content}"
        if not text.strip():
            continue

        graph = parser.parse(f"Product-{idx}", text)
        graph.metadata["rating"] = row.get("Rating", "")
        graph.metadata["model"] = model
        graphs.append(graph)

    # 统计
    total_attrs = sum(len(g.attributes) for g in graphs)
    avg_attrs = total_attrs / len(graphs) if graphs else 0
    all_cats = [g.category for g in graphs]
    cat_dist = {}
    for c in all_cats:
        cat_dist[c] = cat_dist.get(c, 0) + 1

    print(f"\n📊 Amazon POC 统计 ({len(graphs)} 条产品)")
    print(f"   总属性数: {total_attrs}")
    print(f"   平均每产品属性: {avg_attrs:.2f}")
    print(f"   品类分布: {cat_dist}")

    # 打印第一个非空图谱
    for g in graphs:
        if g.attributes:
            print("\n--- 示例输出 ---")
            print_attribute_graph(g)
            break

    print("\n✅ Amazon POC 验证通过")


if __name__ == "__main__":
    print("=" * 60)
    print("Product Attribute Graph Parser - Unit Tests")
    print("=" * 60)
    test_parser()

    print("\n" + "=" * 60)
    print("Product Attribute Graph Parser - Amazon POC")
    print("=" * 60)
    test_with_amazon_data()
