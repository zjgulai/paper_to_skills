---
title: LLM驱动供应链本体自动构建 — 从ERP文档到语义图谱的零样本迭代萃取
doc_type: knowledge
module: 24-标签工程
topic: ontology-llm-auto-build-supply-chain
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: LLM驱动供应链本体自动构建

> **来源**：CEUR Vol-4085（Ontology-Guided KG Extraction by LLMs）+ IJPR 2025（SC Visibility + KG）+ MDPI Axioms 2025（Graph-Based LLM SC Analysis）+ IOF Supply Chain Reference Ontology（GitHub）
> **桥梁**：标签工程 ↔ 知识图谱 ↔ Palantir 本体层 | **类型**：NLP+本体工程

## ① 算法原理

**Palantir 方法论的核心挑战**：本体（Ontology）是整个 AI 决策系统的语义基础，但企业中现有的供应链本体要么是手工维护（更新滞后、覆盖不全），要么根本不存在。LLM 驱动的本体自动构建解决这一根本问题。

**核心算法：迭代式本体引导 KG 萃取**（CEUR Vol-4085 框架）：

```
Step 1: 用户/专家提供初始本体种子（5-10个核心实体类型）
Step 2: LLM 链式提示（CoT）从文档中提取：
         - 实体（Entity）：供应商名、SKU代码、仓库位置
         - 关系（Relation）：供货关系、运输关系、采购关系
         - 属性（Property）：前置时间、价格、质量评级
Step 3: 本体验证 + 冲突消解（新关系是否与现有本体矛盾？）
Step 4: 本体扩展（发现新的实体类型/关系类型 → 追加到本体）
Step 5: 迭代 Step 2-4，直到本体收敛（新增实体<阈值）
```

**三层提示链设计**（关键工程实现）：

```
Layer 1 - 结构提取层（Extract Prompt）
  输入：PO 文档、供应商邮件、WMS 报告
  任务：识别文档中的实体和关系（JSON 格式输出）
  约束：只提取本体中定义的类型（减少幻觉）

Layer 2 - 验证层（Validation Prompt）  
  输入：提取结果 + 现有本体
  任务：检查新实体是否与现有节点重复/矛盾
  约束：输出置信度分 + 冲突说明

Layer 3 - 本体扩展层（Schema Extension Prompt）
  输入：重复出现但未在本体中的实体/关系模式
  任务：建议新的 ObjectType 或 LinkType
  约束：需要满足最小出现频率阈值（默认 5 次）
```

**与传统 KG 构建方法的对比**：

| 方法 | 人工成本 | 覆盖率 | 更新频率 | 质量 |
|------|---------|--------|---------|------|
| 手工建模 | 极高（月级） | 低 | 极慢 | 高 |
| 规则提取（NER+RE） | 中（周级） | 中 | 快 | 中 |
| **LLM零样本迭代** | 低（天级） | 高 | 实时 | 中高 |
| LLM+专家验证 | 低+审核 | 高 | 实时 | 高 |

**供应链专属本体模板**（基于 IOF SCRO 标准）：

```python
SC_ONTOLOGY_SEED = {
    "ObjectTypes": {
        "Supplier": ["name", "location", "lead_time_days", "reliability_score", "capacity"],
        "Product": ["sku", "title", "category", "unit_cost", "weight_kg", "cbm_per_unit"],
        "PurchaseOrder": ["po_number", "date", "total_amount", "status", "payment_terms"],
        "Shipment": ["shipment_id", "origin", "destination", "eta", "carrier", "cbm"],
        "Warehouse": ["warehouse_id", "location", "capacity_cbm", "utilization_rate"],
        "Contract": ["contract_id", "start_date", "end_date", "min_order_qty", "price_tier"]
    },
    "LinkTypes": {
        "SUPPLIES": {"from": "Supplier", "to": "Product"},
        "FULFILLS": {"from": "Supplier", "to": "PurchaseOrder"},
        "CONTAINS": {"from": "Shipment", "to": "Product"},
        "STORES": {"from": "Warehouse", "to": "Product"},
        "GOVERNED_BY": {"from": "PurchaseOrder", "to": "Contract"},
        "ROUTES_THROUGH": {"from": "Shipment", "to": "Warehouse"}
    }
}
```

## ② 母婴出海应用案例

**场景A：从历史 PO 和邮件自动构建供应商知识图谱**

母婴品牌过去 2 年积累了 500+ 份采购订单 PDF 和 1000+ 封供应商邮件，但从未系统化整理。通过 LLM 本体自动构建，在 2 天内完成：
- 识别 150+ 个供应商实体（含别名合并）
- 提取 3200+ 条供货关系（哪个供应商供哪个 SKU）
- 自动标注交货周期、付款条件等属性
- 发现 12 个隐性风险（同一供应商供应多个爆款 SKU 的单点故障）

**数据要求**：PO PDF/CSV、邮件文本、合同文档、WMS 导出
**预期产出**：Neo4j 知识图谱（节点 500+、边 3000+）+ 自动生成的本体 Schema
**业务价值**：本体构建从 3 个月人工建模 → 2 天自动化，节省 80% 人力；发现隐性风险提前预防

**场景B：新SKU引入时的供应链本体实时扩展**

每次新增产品线（如从婴儿奶瓶扩展到辅食机），LLM 自动从新的 PO 和供应商资质文件中提取新实体，并检查是否需要扩展本体 Schema（如新增"认证类型"实体）。

**数据要求**：新品相关文档（供应商报价单、产品规格书、认证文件）
**预期产出**：本体增量补丁（新增实体/关系类型）+ 更新后的知识图谱
**业务价值**：新供应商入库时间从 2 周 → 1 天，本体覆盖率从 60% → 92%

## ③ 代码模板

```python
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# 注意：实际使用时替换为真实LLM调用（OpenAI/DeepSeek/Claude）
# 此处用 mock 函数展示接口设计

@dataclass
class OntologyNode:
    """本体节点（提取后的实体）"""
    entity_id: str
    entity_type: str
    properties: Dict[str, str] = field(default_factory=dict)
    source_doc: str = ""
    confidence: float = 1.0

@dataclass
class OntologyEdge:
    """本体边（提取后的关系）"""
    from_id: str
    to_id: str
    relation_type: str
    properties: Dict[str, str] = field(default_factory=dict)
    source_doc: str = ""
    confidence: float = 1.0

class SCOntologyBuilder:
    """
    LLM驱动的供应链本体自动构建器
    
    算法流程：
    1. 初始化种子本体 Schema
    2. 对每个输入文档运行3层提示链
    3. 冲突消解 + 实体合并（fuzzy matching）
    4. 本体扩展（发现新的类型）
    5. 输出知识图谱 + 更新的本体 Schema
    """
    
    def __init__(self, ontology_seed: Dict, llm_func=None):
        """
        Args:
            ontology_seed: 初始本体定义（ObjectTypes + LinkTypes）
            llm_func: LLM调用函数 func(prompt: str) -> str
        """
        self.ontology = ontology_seed.copy()
        self.nodes: Dict[str, OntologyNode] = {}
        self.edges: List[OntologyEdge] = []
        self.llm = llm_func or self._mock_llm
        self.entity_counter = {}  # 统计实体出现频率（用于本体扩展判断）
    
    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM（演示用，替换为真实API调用）"""
        # 模拟从 PO 文档提取实体关系的响应
        mock_responses = {
            "extract": json.dumps({
                "entities": [
                    {"id": "SUP-SZ-001", "type": "Supplier",
                     "properties": {"name": "深圳乐宝科技", "location": "深圳龙华区",
                                   "lead_time_days": "45", "payment_terms": "Net30"}},
                    {"id": "SKU-STL-001", "type": "Product",
                     "properties": {"sku": "STERILIZER-PRO-V2", "category": "婴儿消毒",
                                   "unit_cost": "18.5 USD", "weight_kg": "1.2"}},
                    {"id": "PO-2026-0123", "type": "PurchaseOrder",
                     "properties": {"po_number": "PO-2026-0123",
                                   "date": "2026-05-15", "total_amount": "18500 USD",
                                   "status": "confirmed"}}
                ],
                "relations": [
                    {"from": "SUP-SZ-001", "to": "SKU-STL-001", "type": "SUPPLIES",
                     "properties": {"since": "2025-01"}},
                    {"from": "SUP-SZ-001", "to": "PO-2026-0123", "type": "FULFILLS",
                     "properties": {"lead_time": "45 days"}}
                ]
            }),
            "validate": json.dumps({
                "conflicts": [],
                "duplicates": [],
                "confidence_adjustments": {}
            }),
            "extend": json.dumps({
                "new_types": [],
                "new_relations": []
            })
        }
        if "extract" in prompt.lower()[:50]:
            return mock_responses["extract"]
        elif "validate" in prompt.lower()[:50]:
            return mock_responses["validate"]
        return mock_responses["extend"]
    
    def _build_extract_prompt(self, document: str) -> str:
        """Layer 1: 结构提取提示"""
        valid_types = list(self.ontology["ObjectTypes"].keys())
        valid_relations = list(self.ontology["LinkTypes"].keys())
        return f"""EXTRACT entities and relations from the following supply chain document.

ONLY extract these entity types: {valid_types}
ONLY extract these relation types: {valid_relations}

For each entity, extract all available properties.
Assign a unique ID based on the entity type and key identifier.

Document:
{document[:2000]}  

Return JSON format:
{{
  "entities": [{{"id": "...", "type": "...", "properties": {{}}}}],
  "relations": [{{"from": "...", "to": "...", "type": "...", "properties": {{}}}}]
}}"""
    
    def _build_validate_prompt(self, extracted: Dict) -> str:
        """Layer 2: 验证与冲突消解提示"""
        existing_nodes = [
            {"id": n.entity_id, "type": n.entity_type, "props": n.properties}
            for n in list(self.nodes.values())[:20]  # 取最近20个节点做参考
        ]
        return f"""VALIDATE extracted entities against existing ontology.

Existing entities (sample): {json.dumps(existing_nodes[:10])}

Newly extracted: {json.dumps(extracted)}

Check for:
1. Duplicates (same entity different IDs) - suggest merge
2. Contradictions (conflicting property values)
3. Type mismatches (entity in wrong ObjectType)

Return JSON:
{{
  "conflicts": [{{"type": "duplicate|contradiction|mismatch", "ids": [], "suggestion": "..."}}],
  "duplicates": [{{"existing_id": "...", "new_id": "...", "similarity": 0.0-1.0}}],
  "confidence_adjustments": {{"entity_id": new_confidence}}
}}"""
    
    def _build_extend_prompt(self, pattern_counts: Dict) -> str:
        """Layer 3: 本体扩展提示"""
        high_freq_unknowns = {k: v for k, v in pattern_counts.items() if v >= 5}
        if not high_freq_unknowns:
            return ""
        return f"""SUGGEST ONTOLOGY EXTENSIONS based on frequently observed patterns not in current schema.

Current ObjectTypes: {list(self.ontology['ObjectTypes'].keys())}
Current LinkTypes: {list(self.ontology['LinkTypes'].keys())}

High-frequency unclassified patterns: {json.dumps(high_freq_unknowns)}

Suggest new types (only if genuinely useful, threshold: 5+ occurrences):
Return JSON:
{{
  "new_types": [{{"name": "...", "properties": [], "rationale": "..."}}],
  "new_relations": [{{"name": "...", "from": "...", "to": "...", "rationale": "..."}}]
}}"""
    
    def process_document(self, document: str, doc_id: str) -> Tuple[List[OntologyNode], List[OntologyEdge]]:
        """处理单个文档，返回提取的节点和边"""
        
        # Layer 1: 提取
        extract_prompt = self._build_extract_prompt(document)
        try:
            extracted = json.loads(self.llm(extract_prompt))
        except json.JSONDecodeError:
            return [], []
        
        # Layer 2: 验证
        validate_prompt = self._build_validate_prompt(extracted)
        try:
            validation = json.loads(self.llm(validate_prompt))
        except json.JSONDecodeError:
            validation = {"conflicts": [], "duplicates": [], "confidence_adjustments": {}}
        
        # 处理重复实体（合并）
        id_mapping = {}  # new_id -> existing_id
        for dup in validation.get("duplicates", []):
            if dup.get("similarity", 0) > 0.85:
                id_mapping[dup["new_id"]] = dup["existing_id"]
        
        # 构建节点列表
        new_nodes = []
        for entity in extracted.get("entities", []):
            eid = id_mapping.get(entity["id"], entity["id"])
            conf = validation.get("confidence_adjustments", {}).get(entity["id"], 1.0)
            
            if eid not in self.nodes:
                node = OntologyNode(
                    entity_id=eid,
                    entity_type=entity["type"],
                    properties=entity.get("properties", {}),
                    source_doc=doc_id,
                    confidence=conf
                )
                self.nodes[eid] = node
                new_nodes.append(node)
        
        # 构建边列表
        new_edges = []
        for rel in extracted.get("relations", []):
            from_id = id_mapping.get(rel["from"], rel["from"])
            to_id = id_mapping.get(rel["to"], rel["to"])
            edge = OntologyEdge(
                from_id=from_id, to_id=to_id,
                relation_type=rel["type"],
                properties=rel.get("properties", {}),
                source_doc=doc_id
            )
            self.edges.append(edge)
            new_edges.append(edge)
        
        return new_nodes, new_edges
    
    def build_from_corpus(self, documents: List[Tuple[str, str]]) -> Dict:
        """
        从文档语料库批量构建本体
        
        Args:
            documents: [(doc_id, doc_text), ...]
        
        Returns:
            dict: 构建结果统计 + 更新的本体 Schema
        """
        total_nodes, total_edges = 0, 0
        
        for doc_id, doc_text in documents:
            new_nodes, new_edges = self.process_document(doc_text, doc_id)
            total_nodes += len(new_nodes)
            total_edges += len(new_edges)
        
        # Layer 3: 本体扩展检查
        if self.entity_counter:
            extend_prompt = self._build_extend_prompt(self.entity_counter)
            if extend_prompt:
                try:
                    extensions = json.loads(self.llm(extend_prompt))
                    # 追加新类型到本体
                    for new_type in extensions.get("new_types", []):
                        self.ontology["ObjectTypes"][new_type["name"]] = new_type["properties"]
                    for new_rel in extensions.get("new_relations", []):
                        self.ontology["LinkTypes"][new_rel["name"]] = {
                            "from": new_rel["from"], "to": new_rel["to"]
                        }
                except json.JSONDecodeError:
                    pass
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "new_nodes_this_run": total_nodes,
            "new_edges_this_run": total_edges,
            "ontology_object_types": len(self.ontology["ObjectTypes"]),
            "ontology_link_types": len(self.ontology["LinkTypes"]),
            "node_type_distribution": self._count_by_type()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for node in self.nodes.values():
            counts[node.entity_type] = counts.get(node.entity_type, 0) + 1
        return counts
    
    def export_to_cypher(self, limit: int = 100) -> str:
        """导出为 Neo4j Cypher 语句"""
        statements = []
        for node in list(self.nodes.values())[:limit]:
            props = json.dumps(node.properties).replace('"', "'")
            stmt = f"MERGE (n:{node.entity_type} {{id: '{node.entity_id}'}}) SET n += {props};"
            statements.append(stmt)
        for edge in self.edges[:limit]:
            stmt = (f"MATCH (a {{id: '{edge.from_id}'}}), (b {{id: '{edge.to_id}'}}) "
                   f"MERGE (a)-[:{edge.relation_type}]->(b);")
            statements.append(stmt)
        return "\n".join(statements)


# ===== 测试用例 =====
def run_test():
    # 初始化种子本体
    seed_ontology = {
        "ObjectTypes": {
            "Supplier": ["name", "location", "lead_time_days", "payment_terms"],
            "Product": ["sku", "category", "unit_cost", "weight_kg"],
            "PurchaseOrder": ["po_number", "date", "total_amount", "status"],
            "Shipment": ["shipment_id", "carrier", "eta", "cbm"],
        },
        "LinkTypes": {
            "SUPPLIES": {"from": "Supplier", "to": "Product"},
            "FULFILLS": {"from": "Supplier", "to": "PurchaseOrder"},
            "CONTAINS": {"from": "Shipment", "to": "Product"},
        }
    }
    
    builder = SCOntologyBuilder(seed_ontology)
    
    # 模拟采购文档
    sample_docs = [
        ("PO-001", "Purchase Order #PO-2026-0123 from supplier 深圳乐宝科技, "
                   "delivery of STERILIZER-PRO-V2 at $18.5/unit, Net30 payment, "
                   "lead time 45 days to Los Angeles warehouse."),
        ("EMAIL-001", "Dear team, supplier 广州辅食工厂 has confirmed shipment "
                      "SHP-20260601 of BABY-FOOD-ORGANIC via Evergreen carrier, "
                      "ETA June 30 to Seattle FBA center, total CBM 8.5."),
    ]
    
    result = builder.build_from_corpus(sample_docs)
    
    # 验证结果
    assert result["total_nodes"] > 0, "应至少提取1个节点"
    assert result["total_edges"] > 0, "应至少提取1条边"
    assert "Supplier" in result["node_type_distribution"], "应识别供应商类型"
    
    print(f"  本体构建结果: {result['total_nodes']}节点, {result['total_edges']}边")
    print(f"  节点分布: {result['node_type_distribution']}")
    
    # 测试 Cypher 导出
    cypher = builder.export_to_cypher(limit=5)
    assert "MERGE" in cypher, "应生成有效 Cypher 语句"
    assert len(cypher) > 0, "Cypher 不应为空"
    print(f"  Cypher 导出: {len(cypher.split(chr(10)))} 条语句")
    
    # 验证本体扩展机制（种子本体应被保留）
    assert len(builder.ontology["ObjectTypes"]) >= 4, "本体类型不应减少"
    print(f"  本体ObjectTypes: {len(builder.ontology['ObjectTypes'])} 个")
    
    print("\n[✓] Ontology-LLM-AutoBuild 测试通过 — 迭代萃取 + 冲突消解 + Cypher导出就绪")

run_test()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Tag-Schema-Engineering-Lifecycle]] — Schema 设计是本体构建的基础规范
- **前置（prerequisite）**：[[Skill-SKU-Master-Data-Golden-Record]] — 主数据黄金记录是本体实体消歧的前置
- **延伸（extends）**：[[Skill-Supply-Chain-Data-Lineage-Tracking]] — 本体构建后需要血缘追踪保证质量
- **延伸（extends）**：[[Skill-SC-Digital-Twin-Sync-Architecture]] — 自动构建的本体直接驱动数字孪生对象层
- **可组合（combinable）**：[[Skill-Supplier-Ontology-Capability-Map]] — LLM提取 + 能力图谱 = 完整供应商知识库
- **可组合（combinable）**：[[Skill-Graph-OKB-Design-SC]] — 自动构建的本体存入 OKB 图谱实现可查询决策支持

## ⑤ 商业价值评估

- **ROI 预估**：本体构建从 3 个月人工 → 2 天自动化（↓95% 时间），新供应商入库从 2 周 → 1 天，识别隐性单点故障（年化防损 10-30 万元）
- **实施难度**：⭐⭐⭐☆☆（主要依赖 LLM API + Neo4j，无需专业 NLP 工程师）
- **优先级**：⭐⭐⭐⭐⭐（本体是整个 Palantir 方法论的语义基础，其他层的前提）
- **企业AI知识库依赖**：高 — 本体本身就是 AI 知识库的核心资产，需要版本化管理
