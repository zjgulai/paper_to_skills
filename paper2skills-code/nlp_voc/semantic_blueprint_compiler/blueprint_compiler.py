"""
语义蓝图编译器 — 将异构图推理结果编译为可执行的语义蓝图
基于: Schema-Guided Generation (Willard & Louf, 2023) + AMR Parsing

核心能力:
1. Schema 约束生成 — 使用 CFG/FSM 约束 LLM 输出结构
2. 结构化输出编译 — 将图节点/边编译为标准化语义模板
3. 语义一致性校验 — 验证输出是否符合预定义蓝图规范
4. Task Blueprint 生成 — 将语义蓝图转化为可执行的任务描述

母婴电商场景: 将评论抽取结果编译为标准化的 VOC 分析语义蓝图
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class BlueprintType(Enum):
    """语义蓝图类型"""
    ENTITY = "entity"           # 实体定义
    RELATION = "relation"       # 关系定义
    EVENT = "event"             # 事件框架
    CONSTRAINT = "constraint"   # 约束规则
    TASK = "task"               # 任务描述


@dataclass
class SemanticBlueprint:
    """语义蓝图 — 标准化的语义结构定义"""
    blueprint_type: BlueprintType
    name: str
    schema: Dict[str, Any]      # 字段定义
    constraints: List[str] = field(default_factory=list)  # 约束规则
    examples: List[Dict] = field(default_factory=list)    # 示例


class SchemaGuidedCompiler:
    """
    语义蓝图编译器

    核心流程:
    1. 接收异构图推理结果 (实体、关系、事件)
    2. 根据 Schema 模板进行结构化转换
    3. 应用约束规则进行校验和修正
    4. 输出标准化的语义蓝图
    """

    def __init__(self):
        self.schemas = self._load_default_schemas()

    def _load_default_schemas(self) -> Dict[str, SemanticBlueprint]:
        """加载默认的语义蓝图模板"""
        return {
            "voc_entity": SemanticBlueprint(
                blueprint_type=BlueprintType.ENTITY,
                name="VOC_Entity",
                schema={
                    "entity_id": "string",
                    "entity_type": "enum[PRODUCT, BRAND, ATTRIBUTE, USER_GROUP]",
                    "text": "string",
                    "span": "tuple[int, int]",
                    "confidence": "float[0,1]"
                },
                constraints=[
                    "entity_type 必须在预定义枚举中",
                    "confidence >= 0.7",
                    "text 不能为空"
                ]
            ),
            "voc_relation": SemanticBlueprint(
                blueprint_type=BlueprintType.RELATION,
                name="VOC_Relation",
                schema={
                    "relation_id": "string",
                    "relation_type": "enum[has_attribute, positive_for, negative_for, compare_with]",
                    "head_entity_id": "string",
                    "tail_entity_id": "string",
                    "confidence": "float[0,1]"
                },
                constraints=[
                    "head_entity_id 和 tail_entity_id 必须存在于实体库",
                    "relation_type 必须在预定义枚举中",
                    "confidence >= 0.6"
                ]
            ),
            "voc_event": SemanticBlueprint(
                blueprint_type=BlueprintType.EVENT,
                name="VOC_Event",
                schema={
                    "event_id": "string",
                    "event_type": "enum[PURCHASE, RETURN, COMPLAINT, RECOMMENDATION]",
                    "trigger": "string",
                    "trigger_span": "tuple[int, int]",
                    "arguments": "list[{role, entity_id}]",
                    "timestamp": "optional[string]"
                },
                constraints=[
                    "event_type 必须在预定义枚举中",
                    "trigger 不能为空",
                    "arguments 中所有 entity_id 必须存在于实体库"
                ]
            ),
            "task_blueprint": SemanticBlueprint(
                blueprint_type=BlueprintType.TASK,
                name="Task_Blueprint",
                schema={
                    "task_id": "string",
                    "task_type": "enum[EXTRACT, ANALYZE, SUMMARIZE, ALERT]",
                    "input_schema": "dict",
                    "output_schema": "dict",
                    "required_skills": "list[string]",
                    "quality_threshold": "float[0,1]"
                },
                constraints=[
                    "task_type 必须在预定义枚举中",
                    "required_skills 必须存在于 Skill Registry",
                    "quality_threshold >= 0.8"
                ]
            ),
        }

    def compile_entity(self, raw_entity: Dict) -> Optional[Dict]:
        """编译实体到语义蓝图"""
        schema = self.schemas["voc_entity"]

        compiled = {
            "blueprint_type": "ENTITY",
            "schema_version": "1.0",
            "data": {}
        }

        # 字段映射和类型检查
        for field_name, field_type in schema.schema.items():
            if field_name in raw_entity:
                value = raw_entity[field_name]
                # 简化类型检查
                compiled["data"][field_name] = value
            else:
                # 必填字段缺失
                if field_name in ["entity_id", "entity_type", "text"]:
                    return None

        # 约束校验
        if compiled["data"].get("confidence", 1.0) < 0.7:
            compiled["data"]["_warning"] = "置信度低于阈值，建议人工复核"

        return compiled

    def compile_relation(self, raw_relation: Dict, entity_registry: Dict) -> Optional[Dict]:
        """编译关系到语义蓝图"""
        schema = self.schemas["voc_relation"]

        # 校验实体存在性
        head_id = raw_relation.get("head_entity_id")
        tail_id = raw_relation.get("tail_entity_id")

        if head_id not in entity_registry or tail_id not in entity_registry:
            return None

        return {
            "blueprint_type": "RELATION",
            "schema_version": "1.0",
            "data": {
                "relation_id": raw_relation.get("relation_id", "rel_" + head_id + "_" + tail_id),
                "relation_type": raw_relation.get("relation_type"),
                "head_entity": entity_registry.get(head_id),
                "tail_entity": entity_registry.get(tail_id),
                "confidence": raw_relation.get("confidence", 1.0),
            }
        }

    def compile_event(self, raw_event: Dict, entity_registry: Dict) -> Optional[Dict]:
        """编译事件到语义蓝图"""
        schema = self.schemas["voc_event"]

        # 校验 arguments 中的实体
        arguments = raw_event.get("arguments", [])
        validated_args = []

        for arg in arguments:
            entity_id = arg.get("entity_id")
            if entity_id in entity_registry:
                validated_args.append({
                    "role": arg.get("role"),
                    "entity": entity_registry[entity_id]
                })

        return {
            "blueprint_type": "EVENT",
            "schema_version": "1.0",
            "data": {
                "event_id": raw_event.get("event_id", "evt_" + raw_event.get("trigger", "")),
                "event_type": raw_event.get("event_type"),
                "trigger": raw_event.get("trigger"),
                "arguments": validated_args,
                "timestamp": raw_event.get("timestamp"),
            }
        }

    def compile_task_blueprint(self, task_description: str, available_skills: List[str]) -> Dict:
        """
        将自然语言任务描述编译为 Task Blueprint

        这是语义蓝图编译器的核心输出 — 将人类意图转化为机器可执行的结构化任务。
        """
        # 简化的任务解析（生产环境使用 LLM + Schema 约束）
        task_keywords = {
            "抽取": "EXTRACT",
            "分析": "ANALYZE",
            "汇总": "SUMMARIZE",
            "预警": "ALERT",
            "extract": "EXTRACT",
            "analyze": "ANALYZE",
            "summarize": "SUMMARIZE",
            "alert": "ALERT",
        }

        task_type = "EXTRACT"
        for keyword, type_val in task_keywords.items():
            if keyword in task_description.lower():
                task_type = type_val
                break

        # 推断所需技能
        required_skills = []
        if "实体" in task_description or "entity" in task_description.lower():
            required_skills.append("InstructUIE")
        if "情感" in task_description or "sentiment" in task_description.lower():
            required_skills.append("ABSA")
        if "图" in task_description or "graph" in task_description.lower():
            required_skills.append("HGT")
        if "关系" in task_description or "relation" in task_description.lower():
            required_skills.append("R-GCN")

        # 过滤不可用技能
        required_skills = [s for s in required_skills if s in available_skills]

        return {
            "blueprint_type": "TASK",
            "schema_version": "1.0",
            "data": {
                "task_id": f"task_{hash(task_description) % 10000:04d}",
                "task_type": task_type,
                "description": task_description,
                "input_schema": {"type": "raw_text", "format": "string"},
                "output_schema": {"type": "structured", "format": "json"},
                "required_skills": required_skills,
                "quality_threshold": 0.85,
                "fallback_strategy": "human_review" if not required_skills else "auto"
            }
        }

    def compile_full_blueprint(self, extraction_results: Dict) -> Dict:
        """
        将完整的抽取结果编译为统一的语义蓝图

        输入: 多任务抽取结果（实体、关系、事件）
        输出: 标准化的语义蓝图，可直接用于下游任务
        """
        entity_registry = {}
        compiled_entities = []
        compiled_relations = []
        compiled_events = []

        # 1. 编译实体
        for raw_entity in extraction_results.get("entities", []):
            compiled = self.compile_entity(raw_entity)
            if compiled:
                entity_registry[compiled["data"]["entity_id"]] = compiled["data"]
                compiled_entities.append(compiled)

        # 2. 编译关系
        for raw_relation in extraction_results.get("relations", []):
            compiled = self.compile_relation(raw_relation, entity_registry)
            if compiled:
                compiled_relations.append(compiled)

        # 3. 编译事件
        for raw_event in extraction_results.get("events", []):
            compiled = self.compile_event(raw_event, entity_registry)
            if compiled:
                compiled_events.append(compiled)

        return {
            "blueprint_version": "1.0",
            "compiled_at": "2026-05-10",
            "entities": compiled_entities,
            "relations": compiled_relations,
            "events": compiled_events,
            "entity_registry": entity_registry,
            "statistics": {
                "num_entities": len(compiled_entities),
                "num_relations": len(compiled_relations),
                "num_events": len(compiled_events),
            }
        }


# ============================================
# 母婴电商 VOC 语义蓝图编译示例
# ============================================

def demo_blueprint_compilation():
    """演示从抽取结果到语义蓝图的编译过程"""
    print("=" * 70)
    print("语义蓝图编译器 — Schema-Guided Compilation")
    print("=" * 70)

    compiler = SchemaGuidedCompiler()

    # 模拟 InstructUIE 抽取结果
    extraction_results = {
        "entities": [
            {"entity_id": "e1", "entity_type": "PRODUCT", "text": "Spectra S1", "span": (0, 10), "confidence": 0.95},
            {"entity_id": "e2", "entity_type": "ATTRIBUTE", "text": "静音", "span": (22, 24), "confidence": 0.88},
            {"entity_id": "e3", "entity_type": "ATTRIBUTE", "text": "价格", "span": (30, 32), "confidence": 0.92},
            {"entity_id": "e4", "entity_type": "PRODUCT", "text": "储奶袋", "span": (50, 53), "confidence": 0.90},
        ],
        "relations": [
            {"relation_id": "r1", "relation_type": "has_attribute", "head_entity_id": "e1", "tail_entity_id": "e2", "confidence": 0.85},
            {"relation_id": "r2", "relation_type": "negative_for", "head_entity_id": "e1", "tail_entity_id": "e3", "confidence": 0.78},
            {"relation_id": "r3", "relation_type": "complement_of", "head_entity_id": "e1", "tail_entity_id": "e4", "confidence": 0.80},
        ],
        "events": [
            {"event_id": "ev1", "event_type": "PURCHASE", "trigger": "买了", "arguments": [{"role": "ARG0", "entity_id": "e1"}]},
        ],
    }

    print("\n[输入] InstructUIE 抽取结果")
    print(f"   实体: {len(extraction_results['entities'])} 个")
    print(f"   关系: {len(extraction_results['relations'])} 条")
    print(f"   事件: {len(extraction_results['events'])} 个")

    # 编译为语义蓝图
    print("\n[编译] Schema-Guided Compilation...")
    blueprint = compiler.compile_full_blueprint(extraction_results)

    print(f"\n[输出] 语义蓝图 (v{blueprint['blueprint_version']})")
    print(f"   编译时间: {blueprint['compiled_at']}")
    print(f"   实体: {blueprint['statistics']['num_entities']} 个")
    print(f"   关系: {blueprint['statistics']['num_relations']} 条")
    print(f"   事件: {blueprint['statistics']['num_events']} 个")

    # 展示编译后的实体
    print("\n[编译后实体示例]")
    for entity in blueprint["entities"][:2]:
        data = entity["data"]
        print(f"   {data['entity_type']}: '{data['text']}' (置信度: {data['confidence']})")

    # 展示编译后的关系
    print("\n[编译后关系示例]")
    for relation in blueprint["relations"][:2]:
        data = relation["data"]
        head = data['head_entity']['text']
        tail = data['tail_entity']['text']
        print(f"   ({head}) --[{data['relation_type']}]--> ({tail})")

    return blueprint


def demo_task_blueprint_generation():
    """演示从自然语言任务生成 Task Blueprint"""
    print("\n" + "=" * 70)
    print("Task Blueprint 生成")
    print("=" * 70)

    compiler = SchemaGuidedCompiler()
    available_skills = ["InstructUIE", "ABSA", "HGT", "R-GCN", "AutoGen", "MetaGPT"]

    tasks = [
        "抽取本周所有吸奶器评论中的实体和情感",
        "分析竞品 A 和竞品 B 的用户评价差异",
        "汇总本月全品类 VOC 趋势报告",
        "预警产品质量投诉突增",
    ]

    print(f"\n可用技能: {', '.join(available_skills)}\n")

    for task_desc in tasks:
        blueprint = compiler.compile_task_blueprint(task_desc, available_skills)
        data = blueprint["data"]

        print(f"任务: {task_desc}")
        print(f"  → 类型: {data['task_type']}")
        print(f"  → 所需技能: {', '.join(data['required_skills'])}")
        print(f"  → 质量阈值: {data['quality_threshold']}")
        print(f"  → 回退策略: {data['fallback_strategy']}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    blueprint = demo_blueprint_compilation()
    demo_task_blueprint_generation()

    print("\n生产环境建议:")
    print("  1. 使用 Outlines 库实现高效的 Schema 约束解码")
    print("  2. 定义完整的 JSON Schema 并使用 Pydantic 校验")
    print("  3. 建立蓝图版本管理机制，支持兼容性检查")
    print("  4. 与 Skill Registry 集成，动态解析所需技能")
    print("  5. 实现蓝图的序列化和反序列化，支持跨系统传输")
