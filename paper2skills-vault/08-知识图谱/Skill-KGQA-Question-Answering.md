---
title: Knowledge Graph Question Answering (KGQA)
module: 08-知识图谱
topic: kgqa
status: stable
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: Knowledge Graph Question Answering (KGQA)

## ① 算法原理

**核心问题**：构建了产品知识图谱后，如何让非技术人员（运营、客服、业务方）用自然语言查询它？KGQA把"图谱查询"从SPARQL/Cypher简化为人话。

**技术架构**：

```
用户提问 → 实体链接 → 关系预测 → 子图检索 → 答案生成
   ↓           ↓           ↓           ↓           ↓
自然语言   识别提到的    预测需要的   在图谱中     整理为
            实体          关系路径    查询子图     自然语言
```

**关键步骤**：

**1. 实体链接（Entity Linking）**
- 从用户问题中识别提到的实体
- "**爱他美**3段奶粉含**DHA**吗？" → 链接到图谱中的`爱他美3段`和`DHA`节点
- 挑战：用户可能用别名（"爱他美"vs"Aptamil"）、错别字、口语化表达

**2. 关系预测（Relation Prediction）**
- 预测用户问题涉及的图谱关系类型
- "含" → `contains`关系
- "适合" → `suitable_for`关系
- "和...有什么区别" → 需要对比两个实体的属性

**3. 子图检索（Subgraph Retrieval）**
- 在知识图谱中找到相关的三元组子集
- "爱他美3段" → `contains` → ? → 检索所有成分

**4. 答案生成（Answer Generation）**
- 用模板或LLM将子图结果转化为自然语言
- 简单问题：模板填充（"是的，爱他美3段含有DHA"）
- 复杂问题：LLM生成（"爱他美3段和美赞臣3段都含有DHA，但爱他美额外含有益生菌"）

**2025年前沿：LLM + KG融合**

- **Retrieve-then-Generate**：先从KG检索相关事实，再用LLM生成答案（减少幻觉）
- **KG-augmented LLM**：在LLM的prompt中注入KG子图作为上下文
- **UniKGQA**：统一编码问题和图谱，端到端训练

**反直觉洞察**：
- KGQA的准确率瓶颈不在"答案生成"，而在**实体链接**——用户不会用标准名称提问
- 80%的电商问题可以用10种标准查询模式覆盖（"含什么"、"适合谁"、"和A的区别"、"价格多少"）
- 复杂推理问题（多跳、比较、计数）仍然是大挑战

---

## ② 母婴出海应用案例

### 场景1：客服知识库问答

**业务问题**：客服每天回答重复问题："这款奶粉含DHA吗？"、"3段和2段有什么区别？"、"这款吸奶器适合背奶妈妈吗？"——答案都在知识图谱里，但客服需要手动查。

**KGQA应用**：

1. **知识图谱构建**：
   ```
   爱他美3段 --contains--> DHA
   爱他美3段 --contains--> 益生菌
   爱他美3段 --suitable_for--> 12-36个月
   爱他美3段 --brand--> 爱他美
   爱他美3段 --price--> $35
   ```

2. **用户提问**："爱他美3段含有哪些成分？"

3. **KGQA处理**：
   - 实体链接：`爱他美3段`
   - 关系预测：`contains`
   - 子图检索：`(爱他美3段, contains, ?)`
   - 答案生成：`"爱他美3段含有DHA、益生菌、GOS/FOS益生元组合"`

4. **复杂提问**："推荐一款含DHA且适合1岁宝宝的奶粉"
   - 多条件查询：`contains=DHA` + `suitable_for=12-36个月`
   - 返回：爱他美3段、美赞臣3段、雀巢3段

**预期产出**：
- 客服响应速度：2分钟查资料 → 5秒自动回答
- 客服培训成本：降低60%（不需要记忆所有产品知识）
- 用户满意度：响应快+答案准确

### 场景2：选品助手

**业务问题**：采购团队需要快速了解竞品差异，用于选品决策。

**KGQA应用**：
- "对比爱他美3段和美赞臣3段的成分差异"
- KGQA检索两个实体的所有属性，对比差异项
- 输出：表格对比 + 自然语言总结

---

## ③ 代码模板

```python
"""
Knowledge Graph Question Answering (KGQA) — 知识图谱问答
支持：实体链接、关系预测、子图检索、答案生成
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class SimpleKGQA:
    """简化版KGQA系统"""

    def __init__(self):
        self.kg = {}  # {entity: {relation: [target_entities]}}
        self.entity_aliases = {}  # {alias: canonical_name}
        self.relation_patterns = {
            'contains': r'含有什么|成分|包含|有什么',
            'suitable_for': r'适合谁|适用|几岁|多大',
            'brand': r'什么品牌|牌子|哪个品牌',
            'price': r'多少钱|价格|售价',
            'compare': r'区别|差异|对比|和.*有什么',
        }

    def add_fact(self, head: str, relation: str, tail: str):
        """添加知识图谱三元组"""
        if head not in self.kg:
            self.kg[head] = defaultdict(list)
        self.kg[head][relation].append(tail)

    def add_alias(self, alias: str, canonical: str):
        """添加实体别名"""
        self.entity_aliases[alias.lower()] = canonical

    def _link_entity(self, question: str) -> Optional[str]:
        """实体链接：从问题中找提到的实体"""
        question_lower = question.lower()

        # 先匹配别名
        for alias, canonical in self.entity_aliases.items():
            if alias in question_lower:
                return canonical

        # 再匹配主实体名
        for entity in self.kg.keys():
            if entity.lower() in question_lower:
                return entity

        return None

    def _predict_relation(self, question: str) -> Optional[str]:
        """关系预测：判断用户问的是什么关系"""
        for relation, pattern in self.relation_patterns.items():
            if re.search(pattern, question.lower()):
                return relation

        # 默认：如果问题含"是""吗"，可能是属性查询
        if '是' in question or '吗' in question:
            return 'contains'

        return None

    def _is_compare_question(self, question: str) -> bool:
        """判断是否是比较问题"""
        return bool(re.search(self.relation_patterns['compare'], question.lower()))

    def _extract_entities_for_compare(self, question: str) -> List[str]:
        """从比较问题中提取两个实体"""
        found = []
        for entity in self.kg.keys():
            if entity.lower() in question.lower():
                found.append(entity)
        return found[:2]

    def answer(self, question: str) -> Dict:
        """回答用户问题"""
        result = {
            'question': question,
            'entity': None,
            'relation': None,
            'answer': None,
            'confidence': 0.0
        }

        # 1. 判断是否是比较问题
        if self._is_compare_question(question):
            entities = self._extract_entities_for_compare(question)
            if len(entities) == 2:
                return self._compare_entities(entities[0], entities[1], question)

        # 2. 实体链接
        entity = self._link_entity(question)
        if not entity:
            result['answer'] = "抱歉，我没有找到相关产品信息。"
            return result
        result['entity'] = entity

        # 3. 关系预测
        relation = self._predict_relation(question)
        if not relation:
            result['answer'] = f"关于{entity}，您可以问它的成分、适用年龄、品牌或价格。"
            return result
        result['relation'] = relation

        # 4. 子图检索
        if entity in self.kg and relation in self.kg[entity]:
            targets = self.kg[entity][relation]
            result['confidence'] = 0.95

            # 5. 答案生成
            if relation == 'contains':
                result['answer'] = f"{entity}含有：{', '.join(targets)}。"
            elif relation == 'suitable_for':
                result['answer'] = f"{entity}适合：{', '.join(targets)}。"
            elif relation == 'brand':
                result['answer'] = f"{entity}的品牌是：{targets[0]}。"
            elif relation == 'price':
                result['answer'] = f"{entity}的价格是：{targets[0]}。"
            else:
                result['answer'] = f"{entity}的{relation}是：{', '.join(targets)}。"
        else:
            result['answer'] = f"关于{entity}的{relation}信息，我暂时没有记录。"
            result['confidence'] = 0.3

        return result

    def _compare_entities(self, entity_a: str, entity_b: str, question: str) -> Dict:
        """对比两个实体"""
        info_a = self.kg.get(entity_a, {})
        info_b = self.kg.get(entity_b, {})

        # 找共同关系和差异关系
        all_relations = set(info_a.keys()) | set(info_b.keys())

        comparison = []
        for rel in all_relations:
            vals_a = set(info_a.get(rel, []))
            vals_b = set(info_b.get(rel, []))
            common = vals_a & vals_b
            only_a = vals_a - vals_b
            only_b = vals_b - vals_a

            if common:
                comparison.append(f"两者都{rel}：{', '.join(common)}")
            if only_a:
                comparison.append(f"只有{entity_a}{rel}：{', '.join(only_a)}")
            if only_b:
                comparison.append(f"只有{entity_b}{rel}：{', '.join(only_b)}")

        return {
            'question': question,
            'entity': f"{entity_a} vs {entity_b}",
            'relation': 'compare',
            'answer': '\n'.join(comparison) if comparison else f"{entity_a}和{entity_b}的对比信息不足。",
            'confidence': 0.9
        }


def build_baby_product_kgqa():
    """构建母婴产品KGQA示例"""
    kgqa = SimpleKGQA()

    # 添加知识
    facts = [
        ('爱他美3段', 'contains', 'DHA'),
        ('爱他美3段', 'contains', '益生菌'),
        ('爱他美3段', 'contains', 'GOS/FOS'),
        ('爱他美3段', 'suitable_for', '12-36个月'),
        ('爱他美3段', 'brand', '爱他美'),
        ('爱他美3段', 'price', '$35'),
        ('美赞臣3段', 'contains', 'DHA'),
        ('美赞臣3段', 'contains', '乳铁蛋白'),
        ('美赞臣3段', 'suitable_for', '12-36个月'),
        ('美赞臣3段', 'brand', '美赞臣'),
        ('美赞臣3段', 'price', '$32'),
        ('Momcozy S12', 'brand', 'Momcozy'),
        ('Momcozy S12', 'price', '$120'),
        ('Momcozy S12', 'suitable_for', '背奶妈妈'),
        ('Momcozy S12', 'contains', '静音马达'),
    ]

    for h, r, t in facts:
        kgqa.add_fact(h, r, t)

    # 添加别名
    aliases = {
        'aptamil 3': '爱他美3段',
        '爱他美三段': '爱他美3段',
        '美赞臣三段': '美赞臣3段',
        'enfamil 3': '美赞臣3段',
        's12': 'Momcozy S12',
    }
    for alias, canonical in aliases.items():
        kgqa.add_alias(alias, canonical)

    return kgqa


if __name__ == '__main__':
    kgqa = build_baby_product_kgqa()

    questions = [
        "爱他美3段含有哪些成分？",
        "美赞臣3段适合多大的宝宝？",
        "爱他美3段和美赞臣3段有什么区别？",
        "Momcozy S12多少钱？",
        "Aptamil 3含DHA吗？",
    ]

    for q in questions:
        result = kgqa.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
```

---


## ④ 技能关联

### 前置技能
- [Skill-Knowledge-Graph-for-Skills-Management](../08-知识图谱/Skill-Knowledge-Graph-for-Skills-Management.md) — KG schema 是 KGQA 的查询语义基础
- [Skill-Dense-Retrieval-Ecommerce-Semantic-Search](../08-知识图谱/Skill-Dense-Retrieval-Ecommerce-Semantic-Search.md) — 稠密检索定位相关子图

### 延伸技能
- [Skill-GraphRAG-Knowledge-Enhanced-Retrieval](../08-知识图谱/Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md) — KGQA + RAG 形成知识增强问答

### 可组合
- [Skill-SQL-Agent-Text-to-SQL](../09-DataAgent-LLM/Skill-SQL-Agent-Text-to-SQL.md) — KGQA 与 Text-to-SQL 共同覆盖结构化问答

## ⑤ 商业价值评估

- **ROI**：客服效率提升3-5倍，知识查询等待从分钟级降到秒级
- **难度**：⭐⭐⭐☆☆（3/5）— 简单模式匹配易实现，复杂推理需要LLM
- **优先级**：⭐⭐⭐⭐☆（4/5）— 知识图谱的"最后一公里"，让业务方真正用上图谱
