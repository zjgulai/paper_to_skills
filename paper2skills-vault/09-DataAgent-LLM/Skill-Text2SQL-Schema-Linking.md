---
title: Schema-Linking感知Text2SQL — 列对齐与表联接推断
doc_type: knowledge
module: 09-DataAgent-LLM
topic: text2sql-schema-linking
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Schema-Linking感知Text2SQL

> **论文/方法来源**：DAIL-SQL: Efficient Prompt Engineering for Large Language Models (Gao et al., 2023) + BRIDGE: Bridging Textual and Tabular Data (Lin et al., 2020) + Schema Linking for Text-to-SQL (Gupta et al., 2022)
> **领域**：09-DataAgent-LLM ↔ 22-数据采集工程 | **类型**: 算法工具

## ① 算法原理

Text2SQL 的核心难点不是 SQL 语法，而是**Schema Linking**：将自然语言中的实体/概念正确对应到数据库表名和列名。错误的 Schema Linking 导致 SQL 逻辑正确但查询错表/错列。

**Schema Linking 流程**：
1. **实体抽取**：从问题中识别候选实体（"昨天的订单" → entity: "订单", time: "昨天"）
2. **列相似度计算**：用 BM25/向量相似度将实体与列名匹配，Top-K 候选列
3. **表联接推断**：根据候选列的所属表，推断需要 JOIN 的表路径（利用外键约束图）
4. **约束注入**：将 Schema Linking 结果注入 Prompt，约束 LLM 只在相关表/列范围内生成 SQL

**关键指标**：
- Schema Linking 准确率（列级）：目标 >90%
- 端到端 SQL 执行正确率（EX）：经 Schema Linking 增强后 +12-20pp

## ② 母婴出海应用案例

**场景A：运营自助查数（无需写 SQL）**
- 业务问题：运营提问"上周吸奶器在美国的退货率"，直接查 BI 需要 JOIN orders/returns/products 三表，非技术人员无从下手
- 数据要求：业务数据库 Schema（表结构 + 外键），数据字典（列名中文映射），历史 SQL 样例（≥20 条）
- 预期产出：从自然语言直接生成正确 SQL，Schema Linking 准确率 >85%，节省人工 30 分钟/次
- 业务价值：运营每日查数需求 10-20 个，节省数据分析师约 3h/天，年化节省 20 万元

**场景B：客服工单数据库自助分析**
- 业务问题：客服主管想查"本月投诉次数超过2次的 ASIN 列表"，需要关联工单、订单、商品三个系统
- 数据要求：多源数据库 Schema，ETL 后的大宽表
- 预期产出：schema-linking 自动识别跨表查询意图，生成带聚合条件的 SQL
- 业务价值：客服分析报表产出效率提升 5 倍

## ③ 代码模板

```python
"""
Schema-Linking 感知 Text2SQL — 列对齐 + 表联接推断
"""
import re
from typing import Dict, List, Tuple
import math


# 模拟业务数据库 Schema
BUSINESS_SCHEMA = {
    "orders": {
        "columns": ["order_id", "asin", "market", "order_date", "quantity", "revenue", "customer_id"],
        "description": "订单表，记录每笔销售",
        "primary_key": "order_id",
        "foreign_keys": {"customer_id": "customers.customer_id", "asin": "products.asin"}
    },
    "returns": {
        "columns": ["return_id", "order_id", "asin", "return_date", "return_reason", "refund_amount"],
        "description": "退货表",
        "primary_key": "return_id",
        "foreign_keys": {"order_id": "orders.order_id", "asin": "products.asin"}
    },
    "products": {
        "columns": ["asin", "product_name", "category", "brand", "cost", "list_price"],
        "description": "商品主数据",
        "primary_key": "asin",
        "foreign_keys": {}
    },
    "customers": {
        "columns": ["customer_id", "country", "registration_date", "customer_tier"],
        "description": "客户信息",
        "primary_key": "customer_id",
        "foreign_keys": {}
    }
}

# 中文→英文列名映射（数据字典）
COLUMN_ALIAS_MAP = {
    "退货": ["return_id", "return_date", "return_reason"],
    "退货率": ["returns.return_id", "orders.order_id"],
    "订单": ["order_id", "order_date", "orders"],
    "吸奶器": ["product_name", "asin", "category"],
    "美国": ["market", "country"],
    "上周": ["order_date", "return_date"],
    "收入": ["revenue"],
    "品牌": ["brand"],
    "类目": ["category"],
    "退款": ["refund_amount"],
}


def tokenize_question(question: str) -> List[str]:
    """简单分词（中文字符 + 英文单词）"""
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z_]+|\d+', question)
    return tokens


def compute_bm25_score(query_token: str, column_name: str, description: str = "") -> float:
    """简化 BM25 相似度（基于字符重叠）"""
    # 检查直接映射
    if query_token in COLUMN_ALIAS_MAP:
        mapped = COLUMN_ALIAS_MAP[query_token]
        if any(column_name in m or m in column_name for m in mapped):
            return 2.0

    # 字符级重叠
    q_chars = set(query_token.lower())
    c_chars = set(column_name.lower())
    d_chars = set(description.lower())

    col_overlap = len(q_chars & c_chars) / (len(q_chars) + 0.5)
    desc_overlap = len(q_chars & d_chars) / (len(q_chars) + 0.5) * 0.5
    return col_overlap + desc_overlap


def schema_linking(question: str, top_k: int = 5) -> Dict:
    """执行 Schema Linking：找出问题相关的表和列"""
    tokens = tokenize_question(question)
    column_scores = {}

    for table_name, table_info in BUSINESS_SCHEMA.items():
        for col in table_info["columns"]:
            full_col = f"{table_name}.{col}"
            score = 0
            for token in tokens:
                score += compute_bm25_score(token, col, table_info["description"])
            if score > 0:
                column_scores[full_col] = score

    # Top-K 候选列
    top_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    linked_tables = list(set(col.split('.')[0] for col, _ in top_columns))

    # 推断表联接路径
    join_paths = infer_join_paths(linked_tables)

    return {
        "question": question,
        "tokens": tokens,
        "top_columns": [(col, round(score, 3)) for col, score in top_columns],
        "linked_tables": linked_tables,
        "join_paths": join_paths
    }


def infer_join_paths(tables: List[str]) -> List[str]:
    """根据外键推断表联接路径"""
    join_conditions = []
    for table in tables:
        if table in BUSINESS_SCHEMA:
            fks = BUSINESS_SCHEMA[table]["foreign_keys"]
            for local_col, ref in fks.items():
                ref_table = ref.split('.')[0]
                if ref_table in tables:
                    join_conditions.append(
                        f"JOIN {ref_table} ON {table}.{local_col} = {ref}"
                    )
    return list(set(join_conditions))


def generate_sql_prompt(question: str, schema_link_result: Dict) -> str:
    """生成包含 Schema Linking 信息的 SQL 生成 Prompt"""
    linked_tables = schema_link_result["linked_tables"]
    top_cols = [c for c, _ in schema_link_result["top_columns"]]
    join_paths = schema_link_result["join_paths"]

    table_schemas = []
    for t in linked_tables:
        if t in BUSINESS_SCHEMA:
            cols = ", ".join(BUSINESS_SCHEMA[t]["columns"])
            table_schemas.append(f"表 {t}({cols})")

    prompt = f"""
你是 SQL 生成专家。根据以下信息生成正确的 SQL 查询。

【问题】{question}

【相关表结构】
{chr(10).join(table_schemas)}

【Schema Linking 分析】
- 相关列（按相关度排序）: {', '.join(top_cols[:3])}
- 需要的表: {', '.join(linked_tables)}
- 推荐 JOIN 路径: {'; '.join(join_paths) if join_paths else '单表查询'}

【SQL 生成规则】
1. 只使用上述相关表和列
2. 日期条件使用 WHERE 子句过滤
3. 聚合查询需要 GROUP BY
"""
    return prompt.strip()


# ===== 测试 =====
if __name__ == "__main__":
    test_questions = [
        "上周吸奶器在美国的退货率",
        "本月订单收入最高的前5个品牌",
        "退款金额超过50美元的退货原因分布",
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        result = schema_linking(q)
        print(f"问题: {q}")
        print(f"分词: {result['tokens']}")
        print(f"关联表: {result['linked_tables']}")
        print(f"Top列: {[c for c, s in result['top_columns'][:3]]}")
        if result['join_paths']:
            print(f"JOIN路径: {result['join_paths'][0]}")

    # 生成 Prompt 示例
    sl = schema_linking("上周吸奶器在美国的退货率")
    prompt = generate_sql_prompt("上周吸奶器在美国的退货率", sl)
    print(f"\n=== 生成的 SQL Prompt ===\n{prompt}")

    # 验证
    assert len(sl["top_columns"]) > 0, "Schema Linking 未返回任何列"
    assert len(sl["linked_tables"]) > 0, "未识别到相关表"
    sl2 = schema_linking("本月订单收入最高的前5个品牌")
    assert "orders" in sl2["linked_tables"] or "products" in sl2["linked_tables"]

    print("\n[✓] Schema-Linking感知Text2SQL测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-SQL-Agent-Text-to-SQL]]（基础 Text2SQL Agent 架构）
- **前置**：[[Skill-LLM-Business-Intelligence-Reasoning]]（LLM 业务推理能力）
- **延伸**：[[Skill-NL2Dashboard-Automation]]（Text2SQL → 自动仪表板）
- **可组合**：[[Skill-Data-to-Dashboard-Multi-Agent-Visualization]]（多 Agent 可视化链路）
- **可组合**：[[Skill-RAG-Enhanced-Data-Analysis]]（RAG + Schema Linking 融合）

## ⑤ 商业价值评估

- ROI 预估：运营自助查数节省数据分析师 3h/天，年化节省 20 万元；查询准确率提升减少错误决策
- 实施难度：⭐⭐⭐☆☆（需要维护数据字典和 Schema，外键关系完备是前提）
- 优先级：⭐⭐⭐⭐⭐
- 评估依据：母婴出海数据需求量大，运营 daily 需要大量临时查询，降低数据获取门槛是最高频需求之一
