---
title: Text-to-SQL Agent for Natural Language Data Query
module: 09-DataAgent-LLM
topic: text-to-sql
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase2
---

# Skill Card: Text-to-SQL Agent

## ① 算法原理

**核心问题**：业务团队（运营、市场、产品）需要数据但不懂SQL。每次提数需求都要排期给数据团队，平均等待2-3天。Text-to-SQL让非技术用户用自然语言直接查询数据库。

**技术演进**：

**v1.0 模板匹配（2010s）**：
- 预定义查询模板，用规则匹配用户意图
- 局限：只能处理预定义问题，灵活性差

**v2.0 Seq2Seq模型（2018-2022）**：
- 用Encoder-Decoder模型将自然语言直接映射到SQL
- 代表：Seq2SQL、SQLNet、IRNet
- 局限：需要大量标注数据，对复杂查询（多表JOIN、子查询）效果差

**v3.0 LLM时代（2023-2025）**：
- 大模型（GPT-4、Claude）直接生成SQL
- 配合Schema Linking（先识别相关表和字段，再生成SQL）
- 配合Self-Correction（生成SQL→执行→报错→修正）

**关键挑战与解法**：

| 挑战 | 解法 |
|------|------|
| Schema理解 | 将表结构、字段描述注入Prompt |
| 歧义消解 | 让用户澄清模糊查询 |
| 复杂JOIN | 分步生成：先确定表→再确定JOIN条件→最后生成完整SQL |
| 安全性 | 只读权限 + SQL注入过滤 + 敏感字段脱敏 |
| 准确性验证 | 执行SQL后用自然语言解释结果，让用户确认 |

**反直觉洞察**：
- Text-to-SQL的准确率瓶颈不在"生成SQL"，而在**Schema理解**——模型不知道`user_id`和`customer_id`是同一个东西
- 80%的业务查询可以用20个标准SQL模板覆盖——与其追求100%的任意查询，不如优先覆盖高频场景
- 最安全的做法是"人机协同"：模型生成草稿SQL，人工审核后执行

---

## ② 母婴出海应用案例

### 场景1：运营自助取数

**业务问题**：运营团队每天问数据团队同样的问题："昨天德国站奶粉类目的转化率是多少？"、"上周各品类的退货率排名"。数据团队80%时间在回答重复问题。

**SQL Agent应用**：

1. **Schema准备**：
   ```yaml
   tables:
     orders:
       - order_id: 订单ID
       - user_id: 用户ID
       - order_date: 订单日期
       - country: 国家（US/DE/UK/FR）
       - category: 品类
       - amount: 订单金额
       - is_returned: 是否退货
     users:
       - user_id: 用户ID
       - register_date: 注册日期
       - country: 注册国家
   ```

2. **用户提问**："上个月德国站各品类的转化率和退货率"

3. **Agent处理**：
   - Schema Linking：识别相关表（orders, users）、字段（country, category, amount, is_returned, order_date）
   - SQL生成：
     ```sql
     SELECT
       o.category,
       COUNT(DISTINCT o.user_id) * 1.0 / COUNT(DISTINCT u.user_id) as conversion_rate,
       SUM(CASE WHEN o.is_returned = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as return_rate
     FROM users u
     LEFT JOIN orders o ON u.user_id = o.user_id
       AND o.order_date >= '2026-04-01'
       AND o.country = 'DE'
     WHERE u.country = 'DE'
       AND u.register_date >= '2026-04-01'
     GROUP BY o.category;
     ```

4. **结果解释**：用自然语言解释SQL结果

**预期产出**：
- 数据团队重复取数工作量：80% → 20%
- 业务方取数等待：2-3天 → 即时
- 数据民主化：运营/产品/市场都能自助分析

### 场景2：高管自然语言报表

**业务问题**：CEO每周需要看核心指标Dashboard，但Dashboard是固定的，无法回答临时问题如"对比去年同期，Q1新客获取成本变化了多少？"

**SQL Agent + 可视化**：
- CEO用自然语言提问
- Agent生成SQL + 执行 + 生成图表
- 输出：带图表的自然语言报告

---

## ③ 代码模板

```python
"""
Text-to-SQL Agent — 自然语言转SQL查询
支持：Schema理解、SQL生成、执行、结果解释
"""

import re
from typing import List, Dict, Optional


class SchemaManager:
    """数据库Schema管理器"""

    def __init__(self):
        self.tables = {}

    def add_table(self, name: str, columns: Dict[str, str], description: str = ""):
        """
        添加表结构

        Args:
            name: 表名
            columns: {字段名: 字段描述}
            description: 表用途描述
        """
        self.tables[name] = {
            'columns': columns,
            'description': description
        }

    def get_schema_prompt(self) -> str:
        """生成Schema描述Prompt"""
        lines = ["数据库Schema:"]
        for table_name, info in self.tables.items():
            lines.append(f"\n表: {table_name}")
            if info['description']:
                lines.append(f"  用途: {info['description']}")
            lines.append("  字段:")
            for col, desc in info['columns'].items():
                lines.append(f"    - {col}: {desc}")
        return "\n".join(lines)

    def find_relevant_tables(self, query: str) -> List[str]:
        """根据查询内容识别相关表"""
        query_lower = query.lower()
        relevant = []
        for table_name, info in self.tables.items():
            score = 0
            # 表名匹配
            if table_name.lower() in query_lower:
                score += 3
            # 字段描述匹配
            for col, desc in info['columns'].items():
                if any(keyword in query_lower for keyword in [col.lower(), desc.lower()]):
                    score += 1
            if score > 0:
                relevant.append((table_name, score))

        relevant.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in relevant[:3]]  # 返回Top-3相关表


class SimpleSQLAgent:
    """简化版Text-to-SQL Agent（规则+模板版）"""

    def __init__(self, schema_manager: SchemaManager):
        self.schema = schema_manager
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """加载SQL模板"""
        return {
            '转化率': {
                'pattern': r'转化率|conversion',
                'sql': """
                    SELECT
                      {dimension},
                      COUNT(DISTINCT CASE WHEN o.order_id IS NOT NULL THEN u.user_id END) * 1.0
                        / COUNT(DISTINCT u.user_id) as conversion_rate
                    FROM users u
                    LEFT JOIN orders o ON u.user_id = o.user_id
                    WHERE {date_filter}
                    GROUP BY {dimension}
                """
            },
            '销售额': {
                'pattern': r'销售额|gmv|revenue|销售金额',
                'sql': """
                    SELECT
                      {dimension},
                      SUM(amount) as total_revenue,
                      COUNT(*) as order_count,
                      AVG(amount) as avg_order_value
                    FROM orders
                    WHERE {date_filter}
                    GROUP BY {dimension}
                """
            },
            '退货率': {
                'pattern': r'退货率|return',
                'sql': """
                    SELECT
                      {dimension},
                      SUM(CASE WHEN is_returned = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as return_rate
                    FROM orders
                    WHERE {date_filter}
                    GROUP BY {dimension}
                """
            },
            '用户数': {
                'pattern': r'用户数|用户数量|新用户',
                'sql': """
                    SELECT
                      {dimension},
                      COUNT(DISTINCT user_id) as user_count
                    FROM users
                    WHERE {date_filter}
                    GROUP BY {dimension}
                """
            }
        }

    def parse_query(self, query: str) -> Dict:
        """解析自然语言查询"""
        query_lower = query.lower()

        # 识别指标类型
        metric = None
        for name, template in self.templates.items():
            if re.search(template['pattern'], query_lower):
                metric = name
                break

        # 识别维度
        dimensions = []
        dim_keywords = {
            '国家': ['国家', 'country', '站'],
            '品类': ['品类', 'category', '类目'],
            '月份': ['月', 'month'],
            '渠道': ['渠道', 'source', 'channel']
        }
        for dim, keywords in dim_keywords.items():
            if any(kw in query_lower for kw in keywords):
                dimensions.append(dim)

        # 识别时间范围
        date_filter = "1=1"  # 默认无过滤
        if '昨天' in query or '昨日' in query:
            date_filter = "DATE(order_date) = CURRENT_DATE - 1"
        elif '上周' in query:
            date_filter = "order_date >= CURRENT_DATE - 7"
        elif '上个月' in query or '上月' in query:
            date_filter = "order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
        elif '本月' in query:
            date_filter = "order_date >= DATE_TRUNC('month', CURRENT_DATE)"

        return {
            'metric': metric or '销售额',
            'dimensions': dimensions or ['品类'],
            'date_filter': date_filter,
            'original_query': query
        }

    def generate_sql(self, query: str) -> str:
        """生成SQL"""
        parsed = self.parse_query(query)
        metric = parsed['metric']
        dimension = parsed['dimensions'][0] if parsed['dimensions'] else 'category'
        date_filter = parsed['date_filter']

        # 维度映射到SQL字段
        dim_map = {'国家': 'country', '品类': 'category', '月份': "DATE_TRUNC('month', order_date)", '渠道': 'source'}
        dim_col = dim_map.get(dimension, dimension)

        # 获取模板
        template = self.templates.get(metric, self.templates['销售额'])
        sql = template['sql'].format(dimension=dim_col, date_filter=date_filter)

        return sql.strip()

    def explain_result(self, query: str, sql: str, rows: List[Dict]) -> str:
        """用自然语言解释查询结果"""
        if not rows:
            return f"查询'{query}'没有返回结果。"

        # 取Top-3结果
        top_rows = rows[:3]
        explanations = []
        for row in top_rows:
            explanations.append(", ".join(f"{k}={v}" for k, v in row.items()))

        return f"关于'{query}'，Top结果：{'; '.join(explanations)}"


# 示例
if __name__ == '__main__':
    # 设置Schema
    schema = SchemaManager()
    schema.add_table('orders', {
        'order_id': '订单ID',
        'user_id': '用户ID',
        'order_date': '订单日期',
        'country': '国家（US/DE/UK/FR/JP）',
        'category': '品类（奶粉/纸尿裤/辅食/玩具/吸奶器）',
        'amount': '订单金额（美元）',
        'is_returned': '是否退货（0/1）'
    }, '订单表')
    schema.add_table('users', {
        'user_id': '用户ID',
        'register_date': '注册日期',
        'country': '注册国家',
        'source': '来源渠道'
    }, '用户表')

    # 创建Agent
    agent = SimpleSQLAgent(schema)

    # 测试查询
    queries = [
        "上周各品类的销售额",
        "上个月德国站的转化率",
        "各国家的退货率排名"
    ]

    for q in queries:
        print(f"\n查询: {q}")
        sql = agent.generate_sql(q)
        print(f"SQL:\n{sql}")
```

---


## ④ 技能关联

### 前置技能
- [Skill-ReAct-Reasoning-Acting](../10-MAS/[[Skill-ReAct-Reasoning-Acting]].md) — SQL Agent 的推理-执行循环建立在 ReAct 范式上

### 延伸技能
- [Skill-Data-to-Dashboard-Multi-Agent-Visualization](../09-DataAgent-LLM/[[Skill-Data-to-Dashboard-Multi-Agent-Visualization]].md) — SQL 结果直接驱动可视化
- [Skill-Root-Cause-Analysis-Agent](../09-DataAgent-LLM/[[Skill-Root-Cause-Analysis-Agent]].md) — SQL Agent 是 RCA 的查询底层

### 可组合
- [Skill-DeepAnalyze-Autonomous-Data-Science-Agent](../09-DataAgent-LLM/[[Skill-DeepAnalyze-Autonomous-Data-Science-Agent]].md) — SQL Agent 是自治数据科学 Agent 的工具节点

## ⑤ 商业价值评估

- **ROI**：数据团队重复取数工作量减少80%，业务决策速度提升10倍
- **难度**：⭐⭐⭐☆☆（3/5）— 高频场景用模板，复杂场景用LLM
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 数据民主化的核心基础设施
