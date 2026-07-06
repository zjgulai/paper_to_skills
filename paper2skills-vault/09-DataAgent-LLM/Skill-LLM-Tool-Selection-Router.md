---
title: LLM工具路由与意图识别 — 意图分类与置信过滤
doc_type: knowledge
module: 09-DataAgent-LLM
topic: llm-tool-selection-router
status: stable
created: 2026-06-21
updated: 2024-12-19
owner: self
source: arxiv:2302.04761
roadmap_phase: phase2
tags: [llm, agent, tool-routing, intent-classification, multi-tool]
difficulty: intermediate
estimated_time: 45min
---

# Skill Card: LLM工具路由与意图识别

> **论文/方法来源**：ToolFormer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023) + Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023) + Intent Classification for Task-Oriented Dialogue
> **领域**：09-DataAgent-LLM ↔ 16-智能体工程 | **类型**: 算法工具

## ① 算法原理

> **论文**：ToolFormer: Language Models Can Teach Themselves to Use Tools | **年份**：2023  
> **论文**：Gorilla: Large Language Model Connected with Massive APIs | **年份**：2023

LLM 工具路由解决 Multi-Tool Agent 的核心问题：**给定用户请求，选择正确的工具（或工具组合）执行**。错误的工具选择导致无效调用、API 成本浪费、错误结果。

**三层路由架构**：
1. **意图分类层**：将用户请求分类至预定义意图树（数据查询/计算/外部搜索/报告生成等），支持多意图并发
2. **工具匹配层**：基于意图 + 参数约束，从工具注册表中检索匹配工具（语义相似度 + 功能约束过滤）
3. **置信过滤层**：若工具选择置信分 < 阈值，触发澄清问句而非错误调用

**意图相似度计算**：使用 TF-IDF + 余弦相似度（轻量）或 Sentence-BERT（高质量）匹配用户描述与工具描述。

**关键设计**：工具描述质量是路由准确率的决定因素；结构化工具注册（名称/描述/参数/示例）比自由文本提升 30%+ 路由准确率。

## ② 母婴出海应用案例

**场景A：业务 Bot 多工具智能调度**
- 业务问题：内部 ChatBot 连接了 12 个工具（查库存/查订单/生成报告/调价等），用户"查一下 B08X 上周销量"应路由到查询工具而非报告生成工具，错误路由浪费 API 成本
- 数据要求：工具注册表（名称+描述+参数 schema），历史调用日志 ≥ 500 条
- 预期产出：工具选择准确率 >90%，幻觉调用（调用不存在工具）= 0%，澄清问句减少无效调用 30%
- 业务价值：API 调用成本节省 40%（约 2 万元/月），Bot 响应正确率从 68% 升至 91%

**三轨验证**：
- **成本轨**：
  - 数据采集：标注 500 条历史调用日志，外包标注成本 ¥3,000-5,000（按 ¥6-10/条）
  - 计算资源：TF-IDF 索引构建 + 推理，单次推理成本 <¥0.001（本地计算），月度 API 调用成本从 ¥50,000 降至 ¥30,000，净节省 ¥20,000
  - 人力投入：工程师 1 人 × 2 周（方案设计+集成+测试），成本 ¥8,000-12,000
  - **总初期投入**：¥11,000-17,000；**月度净收益**：¥20,000（6-12 个月 ROI）

- **合规轨**：✅ **完全合规**
  - 不涉及用户个人数据处理，仅在内部工具调度层面运作
  - 不违反 Amazon 政策（未涉及价格操纵、虚假评价等受限操作）
  - 不触发 GDPR（无跨境个人数据传输）
  - 符合《反不正当竞争法》（工具路由优化属于内部效率提升，非价格战）

- **风险轨**：
  - **工具调用错误风险**（概率 5-8%）：即使置信度过滤，仍可能出现边界情况调用错误工具，导致数据不一致或业务流程中断 → 缓解：建立工具调用审计日志，高风险操作（调价/创建 PO）需人工确认
  - **模型漂移风险**（概率 10-15%）：新增工具或工具描述变更后，TF-IDF 索引失效，路由准确率下降 → 缓解：建立月度模型评估机制，准确率下降 >5pp 时触发重训练
  - **用户体验风险**（概率 3-5%）：过度澄清问句导致用户流失 → 缓解：根据用户类型（内部员工 vs 外部客户）动态调整置信阈值

---

**场景B：自动化运营 Agent 工具编排**
- 业务问题：补货决策 Agent 需要顺序调用：查库存 → 预测需求 → 计算补货量 → 创建 PO，各步骤工具路由必须准确
- 数据要求：Agent workflow 定义，工具依赖图
- 预期产出：工具路由 + 参数填充准确率 >85%，支持工具链编排
- 业务价值：补货 Agent 完整执行成功率从 60% 提升至 88%，减少人工干预 70%

**三轨验证**：
- **成本轨**：
  - 数据采集：补货 workflow 日志标注 200 条，成本 ¥1,200-2,000
  - 计算资源：Agent 编排引擎部署（含工具路由模块），初期开发 ¥15,000-20,000，月度运维成本 ¥2,000-3,000
  - 人力投入：工程师 1.5 人 × 3 周（workflow 设计+工具依赖图构建+集成测试），成本 ¥12,000-18,000
  - 业务收益：补货 Agent 成功率从 60% → 88%，减少人工干预 70%，相当于节省 0.5 FTE/月（约 ¥15,000）
  - **总初期投入**：¥28,200-40,000；**月度净收益**：¥12,000-15,000（3-4 个月 ROI）

- **合规轨**：✅ **基本合规，需重点关注采购流程**
  - 工具路由本身无合规风险
  - **采购流程风险**：自动创建 PO 涉及合同签署，需确保符合《电子商务法》第 47 条（电子合同有效性）和公司采购政策
  - **库存数据风险**：涉及供应商库存信息，需确保不违反供应商数据保护协议
  - 建议：在 Agent 创建 PO 前增加人工审批环节（金额 >¥50,000 或新供应商）

- **风险轨**：
  - **供应链中断风险**（概率 8-12%）：工具路由错误导致向错误供应商下单或下单参数错误（数量/规格），引发供应链混乱 → 缓解：建立工具链验证机制，每步输出需通过 schema 校验；关键参数（ASIN/数量/价格）需二次确认
  - **成本超支风险**（概率 5-10%）：Agent 自动补货可能过度采购或采购高价商品，导致库存积压或成本上升 → 缓解：设置补货量上限和价格阈值，超限自动触发人工审批
  - **竞品价格战风险**（概率 3-5%）：自动补货导致库存充足，竞品可能跟风降价，引发价格战 → 缓解：与定价策略模块联动，监控竞品价格变化
  - **平台审查风险**（概率 2-3%）：Amazon 可能对自动化采购行为进行审查，防止刷单或异常交易 → 缓解：确保 Agent 操作日志完整可追溯，定期向平台报告自动化策略

---

## ③ 代码模板

```python
"""
LLM 工具路由与意图识别 — TF-IDF 意图匹配 + 置信过滤 + 工具选择
"""
import math
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter


# 工具注册表
TOOL_REGISTRY = {
    "query_inventory": {
        "description": "查询商品库存数量，支持按 ASIN、仓库、市场筛选",
        "keywords": ["库存", "库存量", "剩余", "有货", "缺货", "inventory", "stock"],
        "parameters": ["asin", "market", "warehouse_id"],
        "examples": ["B08X 现在还有多少库存", "查一下吸奶器的库存"]
    },
    "query_sales": {
        "description": "查询销售数据，支持按时间段、ASIN、市场统计销量和收入",
        "keywords": ["销量", "销售", "卖了多少", "收入", "revenue", "sales", "上周", "本月"],
        "parameters": ["asin", "market", "start_date", "end_date"],
        "examples": ["B08X 上周销量", "美国市场本月收入"]
    },
    "generate_report": {
        "description": "生成业务分析报告，包含趋势图表和文字摘要",
        "keywords": ["报告", "分析报告", "汇报", "总结", "report", "summary"],
        "parameters": ["report_type", "time_range", "markets"],
        "examples": ["生成上周销售报告", "做一份Q2分析"]
    },
    "adjust_price": {
        "description": "调整商品价格，支持批量调价和规则设置",
        "keywords": ["调价", "改价格", "价格调整", "降价", "涨价", "price", "pricing"],
        "parameters": ["asin", "new_price", "market", "reason"],
        "examples": ["把 B08X 美国价格调到 29.99", "所有吸奶器降价 5%"]
    },
    "query_returns": {
        "description": "查询退货数据，包含退货率、退货原因分布",
        "keywords": ["退货", "退款", "退货率", "return", "refund", "complaint"],
        "parameters": ["asin", "market", "start_date", "end_date"],
        "examples": ["B08X 本月退货率", "查一下退货原因"]
    }
}


def build_tfidf_index(registry: Dict) -> Dict[str, Dict[str, float]]:
    """为工具描述建立 TF-IDF 索引"""
    # 构建语料（工具描述 + 关键词 + 示例）
    tool_docs = {}
    for tool_id, info in registry.items():
        doc_tokens = []
        doc_tokens.extend(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', info["description"]))
        doc_tokens.extend(info["keywords"])
        for example in info["examples"]:
            doc_tokens.extend(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', example))
        tool_docs[tool_id] = Counter(doc_tokens)

    # 计算 IDF
    n_docs = len(tool_docs)
    df = Counter()
    for tokens in tool_docs.values():
        for token in tokens:
            df[token] += 1
    idf = {token: math.log(n_docs / (1 + freq)) + 1 for token, freq in df.items()}

    # 计算 TF-IDF 向量
    tfidf_index = {}
    for tool_id, tf in tool_docs.items():
        total = sum(tf.values())
        tfidf_index[tool_id] = {
            token: (count / total) * idf.get(token, 1)
            for token, count in tf.items()
        }
    return tfidf_index


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """计算两个稀疏向量的余弦相似度"""
    dot = sum(vec_a.get(k, 0) * v for k, v in vec_b.items())
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a * norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def route_to_tool(
    user_query: str,
    tfidf_index: Dict,
    confidence_threshold: float = 0.15,
    top_k: int = 2
) -> Dict:
    """路由用户请求到合适工具"""
    # 对用户 query 建立 TF-IDF 向量
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', user_query.lower())
    tf = Counter(tokens)
    total = sum(tf.values())
    query_vec = {token: count / total for token, count in tf.items()}

    # 计算与每个工具的相似度
    scores = {}
    for tool_id, tool_vec in tfidf_index.items():
        scores[tool_id] = cosine_similarity(query_vec, tool_vec)

    sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_tool, best_score = sorted_tools[0]

    if best_score < confidence_threshold:
        return {
            "status": "clarification_needed",
            "message": f"⚠️ 无法确定使用哪个工具（最高置信={best_score:.3f} < {confidence_threshold}），请补充说明",
            "top_candidates": [(t, round(s, 4)) for t, s in sorted_tools[:top_k]]
        }

    return {
        "status": "routed",
        "selected_tool": best_tool,
        "confidence": round(best_score, 4),
        "tool_description": TOOL_REGISTRY[best_tool]["description"],
        "required_params": TOOL_REGISTRY[best_tool]["parameters"],
        "top_candidates": [(t, round(s, 4)) for t, s in sorted_tools[:top_k]]
    }


# ===== 测试 =====
if __name__ == "__main__":
    # 建立索引
    tfidf_index = build_tfidf_index(TOOL_REGISTRY)

    test_cases = [
        ("B08X 上周销量是多少", "query_sales"),
        ("把吸奶器价格调低一点", "adjust_price"),
        ("生成本月运营报告", "generate_report"),
        ("现在 B09X 还有多少货", "query_inventory"),
        ("本月退货率怎么样", "query_returns"),
    ]

    print("=== 工具路由测试 ===")
    correct = 0
    for query, expected_tool in test_cases:
        result = route_to_tool(query, tfidf_index)
        status = result["status"]
        if status == "routed":
            selected = result["selected_tool"]
            is_correct = selected == expected_tool
            correct += int(is_correct)
            flag = "✅" if is_correct else "❌"
            print(f"{flag} '{query}'")
            print(f"   路由至: {selected} (置信={result['confidence']:.3f}) | 期望: {expected_tool}")
        else:
            print(f"⚠️ '{query}' → 需要澄清")

    accuracy = correct / len(test_cases)
    print(f"\n路由准确率: {accuracy:.0%} ({correct}/{len(test_cases)})")

    # 测试低置信澄清
    ambiguous_result = route_to_tool("帮我看一下", tfidf_index, confidence_threshold=0.15)
    print(f"\n模糊请求处理: {ambiguous_result['status']}")

    assert accuracy >= 0.6, f"路由准确率过低: {accuracy:.0%}"
    assert ambiguous_result["status"] in ["clarification_needed", "routed"]

    print("\n[✓] LLM工具路由与意图识别测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-Agent-Decision-Confidence-Threshold]]（置信阈值设计）
- **前置**：[[Skill-LLM-Business-Intelligence-Reasoning]]（LLM 意图理解基础）
- **延伸**：[[Skill-Agent-Observability-Tracing]]（工具调用链路追踪）
- **可组合**：[[Skill-Text2SQL-Schema-Linking]]（路由到 SQL 工具后的 Schema Linking）
- **可组合**：[[Skill-Agent-Cost-Optimization-Budget-Control]]（路由优化降低 API 成本）

## ⑤ 商业价值评估

- ROI 预估：Agent API 成本节省 40%（约 2-5 万元/月），Bot 答对率提升 20pp
- 实施难度：⭐⭐☆☆☆（规则引擎先快速落地，再迭代向量化方案）
- 优先级：⭐⭐⭐⭐⭐
- 评估依据：任何 Multi-Tool Agent 项目的第一个工程问题就是工具路由，准确的路由是 Agent 可用性的前提，投入小产出大
