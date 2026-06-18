---
title: 自主供应链规划智能体SCPA — 自然语言→结构化规划报告的JD.com完整框架
doc_type: knowledge
module: 16-智能体工程
topic: scpa-autonomous-sc-planning-llm-agent
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 自主供应链规划智能体SCPA

> **来源**：arXiv:2509.03811（Leveraging LLM-Based Agents for Intelligent Supply Chain Planning，JD.com 生产部署 2025）+ arXiv:2602.05524（AI Agent Systems: Structured Decision Prompts and Memory Retrieval）
> **桥梁**：智能体工程 ↔ 供应链运营 ↔ Palantir AIP Copilot Layer | **类型**：LLM智能体+NL-to-Action

## ① 算法原理

**SCPA 架构三核心**（JD.com 2025，已在亿级订单生产环境验证）：

**1. 意图理解层（Intent Classifier）**：

```
用户输入："下周三到的货够卖到黑五吗？如果不够，差多少？"
    ↓ 意图分类（fine-tuned LLM）
意图: inventory_coverage_analysis
参数提取: {sku: null, time_horizon: "黑五前", check_type: "coverage+gap"}
    ↓ 路由到对应工具链
```

**2. 任务编排层（Task Orchestrator）**：

```python
# JD.com 框架的任务分解模式
TASK_REGISTRY = {
    "inventory_coverage_analysis": [
        "fetch_current_inventory",      # Tool 1: 查当前库存
        "fetch_in_transit",             # Tool 2: 查在途货物
        "compute_demand_forecast",      # Tool 3: 调用预测模型
        "compute_coverage_days",        # Tool 4: 计算覆盖天数
        "generate_gap_analysis",        # Tool 5: 生成差距报告
    ],
    "replenishment_recommendation": [
        "compute_demand_forecast",
        "fetch_supplier_constraints",
        "run_multiagent_consensus",     # → 触发 MultiAgent Consensus Skill
        "generate_po_draft",
    ],
    "scenario_planning": [
        "fetch_historical_data",
        "run_whatif_simulation",        # → 触发 WhatIf Engine Skill
        "compare_scenarios",
        "generate_recommendation_report",
    ]
}
```

**3. 记忆检索层（Memory Retrieval）**：

```
短期记忆（Session Context）：当前对话上下文，token budget 管理
长期记忆（Historical RAG）：相似历史决策案例 → 参考"上次同期备货多少/结果如何"
业务规则记忆（Rule Base）：固定策略（如：旺季安全库存×1.5）
```

**Palantir AIP 类比**：
- 意图理解 → AIP Logic（解析 Action 意图）
- 任务编排 → AIP Orchestration（Function 链式调用）
- 记忆检索 → Object Store 查询（获取历史 Action 记录）
- 最终报告 → Action 输出（写回到 Ontology Objects）

## ② 母婴出海应用案例

**场景A：自然语言查询供应链现状**

运营主管早会前在企业微信发消息："帮我看看吸奶器这个月库存健康状况，主要关注断货风险和呆滞库存"

SCPA 在 2 分钟内生成结构化报告：
- 当前健康库存：3 个 SKU（DOS 30-60 天）✅
- 断货风险：2 个 SKU（DOS < 14 天）⚠️ 建议立即补货
- 呆滞库存：1 个 SKU（DOS > 90 天）→ 建议启动清仓
- 附：自动生成的补货草案（可一键确认发 PO）

**数据要求**：ERP 库存数据 API + 销售预测服务 + 供应商档案数据库
**预期产出**：结构化 Markdown 报告 + 自动生成的行动建议 + 草案 PO
**业务价值**：日常供应链晨报从 30 分钟人工准备 → 2 分钟自动生成，运营效率提升 90%

**场景B：多步骤供应链规划任务**

采购主管提问："如果我们要做双十一3倍于平时的备货，最晚什么时候需要下单？每个供应商分别需要提前多少天？有没有哪个供应商可能成为瓶颈？"

SCPA 自动拆解为 6 步任务：预测双十一需求 → 计算各 SKU 备货量 → 查询各供应商 PLT → 计算最晚下单日期 → 识别产能瓶颈 → 生成甘特图式采购计划。

**数据要求**：历史大促数据、供应商 PLT/产能数据、当前库存
**预期产出**：各供应商最晚下单日期表 + 产能瓶颈预警 + 采购甘特图
**业务价值**：双十一备货规划从 3 天 → 1 小时，供应商瓶颈提前 60 天识别

## ③ 代码模板

```python
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

class IntentType(Enum):
    INVENTORY_STATUS = "inventory_status"
    COVERAGE_ANALYSIS = "coverage_analysis"
    REPLENISHMENT_PLAN = "replenishment_plan"
    SCENARIO_PLANNING = "scenario_planning"
    SUPPLIER_RISK = "supplier_risk"
    UNKNOWN = "unknown"

@dataclass
class SCPAMemory:
    """SCPA 三层记忆"""
    session_context: List[Dict] = field(default_factory=list)    # 短期：对话历史
    historical_decisions: List[Dict] = field(default_factory=list)  # 长期：历史决策
    business_rules: Dict[str, Any] = field(default_factory=lambda: {
        "peak_season_safety_multiplier": 1.5,
        "min_dos_trigger_replenishment": 21,
        "max_dos_trigger_clearance": 90,
        "urgent_replenishment_threshold_days": 14,
    })
    
    def add_session_message(self, role: str, content: str):
        self.session_context.append({"role": role, "content": content})
        # 保持最近20条（token budget）
        if len(self.session_context) > 20:
            self.session_context = self.session_context[-20:]
    
    def retrieve_similar_decisions(self, intent: IntentType, k: int = 3) -> List[Dict]:
        """RAG 检索相似历史决策"""
        return [d for d in self.historical_decisions 
                if d.get("intent") == intent.value][-k:]

class SCPAAgent:
    """
    自主供应链规划智能体（SCPA）
    
    实现 JD.com 三层架构：
    意图理解 → 任务编排 → 工具执行 → 报告生成
    """
    
    # 意图-任务映射表（JD.com Task Registry 模式）
    TASK_REGISTRY: Dict[IntentType, List[str]] = {
        IntentType.INVENTORY_STATUS: [
            "fetch_inventory_snapshot",
            "classify_health_status",
            "generate_status_report",
        ],
        IntentType.COVERAGE_ANALYSIS: [
            "fetch_inventory_snapshot",
            "fetch_in_transit",
            "compute_demand_forecast",
            "compute_coverage_days",
            "identify_risks",
            "generate_coverage_report",
        ],
        IntentType.REPLENISHMENT_PLAN: [
            "fetch_inventory_snapshot",
            "compute_demand_forecast",
            "fetch_supplier_constraints",
            "compute_replenishment_qty",
            "generate_po_draft",
        ],
        IntentType.SCENARIO_PLANNING: [
            "fetch_historical_data",
            "run_scenario_simulation",
            "compare_scenarios",
            "generate_scenario_report",
        ],
        IntentType.SUPPLIER_RISK: [
            "fetch_supplier_profiles",
            "compute_risk_scores",
            "identify_bottlenecks",
            "generate_risk_report",
        ],
    }
    
    def __init__(self, tools: Dict[str, Callable], llm_func=None):
        """
        Args:
            tools: 工具函数字典 {"tool_name": callable}
            llm_func: LLM调用函数
        """
        self.tools = tools
        self.llm = llm_func or self._mock_llm
        self.memory = SCPAMemory()
    
    def _mock_llm(self, prompt: str, mode: str = "chat") -> str:
        """Mock LLM（替换为真实 API）"""
        if mode == "intent":
            if "库存" in prompt or "inventory" in prompt.lower():
                if "覆盖" in prompt or "够卖" in prompt or "coverage" in prompt:
                    return json.dumps({"intent": "coverage_analysis", 
                                       "params": {"time_horizon": "30天", "focus": "all_skus"}})
                return json.dumps({"intent": "inventory_status", 
                                   "params": {"focus": "all"}})
            elif "补货" in prompt or "replenishment" in prompt.lower():
                return json.dumps({"intent": "replenishment_plan",
                                   "params": {"urgency": "normal"}})
            elif "情景" in prompt or "scenario" in prompt.lower() or "如果" in prompt:
                return json.dumps({"intent": "scenario_planning",
                                   "params": {}})
            return json.dumps({"intent": "unknown", "params": {}})
        else:
            return "根据分析，您的供应链当前有3个SKU面临断货风险，建议优先处理STERILIZER-PRO的补货。"
    
    def _parse_intent(self, user_query: str) -> tuple:
        """意图识别（Step 1）"""
        prompt = f"intent: {user_query}"
        response = self.llm(prompt, mode="intent")
        try:
            parsed = json.loads(response)
            intent_str = parsed.get("intent", "unknown")
            try:
                intent = IntentType(intent_str)
            except ValueError:
                intent = IntentType.UNKNOWN
            params = parsed.get("params", {})
        except json.JSONDecodeError:
            intent = IntentType.UNKNOWN
            params = {}
        return intent, params
    
    def _execute_task_chain(self, intent: IntentType,
                             params: Dict) -> Dict[str, Any]:
        """任务编排执行（Step 2）"""
        tasks = self.TASK_REGISTRY.get(intent, ["generate_fallback_response"])
        execution_results = {}
        
        for task_name in tasks:
            tool = self.tools.get(task_name)
            if tool is None:
                # 工具不存在，使用 LLM 模拟
                execution_results[task_name] = {"status": "simulated", "data": {}}
                continue
            try:
                result = tool(params, execution_results)
                execution_results[task_name] = {"status": "success", "data": result}
            except Exception as e:
                execution_results[task_name] = {"status": "error", "error": str(e)}
        
        return execution_results
    
    def _retrieve_memory_context(self, intent: IntentType) -> str:
        """记忆检索（Step 3）"""
        similar = self.memory.retrieve_similar_decisions(intent, k=2)
        rules = self.memory.business_rules
        
        context_parts = [f"业务规则: 旺季安全库存×{rules['peak_season_safety_multiplier']}, "
                        f"触发补货DOS<{rules['min_dos_trigger_replenishment']}天, "
                        f"触发清仓DOS>{rules['max_dos_trigger_clearance']}天"]
        
        if similar:
            context_parts.append(f"历史相似决策({len(similar)}条): " + 
                                 "; ".join([f"{d.get('summary','')}" for d in similar]))
        
        return "\n".join(context_parts)
    
    def _generate_report(self, intent: IntentType,
                          user_query: str,
                          execution_results: Dict,
                          memory_context: str) -> str:
        """报告生成（Step 4）"""
        # 汇总执行结果
        successful_results = {k: v["data"] for k, v in execution_results.items() 
                              if v.get("status") == "success"}
        
        prompt = f"""基于以下供应链分析结果，生成专业简洁的中文报告回答用户问题。

用户问题: {user_query}
分析结果: {json.dumps(successful_results, ensure_ascii=False, indent=2)[:2000]}
业务背景: {memory_context}

要求：
1. 直接回答问题，不要废话
2. 用表格/列表展示多条数据
3. 量化数字（件数/天数/金额）
4. 末尾附行动建议（最多3条）"""
        
        return self.llm(prompt, mode="report")
    
    def chat(self, user_query: str) -> Dict:
        """
        主入口：处理用户自然语言查询
        
        Returns:
            dict: {intent, report, execution_summary, action_items}
        """
        # 保存用户消息
        self.memory.add_session_message("user", user_query)
        
        # Step 1: 意图识别
        intent, params = self._parse_intent(user_query)
        
        # Step 2: 记忆检索
        memory_ctx = self._retrieve_memory_context(intent)
        
        # Step 3: 任务编排执行
        execution_results = self._execute_task_chain(intent, params)
        
        # Step 4: 报告生成
        report = self._generate_report(intent, user_query, execution_results, memory_ctx)
        
        # 保存 AI 回复
        self.memory.add_session_message("assistant", report)
        
        # 记录决策历史
        self.memory.historical_decisions.append({
            "intent": intent.value,
            "query": user_query[:100],
            "summary": report[:200],
            "tasks_executed": list(execution_results.keys())
        })
        
        return {
            "intent": intent.value,
            "report": report,
            "tasks_executed": len([v for v in execution_results.values() 
                                   if v.get("status") == "success"]),
            "memory_context_used": len(memory_ctx) > 0
        }


# ===== 测试用例 =====
def run_test():
    # Mock 工具函数（生产中替换为真实API调用）
    def mock_fetch_inventory(params, prev_results):
        return {
            "skus": [
                {"sku": "STERILIZER-PRO", "stock": 120, "dos": 12, "status": "risk"},
                {"sku": "BOTTLE-SET-4PC", "stock": 450, "dos": 35, "status": "healthy"},
                {"sku": "BABY-FOOD-ORG", "stock": 980, "dos": 95, "status": "overstock"},
            ],
            "total_skus": 3
        }
    
    def mock_compute_forecast(params, prev_results):
        return {"daily_demand": {"STERILIZER-PRO": 10, "BOTTLE-SET-4PC": 13, "BABY-FOOD-ORG": 10}}
    
    def mock_classify_health(params, prev_results):
        inv = prev_results.get("fetch_inventory_snapshot", {}).get("data", {})
        return {"risk_skus": 1, "healthy_skus": 1, "overstock_skus": 1}
    
    def mock_generate_report(params, prev_results):
        return {"report_type": "inventory_status", "generated": True}
    
    tools = {
        "fetch_inventory_snapshot": mock_fetch_inventory,
        "compute_demand_forecast": mock_compute_forecast,
        "classify_health_status": mock_classify_health,
        "generate_status_report": mock_generate_report,
    }
    
    agent = SCPAAgent(tools=tools)
    
    # Test 1: 库存状态查询
    result1 = agent.chat("帮我看看当前库存健康状况")
    assert result1["intent"] == "inventory_status", f"意图识别错误: {result1['intent']}"
    assert result1["tasks_executed"] > 0, "应执行至少一个任务"
    print(f"  Test1: 意图={result1['intent']}, 执行任务={result1['tasks_executed']}个")
    
    # Test 2: 覆盖分析查询
    result2 = agent.chat("库存还能覆盖多少天？有没有断货风险？")
    assert result2["intent"] == "coverage_analysis", f"意图识别错误: {result2['intent']}"
    print(f"  Test2: 意图={result2['intent']}, 报告长度={len(result2['report'])}字")
    
    # Test 3: 记忆功能验证
    assert len(agent.memory.session_context) == 4, f"应有4条消息(2轮×2)，实际{len(agent.memory.session_context)}"
    assert len(agent.memory.historical_decisions) == 2, "应记录2条历史决策"
    print(f"  Test3: 记忆={len(agent.memory.session_context)}条, 历史决策={len(agent.memory.historical_decisions)}条")
    
    # Test 4: 情景规划意图
    result4 = agent.chat("如果双十一需求是平时3倍，我们需要提前多久备货？")
    assert result4["intent"] == "scenario_planning", f"情景规划意图识别失败: {result4['intent']}"
    print(f"  Test4: 情景规划意图识别 ✓")
    
    print("\n[✓] SCPA-Autonomous-Planning-Agent 测试通过 — 意图识别+任务编排+记忆检索就绪")

run_test()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supply-Chain-KPI-Health-Dashboard]] — 库存健康数据是 SCPA 的核心信息源
- **前置（prerequisite）**：[[Skill-Demand-Forecasting-Supply-Chain]] — SCPA 需要预测服务作为工具
- **延伸（extends）**：[[Skill-LLM-SC-MultiAgent-Consensus-Replenishment]] — SCPA 补货任务链调用多智能体共识
- **延伸（extends）**：[[Skill-SC-WhatIf-Scenario-Analysis-Engine]] — SCPA 情景规划任务链调用 WhatIf 引擎
- **可组合（combinable）**：[[Skill-Supply-Chain-Ontology-Action-Trigger]] — SCPA 生成的决策通过 Action Type 写回 ERP
- **可组合（combinable）**：[[Skill-RAG-Enhanced-Data-Analysis]] — SCPA 长期记忆层基于 RAG 实现历史案例检索

## ⑤ 商业价值评估

- **ROI 预估**：日常供应链报告从 30 分钟手工 → 2 分钟自动（↓93%），双十一备货规划从 3 天 → 1 小时（↓87%），运营人效提升约 2-3 倍
- **实施难度**：⭐⭐⭐☆☆（核心是工具函数接入 ERP API，LLM 框架相对标准）
- **优先级**：⭐⭐⭐⭐⭐（Palantir AIP Copilot 的核心场景，JD.com 生产验证，直接可迁移）
- **企业AI知识库依赖**：高 — 工具函数需接入 ERP/WMS API；长期记忆需要历史决策数据库
