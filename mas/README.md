# MAS 部署与运维指南

> **最新更新**:2026-05-17 下午
> Skill 卡片库已达 **107** 个,Sprint 1+2 新增 10 个 P0 Skill 待集成到 MAS Agents

母婴跨境电商多智能体系统 (MAS) - 阶段 0~6 MVP 已就绪。

## 集成现状与下一步

- **MAS 基础设施**:5 工作流 + 14 模块 + 37/37 测试全绿(2026-05-17 第一轮)
- **Skill 库存**:107 个 Skill 卡片,18 个有对应代码模板
- **下一步**:将 Week 4-5 萃取的 19 个新 Skill(6h + Sprint1 + Sprint2)接入对应 Agent
  - SupplyChainAgent:Bass + Gen-QOT + HiFoReAd
  - MarketingAgent:PVM + Hierarchical-Search-Intent
  - CustomerServiceAgent:LACA + Dial-In LLM + AGRS + MAA + StaR
  - SelectionAgent:Hierarchical-Product-KG + CoLaKG

## 一、架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                  HITL 审批层 (mas/hitl/)                       │
│   approval_api.py · ApprovalStore · feishu_webhook_notifier   │
└──────────┬────────────────────────────────────────────────────┘
           │ interrupt_fn callback
┌──────────▼────────────────────────────────────────────────────┐
│              Orchestrator (mas/agents/orchestrator.py)         │
│   orchestrator_route · human_approval_gate                     │
│   execute_approved_action · handle_rejection                   │
└──┬──────────┬──────────┬───────────┬──────────────┬───────────┘
   │          │          │           │              │
┌──▼──┐ ┌─────▼────┐ ┌───▼─────┐ ┌───▼──────┐ ┌─────▼─────┐
│ Sup │ │Marketing │ │Customer │ │Selection │ │   QA      │
│Chain│ │  Agent   │ │ Service │ │  Agent   │ │  Agent    │
│Agent│ │          │ │  Agent  │ │          │ │           │
└─────┘ └──────────┘ └─────────┘ └──────────┘ └───────────┘
   │          │          │           │              │
┌──▼──────────▼──────────▼───────────▼──────────────▼────────┐
│         Skill 工具层 (mas/skills/)                          │
│  supply_chain_tools / marketing_tools /                     │
│  customer_service_tools / selection_tools / registry        │
└────────────────────────────────────────────────────────────┘
   │
┌──▼─────────────────────────────────────────────────────────┐
│      基础设施 (mas/checkpointing/ + mas/observability/)     │
│  SQLiteCheckpointer (生产可换 PostgresSaver)                │
│  TraceCollector (生产可换 LangSmith)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、5 个端到端工作流

| 工作流 | workflow_type | Agent | 主要 Skill 链 | 典型 HITL 触发条件 |
|---|---|---|---|---|
| WF-A 智能补货 | `restock`/`large_restock`/`price_change` | SupplyChainAgent | 需求预测 → 反事实 GCF → 安全库存 → DRL 补货 | 单 PO > ¥1万 |
| WF-B 广告优化 | `ad_campaign`/`ad_budget_increase`/`promotion_launch` | MarketingAgent | 搜索词清洗 → 否定词 → Uplift 提价 → MMM → DARA 跨渠道 | 单次预算变动 > ¥1万 |
| WF-C 客服分诊 | `customer_ops` | CustomerServiceAgent | 意图分类 → 情感分析 → GraphRAG FAQ → 决策树 | 大额退款 (>¥500) |
| WF-D 选品扫描 | `product_selection` | SelectionAgent | 市场空间 → 毛利 → 合规 → KGQA → Uplift | 自动通过 (cost=0) |
| WF-E Review 监控 | `review_monitor` | CustomerServiceAgent (复用) | Review 主题聚类 → 健康度评分 | 评分 < 4 触发行动 |

---

## 三、本地运行

### 安装

```bash
# 标准库即可,无第三方依赖 (生产部署再装 langgraph/fastapi/etc.)
git clone <repo>
cd paper_to_skills
python3 --version  # 需要 3.12+
```

### 跑全部 31 个集成测试

```bash
python3 -m mas.tests.test_phase0_skeleton     # 12 项 MAS 骨架
python3 -m mas.tests.test_wfa_restock_e2e      # 5 项 WF-A 补货
python3 -m mas.tests.test_wfb_ad_campaign_e2e  # 5 项 WF-B 广告
python3 -m mas.tests.test_wfce_customer_ops_e2e # 5 项 WF-C/E 客服
python3 -m mas.tests.test_wfd_selection_e2e    # 4 项 WF-D 选品
python3 -m mas.tests.test_phase6_integration   # 6 项 MAS 集成
```

### 入口 demo

```bash
python3 -m mas.main
```

### Python API 调用

```python
from mas.main import MAS
from mas.hitl.approval_api import make_blocking_interrupt

# 1. 同步审批模式 (适合自动化测试)
mas = MAS(interrupt_fn=lambda req: {"action": "approve", "note": "ok"})

# 2. 异步审批模式 (生产推荐,接飞书 webhook)
mas = MAS(interrupt_fn=make_blocking_interrupt())

result = mas.trigger(
    workflow_type="restock",
    operator_id="ops-shen",
    payload={
        "sku_id": "AB-FORMULA-S1",
        "history_daily_sales": [...],
        "current_stock": 800,
        "in_transit": 0,
        "lead_time_days": 35,
        "moq": 1000,
        "unit_cost_rmb": 80.0,
    },
    token_budget=20_000,
)

print(result["final_output"])
print(mas.trace_summary(result["workflow_id"]))
```

---

## 四、生产部署清单

### 必须替换的 stub

| 当前 stub | 生产替换 | 说明 |
|---|---|---|
| `mas/graphs/base_graph.py` 纯 Python StateGraph | `from langgraph.graph import StateGraph` | 接口已对齐,迁移仅需改 import |
| `mas/checkpointing/sqlite_saver.py` | `langgraph.checkpoint.postgres.PostgresSaver` | 改 db_path → POSTGRES_URL |
| `mas/observability/tracer.py` | LangSmith SDK | `langsmith.Client()` |
| `mas/hitl/approval_api.py` ApprovalStore | FastAPI + 飞书 OpenAPI | 配合 `interrupt()` Async |
| `mas/skills/registry.py` SkillTool stub | LangChain `@tool` 装饰器 / MCP Server | 阶段 2+ 切换 |
| `mas/agents/base.py` `_stub_llm` | `langchain_anthropic.ChatAnthropic("claude-opus-4-5")` 或同等 | LLM provider 替换 |

### 推荐 pip 依赖（生产）

```bash
pip install langgraph>=1.1.0 \
            langchain-anthropic \
            langsmith \
            psycopg2-binary \
            fastapi \
            uvicorn \
            requests
```

### 环境变量

```bash
export ANTHROPIC_API_KEY=...
export POSTGRES_URL=postgresql://user:pass@host:5432/mas
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=mas-baby-cross-border
export PAPER2SKILLS_FEISHU_WEBHOOK=https://open.feishu.cn/...
```

### 配置阈值

`mas/agents/orchestrator.py`:
- `HIGH_RISK_THRESHOLD = 10_000.0` 触发 HITL 的金额阈值（人民币）
- `RISKY_WORKFLOW_TYPES` 强制审批的工作流类型集合

`mas/state/schema.py`:
- `init_state()` 默认 `token_budget=50_000`

---

## 五、扩展路线

### 阶段 7：替换为真实 LangGraph 1.1+

1. 在 `requirements.txt` 添加 langgraph
2. 把 `mas/graphs/base_graph.py` 改为 `from langgraph.graph import StateGraph, START, END`
3. `interrupt()` 改用 `langgraph.types.interrupt`
4. 31 个集成测试零修改通过即视为迁移完成

### 阶段 8：MCP Server 化 Skill 工具

1. 在 `mas/skills/mcp_servers/` 下每个领域写一个 server
2. Agent 使用 `MultiServerMCPClient` 替换直接 import
3. 工具权限隔离 / 审计

### 阶段 9：多租户与成本控制

1. WorkflowContext 增加 `tenant_id`
2. SQLite → Postgres 多租户分库
3. tracer 增加 token / cost 看板

---

## 六、当前 MVP 状态

| 维度 | 状态 |
|---|---|
| **集成测试** | ✅ 37/37 全部通过 (含 6 项阶段 6 集成) |
| **5 个工作流** | ✅ 全部端到端可跑 |
| **HITL 双路径** | ✅ 同步/异步 callback 都验证 |
| **Checkpointing** | ✅ SQLite 持久化跨进程加载 |
| **Tracer** | ✅ token / 事件聚合 |
| **依赖** | ✅ 纯 Python 标准库 (生产可一键迁 LangGraph) |
| **总代码量** | ~2500 行,涵盖 14 个核心模块 |

### 已知限制

1. **LLM 仍是 stub** — 生产替换 `BaseAgent._stub_llm` 为真实 ChatAnthropic
2. **Skill 工具是规则版** — 当前是论文公式的简化实现,生产可接真实 Skill 卡片代码
3. **HITL 仅 callback 模式** — 异步审批仍需补 FastAPI + 飞书 OpenAPI
4. **没有并发隔离** — checkpointer SQLite 写入未做行锁,高并发需切 Postgres

---

## 七、文件清单 (14 个核心模块)

```
mas/
├── __init__.py
├── main.py                          # MAS 入口 + 5 工作流注册
├── state/
│   └── schema.py                    # WorkflowContext TypedDict
├── agents/
│   ├── base.py                      # BaseAgent 基类
│   ├── orchestrator.py              # 路由 + HITL gate
│   ├── supply_chain_agent.py        # WF-A
│   ├── marketing_agent.py           # WF-B
│   ├── customer_service_agent.py    # WF-C / WF-E
│   ├── selection_agent.py           # WF-D
│   └── qa_agent.py                  # 横切质检
├── skills/
│   ├── registry.py                  # SkillRegistry (90+ stub 工具)
│   ├── supply_chain_tools.py        # WF-A 业务工具
│   ├── marketing_tools.py           # WF-B 业务工具
│   ├── customer_service_tools.py    # WF-C/E 业务工具
│   └── selection_tools.py           # WF-D 业务工具
├── graphs/
│   ├── base_graph.py                # StateGraph + RetryPolicy
│   ├── restock_graph.py             # WF-A
│   ├── ad_campaign_graph.py         # WF-B
│   ├── customer_ops_graph.py        # WF-C / WF-E
│   └── selection_graph.py           # WF-D
├── hitl/
│   └── approval_api.py              # 审批 stub
├── checkpointing/
│   └── sqlite_saver.py              # SQLite 持久化
├── observability/
│   └── tracer.py                    # in-memory tracer
└── tests/
    ├── test_phase0_skeleton.py
    ├── test_wfa_restock_e2e.py
    ├── test_wfb_ad_campaign_e2e.py
    ├── test_wfce_customer_ops_e2e.py
    ├── test_wfd_selection_e2e.py
    └── test_phase6_integration.py
```
