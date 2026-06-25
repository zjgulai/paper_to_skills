---
name: phase3-commercial-platform
description: paper2skills Phase 3 商业化平台计划（3-6 月）。4 个战略目标：MAS 与前端打通（三层资产闭环）、数据源集成（飞书多维表格 + CSV）、对外 REST API（B2B 收费入口）、Freemium 付费墙（真实变现）。完成后商业化就绪度从 ~6/10 提升到 ~8.5/10。当被要求执行「Phase 3」或「商业化」时使用。
---

# paper2skills Phase 3 — 商业化平台计划（3-6 月）

**目标**：三层资产（Skill 内容 / MAS 框架 / 前端 Agent）形成闭环，建立可收费的产品。  
**前置条件**：Phase 1 + Phase 2 全部完成  
**预期商业化就绪度**：~6/10 → ~8.5/10

---

## S1：MAS 框架与前端 Agent 打通

**问题**：MAS 框架（5 工作流 / 112 工具 / 61 测试）和前端 21 个 Agent 是两套平行系统，互不调用。MAS 的多智能体编排能力完全未被前端用户感知。

**目标**：前端 Agent 调用 MAS 编排，而不是直接调 DeepSeek 一问一答。

### S1.1 MAS HTTP API 层（p2s-service）

在 FastAPI 服务新增 MAS 触发端点：

```python
# p2s-service/app.py 新增
POST /api/mas/run
Request: {
  "workflow_type": "restock" | "ad_campaign" | "customer_ops" | "product_selection" | "review_monitor",
  "payload": {},       # 工作流输入
  "operator_id": str,  # 用户 ID（来自飞书 OAuth）
  "session_key": str   # 匿名标识（未登录时）
}
Response: {
  "run_id": "uuid",
  "status": "pending" | "running" | "hitl_required" | "completed" | "failed",
  "result": {},
  "hitl_actions": []   # 需要人工审批的操作列表
}

GET /api/mas/run/{run_id}    # 查询执行状态
POST /api/mas/approve/{run_id}  # HITL 审批
POST /api/mas/reject/{run_id}   # HITL 拒绝
```

```python
# 服务端实现
from mas.main import MAS
mas_instance = MAS()

@app.post("/api/mas/run")
async def run_mas(body: MASRunRequest):
    run_id = str(uuid.uuid4())
    # 异步执行（避免 HTTP timeout）
    asyncio.create_task(execute_mas_workflow(run_id, body))
    return {"run_id": run_id, "status": "pending"}
```

### S1.2 前端 Agent 改造

**映射关系**（前端 Agent ID → MAS workflow_type）：

| 前端 Agent | MAS 工作流 | 触发条件 |
|---|---|---|
| agent-supply-sentinel | `restock` | 有库存数据输入 |
| agent-festival-replenishment | `large_restock` | 大促补货场景 |
| agent-ad-attribution | `ad_campaign` | 广告分析输入 |
| agent-voc-decoder | `customer_ops` | 评论数据输入 |
| agent-product-radar | `product_selection` | 选品分析输入 |
| 其余 16 个 | 直接 DeepSeek | 无对应工作流（暂不改） |

**改造方式**：Agent 运行前判断是否有对应 MAS 工作流，有则走 `/api/mas/run`，无则降级到现有 DeepSeek 直调。

### S1.3 HITL 飞书通知升级

MAS 的 HITL 审批目前是 `approval_api.py` 内存存储。接入飞书：
- 需要审批时：`POST /api/feishu-callback` 推送包含 `run_id` 的审批卡片
- 飞书卡片按钮：`[确认执行]` → `POST /api/mas/approve/{run_id}`
- 结果完成时：推送结果卡片（已有 `pushToFeishu` 机制，复用即可）

**验收**：
```bash
# 调用供应链哨兵 Agent → 触发 MAS restock 工作流
curl -X POST http://localhost:8765/api/mas/run \
  -d '{"workflow_type":"restock","payload":{"sku":"BABY-001","current_stock":150},"session_key":"test"}'
# 返回 run_id
# 飞书群收到 HITL 审批卡片
```

---

## S2：数据源集成

**问题**：所有 Agent 输入靠手工粘贴文本，这是真实使用的最大摩擦点。

**优先级**：飞书多维表格（最快路径）→ CSV 上传（最广覆盖）→ Amazon SP-API（最高价值）

### S2.1 飞书多维表格 Webhook 自动触发（最快路径，2 周）

**场景**：用户在飞书多维表格更新库存数据 → 自动触发供应链哨兵分析 → 结果写回表格

```python
# p2s-service/app.py 新增飞书事件处理
@app.post("/api/feishu-event")
async def handle_feishu_event(request: Request):
    body = await request.json()
    
    # 多维表格记录变更事件
    if body.get("header", {}).get("event_type") == "bitable.record.updated":
        record = body["event"]["record"]
        table_id = body["event"]["table_id"]
        
        # 根据表格 ID 路由到对应 Agent/MAS 工作流
        if table_id in WATCHED_TABLES:
            workflow = WATCHED_TABLES[table_id]["workflow"]
            payload = extract_payload(record, WATCHED_TABLES[table_id]["field_map"])
            await run_mas_workflow(workflow, payload)
    
    return {"code": 0}

# 配置：表格 ID → 工作流映射
WATCHED_TABLES = {
    "tbl_inventory_sku": {
        "workflow": "restock",
        "field_map": {"SKU": "sku", "当前库存": "current_stock", "日均销量": "daily_sales"}
    }
}
```

**用户配置步骤**：
1. 飞书多维表格创建自动化规则：「当记录更新」→「发送 Webhook」→ `https://skills.lute-tlz-dddd.top/api/feishu-event`
2. 在 `diagnostic.html` 或独立「集成配置」页面，用户配置表格 ID 和字段映射

### S2.2 CSV 上传（覆盖最广，1 周）

在 `agents.html` 的每个 Agent 输入区增加「上传 CSV」按钮：

```javascript
// 前端解析 CSV，转为 Agent 输入格式
async function handleCSVUpload(agentId, file) {
  const text = await file.text();
  const rows = parseCSV(text);
  // 根据 agentId 的字段定义自动映射列名
  const mapped = mapCSVToAgentInput(agentId, rows);
  document.getElementById(`${agentId}-input`).value = JSON.stringify(mapped, null, 2);
}
```

CSV 字段映射规则存在 `config/agents_data.py` 的 `csv_field_map` 字段（新增）。

### S2.3 Amazon SP-API（高价值，6-8 周）

**范围**：只接入只读端点，降低审核难度

- `GET /catalog/items`：获取 Listing 信息（供 Listing Doctor Agent）
- `GET /reports`：获取销售报告（供 P&L 分析 Agent）
- `GET /fba/inventory`：获取库存（供供应链哨兵 Agent）

**认证流程**：
1. 用户在「集成配置」页面输入 SP-API 凭据（Marketplace ID + Refresh Token）
2. p2s-service 加密存储（AES-256），用时解密调用
3. 定时刷新 Access Token（SP-API 每小时过期）

**验收**：
```bash
# 测试 CSV 上传
# 上传包含 SKU/库存/销量列的 CSV → Agent 输入框自动填充
# 测试飞书 Webhook
# 多维表格更新 → p2s-service 日志显示接收到事件 → MAS 工作流触发
```

---

## S3：对外 REST API

**问题**：无公开 API = 企业客户无法集成 = B2B SaaS 最核心收费入口缺失。

**目标**：发布 v1 API，支持企业客户把 paper2skills 的能力嵌入自己的系统。

### S3.1 API 设计

```
Base URL: https://skills.lute-tlz-dddd.top/api/v1

# Skill 查询
GET  /v1/skills                          # 列表（支持 domain/keyword 过滤）
GET  /v1/skills/{skill_id}               # 单个 Skill 完整内容
GET  /v1/skills/search?q=定价&domain=17  # 搜索

# Agent 调用
POST /v1/agent/run                       # 运行 Agent（同步，max 30s）
GET  /v1/agent/run/{run_id}              # 异步查询状态

# 图谱
GET  /v1/graph/neighbors/{skill_id}      # 获取关联 Skill
GET  /v1/graph/path?from=X&to=Y          # Skill 间最短路径

# 诊断
POST /v1/diagnose                        # 症状 → Skill 链推荐
```

### S3.2 API 认证

API Key 认证（最简单，B2B 标准）：
```
Authorization: Bearer p2s_sk_xxxxxxxxxxxxxxxx
```

```python
# p2s-service/app.py
from functools import wraps

def require_api_key(f):
    @wraps(f)
    async def wrapper(request: Request, *args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer p2s_sk_"):
            raise HTTPException(401, "Invalid API key")
        key = auth.removeprefix("Bearer ")
        user = await get_user_by_api_key(key)  # 从 SQLite 查
        if not user:
            raise HTTPException(401, "API key not found")
        request.state.user = user
        return await f(request, *args, **kwargs)
    return wrapper
```

### S3.3 API Key 管理页面

在 `agents.html` 或新页面 `/settings.html` 添加：
- 「生成 API Key」按钮（需登录）
- 显示已有 Key（只显示前 8 位）
- 使用量统计（本月调用次数）

### S3.4 速率限制

```python
# 按 API Key 限流
RATE_LIMITS = {
    "free": {"per_min": 10, "per_day": 100},
    "pro": {"per_min": 60, "per_day": 2000},
    "enterprise": {"per_min": 300, "per_day": 50000}
}
```

### S3.5 API 文档

生成 OpenAPI 文档（FastAPI 自带），发布到 `/api/v1/docs`。写 Quickstart Guide（paper2skills-services/API.md）。

**验收**：
```bash
# 生成 API Key
curl -X POST https://skills.lute-tlz-dddd.top/api/v1/auth/api-key \
  -H "Cookie: session=xxx"
# {"key": "p2s_sk_xxxxxx", "created_at": "..."}

# 调用 Skill 搜索
curl https://skills.lute-tlz-dddd.top/api/v1/skills/search?q=动态定价 \
  -H "Authorization: Bearer p2s_sk_xxxxxx"
# 返回相关 Skill 列表
```

---

## S4：Freemium 付费墙

**问题**：所有功能完全免费，AI API 成本由运营者承担，没有任何收入来源。

**策略**：Freemium 分层，免费用户可感知价值，付费用户解锁效率工具。

### S4.1 分层定义

| 层级 | 价格 | 权限 |
|------|------|------|
| **Free** | ¥0 | 浏览所有 Skill（只读）/ Agent 每月 10 次 / Chat 每月 20 条 |
| **Pro** | ¥299/月 | 无限 Agent / Chat / API 2000次/天 / 数据源集成 / 报告导出 |
| **Enterprise** | 定价 | 私有部署 / 无限 API / 飞书深度集成 / 专属客服 |

### S4.2 实现方式（最小改动）

**不做强制登录**（降低获客摩擦），改为：
- 匿名用户：使用量追踪用 `session_key`（浏览器指纹），超限后提示注册
- 登录用户：追踪用 `feishu_id`，升级 Pro 后重置配额

**使用量追踪**（p2s-service）：
```python
@app.post("/api/agent")
async def run_agent(request: Request, body: AgentRequest):
    session_key = body.session_key
    user = get_or_create_session(session_key)  # 匿名或登录用户
    
    if user.tier == "free" and user.monthly_agent_calls >= 10:
        raise HTTPException(429, {
            "error": "monthly_limit_reached",
            "message": "免费版每月 10 次 Agent 调用已用完",
            "upgrade_url": "/pricing.html"
        })
    
    # 正常处理...
    increment_usage(user)
```

**前端提示**（agents.html）：
```javascript
if (error.status === 429) {
  showUpgradeModal({
    title: '本月免费次数已用完',
    desc: '升级 Pro 版解锁无限 Agent 调用',
    ctaUrl: '/pricing.html'
  });
}
```

### S4.3 支付集成

**优先支付渠道**：微信支付（目标用户是中国跨境卖家）

```python
# p2s-service/app.py
POST /api/payment/create   # 创建支付订单 → 返回微信支付二维码
GET  /api/payment/notify   # 微信支付回调 → 升级用户 tier
GET  /api/payment/status   # 查询支付状态
```

升级成功后：
1. `users.tier = 'pro'`
2. 飞书推送开通确认消息
3. 用户获得 API Key（自动生成）

### S4.4 定价页面（/pricing.html）

在 `build_playbook.py` 新增 `render_pricing_page()` 函数，生成定价对比页：
- 三列对比（Free / Pro / Enterprise）
- 扫码支付二维码（Pro）
- 企业版联系方式（Enterprise）

**验收**：
```bash
# 验证使用量限制
for i in {1..11}; do
  curl -X POST https://skills.lute-tlz-dddd.top/api/agent \
    -d '{"session_key":"test-limit","agent_id":"agent-pricing-advisor","inputs":{}}'
done
# 第 11 次返回 429，含 upgrade_url
```

---

## 执行顺序与里程碑

```
Month 1 (Week 1-4):   S1（MAS 打通）
Month 2 (Week 5-8):   S2（数据源集成：飞书 + CSV）
Month 3 (Week 9-12):  S3（REST API）
Month 4 (Week 13-16): S4（Freemium 付费墙）
Month 5 (Week 17-20): S2.3（Amazon SP-API）+ 压测
Month 6 (Week 21-24): 商业化 GTM（定价页 + 销售材料）
```

## 完成标志（商业化就绪）

- [ ] 供应链哨兵 Agent → 触发 MAS restock 工作流 → 飞书收到 HITL 审批卡片
- [ ] 上传库存 CSV → Agent 自动填充输入并分析
- [ ] `GET /api/v1/skills/search?q=动态定价` 返回结构化结果
- [ ] 使用 10 次后出现升级弹窗，支付后配额重置
- [ ] 整体 AI API 成本 < 月收入的 30%

---

## 附：三阶段商业化就绪度路线图

| 阶段 | 完成后就绪度 | 核心里程碑 |
|------|-------------|-----------|
| Phase 1（1-2周） | 3/10 → 4/10 | 数据准确 + 代码安全 + 服务入库 |
| Phase 2（1-3月） | 4/10 → 6/10 | 工程可维护 + 用户有身份 + CI/CD 自动化 + Chat 有真实 RAG |
| Phase 3（3-6月） | 6/10 → 8.5/10 | 三层打通 + 数据源接入 + API 收费 + Freemium 变现 |
