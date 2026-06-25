"""
paper2skills 飞书回调服务
- POST /api/feishu-callback  飞书卡片按钮回调
- POST /api/feishu-event     飞书事件订阅（文档分享 → Skill 草稿）
- POST /api/daily-inspect    cron 触发每日巡检
- POST /api/reports          保存 Agent 运行报告
- GET  /api/reports          查询 Agent 运行报告（按 session_key）
- DELETE /api/reports/{id}   删除单条报告
"""
import os, json, requests
import sqlite3, uuid
from datetime import datetime, timezone
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from feishu_oauth import auth_callback as _auth_callback
from feishu_oauth import auth_login as _auth_login
from feishu_oauth import auth_logout as _auth_logout
from feishu_oauth import auth_me as _auth_me

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://skills.lute-tlz-dddd.top", "http://localhost:8765", "http://127.0.0.1:8765"],
    allow_origin_regex=r"http://localhost:.*",
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

DB_PATH = os.environ.get("P2S_DB_PATH", "/opt/paper2skills/service/reports.db")


def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_reports (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                inputs TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session
            ON agent_reports(session_key, created_at DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                feishu_id TEXT PRIMARY KEY,
                name TEXT,
                org TEXT,
                tier TEXT DEFAULT 'free',
                created_at TEXT,
                last_seen TEXT,
                usage_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()


_init_db()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
FEISHU_WEBHOOK   = os.environ.get("FEISHU_WEBHOOK_URL", "")
DS_URL           = "https://api.deepseek.com/chat/completions"

AGENT_PROMPTS = {
    "agent-supply-sentinel":   "你是供应链分析专家。根据数据给出风险评级、DOS分析、断货概率、补货建议、紧急行动。每项用【】标题，禁止模糊表达。",
    "agent-pricing-advisor":   "你是定价顾问。给出定价诊断、竞争价格带、最优定价区间、执行路径、ROI影响。每项用【】标题。",
    "agent-pnl-analyzer":      "你是P&L分析师。拆解利润瀑布、亏损根因TOP3、行业对标、提利优先级。每项用【】标题。",
    "agent-ad-attribution":    "你是广告归因分析师。给出健康评分、ROAS拆解、预算浪费、优化行动、重分配方案。每项用【】标题。",
    "agent-competitor-radar":  "你是竞品情报分析师。给出竞争格局、定价策略、差评规律、市场空白、切入建议。每项用【】标题。",
    "agent-listing-doctor":    "你是Listing优化专家。给出评分、Title诊断+重写、Bullet诊断、关键词缺口、预期效果。每项用【】标题。",
    "agent-voc-decoder":       "你是VOC分析师。给出情感分布、痛点TOP5、用户期待、竞品信号、迭代建议。每项用【】标题。",
    "agent-cs-triage":         "你是客服分诊专家。给出工单分类、高风险识别、根因TOP3、回复模板、预防建议。每项用【】标题。",
    "agent-account-guardian":  "你是账号风险专家。给出健康评分、违规风险点、封号概率、48h行动、长期方案。每项用【】标题。",
    "agent-brand-guardian":    "你是品牌合规专家。给出评分、违禁词清单、风险等级、合规改写、上架可行性。每项用【】标题。",
    "agent-product-radar":     "你是选品分析师。给出机会评分/GO-NOGO、市场规模、竞争强度、利润空间、首批建议。每项用【】标题。",
    "agent-festival-replenishment": "你是大促补货专家。给出需求预测、安全库存、资金评估、物流时间线、清仓预案。每项用【】标题。",
}

def call_deepseek(system_prompt: str, user_content: str, max_tokens: int = 1200) -> str:
    r = requests.post(DS_URL, json={
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": user_content}],
        "max_tokens": max_tokens, "temperature": 0.5,
    }, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"}, timeout=90)
    return r.json()["choices"][0]["message"]["content"].strip()

def push_feishu_text(title: str, content: str, template: str = "blue"):
    clean = content.replace("**","").replace("##","").strip()[:2000]
    requests.post(FEISHU_WEBHOOK, json={
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title}, "template": template},
            "elements": [{"tag": "div", "text": {"tag": "plain_text", "content": clean}}]
        }
    }, timeout=10)

def push_feishu_card_with_buttons(title: str, summary: str, action_value: dict, template: str = "blue"):
    requests.post(FEISHU_WEBHOOK, json={
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title}, "template": template},
            "elements": [
                {"tag": "div", "text": {"tag": "plain_text", "content": summary}},
                {"tag": "hr"},
                {"tag": "action", "actions": [
                    {"tag": "button", "text": {"tag": "plain_text", "content": "✅ 确认执行"},
                     "type": "primary", "value": {**action_value, "action": "confirm"}},
                    {"tag": "button", "text": {"tag": "plain_text", "content": "🔗 深度分析"},
                     "type": "default", "value": {**action_value, "action": "deep_analyze"}},
                ]}
            ]
        }
    }, timeout=10)

# ── Agent Report Storage ─────────────────────────────────────────────────

class ReportIn(BaseModel):
    session_key: str
    agent_id: str
    agent_name: str
    inputs: dict
    result: str
    metadata: dict = {}


@app.post("/api/reports")
async def save_report(body: ReportIn):
    report_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO agent_reports VALUES (?,?,?,?,?,?,?,?)",
            (report_id, body.session_key, body.agent_id, body.agent_name,
             json.dumps(body.inputs, ensure_ascii=False),
             body.result, created_at,
             json.dumps(body.metadata, ensure_ascii=False))
        )
        conn.commit()
    return {"id": report_id, "created_at": created_at}


@app.get("/api/reports")
async def get_reports(session_key: str, limit: int = 20):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM agent_reports WHERE session_key=? ORDER BY created_at DESC LIMIT ?",
            (session_key, min(limit, 50))
        ).fetchall()
    return [dict(r) for r in rows]


@app.delete("/api/reports/{report_id}")
async def delete_report(report_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM agent_reports WHERE id=?", (report_id,))
        conn.commit()
    return {"deleted": report_id}


@app.get("/auth/login")
async def auth_login():
    return await _auth_login()


@app.get("/auth/callback")
async def auth_callback(code: str = "", state: str = ""):
    return await _auth_callback(code=code, state=state)


@app.get("/auth/me")
async def auth_me(request: Request):
    return await _auth_me(request)


@app.post("/auth/logout")
async def auth_logout():
    return await _auth_logout()


@app.post("/api/feishu-callback")
async def feishu_callback(request: Request, background: BackgroundTasks):
    body = await request.json()
    action_val = body.get("action", {}).get("value", {})
    agent_id   = action_val.get("agent_id", "")
    action     = action_val.get("action", "")
    inputs_str = action_val.get("inputs_str", "")

    if action == "confirm":
        background.add_task(push_feishu_text,
            "✅ 已确认执行 — paper2skills",
            f"决策已记录。\n时间: {time.strftime('%Y/%m/%d %H:%M')}\nAgent: {agent_id}\n输入: {inputs_str[:200]}",
            "green"
        )
    elif action == "deep_analyze":
        prompt = AGENT_PROMPTS.get(agent_id, "你是跨境电商AI分析助手，给出详细深度分析。")
        background.add_task(_run_deep_analyze, agent_id, prompt, inputs_str)

    return JSONResponse({"code": 0})

async def _run_deep_analyze(agent_id: str, prompt: str, inputs_str: str):
    try:
        result = call_deepseek(prompt, f"深度分析（详细版）：\n{inputs_str}", max_tokens=1800)
        push_feishu_text(f"🔬 深度分析报告 — {agent_id}", result, "purple")
    except Exception as e:
        push_feishu_text("深度分析失败", str(e), "red")

@app.post("/api/feishu-event")
async def feishu_event(request: Request, background: BackgroundTasks):
    body = await request.json()
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}
    event = body.get("event", {})
    msg   = event.get("message", {})
    if msg.get("message_type") == "text":
        content = json.loads(msg.get("content", "{}")).get("text", "")
        if "feishu.cn/docx/" in content or "feishu.cn/wiki/" in content:
            background.add_task(_gen_skill_draft, content)
    return {"code": 0}

async def _gen_skill_draft(doc_url: str):
    try:
        result = call_deepseek(
            "你是 paper2skills Skill 卡片生成专家。根据飞书文档 URL 和描述，生成标准 Skill 草稿。"
            "格式：【Skill名称】【算法原理】【业务应用案例】【代码思路】【技能关联】【商业价值】",
            f"请根据这个飞书文档生成 Skill 草稿：{doc_url}\n"
            "如果无法访问文档，请根据 URL 路径中的关键词推断主题，生成合理的草稿框架。",
            max_tokens=1500
        )
        push_feishu_text("📝 Skill 草稿生成 — paper2skills", result, "wathet")
    except Exception as e:
        push_feishu_text("Skill草稿生成失败", str(e), "red")

@app.post("/api/daily-inspect")
async def daily_inspect(request: Request, background: BackgroundTasks):
    body = await request.json()
    secret = body.get("secret", "")
    if secret != os.environ.get("P2S_INSPECT_SECRET", ""):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    skus = body.get("skus", [])
    background.add_task(_run_daily_inspect, skus)
    return {"code": 0, "message": f"巡检启动，共 {len(skus)} 个SKU"}

async def _run_daily_inspect(skus: list):
    if not skus:
        push_feishu_text("📊 每日巡检 — 无数据", "未提供 SKU 数据，请检查 cron 配置。", "yellow")
        return
    alerts = []
    for sku in skus:
        dos = sku.get("dos", 999)
        acos = sku.get("acos", 0)
        name = sku.get("name", sku.get("id", "unknown"))
        if dos < 30:
            alerts.append(f"⚠️ {name}: DOS={dos}天 低于安全线30天")
        if acos > 40:
            alerts.append(f"⚠️ {name}: ACOS={acos}% 超过阈值40%")
    if not alerts:
        push_feishu_text("📊 每日巡检 — 全部正常", f"共巡检 {len(skus)} 个SKU，无异常。\n{time.strftime('%Y/%m/%d %H:%M')}", "green")
        return
    summary = f"发现 {len(alerts)} 个异常：\n" + "\n".join(alerts)
    try:
        detail = call_deepseek(
            "你是供应链和广告分析专家。根据以下异常指标，给出优先级排序和今日必须执行的3个行动。",
            summary, max_tokens=800
        )
        push_feishu_text(f"🚨 每日巡检异常 — {len(alerts)}项", summary + "\n\n" + detail, "red")
    except Exception:
        push_feishu_text(f"🚨 每日巡检异常 — {len(alerts)}项", summary, "red")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
