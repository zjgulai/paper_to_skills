# AGENTS.md

This file provides guidance to AI agents (OpenCode/Codex) working with this repository.

## Project Overview

**paper2skills** — 将顶刊学术论文转化为可落地的母婴跨境电商 AI 决策 Skill 卡片系统，并通过 Playbook、Agent 市场、飞书集成，形成从「知识库」到「日常决策基础设施」的完整闭环。

4 步内容流水线：论文筛选 → Skill 萃取 → 质量审核 → 同步发布。
产品形态：静态 Playbook 网站 + 21 个真实 AI Agent + 飞书双向集成 + 每日自动巡检。

**Live Playbook**: https://skills.lute-tlz-dddd.top

---

## Current State (2026-06-25)

| Metric | Value |
|--------|-------|
| Skill pages | **1037** |
| Domains | **25** |
| Graph edges | **11,577**（有效边，两端节点均在图谱内） |
| Agents | **21** 个真实 AI Agent（默认 DeepSeek，有结构化 prompt） |
| Solutions | **15** published |
| ps_override entries | **1037** (100% coverage) |
| Playbooks | **33** scene handbooks（每个含「推送到飞书」按钮） |
| Workflows | **27** business workflows |
| Topics | **32** topic pages |
| Risk Ontology | **17 events / 193 Skill mappings** (diagnostic.html) |
| VOID Framework | **运行中** — Q2-2026 Session 完成，8 个问题入库，1 个原型 Skill |
| GA4 | **G-N9HJR3G0MR**（全站埋点：skill_view / skill_code_copy / agent_run） |
| 飞书集成 | **双向** — Agent 报告推送 + Playbook 进度推送 + 每日巡检 + 卡片按钮回调 |

---

## VOID 框架（第三象限创新引擎）

> **理念**：第三象限 = 我不知道 + AI 也不知道的问题。不能用"解题框架"来框，只能在已知体系遭遇压力时从裂缝里显现。

```
VOID = Venture into Original Ignorance Deliberately

V — Venture     主动走到边界（崩溃设计/跨行业移植/隐性知识访谈）
O — Observe     观察摩擦和断裂（盲点登记册/放弃问题存档）
I — Interrogate 审讯已有答案（反转/排除人群/10年测试）
D — Distill     蒸馏未命名问题（三测试→问题银行→原型Skill）
```

**文件位置**：`paper2skills-vault/00-项目管理/VOID/`

**运转节奏**：每季度一次 VOID Session（半天）+ 每月 30 分钟问题银行维护

**第一次第三象限→主线转化（2026-06-24）**：
盲点 B003 → AQ-004 → 婴儿月龄时钟 → 3 个正式 Skill 上线：
- `Skill-Baby-Age-Clock-RFM-Enhancement`（14-用户分析）
- `Skill-Infant-Lifecycle-Purchase-Rhythm`（06-增长模型）
- `Skill-Baby-Age-Aware-Recommendation`（05-推荐系统）

---

## Project Structure

```
├── paper2skills-skills/
│   ├── paper-workflow/          # 完整工作流编排 skill
│   ├── paper-选题/              # Step 1: 论文筛选
│   ├── paper-萃取/              # Step 2: Skill 萃取
│   ├── paper-审核/              # Step 3: 质量审核
│   ├── paper-同步/              # Step 4: 多端同步
│   └── playbook-generator/
│       └── scripts/
│           ├── build_playbook.py              # ⚠️ CSS + 全部页面由此生成，不要直接改输出文件
│           └── config/
│               ├── skill_ps_override.yaml     # problem_solved 覆盖（1037条，100%覆盖）
│               ├── risk_events_ontology.yaml  # ⚠️ 风险事件Ontology（17事件/193映射）
│               ├── agents_data.py             # 21个 Agent 定义（输入字段/demo输出/关联Skill）
│               ├── playbooks_data.py          # 33个 Playbook 定义
│               ├── skill_biz_context_override.yaml
│               └── skill_handbook_map.yaml
├── paper2skills-vault/          # 1037 个 Skill 卡片（Obsidian 兼容 Markdown）
│   ├── 01-因果推断/    (21 skills)
│   ├── 02-A_B实验/     (20 skills)
│   ├── 03-时间序列/    (38 skills)
│   ├── 04-供应链/      (129 skills) ← 最大域
│   ├── 05-推荐系统/    (27 skills)
│   ├── 06-增长模型/    (59 skills)
│   ├── 07-NLP-VOC/     (36 skills)
│   ├── 08-知识图谱/    (52 skills)
│   ├── 09-DataAgent-LLM/ (20 skills)
│   ├── 10-MAS/         (65 skills)
│   ├── 11-AI人文/      (20 skills)
│   ├── 12-ML基础/      (20 skills)
│   ├── 13-广告分析/    (52 skills)
│   ├── 14-用户分析/    (52 skills)
│   ├── 15-营销投放分析/ (42 skills)
│   ├── 16-智能体工程/  (66 skills)
│   ├── 17-价格优化/    (36 skills)
│   ├── 18-物流履约/    (20 skills)
│   ├── 19-风控反欺诈/  (36 skills)
│   ├── 20-AI视频生成/  (35 skills)
│   ├── 21-合规决策/    (31 skills)
│   ├── 22-数据采集工程/ (26 skills)
│   ├── 23-运营财务/    (35 skills)
│   ├── 24-标签工程/    (61 skills)
│   ├── 25-搜索流量工程/ (35 skills)
│   └── 07-资源库/
│       ├── MasterPrompt.md
│       ├── 关键词库.md
│       └── SC-TagToDecision-Architecture.md
├── paper2skills-code/           # Python 代码模板（各域子目录）
└── playbook/                    # build 输出目录（不要手动修改）
    ├── assets/
    │   ├── graph-data.json      # 知识图谱数据（节点+有效边）
    │   └── risk-events.json     # 风险事件 Ontology
    └── build-report.json        # 构建摘要（skill_pages/domains/edges）
```

---

## Key Files

| File | Purpose |
|------|---------|
| `paper2skills-skills/playbook-generator/scripts/build_playbook.py` | **全站构建入口**（CSS/JS/HTML 全部在这里）|
| `paper2skills-skills/playbook-generator/scripts/config/agents_data.py` | 21 个 Agent 定义，修改 Agent 必须改这里 |
| `paper2skills-skills/playbook-generator/scripts/config/playbooks_data.py` | 33 个 Playbook 定义 |
| `paper2skills-skills/playbook-generator/scripts/config/skill_ps_override.yaml` | problem_solved 语句覆盖 |
| `paper2skills-vault/07-资源库/MasterPrompt.md` | 论文转 Skill 的核心 Prompt |
| `paper2skills-vault/07-资源库/关键词库.md` | ArXiv 搜索关键词分域索引 |

---

## ⚠️ 关键工程注意事项

### CSS / JS 修改必须改 `build_playbook.py`

`playbook/assets/style.css`、`playbook/assets/search.js` 等**全部由 `build_playbook.py` 生成**。
每次 build 都会重建输出目录。**直接改 playbook/ 下的文件无效**，必须改对应的 Python 函数。

| 输出文件 | 来源函数 |
|---------|---------|
| `assets/style.css` | `build_css()` |
| `assets/graph.js` | `build_graph_js()` |
| `assets/chat-page.js` | `build_chat_page_js()` |
| `agents.html` | `render_agents_page()` + 7300 行后的 patch 逻辑 |
| `diagnostic.html` | `render_diagnostic_page()` |
| `chat.html` | `render_chat_page()` |

### Agent 系统关键常量（build_playbook.py 顶部）

```python
GA4_MEASUREMENT_ID = "G-N9HJR3G0MR"        # GA4 埋点 ID，placeholder 时自动跳过注入
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/a32b3ab7-6cfb-498d-bc3f-91d9f48b47e9"
```

### Agent 架构

- **前端**：`agents.html` 的 21 个 Agent 默认走「AI 真实分析」模式
- **AI 调用**：`POST /api/agent` → nginx 代理 → DeepSeek API（`sk-aae11f...`）
- **结果推送**：`pushToFeishu()` 把真实输入+AI输出推飞书群
- **系统 prompt**：`AGENT_PROMPTS` dict（build_playbook.py 约 7355 行），每个 Agent 有结构化格式要求
- **新增/修改 Agent**：改 `config/agents_data.py`（定义卡片），同步更新 `AGENT_PROMPTS`（定义 prompt）

### 知识图谱边数说明

- `build-report.json` 的 `edges` = **有效边**（两端节点均存在于图谱）
- 原始边（含悬挂边）数量更多，不反映真实图谱质量
- 两处数字现在保持一致（2026-06-25 修复）

### nginx 配置注意事项

skills 域名的 nginx 配置存放在**独立文件**，不在主 nginx.conf 里，避免被其他服务部署覆盖：

```
宿主机路径：/opt/ai-video/deploy/lighthouse/skills.conf
容器内挂载：/etc/nginx/skills.conf（只读）
主 nginx.conf 末尾包含：include /etc/nginx/skills.conf;
docker-compose volume：/opt/paper2skills/html:/var/www/skills:ro
                        /opt/ai-video/deploy/lighthouse/skills.conf:/etc/nginx/skills.conf:ro
```

**修改 nginx 路由**：改宿主机 `skills.conf`，然后 `docker exec ai_video_nginx nginx -s reload`。
**若其他服务部署覆盖了 nginx.conf**：重新执行 `docker compose up -d --no-deps --force-recreate nginx` 重建容器。

---

## 服务器后端服务

### p2s-service（FastAPI，端口 8765）

```
宿主机路径：/opt/paper2skills/service/app.py
systemd 服务：p2s-service.service（开机自启，崩溃自动重启）
日志：journalctl -u p2s-service -f
```

提供以下接口（通过 nginx 代理到公网）：

| 接口 | 用途 |
|------|------|
| `POST /api/feishu-callback` | 飞书卡片按钮回调（确认执行 / 深度分析） |
| `POST /api/feishu-event` | 飞书事件订阅（文档分享 → Skill 草稿，需飞书开放平台配置） |
| `POST /api/daily-inspect` | 每日 SKU 巡检，异常推飞书（secret: `p2s_inspect_2026`） |

### cron 每日巡检

```bash
# crontab -l 查看
0 9 * * * /opt/paper2skills/service/daily_inspect.sh
```

脚本路径：`/opt/paper2skills/service/daily_inspect.sh`
**修改巡检 SKU 列表**：编辑该脚本的 `SKUS_JSON`，或接入飞书多维表格 Webhook。

---

## Playbook Pages

| Page | URL | Description |
|------|-----|-------------|
| `/` | index.html | 总览仪表盘（含 Freemium 升级 CTA） |
| `/diagnostic.html` | 业务诊断中心 | 症状→Skill链 + BFS路径规划器 + 风险事件 Ontology |
| `/chat.html` | AI知识库对话 | DeepSeek RAG + 角色分层（运营/分析师/CEO）+ 症状路由 |
| `/playbooks/` | 场景手册 (33个) | 可执行业务手册 + 飞书进度推送按钮 |
| `/solutions/` | 方案库 (15个) | 系统架构方案 |
| `/agents.html` | 智能体广场 | 21 个真实 AI Agent（默认 DeepSeek，结果推飞书） |
| `/agent-report.html` | 智能体报告 | localStorage 持久化历史报告 |
| `/ai-roadmap.html` | AI 能力路线图 | CEO-facing 白皮书 |
| `/skills/` | 全部 Skills (1037个) | 每 Skill 独立详情页（GA4 skill_view 埋点） |
| `/domains/` | 25 个领域页 | 按领域浏览（含 Freemium 升级 CTA） |
| `/graph/overview.html` | 技能关系图谱 | D3 可视化（11,577 有效边） |
| `/workflows/` | 业务工作流 (27个) | 场景 → Skill 的执行链路 |
| `/topics/` | 专题页面 (32个) | 跨域主题聚合 |
| `/maturity-report.html` | AI成熟度报告 | 母婴跨境 AI 能力成熟度白皮书 |

---

## Build & Deploy SOP

```bash
# 1. Build（验证无 WARN dup_ps）
cd /Users/lute/project/paper_to_skills
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . --vault paper2skills-vault --out playbook 2>&1 | tail -8

# 2. 打包
cd playbook && tar -czf /tmp/pb.tar.gz \
  assets/ domains/ graph/ playbooks/ topics/ workflows/ skills/ solutions/ \
  *.html build-report.json README.md

# 3. 上传 + 解压
rsync -avz --timeout=60 -e "ssh -i ../ai_video.pem -o StrictHostKeyChecking=no" \
  /tmp/pb.tar.gz ubuntu@101.34.52.232:/tmp/
ssh -i ../ai_video.pem -o StrictHostKeyChecking=no ubuntu@101.34.52.232 \
  "rm -rf /opt/paper2skills/html/* && tar -xzf /tmp/pb.tar.gz -C /opt/paper2skills/html/ && rm /tmp/pb.tar.gz"

# 4. 验证
python3 -c "
import urllib.request, ssl, json
ctx = ssl.create_default_context()
r = json.loads(urllib.request.urlopen('https://skills.lute-tlz-dddd.top/build-report.json', timeout=8, context=ctx).read())
print(f'✅ {r[\"skill_pages\"]} Skills / {r[\"domains\"]}域 / {r[\"edges\"]}边')
"
```

---

## Design System v5（当前版本）

**风格**：Smartisan 克制科技感 · Linear 精密线条 · 顶级咨询高级感

| Token | 值 | 含义 |
|-------|-----|------|
| `--nav-bg` | `#111111` | 深黑顶栏 |
| `--bg` | `#F6F6F6` | 冷灰大背景 |
| `--accent` | `#B5323E` | 深玫红品牌色 |
| `--r-lg` | `8px` | 卡片圆角（克制） |
| `--ink` | `#0C0C0C` | 接近纯黑文字 |
| topbar height | `52px` | 深色压舱石 |

**卡片原则**：无默认阴影，仅 `1px border`；hover 时 `translateY(-2px)` + 轻阴影 + 边框加深。

---

## Agent Marketplace（21个，全部真实 AI）

| 分类 | Agent ID | 名称 |
|------|---------|------|
| 供应链 | agent-supply-sentinel | 供应链哨兵 |
| 供应链 | agent-festival-replenishment | 大促补货决策师 |
| 定价 | agent-pricing-advisor | 动态定价顾问 |
| 定价 | agent-dml-counterfactual-pricing | 反事实定价引擎 |
| 财务 | agent-pnl-analyzer | P&L透视镜 |
| 财务 | agent-margin-calculator | SKU利润归因计算器 |
| 广告 | agent-ad-attribution | 广告归因侦探 |
| 竞品 | agent-competitor-radar | 竞品雷达站 |
| 运营 | agent-listing-doctor | Listing医生 |
| 运营 | agent-voc-decoder | 用户之声解码器 |
| 运营 | agent-cs-triage | 客服分诊台 |
| 运营 | agent-tiktok-content | TikTok内容官 |
| 运营 | agent-cold-start-advisor | 新品冷启动顾问 |
| 选品 | agent-product-radar | 选品雷达 |
| 风控 | agent-account-guardian | 账号风险卫士 |
| 风控 | agent-brand-guardian | 品牌合规卫士 |
| 风控 | agent-geopolitical-risk | 地缘风险评估仪 |
| 合规 | agent-compliance-matrix | 多市场合规矩阵 |
| 合规 | agent-epr-calculator | EPR合规费用测算 |
| 标签 | agent-sku-tag-scanner | SKU标签质量扫描器 |
| 物流 | agent-return-analyzer | 退货根因分析师 |

- **定义文件**：`config/agents_data.py`（输入字段、demo输出、关联Skill）
- **AI Prompt**：`build_playbook.py` 约 7355 行的 `AGENT_PROMPTS` dict
- **运行结果**：自动保存 `localStorage('agentReports')` + 推飞书群

---

## 飞书集成架构

```
用户行为
  │
  ├─ Agent 运行 ──────────────→ pushToFeishu() ──→ 飞书群（AI分析卡片）
  │                                                     │
  │                                              [确认执行][深度分析]
  │                                                     │
  │                                         POST /api/feishu-callback
  │                                                     │
  │                                          p2s-service:8765 处理
  │
  ├─ Playbook 进度推送 ────────→ /api/feishu-callback ──→ 飞书群（进度卡片）
  │
  ├─ cron 09:00 每日巡检 ──────→ /api/daily-inspect ───→ 飞书群（异常告警）
  │
  └─ 飞书多维表格自动化（配置后）→ /api/daily-inspect ───→ 飞书群（实时告警）

待办（下次做）：
  ├─ 飞书多维表格 → 自动触发 Agent（Webhook 自动化规则，无需改代码）
  └─ 飞书文档分享 → Skill 草稿生成（需飞书开放平台事件订阅配置）
```

---

## Solutions（方案库）

位置：`playbook/solutions/`，在侧边导航「◆ 方案库」入口。

**扩展方式**：在 `build_playbook.py` 的 `SOLUTIONS_CATALOG` 列表添加 dict → rebuild 自动生成。

---

## 通用业务诊断 SOP

**Web 端**（推荐）：打开 `diagnostic.html` → 输入症状关键词 → 匹配风险事件 → 获取三层 Skill 链。
支持 BFS Skill 路径规划：侧边栏「Skill 路径规划」折叠区，选起点+终点查找最短路径。

**CLI 端**：`paper2skills-skills/diagnostic-sop/diagnose.py`

```bash
python3 paper2skills-skills/diagnostic-sop/diagnose.py \
  --sku "暖奶器" --channel amazon --problem repurchase_drop
```

支持问题类型：`repurchase_drop` / `roas_decline` / `traffic_drop` / `new_cvr_low` / `inventory_issue` / `review_attack`

---

## Skill Card Format

5 模块结构（每个模块必须有实质内容）：

1. **① 算法原理**（≤300字）— 核心思想、数学直觉、关键假设
2. **② 母婴出海应用案例**（1-2个场景）— 具体业务痛点、数据要求、量化产出
3. **③ 代码模板** — 完整可运行 Python，末尾输出 `[✓] XXX 测试通过`
4. **④ 技能关联** — 前置/延伸/可组合，至少 2 条 `[[双括号链接]]`
5. **⑤ 商业价值评估** — ROI 量化数字，实施难度 ⭐，优先级 ⭐

**Pass threshold**: 总分 ≥ 7/10，代码维度 ≥ 7/10

---

## Domain Mapping（全量）

| 中文目录 | 域 | Skills |
|---------|-----|--------|
| 01-因果推断 | Causal inference, uplift modeling | 21 |
| 02-A_B实验 | A/B testing, multi-armed bandits | 20 |
| 03-时间序列 | Demand forecasting | 38 |
| 04-供应链 | Inventory, SC optimization | 129 |
| 05-推荐系统 | Recommendation systems | 27 |
| 06-增长模型 | Churn, LTV, growth | 59 |
| 07-NLP-VOC | Sentiment, VOC mining | 36 |
| 08-知识图谱 | KG, GNN, hyperbolic | 52 |
| 09-DataAgent-LLM | Data agent, LLM analytics | 20 |
| 10-MAS | Multi-agent systems | 65 |
| 11-AI人文 | AI ethics, AIGC | 20 |
| 12-ML基础 | ML fundamentals | 20 |
| 13-广告分析 | Ad attribution, ROAS | 52 |
| 14-用户分析 | Funnel, cohort, RFM | 52 |
| 15-营销投放分析 | MMM, promo effectiveness | 42 |
| 16-智能体工程 | LLM Agent engineering | 66 |
| 17-价格优化 | Dynamic pricing | 36 |
| 18-物流履约 | Cross-border logistics | 20 |
| 19-风控反欺诈 | Fraud detection | 36 |
| 20-AI视频生成 | Virtual anchor, brand video | 35 |
| 21-合规决策 | Compliance decisions | 31 |
| 22-数据采集工程 | Data collection, quality | 26 |
| 23-运营财务 | FBA finance, P&L, tariff | 35 |
| 24-标签工程 | Tag engineering, Palantir ontology, action triggers | 61 |
| 25-搜索流量工程 | Search traffic, SEO, A9 algorithm | 35 |

---

## ps_override 规范

格式：`{业务角色}面临{具体场景}——{方法}将{A}改善为{B}，年化{ROI数字}`

```bash
# 追加新条目
cat >> paper2skills-skills/playbook-generator/scripts/config/skill_ps_override.yaml << 'EOF'
Skill-XXX: 运营面临"具体痛点"——方法将指标A→指标B，年化节省X万元
EOF

# 验证无 WARN dup_ps
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . --vault paper2skills-vault --out playbook 2>&1 | grep "WARN"
```

---

## Dependencies

```bash
cd paper2skills-code && pip install -r requirements.txt
```

Key: numpy, pandas, scikit-learn, statsmodels, prophet, causalml, econml

## Local service mirror

- `paper2skills-services/` 用于从生产服务器镜像 `p2s-service` 的后端源码、依赖与运行说明。
