---
name: phase2-product-foundation
description: paper2skills Phase 2 产品基础建设计划（1-3 月）。5 个工程目标：拆解巨文件 build_playbook.py、Agent 结果持久化后端、Chat 真 RAG 实现、飞书 OAuth 用户系统、CI/CD 流水线。完成后商业化就绪度从当前 ~3/10 提升到 ~6/10。当被要求执行「Phase 2」或「产品基础」时使用。
---

# paper2skills Phase 2 — 产品基础建设计划（1-3 月）

**目标**：把三层孤立资产（Skill 内容 / MAS 框架 / 前端 Agent）打通，建立可商业化的产品基础。  
**前置条件**：Phase 1 全部完成  
**预期商业化就绪度**：~3/10 → ~6/10

---

## M1：拆解 build_playbook.py（工程债清偿）

**优先级**：P0 — 后续所有 Phase 2 工作都依赖这个重构

**问题**：7937 行单体文件，54 个函数混杂 CSS/JS/HTML/数据逻辑，任何改动都有全局风险。

**目标架构**：
```
paper2skills-skills/playbook-generator/scripts/
├── build.py                    # 主入口（原 main() + build()，≤200行）
├── config/                     # 已有，保持
├── builders/
│   ├── __init__.py
│   ├── css_builder.py          # build_css()
│   ├── js_builder.py           # build_graph_js() + build_chat_page_js() + build_search_js() + build_ego_graph_js()
│   ├── page_renderers.py       # render_index() + render_graph_page() + render_maturity_report() + render_diagnostic_page() + render_chat_page()
│   ├── skill_renderers.py      # render_skill_page() + render_skill_card() + render_items()
│   ├── agent_renderers.py      # render_agents_page() + render_agent_report_page()
│   ├── playbook_renderers.py   # render_tob_playbook() + _render_playbook_progress_page()
│   ├── workflow_renderers.py   # render_workflow_page() + render_workflow_step()
│   └── solution_renderers.py   # render_solutions_index() + render_solution_detail()
├── extractors/
│   ├── __init__.py
│   ├── skill_parser.py         # parse_frontmatter() + section_map() + extract_*() 系列
│   └── graph_builder.py        # build_graph() + build_skills()
└── utils/
    ├── __init__.py
    └── html_helpers.py         # html_page() + skill_url() + link_list() + slugify() + read_text()
```

**执行步骤**：

1. **基准测试先行**：运行现有 build，记录输出文件 hash 作为回归基准
   ```bash
   find playbook/ -name "*.html" | xargs md5 > /tmp/build-baseline.txt
   ```

2. **逐模块迁移**（顺序：utils → extractors → builders → build.py）：
   - 每迁移一个模块，立即跑 build 验证 hash 不变
   - 使用相对 import，保持 `paper2skills_common` 依赖不变

3. **JS/CSS 单独文件化**：把内联 CSS 字符串（约 1500 行）移到 `builders/css_builder.py`，对外只暴露 `build_css() -> str`

4. **AGENT_PROMPTS 单独配置文件**：把 ~7355 行之后的 `AGENT_PROMPTS` dict 移到 `config/agent_prompts.py`

5. **验收**：重构后 build 产物与基准 hash 完全一致
   ```bash
   find playbook/ -name "*.html" | xargs md5 > /tmp/build-refactored.txt
   diff /tmp/build-baseline.txt /tmp/build-refactored.txt
   # 允许时间戳差异，禁止内容差异
   ```

**风险控制**：每个子模块迁移完即 commit，不攒大 PR。

---

## M2：Agent 结果持久化后端

**问题**：21 个 Agent 结果存在 `localStorage`，换浏览器/清缓存即丢失，无法跨设备共享，无法审计。

**架构**：在 p2s-service（FastAPI）新增 SQLite 持久化层，前端改为调后端存取。

**执行步骤**：

### 2.1 后端 Schema（p2s-service/app.py）

```python
# 新增表结构
CREATE TABLE agent_reports (
    id TEXT PRIMARY KEY,        -- UUID
    session_key TEXT NOT NULL,  -- 浏览器指纹（无登录时用 fingerprint）
    agent_id TEXT NOT NULL,     -- 如 "agent-pricing-advisor"
    agent_name TEXT NOT NULL,
    inputs TEXT NOT NULL,       -- JSON
    result TEXT NOT NULL,       -- AI 输出文本
    created_at TEXT NOT NULL,   -- ISO8601
    metadata TEXT               -- JSON（预留：用户ID、版本等）
);
CREATE INDEX idx_session ON agent_reports(session_key, created_at DESC);
```

### 2.2 新增 API 端点（p2s-service/app.py）

```
POST /api/reports          # 保存一条报告
GET  /api/reports?session_key=xxx&limit=20  # 拉取历史
DELETE /api/reports/{id}   # 删除单条
```

### 2.3 前端改造（build_playbook.py → agents.html）

将 `agentReports` 的存取从 `localStorage` 改为 API 调用：
- 页面加载时：`GET /api/reports?session_key=fingerprint` 初始化历史
- Agent 运行完：`POST /api/reports` 保存（同时保留 localStorage 作为离线降级）
- session_key 用 `navigator.userAgent + screen.width` 的 hash（无需注册即可区分设备）

### 2.4 部署

```bash
# 服务器上
cd /opt/paper2skills/service
python3 -c "import sqlite3; sqlite3.connect('reports.db').execute('CREATE TABLE ...')"
systemctl restart p2s-service
```

**验收**：
```bash
# 本地调用测试
curl -X POST http://localhost:8765/api/reports \
  -H "Content-Type: application/json" \
  -d '{"session_key":"test","agent_id":"agent-pricing-advisor","agent_name":"定价顾问","inputs":{},"result":"测试结果"}'
# 返回 {"id": "uuid..."}
curl http://localhost:8765/api/reports?session_key=test
# 返回报告列表
```

---

## M3：Chat 页面真 RAG 实现

**问题**：`chat.html` 直接把用户问题发给 DeepSeek，没有检索 1054 个 Skill 的任何内容，AI 是在用训练数据幻觉回答，不是在用知识库回答。

**架构**：离线构建 Skill 向量索引，Chat 时先检索 Top-K Skill，再注入 System Prompt。

**执行步骤**：

### 3.1 构建 Skill 摘要索引（build 阶段）

在 `build_playbook.py` 的 `build()` 函数末尾，生成 `assets/skill-index.json`：

```python
# skill-index.json 格式
[
  {
    "id": "Skill-Dynamic-Pricing-Elasticity",
    "title": "动态定价弹性模型",
    "domain": "17-价格优化",
    "summary": "①算法原理摘要（≤200字）②业务场景关键词 ③problem_solved",
    "keywords": ["定价", "弹性", "价格优化", "GMV"]
  },
  ...
]
```

### 3.2 前端检索层（chat.html JS）

用 TF-IDF 近似（无需后端向量服务）：
```javascript
// 加载索引
const skillIndex = await fetch('/assets/skill-index.json').then(r => r.json());

// 用户提问时：关键词匹配 Top-5 Skills
function retrieveRelevantSkills(query, topK=5) {
  const tokens = query.toLowerCase().split(/\W+/);
  return skillIndex
    .map(skill => ({
      ...skill,
      score: tokens.filter(t => 
        skill.keywords.includes(t) || skill.summary.includes(t)
      ).length
    }))
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// 注入 System Prompt
function buildSystemPrompt(retrievedSkills) {
  const ctx = retrievedSkills.map(s => 
    `[${s.id}] ${s.title}: ${s.summary}`
  ).join('\n');
  return `你是 paper2skills 知识库助手。以下是与用户问题最相关的 Skill 卡片，请基于这些内容回答，并引用具体 Skill ID：\n\n${ctx}\n\n规则：只引用上面列出的 Skill，不要编造 Skill ID。`;
}
```

### 3.3 Source 引用显示

AI 回答后，前端在消息气泡下方展示「参考 Skill」列表，点击跳转 `/skills/{id}.html`。

### 3.4 构建 skill-index.json 的 build 步骤

```python
def build_skill_index(skills: list[PlaybookSkill]) -> list[dict]:
    return [
        {
            "id": s.skill_id,
            "title": s.title,
            "domain": s.domain_id,
            "summary": f"{s.algo_summary[:150]} {s.problem_solved}",
            "keywords": s.tags + extract_keywords(s.problem_solved)
        }
        for s in skills
    ]
```

**验收**：
```bash
# 构建后检查索引文件
python3 -c "
import json
idx = json.load(open('playbook/assets/skill-index.json'))
print(f'✅ {len(idx)} skills indexed')
assert len(idx) > 1000
assert all('keywords' in s for s in idx)
"
```

---

## M4：飞书 OAuth 最小用户系统

**问题**：零用户身份 = 无法区分试用/付费用户，无法做权限控制，无法 CRM。

**策略**：用飞书 SSO（企业客户 90% 有飞书），实现最小可用的用户认证，不引入额外 SaaS 依赖。

**执行步骤**：

### 4.1 飞书应用创建

1. 在飞书开放平台创建「paper2skills」企业自建应用
2. 开启「网页应用」能力，配置回调 URL：`https://skills.lute-tlz-dddd.top/auth/callback`
3. 申请权限：`contact:user.id:readonly`（获取 User ID）

### 4.2 后端认证端点（p2s-service/app.py）

```python
# 新增端点
GET  /auth/login           # 重定向到飞书 OAuth 授权页
GET  /auth/callback        # 换取 access_token，写 JWT cookie
GET  /auth/me              # 返回当前用户信息（id, name, avatar）
POST /auth/logout          # 清除 cookie

# JWT payload
{
  "sub": "feishu_user_id",
  "name": "张三",
  "org": "company_name",
  "tier": "free",           # free | pro（预留）
  "exp": timestamp
}
```

### 4.3 前端接入（最小改动）

在 `html_page()` 函数生成的 `<head>` 中注入：
```javascript
// 检查登录态（不强制，仅用于个性化）
async function checkAuth() {
  try {
    const user = await fetch('/auth/me').then(r => r.json());
    if (user?.name) {
      document.getElementById('user-avatar')?.textContent = user.name[0];
    }
  } catch {}
}
```

顶栏增加「登录」按钮，未登录时显示，登录后显示用户头像。**第一版不做强制登录门槛**，只做身份识别，为后续 Freemium 打基础。

### 4.4 用户表（p2s-service SQLite）

```sql
CREATE TABLE users (
    feishu_id TEXT PRIMARY KEY,
    name TEXT,
    org TEXT,
    tier TEXT DEFAULT 'free',   -- free | pro | enterprise
    created_at TEXT,
    last_seen TEXT,
    usage_count INTEGER DEFAULT 0  -- Agent 调用次数
);
```

**验收**：
```bash
# 本地模拟 OAuth 回调
curl http://localhost:8765/auth/me
# {"feishu_id": null, "tier": "free"}（未登录时）
```

---

## M5：CI/CD 流水线

**问题**：全手工 5 步部署 SOP，无自动化，无质量门控。

**目标**：push 到 `main` 分支自动触发 build + 部署。

**执行步骤**：

### 5.1 创建 GitHub Actions 工作流

文件：`.github/workflows/deploy.yml`

```yaml
name: Build & Deploy Playbook

on:
  push:
    branches: [main]
    paths:
      - 'paper2skills-vault/**'
      - 'paper2skills-skills/playbook-generator/**'
      - 'paper2skills-skills/paper-skills-graph/**'

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          
      - name: Install dependencies
        run: pip install -r paper2skills-skills/playbook-generator/requirements.txt
        
      - name: Build Playbook
        run: |
          python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
            --root . --vault paper2skills-vault --out playbook
        env:
          P2S_FEISHU_WEBHOOK_URL: ${{ secrets.P2S_FEISHU_WEBHOOK_URL }}
          
      - name: Verify build
        run: |
          python3 -c "
          import json
          r = json.load(open('playbook/build-report.json'))
          assert r['skill_pages'] > 1000, f'Only {r[\"skill_pages\"]} skills'
          assert r['edges'] > 10000, f'Only {r[\"edges\"]} edges'
          print(f'✅ {r[\"skill_pages\"]} Skills / {r[\"domains\"]} Domains / {r[\"edges\"]} Edges')
          "
          
      - name: Package
        run: |
          cd playbook && tar -czf /tmp/pb.tar.gz \
            assets/ domains/ graph/ playbooks/ topics/ workflows/ skills/ solutions/ \
            *.html build-report.json README.md
            
      - name: Deploy to server
        uses: appleboy/scp-action@v0.1.7
        with:
          host: 101.34.52.232
          username: ubuntu
          key: ${{ secrets.SERVER_SSH_KEY }}
          source: /tmp/pb.tar.gz
          target: /tmp/
          
      - name: Unpack on server
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: 101.34.52.232
          username: ubuntu
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            rm -rf /opt/paper2skills/html/*
            tar -xzf /tmp/pb.tar.gz -C /opt/paper2skills/html/
            rm /tmp/pb.tar.gz
            echo "✅ Deployed at $(date)"
```

### 5.2 GitHub Secrets 配置

在 GitHub 仓库 Settings → Secrets 添加：
- `SERVER_SSH_KEY`：ai_video.pem 内容
- `P2S_FEISHU_WEBHOOK_URL`：飞书 Webhook URL

### 5.3 质量门控（build 失败则停止部署）

在 Build 步骤后加 `check_quality.py` 检查：
```bash
python3 paper2skills-skills/playbook-generator/scripts/check_quality.py \
  --vault paper2skills-vault --threshold 7.0
# 平均质量分 < 7.0 时 exit(1)，阻断部署
```

**验收**：
```bash
git push origin main
# 5 分钟后检查
python3 -c "
import urllib.request, ssl, json
ctx = ssl.create_default_context()
r = json.loads(urllib.request.urlopen('https://skills.lute-tlz-dddd.top/build-report.json', timeout=8, context=ctx).read())
print(f'✅ 自动部署成功: {r[\"skill_pages\"]} Skills')
"
```

---

## 执行顺序与里程碑

```
Week 1-2:   M1（拆解 build_playbook.py）
Week 3-4:   M2（Agent 持久化）+ M5（CI/CD）
Week 5-6:   M3（Chat RAG）
Week 7-8:   M4（飞书 OAuth）
Week 9-10:  集成测试 + 压测 + 文档更新
Week 11-12: 缓冲 + 回归验证
```

## 完成标志

- [ ] `build_playbook.py` < 500 行，其余逻辑分散在 `builders/` / `extractors/` / `utils/`
- [x] Agent 报告刷新浏览器后仍在，且可在手机端看到
- [x] Chat 回答中出现具体 Skill ID 引用（如 `Skill-Dynamic-Pricing-Elasticity`）
- [ ] 飞书扫码后顶栏显示用户名
- [x] push 代码后 5 分钟内生产站自动更新
