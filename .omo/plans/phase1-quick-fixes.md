---
name: phase1-quick-fixes
description: paper2skills Phase 1 立即修复计划（1-2 周）。修复 4 项高优先级工程债：硬编码数字、根目录垃圾文件、飞书 Webhook 安全暴露、p2s-service 源码入库。所有修复不涉及功能变更，只消除已知风险和数据错误。当被要求执行「Phase 1」或「立即修复」时使用。
---

# paper2skills Phase 1 — 立即修复计划

**目标**：消除 4 个现有风险，无需用户感知，不新增功能。  
**预期工时**：2-3 天  
**基线**：2026-06-25 | Skills: 1054 | Domains: 25 | Edges: 11736

---

## T1：修复硬编码数字（数据准确性）

**问题**：build_playbook.py 中 6 个函数的 `skill_count`/`edge_count` 使用过期默认值（849/931/1010/17419），与 build-report.json 实际值不符，导致对外白皮书展示虚假数据。

**文件**：`paper2skills-skills/playbook-generator/scripts/build_playbook.py`

**执行步骤**：

1. 找到 `build()` 函数的返回值（已包含 `skill_pages`/`edges`/`domains`）
2. 在 `render_pages()` 调用链中，把 `stats` 字典向下传递到以下 6 个函数：
   - `render_maturity_report(skill_count=..., edge_count=..., domain_count=...)`
   - `render_diagnostic_page(skill_count=..., build_ts=...)`
   - `render_solutions_index(total_skill_count=...)`
   - `render_solution_detail(sol, total_skill_count=...)`
   - `render_roadmap_page(skill_lookup, skill_count=...)`
   - `render_chat_page(nav=..., skill_count=...)`
3. 删除各函数的硬编码默认值，改为必填参数（无默认值），让调用方显式传递
4. 在 `build()` 函数末尾验证：`assert stats["skill_pages"] > 1000`（防回归）

**验收**：
```bash
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . --vault paper2skills-vault --out playbook 2>&1 | grep -E "skill_pages|edges"
# maturity-report.html 内搜索 "1054" 应出现
grep -c "1054" playbook/maturity-report.html
```

---

## T2：清理根目录垃圾文件

**问题**：根目录堆放 7 个临时 fix 脚本 + 72 张 PNG 截图，污染 git 历史，使 AI Agent 难以理解项目结构。

**执行步骤**：

1. 创建归档目录：
   ```bash
   mkdir -p /Users/lute/project/paper_to_skills/archive/fix-scripts-2026
   mkdir -p /Users/lute/project/paper_to_skills/archive/screenshots-2026
   ```

2. 移动临时脚本（7 个）：
   ```bash
   mv fix_architecture_and_ui.py fix_lead_gen.py fix_modal_fstring.py \
      inject_daas_modal.py inject_lead_gen.py \
      upgrade_architecture.py upgrade_graph.py \
      archive/fix-scripts-2026/
   ```

3. 移动 PNG 截图（72 张）：
   ```bash
   mv *.png archive/screenshots-2026/
   ```

4. 把 `archive/` 加入 `.gitignore`（如无则创建）：
   ```
   archive/
   venv/
   __pycache__/
   *.pyc
   /tmp/
   ```

5. 检查根目录剩余文件是否合理：
   ```bash
   ls *.py *.md 2>/dev/null
   # 应只剩：README.md、AGENTS.md、CLAUDE.md、pyproject.toml 等正式文件
   ```

**验收**：
```bash
ls /Users/lute/project/paper_to_skills/*.png 2>&1 | grep -c "No such" 
# 输出 1（无 PNG 残留）
ls /Users/lute/project/paper_to_skills/*.py 2>&1 | grep -c "No such"
# 输出 1（无临时脚本残留）
```

---

## T3：飞书 Webhook URL 移出代码

**问题**：`FEISHU_WEBHOOK_URL` 硬编码在 `build_playbook.py` 顶部，被打包进 HTML 输出，任何访客可从前端 JS 提取后向企业飞书群发消息。

**文件**：`paper2skills-skills/playbook-generator/scripts/build_playbook.py`

**执行步骤**：

1. 在 `build_playbook.py` 顶部找到硬编码行：
   ```python
   FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/a32b3ab7-..."
   ```

2. 替换为环境变量读取：
   ```python
   import os
   FEISHU_WEBHOOK_URL = os.environ.get("P2S_FEISHU_WEBHOOK_URL", "")
   ```

3. 在构建时注入：如果 `FEISHU_WEBHOOK_URL` 为空，前端 `pushToFeishu()` 函数静默跳过（不报错）。找到 build_playbook.py 里把 `FEISHU_WEBHOOK_URL` 注入 HTML/JS 的位置，改为：
   ```python
   webhook_js = f"const FEISHU_WEBHOOK_URL = '{FEISHU_WEBHOOK_URL}';" if FEISHU_WEBHOOK_URL else "const FEISHU_WEBHOOK_URL = '';"
   ```

4. 创建本地 `.env.local` 文件（加入 `.gitignore`）：
   ```
   P2S_FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/a32b3ab7-6cfb-498d-bc3f-91d9f48b47e9
   ```

5. 更新 Build & Deploy SOP（AGENTS.md）：
   ```bash
   # 构建前加载环境变量
   export P2S_FEISHU_WEBHOOK_URL="$(cat .env.local | grep P2S_FEISHU_WEBHOOK_URL | cut -d= -f2)"
   python3 ... build_playbook.py ...
   ```

6. 同样处理 `GA4_MEASUREMENT_ID`（当前已有 placeholder 逻辑，确认其不被硬编码进 git）

**验收**：
```bash
# build 后检查输出 HTML 不含 Webhook URL 明文
grep -r "a32b3ab7" playbook/ | wc -l
# 输出 0
# 有环境变量时功能正常
P2S_FEISHU_WEBHOOK_URL="https://..." python3 build_playbook.py --root . --vault paper2skills-vault --out playbook
grep -r "a32b3ab7" playbook/ | wc -l  # 这次允许有（来自env）
```

---

## T4：p2s-service 源码入库

**问题**：FastAPI 后端服务直接存在服务器 `/opt/paper2skills/service/app.py`，无版本控制，无本地副本，一次服务器故障即永久丢失。

**执行步骤**：

1. 从服务器拉取源码：
   ```bash
   scp -i ai_video.pem ubuntu@101.34.52.232:/opt/paper2skills/service/app.py \
     /Users/lute/project/paper_to_skills/paper2skills-services/app.py
   scp -i ai_video.pem ubuntu@101.34.52.232:/opt/paper2skills/service/daily_inspect.sh \
     /Users/lute/project/paper_to_skills/paper2skills-services/daily_inspect.sh
   ```

2. 创建服务目录结构：
   ```
   paper2skills-services/
   ├── app.py                 # FastAPI 主服务
   ├── daily_inspect.sh       # cron 巡检脚本
   ├── requirements.txt       # fastapi, uvicorn, httpx 等
   ├── .env.example           # 环境变量示例（不含真实值）
   └── README.md              # 部署 SOP
   ```

3. 检查 `app.py` 中的硬编码敏感信息（API Key、Secret），替换为 `os.environ.get()`，并把真实值加入服务器 `/opt/paper2skills/service/.env`（不入库）

4. 写 `paper2skills-services/README.md`，包含：
   - 本地运行：`uvicorn app:app --reload --port 8765`
   - 服务器部署：`systemd` 服务名 `p2s-service`
   - 环境变量列表
   - 接口文档（3 个端点 + 请求/响应格式）

5. 更新 `AGENTS.md` 的「服务器后端服务」章节，改为指向 `paper2skills-services/`

**验收**：
```bash
ls /Users/lute/project/paper_to_skills/paper2skills-services/
# 应有 app.py、requirements.txt、README.md
python3 -c "import ast; ast.parse(open('paper2skills-services/app.py').read()); print('✅ syntax OK')"
```

---

## 执行顺序

```
T1（修复数字）→ T3（Webhook 安全）→ rebuild 验证 → T2（清理目录）→ T4（入库服务）
```

T1 和 T3 同属 build_playbook.py 修改，合并到一次 build 验证。T2 纯文件操作，不影响构建。T4 独立进行。

## 完成标志

- [x] `maturity-report.html` 显示真实 skill/edge 数字
- [x] 根目录无 `.png` 无临时 `.py` 脚本
- [x] `playbook/agents.html` 源码中无 `a32b3ab7` 字符串
- [x] `paper2skills-services/app.py` 存在且语法正常
- [x] `git status` 显示合理的变更（无 venv/、无 archive/）
