# 会话摘要：商业化基础设施建设 — 2026-06-24/25

## 本轮会话做了什么

这轮会话从「分析商业化差距」出发，完整执行了 Phase 1 基础设施补强 + 飞书双向集成，把 paper2skills 从「静态知识展示站」推进到「具有数据收集能力和飞书工作流集成的决策基础设施」。

---

## 核心交付（全部已上线 https://skills.lute-tlz-dddd.top）

### 1. 知识图谱修复
- **边数不一致问题**：`build-report.json` 的 edges 之前是全量原始边（含悬挂边），与 `graph-data.json` 实际渲染的有效边不一致。修复后统一为**有效边**（两端节点均存在），当前 11,577 条，两处数字完全一致。
- **修改位置**：`build_playbook.py` 的 `render_pages()` 函数，先计算 `valid_links`，再赋值给 `edge_count`。

### 2. GA4 全站埋点
- **Measurement ID**：`G-N9HJR3G0MR`（在 `build_playbook.py` 顶部 `GA4_MEASUREMENT_ID` 常量管理）
- **埋点事件**：`skill_view`（每个 Skill 页面加载）、`skill_code_copy`（点击复制代码按钮）、`agent_run`（Agent 开始运行）
- **激活条件**：ID 不以 `G-X` 开头时自动注入，placeholder 状态自动跳过。

### 3. chat.html 角色分层
- topbar 加入角色下拉：**运营视角 / 数据分析师 / CEO 战略**
- 选择后动态注入 system prompt 角色指令，使回答深度随角色变化。
- 位置：`render_chat_page()` HTML + `build_chat_page_js()` JS 逻辑。

### 4. diagnostic.html BFS Skill 路径规划器
- 侧边栏「Skill 路径规划」折叠区，选起点+终点 Skill，前端 JS BFS 遍历 `graph-data.json` 的 11,577 条边。
- 输出：步骤链 + 边类型标注（前置/延伸/可组合）+ Skill 直达链接。
- 懒加载：展开时才 fetch graph-data.json，不影响首屏性能。

### 5. Freemium 门槛 CTA
- 首页（`index.html`）hero 下方加升级 banner。
- 领域首页（`domains/index.html`）顶部加升级 banner。
- 定位：免费5域 / 企业版25域，联系邮件 `skills@lute-tlz-dddd.top`。

### 6. 飞书 Webhook 集成（单向推送）
- **Webhook URL**：`https://open.feishu.cn/open-apis/bot/v2/hook/a32b3ab7-6cfb-498d-bc3f-91d9f48b47e9`
- 在 `build_playbook.py` 顶部 `FEISHU_WEBHOOK_URL` 常量管理。
- Agent 运行后自动调用 `pushToFeishu()`，推送用户真实输入 + DeepSeek 分析结果到飞书群。
- 推送格式：`plain_text`（不用 lark_md），推送前清理 `**` `# ` 等 markdown 符号。

### 7. Agent 系统全面升级（21个，全部真实 AI）
- **默认模式**：从「本地演示」改为「AI 真实分析」（`value='ai' checked`）。
- **全量覆盖**：模式切换控件从固定12个 ID 改为动态读 `AGENT_CATALOG`，覆盖全部21个。
- **结构化 prompt**：18个已有 Agent 的 system prompt 全部重写，统一用【】标题格式，要求输出具体数字。
- **3个新 prompt**：`agent-dml-counterfactual-pricing`、`agent-cold-start-advisor`、`agent-festival-replenishment` 补全。
- **中文字段名**：新增 `getInputsLabeled()` 函数，从 DOM label 取中文字段名传给 DeepSeek，替代原来的英文 field id。
- **AI 模式飞书推送**：`runAgentAI` 里直接传真实 `inp` 给 `pushToFeishu`，绕过 DEMO_DATA 的演示输入。
- **ADISP 补全**：3个新 Agent 中文名加入 `ADISP` 字典。

### 8. nginx skills.conf 独立文件（重要基础设施修复）
- **根因**：skills server block 原来写在主 nginx.conf 里，每次其他服务部署都会覆盖 nginx.conf 导致 skills 域名失效。
- **修复**：创建独立文件 `/opt/ai-video/deploy/lighthouse/skills.conf`，主 nginx.conf 末尾 `include /etc/nginx/skills.conf`，docker-compose 独立挂载。
- **好处**：其他服务部署不再影响 paper2skills，skills.conf 只由 paper2skills 维护。

### 9. Playbook 进度推飞书按钮
- 每个 Playbook 页面步骤区上方加「推送进度到飞书」绿色按钮。
- 点击后统计当前已完成步骤数，推送格式：`手册名 | 进度 N/M | 时间`。
- 调用 `/api/feishu-callback` 接口（非直接 Webhook，经过回调服务记录）。

### 10. p2s-service 后端服务（FastAPI 8765）
- **文件**：`/opt/paper2skills/service/app.py`
- **systemd**：`p2s-service.service`，开机自启，崩溃自动重启。
- 提供三个接口：
  - `POST /api/feishu-callback`：处理飞书卡片「确认执行」/ 「深度分析」按钮回调
  - `POST /api/feishu-event`：接收飞书事件订阅（文档分享 → Skill 草稿，待飞书开放平台配置）
  - `POST /api/daily-inspect`：接收 SKU 巡检数据，异常推飞书（secret: `p2s_inspect_2026`）

### 11. 每日自动巡检 cron
- **脚本**：`/opt/paper2skills/service/daily_inspect.sh`
- **触发时间**：每天 09:00（`0 9 * * * /opt/paper2skills/service/daily_inspect.sh`）
- **逻辑**：传入 SKU 列表 → DOS < 30 或 ACOS > 40% 触发异常 → 调 DeepSeek 生成优先级建议 → 推飞书。
- **修改 SKU 列表**：编辑脚本的 `SKUS_JSON` 变量，或改造成从飞书多维表格拉取。

---

## 关键常量一览（下次开发必看）

```python
# build_playbook.py 顶部
GA4_MEASUREMENT_ID = "G-N9HJR3G0MR"
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/a32b3ab7-6cfb-498d-bc3f-91d9f48b47e9"

# p2s-service 巡检 secret
secret = "p2s_inspect_2026"

# nginx 代理配置文件（独立，不随其他服务覆盖）
/opt/ai-video/deploy/lighthouse/skills.conf

# 后端服务
http://10.0.0.16:8765  # nginx container 内访问宿主机的地址
```

---

## 待办（下次会话直接开始）

### 高优先级
1. **飞书多维表格 → Agent 自动触发**
   - 在飞书多维表格建库存/广告数据表
   - 用飞书原生「自动化」规则：`当 DOS < 30` → HTTP 请求 → `POST /api/daily-inspect`
   - Body 模板：`{"secret":"p2s_inspect_2026","skus":[{"name":"{{SKU名}}","dos":{{DOS}},"acos":{{ACOS}}}]}`
   - **无需改代码，纯飞书配置**

2. **飞书文档 → Skill 草稿生成（方向5）**
   - 进飞书开放平台，创建企业自建应用
   - 事件订阅 → `im.message.receive_v1`
   - 请求 URL：`https://skills.lute-tlz-dddd.top/api/feishu-event`
   - 完成后群里分享文档 URL → 机器人自动生成 Skill 草稿

### 中优先级
3. **每日巡检 SKU 列表接入真实数据**：把 `daily_inspect.sh` 里的硬编码 `SKUS_JSON` 改成从飞书多维表格 API 拉取
4. **飞书卡片「确认执行」写回多维表格**：在 `app.py` 的 `confirm` 分支加飞书多维表格写入
5. **Agent 报告持久化升级**：当前用 localStorage，考虑写入服务端（SQLite 或飞书多维表格）

---

## 遇到的坑（下次避免）

| 坑 | 根因 | 解决方案 |
|----|------|---------|
| nginx.conf 被其他服务部署覆盖 | 所有服务共用一个 nginx.conf | skills 配置独立为 `skills.conf`，include 引入 |
| /api/agent 返回 404 | skills server block 丢失导致 `/api/agent` 路由消失 | skills.conf 独立后彻底解决 |
| 飞书推送有 `\n` `**` 符号 | 用了 lark_md 但内容有 markdown 符号 | 改 plain_text + 推送前 `.replace(/\*\*/g,'')` |
| build-report.json edges 与图谱不一致 | 用了全量边（含悬挂边）计数 | 改为先过滤有效边，再计数 |
| getInputs 用英文 field id | DOM id 是 field id 不是 label | 新增 `getInputsLabeled()` 从 label 元素取中文名 |
| location 块追加在 server block 外 | cat >> 追加到文件末尾 server 闭合括号后 | 用 cat > 整体重写配置文件 |

---

## 当前产品形态描述

paper2skills 现在是一个**飞书原生的母婴跨境电商 AI 决策系统**：

- **知识层**：1037 个从顶会论文萃取的 Skill 卡片，25 个领域，11,577 条知识关联边
- **诊断层**：`diagnostic.html` 支持症状→Skill链匹配 + BFS路径规划，17个风险事件 Ontology
- **执行层**：21 个真实 AI Agent（DeepSeek 驱动），有结构化输出 prompt，支持角色分层对话
- **集成层**：飞书双向集成（Agent 推送 + Playbook 进度 + 每日巡检），p2s-service 8765 端口
- **商业层**：GA4 全站埋点（skill_view/code_copy/agent_run），Freemium CTA（5域免费/企业版25域）
