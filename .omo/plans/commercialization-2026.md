# paper2skills 商业化落地执行计划 — 2026-06-24

## 数据基线（2026-06-24）
- Skills: **1037** | Domains: **25** | Edges: **11,758**（build-report）
- Agent: **15+**（前端模拟，无真实数据接入）
- 商业化就绪度: **~5.5/10**

## 核心差距（按优先级）
1. 图谱推理引擎缺失 — 只能展示，无法推理路径
2. Agent 执行停留在模拟层 — 无真实 API 接入
3. 用户行为数据空白 — 无埋点，无用量感知
4. 诊断工具未 Web 化 — diagnose.py 无法在浏览器运行
5. 变现链路缺失 — 无 Freemium，无账户体系
6. RAG 无上下文感知 — 单轮检索，无用户画像

---

## Phase 1 — 基础设施补强（目标：2周内完成）

### P1-T1：用户行为埋点 + 热力图
- 在 build_playbook.py 的 render_skill_page() 里注入 GA4 事件
- 埋点目标：Skill 点击 / Workflow Step 完成 / Agent 运行 / Playbook 完成
- 用 gtag('event', ...) 写入 dataLayer，无需外部服务
- 验收：build 后 skill 页面 HTML 含 gtag 事件代码

### P1-T2：诊断工具 Web 化（diagnostic.html 真正可运行）
- 当前 diagnostic.html 只展示 Skill 链，diagnose.py 逻辑在 Python 侧
- 目标：在 diagnostic.html 页面内用 JS 实现简版诊断逻辑
  - 用户选择：SKU类型 + 问题类型 → 输出三层诊断报告 + Skill直达链接
  - 6 种问题类型对应的诊断树逻辑（repurchase_drop / roas_decline / traffic_drop / new_cvr_low / inventory_issue / review_attack）
- 验收：浏览器内点击触发诊断，输出结构化报告

### P1-T3：图谱边数不一致修复
- build-report.json 显示 11,758 边，AGENTS.md 显示 17,913
- 找到 build_playbook.py 里 edge 计数逻辑，统一来源
- 验收：build 后 build-report.json edge 与 graph/overview.html 显示一致

### P1-T4：chat.html RAG 多轮增强
- 增加用户角色选择（运营 / 数据分析师 / CEO）影响 system prompt 深度
- 在 system prompt 里注入角色上下文，使回答深度随角色变化
- 验收：选择不同角色后，相同问题的回答详略明显不同

---

## Phase 2 — 图谱推理引擎（目标：3-6周）

### P2-T1：Skill 路径规划 API
- 在 build_playbook.py 生成 /assets/skill-graph.json（节点+边+权重）
- 前端 JS 实现图上路径推荐：给定业务问题 → 自动规划 Skill 执行路径
- 算法：BFS + 业务价值权重，输出 Top-3 路径
- 验收：diagnostic.html 或新页面能展示"解决XX问题的推荐 Skill 路径"

### P2-T2：Agent 数据接入层（POC）
- 为「供应链哨兵」Agent 接入飞书多维表格 API（最低门槛的真实数据源）
- Agent 结果可写回飞书表格（创建新行记录诊断结果）
- 验收：真实运行一次 Agent，结果出现在飞书表格

### P2-T3：Freemium 门槛设计
- 设计付费墙逻辑：免费用户 5 个 Domain，付费解锁全部 + Agent 运行次数
- 在 build_playbook.py 里加入 domain_tier 字段区分 free/pro
- 验收：build 后 free tier Domain 页面有升级 CTA

---

## Phase 3 — 商业化扩张（目标：6-12周）

### P3-T1：知识图谱 API 化
- 暴露 REST API：GET /api/skills/:id/related
- 使外部系统可查询 Skill 关系

### P3-T2：企业知识库定制
- 允许上传私有数据生成企业专属 Playbook

### P3-T3：内容生产自动化
- ArXiv API 定时拉取 → LLM 初筛 → 草稿生成
- 人工审核比例从 ~50% → ~10%

---

## 执行序列（严格按序）

```
Week 1: P1-T3（边修复）→ P1-T1（埋点）→ P1-T2（诊断Web化）
Week 2: P1-T4（RAG增强）→ build + deploy 验证
Week 3-4: P2-T1（路径规划）
Week 5-6: P2-T2（Agent接入POC）+ P2-T3（Freemium设计）
```
