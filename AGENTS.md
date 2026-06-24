# AGENTS.md

This file provides guidance to AI agents (OpenCode/Codex) working with this repository.

## Project Overview

**paper2skills** — 将顶刊学术论文转化为可落地的母婴跨境电商 AI 决策 Skill 卡片系统。
4 步流水线：论文筛选 → Skill 萃取 → 质量审核 → 同步发布。

**Live Playbook**: https://skills.lute-tlz-dddd.top

---

## Current State (2026-06-22)

| Metric | Value |
|--------|-------|
| Skill pages | **931** |
| Domains | **25** |
| Graph edges | **16,929** |
| Agents | **15+** callable agents (local compute engines) |
| Solutions | **12** published |
| ps_override entries | **931** (100% coverage) |
| Playbooks | **33** scene handbooks |
| Workflows | **16** business workflows |
| Risk Ontology | **10 events / 131 Skill mappings** (diagnostic.html) |

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
│           ├── build_playbook.py              # ⚠️ CSS 由此函数生成，不要直接改 playbook/assets/style.css
│           └── config/
│               ├── skill_ps_override.yaml     # problem_solved 覆盖（931条，100%覆盖）
│               ├── risk_events_ontology.yaml  # ⚠️ 风险事件Ontology（10事件/131映射）
│               ├── skill_biz_context_override.yaml
│               └── skill_handbook_map.yaml
├── paper2skills-vault/
│   ├── 01-因果推断/    (20 skills)
│   ├── 02-A_B实验/     (20 skills)
│   ├── 03-时间序列/    (23 skills)
│   ├── 04-供应链/      (124 skills) ← 最大域
│   ├── 05-推荐系统/    (27 skills)
│   ├── 06-增长模型/    (49 skills)
│   ├── 07-NLP-VOC/     (31 skills)
│   ├── 08-知识图谱/    (51 skills)
│   ├── 09-DataAgent-LLM/ (20 skills)
│   ├── 10-MAS/         (65 skills)
│   ├── 11-AI人文/      (20 skills)
│   ├── 12-ML基础/      (20 skills)
│   ├── 13-广告分析/    (47 skills)
│   ├── 14-用户分析/    (48 skills)
│   ├── 15-营销投放分析/ (37 skills)
│   ├── 16-智能体工程/  (66 skills)
│   ├── 17-价格优化/    (36 skills)
│   ├── 18-物流履约/    (20 skills)
│   ├── 19-风控反欺诈/  (23 skills)
│   ├── 20-AI视频生成/  (23 skills)
│   ├── 21-合规决策/    (28 skills)
│   ├── 22-数据采集工程/ (25 skills)
│   ├── 23-运营财务/    (27 skills)
│   ├── 24-标签工程/    (61 skills) ← 新域，Palantir本体论+决策自动化
│   ├── 25-搜索流量工程/ (20 skills) ← 新域
│   └── 07-资源库/
│       ├── MasterPrompt.md
│       ├── 关键词库.md
│       ├── sync_status.json
│       └── SC-TagToDecision-Architecture.md  ← 供应链全链路架构方案文档
└── paper2skills-code/  # Python 代码模板（各域子目录）
```

---

## Key Files

| File | Purpose |
|------|---------|
| `paper2skills-vault/07-资源库/MasterPrompt.md` | 论文转 Skill 的核心 Prompt |
| `paper2skills-vault/07-资源库/关键词库.md` | ArXiv 搜索关键词分域索引 |
| `paper2skills-vault/07-资源库/SC-TagToDecision-Architecture.md` | 供应链「标签→决策」完整架构方案 |
| `paper2skills-skills/playbook-generator/scripts/build_playbook.py` | **Playbook 构建脚本（CSS 源头在这里）** |
| `paper2skills-skills/playbook-generator/scripts/config/skill_ps_override.yaml` | problem_solved 语句覆盖 |
| `paper2skills-skills/paper-同步/scripts/sync.py` | 多平台同步脚本 |

---

## ⚠️ 关键工程注意事项

### CSS 修改必须改 `build_playbook.py`

`playbook/assets/style.css` 由 `build_playbook.py` 里的 `build_css()` 函数生成。
每次 build 都会覆盖该文件。**直接改 style.css 无效**，必须改 `build_css()` 函数。

```python
# build_playbook.py 第 4300 行
def build_css() -> str:
    return """...(完整 CSS 字符串)..."""
```

### Solutions 页面有内嵌 `<style>`

`render_solutions_index()` 和 `render_solution_detail()` 函数在 HTML body 里内嵌了独立的 `<style>` 块。
这些样式**独立于全局 CSS**，需要在 Python 函数里同步修改（不随 build_css() 更新）。

### SOLUTIONS_CATALOG 扩展方式

在 `build_playbook.py` 的 `SOLUTIONS_CATALOG` 列表里追加一个 dict，rebuild 后自动生成首页卡片 + 详情页，不需要手写 HTML。

---

## Playbook Pages

| Page | URL | Description |
|------|-----|-------------|
| `/` | index.html | 总览仪表盘 |
| `/diagnostic.html` | 业务诊断中心 | 症状→Skill链 / 风险事件Ontology（新增） |
| `/chat.html` | AI知识库对话 | DeepSeek RAG + 症状路由 + 对话历史持久化 |
| `/playbooks/` | 场景手册 (33个) | 可执行业务手册 + 进度追踪 |
| `/solutions/` | 方案库 (12个) | 系统架构方案 |
| `/agents.html` | 智能体广场 | 15+ 本地计算 Agent |
| `/agent-report.html` | 智能体报告 | localStorage 持久化报告 |
| `/ai-roadmap.html` | AI 能力路线图 | CEO-facing 白皮书 |
| `/skills/` | 全部 Skills (931个) | 每 Skill 独立详情页 |
| `/domains/` | 25 个领域页 | 按领域浏览 |
| `/graph/overview.html` | 技能关系图谱 | D3 可视化 |
| `/workflows/` | 业务工作流 (16个) | 场景 → Skill 的执行链路 |

---

## Build & Deploy SOP

```bash
# 1. Build（验证无 WARN dup_ps）
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . --vault paper2skills-vault --out playbook 2>&1 | tail -8

# 2. 打包（用 *.html 通配符确保所有顶层页面都包含）
cd playbook && tar -czf /tmp/pb.tar.gz \
  assets/ domains/ graph/ playbooks/ topics/ workflows/ skills/ solutions/ \
  *.html \
  build-report.json README.md

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

## Agent Marketplace

15+ callable agents，分类如下：

| 分类 | Agent | 核心输入 |
|------|-------|---------|
| 供应链 | 供应链哨兵 / SKU标签质量扫描器 | 库存/速度/前置期 |
| 定价 | 动态定价顾问 | price, cost, BSR |
| 财务 | P&L透视镜 | revenue, cogs, fba |
| 广告 | 广告归因侦探 | platform, spend |
| 竞品 | 竞品雷达站 | asins, period |
| 运营 | Listing医生 / VOC解码器 / 客服分诊台 | text inputs |
| 风控 | 账号风险卫士 / 品牌合规卫士 | notice/copy |
| 标签工程 | SKU标签质量扫描器 | SKU list, market |
| 合规 | 多市场合规矩阵 | product, markets |
| 逆向物流 | 退货根因分析师 | return data |

每次 Agent 运行自动保存到 `localStorage('agentReports')`，在 `agent-report.html` 查看。

---

## Solutions（方案库）

位置：`playbook/solutions/`，在侧边导航「◆ 方案库」入口。

| ID | 标题 | 状态 |
|----|------|------|
| `sol-sc-tag-to-decision` | 供应链标签工程→决策全链路架构 | ✅ Published |
| `sol-voc-product-selection` | VOC→选品决策全链路 | ✅ Published |
| `sol-ads-organic-synergy` | 广告-自然流量协同增长架构 | ✅ Published |
| `sol-inventory-risk-management` | 库存风险管理架构 | ✅ Published |
| `sol-viral-growth-engine` | 病毒增长引擎架构 | ✅ Published |
| `sol-counterfactual-pricing` | 反事实定价系统 | ✅ Published |
| （+6 more） | ... | ✅ Published |

**扩展方式**：在 `build_playbook.py` 的 `SOLUTIONS_CATALOG` 添加 dict → rebuild 自动生成。

**架构文档**：`paper2skills-vault/07-资源库/SC-TagToDecision-Architecture.md`

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
| 01-因果推断 | Causal inference, uplift modeling | 20 |
| 02-A_B实验 | A/B testing, multi-armed bandits | 20 |
| 03-时间序列 | Demand forecasting | 23 |
| 04-供应链 | Inventory, SC optimization | 124 |
| 05-推荐系统 | Recommendation systems | 27 |
| 06-增长模型 | Churn, LTV, growth | 49 |
| 07-NLP-VOC | Sentiment, VOC mining | 31 |
| 08-知识图谱 | KG, GNN, hyperbolic | 51 |
| 09-DataAgent-LLM | Data agent, LLM analytics | 20 |
| 10-MAS | Multi-agent systems | 65 |
| 11-AI人文 | AI ethics, AIGC | 20 |
| 12-ML基础 | ML fundamentals | 20 |
| 13-广告分析 | Ad attribution, ROAS | 47 |
| 14-用户分析 | Funnel, cohort, RFM | 48 |
| 15-营销投放分析 | MMM, promo effectiveness | 37 |
| 16-智能体工程 | LLM Agent engineering | 66 |
| 17-价格优化 | Dynamic pricing | 36 |
| 18-物流履约 | Cross-border logistics | 20 |
| 19-风控反欺诈 | Fraud detection | 23 |
| 20-AI视频生成 | Virtual anchor, brand video | 23 |
| 21-合规决策 | Compliance decisions | 28 |
| 22-数据采集工程 | Data collection, quality | 25 |
| 23-运营财务 | FBA finance, P&L, tariff | 27 |
| 24-标签工程 | Tag engineering, Palantir ontology, action triggers | 61 |
| 25-搜索流量工程 | Search traffic, SEO, A9 algorithm | 20 |
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
