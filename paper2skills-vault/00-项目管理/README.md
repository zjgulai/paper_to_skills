# 项目管理

> **最新更新**：script-derived governance baseline

## 进度追踪
- [Week 1] 建系统（Day 1-7）✅
- [Week 2] 建标准（Day 8-14）✅
- [Week 3] 新领域扩展（VOC + 6 个业务垂类）✅
- [Week 4-5] MAS 系统化与三轮迭代萃取（2026-05-17）✅
- [Sprint 3-5] 220 个 Skill / 22 个领域 / 5+6 Sprint 完成（2026-05-25）✅
- [技术债清零] MCP Server + 图谱断链修复 + 环境治理（2026-06-01）✅

## 当前状态
- **Skill 总数**：**302** / 图谱边数：**5390** / P0/P1 缺口：**0/0** / P2 复核项：**9**
- **MAS MVP**：5 工作流 / 标准库测试由 `python3 -m pytest` 验证 / MCP Server 4 个 domain server
- **工具注册**：112 个工具 / 14 个域
- **累计 ROI**：12000-22000 万元/年潜在（中型品牌）

> 计数类状态不再手动维护为权威来源。请用 `doctor`、`skills_graph_analyzer.py --gaps`、`build_asset_inventory.py --dry-run` 重建。

## 核心文档导航

### 状态与复盘
- [`项目总结.md`](./项目总结.md) — **首先阅读**,整体里程碑与最新数据
- [`进度追踪.md`](./进度追踪.md) — 按周追踪的任务清单
- [`sprint1-2-iteration-report-20260517.md`](./sprint1-2-iteration-report-20260517.md) — Sprint 1+2 完整复盘
- [`6h-iteration-report-20260517.md`](./6h-iteration-report-20260517.md) — 6h 迭代复盘

### 下一步规划
- [`next-papers-roadmap.md`](./next-papers-roadmap.md) — Sprint 3 P1 候选(6 个 Skill)

### 图谱与结构
- [`Skill关联图谱.md`](./Skill关联图谱.md) — 图谱可视化 + 缺口热力图
- [`知识图谱架构与分类体系.md`](./知识图谱架构与分类体系.md)
- [`知识图谱操作手册.md`](./知识图谱操作手册.md)
- [`知识图谱速查卡.md`](./知识图谱速查卡.md)

### 历史方案
- [`Phase1落地计划.md`](./Phase1落地计划.md) — 早期 MAB + Demand Forecasting 落地方案
- [`audits/`](./audits/) — 历史审计报告(KG 自动构建/Dense Retrieval)

## 常用工具索引

| 工具 | 路径 |
|---|---|
| 项目健康检查 | `python3 -m paper2skills_common.doctor --json` |
| Paper 候选队列 | `paper2skills-skills/paper-选题/scripts/build_candidate_queue.py` |
| 增量 workflow driver | `paper2skills-skills/paper-workflow/scripts/run_incremental_workflow.py` |
| Darwin 20-loop 进化 | `paper2skills-skills/paper-workflow/scripts/run_darwin_evolution.py` |
| Skill 图谱分析 | `paper2skills-skills/paper-skills-graph/scripts/skills_graph_analyzer.py` |
| Skill 别名映射表 | `paper2skills-vault/07-资源库/skill-aliases.json` |
| sync_status 重建 | `paper2skills-skills/paper-同步/scripts/rebuild_sync_status.py` |
| 孤立 Skill 回填 | `paper2skills-skills/paper-skills-graph/scripts/backfill_skill_relations.py` |
| MAS 系统入口 | `mas/main.py` + `mas/README.md` |
