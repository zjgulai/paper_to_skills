# 项目管理

> **最新更新**:2026-05-17 下午(Sprint 1+2 完成,107 个 Skill)

## 进度追踪
- [Week 1] 建系统 (Day 1-7) ✅
- [Week 2] 建标准 (Day 8-14) ✅
- [Week 3] 新领域扩展(VOC + 6 个业务垂类) ✅
- [Week 4-5] MAS 系统化与三轮迭代萃取(2026-05-17) ✅

## 当前状态
- **Skill 总数**:107 / 图谱边数:581 / HIGH 缺口:1(CausalRAG)
- **MAS MVP**:5 工作流 / 37 项集成测试全绿
- **累计 ROI**:6550-13160 万元/年潜在(中型品牌)

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
| Skill 图谱分析 | `paper2skills-skills/paper-skills-graph/scripts/skills_graph_analyzer.py` |
| Skill 别名映射表 | `paper2skills-vault/07-资源库/skill-aliases.json` |
| sync_status 重建 | `paper2skills-skills/paper-同步/scripts/rebuild_sync_status.py` |
| 孤立 Skill 回填 | `paper2skills-skills/paper-skills-graph/scripts/backfill_skill_relations.py` |
| MAS 系统入口 | `mas/main.py` + `mas/README.md` |
