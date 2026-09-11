---
plan_id: voc-dept-rename-2026-05-13
status: AWAITING_GO
related: 01-rename-canonical-mapping.md
---

# 部门重命名 · 7 层执行清单 · 2026-05-13

> 阅读顺序：执行清单 → 回滚预案 → 顶层执行 GO。
> 每一层有「输入 / 动作 / 输出 / 验证 / 失败处置」。前一层全过才进下一层。

## 总流程图

```
L1 字典 v4.2          ─┐
L2 dict generator     ─┤  本地分支提交，单元测试
L3 BI factory         ─┤
L4 SQL views          ─┤
L5 report-gen + yaml  ─┤
L6 文档               ─┤
L7 演讲 HTML          ─┘
                       ↓
L8 腾讯云 PG dim_tag UPDATE
                       ↓
L9 Superset 重跑 factories + 重新导出 ZIP + 重启容器
                       ↓
全栈端到端验证（Playwright + curl + SQL）
```

---

## L0 · 准备工作（执行前一次性）

| 步 | 动作 | 验证 |
|---|---|---|
| 0.1 | git 当前 working tree 是否 clean | `git status` 应无 uncommitted changes |
| 0.2 | 建工作分支 `chore/dept-rename-2026-05-13` | `git checkout -b ...` |
| 0.3 | 备份生产 PG dim_tag | `docker exec voc_bi_pg psql -U voc_bi -d voc_bi -c "\copy dim_tag TO '/tmp/dim_tag_backup_<ts>.csv' CSV HEADER"` 然后 scp 回本地 `~/.secrets/backups/` |
| 0.4 | 备份 Superset metadata sqlite | `docker cp voc_superset:/app/superset_home/superset.db ~/backup/superset_db_<ts>.db` |
| 0.5 | 备份 v4.1 字典 | 已在 `01-字典版本/tag_dictionary_v4.1.xlsx`（不动），新版另存为 v4.2 |

---

## L1 · 字典数据 v4.1 → v4.2

**输入**：`tag_dictionary_v4.1.xlsx`（11 个 sheet）
**动作**：写脚本 `scripts/rename_dept_v42.py`（约 80 行）
- 用 openpyxl 读取
- 对每个 sheet，遍历 `主责部门` / `协同部门` / `业务动作/责任部门` 列
- 应用映射规则（含 Q3=A 复合字符串的「仅前缀替换」逻辑）
- 写出 `tag_dictionary_v4.2.xlsx`，并产出 diff 报告 `dept_rename_v42_diff.md`

**输出文件**：
- 新增：`research/04-输出结果/01-字典版本/tag_dictionary_v4.2.xlsx`
- 新增：`research/04-输出结果/01-字典版本/dept_rename_v42_diff.md`（变更对照）
- **保留** v4.1 不动

**验证**：
- 行数：v4.2 总行数 = v4.1 总行数（无丢失）
- 部门白名单：v4.2 三列里只能出现 D1-D8 + 「未分类」
- diff 报告：每条变更可追溯
- `dictionary_validator.py` 跑过，0 字段错误

**失败处置**：脚本不会动 v4.1 原文件；删除 v4.2 重跑即可。

---

## L2 · 字典生成器代码

**文件**：
- `02-脚本工具/01-标签进化/tag_dictionary_v38_generator.py`
- `02-脚本工具/01-标签进化/tag_dictionary_v39_generator.py`

**动作**：精确替换文件里的 hardcoded 部门字符串（约 30 处）

**验证**：
- `python3 tag_dictionary_v39_generator.py --dry-run` 跑过（如脚本支持），输出含新部门名
- ripgrep 校验：这两个文件不再出现任何 old name
- self-test：`python3 -m py_compile <file>`（保证语法正确）

**失败处置**：git revert 单文件。

---

## L3 · BI 工厂脚本（最关键）

**文件**：
- `02-脚本工具/01-标签进化/docker/superset_charts_factory.py`
- `02-脚本工具/01-标签进化/docker/superset_filters_factory.py`

**关键变更点**：
- `superset_charts_factory.py` line 45-48 的 `DEPARTMENTS = [...]` 列表
- `superset_filters_factory.py` line 43-51 的 `DEPT_DASHBOARDS = [...]` 列表（同时改 dept + chart_name）

**动作**：用 `edit` 工具精确改 7 项

**验证**：
- `python3 -m py_compile` 通过
- ripgrep 校验文件里只有 D1-D8
- `python3 superset_charts_factory.py --print-spec` 干跑（不连 API），看 chart_name 是新名

**失败处置**：git revert 这两文件，腾讯云 Superset 不重跑就还是老配置。

---

## L4 · SQL 视图

**文件**：`02-脚本工具/01-标签进化/sql/voc_bi_views.sql`

**变更点**：注释里 1 处「品牌市场中心」（line 待定，1 处命中）

**注**：视图 SQL 本身不依赖部门字符串（部门是 `dept_owner` 字段值，不是 SQL 关键字），所以仅是注释更新。

**验证**：`psql -h localhost -U voc_bi -d voc_bi -f voc_bi_views.sql --dry-run`（本地）

---

## L5 · 报告生成器 + 规则配置

**文件**（5 个）：
- `bi_dashboard_generator.py`
- `maa_strategy_generator.py`
- `week1-2_P0_aip1_dynamic_anchor_rules.yaml`
- `week1-2_P0_audit_84_pending_tags.yaml`
- `week1-2_P0_fix_20_missing_fields.yaml`

**动作**：精确字符串替换。yaml 里部门是字段值，不是 key，sed 安全。

**特殊**：`maa_strategy_generator.py` 里有 7 部门 dispatch 函数（dept_owner → 部门策略），需要把 dict key 全改。

**验证**：
- `python3 -m py_compile`
- yaml 里的部门只能出现 D1-D8 / 「未分类」 / 「内容电商运营部」/「KOL电商运营部」/「培训部」（保留）

---

## L6 · 文档（45 个 .md）

**策略**：批量 sed 但**人工审阅**（diff 必须可读）

**文件清单**（按目录归类）：

```
07-NLP-VOC/README.md
07-NLP-VOC/CLAUDE.md
07-NLP-VOC/research/01-设计文档/00-Phase5-汇报与复盘/  (8 份)
07-NLP-VOC/research/01-设计文档/06-设计草稿/         (5 份)
07-NLP-VOC/research/01-设计文档/02-工作流设计/        (3 份)
07-NLP-VOC/research/01-设计文档/07-操作指南/          (3 份)
07-NLP-VOC/research/01-设计文档/08-Phase计划/         (4 份)
07-NLP-VOC/research/01-设计文档/03-自动打标调研/      (3 份)
07-NLP-VOC/research/01-设计文档/04-NPS校准方法/       (3 份)
07-NLP-VOC/research/01-设计文档/phase5-bi-dashboard-spec.md
07-NLP-VOC/research/04-输出结果/03-审计报告/         (8 份审计)
07-NLP-VOC/00-知识库-Skill卡片/                       (2 份)
07-NLP-VOC/00-知识库-架构图谱/                        (2 份)
00-项目管理/                                          (1 份)
01-MasterPrompt设计/                                  (1 份)
```

**动作**：用 `scripts/replace_dept_in_docs.sh`（带 git diff 预览模式）

**验证**：
- 全部 ripgrep 校验：仓库内 .md 文件只能出现 D1-D8
- diff 总行数估算 ~500 行
- 抽 5 份文档人工预览（README / CLAUDE / phase5-bi-dashboard-spec / Superset_BI_SOP / phase6-7-executive-brief）

---

## L7 · 演讲 HTML

**文件**（4 个）：

1. `presentation/index.html` — 主线 16 段 + 附录 6 段，部门出现在 S10 mockup-aside 和 A4 mockup-mini-title
2. `presentation/showcase/02-superset-architecture.html` — 4 部门 mockup
3. `research/01-设计文档/00-Phase5-汇报与复盘/phase7-architecture-diagrams.html` — Mermaid 图里的部门名
4. `research/00-归档资料/phase6_html_dashboard/dashboard-2026-W19.html` — 已归档，**只改命中文字 / 不重跑生成器**

**动作**：edit tool 精确替换

**验证**：
- 浏览器打开本地 server 看 mockup 文字正确
- 重新跑 Playwright 全段截图（22 段都要重出）
- 重新导出 PDF（主版 + with-notes 共 2 份）

**特殊**：`/dist/*.pdf` 必须重新生成。

---

## L8 · 腾讯云生产 PG dim_tag UPDATE

**前置条件**：L1-L7 全过 + 已 git commit。

**动作**：通过 SSH 在 voc_bi_pg 里执行：

```sql
BEGIN;

-- 备份当前 distinct 值
\copy (SELECT dept_owner, COUNT(*) FROM dim_tag GROUP BY dept_owner) TO '/tmp/dept_before_<ts>.tsv' CSV HEADER;

-- 应用映射
UPDATE dim_tag SET dept_owner = '全球客服中心'   WHERE dept_owner = '全球客服中心';
UPDATE dim_tag SET dept_owner = '产品中心'       WHERE dept_owner = '产品中心';
UPDATE dim_tag SET dept_owner = '仓储物流部'     WHERE dept_owner = '仓储物流部';
UPDATE dim_tag SET dept_owner = '品质管理中心'   WHERE dept_owner = '品质管理中心';
UPDATE dim_tag SET dept_owner = '法务合规部'     WHERE dept_owner = '法务合规部';
-- E1/E2 等你确认后追加：
-- UPDATE dim_tag SET dept_owner = '品牌市场中心' WHERE dept_owner = '品牌市场中心';
-- UPDATE dim_tag SET dept_owner = '电商运营部'   WHERE dept_owner = '电商运营部';

-- 验证：所有部门只能在新白名单里
SELECT dept_owner, COUNT(*) FROM dim_tag GROUP BY dept_owner ORDER BY 2 DESC;

-- 看到结果后再决定 COMMIT 或 ROLLBACK
-- 默认我会先 ROLLBACK 让你看一遍，再 COMMIT
```

**验证**：
- `SELECT dept_owner` distinct 必须 ⊆ {D1-D8, 「未分类」}
- 行数总和 = 267（不变）

**失败处置**：`ROLLBACK;` 立即回滚，事务原子性保证。

---

## L9 · Superset 重新装配 + 重启

**前置**：L8 已 COMMIT。

**动作**：
```bash
ssh voc-prod
cd /opt/voc-superset

# 9.1 拉最新代码（git pull）
# 9.2 重跑 3 个 factories（删除旧 charts/dashboards/filters，重建）
sudo docker exec voc_superset python3 /opt/voc-superset/superset_charts_factory.py
sudo docker exec voc_superset python3 /opt/voc-superset/superset_filters_factory.py

# 9.3 导出 8 个新 dashboard ZIP（覆盖 superset_exports/）
# 9.4 清 Redis 缓存
sudo docker exec voc_superset_redis redis-cli FLUSHDB

# 9.5 重启 superset 容器（确保拿到新 superset_config.py / charts）
cd /opt/voc-superset && sudo docker compose restart voc_superset

# 9.6 等 healthy
```

**验证**：
- `curl https://voc.lute-tlz-dddd.top/health` → 200
- 浏览器登录后 dashboard list 8 个名字都是新部门
- Playwright 实测 dashboard 3（产品中心）的 polarity filter 切换 Top-3 仍正常（数据没乱）
- SQL Lab 跑 `SELECT dept_owner, count(*) FROM dim_tag GROUP BY 1`，结果 = L8 的最终值

**失败处置**：见回滚预案 L9。

---

## 执行总耗时估算

| 层 | 估算时间 |
|---|---|
| L0 准备 | 5 min |
| L1 字典 v4.2 | 15 min（写脚本 + 跑 + diff review）|
| L2 generator | 5 min |
| L3 factory | 5 min |
| L4 SQL | 2 min |
| L5 report-gen + yaml | 10 min |
| L6 文档 (45 份) | 20 min |
| L7 HTML + PDF 重出 | 15 min |
| L8 PG UPDATE | 5 min |
| L9 Superset 重装 + 验证 | 15 min |
| **合计** | **~95 min** |

---

## 单点失败原则

- L1-L7：本地分支，git revert 即可回滚
- L8：事务保护，ROLLBACK 即可
- L9：保留旧 ZIP 备份，可重导入

**任何一层失败都不允许进入下一层**。
