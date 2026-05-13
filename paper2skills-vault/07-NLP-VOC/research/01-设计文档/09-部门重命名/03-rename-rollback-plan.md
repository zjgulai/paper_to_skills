---
plan_id: voc-dept-rename-2026-05-13
status: AWAITING_GO
related: 02-rename-execution-plan.md
---

# 部门重命名 · 回滚预案 · 2026-05-13

## 设计原则

每一层都必须**独立、单次命令**回滚。不允许「需要先做 X 再做 Y 才能回滚」。

## 备份清单（执行前一次性产出）

| 备份 | 路径 | 用途 |
|---|---|---|
| dim_tag 当前数据 | `~/.secrets/backups/voc-dept-rename/dim_tag_<ts>.csv` | L8 PG 回滚 |
| Superset metadata | `~/.secrets/backups/voc-dept-rename/superset_<ts>.db` | L9 重装失败回滚 |
| 8 个旧 dashboard ZIP | `~/.secrets/backups/voc-dept-rename/superset_exports_<ts>/` | L9 重装失败回滚 |
| 字典 v4.1 | 仓库内 `01-字典版本/tag_dictionary_v4.1.xlsx`（不删）| L1 回滚 |
| git working tree | git commit hash before any change | L2-L7 回滚 |

> 所有备份不入 git，统一放本地 `~/.secrets/backups/voc-dept-rename/<timestamp>/`。

## 分层回滚命令

### L1 字典 v4.2

```bash
# 删除新生成的 v4.2，v4.1 完全不动
rm -f paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/tag_dictionary_v4.2.xlsx
rm -f paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/dept_rename_v42_diff.md
```

**风险**：无。v4.1 在生产生效，v4.2 是新文件。

### L2 字典生成器

```bash
git checkout HEAD~1 -- paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/tag_dictionary_v3{8,9}_generator.py
```

或更安全：

```bash
git revert <commit-of-L2>
```

### L3 BI 工厂脚本

```bash
git checkout HEAD~1 -- paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/docker/superset_*_factory.py
```

**注**：L3 回滚后**不要**重跑生产 factories（否则会把新名字打回老名字）。L9 已重装的话需要按 L9 回滚先。

### L4 SQL 视图

```bash
git checkout HEAD~1 -- paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/sql/voc_bi_views.sql
```

**注**：仅注释变更，无数据风险。

### L5 报告生成器 + yaml

```bash
git checkout HEAD~1 -- paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/{bi_dashboard_generator,maa_strategy_generator}.py
git checkout HEAD~1 -- paper2skills-vault/07-NLP-VOC/research/04-输出结果/05-设计方案/week1-2_P0_*.yaml
```

### L6 文档（45 份 .md）

```bash
git checkout HEAD~1 -- $(cat /tmp/voc-dept-scan/06_md_files.txt)
```

或全量：

```bash
git revert <commit-of-L6>
```

### L7 演讲 HTML + PDF

```bash
# 文件回滚
git checkout HEAD~1 -- paper2skills-vault/07-NLP-VOC/presentation/

# PDF 重新生成（如已 git revert 则自动用旧 HTML）
cd paper2skills-vault/07-NLP-VOC/presentation
node scripts/export-pdf.mjs
```

### L8 PG dim_tag UPDATE

```sql
-- 方式 1: 事务还在执行中（未 COMMIT）
ROLLBACK;

-- 方式 2: 已 COMMIT，从备份 CSV 恢复
TRUNCATE dim_tag;
\copy dim_tag FROM '/tmp/dim_tag_backup_<ts>.csv' CSV HEADER;
SELECT dept_owner, COUNT(*) FROM dim_tag GROUP BY 1;
```

### L9 Superset 重装

```bash
# 9a. 停容器
cd /opt/voc-superset
sudo docker compose stop voc_superset

# 9b. 还原 metadata sqlite（恢复旧 8 dashboard 配置）
sudo docker cp ~/.secrets/backups/voc-dept-rename/superset_<ts>.db \
  voc_superset:/app/superset_home/superset.db

# 9c. 启容器
sudo docker compose start voc_superset

# 9d. 清 Redis
sudo docker exec voc_superset_redis redis-cli FLUSHDB

# 9e. 浏览器验证 dashboard 名字回到旧名
```

或更暴力：从 `~/.secrets/backups/voc-dept-rename/superset_exports_<ts>/` 重新导入 8 个 ZIP。

## 全栈回滚（核选项）

如果 L1-L9 任何一步执行后发现**多层失败需要全部回滚**：

```bash
# 1. 回滚生产数据（L8）
ssh voc-prod
sudo docker exec voc_bi_pg psql -U voc_bi -d voc_bi -c "
  TRUNCATE dim_tag;
  \\copy dim_tag FROM '/tmp/dim_tag_backup_<ts>.csv' CSV HEADER;
"

# 2. 回滚生产 Superset（L9）
sudo docker compose stop voc_superset
sudo docker cp ~/.secrets/backups/voc-dept-rename/superset_<ts>.db \
  voc_superset:/app/superset_home/superset.db
sudo docker compose start voc_superset

# 3. 回滚本地代码（L1-L7）
cd /Users/pray/project/paper_to_skills
git checkout main
git branch -D chore/dept-rename-2026-05-13
```

**RTO（Recovery Time Objective）**：~5 分钟。

## 验证回滚成功的 checklist

- [ ] `git status` clean，分支回到 main
- [ ] 仓库内 ripgrep 老部门名命中数恢复到执行前的基线
- [ ] 腾讯云 PG `dim_tag` distinct 部门 = 执行前的 10 个值（含「全球营销服务中心」「营运部」）
- [ ] Superset 浏览器登录后 8 个 dashboard 标题为旧名
- [ ] Playwright 端到端测试通过（dashboard 3 polarity 切换 Top-3 还是「质量感知 / 易用性 / 延迟」）

## 风险登记

| 风险 | 缓解 |
|---|---|
| L8 UPDATE 跑过头但忘备份 | L0 备份 CSV 是硬约束，跑 UPDATE 前先 ls 确认存在 |
| L9 重启 Superset 时配置错乱 | 备份 superset.db + 备份 8 个旧 ZIP，二选一恢复 |
| git revert 引入冲突 | 单层回滚优先用 `git checkout HEAD~1 -- <path>` 不动其他文件 |
| 公网访问者命中半改状态 | L1-L7 全过 + commit 后再做 L8/L9，公网 BI 切换在最后一步 |

## 不会做的事（明确边界）

- ❌ 删除 v4.1 字典
- ❌ DROP 任何表
- ❌ 改 git 历史（rebase / force-push）
- ❌ 修改其他 ai-video 项目的容器（同 docker host 上）
- ❌ 改 nginx 配置

---

## 执行 GO 之前必须确认的清单

- [ ] 用户已审阅 `01-rename-canonical-mapping.md`
- [ ] 用户已审阅 `02-rename-execution-plan.md`
- [ ] 用户已审阅 `03-rename-rollback-plan.md`（本文档）
- [ ] 用户已就 E1/E2/E3 三项边缘值给出决策
- [ ] L0 备份完成，文件存在 `~/.secrets/backups/voc-dept-rename/<ts>/`
- [ ] git 工作树 clean，已建 `chore/dept-rename-2026-05-13` 分支
