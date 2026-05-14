---
name: superset-bi-sop
description: Superset BI 看板运维 SOP — 高详细度操作手册。覆盖从零启动、日常访问、添加 chart/filter、迁移重建、故障处置、权限管理、备份恢复的全流程。面向接手的工程/运维同事，逐行命令 + 故障判断树 + 底脱图。v2 (L7) 升级：覆盖 Phase 7 8-dashboard + MVP L4-L6 新增 5 dashboard / 30 chart / 4 dataset，部署在腾讯云 https://voc.lute-tlz-dddd.top。
title: Superset BI 看板运维 SOP
doc_type: sop
module: voc-nlp
topic: superset-bi-operations
status: stable
created: 2026-05-11
updated: 2026-05-14
version: v2
owner: self
source: ai
audience: engineer
---

# Superset BI 看板运维 SOP（v2 / L7 升级）

> **文档定位**：接手人能照着逐行做，不需要去翻 Phase 7 的 4 份进度报告 + MVP L4/L5/L6 三份进度报告。
>
> **v2 关键升级（2026-05-14 / L7）**：
> - 部署地址从 `http://localhost:8088` 升级为腾讯云 `https://voc.lute-tlz-dddd.top`
> - 资产规模：8 dashboard / 12 chart / 6 dataset → **13 dashboard / 42 chart / 10 dataset**
> - 新增覆盖：MVP L4 (D-Health) + L5 (D-Diag 三专题) + L6 (D-Action 7 部门行动队列)
> - 新增 §B「MVP L4-L6 资产清单 + 重建/回滚」专章
>
> **配套文档**：
> - 白话背景：[phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md)
> - 架构图：[phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md)
> - ETL 上游：[ETL_pipeline_SOP.md](ETL_pipeline_SOP.md)
> - MVP 设计：[01-mvp-design.md](../10-VOC深度分析MVP/01-mvp-design.md)
> - L4 进度：[mvp_l4_progress_report.md](../../04-输出结果/03-审计报告/mvp_l4_progress_report.md)（如已归档）
> - L5 进度：[mvp_l5_progress_report.md](../../04-输出结果/03-审计报告/mvp_l5_progress_report.md)
> - L6 进度：[mvp_l6_progress_report.md](../../04-输出结果/03-审计报告/mvp_l6_progress_report.md)
>
> **当前生产配置（2026-05-14）**：
> - Superset 版本：4.1.1（Docker）
> - **生产访问**：`https://voc.lute-tlz-dddd.top`（腾讯云 · HTTPS）
> - 本机开发（仅可选）：`http://localhost:8088`
> - 管理员：`admin / voc_admin_2026`（⚠️ 默认密码，未来 P8 加固时更换）
> - 后端数据库：`voc_bi`（Postgres，腾讯云容器内）
> - 数据规模：364,569 条 review · 1.4M+ labels · ~13 工日累计建设

---

## §0 目录

1. [§1 环境前置检查](#§1-环境前置检查)
2. [§2 从零启动 Superset](#§2-从零启动-superset)
3. [§3 日常访问与使用](#§3-日常访问与使用)
4. [§4 添加新 chart / 新 dashboard / 新 filter](#§4-添加新-chart--新-dashboard--新-filter)
5. [§5 迁移到新机器（完整 10 分钟重建）](#§5-迁移到新机器完整-10-分钟重建)
6. [§6 数据刷新（重跑 ETL 后）](#§6-数据刷新重跑-etl-后)
7. [§7 备份与恢复](#§7-备份与恢复)
8. [§8 故障处置判断树](#§8-故障处置判断树)
9. [§9 安全与权限](#§9-安全与权限)
10. [§A 附录：Docker / API 常用命令](#§a-附录docker--api-常用命令)
11. [§B MVP L4-L6 资产清单 + 重建/回滚（v2 新增）](#§b-mvp-l4-l6-资产清单--重建--回滚v2-新增)

---

## §1 环境前置检查

执行以下命令确认依赖齐全：

```bash
# 1. Docker 可用
docker --version
# 期望：Docker version 20.x+

docker compose version
# 期望：Docker Compose version v2.x+

# 2. Postgres voc_bi 可达（Superset 要连它）
psql -h localhost -p 5432 -U voc_user -d voc_bi -c "SELECT count(*) FROM review;"
# 期望：count=364569（或 ETL 后的最新行数）

# 3. Python 3.9+（跑 factory 脚本）
python3 --version
# 期望：3.9.x 或更高

# 4. 网络端口 8088 未被占用
lsof -i :8088 | grep LISTEN
# 期望：无输出 或 docker-proxy 已在占
```

**如果 voc_bi 不可达**：先去跑 [ETL_pipeline_SOP.md](ETL_pipeline_SOP.md) 的 §2 建库 + §3 导入。

---

## §2 从零启动 Superset

### 2.1 进入工作目录

```bash
cd /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/docker/
ls
# 期望看到：
#   docker-compose.superset.yml
#   superset_bootstrap.py
#   superset_bootstrap.sh
#   superset_charts_factory.py
#   superset_config.py
#   superset_filters_factory.py
```

### 2.2 启动容器

```bash
docker compose -f docker-compose.superset.yml up -d
# 首次运行耗时 ~30s（拉镜像 + init）

# 等待 healthy
for i in $(seq 1 30); do
  status=$(docker inspect voc_superset --format='{{.State.Health.Status}}' 2>/dev/null)
  echo "[$i] status=$status"
  if [ "$status" = "healthy" ]; then break; fi
  sleep 10
done
# 期望最后：status=healthy（约 1-2 分钟）
```

### 2.3 验证容器运行

```bash
docker ps | grep voc_
# 期望 2 行：
#   voc_superset        Up X (healthy)
#   voc_superset_redis  Up X

# 健康检查
curl -sS -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8088/health
# 期望：HTTP 200
```

### 2.4 浏览器验证

```bash
# macOS 直接打开
open http://localhost:8088/login/

# 登录：admin / voc_admin_2026
# 成功标志：看到 Welcome 页面，左上角「Dashboards / Charts / Datasets」菜单可点
```

### 2.5 数据源 + Datasets 就绪（如果是第一次）

```bash
cd /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC
python3 research/02-脚本工具/01-标签进化/docker/superset_bootstrap.py
# 输出期望包含：
#   ✅ Database 'voc_bi' connected (id=1)
#   ✅ Dataset v_review_overview (id=1)
#   ✅ Dataset v_label_with_dept (id=2)
#   ...（共 6 个）
```

### 2.6 Charts + Dashboards

```bash
python3 research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py
# 期望输出末尾：
#   📊 Total charts: 12 / 12
#   overview dashboard → id=1 (5 charts)
#   dept 全球客服中心 → id=2
#   ...
```

### 2.7 Native Filters

```bash
python3 research/02-脚本工具/01-标签进化/docker/superset_filters_factory.py
# 期望末尾：
#   ✅ Done. Filters applied to 1 overview + 7 dept dashboards.
```

### 2.8 端到端验证

```bash
curl -sS -X POST http://localhost:8088/api/v1/security/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"voc_admin_2026","provider":"db"}' \
  | python3 -c "import sys,json; print('OK' if 'access_token' in json.load(sys.stdin) else 'FAIL')"
# 期望：OK
```

**全部通过 = §2 完成。** 浏览器打开 `http://localhost:8088/dashboard/list/` 应看到 8 个 dashboards。

---

## §3 日常访问与使用

### 3.1 URL 速查

| 用途 | 生产 URL（腾讯云） | 本机 URL（开发） |
|---|---|---|
| 首页 | `https://voc.lute-tlz-dddd.top/` | `http://localhost:8088/` |
| 登录 | `https://voc.lute-tlz-dddd.top/login/` | `http://localhost:8088/login/` |
| Dashboard 列表 | `https://voc.lute-tlz-dddd.top/dashboard/list/` | `http://localhost:8088/dashboard/list/` |
| Chart 列表 | `https://voc.lute-tlz-dddd.top/chart/list/` | `http://localhost:8088/chart/list/` |
| Dataset 列表 | `https://voc.lute-tlz-dddd.top/tablemodelview/list/` | 同上 |
| SQL Lab | `https://voc.lute-tlz-dddd.top/sqllab/` | 同上 |
| API 文档 | `https://voc.lute-tlz-dddd.top/swagger/v1` | 同上 |

### 3.2 13 个 Dashboard 直达链接（v2）

#### Phase 7 看板（id 1-8 · 8 个）

| ID | Dashboard | URL | slug |
|----|---|---|---|
| 1 | VOC Overview · 全局总览 | `https://voc.lute-tlz-dddd.top/superset/dashboard/1/` | `voc-overview` |
| 2 | VOC · 全球客服中心 | `https://voc.lute-tlz-dddd.top/superset/dashboard/2/` | `voc-dept-全球客服与体验中心` |
| 3 | VOC · 产品中心 ⭐ | `https://voc.lute-tlz-dddd.top/superset/dashboard/3/` | `voc-dept-产品中心品线` |
| 4 | VOC · 仓储物流部 | `https://voc.lute-tlz-dddd.top/superset/dashboard/4/` | `voc-dept-供应链中心` |
| 5 | VOC · 品牌市场中心 | `https://voc.lute-tlz-dddd.top/superset/dashboard/5/` | `voc-dept-品牌市场中心` |
| 6 | VOC · 电商运营部 | `https://voc.lute-tlz-dddd.top/superset/dashboard/6/` | `voc-dept-电商运营部` |
| 7 | VOC · 品质管理中心 | `https://voc.lute-tlz-dddd.top/superset/dashboard/7/` | `voc-dept-品控部` |
| 8 | VOC · 法务合规部 | `https://voc.lute-tlz-dddd.top/superset/dashboard/8/` | `voc-dept-质量与法规部` |

> ⭐ 部门 dashboard（2-8）在 L6 后已经在末尾追加 D-Action 行动队列 chart。

#### MVP 看板（id 9-13 · 5 个新增）

| ID | Dashboard | URL | slug |
|----|---|---|---|
| 9 | VOC · 深度分析 D-Health（L4） | `https://voc.lute-tlz-dddd.top/superset/dashboard/9/` | `voc-deep-health` |
| 10 | VOC · 深度分析 D-Diag-Product 产品力诊断（L5） | `https://voc.lute-tlz-dddd.top/superset/dashboard/10/` | `voc-deep-diag-product` |
| 11 | VOC · 深度分析 D-Diag-Service 服务力诊断（L5） | `https://voc.lute-tlz-dddd.top/superset/dashboard/11/` | `voc-deep-diag-service` |
| 12 | VOC · 深度分析 D-Diag-Brand 品牌内容力诊断（L5） | `https://voc.lute-tlz-dddd.top/superset/dashboard/12/` | `voc-deep-diag-brand` |
| 13 | VOC · 深度分析 D-Action 7 部门行动总队列（L6）⭐⭐ | `https://voc.lute-tlz-dddd.top/superset/dashboard/13/` | `voc-deep-action-overview` |

> ⭐⭐ id 13 是管理层入口：一页看到全部门 Top 20 行动项 + 7 个部门各自 Top 10。

### 3.3 使用过滤器

1. 打开任一部门 dashboard（例如 `/dashboard/3/` 产品中心）
2. 左侧「过滤」面板 → 点击「情感极性」下拉
3. 选择「负向」→ 点击底部「Apply filters」
4. 3-5s 后图表重绘，显示「Applied filters (1)」badge
5. 若要清空，点击「清除所有」

**Overview dashboard**：同理，但过滤器仅对饼图（chart 13/14）生效（因 dataset 限制，见 [phase7-architecture-diagrams.md §6](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md#图-6--10-native-filters-作用域)）。

---

## §4 添加新 chart / 新 dashboard / 新 filter

### 4.1 添加新 chart（基于现有 dataset）

**推荐方式：修改工厂脚本 + 重跑（声明式）**

```bash
# 1. 编辑 superset_charts_factory.py，在 chart_specs() 的返回列表中追加一个 spec dict
# 参考 existing 的 spec 结构（viz_type / params / datasource_id / datasource_type / tag / name）

# 2. 重跑工厂（idempotent：已存在的 chart 会 skip，只创建新的）
python3 research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py
# 期望：已有 12 个 skip，新的显示 'id=15'
```

**⚠️ 如果 dataset 还不存在**（比如你要加的 chart 是基于新视图）：
- 先在 `voc_bi_views.sql` 添加视图定义
- 执行 `psql -U voc_user -d voc_bi -f voc_bi_views.sql`
- 然后跑 `superset_bootstrap.py` 注册新 dataset
- 最后跑 `superset_charts_factory.py`

### 4.2 添加新 dashboard

```bash
# 编辑 superset_charts_factory.py 的 main() 末尾
# 按 existing dept dashboard pattern 追加：
#   d_new_id = upsert_dashboard(token, csrf, "VOC · 新部门", "voc-dept-new", ...)
#   attach_charts_to_dashboard(token, csrf, d_new_id, [chart_id_X], ["Chart Name X"])

# 重跑
python3 research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py
```

### 4.3 添加新 filter

编辑 `superset_filters_factory.py`：

```python
# 如果是 Overview-style filter（作用于饼图 dataset）
def configure_overview(token, csrf, chart_ids_by_name):
    overview_ids = [...]  # 保持不变
    filters = [
        # existing filters...
        build_filter(
            "NATIVE_FILTER-new-column", "新维度",
            DATASET_REVIEW, "new_column_name", overview_ids,
            "新过滤器描述"
        ),
    ]
    ...

# 如果是 Dept-style filter
def configure_dept_dashboards(token, csrf, chart_ids_by_name):
    for dd in DEPT_DASHBOARDS:
        chart_id = chart_ids_by_name.get(dd["chart_name"])
        filters = [
            # existing polarity filter...
            build_filter(
                "NATIVE_FILTER-new-dept-col", "新维度",
                DATASET_DEPT_TOPIC, "new_column", [chart_id],
                "描述"
            ),
        ]
```

重跑：`python3 superset_filters_factory.py`

**约束**：filter 的 dataset 必须有对应 column；chartsInScope 的 chart 必须使用同一个 dataset（否则 Superset 静默忽略）。

### 4.4 通过 UI 手改 vs 脚本改

| 场景 | 推荐方式 |
|---|---|
| 探索性尝试 / 一次性看看 | UI |
| 生产 / 需要可重现 | **脚本**（入 git） |
| 改完要保留到生产 | 改脚本 + 重跑 + 删 UI 临时改动 |

**NEVER** 在 UI 里改生产 dashboard 不同步到脚本——下次迁移/重建就丢了。

---

## §5 迁移到新机器（完整 10 分钟重建）

### 5.1 前置

在新机器上：
- Docker + Docker Compose 已装
- Postgres 实例可用（或用 docker-compose 新起一个）
- Git clone 的仓库就位

### 5.2 迁移步骤

```bash
# 1. 进仓库
cd paper_to_skills/paper2skills-vault/07-NLP-VOC

# 2. 启 voc_bi（假设用 docker）— 或连已有 Postgres
# （详见 ETL_pipeline_SOP.md §2）

# 3. 跑 ETL 导入数据（~37s）
python3 research/02-脚本工具/01-标签进化/etl_to_postgres.py \
  --input research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl \
  --dict research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx

# 4. 启 Superset（~2min 冷启动）
cd research/02-脚本工具/01-标签进化/docker
docker compose -f docker-compose.superset.yml up -d
# 等 healthy...

# 5. 三个工厂脚本（~20s）
cd ../../../../
python3 research/02-脚本工具/01-标签进化/docker/superset_bootstrap.py
python3 research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py
python3 research/02-脚本工具/01-标签进化/docker/superset_filters_factory.py

# 6. 验证
curl -sS http://localhost:8088/health
open http://localhost:8088/dashboard/list/
```

### 5.3 或：直接导入 ZIP（更快）

如果 Superset 已运行 + voc_bi 已导入 + datasets 已注册：

```bash
# 通过 UI 导入（最省事）
# Settings → Import Dashboards → 上传 8 个 ZIP
# 位置：research/04-输出结果/11-BI看板/superset_exports/dashboard_*.zip

# 或通过 API 批量导入
for f in research/04-输出结果/11-BI看板/superset_exports/dashboard_*.zip; do
  curl -sS -X POST http://localhost:8088/api/v1/dashboard/import/ \
    -H "Authorization: Bearer $TOKEN" \
    -F "formData=@$f" \
    -F "overwrite=true"
done
```

---

## §6 数据刷新（重跑 ETL 后）

### 6.1 无需重建 Superset

ETL 重跑只是更新 Postgres 数据；Superset 的 chart/dashboard/filter 配置保持。**不需要重跑 charts_factory / filters_factory**。

### 6.2 清缓存

```bash
# Redis 缓存会让旧数据保留 5min，可手动清
docker exec voc_superset_redis redis-cli FLUSHDB
# 期望：OK

# 或：浏览器强制刷新 dashboard
# URL 加 ?force=true 参数
open "http://localhost:8088/dashboard/3/?force=true"
```

### 6.3 验证新数据已上

```bash
# 在 Superset SQL Lab 里跑
SELECT count(*) FROM review;
# 期望：新数字（比如 400K）
```

---

## §7 备份与恢复

### 7.1 Superset 元数据备份（dashboards + charts + filters 配置）

```bash
# 每个 dashboard 导出 ZIP（含其所属 charts 和 filters）
for did in 1 2 3 4 5 6 7 8; do
  curl -sS "http://localhost:8088/api/v1/dashboard/export/?q=!($did)" \
    -H "Authorization: Bearer $TOKEN" \
    -o "backup_dashboard_$did.zip"
done

# 更好：直接使用生产 ZIP（已在仓库）
ls research/04-输出结果/11-BI看板/superset_exports/
# 8 个 ZIP + README.md
```

### 7.2 Postgres 数据备份

```bash
pg_dump -h localhost -U voc_user -d voc_bi \
  --format=custom \
  --file=voc_bi_backup_$(date +%Y%m%d).dump
# 期望：约 500MB 的压缩文件
```

### 7.3 Superset metadata DB（SQLite）备份

```bash
# Superset 的 user/role/dashboard-to-chart 关系在此
docker cp voc_superset:/app/superset_home/superset.db \
  ./superset_metadata_backup_$(date +%Y%m%d).db
```

### 7.4 恢复流程

```bash
# 1. Postgres 恢复
pg_restore -h localhost -U voc_user -d voc_bi \
  --clean --if-exists \
  voc_bi_backup_YYYYMMDD.dump

# 2. Superset metadata 恢复
docker cp superset_metadata_backup_YYYYMMDD.db \
  voc_superset:/app/superset_home/superset.db
docker restart voc_superset

# 3. 等 healthy → 浏览器验证
```

---

## §8 故障处置判断树

```
          ┌─ 浏览器打不开 http://localhost:8088 ─┐
          │                                   │
     是否 503 错误？──► 是 ──► §8.1 容器没起来
          │ 否
          ▼
     是否 connection refused？──► 是 ──► §8.2 端口没监听
          │ 否
          ▼
     是否 200 但页面空白？──► 是 ──► §8.3 前端资源加载失败
          │ 否
          ▼
     登录后 dashboard 打不开？──► 是 ──► §8.4 数据层问题
          │ 否
          ▼
     Chart 显示「意外错误」？──► 是 ──► §8.5 SQL 查询问题
          │ 否
          ▼
     Filter 不生效？──► 是 ──► §8.6 Filter scope 错配
          │ 否
          ▼
     其他 ──► 查 Superset 日志（§A.2）
```

### §8.1 容器没起来

```bash
# 诊断
docker ps -a | grep voc_superset
# 如果 Exited → 看日志
docker logs voc_superset --tail 50

# 常见原因
# 1. 镜像没拉下来 → docker pull apache/superset:4.1.1
# 2. 端口冲突 → lsof -i :8088 找到占用者 → kill 或改 docker-compose 端口
# 3. 磁盘满 → df -h，清理 docker system prune
# 4. voc_bi 不可达（启动时连不上）→ 检查 Postgres 运行状态

# 处置
docker compose -f docker-compose.superset.yml restart superset
# 或彻底重启
docker compose -f docker-compose.superset.yml down
docker compose -f docker-compose.superset.yml up -d
```

### §8.2 端口没监听

```bash
# 诊断
curl -v http://localhost:8088/health 2>&1 | grep -i "connection"
# "Connection refused" = 端口没人监听

# 检查
docker port voc_superset
# 期望：8088/tcp -> 0.0.0.0:8088

# 如果没映射，改 docker-compose.superset.yml 的 ports 段，然后
docker compose -f docker-compose.superset.yml up -d --force-recreate
```

### §8.3 前端资源加载失败

打开浏览器开发者工具 → Network tab → 重新刷新 → 看哪个资源 404/500

```bash
# 常见：nginx/反向代理配置问题，或 Superset static assets 路径错
# 直接访问 localhost:8088 绕过代理测试

# 如果 docker 日志有 "static asset not found" → 重启容器
docker restart voc_superset
# 等 healthy → 再访问
```

### §8.4 数据层问题

```bash
# Superset 报 "Unable to connect to database"
docker exec voc_superset bash -c "
  cat /app/pythonpath/superset_config.py | grep -A 3 'SQLALCHEMY_DATABASE'
"
# 看 Superset 配置是否指向正确的 voc_bi

# 测试从 Superset 容器内 ping voc_bi
docker exec voc_superset bash -c "
  apt-get install -y --no-install-recommends postgresql-client > /dev/null
  psql postgresql://voc_user:voc_pass@host.docker.internal:5432/voc_bi -c 'SELECT 1'
"
# 期望：返回 1

# 如果 host.docker.internal 不解析（Linux）→ 检查 docker-compose 的 extra_hosts 段
```

### §8.5 SQL 查询问题

**饼图显示「意外错误：Columns missing in dataset: ['review_id']」**：

这是 D3 遗留 bug，D4 已修（见 [phase7_d4_progress_report.md §五](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md)）。如果你看到这个错，说明 chart 4/5 没有被 D4 重建。处置：

```bash
# 重建 chart 4, 5 为 chart 13, 14（使用 COUNT(*) SQL form）
python3 research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py
# 如果 chart 4,5 还在但 spec 不对，先删再重建：
# curl -X DELETE http://localhost:8088/api/v1/chart/4 -H "Authorization: Bearer $TOKEN"
# curl -X DELETE http://localhost:8088/api/v1/chart/5 -H "Authorization: Bearer $TOKEN"
# python3 superset_charts_factory.py
```

**Chart 超时（>30s）**：

```bash
# 去 SQL Lab 直接跑 chart 背后的 SQL
# 加 EXPLAIN ANALYZE 看是不是索引问题
# 常见：v_dept_topic_summary 没聚合好，改进 voc_bi_views.sql
```

### §8.6 Filter 不生效

```bash
# 诊断：Filter 选了值，点 Apply，图表数据没变

# 检查 1：filter 的 chartsInScope 是否包含当前 chart
curl -sS http://localhost:8088/api/v1/dashboard/1 \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
meta = json.loads(d['json_metadata'])
for f in meta.get('native_filter_configuration', []):
    print(f['name'], '→', f.get('chartsInScope', []))
"
# 期望：数据源 → [13, 14]（如果你的 chart 是 13 或 14）

# 检查 2：chart 用的 dataset 是否含 filter 的 column
# Overview filters 用 v_review_overview，所以只对 chart 13 (数据源 pie) 和 14 (NPS pie) 生效
# chart 1,2 用 v_dept_kpi（没有 data_source 列），filter 会静默忽略

# 处置：如果你加的新 chart 想被 filter 作用，确保它用 v_review_overview
# 或者加新 filter 到新 chart 用的 dataset 上
```

### §8.7 Superset 饼图 dashboard-mode 不渲染（已知 bug）

**症状**：Overview dashboard 打开后，饼图 canvas 是透明的（所有像素 RGBA=0,0,0,0），但 `/explore/?slice_id=13` 单图模式正常。

**根因**：Superset 4.1.1 前端 ECharts 在 dashboard 模式 + pie + SQL-form metric 组合下的实例化 bug。

**当前处置**：
- 不阻塞业务：部门 dashboard 的 top-10 柱状图全部正常
- 详细分析：[phase7_d4_progress_report.md §五](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md)
- 修复方案（Phase 8 候选）：
  - 选项 A：把饼图改为 bar chart
  - 选项 B：升级 Superset 到 4.2+
  - 选项 C：改 metric 为 SIMPLE 形式但用 dummy column

---

## §9 安全与权限

### 9.1 当前生产状态（2026-05-11）

| 维度 | 当前配置 | 风险评级 |
|---|---|---|
| 访问地址 | localhost only | 🟢 安全（仅本机） |
| 认证 | admin + password | 🟡 默认密码需换 |
| 授权 | 全部 dashboard 公开 | 🟡 无 RBAC |
| HTTPS | 无 | 🔴 仅 localhost 可接受 |
| 跨域 | 默认 | 🟢 无对外暴露 |

### 9.2 生产部署前必做

1. **改默认密码**
   ```bash
   # 进 Superset 容器改
   docker exec -it voc_superset superset fab reset-password \
     --username admin --password '<新密码>'
   ```
2. **改 SECRET_KEY**（编辑 `docker-compose.superset.yml` 的 `SUPERSET_SECRET_KEY`）
3. **加 HTTPS**（Nginx 反向代理 + Let's Encrypt）
4. **启用 RBAC**（Settings → Security → Roles）
5. **SSO 接入**（若要跨部门访问）

### 9.3 RBAC 推荐 Role 设计

| Role | 能看 | 能改 |
|---|---|---|
| `voc_admin` | 全部 | 全部 |
| `voc_analyst` | 全部 dashboard | 改自己 chart |
| `voc_viewer_dept_<X>` | 仅部门 X 的 dashboard | 无 |

配置：Settings → List Roles → New Role → 添加对应 permissions。

---

## §B MVP L4-L6 资产清单 + 重建 / 回滚（v2 新增）

> 本章覆盖 MVP L4 (D-Health) / L5 (D-Diag 三专题) / L6 (D-Action 7 部门行动队列) 共 5 个新 dashboard + 30 个新 chart + 4 个新 dataset 的：完整对象清单、重建脚本、一键回滚。

### §B.1 资产规模对比（Phase 7 → MVP L7）

| 维度 | Phase 7 末态 | MVP L7 末态 | Δ |
|---|---:|---:|---:|
| Dataset | 6 | **10** | +4（L4 新增） |
| Chart | 19（含 5/4 D3 bug 修复后） | **42** | +23（L4:7 + L5:15 + L6:8 - 注：L4 重建 ch4/5 → 13/14 是覆盖关系） |
| Dashboard | 8 | **13** | +5（L4:1 + L5:3 + L6:1） |
| Postgres view | 6 | **10** | +4（L3 新增 4 个深度分析视图） |
| voc_review 列 | 15 | **17** | +2（country + ts_inferred，L1） |
| dim_tag 列 | 11 | **12** | +1（atomic_indicator_id，L2） |
| 工程周期 | Phase 5+6+7 = 22d | + MVP L0-L7 = 13d → **35d 累计** | +13 |

### §B.2 4 个 MVP 新视图（L3 产出）

```bash
# 视图定义文件
research/02-脚本工具/01-标签进化/sql/mvp_l3_views.sql

# 4 个视图（普通 VIEW，无物化）
1. v_atomic_indicator_score   · 50 SAT × month × country × product_line 得分
2. v_aipl_node_score          · L1-L4 AIPL 节点得分
3. v_country_dept_health      · country × dept × month 健康度
4. v_proxy_nps_calibrated     · 按 04-NPS 偏差校准 Proxy NPS

# 重建命令
psql -h <prod-pg> -U voc_user -d voc_bi \
  -f research/02-脚本工具/01-标签进化/sql/mvp_l3_views.sql

# 回滚命令
psql -h <prod-pg> -U voc_user -d voc_bi -c "
  DROP VIEW v_aipl_node_score, v_atomic_indicator_score,
           v_country_dept_health, v_proxy_nps_calibrated CASCADE;
"
```

### §B.3 MVP L4 D-Health（dashboard 9）

| Chart ID | viz | 用途 |
|---:|---|---|
| 13 | big_number_total | 总 reviews 数（重建后用） |
| 14 | big_number_total | 平均 SAT 分 |
| 15 | big_number_total | 校准 NPS |
| 16 | dist_bar | AIPL 节点健康度 by country |
| 17 | table | SAT 排行 Top/Bottom 10 |
| 18 | heatmap | Country × Dept 负向率热力 |
| 19 | dist_bar | 校准前后 NPS by source |

**重建脚本**：

```bash
# 1. 注册 4 个 dataset（v_atomic_indicator_score 等）
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l4_create_charts.py
# 期望末尾：created 7 charts (id 13-19)

# 2. 装配 dashboard
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l4_create_dashboard.py
# 期望末尾：DASHBOARD ID = 9
```

**回滚**：

```bash
# 删除 1 dashboard + 7 charts + 4 datasets
# 详见 ~/.secrets/backups/voc-deep-analysis-mvp/L4-*/MANIFEST.md
```

### §B.4 MVP L5 D-Diag 三专题（dashboard 10/11/12）

| Dashboard | Charts | 主题 |
|---:|---|---|
| 10 | 20-24 (P-1~P-5) | 产品力诊断 |
| 11 | 25-29 (S-1~S-5) | 服务力诊断 |
| 12 | 30-34 (B-1~B-5) | 内容/品牌力诊断 |

**15 charts 概览**：

| code | dashboard | viz | name |
|---|:---:|---|---|
| P-1 | 10 | table | 产品中心 SAT 失血 Top 10 |
| P-2 | 10 | heatmap | 产品中心 × 国家 负向率热力 |
| P-3 | 10 | table | 产品中心 × 国家 体量+负向率 |
| P-4 | 10 | dist_bar | L1 使用层 SAT 全量表现 |
| P-5 | 10 | dist_bar | 产品中心 国家×负向率（单批快照） |
| S-1 | 11 | heatmap | 服务三部门 × 国家 负向率热力 |
| S-2 | 11 | table | L3 服务层 SAT 失血 Top 10 |
| S-3 | 11 | heatmap | 数据源 × 国家 NPS 校准矩阵 |
| S-4 | 11 | dist_bar | 仓储物流 SAT 明细排行 |
| S-5 | 11 | pie | 核心质量 SAT 体量分布 |
| B-1 | 12 | table | 品牌市场中心 SAT Top 10 |
| B-2 | 12 | dist_bar | L4 品牌层 SAT × 国家 |
| B-3 | 12 | dist_bar | SAT_L4_01 品牌声量 by 国家（单批快照） |
| B-4 | 12 | dist_bar | SAT_L3_17 内容/社群 × 国家 |
| B-5 | 12 | dist_bar | 节点体量 vs 节点得分 双轴 |

**重建**：

```bash
# 1. 15 chart
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l5_create_charts.py
# 期望末尾：created [{id:20,code:P-1,...}, ...{id:34}]

# 2. 3 dashboards（自动 link chart 到 dashboard）
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l5_create_dashboards.py
# 期望末尾：3 dashboards 10/11/12 + chart links 全部 200
```

**回滚**：

```bash
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l5_rollback.py
# 删除 dashboard 10/11/12 + charts 20-34
```

> **已知非阻塞限制**：`ts_inferred = voc_review.loaded_at`（ETL 入库时间，单批 ETL 下所有 review 共享 month=2026-05-01）→ P-5 / B-3 line 退化为单点，已切换为 dist_bar 单批快照。upgrade path：真实 timestamp 入库后，反转 viz_type 即可。

### §B.5 MVP L6 D-Action 7 部门行动总队列（dashboard 13 + 7 dept dashboard 追加）

#### B.5.1 双路径

- **Path A**：现有 7 dept dashboard (id 2-8) 末尾追加 D-Action 行动队列 chart
- **Path B**：新建总览 dashboard 13（含 8 charts：master Top 20 + 7 dept Top 10）

#### B.5.2 8 chart 清单（id 35-42）

| Chart | dept | 关联 dashboard |
|---:|---|---|
| 35 | 产品中心 | 3, 13 |
| 36 | 全球客服中心 | 2, 13 |
| 37 | 仓储物流部 | 4, 13 |
| 38 | 品牌市场中心 | 5, 13 |
| 39 | 品质管理中心 | 7, 13 |
| 40 | 电商运营部 | 6, 13 |
| 41 | 法务合规部 | 8, 13 |
| 42 | 全部门 master Top 20 | 13 |

#### B.5.3 优先级公式

```sql
priority_score = hit_negative × pct_negative / 100
-- 体量 × 负向占比 双重加权 · 与 SGCS 失血指标排序逻辑一致
```

#### B.5.4 关键过滤

⚠️ dim_tag 里 15 个产品中心标签 biz_action 为占位符 `【待填写】`，过滤时必须排除：

```sql
WHERE biz_action IS NOT NULL
  AND biz_action <> ''
  AND biz_action NOT LIKE '%待填写%'
```

#### B.5.5 重建

```bash
# 1. 8 charts
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_create_charts.py
# 期望末尾：created 8 charts (id 35-42)

# 2. Path A: 追加到 7 dept dashboards
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_append_to_dept_dashboards.py
# 期望末尾：7 entries 'appended'

# 3. Path B: 创建 D-Action 总览
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_create_overview.py
# 期望末尾：dashboard_id=13 + chart links 8 个 200

# 4. （如需）修复过滤排除占位符
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_fix_action_filter.py
```

#### B.5.6 回滚

```bash
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_rollback.py
# Step 1: 从 pre_l6_snapshot.json 还原 7 dept dashboard position_json
# Step 2: 删除 dashboard 13
# Step 3: 删除 charts 35-42
```

### §B.6 完整 MVP 重建顺序（10-15 分钟）

```bash
# 前置：voc_bi Postgres 已就位 + Phase 7 看板已 bootstrap

# Layer 1 (SQL views) ────────────────────
psql -h <prod-pg> -U voc_user -d voc_bi \
  -f research/02-脚本工具/01-标签进化/sql/mvp_l3_views.sql
# 4 视图就位

# Layer 2 (L4 D-Health) ─────────────────
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l4_create_charts.py
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l4_create_dashboard.py
# dataset 7-10 + chart 13-19 + dashboard 9

# Layer 3 (L5 D-Diag) ────────────────────
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l5_create_charts.py
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l5_create_dashboards.py
# chart 20-34 + dashboard 10/11/12

# Layer 4 (L6 D-Action) ──────────────────
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_create_charts.py
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_append_to_dept_dashboards.py
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_create_overview.py
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_fix_action_filter.py
# chart 35-42 + dashboard 13 + dept dashboard 2-8 各追加 1 row

# 验证
curl -k https://voc.lute-tlz-dddd.top/health
# 期望：200
open https://voc.lute-tlz-dddd.top/superset/dashboard/13/
# 应看到 8 个 D-Action 表格
```

### §B.7 完整 MVP 回滚顺序（5-10 分钟）

```bash
# 倒序回滚：L6 → L5 → L4 → L3 views
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l6_rollback.py
python3 research/02-脚本工具/01-标签进化/scripts/mvp_l5_rollback.py
# L4 rollback：参考 ~/.secrets/backups/voc-deep-analysis-mvp/L4-*/MANIFEST.md
psql -h <prod-pg> -U voc_user -d voc_bi -c "
  DROP VIEW v_aipl_node_score, v_atomic_indicator_score,
           v_country_dept_health, v_proxy_nps_calibrated CASCADE;
"
```

### §B.8 备份索引（5 个 L 阶段 baseline）

```
~/.secrets/backups/voc-deep-analysis-mvp/
├── L0-20260514-102823/      # MVP 准备
├── L1-20260514-112119/      # voc_review 扩展
├── L2-20260514-115502/      # 50 SAT 映射
├── L3-20260514-122215/      # 4 深度视图
├── L4-20260514-144802/      # D-Health
├── L5-20260514-163344/      # D-Diag 三专题
├── L6-20260514-171018/      # D-Action 7 部门
└── L7-<timestamp>/          # 本 SOP v2 升级
```

每个 L 目录均有 MANIFEST.md + 重建脚本 + 回滚脚本 + 截图。

---

### A.1 容器管理

```bash
# 启动
docker compose -f docker-compose.superset.yml up -d

# 停止（保留数据）
docker compose -f docker-compose.superset.yml stop

# 停止并删除容器（保留 volume）
docker compose -f docker-compose.superset.yml down

# 停止并删除容器 + volume（彻底清零）
docker compose -f docker-compose.superset.yml down -v

# 查看日志
docker logs voc_superset --tail 100 -f

# 进容器 shell
docker exec -it voc_superset bash

# 重启单容器
docker restart voc_superset
```

### A.2 Superset 日志排查

```bash
# 应用日志
docker logs voc_superset 2>&1 | grep -i "error\|exception" | tail 30

# 容器内 Flask logs
docker exec voc_superset tail -f /app/superset_home/superset.log

# Redis 状态
docker exec voc_superset_redis redis-cli INFO memory
```

### A.3 REST API 常用（需先 login 拿 token）

```bash
# Login
TOKEN=$(curl -sS -X POST http://localhost:8088/api/v1/security/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"voc_admin_2026","provider":"db","refresh":true}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# List charts
curl -sS "http://localhost:8088/api/v1/chart/?q=(page_size:100)" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool | head 50

# Get dashboard detail
curl -sS "http://localhost:8088/api/v1/dashboard/1" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

# Export dashboard
curl -sS "http://localhost:8088/api/v1/dashboard/export/?q=!(1,2,3)" \
  -H "Authorization: Bearer $TOKEN" -o "dashboards_1_2_3.zip"

# Delete chart
curl -sS -X DELETE "http://localhost:8088/api/v1/chart/4" \
  -H "Authorization: Bearer $TOKEN"
```

### A.4 Postgres 常用

```bash
# 进 psql
psql -h localhost -U voc_user -d voc_bi

# 看表行数
psql -h localhost -U voc_user -d voc_bi -c "
  SELECT 'review' t, count(*) FROM review
  UNION ALL SELECT 'label', count(*) FROM label;
"

# 看视图定义
psql -h localhost -U voc_user -d voc_bi -c "\sv v_dept_topic_summary"

# 手改视图（慎用！改完 Superset 数据会立刻变）
psql -h localhost -U voc_user -d voc_bi -f research/02-脚本工具/01-标签进化/sql/voc_bi_views.sql
```

---

## 附：关联文档

| 类型 | 链接 |
|---|---|
| 白话汇报 | [phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md) |
| 架构图集（Markdown） | [phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md) |
| 架构图集（HTML） | [phase7-architecture-diagrams.html](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.html) |
| ETL 上游 SOP | [ETL_pipeline_SOP.md](ETL_pipeline_SOP.md) |
| Phase 7 完整复盘 | [phase7_complete_retrospective.md](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) |
| Phase 7 D4 进度报告 | [phase7_d4_progress_report.md](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md) |

---

> **SOP 维护约定**：每次 Superset 配置、factory 脚本、或底层数据层发生变化时，更新对应章节。如果整个架构升级（如 Phase 8），本文挪至 00-归档资料/ 并建新版。
