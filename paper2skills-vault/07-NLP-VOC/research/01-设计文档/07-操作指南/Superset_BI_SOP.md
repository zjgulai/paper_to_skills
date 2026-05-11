---
name: superset-bi-sop
description: Superset BI 看板运维 SOP — 高详细度操作手册。覆盖从零启动、日常访问、添加 chart/filter、迁移重建、故障处置、权限管理、备份恢复的全流程。面向接手的工程/运维同事，逐行命令 + 故障判断树 + 底脱图。
title: Superset BI 看板运维 SOP
doc_type: sop
module: voc-nlp
topic: superset-bi-operations
status: stable
created: 2026-05-11
updated: 2026-05-11
owner: self
source: ai
audience: engineer
---

# Superset BI 看板运维 SOP

> **文档定位**：接手人能照着逐行做，不需要去翻 Phase 7 的 4 份进度报告。
>
> **配套文档**：
> - 白话背景：[phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md)
> - 架构图：[phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md)
> - ETL 上游：[ETL_pipeline_SOP.md](ETL_pipeline_SOP.md)
>
> **当前生产配置**：
> - Superset 版本：4.1.1（Docker）
> - 访问地址：`http://localhost:8088`
> - 管理员：`admin / voc_admin_2026`
> - 后端数据库：`voc_bi`（Postgres，host network）

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
#   dept 客服部 → id=2
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

| 用途 | URL |
|---|---|
| 首页 | `http://localhost:8088/` |
| 登录 | `http://localhost:8088/login/` |
| Dashboard 列表 | `http://localhost:8088/dashboard/list/` |
| Chart 列表 | `http://localhost:8088/chart/list/` |
| Dataset 列表 | `http://localhost:8088/tablemodelview/list/` |
| SQL Lab | `http://localhost:8088/sqllab/` |
| API 文档 | `http://localhost:8088/swagger/v1` |

### 3.2 8 个 Dashboard 直达链接

| ID | Dashboard | URL |
|---|---|---|
| 1 | VOC Overview · 全局总览 | `http://localhost:8088/dashboard/1/` |
| 2 | VOC · 客服部 | `http://localhost:8088/dashboard/2/` |
| 3 | VOC · 产品研发部 | `http://localhost:8088/dashboard/3/` |
| 4 | VOC · 国际物流部 | `http://localhost:8088/dashboard/4/` |
| 5 | VOC · 市场部 | `http://localhost:8088/dashboard/5/` |
| 6 | VOC · 电商运营部 | `http://localhost:8088/dashboard/6/` |
| 7 | VOC · 品控部 | `http://localhost:8088/dashboard/7/` |
| 8 | VOC · 质量与法规部 | `http://localhost:8088/dashboard/8/` |

### 3.3 使用过滤器

1. 打开任一部门 dashboard（例如 `/dashboard/3/` 产品研发部）
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

## §A 附录：Docker / API 常用命令

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
