---
name: phase7-d2-progress-report
description: Phase 7 D2 进度报告 — Superset Docker 部署 + voc_bi 接入 + 6 dataset 注册 + SQL Lab 烟测通过。BI B 路径的可视化层就绪，下一步可拼装 dashboard。
date: 2026-05-10
phase: phase7
day: D2
status: 🟢 Superset 实质上线 — 4.1.1 healthy, 6 datasets, SQL Lab 烟测 ✅
doc_type: audit-report
module: voc-nlp
---

# Phase 7 D2 进度报告 — Superset Docker 部署

> **总判定**：🟢 **Superset 4.1.1 实质上线**：Docker compose 启 + Redis 缓存 + voc_bi 数据库连接 + 6 SQL 视图注册为 dataset + Playwright 浏览器实跑验证 6/6 dataset 可见 + SQL Lab 烟测返回 7 部门 KPI 真实数据。`superset_bootstrap.py` idempotent 自动化，重跑 0 副作用。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| D2.1 docker-compose.superset.yml | ✅ | Superset 4.1.1 + Redis 7 |
| D2.2 superset_bootstrap.sh | ✅ | 容器启动时自动 pip install psycopg2-binary |
| D2.3 superset_config.py | ✅ | 中文 locale + Redis 缓存 + 关 CSRF（dev）|
| D2.4 admin user + db init | ✅ | admin / voc_admin_2026 |
| D2.5 voc_bi DB 接入 | ✅ | postgres+psycopg2://voc_bi@host.docker.internal:5432/voc_bi |
| D2.6 6 视图注册 dataset | ✅ | id 1-6 |
| D2.7 superset_bootstrap.py | ✅ | idempotent REST API 自动化 |
| D2.8 Playwright 浏览器验证 | ✅ | 6/6 datasets 在 UI 可见，SQL Lab 烟测返回 7 行 |

## 二、部署架构

```
┌──────────────────────────────────────────────┐
│ host (macOS)                                 │
│                                              │
│ ┌──────────────────────┐                     │
│ │ ai_video_pg (existing)                     │
│ │ postgres:16          │                     │
│ │ port 5432            │   ← voc_bi DB 在此  │
│ └──────────────────────┘                     │
│                  ↑                           │
│                  │ host.docker.internal:5432 │
│                  │                           │
│ ┌──────────────────────┐  ┌─────────────────┐│
│ │ voc_superset         │  │ voc_superset_redis││
│ │ apache/superset:4.1.1│←→│ redis:7         ││
│ │ port 8088            │  │ (cache + bg)    ││
│ └──────────────────────┘  └─────────────────┘│
│           ↑                                  │
│           │ http://localhost:8088            │
│           ↓                                  │
│       Browser                                │
└──────────────────────────────────────────────┘
```

## 三、关键工程踩坑 + 修复

### 3.1 Superset 4.1.1 默认镜像无 psycopg2

**症状**：`from sqlalchemy import create_engine; conn = create_engine('postgresql://...')` 抛 `ModuleNotFoundError: No module named 'psycopg2'`。

**根因**：apache/superset:4.1.1 official image 不预装 postgres 驱动（避免镜像膨胀）。

**修复**：`superset_bootstrap.sh` entrypoint 替换：
```bash
#!/bin/bash
if ! python3 -c "import psycopg2" >/dev/null 2>&1; then
  pip install --quiet psycopg2-binary
fi
exec /usr/bin/run-server.sh
```

`docker-compose.superset.yml` 用 `entrypoint: ["/bin/bash", "/app/docker/superset_bootstrap.sh"]` + `user: root` 让 pip install 有权限。重启容器后 psycopg2 自动重装（~3s）。

### 3.2 SQL Lab 默认 async 但无 Celery results backend

**症状**：API 烟测返回 `RESULTS_BACKEND_NOT_CONFIGURED_ERROR`。

**根因**：Superset 默认 SQL Lab 走 async（提交查询 → Celery worker → 结果存 Redis），但社区 Docker 不预配 Celery + results backend。

**修复**：在 voc_bi 数据库设置里 `allow_run_async: false`，强制同步执行。本地 dev 数据量（689K labels）下查询 < 5 秒，同步可接受。

`superset_bootstrap.py` 在 upsert_voc_bi_db 内做这个 PUT，**idempotent 重跑也自动修正**。

## 四、API 自动化（superset_bootstrap.py）

```
1. /api/v1/security/login        → JWT
2. /api/v1/security/csrf_token/  → CSRF
3. /api/v1/database/             → upsert voc_bi
4. /api/v1/database/{id} (PUT)   → allow_run_async: false
5. /api/v1/dataset/              → upsert 6 views
6. /api/v1/sqllab/execute/       → smoke test
```

**idempotent**：列出已有 → 存在则跳过 / 更新；不存在则创建。重跑 0 副作用。

烟测返回（部门 7 行）：
```json
[
  {"dept_owner": "产品中心/品线", "total_label_hits": 136205},
  {"dept_owner": "私域运营部", "total_label_hits": 77417},
  {"dept_owner": "品牌市场中心",     "total_label_hits": 76480},
  {"dept_owner": "全球客服与体验中心",     "total_label_hits": 58664},
  {"dept_owner": "供应链中心", "total_label_hits": 45113},
  {"dept_owner": "电商运营部", "total_label_hits": 33145},
  {"dept_owner": "产品中心/品线",     "total_label_hits": 12092}
]
```

## 五、Playwright 浏览器验证

| 检查 | 结果 |
|---|---|
| 登录页加载 | ✅ Page Title: Superset |
| admin / voc_admin_2026 登录 | ✅ 跳到 /superset/welcome/ |
| /tablemodelview/list/ 6 datasets | ✅ totalRows=6, 全部 v_* 链接可见 |
| Console errors | 0（仅 5 warnings 为业务无关）|

## 六、产出清单

| 文件 | 用途 | 大小 |
|---|---|---:|
| `02-脚本工具/01-标签进化/docker/docker-compose.superset.yml` | Superset + Redis 编排 | 1 KB |
| `02-脚本工具/01-标签进化/docker/superset_config.py` | 中文 locale + Redis 缓存 | 1 KB |
| `02-脚本工具/01-标签进化/docker/superset_bootstrap.sh` | 容器启动 psycopg2 自动安装 | 0.6 KB |
| `02-脚本工具/01-标签进化/docker/superset_bootstrap.py` | REST API 自动化 | 8 KB |
| `04-输出结果/03-审计报告/phase7_d2_progress_report.md` | 本文档 | 6 KB |

## 七、访问凭据 + 操作手册

```bash
# 1. 启动
cd paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/docker
docker compose -f docker-compose.superset.yml up -d

# 2. 等 healthy（首次约 60 秒，含 psycopg2 install）
docker inspect voc_superset --format='{{.State.Health.Status}}'

# 3. 初始化（首次）
docker exec voc_superset superset db upgrade
docker exec voc_superset superset fab create-admin \
  --username admin --firstname VOC --lastname Admin \
  --email admin@voc-bi.local --password voc_admin_2026
docker exec voc_superset superset init

# 4. 自动注册 voc_bi + 6 视图
python3 superset_bootstrap.py

# 5. 浏览器访问
open http://localhost:8088
# admin / voc_admin_2026
```

## 八、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | 凭据写在 sqlalchemy_uri 里（明文）| 中 | 当前 dev；生产用 Superset secret |
| R2 | CSRF 关闭（dev 友好但有风险）| 中 | 同上 |
| R3 | 同步 SQL Lab，大查询会阻塞 worker | 低 | 689K 数据下查询 < 5s 可接受 |
| R4 | 镜像 ~1.5GB，容器内存 ~500MB | 低 | 个人 mac 无压力 |
| R5 | 容器重启 psycopg2 重装（~3s）| 低 | bootstrap.sh idempotent |

## 九、D3 解锁条件

| 前置 | 状态 |
|---|---|
| Superset 健康运行 | ✅ |
| voc_bi DB 接入 + 6 dataset | ✅ |
| SQL Lab 烟测通过 | ✅ |
| 自动化脚本可重跑 | ✅ |

🟢 **D3 解锁**。下一步：

1. 在 Superset UI 创建 chart：复制 D10 静态看板的 9 个图表（overview source/sentiment + 7 dept SRAC）
2. 用 dataset v_dept_topic_summary 创建 7 部门 dashboard
3. 配置 dashboard filter（dept_owner, polarity, data_source）
4. 用户分发：建 dept-specific role + permissions（或 7 个 dashboard tab 给所有用户）

## 十、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-10 17:50 | docker-compose.superset.yml 写完 |
| 2026-05-10 17:55 | docker compose up -d，~3 min pull 1.5GB image |
| 2026-05-10 17:58 | 容器 healthy，但发现 psycopg2 缺失 |
| 2026-05-10 18:00 | 加 superset_bootstrap.sh 自动 pip install + 重建容器 |
| 2026-05-10 18:05 | admin user + db init + REST API 注册 voc_bi |
| 2026-05-10 18:08 | 6 datasets 注册成功 |
| 2026-05-10 18:10 | SQL Lab 烟测：RESULTS_BACKEND 错误 → 关 async 修复 |
| 2026-05-10 18:13 | 写 superset_bootstrap.py idempotent 自动化，重跑 0 副作用 |
| 2026-05-10 18:18 | Playwright 浏览器验证 6 dataset 在 UI 可见 |
| 2026-05-10 18:22 | 本报告归档，D3 解锁 |

## 十一、一行总结

> Phase 7 D2 **Superset 实质上线**：4.1.1 image + Redis 缓存 + 中文 locale + voc_bi 接入 + 6 视图 dataset 注册 + sync SQL Lab 烟测真实数据。修了 2 个工程踩坑（psycopg2 缺失 + async 无后端）。`superset_bootstrap.py` REST API 自动化重跑 idempotent。Playwright 浏览器验证 dataset 可见 + 登录流程畅通。**B 路径数据可视化层就绪**，D3 拼装 dashboard。
