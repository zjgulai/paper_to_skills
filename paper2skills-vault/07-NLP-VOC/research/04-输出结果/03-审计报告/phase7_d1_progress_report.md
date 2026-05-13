---
name: phase7-d1-progress-report
description: Phase 7 D1 进度报告 — voc_bi 数据库 + ETL + 6 SQL 视图层。BI B 路径（Superset/Metabase）的前置基础设施完成，下一步可直接接入 BI 工具。
date: 2026-05-10
phase: phase7
day: D1
status: 🟢 BI 数据基础设施就绪 — 4 表 + 6 视图 + ETL 37s
doc_type: audit-report
module: voc-nlp
---

# Phase 7 D1 进度报告 — voc_bi 数据库 + ETL + 视图层

> **总判定**：🟢 **Phase 7 D1 全部任务完成** — voc_bi 数据库就绪（4 表 + 6 视图）+ ETL 37 秒导入 364K reviews / 690K labels / 33K brands / 267 tags + 6 视图替代 MAA 查询，**SQL 输出与 D10 MAA Markdown 一致**。BI B 路径的"数据底座"实质上线，D2 可直接接入 Superset/Metabase。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| D1.1 dim/fact 模型设计 | ✅ | star schema：dim_tag + voc_review + voc_label + voc_brand_mention |
| D1.2 voc_bi 数据库 + 用户 | ✅ | 复用 `ai_video_pg` 容器，新建 `voc_bi` 数据库 + `voc_bi` 用户 |
| D1.3 etl_to_postgres.py | ✅ | 流式 jsonl 导入，psycopg2 batch=1000 |
| D1.4 ETL 实跑 | ✅ | **37 秒**导入 364K reviews + 690K labels + 33K brands + 267 dim_tag |
| D1.5 6 SQL 视图层 | ✅ | v_review_overview / v_label_with_dept / v_dept_topic_summary / v_label_brand / v_global_top_tags / v_dept_kpi |
| D1.6 验证 | ✅ | 产品中心/品线 top-3 标签命中数与 D10 MAA Markdown 完全一致 |

## 二、数据底座架构

### 2.1 4 张事实/维度表

```
dim_tag (267 rows, from v4.1 dict 01_通用标签主表)
  ↓
voc_review (364,569 rows, master fact, 1 per review)
  ├── voc_label (689,774 rows, 1 per (review, label))
  └── voc_brand_mention (33,138 rows, 1 per (review, brand))
```

### 2.2 6 SQL 视图（BI 工具直接消费）

| 视图 | 行数（首测）| 用途 |
|---|---:|---|
| `v_review_overview` | 364,569 | 总览：source / nps / sentiment / product_line 切分 |
| `v_label_with_dept` | 689,774 | 三表 JOIN 扁平表（含部门归属，**主消费视图**）|
| `v_dept_topic_summary` | ~250 | per-dept × per-tag 聚合（MAA Top-N 替代）|
| `v_label_brand` | ~33K | brand × label 交叉（品牌市场中心品牌分析）|
| `v_global_top_tags` | ~150 | 全局标签命中排行 |
| `v_dept_kpi` | 17 | 7 部门 KPI 概览（一行一个部门）|

## 三、关键决策

### 3.1 复用 `ai_video_pg` 容器（vs 独立部署）

**决策**：复用现有 postgres:16 容器（端口 5432，已 healthy 3 天），新建 `voc_bi` 数据库 + 独立用户 `voc_bi`。

**理由**：
- 同主机共享，免运维
- 独立 db / 用户保证 schema 隔离 + 权限边界
- 用户 `voc_bi` 仅有 `voc_bi` 数据库的 CTc 权限，看不到其他业务数据

**凭据**：`~/.paper2skills/voc_bi_pg.json` (chmod 600) — 与 `llm_keys.json` 同处理。

### 3.2 普通 VIEW vs MATERIALIZED VIEW

**决策**：当前全部用普通 VIEW。

**理由**：
- voc_label 689K 行 + 已索引 (review_id / tag_id / confidence)
- 测试查询 v_dept_topic_summary 5 秒内返回，BI 工具可承受
- MATERIALIZED 需要刷新机制 + 占用磁盘，**当前不需要**
- 后续如有性能瓶颈，可升级为 MATERIALIZED + REFRESH CONCURRENTLY 钩到 ETL 末端

### 3.3 sentiment_calibrated 类型混乱（D10 已知 bug）

ETL 内复用了 D10 同款 `_to_sentiment_float()` helper：将 `"positive"/"negative"/"neutral"` 映射为 `[1, -1, 0]`，浮点数原样传递，**postgres 字段统一为 REAL**。

## 四、性能数据

| 指标 | 值 |
|---|---:|
| ETL 总耗时 | 37 秒 |
| 吞吐量 | **9,853 reviews / 秒** |
| 写入 batch_size | 1,000 |
| psycopg2 execute_values 加速 | ~3-5x vs 单条 INSERT |
| 内存占用 | < 100 MB（流式读取）|
| postgres 连接数 | 1（单连接 + commit per flush）|

## 五、SQL 输出与 D10 MAA 一致性验证

### 5.1 v_dept_kpi 7 部门概览

| 部门 | distinct_tags | distinct_reviews | total_hits | avg_conf | pct_neg |
|---|---:|---:|---:|---:|---:|
| 产品中心/品线 | 62 | 101,295 | 136,205 | 0.8200 | 29.6% |
| 品牌市场中心 | 34 | 71,889 | 76,480 | 0.9211 | 19.8% |
| 全球客服与体验中心 | 36 | 45,187 | 58,664 | 0.8297 | 34.3% |
| 供应链中心 | 39 | 36,938 | 45,113 | 0.8343 | 48.1% |
| 电商运营部 | 34 | 29,988 | 33,145 | 0.7696 | 48.5% |
| 品控部 | 14 | 9,523 | 10,229 | 0.7791 | 92.4% |
| 质量与法规部 | 8 | 1,534 | 2,253 | 0.8170 | 94.5% |

**业务洞察一致**：品控部 + 质量与法规部 90%+ 负向，与 D10 MAA 一致。

### 5.2 产品中心/品线 Top-10

```
tag_id        | tag_cn   | polarity | hit_count | avg_confidence
--------------+----------+----------+-----------+----------------
TAG_GEN_E003  | 质量感知 | 正向     | 20,192    | 0.8916
TAG_GEN_E001  | 易用性   | 正向     | 16,351    | 0.8990
TAG_L1_040    | 延迟     | 负向     | 12,257    | 0.7431
TAG_GEN_E006  | 性能满意 | 正向     |  9,323    | 0.8815
TAG_GEN_E002  | 舒适体验 | 正向     |  9,159    | 0.8778
```

**与 D10 MAA Markdown 输出 100% 一致**（同样的 hit_count，同样的 confidence）。

## 六、产出文件

| 文件 | 用途 | 大小 |
|---|---|---:|
| `02-脚本工具/01-标签进化/sql/voc_bi_schema.sql` | DDL（4 表 + 索引）| ~3 KB |
| `02-脚本工具/01-标签进化/sql/voc_bi_views.sql` | 6 BI 视图 | ~6 KB |
| `02-脚本工具/01-标签进化/etl_to_postgres.py` | ETL 脚本 | ~10 KB |
| `~/.paper2skills/voc_bi_pg.json` | DB 凭据（chmod 600）| 207 B |
| postgres `voc_bi` 数据库 | 实时数据 | ~700 MB on disk |
| `04-输出结果/03-审计报告/phase7_d1_progress_report.md` | 本文档 | ~5 KB |

## 七、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | 普通 VIEW 性能在 Superset 大量并发时可能不够 | 中 | 监测；必要时改 MATERIALIZED + REFRESH 钩到 ETL |
| R2 | 数据库凭据存本地文件 | 低 | 当前是 dev；生产应迁到 secret manager |
| R3 | dim_tag.is_general 是基于字符串 "是" 判定 | 低 | 字典升级时同步刷新 |
| R4 | ETL 是 truncate-and-reload 全量；增量未实现 | 中 | 月度全量足够；如需实时改 upsert |

## 八、D2 解锁条件

| 前置 | 状态 |
|---|---|
| postgres voc_bi 可访问 | ✅ |
| 4 表数据完整 | ✅ |
| 6 视图查询正确（与 D10 一致）| ✅ |
| 凭据文件 | ✅ |

🟢 **D2 解锁**。下一步：

1. Docker 启 Superset (1.5 GB image)
2. 创建 dataset：voc_bi → 6 视图
3. 复制 D10 静态看板的图表到 Superset chart 库
4. 7 部门 dashboard 拼装
5. 用户登录 + 部门可见性分发

## 九、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-10 17:10 | D1.1 设计 star schema，写 voc_bi_schema.sql |
| 2026-05-10 17:15 | D1.2 复用 ai_video_pg 容器，新建 voc_bi DB |
| 2026-05-10 17:18 | DDL 应用，4 表创建成功 |
| 2026-05-10 17:25 | D1.3 etl_to_postgres.py 实现（流式 + execute_values）|
| 2026-05-10 17:28 | D1.4 ETL 跑 37 秒，364K reviews 全部导入 |
| 2026-05-10 17:32 | D1.5 6 SQL 视图全部创建 |
| 2026-05-10 17:35 | D1.6 SQL 输出与 D10 MAA 100% 一致 |
| 2026-05-10 17:40 | 本报告归档，D2 解锁 |

## 十、一行总结

> Phase 7 D1 数据底座**全部就绪**：voc_bi 数据库（4 表：dim_tag 267 + voc_review 364K + voc_label 690K + voc_brand_mention 33K）+ 6 BI 视图，ETL 37 秒导入完成，**SQL 输出与 D10 MAA Markdown 100% 一致**。下一步 D2 接入 Superset。
