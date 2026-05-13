---
name: etl-pipeline-sop
description: ETL 流水线 SOP — 高详细度操作手册。从原始 jsonl + 字典 Excel 一键跑通到 voc_bi Postgres 数据库。覆盖：数据库准备、ETL 执行、6 SQL 视图建立、验证、增量重跑、故障处置。面向接手的工程同事，逐行命令 + SQL 校验 + 性能调优。
title: ETL Pipeline SOP — jsonl → voc_bi → BI 看板
doc_type: sop
module: voc-nlp
topic: etl-pipeline-operations
status: stable
created: 2026-05-11
updated: 2026-05-11
owner: self
source: ai
audience: engineer
---

# ETL Pipeline SOP — jsonl → voc_bi → BI 看板

> **文档定位**：从原始 `phase6_d9_filtered.jsonl`（560M）+ 字典 `tag_dictionary_v4.1.xlsx`，到 Postgres `voc_bi` 里 4 张表 + 6 个视图就绪供 Superset 用，逐步走通。
>
> **预期耗时**：完整流程 ~5 分钟（37s ETL + 余下网络/初始化）
>
> **关联文档**：
> - 下游 BI：[Superset_BI_SOP.md](Superset_BI_SOP.md)
> - 架构图：[phase7-architecture-diagrams.md §2 ETL Data Flow](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md#图-2--etl-数据流d137s-导入-364k)
> - 上游：Phase 6 D9 产出 `phase6_d9_filtered.jsonl`

---

## §0 目录

1. [§1 数据流总图](#§1-数据流总图)
2. [§2 Postgres voc_bi 数据库准备](#§2-postgres-voc_bi-数据库准备)
3. [§3 执行 ETL 导入](#§3-执行-etl-导入)
4. [§4 6 SQL 视图建立](#§4-6-sql-视图建立)
5. [§5 验证检查清单](#§5-验证检查清单)
6. [§6 增量重跑 / 字典升级](#§6-增量重跑--字典升级)
7. [§7 性能调优](#§7-性能调优)
8. [§8 故障处置判断树](#§8-故障处置判断树)
9. [§A 附录：表结构 + 视图 + 字段语义](#§a-附录表结构--视图--字段语义)

---

## §1 数据流总图

```
┌─────────────────────────────────┐
│ 输入                             │
├─────────────────────────────────┤
│ • phase6_d9_filtered.jsonl 560M  │  → Phase 6 D9 Method C 过滤后产物
│ • tag_dictionary_v4.1.xlsx 2.6M  │  → v4.1 字典（643 tag_ids 元信息）
│ • ~/.paper2skills/voc_bi_pg.json │  → Postgres 连接配置
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ etl_to_postgres.py                │
├─────────────────────────────────┤
│ 1. Stream jsonl (O(1) memory)     │
│ 2. 字典 enrich                    │
│ 3. Batch INSERT (1000/batch)      │
│ 4. TRUNCATE first → idempotent   │
└────────────┬────────────────────┘
             │ ~37 秒
             ▼
┌─────────────────────────────────┐
│ voc_bi Postgres 4 张基表          │
├─────────────────────────────────┤
│ • dim_tag           267 rows     │
│ • voc_review        364,569      │
│ • voc_label         ~689K        │
│ • voc_brand_mention ~varies      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ voc_bi_views.sql 6 视图           │
├─────────────────────────────────┤
│ • v_review_overview              │
│ • v_label_with_dept              │
│ • v_dept_topic_summary           │
│ • v_label_brand                  │
│ • v_global_top_tags              │
│ • v_dept_kpi                     │
└────────────┬────────────────────┘
             │
             ▼
        Superset 8 dashboards
```

---

## §2 Postgres voc_bi 数据库准备

### 2.1 选项 A：Docker 起一个 Postgres（推荐：干净 + 隔离）

```bash
docker run -d \
  --name voc_bi_pg \
  -e POSTGRES_USER=voc_user \
  -e POSTGRES_PASSWORD=voc_pass \
  -e POSTGRES_DB=voc_bi \
  -p 5432:5432 \
  -v voc_bi_pg_data:/var/lib/postgresql/data \
  postgres:15

# 验证（约 5-10s 后启动好）
sleep 10
docker exec voc_bi_pg pg_isready -U voc_user -d voc_bi
# 期望：localhost:5432 - accepting connections
```

### 2.2 选项 B：连已有 Postgres 实例

```bash
# 在你的 Postgres 上创建用户和库
psql -h <pg_host> -U postgres <<EOF
CREATE USER voc_user WITH PASSWORD 'voc_pass';
CREATE DATABASE voc_bi OWNER voc_user;
GRANT ALL PRIVILEGES ON DATABASE voc_bi TO voc_user;
EOF
```

### 2.3 写入 ETL 连接配置

```bash
mkdir -p ~/.paper2skills
cat > ~/.paper2skills/voc_bi_pg.json <<EOF
{
  "host": "localhost",
  "port": 5432,
  "database": "voc_bi",
  "user": "voc_user",
  "password": "voc_pass"
}
EOF
chmod 600 ~/.paper2skills/voc_bi_pg.json

# 验证连接
python3 -c "
import json, psycopg2
cfg = json.load(open('$HOME/.paper2skills/voc_bi_pg.json'))
conn = psycopg2.connect(**cfg)
print('✅ 连接成功')
conn.close()
"
```

### 2.4 创建 schema（基表）

```bash
cd /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC
psql -h localhost -U voc_user -d voc_bi -f research/02-脚本工具/01-标签进化/sql/voc_bi_schema.sql
# 期望输出 4 个 CREATE TABLE 语句执行成功

# 验证
psql -h localhost -U voc_user -d voc_bi -c "\dt"
# 期望看到 4 张表：dim_tag / voc_review / voc_label / voc_brand_mention
```

---

## §3 执行 ETL 导入

### 3.1 前置检查

```bash
cd /Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC

# 1. 输入文件存在
ls -lh research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl
# 期望：560M（约 36 万行）

# 2. 字典存在
ls -lh research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx
# 期望：~2.6M

# 3. Python 依赖
python3 -c "import psycopg2, openpyxl, pydantic; print('✅ deps ok')"
# 如缺：pip install psycopg2-binary openpyxl pydantic
```

### 3.2 执行 ETL

```bash
python3 research/02-脚本工具/01-标签进化/etl_to_postgres.py \
  --input research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl \
  --dict research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx

# 期望输出（约 37 秒）：
#   ⏳ Connecting to voc_bi @ localhost:5432
#   ⏳ Loading dictionary: 267 tags from sheet 01_通用标签主表
#   ⏳ TRUNCATE dim_tag, voc_review, voc_label, voc_brand_mention
#   ⏳ Inserting dim_tag: 267 rows
#   ⏳ Streaming jsonl, batch_size=1000
#     batch 100 - 100,000 reviews
#     batch 200 - 200,000 reviews
#     ...
#     batch 365 - 364,569 reviews
#   ✅ voc_review: 364,569 rows
#   ✅ voc_label:  689,XXX rows
#   ✅ voc_brand_mention: XX,XXX rows
#   ✅ Total: 37.X seconds
```

### 3.3 即时验证

```bash
psql -h localhost -U voc_user -d voc_bi <<EOF
SELECT 'dim_tag' AS t, count(*) FROM dim_tag
UNION ALL SELECT 'voc_review', count(*) FROM voc_review
UNION ALL SELECT 'voc_label', count(*) FROM voc_label
UNION ALL SELECT 'voc_brand_mention', count(*) FROM voc_brand_mention;
EOF

# 期望：
#  t                  | count
# --------------------+--------
#  dim_tag            |    267
#  voc_review         | 364569
#  voc_label          | ~689000
#  voc_brand_mention  | ~XXXXX
```

---

## §4 6 SQL 视图建立

### 4.1 执行

```bash
psql -h localhost -U voc_user -d voc_bi \
  -f research/02-脚本工具/01-标签进化/sql/voc_bi_views.sql

# 期望：6 个 CREATE OR REPLACE VIEW 语句执行成功
```

### 4.2 验证

```bash
psql -h localhost -U voc_user -d voc_bi -c "\dv"
# 期望 6 视图：
#   v_review_overview
#   v_label_with_dept
#   v_dept_topic_summary
#   v_label_brand
#   v_global_top_tags
#   v_dept_kpi

# 每个视图行数检查
psql -h localhost -U voc_user -d voc_bi <<EOF
SELECT 'v_review_overview' AS v, count(*) FROM v_review_overview
UNION ALL SELECT 'v_label_with_dept', count(*) FROM v_label_with_dept
UNION ALL SELECT 'v_dept_topic_summary', count(*) FROM v_dept_topic_summary
UNION ALL SELECT 'v_label_brand', count(*) FROM v_label_brand
UNION ALL SELECT 'v_global_top_tags', count(*) FROM v_global_top_tags
UNION ALL SELECT 'v_dept_kpi', count(*) FROM v_dept_kpi;
EOF
```

期望大致：

| view | rows 量级 |
|---|---|
| v_review_overview | ~364K |
| v_label_with_dept | ~689K |
| v_dept_topic_summary | ~2K（按 dept × tag 聚合） |
| v_label_brand | ~5-50K |
| v_global_top_tags | 30 |
| v_dept_kpi | 7 |

---

## §5 验证检查清单

跑完 ETL + 建视图后，逐项确认：

```bash
psql -h localhost -U voc_user -d voc_bi <<'EOF'
\echo '✅ 1. data_source 分布（应有 5 个数据源）'
SELECT data_source, count(*) FROM v_review_overview GROUP BY data_source ORDER BY 2 DESC;

\echo '✅ 2. proxy_nps 分布（应有 promoter / passive / detractor）'
SELECT proxy_nps, count(*) FROM v_review_overview GROUP BY proxy_nps;

\echo '✅ 3. 7 个部门都有数据'
SELECT dept_owner, count(*) FROM v_dept_kpi GROUP BY dept_owner;

\echo '✅ 4. polarity 三类齐全'
SELECT polarity, count(*) FROM v_label_with_dept GROUP BY polarity;

\echo '✅ 5. 一个具体部门的 top-3 话题'
SELECT * FROM v_dept_topic_summary WHERE dept_owner='产品中心' ORDER BY hit_count DESC LIMIT 3;

\echo '✅ 6. 全局 top-3 标签'
SELECT * FROM v_global_top_tags LIMIT 3;
EOF
```

期望结果：

| 检查 | 期望 |
|---|---|
| 1 | 5 行：amazon_competitor 194K / trustpilot 100K / zendesk 47K / momcozy 20K / reddit 3K |
| 2 | 3 行 |
| 3 | 7 行（客服 / 产品研发 / 国际物流 / 市场 / 电商运营 / 品控 / 质量与法规） |
| 4 | 3 行（正向 / 负向 / 中性） |
| 5 | 产品中心 top-3：质量感知 ~20.2k / 易用性 ~16.4k / 延迟 ~12.3k |
| 6 | 第一名 hit_count > 30,000 |

**全部通过** = ETL 成功，下一步去 [Superset_BI_SOP.md §2](Superset_BI_SOP.md#§2-从零启动-superset) 起 Superset。

---

## §6 增量重跑 / 字典升级

### 6.1 场景：上游 phase6_d9_filtered.jsonl 更新了

```bash
# ETL 是 idempotent 的（TRUNCATE before insert），直接重跑
python3 research/02-脚本工具/01-标签进化/etl_to_postgres.py \
  --input research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl \
  --dict research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx

# 视图基于基表，自动反映新数据，不需要重建

# 但 Superset 缓存可能有旧数据
docker exec voc_superset_redis redis-cli FLUSHDB
```

### 6.2 场景：字典升级（v4.1 → v5.0）

```bash
# 1. 用新字典重跑 ETL（dim_tag 会被新内容替换）
python3 research/02-脚本工具/01-标签进化/etl_to_postgres.py \
  --input research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl \
  --dict research/04-输出结果/01-字典版本/tag_dictionary_v5.0.xlsx

# 2. 视图自动反映新 dept_owner 映射

# 3. 检查 Superset chart 配置中是否有硬编码部门名
# （超过 7 个部门？某 chart 引用了消失的 dept_owner？）
grep -n "dept_owner" research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py

# 4. 如有变化，重跑 Superset 工厂
python3 research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py
python3 research/02-脚本工具/01-标签进化/docker/superset_filters_factory.py
```

### 6.3 场景：只改某个视图（不动数据）

```bash
# 比如要新增 v_dept_topic_summary 的某列
vim research/02-脚本工具/01-标签进化/sql/voc_bi_views.sql
# 编辑相应 CREATE OR REPLACE VIEW

# 重新应用（不影响基表）
psql -h localhost -U voc_user -d voc_bi -f research/02-脚本工具/01-标签进化/sql/voc_bi_views.sql

# Superset 缓存清
docker exec voc_superset_redis redis-cli FLUSHDB
```

---

## §7 性能调优

### 7.1 ETL 速度优化

当前：~37s / 364K reviews + 689K labels。

| 瓶颈 | 现状 | 改进点 |
|---|---|---|
| jsonl 解析 | streaming Pydantic 校验 | 已是 O(1) 内存 |
| 字典 enrich | 内存 lookup | 已是 O(1) per row |
| Postgres INSERT | `execute_values` batch 1000 | 可调到 5000（视内存） |
| 网络 | localhost | 跨机时考虑 COPY FROM STDIN |

### 7.2 视图查询性能

```bash
# 慢查询诊断
psql -h localhost -U voc_user -d voc_bi -c "
EXPLAIN ANALYZE
SELECT * FROM v_dept_topic_summary
WHERE dept_owner='产品中心' AND polarity='负向'
ORDER BY hit_count DESC LIMIT 10;
"
# 期望：Execution Time < 100ms（无索引）
```

### 7.3 加索引（如果数据量增长 10x+）

```sql
-- 在 voc_label 上加常用查询索引
CREATE INDEX idx_label_dept_polarity
ON voc_label(dept_owner, polarity, hit_count DESC);

CREATE INDEX idx_review_data_source ON voc_review(data_source);
CREATE INDEX idx_review_proxy_nps ON voc_review(proxy_nps);
```

### 7.4 Materialized View（如查询慢且数据更新不频繁）

```sql
-- 把 v_dept_topic_summary 改物化（节省 Superset 查询时间）
DROP VIEW v_dept_topic_summary CASCADE;
CREATE MATERIALIZED VIEW v_dept_topic_summary AS
SELECT ... FROM v_label_with_dept GROUP BY ...;

-- ETL 后刷新
REFRESH MATERIALIZED VIEW v_dept_topic_summary;
```

---

## §8 故障处置判断树

```
            ┌─ 跑 ETL 时报错 ─┐
            │              │
       连接失败 ──► §8.1 Postgres 连接
            │
            ▼
       字典读不进 ──► §8.2 Excel 解析
            │
            ▼
       某行 jsonl 报错 ──► §8.3 数据格式
            │
            ▼
       INSERT 太慢 ──► §7 性能调优
            │
            ▼
       完成但行数不对 ──► §8.4 数据完整性
            │
            ▼
       视图建不出来 ──► §8.5 视图依赖
```

### §8.1 Postgres 连接失败

```bash
# 诊断
python3 -c "
import json, psycopg2
cfg = json.load(open('$HOME/.paper2skills/voc_bi_pg.json'))
print(cfg)
try:
    conn = psycopg2.connect(**cfg)
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
"

# 常见错误
# - 'could not connect' → Postgres 没起来 → docker start voc_bi_pg
# - 'authentication failed' → password 错 → 改 ~/.paper2skills/voc_bi_pg.json
# - 'database voc_bi does not exist' → 跑 §2.2 建库
# - 'permission denied' → 跑 §2.2 的 GRANT
```

### §8.2 字典 Excel 解析失败

```bash
# 诊断
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx')
print('Sheets:', wb.sheetnames)
ws = wb['01_通用标签主表']
print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')
"
# 期望：含 '01_通用标签主表' sheet，267+ rows

# 如果缺 sheet
# → 用了错版本字典；检查文件路径和版本号
```

### §8.3 数据格式错误（某行 jsonl 报错）

```bash
# 找出报错行号（ETL 输出有提示）
sed -n '12345p' research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl | python3 -m json.tool

# 常见
# - 字段缺失 → 上游 Phase 6 D9 过滤逻辑有 bug，需重跑 label_filter_kimi.py
# - sentiment_calibrated 是 'positive' 而非浮点 → ETL 已有 _to_sentiment_float 兼容
# - 编码错误 → 用 file -bi 检查文件编码，应为 utf-8
```

### §8.4 行数不对

```bash
# 原始 jsonl 行数
wc -l research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl
# 期望：364,569

# Postgres voc_review 行数
psql -h localhost -U voc_user -d voc_bi -c "SELECT count(*) FROM voc_review"
# 期望相同

# 如果少了
# → ETL 中途断了 → 重跑（idempotent）
# → 数据有 review_id duplicate → 检查上游
```

### §8.5 视图建不出来

```bash
# 看具体错误
psql -h localhost -U voc_user -d voc_bi \
  -f research/02-脚本工具/01-标签进化/sql/voc_bi_views.sql 2>&1 | grep ERROR

# 常见
# - 'column "xxx" does not exist' → 基表 schema 和视图 SQL 不匹配
# - 'permission denied' → 跑 §2.2 GRANT
# - 'relation v_label_with_dept does not exist' → 视图依赖顺序错，重跑整个 SQL 文件
```

---

## §A 附录：表结构 + 视图 + 字段语义

### A.1 4 张基表

#### `dim_tag`（267 rows）

| 列 | 类型 | 说明 |
|---|---|---|
| `tag_id` | TEXT (PK) | 标签 ID（如 `TAG_L1_087`） |
| `tag_cn` | TEXT | 中文名 |
| `tag_en` | TEXT | 英文名 |
| `dept_owner` | TEXT | 主责部门 |
| `polarity` | TEXT | 极性（正向/负向/中性） |
| `aipl_node` | TEXT | AIPL 节点 |
| `biz_action` | TEXT | 业务动作建议 |

#### `voc_review`（364,569 rows）

| 列 | 类型 | 说明 |
|---|---|---|
| `review_id` | TEXT (PK) | 评论唯一 ID |
| `text` | TEXT | 评论原文 |
| `data_source` | TEXT | amazon_competitor / trustpilot / zendesk / momcozy / reddit |
| `platform` | TEXT | 平台子类 |
| `product_line` | TEXT | 产品品线（吸奶器 / 内衣服饰 / ...） |
| `language` | TEXT | 语言代码 |
| `proxy_nps` | TEXT | promoter / passive / detractor |
| `rating` | REAL | 星级（1-5） |
| `sentiment_polarity` | REAL | 整体情感 [-1, 1] |
| `aipl_stage` | TEXT | A/I/P/L 阶段 |
| `persona_derived` | TEXT | 衍生画像 |
| `brand_count` | INT | 提及品牌数 |
| `quality_score` | REAL | 文本质量分 |
| `n_tags` | INT | 该 review 的标签数 |

#### `voc_label`（~689K rows）

| 列 | 类型 | 说明 |
|---|---|---|
| `label_id` | BIGINT (PK) | 自增 |
| `review_id` | TEXT (FK) | → voc_review |
| `tag_id` | TEXT (FK) | → dim_tag |
| `confidence` | REAL | LLM 给的置信度 |
| `confidence_original` | REAL | 重赋前 |
| `polarity` | TEXT | 该标签的极性 |
| `sentiment_calibrated` | REAL | 校准后情感 [-1, 1] |
| `label_source` | TEXT | rule / llm / consensus / persona |

#### `voc_brand_mention`（~varies rows）

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | BIGSERIAL (PK) | |
| `review_id` | TEXT (FK) | |
| `brand_name` | TEXT | 提及的品牌 |
| `is_competitor` | BOOLEAN | |

### A.2 6 个视图（在 [voc_bi_views.sql](../../02-脚本工具/01-标签进化/sql/voc_bi_views.sql) 中定义）

详见 [phase7-architecture-diagrams.md §4](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md#图-4--6-sql-视图依赖关系)。

### A.3 字段语义关键约定

- **`polarity` vs `sentiment_polarity` vs `sentiment_calibrated`**：
  - `polarity`（TEXT）：标签级别的离散极性（正向/负向/中性）— 来自字典 dim_tag.polarity
  - `sentiment_polarity`（REAL [-1,1]）：review 整体情感连续值 — 来自 NLP 模型
  - `sentiment_calibrated`（REAL [-1,1]）：label 级别校准后的情感强度 — 来自 ABSA + 校准

- **`confidence` vs `confidence_original`**：
  - `confidence_original`：LLM 原始输出
  - `confidence`：Phase 6 D3 离线 confidence_rebalancer 重赋后值（Method C 用这个判过滤）

- **`label_source` 取值**：
  - `rule` — L0 规则层（Phase 4 保留）
  - `llm` — L1 LLM 闭集层
  - `consensus` — L1 共识增强
  - `persona` — L3 画像规则

---

## 附：关联文档

| 类型 | 链接 |
|---|---|
| 下游 BI | [Superset_BI_SOP.md](Superset_BI_SOP.md) |
| 架构图 | [phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md) |
| Phase 7 D1 进度报告 | [phase7_d1_progress_report.md](../../04-输出结果/03-审计报告/phase7_d1_progress_report.md) |
| 字典管理 | [tag_dictionary_v4.1.xlsx](../../04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx) |
| Phase 6 D9 上游 | [phase6_d9_progress_report.md](../../04-输出结果/03-审计报告/phase6_d9_progress_report.md) |

---

> **SOP 维护约定**：当 voc_bi 表结构、字典字段、ETL 脚本接口变化时，更新本文 §A 附录。如果 ETL 改换技术栈（如换成 dbt），本文挪至 00-归档资料/ 并建新版。
