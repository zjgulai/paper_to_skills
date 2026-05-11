---
name: phase7-architecture-diagrams
description: Phase 7 BI B 路径架构图集（Superset 实时交互看板）。10 张 Mermaid 图覆盖：系统分层、ETL 数据流、Superset 容器栈、6 SQL 视图依赖、12 charts × 8 dashboards 关系、10 native filters 作用域、用户交互时序、看板访问控制、Phase 7 时间线、与 Phase 6 C 路径的对比。所有图遵循 Lute 全局 Mermaid 规范（classDef + 横向布局）。
title: Phase 7 BI 架构图集
doc_type: diagrams
module: voc-nlp
topic: phase7-architecture-diagrams
status: stable
created: 2026-05-11
updated: 2026-05-11
owner: self
source: ai
---

# Phase 7 BI 架构图集（Mermaid 版）

> 配套白话汇报：[phase6-7-executive-brief.md](phase6-7-executive-brief.md)
> 配套高保真 HTML 版：[phase7-architecture-diagrams.html](phase7-architecture-diagrams.html)（Blueprinter 风格）
> 关联文档：[Superset_BI_SOP.md](../07-操作指南/Superset_BI_SOP.md) / [ETL_pipeline_SOP.md](../07-操作指南/ETL_pipeline_SOP.md)

## 图 1 — 系统分层（Phase 5 → Phase 6 → Phase 7）

> **看点**：Phase 7 在 Phase 5/6 之上加了 BI 层，但不动下层。

```mermaid
flowchart LR
    subgraph L1["Phase 5：AI 打标管道"]
        P5_RAW["原始 jsonl<br/>364K reviews"]
        P5_L0["L0 规则层<br/>关键词 + 品牌"]
        P5_L1["L1 LLM 闭集<br/>DeepSeek + Kimi"]
        P5_L2["L2 ABSA"]
        P5_L3["L3 画像 + NPS"]
        P5_OUT["phase5_intermediate_merged.jsonl"]
    end

    subgraph L2["Phase 6：质量提升"]
        P6_F["Method C 后处理过滤<br/>label_filter_kimi.py"]
        P6_OUT["phase6_d9_filtered.jsonl<br/>precision 0.896"]
    end

    subgraph L3["Phase 7：BI B 路径"]
        P7_ETL["ETL<br/>etl_to_postgres.py"]
        P7_DB[("voc_bi<br/>Postgres + 6 视图")]
        P7_SS["Superset 4.1.1<br/>Docker"]
        P7_DASH["8 dashboards<br/>12 charts<br/>10 filters"]
    end

    P5_RAW --> P5_L0 --> P5_L1 --> P5_L2 --> P5_L3 --> P5_OUT
    P5_OUT --> P6_F --> P6_OUT
    P6_OUT --> P7_ETL --> P7_DB --> P7_SS --> P7_DASH

    classDef phase5 fill:#e3f2fd,stroke:#1976d2,stroke-width:1px;
    classDef phase6 fill:#fff3e0,stroke:#f57c00,stroke-width:1px;
    classDef phase7 fill:#c8e6c9,stroke:#388e3c,stroke-width:2px;
    classDef store fill:#e0f2f1,stroke:#00796b,stroke-width:1px;

    class P5_RAW,P5_L0,P5_L1,P5_L2,P5_L3 phase5;
    class P5_OUT,P6_F,P6_OUT store;
    class P7_ETL,P7_SS,P7_DASH phase7;
    class P7_DB store;
```

## 图 2 — ETL 数据流（D1：37s 导入 364K）

```mermaid
flowchart LR
    SRC["phase6_d9_filtered.jsonl<br/>560M / 364,569 reviews"]
    DICT["tag_dictionary_v4.1.xlsx<br/>v4.1 字典"]

    subgraph ETL["etl_to_postgres.py（37 秒）"]
        PARSE["1. 解析 jsonl<br/>逐行 + Pydantic 校验"]
        FLATTEN["2. 扁平化 labels<br/>1 review × N labels → row"]
        ENRICH["3. 字典 enrich<br/>tag_id → tag_cn / tag_en / dept_owner / polarity"]
        BULK["4. Postgres COPY<br/>review + label 双表"]
    end

    subgraph DB[("voc_bi Postgres")]
        T_REVIEW["review<br/>364,569 行"]
        T_LABEL["label<br/>1,103,287 行"]
        V1["v_review_overview"]
        V2["v_label_with_dept"]
        V3["v_dept_topic_summary"]
        V4["v_label_brand"]
        V5["v_global_top_tags"]
        V6["v_dept_kpi"]
    end

    SRC --> PARSE
    DICT --> ENRICH
    PARSE --> FLATTEN --> ENRICH --> BULK
    BULK --> T_REVIEW
    BULK --> T_LABEL
    T_REVIEW -.SQL CREATE VIEW.-> V1
    T_LABEL -.JOIN.-> V2
    V2 -.GROUP BY.-> V3
    V2 -.GROUP BY.-> V4
    V2 -.GROUP BY.-> V5
    V2 -.GROUP BY.-> V6

    classDef src fill:#e3f2fd,stroke:#1976d2;
    classDef etl fill:#fff3e0,stroke:#f57c00;
    classDef table fill:#e0f2f1,stroke:#00796b,stroke-width:2px;
    classDef view fill:#f3e5f5,stroke:#7b1fa2;

    class SRC,DICT src;
    class PARSE,FLATTEN,ENRICH,BULK etl;
    class T_REVIEW,T_LABEL table;
    class V1,V2,V3,V4,V5,V6 view;
```

## 图 3 — Superset 容器栈（D2 部署）

```mermaid
flowchart LR
    subgraph HOST["宿主机"]
        BROWSER["浏览器<br/>localhost:8088"]
        CLI["Python REST API<br/>charts/filters factory"]
    end

    subgraph DOCKER["Docker Compose"]
        SS["voc_superset<br/>apache/superset:4.1.1"]
        REDIS["voc_superset_redis<br/>redis:7"]
        PG["voc_bi Postgres<br/>(host network)"]
    end

    SUPERSET_DB[("superset.db<br/>SQLite metadata")]

    BROWSER -- HTTP :8088 --> SS
    CLI -- REST API :8088 --> SS
    SS -. metadata .-> SUPERSET_DB
    SS -. cache .-> REDIS
    SS -- SQL :5432 --> PG

    classDef external fill:#e3f2fd,stroke:#1976d2;
    classDef container fill:#c8e6c9,stroke:#388e3c,stroke-width:2px;
    classDef store fill:#e0f2f1,stroke:#00796b;

    class BROWSER,CLI external;
    class SS,REDIS container;
    class PG,SUPERSET_DB store;
```

## 图 4 — 6 SQL 视图依赖关系

```mermaid
flowchart LR
    subgraph BASE["基表"]
        REVIEW["review<br/>review_id / data_source / product_line / proxy_nps / ..."]
        LABEL["label<br/>review_id / tag_id / polarity / confidence / ..."]
        DICT["tag_dictionary<br/>tag_id / tag_cn / dept_owner / polarity / aipl_node"]
    end

    V1["v_review_overview<br/>review 投影<br/>data_source / product_line / proxy_nps"]
    V2["v_label_with_dept<br/>label JOIN dictionary<br/>+ dept_owner / aipl_node"]
    V3["v_dept_topic_summary<br/>v_label_with_dept GROUP BY dept_owner, tag_id"]
    V4["v_label_brand<br/>v_label_with_dept WHERE has_brand"]
    V5["v_global_top_tags<br/>v_label_with_dept GROUP BY tag_id<br/>ORDER BY hit_count DESC LIMIT 30"]
    V6["v_dept_kpi<br/>v_label_with_dept GROUP BY dept_owner<br/>+ pct_negative / data_source_count"]

    REVIEW --> V1
    LABEL --> V2
    DICT --> V2
    V2 --> V3
    V2 --> V4
    V2 --> V5
    V2 --> V6

    classDef base fill:#e0f2f1,stroke:#00796b,stroke-width:2px;
    classDef view fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;
    class REVIEW,LABEL,DICT base;
    class V1,V2,V3,V4,V5,V6 view;
```

## 图 5 — 12 charts × 8 dashboards 映射

```mermaid
flowchart LR
    subgraph DASH1["Dashboard 1 - Overview"]
        D1[("VOC Overview")]
    end
    subgraph DEPT["Dashboards 2-8（7 部门）"]
        D2[("客服部")]
        D3[("产品研发部")]
        D4[("国际物流部")]
        D5[("市场部")]
        D6[("电商运营部")]
        D7[("品控部")]
        D8[("质量与法规部")]
    end

    subgraph CHARTS_OV["Overview Charts (5)"]
        C1["1. 7 部门标签命中数<br/>(dist_bar / v_dept_kpi)"]
        C2["2. 部门极性分布<br/>(stack_bar / v_dept_kpi)"]
        C3["3. Top-30 全局标签<br/>(table / v_global_top_tags)"]
        C13["13. 数据源分布<br/>(pie / v_review_overview)"]
        C14["14. Proxy NPS 分布<br/>(pie / v_review_overview)"]
    end

    subgraph CHARTS_DEPT["Per-Dept Charts (7)"]
        C6["6. 客服部 Top-10"]
        C7["7. 产品研发部 Top-10"]
        C8["8. 国际物流部 Top-10"]
        C9["9. 市场部 Top-10"]
        C10["10. 电商运营部 Top-10"]
        C11["11. 品控部 Top-10"]
        C12["12. 质量与法规部 Top-10"]
    end

    D1 --> C1
    D1 --> C2
    D1 --> C3
    D1 --> C13
    D1 --> C14

    D2 --> C6
    D3 --> C7
    D4 --> C8
    D5 --> C9
    D6 --> C10
    D7 --> C11
    D8 --> C12

    classDef dashOV fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef dashDept fill:#e8f5e9,stroke:#388e3c;
    classDef chartOV fill:#e3f2fd,stroke:#1976d2;
    classDef chartDept fill:#c8e6c9,stroke:#388e3c;

    class D1 dashOV;
    class D2,D3,D4,D5,D6,D7,D8 dashDept;
    class C1,C2,C3,C13,C14 chartOV;
    class C6,C7,C8,C9,C10,C11,C12 chartDept;
```

## 图 6 — 10 native filters 作用域

```mermaid
flowchart LR
    subgraph OV_F["Overview Filters (3)"]
        F1["数据源<br/>data_source<br/>5 options"]
        F2["产品线<br/>product_line<br/>7 options"]
        F3["Proxy NPS<br/>proxy_nps<br/>3 options"]
    end

    subgraph DEPT_F["Dept Filters (7)"]
        FP["情感极性 polarity<br/>3 options：正向 / 负向 / 中性"]
    end

    subgraph SCOPE_OV["chartsInScope = [13, 14]"]
        S1["数据源饼图"]
        S2["NPS 饼图"]
    end

    subgraph SCOPE_DEPT["chartsInScope = 各部门 Top-10"]
        SD["chart 6-12<br/>逐 dashboard 一一对应"]
    end

    F1 --> S1
    F1 --> S2
    F2 --> S1
    F2 --> S2
    F3 --> S1
    F3 --> S2
    FP --> SD

    classDef filterOV fill:#fff3e0,stroke:#f57c00;
    classDef filterDept fill:#e8f5e9,stroke:#388e3c;
    classDef scope fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;

    class F1,F2,F3 filterOV;
    class FP filterDept;
    class S1,S2,SD,SCOPE_OV,SCOPE_DEPT scope;
```

## 图 7 — 用户交互时序（部门 polarity 过滤）

```mermaid
sequenceDiagram
    autonumber
    actor U as 业务用户
    participant B as 浏览器
    participant SS as Superset
    participant PG as voc_bi
    participant CACHE as Redis

    rect rgb(227, 242, 253)
        Note over U,SS: 首次加载 dashboard
        U->>B: 打开 /dashboard/3/
        B->>SS: GET /dashboard/3/
        SS->>PG: SELECT * FROM v_dept_topic_summary<br/>WHERE dept_owner='产品研发部'<br/>ORDER BY hit_count DESC LIMIT 10
        PG-->>SS: 10 行结果
        SS->>CACHE: SET cache_key (TTL 5min)
        SS-->>B: HTML + chart-data API 调用
        B-->>U: 渲染 ECharts<br/>top-1: 质量感知 20.2k
    end

    rect rgb(200, 230, 201)
        Note over U,SS: 应用 polarity=负向
        U->>B: 选择「情感极性=负向」+ Apply
        B->>SS: POST /api/v1/chart/data<br/>+ filter_state.value=['负向']
        SS->>PG: SELECT * FROM v_dept_topic_summary<br/>WHERE dept_owner='产品研发部'<br/>  AND polarity='负向'<br/>ORDER BY hit_count DESC LIMIT 10
        PG-->>SS: 10 行（不同结果）
        SS-->>B: chart-data JSON
        B-->>U: 重渲染<br/>top-1: 延迟 12.3k<br/>显示 "Applied filters (1)" badge
    end
```

## 图 8 — 看板访问控制（当前 + Phase 8 候选）

```mermaid
flowchart LR
    subgraph NOW["当前（Phase 7 末态）"]
        U_NOW["用户"] --> LOCAL["localhost:8088<br/>admin / voc_admin_2026"]
        LOCAL --> SS_NOW["Superset"]
    end

    subgraph FUTURE["Phase 8 候选"]
        U_FU["7 部门用户"] --> SSO["SSO / 域账号"]
        SSO --> NGINX["Nginx + HTTPS<br/>voc-bi.内网域名"]
        NGINX --> SS_FU["Superset"]
        SS_FU --> RBAC["RBAC<br/>部门隔离 row-level filter"]
        FEISHU["飞书工作台"] --> NGINX
        DINGTALK["钉钉工作台"] --> NGINX
    end

    classDef now fill:#fff3e0,stroke:#f57c00;
    classDef future fill:#e8f5e9,stroke:#388e3c,stroke-dasharray: 5 5;

    class U_NOW,LOCAL,SS_NOW now;
    class U_FU,SSO,NGINX,SS_FU,RBAC,FEISHU,DINGTALK future;
```

## 图 9 — Phase 7 时间线（D1-D4 ~7h）

```mermaid
gantt
    title Phase 7 BI B 路径开发时间线
    dateFormat YYYY-MM-DD
    axisFormat %m-%d

    section D1 数据底座
    voc_bi 数据库 + ETL          :done, d1a, 2026-05-08, 1d
    6 SQL 视图                    :done, d1b, after d1a, 0d

    section D2 Superset 部署
    Docker Compose                :done, d2a, 2026-05-09, 1d
    voc_bi 接入 + 6 datasets       :done, d2b, after d2a, 0d

    section D3 Charts + Dashboards
    12 charts (REST API)          :done, d3a, 2026-05-10, 1d
    8 dashboards 装配              :done, d3b, after d3a, 0d
    Playwright 验证                :done, d3c, after d3b, 0d
    导出 ZIP                       :done, d3d, after d3c, 0d

    section D4 Native Filters
    Probe schema                  :done, d4a, 2026-05-11, 1d
    Filters factory + 10 filters   :done, d4b, after d4a, 0d
    Browser 端到端验证             :done, d4c, after d4b, 0d
    修订（D4 fix）                  :done, crit, d4d, after d4c, 0d
```

## 图 10 — C 路径 vs B 路径对比

```mermaid
flowchart LR
    DATA["phase6_d9_filtered.jsonl"]

    subgraph C_PATH["C 路径（Phase 6 D10）"]
        BIDG["bi_dashboard_generator.py<br/>离线渲染"]
        HTML["dashboard-2026-W19.html<br/>125KB 单文件"]
        EMAIL["邮件 / IM 分发"]
    end

    subgraph B_PATH["B 路径（Phase 7）⭐"]
        ETL["ETL → voc_bi"]
        SS["Superset Docker"]
        WEB["浏览器实时交互"]
    end

    DATA --> BIDG --> HTML --> EMAIL
    DATA --> ETL --> SS --> WEB

    classDef cpath fill:#fff3e0,stroke:#f57c00,stroke-dasharray: 5 5;
    classDef bpath fill:#c8e6c9,stroke:#388e3c,stroke-width:2px;

    class BIDG,HTML,EMAIL cpath;
    class ETL,SS,WEB bpath;
```

| 维度 | C 路径（HTML） | B 路径（Superset） |
|---|---|---|
| 上线时间 | Phase 6 D10 | Phase 7 D1-D4 |
| 交付方式 | 单文件 HTML | 实时 web app |
| 交互 | 静态 | filter / drill-down |
| 分发 | 邮件 / IM 附件 | URL |
| 维护 | 手动重生 | REST API 自动化 |
| 适用场景 | 周报快照 / 离线档案 | 日常决策 / 探索 |

**两条路径互补**——C 路径作快照存档，B 路径作日常工具。

---

## 引用脚本与产物

| 图涉及 | 脚本 / 文件 |
|---|---|
| 图 2 ETL | [etl_to_postgres.py](../../02-脚本工具/01-标签进化/etl_to_postgres.py) |
| 图 3 Docker | [docker-compose.yml](../../02-脚本工具/01-标签进化/docker/docker-compose.yml) |
| 图 4 SQL 视图 | [voc_bi_views.sql](../../02-脚本工具/01-标签进化/sql/voc_bi_views.sql) |
| 图 5 Charts | [superset_charts_factory.py](../../02-脚本工具/01-标签进化/docker/superset_charts_factory.py) |
| 图 6 Filters | [superset_filters_factory.py](../../02-脚本工具/01-标签进化/docker/superset_filters_factory.py) |
| 图 7 时序 | Playwright 实测：[phase7_d4_progress_report.md §四](../../04-输出结果/03-审计报告/phase7_d4_progress_report.md) |
| 图 10 C 路径 | [bi_dashboard_generator.py](../../02-脚本工具/01-标签进化/bi_dashboard_generator.py) → [phase6_html_dashboard/](../../00-归档资料/phase6_html_dashboard/) |
