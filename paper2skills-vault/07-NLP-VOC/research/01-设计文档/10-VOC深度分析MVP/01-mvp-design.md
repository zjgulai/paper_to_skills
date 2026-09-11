---
plan_id: voc-deep-analysis-mvp
status: L0_DRAFT_AWAITING_REVIEW
created: 2026-05-14
owner: voc-nlp
intent: 基于 SGCS 方法论 + 现有 v4.3 字典 + 内部 5 数据源，2 周交付 3 张深度分析看板
mvp_parameters:
  Q1_external_voc: B (不引入)
  Q2_atomic_indicators: B (50 个自定义 · 模板参考 SGCS)
  Q3_aipl_node_scheme: B (L1-L4 4 节点)
  Q4_timestamp: A (loaded_at proxy)
  Q5_country: A (language 推断)
  Q6_deploy_target: A (同 Superset 实例)
  Q7_indicator_owner: 2A (Sisyphus 出初稿)
  Q8_timestamp_strategy: 3A (loaded_at proxy)
---

# VOC 深度分析看板 · MVP 设计文档

## 一、目标

把 SGCS 已交付的「7 节点诊断 + 3 专题 + 机会清单 + 部门动作分发」方法论，
落地为 3 张 Superset 看板，在 v4.3 字典 + 现有生产 BI 上扩展。

## 二、范围（什么会做、什么不做）

### 会做
- 50 个原子指标定义 + 与现有 267 标签的映射
- 4 节点（L1-L4）AIPL 得分公式 · 套用 SGCS：S = (正向 + 0.5 中性) / 总数 × 100
- 时间维度（用 loaded_at proxy）
- 国家维度（language → country）
- 3 张深度看板：D-Health / D-Diag / D-Action
- 与现有 8 个部门看板兼容共存

### 不做
- 引入外部 Amazon 竞品 VOC
- 7 节点细分（A/I/P1/P2/L1/L2/L3）
- 重做 v4.3 字典（267 标签 + 8 部门保持不变）
- 改动现有 8 个 Superset 看板（仅追加）
- LLM 内容生成
- 真实 timestamp（爬虫补抓）

## 三、关键决策回顾

### Q4 timestamp 用 loaded_at proxy

实测确认 `voc_review.loaded_at` 已存在（TIMESTAMP WITH TIME ZONE，default now()）：

- **业务含义**：评论被 ETL 入库的时间，非用户写评论时间
- **精度**：批次级，按 ETL 跑批日聚合
- **局限**：amazon/trustpilot/reddit 的真实写评时间需要爬虫补抓
- **当前用途**：作为「数据可见时间」用，能做月环比 / 季度趋势
- **upgrade path**：未来真实 timestamp 入库后切换字段，视图不变

### Q5 language → country 映射

依据实际数据分布（L0.5 探明）：

| language | data_source | → country | 依据 |
|---|---|---|---|
| en | amazon_competitor | **US** | 美国主导（94K rows）|
| en | trustpilot | **UK** | Trustpilot 英语主要 UK |
| en | zendesk | **OTHER** | 客服多国，无法细分 |
| en | momcozy / reddit | **US** | 自有渠道默认 US |
| de | * | **DE** | 28K rows 全德语 |
| fr | * | **FR** | 21K rows 全法语 |
| es | * | **ES** | 632 rows |
| it | * | **IT** | 301 rows |
| 其他 | * | **OTHER** | pl / nl / sv 等 |

5 主要国家：US / UK / DE / FR / OTHER

## 四、视图层架构（L3 阶段产出）

```
voc_review (已有)
   ├── 新列 country (L1.2)
   └── 新列 loaded_at 已有，直接当 ts_inferred
   
dim_tag (已有)
   └── 新列 atomic_indicator_id (L2.1，关联 50 SAT_xxx)
   
voc_label (已有，不动)

新视图：
   ├── v_aipl_node_score      L1-L4 节点得分按 month × country × product_line
   ├── v_atomic_indicator_score    50 原子指标得分按 month × country × product_line
   ├── v_country_dept_health   country × dept_owner × month 健康度
   └── v_proxy_nps_calibrated      按 04-NPS 偏差分析方法校准

现有 6 视图（保持不动）：
   ├── v_review_overview
   ├── v_label_with_dept
   ├── v_dept_topic_summary
   ├── v_label_brand
   ├── v_global_top_tags
   └── v_dept_kpi
```

## 五、3 张深度看板设计

### D-Health · AIPL 4 节点健康度看板（5 charts）

| Chart | viz_type | datasource | 业务问题 |
|---|---|---|---|
| KPI 卡 × 4 | big_number | v_aipl_node_score | L1/L2/L3/L4 月度总分 + 月环比 |
| 节点雷达图 | radar | v_aipl_node_score | 4 节点同心圈，按品类分组 |
| 节点得分趋势线 | line | v_aipl_node_score | 时间 × 节点得分 4 条线 |
| 品类对比柱图 | dist_bar | v_aipl_node_score | 品类 × 节点得分 |
| 失血指标 Top 10 | table | v_atomic_indicator_score | sat_score 升序前 10 |

### D-Diag · 3 专题诊断看板（共 12-15 charts）

#### 专题 1：承诺偏差诊断
- 关键指标链表（A 节点原子指标 × 极性分布）
- 时间趋势 + 国家差异

#### 专题 2：信任恢复效率
- L4_01 响应速度 vs L4_02 解决效率 vs L4_06 信任恢复 三卡对比
- 「响应快 vs 解决差」缺口柱图

#### 专题 3：国家 × 品类 Gap 聚类
- 国家健康度表（US/UK/DE/FR/OTHER × 4 品类）
- 跨国家重复性 Top 5

### D-Action · 7 部门行动分发看板（1 dashboard + 1 table）

合并到现有 7 部门 dashboard 末尾。
字段：动作编号 / 主题 / 触发专题 / 主责 / 协同 / 优先级 / 截止 / 成功判定

## 六、执行 checkpoint（7 个 GO 点）

| Phase | 产出 | 用户审 | 工日 |
|---|---|---|---|
| L0 | mvp-design.md + sat-indicators-draft.md + 备份就绪 | ✓ | 1 |
| L1 | voc_review 新增 ts_inferred + country 列 + 验证 | ✓ | 0.5 |
| L2 | 50 原子指标定义 + 267 标签映射 + dim_tag v4.4 | ✓ | 2 |
| L3 | 4 新视图 + 性能验证 | ✓ | 1 |
| L4 | D-Health 看板上线 | ✓ | 2 |
| L5 | D-Diag 3 专题上线 | ✓ | 4 |
| L6 | D-Action 部门分发 | ✓ | 2 |
| L7 | 部署 + 文档 + 培训 | ✓ | 1 |

**合计 13.5 工日 / 0 LLM 成本 / ~30 秒 Superset 重启**

## 七、回滚预案

| 层 | 回滚命令 |
|---|---|
| L1 | `ALTER TABLE voc_review DROP COLUMN ts_inferred, country` |
| L2 | 重导 dim_tag.csv 备份 |
| L3 | DROP 4 新视图（不影响旧视图） |
| L4-L6 | Superset PUT/DELETE 各 dashboard（与 L9 经验一致） |
| 全栈 | 切回 chore/dept-rename-2026-05-13 分支，PG 还原 dim_tag.csv |

## 八、备份索引

`~/.secrets/backups/voc-deep-analysis-mvp/L0-20260514-102823/`

| 文件 | 大小 | 用途 |
|---|---|---|
| voc_review.schema.sql | 2.5K | DDL 回滚参考 |
| voc_review_sample.csv | 3.1M (18K rows, 1.5% 抽样) | 字段值验证 |
| dim_tag_post_mvp.csv | 52K (267 rows) | dim_tag 当前 baseline |
| superset.db | 992K | 看板配置回滚 |

## 九、后续可扩展（不在本 MVP）

- 引入外部 Amazon 竞品 VOC → D-Comp 看板
- D-Opp 场景×痛点机会缺口看板
- LLM 自动生成 P0/P1 行动建议文案
- 真实 timestamp 入库（替代 loaded_at proxy）
- 7 节点细分（A/I/P1/P2/L1/L2/L3）

---

**等审：sat-indicators-draft.md（50 个原子指标定义）→ GO L1**
