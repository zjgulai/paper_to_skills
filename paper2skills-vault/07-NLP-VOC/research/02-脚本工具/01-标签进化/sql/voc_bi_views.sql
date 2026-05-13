-- Phase 7 D1.5 — voc_bi 视图层 (Superset / Metabase 直接消费)
-- ----------------------------------------------------------------------------
-- 视图定位：把 ETL 后的 4 张表（dim_tag + voc_review + voc_label + voc_brand_mention）
-- 转换成 BI 工具直接可拖拽的"业务化"视图。任何 BI 工具读这些视图，无需写 JOIN。
--
-- 视图清单：
--   v_review_overview         总览 KPI（按 source / nps / sentiment / product_line）
--   v_label_with_dept         label × dim_tag JOIN 后的扁平表（含部门归属）
--   v_dept_topic_summary      MAA Top-N 替代 SQL 版（per-dept 命中数 + 平均置信度）
--   v_label_brand              label + brand 交叉视图
--   v_global_top_tags         全局 Top-N 标签命中数
--   v_dept_kpi                7 部门 KPI 概览（命中数 / 平均极性 / NPS 分布）
--
-- 设计原则：
--   - 视图只做 JOIN + 聚合，不写业务规则 (SRAC 排序由 Superset 端做)
--   - 用 MATERIALIZED VIEW 仅在确实慢时；当前 voc_label 689K 行，普通 VIEW 即可
--   - 每个视图含 source 注释指明原始 4 表
-- ----------------------------------------------------------------------------

-- ============================================================================
-- 1. v_review_overview — 总览 KPI（基础事实表的轻聚合）
-- ============================================================================
DROP VIEW IF EXISTS v_review_overview CASCADE;
CREATE VIEW v_review_overview AS
SELECT
  data_source,
  platform,
  product_line,
  language,
  proxy_nps,
  rating,
  sentiment_polarity,
  aipl_stage,
  persona_derived,
  brand_count,
  brand_comparison,
  quality_score,
  n_tags
FROM voc_review;

COMMENT ON VIEW v_review_overview IS '总览：按 data_source / product_line / proxy_nps 切分；BI 直接拖入 chart filter';


-- ============================================================================
-- 2. v_label_with_dept — voc_label JOIN dim_tag 加部门归属（最常用视图）
-- ============================================================================
DROP VIEW IF EXISTS v_label_with_dept CASCADE;
CREATE VIEW v_label_with_dept AS
SELECT
  l.id                          AS label_id,
  l.review_id,
  l.tag_id,
  COALESCE(l.tag_cn, d.tag_cn) AS tag_cn,
  COALESCE(l.tag_en, d.tag_en) AS tag_en,
  d.aipl_node                   AS aipl_node,
  d.polarity                    AS polarity,        -- 字典侧权威极性
  d.dept_owner                  AS dept_owner,      -- 主责部门（核心 BI 切分维度）
  d.biz_action                  AS biz_action,
  d.strategy_pkg                AS strategy_pkg,
  l.sentiment_preset,
  l.sentiment_calibrated,
  l.confidence,
  l.confidence_original,
  l.confidence_lift,
  l.source                      AS label_source,
  -- review-level 关键字段（避免每次都 JOIN voc_review）
  r.data_source,
  r.platform,
  r.product_line,
  r.language,
  r.proxy_nps,
  r.rating,
  r.sentiment_polarity          AS review_sentiment_polarity,
  r.aipl_stage                  AS review_aipl_stage,
  r.brand_count,
  r.brand_comparison,
  r.quality_score
FROM voc_label l
LEFT JOIN dim_tag d ON l.tag_id = d.tag_id
LEFT JOIN voc_review r ON l.review_id = r.review_id;

COMMENT ON VIEW v_label_with_dept IS 'label × tag × review 三表 JOIN 的扁平视图，含部门归属。BI 主消费';


-- ============================================================================
-- 3. v_dept_topic_summary — Per-dept Per-tag 聚合（MAA 替代）
-- ============================================================================
DROP VIEW IF EXISTS v_dept_topic_summary CASCADE;
CREATE VIEW v_dept_topic_summary AS
SELECT
  dept_owner,
  tag_id,
  MAX(tag_cn) FILTER (WHERE tag_cn IS NOT NULL)             AS tag_cn,
  MAX(tag_en) FILTER (WHERE tag_en IS NOT NULL)             AS tag_en,
  MAX(polarity)                                              AS polarity,
  MAX(biz_action)                                            AS biz_action,
  COUNT(*)                                                   AS hit_count,
  ROUND(AVG(confidence)::numeric, 4)                         AS avg_confidence,
  ROUND(AVG(ABS(sentiment_calibrated))::numeric, 4)          AS avg_abs_sentiment,
  COUNT(*) FILTER (WHERE polarity = '负向')::INT             AS hit_negative,
  COUNT(*) FILTER (WHERE polarity = '正向')::INT             AS hit_positive,
  COUNT(*) FILTER (WHERE polarity = '中性')::INT             AS hit_neutral,
  COUNT(DISTINCT review_id)                                  AS distinct_reviews
FROM v_label_with_dept
WHERE dept_owner IS NOT NULL AND dept_owner <> ''
GROUP BY dept_owner, tag_id
ORDER BY dept_owner, hit_count DESC;

COMMENT ON VIEW v_dept_topic_summary IS '每部门每标签的命中量 + 平均置信度 + 极性分布。BI 拖出来即 MAA Top-N';


-- ============================================================================
-- 4. v_label_brand — label JOIN brand_mention（交叉分析）
-- ============================================================================
DROP VIEW IF EXISTS v_label_brand CASCADE;
CREATE VIEW v_label_brand AS
SELECT
  bm.brand_name,
  bm.review_id,
  l.tag_id,
  l.tag_cn,
  l.dept_owner,
  l.polarity,
  l.confidence,
  l.review_sentiment_polarity
FROM voc_brand_mention bm
LEFT JOIN v_label_with_dept l ON bm.review_id = l.review_id;

COMMENT ON VIEW v_label_brand IS 'brand × label 交叉。品牌市场中心「品牌提及 × 标签」分析';


-- ============================================================================
-- 5. v_global_top_tags — 全局 Top-N 标签
-- ============================================================================
DROP VIEW IF EXISTS v_global_top_tags CASCADE;
CREATE VIEW v_global_top_tags AS
SELECT
  l.tag_id,
  MAX(l.tag_cn) FILTER (WHERE l.tag_cn IS NOT NULL)  AS tag_cn,
  MAX(l.tag_en) FILTER (WHERE l.tag_en IS NOT NULL)  AS tag_en,
  d.dept_owner,
  d.polarity,
  COUNT(*)                                            AS hit_count,
  ROUND(AVG(l.confidence)::numeric, 4)                AS avg_confidence,
  COUNT(DISTINCT l.review_id)                         AS distinct_reviews
FROM voc_label l
LEFT JOIN dim_tag d ON l.tag_id = d.tag_id
GROUP BY l.tag_id, d.dept_owner, d.polarity
ORDER BY hit_count DESC;

COMMENT ON VIEW v_global_top_tags IS '全局标签命中排行（不限部门）。Overview 大屏首页';


-- ============================================================================
-- 6. v_dept_kpi — 7 部门 KPI 概览（一行一个部门）
-- ============================================================================
DROP VIEW IF EXISTS v_dept_kpi CASCADE;
CREATE VIEW v_dept_kpi AS
WITH base AS (
  SELECT * FROM v_label_with_dept
  WHERE dept_owner IS NOT NULL AND dept_owner <> ''
)
SELECT
  dept_owner,
  COUNT(DISTINCT tag_id)                              AS distinct_tags,
  COUNT(DISTINCT review_id)                           AS distinct_reviews,
  COUNT(*)                                            AS total_label_hits,
  ROUND(AVG(confidence)::numeric, 4)                  AS avg_confidence,
  COUNT(*) FILTER (WHERE polarity = '负向')           AS hits_negative,
  COUNT(*) FILTER (WHERE polarity = '正向')           AS hits_positive,
  COUNT(*) FILTER (WHERE polarity = '中性')           AS hits_neutral,
  ROUND(
    COUNT(*) FILTER (WHERE polarity = '负向')::numeric /
    NULLIF(COUNT(*), 0)::numeric,
  4)                                                  AS pct_negative,
  COUNT(DISTINCT data_source)                         AS data_source_count
FROM base
GROUP BY dept_owner
ORDER BY total_label_hits DESC;

COMMENT ON VIEW v_dept_kpi IS '7 部门 KPI 概览：标签数 / 评论数 / 命中数 / 极性分布。Overview 卡片';
