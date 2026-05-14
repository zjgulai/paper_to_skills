-- MVP L3 · VOC 深度分析 4 个新视图
-- 依赖：voc_review (含 country + ts_inferred) + dim_tag (含 atomic_indicator_id)
-- 设计：普通 VIEW（与现有 6 视图一致），查询 < 5s 即可，避免 REFRESH 复杂度
-- 公式：S = (正向条数 + 0.5 × 中性条数) / 总有效条数 × 100  (per SGCS spec)


-- ============================================================================
-- 1. v_atomic_indicator_score · 50 SAT 得分按 month × country × product_line
-- ============================================================================
DROP VIEW IF EXISTS v_atomic_indicator_score CASCADE;
CREATE VIEW v_atomic_indicator_score AS
SELECT
  d.atomic_indicator_id,
  r.product_line,
  r.country,
  date_trunc('month', r.ts_inferred)::date AS month,
  COUNT(*)                                  AS total_hits,
  COUNT(*) FILTER (WHERE l.sentiment_preset = 'positive') AS pos,
  COUNT(*) FILTER (WHERE l.sentiment_preset = 'neutral')  AS neu,
  COUNT(*) FILTER (WHERE l.sentiment_preset = 'negative') AS neg,
  ROUND(
    ((COUNT(*) FILTER (WHERE l.sentiment_preset = 'positive')
      + 0.5 * COUNT(*) FILTER (WHERE l.sentiment_preset = 'neutral'))
     / NULLIF(COUNT(*), 0)::numeric * 100)::numeric, 2
  ) AS sat_score,
  ROUND(AVG(l.confidence)::numeric, 3)      AS avg_confidence,
  COUNT(DISTINCT l.review_id)               AS distinct_reviews
FROM voc_label l
JOIN dim_tag d   ON d.tag_id   = l.tag_id
JOIN voc_review r ON r.review_id = l.review_id
WHERE d.atomic_indicator_id IS NOT NULL
GROUP BY 1, 2, 3, 4;
COMMENT ON VIEW v_atomic_indicator_score IS '50 原子指标得分 (SAT_xxx) 按 month × country × product_line · MVP L3';


-- ============================================================================
-- 2. v_aipl_node_score · L1-L4 节点得分（atomic 平均）按 month × country × product_line
-- ============================================================================
DROP VIEW IF EXISTS v_aipl_node_score CASCADE;
CREATE VIEW v_aipl_node_score AS
SELECT
  CASE
    WHEN atomic_indicator_id LIKE 'SAT_L1_%' THEN 'L1'
    WHEN atomic_indicator_id LIKE 'SAT_L2_%' THEN 'L2'
    WHEN atomic_indicator_id LIKE 'SAT_L3_%' THEN 'L3'
    WHEN atomic_indicator_id LIKE 'SAT_L4_%' THEN 'L4'
  END AS aipl_node,
  product_line,
  country,
  month,
  COUNT(DISTINCT atomic_indicator_id)        AS distinct_sat,
  ROUND(AVG(sat_score)::numeric, 2)          AS node_score,
  SUM(total_hits)                            AS total_hits,
  SUM(neg)                                   AS total_neg,
  SUM(pos)                                   AS total_pos,
  ROUND((SUM(neg)::numeric / NULLIF(SUM(total_hits), 0) * 100)::numeric, 2) AS pct_negative
FROM v_atomic_indicator_score
WHERE atomic_indicator_id IS NOT NULL
GROUP BY 1, 2, 3, 4;
COMMENT ON VIEW v_aipl_node_score IS 'L1-L4 节点得分 (= 该节点下所有 SAT 平均) · 按 month × country × product_line · MVP L3';


-- ============================================================================
-- 3. v_country_dept_health · country × dept_owner × month 健康度
-- ============================================================================
DROP VIEW IF EXISTS v_country_dept_health CASCADE;
CREATE VIEW v_country_dept_health AS
SELECT
  r.country,
  d.dept_owner,
  date_trunc('month', r.ts_inferred)::date AS month,
  COUNT(DISTINCT r.review_id)              AS reviews,
  COUNT(*)                                 AS label_hits,
  COUNT(*) FILTER (WHERE l.sentiment_preset = 'negative') AS hits_negative,
  COUNT(*) FILTER (WHERE l.sentiment_preset = 'positive') AS hits_positive,
  COUNT(*) FILTER (WHERE l.sentiment_preset = 'neutral')  AS hits_neutral,
  ROUND(
    (COUNT(*) FILTER (WHERE l.sentiment_preset = 'negative')::numeric
     / NULLIF(COUNT(*), 0) * 100)::numeric, 2
  ) AS pct_negative,
  ROUND(AVG(l.confidence)::numeric, 3)     AS avg_confidence
FROM voc_label l
JOIN dim_tag d   ON d.tag_id   = l.tag_id
JOIN voc_review r ON r.review_id = l.review_id
WHERE d.dept_owner IS NOT NULL AND d.dept_owner <> ''
GROUP BY 1, 2, 3;
COMMENT ON VIEW v_country_dept_health IS 'country × dept_owner × month 健康度 (评论/命中/负向占比) · MVP L3';


-- ============================================================================
-- 4. v_proxy_nps_calibrated · 按数据源 baseline 校准 NPS · per 04-NPS偏差分析方法
-- ============================================================================
DROP VIEW IF EXISTS v_proxy_nps_calibrated CASCADE;
CREATE VIEW v_proxy_nps_calibrated AS
WITH source_baseline AS (
  SELECT
    data_source,
    AVG(CASE
      WHEN proxy_nps = 'promoter'  THEN 1.0
      WHEN proxy_nps = 'detractor' THEN -1.0
      ELSE 0.0
    END) AS baseline
  FROM voc_review
  WHERE proxy_nps IS NOT NULL
  GROUP BY data_source
),
global_baseline AS (
  SELECT AVG(CASE
    WHEN proxy_nps = 'promoter'  THEN 1.0
    WHEN proxy_nps = 'detractor' THEN -1.0
    ELSE 0.0
  END) AS baseline
  FROM voc_review
  WHERE proxy_nps IS NOT NULL
)
SELECT
  r.review_id,
  r.data_source,
  r.product_line,
  r.country,
  date_trunc('month', r.ts_inferred)::date AS month,
  r.proxy_nps                              AS proxy_nps_raw,
  CASE
    WHEN r.proxy_nps = 'promoter'  THEN 1.0
    WHEN r.proxy_nps = 'detractor' THEN -1.0
    ELSE 0.0
  END AS nps_score_raw,
  ROUND(
    (CASE
       WHEN r.proxy_nps = 'promoter'  THEN 1.0
       WHEN r.proxy_nps = 'detractor' THEN -1.0
       ELSE 0.0
     END
     - (sb.baseline - gb.baseline))::numeric,
    4
  ) AS nps_score_calibrated,
  ROUND(sb.baseline::numeric, 4) AS source_baseline,
  ROUND(gb.baseline::numeric, 4) AS global_baseline
FROM voc_review r
LEFT JOIN source_baseline sb ON sb.data_source = r.data_source
CROSS JOIN global_baseline gb
WHERE r.proxy_nps IS NOT NULL;
COMMENT ON VIEW v_proxy_nps_calibrated IS '按 04-NPS偏差分析 方法校准 Proxy NPS · 输出 raw + calibrated · MVP L3';
