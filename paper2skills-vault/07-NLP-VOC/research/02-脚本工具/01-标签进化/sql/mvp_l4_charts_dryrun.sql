\echo '=== Chart 1: 总览 KPI 卡片 ==='
\timing on
SELECT
  (SELECT COUNT(*) FROM voc_review)                                                    AS total_reviews,
  ROUND((100.0 * COUNT(*) FILTER (WHERE sentiment_preset='negative') / NULLIF(COUNT(*),0))::numeric, 2) AS neg_label_rate_pct,
  (SELECT ROUND(AVG(sat_score)::numeric, 2) FROM v_atomic_indicator_score)             AS avg_sat_score,
  (SELECT ROUND(AVG(nps_score_calibrated)::numeric, 4) FROM v_proxy_nps_calibrated)    AS avg_nps_cal,
  (SELECT ROUND(AVG(nps_score_raw)::numeric, 4) FROM v_proxy_nps_calibrated)           AS avg_nps_raw
FROM voc_label;
\timing off

\echo '=== Chart 2: AIPL 节点健康度 by country ==='
\timing on
SELECT
  country,
  aipl_node,
  ROUND(AVG(node_score)::numeric, 2)              AS avg_score,
  ROUND(AVG(pct_negative)::numeric, 2)            AS pct_neg,
  SUM(total_hits)                                 AS hits
FROM v_aipl_node_score
WHERE country IS NOT NULL
GROUP BY country, aipl_node
ORDER BY country, aipl_node;
\timing off

\echo '=== Chart 3: SAT Top10 / Bottom10 (体量加权) ==='
\timing on
WITH agg AS (
  SELECT
    atomic_indicator_id,
    SUM(total_hits)                                                                            AS hits,
    SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0)                          AS score_w,
    SUM(neg)                                                                                   AS neg_hits,
    ROUND((100.0 * SUM(neg) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)                AS pct_neg
  FROM v_atomic_indicator_score
  GROUP BY atomic_indicator_id
)
(SELECT atomic_indicator_id, hits, ROUND(score_w::numeric, 2) AS score_w, pct_neg, '顶部高满意' AS bucket
 FROM agg WHERE hits > 0 ORDER BY score_w DESC LIMIT 10)
UNION ALL
(SELECT atomic_indicator_id, hits, ROUND(score_w::numeric, 2) AS score_w, pct_neg, '底部低满意' AS bucket
 FROM agg WHERE hits > 0 ORDER BY score_w ASC LIMIT 10)
ORDER BY bucket DESC, score_w DESC;
\timing off

\echo '=== Chart 4: Country × Dept 健康度热力图 ==='
\timing on
SELECT
  country,
  dept_owner,
  ROUND(AVG(pct_negative)::numeric, 2)            AS pct_neg,
  SUM(label_hits)                                 AS hits,
  SUM(reviews)                                    AS reviews
FROM v_country_dept_health
WHERE label_hits > 0 AND country IS NOT NULL
GROUP BY country, dept_owner
ORDER BY country, pct_neg DESC;
\timing off

\echo '=== Chart 5: 校准后 NPS by source ==='
\timing on
SELECT
  data_source,
  COUNT(*)                                                                           AS reviews,
  ROUND(AVG(nps_score_raw)::numeric, 4)                                              AS nps_raw,
  ROUND(AVG(nps_score_calibrated)::numeric, 4)                                       AS nps_cal,
  ROUND((AVG(nps_score_calibrated) - AVG(nps_score_raw))::numeric, 4)                AS calibration_delta
FROM v_proxy_nps_calibrated
GROUP BY data_source
ORDER BY reviews DESC;
\timing off
