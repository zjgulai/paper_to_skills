\echo '=== BEFORE ==='
SELECT COUNT(*) FROM voc_review;

BEGIN;

ALTER TABLE voc_review ADD COLUMN IF NOT EXISTS country TEXT;
ALTER TABLE voc_review ADD COLUMN IF NOT EXISTS ts_inferred TIMESTAMP WITH TIME ZONE;
ALTER TABLE voc_review ADD COLUMN IF NOT EXISTS time_resolution TEXT;

UPDATE voc_review SET country = CASE
  WHEN language = 'en' AND data_source = 'amazon_competitor' THEN 'US'
  WHEN language = 'en' AND data_source = 'trustpilot'        THEN 'UK'
  WHEN language = 'en' AND data_source = 'zendesk'           THEN 'OTHER'
  WHEN language = 'en' AND data_source IN ('momcozy','reddit') THEN 'US'
  WHEN language = 'de' THEN 'DE'
  WHEN language = 'fr' THEN 'FR'
  WHEN language = 'es' THEN 'ES'
  WHEN language = 'it' THEN 'IT'
  WHEN language IS NULL THEN 'OTHER'
  ELSE 'OTHER'
END;

UPDATE voc_review SET
  ts_inferred = loaded_at,
  time_resolution = CASE
    WHEN data_source IN ('zendesk','momcozy') THEN 'load_batch_proxy_for_real_ts'
    ELSE 'load_batch'
  END;

CREATE INDEX IF NOT EXISTS idx_voc_review_country     ON voc_review(country);
CREATE INDEX IF NOT EXISTS idx_voc_review_ts_inferred ON voc_review(ts_inferred);

\echo '=== AFTER (in tx) ==='
SELECT country, COUNT(*) FROM voc_review GROUP BY 1 ORDER BY 2 DESC;
SELECT time_resolution, COUNT(*) FROM voc_review GROUP BY 1;
SELECT COUNT(*) AS total, COUNT(country) AS country_filled, COUNT(ts_inferred) AS ts_filled FROM voc_review;

COMMIT;

\echo '=== AFTER COMMIT ==='
SELECT country, COUNT(*) FROM voc_review GROUP BY 1 ORDER BY 2 DESC;
\echo ''
\echo '=== Indexes ==='
\di voc_review*
