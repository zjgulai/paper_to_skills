\echo '=== BEFORE ==='
SELECT COUNT(*) FROM dim_tag;

BEGIN;

ALTER TABLE dim_tag ADD COLUMN IF NOT EXISTS atomic_indicator_id TEXT;

CREATE TEMP TABLE _tag_sat_map (tag_id TEXT, atomic_indicator_id TEXT);
\copy _tag_sat_map FROM '/tmp/voc_l2_tag_sat_map.csv' WITH CSV HEADER;
\echo 'map rows loaded:'
SELECT COUNT(*) FROM _tag_sat_map;

UPDATE dim_tag d
SET atomic_indicator_id = m.atomic_indicator_id
FROM _tag_sat_map m
WHERE d.tag_id = m.tag_id;

CREATE INDEX IF NOT EXISTS idx_dim_tag_sat ON dim_tag(atomic_indicator_id);

\echo ''
\echo '=== AFTER (in tx) ==='
SELECT COUNT(*) AS total_rows,
       COUNT(atomic_indicator_id) AS sat_filled,
       COUNT(*) - COUNT(atomic_indicator_id) AS sat_null
FROM dim_tag;

COMMIT;

\echo ''
\echo '=== AFTER COMMIT (sanity) ==='
SELECT atomic_indicator_id, COUNT(*) AS tags
FROM dim_tag GROUP BY 1 ORDER BY 1 NULLS FIRST;

\echo ''
\echo '=== indexes ==='
SELECT indexname FROM pg_indexes WHERE tablename='dim_tag' ORDER BY 1;
