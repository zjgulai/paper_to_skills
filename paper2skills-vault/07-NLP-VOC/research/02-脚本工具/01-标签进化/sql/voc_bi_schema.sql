-- Phase 7 D1 — voc_bi schema (star)
-- ----------------------------------------------------------------------------
-- 4 tables based on phase6_d9_filtered.jsonl + tag_dictionary_v4.1.xlsx
--
-- voc_review        - 1 row per review_id (master fact)
-- voc_label         - 1 row per (review_id, label) (1:N from review)
-- voc_brand_mention - 1 row per (review_id, brand) (1:N)
-- dim_tag           - 1 row per tag_id (from v4.1 dict, dim)
--
-- All FK enforce ON DELETE CASCADE so re-importing a review wipes its labels.
-- ----------------------------------------------------------------------------

DROP TABLE IF EXISTS voc_label CASCADE;
DROP TABLE IF EXISTS voc_brand_mention CASCADE;
DROP TABLE IF EXISTS voc_review CASCADE;
DROP TABLE IF EXISTS dim_tag CASCADE;

-- ============================================================================
-- dim_tag — from v4.1 dict 01_通用标签主表
-- ============================================================================
CREATE TABLE dim_tag (
  tag_id        TEXT PRIMARY KEY,
  tag_cn        TEXT,
  tag_en        TEXT,
  aipl_node     TEXT,
  polarity      TEXT,                  -- 正向 / 负向 / 中性
  dept_owner    TEXT,                  -- 主责部门
  biz_action    TEXT,                  -- 业务动作
  strategy_pkg  TEXT,
  is_general    BOOLEAN,
  audit_status  TEXT,
  loaded_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_tag_dept ON dim_tag(dept_owner);
CREATE INDEX idx_dim_tag_polarity ON dim_tag(polarity);

-- ============================================================================
-- voc_review — master fact (1 row per review)
-- ============================================================================
CREATE TABLE voc_review (
  review_id            TEXT PRIMARY KEY,
  text                 TEXT,
  data_source          TEXT,           -- amazon_competitor / trustpilot / zendesk / momcozy / reddit
  platform             TEXT,           -- amazon / trustpilot / zendesk / reddit
  source_type          TEXT,           -- review / customer_service / etc
  asin                 TEXT,
  spu_code             TEXT,
  product_line         TEXT,           -- 内衣服饰 / 喂养电器 / etc
  category             TEXT,
  rating               REAL,
  language             TEXT,
  -- analytics fields
  sentiment_polarity   REAL,           -- numeric in [-1, 1]
  sentiment_calibration TEXT,
  proxy_nps            TEXT,           -- promoter / passive / detractor
  aipl_stage           TEXT,
  persona_derived      TEXT,
  brand_count          INT,            -- count of brand_mentions
  brand_comparison     BOOLEAN,
  -- quality
  quality_score        REAL,           -- _quality_score
  n_tags               INT,
  -- provenance
  label_source         TEXT,
  label_sources        TEXT[],         -- array of sources (D5 / D8 / D9 etc.)
  has_phase6_d4_meta   BOOLEAN,        -- whether _phase6_d4_meta present
  loaded_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_voc_review_data_source ON voc_review(data_source);
CREATE INDEX idx_voc_review_product_line ON voc_review(product_line);
CREATE INDEX idx_voc_review_proxy_nps ON voc_review(proxy_nps);
CREATE INDEX idx_voc_review_rating ON voc_review(rating);
CREATE INDEX idx_voc_review_language ON voc_review(language);

-- ============================================================================
-- voc_label — 1 row per (review, label)
-- ============================================================================
CREATE TABLE voc_label (
  id                 BIGSERIAL PRIMARY KEY,
  review_id          TEXT NOT NULL REFERENCES voc_review(review_id) ON DELETE CASCADE,
  tag_id             TEXT NOT NULL,
  tag_cn             TEXT,
  tag_en             TEXT,
  aipl_node          TEXT,
  sentiment_preset   TEXT,             -- positive / negative / neutral
  sentiment_calibrated REAL,           -- numeric (post-D10 helper normalize)
  confidence         REAL,             -- final confidence after rebalance
  confidence_original REAL,            -- pre-D3 lift
  confidence_lift    REAL,             -- D3 lift amount
  source             TEXT,             -- _source field if present
  loaded_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_voc_label_review ON voc_label(review_id);
CREATE INDEX idx_voc_label_tag ON voc_label(tag_id);
CREATE INDEX idx_voc_label_confidence ON voc_label(confidence);

-- ============================================================================
-- voc_brand_mention — 1 row per (review, brand)
-- ============================================================================
CREATE TABLE voc_brand_mention (
  id          BIGSERIAL PRIMARY KEY,
  review_id   TEXT NOT NULL REFERENCES voc_review(review_id) ON DELETE CASCADE,
  brand_name  TEXT NOT NULL,
  loaded_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_voc_brand_review ON voc_brand_mention(review_id);
CREATE INDEX idx_voc_brand_name ON voc_brand_mention(brand_name);
