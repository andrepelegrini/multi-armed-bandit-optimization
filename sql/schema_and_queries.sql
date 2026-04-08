-- =================================================================
-- Multi-Armed Bandit Optimization API – SQL Schema Reference
-- (SQLite dialect; adapt for PostgreSQL/MySQL as needed)
-- =================================================================

-- 1. EXPERIMENTS – top-level registry
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT,
    metric      TEXT    NOT NULL DEFAULT 'ctr',
    created_at  DATETIME DEFAULT (datetime('now'))
);

-- 2. VARIANTS – arms of the bandit (supports 2+ variants)
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS variants (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name          TEXT    NOT NULL,
    is_control    INTEGER NOT NULL DEFAULT 0,   -- 1 = control arm
    created_at    DATETIME DEFAULT (datetime('now')),
    UNIQUE (experiment_id, name)
);
CREATE INDEX IF NOT EXISTS idx_variants_experiment ON variants(experiment_id);

-- 3. OBSERVATIONS – one row per (variant, date)
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id    INTEGER NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    obs_date      TEXT    NOT NULL,   -- ISO-8601 'YYYY-MM-DD'
    impressions   INTEGER NOT NULL DEFAULT 0,
    clicks        INTEGER NOT NULL DEFAULT 0,
    created_at    DATETIME DEFAULT (datetime('now')),
    UNIQUE (variant_id, obs_date)
);
CREATE INDEX IF NOT EXISTS idx_obs_experiment_date ON observations(experiment_id, obs_date);

-- 4. ALLOCATIONS – computed traffic split recommendations
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS allocations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    variant_id     INTEGER NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
    target_date    TEXT    NOT NULL,
    allocation_pct REAL    NOT NULL,
    algorithm      TEXT    NOT NULL DEFAULT 'thompson_sampling',
    computed_at    DATETIME DEFAULT (datetime('now')),
    UNIQUE (experiment_id, variant_id, target_date)
);


-- =================================================================
-- Useful Analytical Queries
-- =================================================================

-- A) Aggregate CTR per variant (all-time)
SELECT
    v.name                  AS variant,
    SUM(o.impressions)      AS total_impressions,
    SUM(o.clicks)           AS total_clicks,
    CAST(SUM(o.clicks) AS REAL) / NULLIF(SUM(o.impressions), 0) AS raw_ctr,
    -- Bayesian smoothed estimate (Beta prior α=1, β=1)
    CAST(SUM(o.clicks) + 1 AS REAL) / CAST(SUM(o.impressions) + 2 AS REAL) AS smoothed_ctr
FROM variants v
LEFT JOIN observations o ON o.variant_id = v.id
WHERE v.experiment_id = 1
GROUP BY v.id, v.name
ORDER BY smoothed_ctr DESC;


-- B) Daily CTR time-series for a specific experiment
SELECT
    o.obs_date,
    v.name                  AS variant,
    o.impressions,
    o.clicks,
    CAST(o.clicks AS REAL) / NULLIF(CAST(o.impressions AS REAL), 0) AS daily_ctr
FROM observations o
JOIN variants v ON v.id = o.variant_id
WHERE o.experiment_id = 1
ORDER BY o.obs_date, v.name;


-- C) 7-day rolling CTR (SQLite window function)
SELECT
    obs_date,
    variant_id,
    SUM(clicks)      OVER w AS rolling_clicks,
    SUM(impressions) OVER w AS rolling_impressions,
    CAST(SUM(clicks) OVER w AS REAL) /
        NULLIF(CAST(SUM(impressions) OVER w AS REAL), 0) AS rolling_ctr
FROM observations
WHERE experiment_id = 1
WINDOW w AS (
    PARTITION BY variant_id
    ORDER BY obs_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
)
ORDER BY obs_date, variant_id;


-- D) Latest allocation recommendation per experiment
SELECT
    e.name  AS experiment,
    v.name  AS variant,
    a.allocation_pct,
    a.algorithm,
    a.target_date,
    a.computed_at
FROM allocations a
JOIN experiments e ON e.id = a.experiment_id
JOIN variants    v ON v.id = a.variant_id
WHERE a.target_date = (
    SELECT MAX(target_date) FROM allocations WHERE experiment_id = a.experiment_id
)
ORDER BY a.experiment_id, a.allocation_pct DESC;


-- E) UCB1 scores (manual calculation for inspection)
WITH stats AS (
    SELECT
        v.id, v.name,
        SUM(o.impressions) AS n,
        SUM(o.clicks)      AS k,
        (SELECT SUM(impressions) FROM observations WHERE experiment_id = 1) AS N
    FROM variants v
    LEFT JOIN observations o ON o.variant_id = v.id AND o.experiment_id = 1
    WHERE v.experiment_id = 1
    GROUP BY v.id, v.name
)
SELECT
    name,
    n AS impressions,
    k AS clicks,
    CAST(k AS REAL) / NULLIF(n, 0) AS ctr,
    CAST(k AS REAL) / NULLIF(n, 0)
        + SQRT(2.0 * LOG(NULLIF(N, 0)) / NULLIF(n, 0)) AS ucb1_score
FROM stats;
