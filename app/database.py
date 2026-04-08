"""
database.py - SQLAlchemy setup and SQL schema for the Multi-Armed Bandit API.

Schema Design:
  - experiments: top-level experiment registry (supports multiple experiments)
  - variants:    each arm of the bandit (control, variant_a, variant_b, ...)
  - observations: one row per day per variant (impressions + clicks)
  - allocations:  stores computed allocation recommendations per day

This design allows multiple concurrent experiments and more than 2 variants (bonus).
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, ForeignKey, UniqueConstraint, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime

DATABASE_URL = "sqlite:///./bandit.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────
# ORM Models (mirrors the SQL DDL below)
# ──────────────────────────────────────────────

class Experiment(Base):
    __tablename__ = "experiments"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(120), unique=True, nullable=False)
    description = Column(String(500))
    metric      = Column(String(60), default="ctr")   # e.g. "ctr", "conversion_rate"
    created_at  = Column(DateTime, default=datetime.utcnow)


class Variant(Base):
    __tablename__ = "variants"

    id            = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    name          = Column(String(60), nullable=False)   # "control", "variant_a", ...
    is_control    = Column(Integer, default=0)            # 1 = control arm
    created_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("experiment_id", "name", name="uq_variant_per_experiment"),
    )


class Observation(Base):
    """One row per (variant, date).  Impressions and clicks for that day."""
    __tablename__ = "observations"

    id            = Column(Integer, primary_key=True, index=True)
    variant_id    = Column(Integer, ForeignKey("variants.id", ondelete="CASCADE"), nullable=False)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    obs_date      = Column(String(10), nullable=False)   # ISO-8601 "YYYY-MM-DD"
    impressions   = Column(Integer, default=0)
    clicks        = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("variant_id", "obs_date", name="uq_obs_per_variant_day"),
    )


class Allocation(Base):
    """Stores the computed recommendation for a given target date."""
    __tablename__ = "allocations"

    id            = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    variant_id    = Column(Integer, ForeignKey("variants.id", ondelete="CASCADE"), nullable=False)
    target_date   = Column(String(10), nullable=False)   # date the allocation is FOR
    allocation_pct= Column(Float, nullable=False)
    algorithm     = Column(String(60), default="thompson_sampling")
    computed_at   = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("experiment_id", "variant_id", "target_date",
                         name="uq_alloc_per_variant_day"),
    )


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency – yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ──────────────────────────────────────────────
# SQL helper queries (raw SQL via SQLAlchemy text())
# ──────────────────────────────────────────────

SQL_AGGREGATE_VARIANT_STATS = text("""
    SELECT
        v.id            AS variant_id,
        v.name          AS variant_name,
        v.is_control,
        SUM(o.impressions)  AS total_impressions,
        SUM(o.clicks)       AS total_clicks,
        -- Bayesian CTR estimate (add-1 Laplace smoothing to avoid div/0)
        CAST(SUM(o.clicks) + 1 AS REAL) /
            CAST(SUM(o.impressions) + 2 AS REAL) AS smoothed_ctr,
        -- Wilson score lower bound for UCB reference
        CAST(SUM(o.clicks) AS REAL) /
            NULLIF(CAST(SUM(o.impressions) AS REAL), 0) AS raw_ctr
    FROM variants v
    LEFT JOIN observations o
        ON o.variant_id = v.id
       AND o.experiment_id = :experiment_id
    WHERE v.experiment_id = :experiment_id
    GROUP BY v.id, v.name, v.is_control
    ORDER BY v.is_control DESC, v.name
""")

SQL_LAST_N_DAYS_STATS = text("""
    SELECT
        v.id            AS variant_id,
        v.name          AS variant_name,
        SUM(o.impressions)  AS total_impressions,
        SUM(o.clicks)       AS total_clicks
    FROM variants v
    LEFT JOIN observations o
        ON o.variant_id = v.id
       AND o.experiment_id = :experiment_id
       AND o.obs_date >= :since_date
    WHERE v.experiment_id = :experiment_id
    GROUP BY v.id, v.name
""")

SQL_DAILY_SERIES = text("""
    SELECT
        o.obs_date,
        v.name          AS variant_name,
        o.impressions,
        o.clicks,
        CAST(o.clicks AS REAL) / NULLIF(CAST(o.impressions AS REAL), 0) AS daily_ctr
    FROM observations o
    JOIN variants v ON v.id = o.variant_id
    WHERE o.experiment_id = :experiment_id
    ORDER BY o.obs_date, v.name
""")
