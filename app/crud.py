"""
crud.py - Data-access layer.
All heavy lifting uses the named SQL queries defined in database.py;
simple inserts/lookups use the ORM for clarity.
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import date, timedelta
from typing import Optional

from . import database as db
from . import schemas, bandit


# ──────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────

def create_experiment(session: Session, payload: schemas.ExperimentCreate) -> db.Experiment:
    experiment = db.Experiment(
        name=payload.name,
        description=payload.description,
        metric=payload.metric,
    )
    session.add(experiment)
    session.flush()   # get experiment.id before committing

    for idx, vname in enumerate(payload.variants):
        variant = db.Variant(
            experiment_id=experiment.id,
            name=vname,
            is_control=1 if idx == 0 else 0,
        )
        session.add(variant)

    session.commit()
    session.refresh(experiment)
    return experiment


def get_experiment(session: Session, experiment_id: int) -> Optional[db.Experiment]:
    return session.get(db.Experiment, experiment_id)


def list_experiments(session: Session):
    return session.query(db.Experiment).order_by(db.Experiment.id).all()


# ──────────────────────────────────────────────
# Variants
# ──────────────────────────────────────────────

def get_variant_by_name(session: Session, experiment_id: int, name: str) -> Optional[db.Variant]:
    return (
        session.query(db.Variant)
        .filter(db.Variant.experiment_id == experiment_id, db.Variant.name == name)
        .first()
    )


def list_variants(session: Session, experiment_id: int):
    return (
        session.query(db.Variant)
        .filter(db.Variant.experiment_id == experiment_id)
        .order_by(db.Variant.is_control.desc(), db.Variant.name)
        .all()
    )


# ──────────────────────────────────────────────
# Observations  (INSERT / UPSERT via raw SQL)
# ──────────────────────────────────────────────

UPSERT_OBSERVATION = text("""
    INSERT INTO observations (variant_id, experiment_id, obs_date, impressions, clicks, created_at)
    VALUES (:variant_id, :experiment_id, :obs_date, :impressions, :clicks, datetime('now'))
    ON CONFLICT (variant_id, obs_date)
    DO UPDATE SET
        impressions = excluded.impressions,
        clicks      = excluded.clicks
""")


def upsert_observations(
    session: Session,
    experiment_id: int,
    payload: schemas.ObservationBatch,
) -> int:
    """
    Insert-or-replace observations.
    Returns the number of rows processed.
    """
    count = 0
    for obs in payload.observations:
        variant = get_variant_by_name(session, experiment_id, obs.variant_name)
        if variant is None:
            raise ValueError(
                f"Variant '{obs.variant_name}' not found in experiment {experiment_id}."
            )
        session.execute(
            UPSERT_OBSERVATION,
            {
                "variant_id":    variant.id,
                "experiment_id": experiment_id,
                "obs_date":      obs.obs_date,
                "impressions":   obs.impressions,
                "clicks":        obs.clicks,
            },
        )
        count += 1
    session.commit()
    return count


# ──────────────────────────────────────────────
# Allocation (Multi-Armed Bandit recommendation)
# ──────────────────────────────────────────────

UPSERT_ALLOCATION = text("""
    INSERT INTO allocations
        (experiment_id, variant_id, target_date, allocation_pct, algorithm, computed_at)
    VALUES
        (:experiment_id, :variant_id, :target_date, :allocation_pct, :algorithm, datetime('now'))
    ON CONFLICT (experiment_id, variant_id, target_date)
    DO UPDATE SET
        allocation_pct = excluded.allocation_pct,
        algorithm      = excluded.algorithm,
        computed_at    = excluded.computed_at
""")


def compute_and_store_allocation(
    session: Session,
    experiment_id: int,
    algorithm: str = "thompson_sampling",
    target_date: Optional[str] = None,
) -> schemas.AllocationResponse:
    """
    1. Pull aggregated stats via SQL.
    2. Run the chosen MAB algorithm.
    3. Persist allocations via SQL.
    4. Return a structured response.
    """
    experiment = get_experiment(session, experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id} not found.")

    if target_date is None:
        target_date = (date.today() + timedelta(days=1)).isoformat()

    # ── Step 1: aggregate stats with SQL ──────
    rows = session.execute(
        db.SQL_AGGREGATE_VARIANT_STATS,
        {"experiment_id": experiment_id},
    ).mappings().all()

    if not rows:
        raise ValueError("No variants found for this experiment.")

    # ── Step 2: run MAB algorithm ─────────────
    arm_data = [
        {
            "variant_id":        r["variant_id"],
            "variant_name":      r["variant_name"],
            "is_control":        bool(r["is_control"]),
            "total_impressions": r["total_impressions"] or 0,
            "total_clicks":      r["total_clicks"] or 0,
            "smoothed_ctr":      r["smoothed_ctr"] or 0.0,
        }
        for r in rows
    ]

    percentages = bandit.compute_allocations(arm_data, algorithm=algorithm)

    # ── Step 3: persist allocations via SQL ───
    for arm, pct in zip(arm_data, percentages):
        session.execute(
            UPSERT_ALLOCATION,
            {
                "experiment_id":  experiment_id,
                "variant_id":     arm["variant_id"],
                "target_date":    target_date,
                "allocation_pct": round(pct, 4),
                "algorithm":      algorithm,
            },
        )
    session.commit()

    # ── Step 4: build response ─────────────────
    allocations = [
        schemas.VariantAllocation(
            variant_name=arm["variant_name"],
            is_control=arm["is_control"],
            allocation_pct=round(pct, 2),
            total_impressions=arm["total_impressions"],
            total_clicks=arm["total_clicks"],
            smoothed_ctr=round(arm["smoothed_ctr"], 6),
        )
        for arm, pct in zip(arm_data, percentages)
    ]

    note = None
    if all(a.total_impressions == 0 for a in allocations):
        note = "No observations yet – equal allocation applied."

    return schemas.AllocationResponse(
        experiment_id=experiment_id,
        experiment_name=experiment.name,
        target_date=target_date,
        algorithm=algorithm,
        allocations=allocations,
        note=note,
    )


# ──────────────────────────────────────────────
# Analytics – daily time-series
# ──────────────────────────────────────────────

def get_daily_series(session: Session, experiment_id: int) -> schemas.AnalyticsResponse:
    rows = session.execute(
        db.SQL_DAILY_SERIES,
        {"experiment_id": experiment_id},
    ).mappings().all()

    series = [
        schemas.DailyPoint(
            obs_date=r["obs_date"],
            variant_name=r["variant_name"],
            impressions=r["impressions"],
            clicks=r["clicks"],
            daily_ctr=r["daily_ctr"],
        )
        for r in rows
    ]

    return schemas.AnalyticsResponse(experiment_id=experiment_id, series=series)
