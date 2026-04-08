"""
main.py - FastAPI application entry point.

Endpoints
─────────
POST   /experiments                         create a new experiment
GET    /experiments                         list all experiments
GET    /experiments/{id}                    get experiment details

POST   /experiments/{id}/observations       push temporal data (batch)
GET    /experiments/{id}/observations       list time-series data

GET    /experiments/{id}/allocations        compute & return next-day allocation
       ?algorithm=thompson_sampling|ucb1
       &target_date=YYYY-MM-DD (optional)

GET    /experiments/{id}/analytics          raw daily series for charts
GET    /health                              health-check
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from .database import create_tables, get_db
from . import schemas, crud

# ── App bootstrap ─────────────────────────────
app = FastAPI(
    title="Multi-Armed Bandit Optimization API",
    description=(
        "RESTful API for running Multi-Armed Bandit experiments with SQL-backed "
        "storage and Thompson Sampling / UCB1 allocation recommendations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()


# ── Health ────────────────────────────────────

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


# ── Experiments ───────────────────────────────

@app.post("/experiments", response_model=schemas.ExperimentOut, status_code=201, tags=["experiments"])
def create_experiment(
    payload: schemas.ExperimentCreate,
    db: Session = Depends(get_db),
):
    """
    Register a new experiment.

    The first entry in `variants` is treated as the **control** arm.
    You may supply 2+ variants to run a multi-arm bandit (bonus feature).
    """
    try:
        return crud.create_experiment(db, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/experiments", response_model=List[schemas.ExperimentOut], tags=["experiments"])
def list_experiments(db: Session = Depends(get_db)):
    return crud.list_experiments(db)


@app.get("/experiments/{experiment_id}", response_model=schemas.ExperimentOut, tags=["experiments"])
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = crud.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(404, detail="Experiment not found")
    return exp


# ── Observations ──────────────────────────────

@app.post(
    "/experiments/{experiment_id}/observations",
    status_code=201,
    tags=["observations"],
    summary="Push temporal data for one or more (variant, date) pairs",
)
def push_observations(
    experiment_id: int,
    payload: schemas.ObservationBatch,
    db: Session = Depends(get_db),
):
    """
    Accept a batch of daily observations.

    Each entry requires:
    - `variant_name` – must match a registered variant
    - `obs_date`     – ISO-8601 date (YYYY-MM-DD)
    - `impressions`  – total traffic on that day
    - `clicks`       – number of conversions/clicks

    Duplicate (variant, date) pairs are **upserted** (updated in-place).
    """
    if not crud.get_experiment(db, experiment_id):
        raise HTTPException(404, detail="Experiment not found")
    try:
        count = crud.upsert_observations(db, experiment_id, payload)
        return {"inserted_or_updated": count}
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))


@app.get(
    "/experiments/{experiment_id}/observations",
    response_model=schemas.AnalyticsResponse,
    tags=["observations"],
)
def list_observations(experiment_id: int, db: Session = Depends(get_db)):
    if not crud.get_experiment(db, experiment_id):
        raise HTTPException(404, detail="Experiment not found")
    return crud.get_daily_series(db, experiment_id)


# ── Allocations ───────────────────────────────

@app.get(
    "/experiments/{experiment_id}/allocations",
    response_model=schemas.AllocationResponse,
    tags=["allocations"],
    summary="Compute next-day traffic allocation using the chosen MAB algorithm",
)
def get_allocation(
    experiment_id: int,
    algorithm: str = Query(
        "thompson_sampling",
        description="Algorithm to use: `thompson_sampling` (default) or `ucb1`",
    ),
    target_date: Optional[str] = Query(
        None,
        description="Date to target (YYYY-MM-DD). Defaults to tomorrow.",
    ),
    db: Session = Depends(get_db),
):
    """
    Runs the Multi-Armed Bandit algorithm over all historical observations
    stored in SQL and returns the recommended traffic split for `target_date`.

    **Thompson Sampling** (default): Bayesian Beta-Binomial model –
    samples from each arm's posterior and allocates proportionally to win probability.

    **UCB1**: Upper Confidence Bound – balances exploitation of best CTR with
    exploration proportional to 1/sqrt(impressions).
    """
    if not crud.get_experiment(db, experiment_id):
        raise HTTPException(404, detail="Experiment not found")
    if algorithm not in ("thompson_sampling", "ucb1"):
        raise HTTPException(400, detail="algorithm must be 'thompson_sampling' or 'ucb1'")
    try:
        return crud.compute_and_store_allocation(
            db, experiment_id, algorithm=algorithm, target_date=target_date
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))


# ── Analytics ─────────────────────────────────

@app.get(
    "/experiments/{experiment_id}/analytics",
    response_model=schemas.AnalyticsResponse,
    tags=["analytics"],
)
def analytics(experiment_id: int, db: Session = Depends(get_db)):
    """Return the full daily time-series for charting purposes."""
    if not crud.get_experiment(db, experiment_id):
        raise HTTPException(404, detail="Experiment not found")
    return crud.get_daily_series(db, experiment_id)
