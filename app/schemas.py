"""
schemas.py - Pydantic request / response models.
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from datetime import date


# ── Experiment ────────────────────────────────

class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120,
                      example="Homepage CTR Experiment")
    description: Optional[str] = Field(None, max_length=500,
                                       example="A/B test comparing two hero layouts")
    metric: str = Field("ctr", example="ctr")
    variants: List[str] = Field(
        ..., min_length=2,
        example=["control", "variant_a"],
        description="First entry is treated as the control arm."
    )


class ExperimentOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    metric: str

    class Config:
        from_attributes = True


# ── Observations ──────────────────────────────

class DayObservation(BaseModel):
    variant_name: str = Field(..., example="control")
    obs_date: str     = Field(..., example="2024-05-01",
                              description="ISO-8601 date YYYY-MM-DD")
    impressions: int  = Field(..., ge=0, example=1000)
    clicks: int       = Field(..., ge=0, example=42)

    @model_validator(mode="after")
    def clicks_lte_impressions(self):
        if self.clicks > self.impressions:
            raise ValueError("clicks cannot exceed impressions")
        return self


class ObservationBatch(BaseModel):
    """Push one or more (variant, date) rows in a single call."""
    observations: List[DayObservation]


class ObservationOut(BaseModel):
    variant_name: str
    obs_date: str
    impressions: int
    clicks: int


# ── Allocation (recommendation) ───────────────

class VariantAllocation(BaseModel):
    variant_name: str
    is_control: bool
    allocation_pct: float = Field(..., ge=0.0, le=100.0)
    total_impressions: int
    total_clicks: int
    smoothed_ctr: float

    class Config:
        from_attributes = True


class AllocationResponse(BaseModel):
    experiment_id: int
    experiment_name: str
    target_date: str
    algorithm: str
    allocations: List[VariantAllocation]
    note: Optional[str] = None


# ── Analytics ─────────────────────────────────

class DailyPoint(BaseModel):
    obs_date: str
    variant_name: str
    impressions: int
    clicks: int
    daily_ctr: Optional[float]


class AnalyticsResponse(BaseModel):
    experiment_id: int
    series: List[DailyPoint]
