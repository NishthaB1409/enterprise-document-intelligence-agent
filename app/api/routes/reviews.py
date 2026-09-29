"""The human side of the review gate: see what was held, and decide it.

    GET  /reviews?status=pending     the queue, oldest first
    GET  /reviews/{id}               one held answer, with everything it was held with
    POST /reviews/{id}/decision      approve or reject it

The requester polls the same `GET /reviews/{id}` for the outcome, so there is
one representation of a held answer rather than a reviewer's view and a
requester's view that can drift apart.

These routes are as unauthenticated as the rest of the service. `reviewer` is a
name the caller supplies, recorded for the audit trail but not verified — which
is fine for a demo and not fine for a queue that guards real decisions.
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import ServicesDep
from app.api.routes.query import QueryResponse
from app.review.store import ReviewAlreadyDecided, ReviewNotFound, ReviewRecord

router = APIRouter(tags=["review"])


class Review(BaseModel):
    id: str
    status: Literal["pending", "approved", "rejected"]
    question: str
    reasons: list[str]
    # The answer exactly as it would have been released, including citations
    # and contradictions. What the reviewer approves is what the requester gets.
    response: QueryResponse
    trace_id: str | None
    created_at: datetime
    decided_at: datetime | None
    reviewer: str | None
    note: str | None


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    # Required: a decision nobody is accountable for is not much of a review.
    reviewer: str = Field(min_length=1, max_length=200)
    # Most useful on a rejection, where it is the only explanation the
    # requester gets.
    note: str | None = Field(default=None, max_length=4000)


def _review(record: ReviewRecord) -> Review:
    return Review(
        id=record.id,
        status=record.status,
        question=record.question,
        reasons=record.reasons,
        response=QueryResponse.model_validate(record.response),
        trace_id=record.trace_id,
        created_at=record.created_at,
        decided_at=record.decided_at,
        reviewer=record.reviewer,
        note=record.note,
    )


# Plain `def` handlers: the store is blocking SQLite, and FastAPI runs sync
# handlers on its worker threads rather than on the event loop.


@router.get("/reviews", response_model=list[Review])
def list_reviews(
    services: ServicesDep,
    status_: Literal["pending", "approved", "rejected"] | None = Query(
        default="pending", alias="status"
    ),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[Review]:
    return [_review(r) for r in services.reviews.list_reviews(status_, limit)]


@router.get("/reviews/{review_id}", response_model=Review)
def get_review(review_id: str, services: ServicesDep) -> Review:
    try:
        return _review(services.reviews.get(review_id))
    except ReviewNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such review") from None


@router.post("/reviews/{review_id}/decision", response_model=Review)
def decide(review_id: str, decision: Decision, services: ServicesDep) -> Review:
    try:
        record = services.reviews.decide(
            review_id,
            approve=decision.decision == "approve",
            reviewer=decision.reviewer,
            note=decision.note,
        )
    except ReviewNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such review") from None
    except ReviewAlreadyDecided as exc:
        # 409 rather than a silent overwrite: a second reviewer reversing the
        # first without either knowing is the failure a review trail prevents.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"already {exc.record.status} by {exc.record.reviewer}",
        ) from None
    return _review(record)
