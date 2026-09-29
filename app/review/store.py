"""The review queue: held answers, and what a reviewer decided about each.

SQLite, because a queue that forgets its contents on restart is not a queue. A
held answer is a promise to the person who asked — "a human will look at this" —
and a deploy that silently drops every pending review breaks it without anyone
noticing. SQLite is in the standard library, needs no service beside the app,
and a file on a volume survives exactly the restarts that matter.

Each operation opens its own connection. That costs microseconds against a
request that has just spent seconds on a model call, and it is what makes the
store safe to call from FastAPI's worker threads without a connection shared
across them. It also means constructing the store touches no disk: the file and
table appear on first use, so importing the app stays side-effect free.

A decision is a single conditional UPDATE, so two reviewers deciding the same
answer at once cannot both win — the second sees it already decided.

Like the rest of this service it has no authentication of its own. The reviewer
name on a decision is recorded, not verified; put the review routes behind your
identity provider before the queue guards anything that matters.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Status = Literal["pending", "approved", "rejected"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    question    TEXT NOT NULL,
    reasons     TEXT NOT NULL,
    response    TEXT NOT NULL,
    trace_id    TEXT,
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    reviewer    TEXT,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS reviews_by_status ON reviews (status, created_at);
"""


class ReviewNotFound(LookupError):
    pass


class ReviewAlreadyDecided(RuntimeError):
    def __init__(self, record: "ReviewRecord") -> None:
        super().__init__(f"review {record.id} was already {record.status}")
        self.record = record


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    id: str
    status: Status
    question: str
    reasons: list[str]
    # The complete response as it would have been released, citations and all.
    # Stored whole rather than re-derived on approval: the reviewer approves
    # what they read, and re-running the pipeline could produce something else.
    response: dict[str, Any]
    trace_id: str | None
    created_at: datetime
    decided_at: datetime | None = None
    reviewer: str | None = None
    note: str | None = None


class ReviewStore:
    def __init__(self, path: str) -> None:
        self._path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as db:
            db.row_factory = sqlite3.Row
            # Idempotent and cheap; running it per connection is what lets the
            # constructor stay free of I/O.
            db.executescript(_SCHEMA)
            with db:  # commits on success, rolls back on error
                yield db

    def create(
        self,
        *,
        question: str,
        reasons: list[str],
        response: dict[str, Any],
        trace_id: str | None,
    ) -> ReviewRecord:
        record = ReviewRecord(
            id=uuid.uuid4().hex,
            status="pending",
            question=question,
            reasons=list(reasons),
            response=response,
            trace_id=trace_id,
            created_at=datetime.now(UTC),
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO reviews (id, status, question, reasons, response, trace_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.status,
                    record.question,
                    json.dumps(record.reasons),
                    json.dumps(record.response),
                    record.trace_id,
                    record.created_at.isoformat(),
                ),
            )
        return record

    def get(self, review_id: str) -> ReviewRecord:
        with self._connect() as db:
            row = db.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if row is None:
            raise ReviewNotFound(review_id)
        return _record(row)

    def list_reviews(
        self, status: Status | None = None, limit: int = 50
    ) -> list[ReviewRecord]:
        """Oldest first: a queue is worked from the front."""
        query = "SELECT * FROM reviews"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at, id LIMIT ?"

        with self._connect() as db:
            rows = db.execute(query, (*params, limit)).fetchall()
        return [_record(row) for row in rows]

    def decide(
        self, review_id: str, *, approve: bool, reviewer: str, note: str | None = None
    ) -> ReviewRecord:
        """Approve or reject a pending review. A decision is final."""
        with self._connect() as db:
            updated = db.execute(
                "UPDATE reviews SET status = ?, decided_at = ?, reviewer = ?, note = ?"
                " WHERE id = ? AND status = 'pending'",
                (
                    "approved" if approve else "rejected",
                    datetime.now(UTC).isoformat(),
                    reviewer,
                    note,
                    review_id,
                ),
            ).rowcount

        record = self.get(review_id)  # raises ReviewNotFound for an unknown id
        if not updated:
            raise ReviewAlreadyDecided(record)
        return record


def _record(row: sqlite3.Row) -> ReviewRecord:
    return ReviewRecord(
        id=row["id"],
        status=row["status"],
        question=row["question"],
        reasons=json.loads(row["reasons"]),
        response=json.loads(row["response"]),
        trace_id=row["trace_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        decided_at=datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
        reviewer=row["reviewer"],
        note=row["note"],
    )
