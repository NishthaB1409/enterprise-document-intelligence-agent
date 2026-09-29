"""The review gate's rules, contradiction resolution, and the queue's storage.

The gate is pure: signals in, reasons out. So every rule is pinned here directly,
including the ones that must *not* fire — an honest "not covered" and an
unreviewed-because-disabled answer are both correct outcomes, and a gate that
held them would bury the answers that actually need a human.
"""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.generation.answerer import GeneratedAnswer
from app.generation.citations import ground
from app.graph.nodes import Critique, SourceConflict
from app.graph.pipeline import PipelineResult
from app.ingest.chunking import Chunk
from app.main import create_app
from app.review.contradictions import resolve_conflicts
from app.review.gate import review_reasons
from app.review.store import ReviewAlreadyDecided, ReviewNotFound, ReviewStore
from app.vectorstore.store import ScoredChunk


def _chunk(index: int, text: str, source: str = "contract.pdf") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"chunk-{index}",
            doc_id=f"doc-{source}",
            index=index,
            page=index + 1,
            text=text,
            char_start=0,
            char_end=len(text),
        ),
        source=source,
        score=0.9,
    )


THIRTY = _chunk(0, "Invoices are payable within thirty (30) days of receipt.")
FORTY_FIVE = _chunk(1, "All fees are due within forty-five (45) days.", "schedule-b.pdf")
CHUNKS = [THIRTY, FORTY_FIVE]

CITED = GeneratedAnswer.model_validate(
    {
        "answer": "Within thirty days.",
        "answerable": True,
        "claims": [{"text": "Invoices are payable within thirty days.", "sources": [1]}],
    }
)
CLEAN = Critique(supported=True, confidence=0.9, concerns=[])


def _reasons(
    answer: GeneratedAnswer = CITED,
    *,
    critique: Critique | None = None,
    critique_failed: bool = False,
    chunks=CHUNKS,
    min_confidence: float = 0.7,
) -> list[str]:
    return review_reasons(
        ground(answer, chunks),
        critique=critique,
        critique_failed=critique_failed,
        contradictions=resolve_conflicts(critique.conflicts if critique else [], chunks),
        min_confidence=min_confidence,
    )


# --- the gate -------------------------------------------------------------------


class TestGate:
    def test_a_cited_answer_with_a_clean_critique_is_released(self):
        assert _reasons(critique=CLEAN) == []

    def test_a_cited_answer_with_no_critic_is_released(self):
        """The single-pass default has no critic. Holding every answer for
        lacking one would make the gate useless in the shipped configuration."""
        assert _reasons(critique=None) == []

    def test_an_honest_not_covered_is_released(self):
        declined = GeneratedAnswer(
            answer="No indexed document covers this.", claims=[], answerable=False
        )
        assert _reasons(declined, chunks=[]) == []

    def test_an_uncited_claim_holds_the_answer(self):
        answer = GeneratedAnswer.model_validate(
            {
                "answer": "x",
                "answerable": True,
                "claims": [{"text": "A fee applies.", "sources": [7]}],
            }
        )
        assert _reasons(answer) == ["claim cites no retrieved source: 'A fee applies.'"]

    def test_answering_without_citing_anything_holds_it(self):
        answer = GeneratedAnswer(answer="Thirty days.", claims=[], answerable=True)
        assert _reasons(answer) == ["answered from the documents but cited nothing"]

    def test_an_unreachable_critic_holds_it(self):
        """Unreviewed is not reviewed-and-clean."""
        reasons = _reasons(critique=None, critique_failed=True)
        assert reasons == ["the critic could not be reached; the answer is unreviewed"]

    def test_an_unsupported_verdict_holds_it(self):
        critique = Critique(supported=False, confidence=0.9, concerns=["figure"])
        assert _reasons(critique=critique) == [
            "the critic judged the answer not supported by its sources"
        ]

    @pytest.mark.parametrize(("confidence", "held"), [(0.69, True), (0.7, False), (0.95, False)])
    def test_confidence_below_the_threshold_holds_it(self, confidence, held):
        critique = Critique(supported=True, confidence=confidence, concerns=[])
        assert bool(_reasons(critique=critique, min_confidence=0.7)) is held

    def test_disagreeing_sources_hold_it_even_when_the_answer_is_fine(self):
        """The answer may have picked the right side. The reader still needs to
        know there were two."""
        critique = CLEAN.model_copy(
            update={
                "conflicts": [
                    SourceConflict(sources=[1, 2], description="30 days vs 45 days")
                ]
            }
        )
        assert _reasons(critique=critique) == ["sources disagree: 30 days vs 45 days"]

    def test_every_reason_is_reported_not_just_the_first(self):
        answer = GeneratedAnswer.model_validate(
            {"answer": "x", "answerable": True, "claims": [{"text": "c", "sources": []}]}
        )
        critique = Critique(supported=False, confidence=0.1, concerns=[])
        assert len(_reasons(answer, critique=critique)) == 3


# --- contradiction resolution ------------------------------------------------


class TestResolveConflicts:
    def test_a_conflict_resolves_to_both_spans(self):
        (contradiction,) = resolve_conflicts(
            [SourceConflict(sources=[1, 2], description="30 vs 45 days")], CHUNKS
        )

        assert contradiction.description == "30 vs 45 days"
        assert [c.source for c in contradiction.citations] == [
            "contract.pdf",
            "schedule-b.pdf",
        ]
        assert contradiction.citations[1].page == 2

    @pytest.mark.parametrize(
        "sources",
        [
            [1],  # one side is not a disagreement
            [1, 1],  # nor is a source disagreeing with itself
            [1, 9],  # a hallucinated source is not evidence
            [],
        ],
    )
    def test_a_conflict_without_two_real_sides_is_dropped(self, sources):
        assert resolve_conflicts([SourceConflict(sources=sources, description="d")], CHUNKS) == []

    def test_out_of_range_numbers_are_dropped_not_clamped(self):
        (contradiction,) = resolve_conflicts(
            [SourceConflict(sources=[0, 1, 2, 3], description="d")], CHUNKS
        )
        assert [c.chunk_id for c in contradiction.citations] == ["chunk-0", "chunk-1"]


# --- the store ------------------------------------------------------------------


@pytest.fixture
def reviews(tmp_path) -> ReviewStore:
    return ReviewStore(str(tmp_path / "nested" / "reviews.sqlite3"))


def _create(store: ReviewStore, question: str = "q"):
    return store.create(
        question=question, reasons=["r"], response={"answer": "a"}, trace_id="t"
    )


class TestReviewStore:
    def test_constructing_it_touches_no_disk(self, tmp_path):
        """Importing the app builds one. That must not create files."""
        path = tmp_path / "nested" / "reviews.sqlite3"
        ReviewStore(str(path))
        assert not path.parent.exists()

    def test_a_review_survives_a_new_store_on_the_same_file(self, reviews, tmp_path):
        """A restart must not drop the queue."""
        created = _create(reviews)

        reopened = ReviewStore(str(tmp_path / "nested" / "reviews.sqlite3"))
        fetched = reopened.get(created.id)

        assert fetched.status == "pending"
        assert fetched.response == {"answer": "a"}
        assert fetched.reasons == ["r"]
        assert fetched.created_at == created.created_at

    def test_deciding_records_who_and_when(self, reviews):
        created = _create(reviews)

        decided = reviews.decide(created.id, approve=False, reviewer="alice", note="wrong")

        assert decided.status == "rejected"
        assert decided.reviewer == "alice"
        assert decided.note == "wrong"
        assert decided.decided_at is not None

    def test_a_second_decision_is_refused(self, reviews):
        created = _create(reviews)
        reviews.decide(created.id, approve=True, reviewer="alice")

        with pytest.raises(ReviewAlreadyDecided) as caught:
            reviews.decide(created.id, approve=False, reviewer="bob")

        assert caught.value.record.reviewer == "alice"
        assert reviews.get(created.id).status == "approved"

    def test_unknown_ids_are_not_found(self, reviews):
        with pytest.raises(ReviewNotFound):
            reviews.get("missing")
        with pytest.raises(ReviewNotFound):
            reviews.decide("missing", approve=True, reviewer="x")

    def test_listing_filters_by_status_and_bounds_the_page(self, reviews):
        ids = [_create(reviews, f"q{i}").id for i in range(3)]
        reviews.decide(ids[1], approve=True, reviewer="x")

        assert [r.id for r in reviews.list_reviews("pending")] == [ids[0], ids[2]]
        assert [r.id for r in reviews.list_reviews("approved")] == [ids[1]]
        assert len(reviews.list_reviews(None, limit=2)) == 2


# --- through the API, with a critic -------------------------------------------


class _Pipeline:
    """Stands in for the agentic graph: returns a fixed result."""

    def __init__(self, result: PipelineResult) -> None:
        self._result = result

    def run(self, question: str, top_k: int | None = None) -> PipelineResult:
        return self._result


@pytest.fixture
def agent_client(client: TestClient, services):
    """The session app runs the single pass. This one reports what the graph
    would — a critique, and possibly a critic failure — without running it.

    Not entered as a context manager: the lifespan already ran for the session
    app, and running it again would re-initialise the Langfuse singleton.
    """

    def _build(result: PipelineResult) -> TestClient:
        app = create_app(client.app.state.settings, replace(services, pipeline=_Pipeline(result)))
        return TestClient(app)

    return _build


def test_contradictions_reach_the_reviewer_as_spans(agent_client):
    critique = CLEAN.model_copy(
        update={"conflicts": [SourceConflict(sources=[1, 2], description="30 vs 45 days")]}
    )
    test_client = agent_client(
        PipelineResult(answer=CITED, chunks=CHUNKS, critique=critique, steps=["critique:0.90"])
    )

    held = test_client.post("/api/v1/query", json={"question": "When are invoices due?"})

    assert held.status_code == 202
    assert held.json()["contradictions"] == []  # withheld with the rest
    review = test_client.get(f"/api/v1/reviews/{held.json()['review']['id']}").json()
    (contradiction,) = review["response"]["contradictions"]
    assert contradiction["description"] == "30 vs 45 days"
    assert [c["source"] for c in contradiction["citations"]] == [
        "contract.pdf",
        "schedule-b.pdf",
    ]


def test_a_failed_critic_holds_the_answer_through_the_api(agent_client):
    test_client = agent_client(
        PipelineResult(answer=CITED, chunks=CHUNKS, critique_failed=True)
    )

    response = test_client.post("/api/v1/query", json={"question": "When are invoices due?"})

    assert response.status_code == 202
    assert "unreviewed" in response.json()["review"]["reasons"][0]


def test_a_clean_critique_is_released_with_its_verdict(agent_client):
    test_client = agent_client(PipelineResult(answer=CITED, chunks=CHUNKS, critique=CLEAN))

    response = test_client.post("/api/v1/query", json={"question": "When are invoices due?"})

    assert response.status_code == 200
    assert response.json()["critique"]["confidence"] == 0.9
    assert response.json()["contradictions"] == []
