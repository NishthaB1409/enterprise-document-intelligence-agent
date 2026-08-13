"""Grounding: what the model claims versus what the sources actually support.

These are the tests that make "every claim maps to a source span" a property of
the system rather than a hope about the prompt.
"""

from app.generation.answerer import GeneratedAnswer
from app.generation.citations import ground
from app.ingest.chunking import Chunk
from app.vectorstore.store import ScoredChunk


def _chunk(index: int, text: str, *, page: int = 1) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"chunk-{index}",
            doc_id="doc-1",
            index=index,
            page=page,
            text=text,
            char_start=index * 100,
            char_end=index * 100 + len(text),
        ),
        source="contract.pdf",
        # Descending, as a retriever would return them.
        score=1.0 - index / 10,
    )


CHUNKS = [
    _chunk(0, "Either party may terminate on thirty days notice.", page=2),
    _chunk(1, "The agreement begins on 1 January 2026.", page=1),
    _chunk(2, "Governing law is the State of Delaware.", page=7),
]


def _answer(**overrides) -> GeneratedAnswer:
    return GeneratedAnswer.model_validate(
        {"answer": "...", "answerable": True, "claims": []} | overrides
    )


def test_source_numbers_resolve_to_exact_spans():
    grounded = ground(
        _answer(claims=[{"text": "Termination needs thirty days notice.", "sources": [1]}]),
        CHUNKS,
    )

    (claim,) = grounded.claims
    (citation,) = claim.citations

    assert claim.supported
    # 1-based: source [1] is the first chunk, not the second.
    assert citation.chunk_id == "chunk-0"
    assert citation.page == 2
    assert (citation.char_start, citation.char_end) == (0, len(CHUNKS[0].chunk.text))
    assert citation.source == "contract.pdf"
    assert grounded.unsupported_claims == []


def test_a_source_number_we_never_gave_is_dropped():
    grounded = ground(
        _answer(claims=[{"text": "The contract auto-renews annually.", "sources": [9]}]),
        CHUNKS,
    )

    (claim,) = grounded.claims
    # Not clamped to the nearest real source: a hallucinated [9] is not evidence
    # for source 3.
    assert claim.citations == []
    assert not claim.supported
    assert grounded.unsupported_claims == ["The contract auto-renews annually."]
    assert grounded.citations == []


def test_an_unsupported_claim_keeps_its_text():
    """Dropping it would hide a fabrication from the reviewer; keeping it with a
    citation would launder one."""
    grounded = ground(
        _answer(
            claims=[
                {"text": "Notice is thirty days.", "sources": [1]},
                {"text": "Penalties accrue at 5% monthly.", "sources": []},
            ]
        ),
        CHUNKS,
    )

    assert [claim.text for claim in grounded.claims] == [
        "Notice is thirty days.",
        "Penalties accrue at 5% monthly.",
    ]
    assert [claim.supported for claim in grounded.claims] == [True, False]
    assert grounded.unsupported_claims == ["Penalties accrue at 5% monthly."]


def test_repeated_sources_collapse_to_one_citation_in_rank_order():
    grounded = ground(
        _answer(
            claims=[
                {"text": "Notice is thirty days.", "sources": [1, 1]},
                {"text": "Delaware law governs.", "sources": [3, 1]},
            ]
        ),
        CHUNKS,
    )

    # Per claim: the duplicate within a claim is dropped.
    assert [len(claim.citations) for claim in grounded.claims] == [1, 2]
    # Across the answer: one chunk cited twice is one source, and the collected
    # list stays in the order the retriever ranked them, not first-cited order.
    assert [citation.chunk_id for citation in grounded.citations] == ["chunk-0", "chunk-2"]


def test_an_unanswerable_response_carries_no_citations():
    grounded = ground(
        _answer(
            answer="The retrieved documents do not state a penalty rate.",
            answerable=False,
            claims=[],
        ),
        CHUNKS,
    )

    assert not grounded.answerable
    assert grounded.claims == []
    assert grounded.citations == []
    # An honest "I don't know" is not an unsupported claim.
    assert grounded.unsupported_claims == []
