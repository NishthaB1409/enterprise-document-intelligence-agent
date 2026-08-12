"""Turning the model's source numbers back into verifiable spans.

This is the enforcement step. The model is asked to cite; here we check that it
did, that the numbers it used were ones we actually gave it, and that what comes
back points at a real chunk of a real document.

Nothing is repaired. A claim whose citations do not resolve keeps its text and
loses its citations, and is also reported separately — a silently dropped claim
would be a fabrication the reviewer never sees, and a silently kept one would be
a fabrication wearing a citation.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.generation.answerer import GeneratedAnswer
from app.vectorstore.store import ScoredChunk


@dataclass(frozen=True, slots=True)
class Citation:
    """One source span, addressed precisely enough for a reviewer to check it."""

    chunk_id: str
    doc_id: str
    source: str
    page: int
    # Offsets into the normalized page text; -1 when the splitter reshaped
    # whitespace and the exact span was lost (see `app.ingest.chunking`).
    char_start: int
    char_end: int
    # The retrieval score of the chunk, not a confidence in the claim.
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    text: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return bool(self.citations)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str
    answerable: bool
    claims: list[GroundedClaim]
    # Every citation used anywhere in the answer, deduplicated, in the order the
    # retriever ranked them. What a UI renders as the source list.
    citations: list[Citation]
    # Claims that cited nothing resolvable. Non-empty means the answer contains
    # an assertion no source backs — the signal phase 4's review gate keys on.
    unsupported_claims: list[str]


def _citation_of(scored: ScoredChunk) -> Citation:
    return Citation(
        chunk_id=scored.chunk.id,
        doc_id=scored.chunk.doc_id,
        source=scored.source,
        page=scored.chunk.page,
        char_start=scored.chunk.char_start,
        char_end=scored.chunk.char_end,
        score=scored.score,
        text=scored.chunk.text,
    )


def ground(generated: GeneratedAnswer, chunks: Sequence[ScoredChunk]) -> GroundedAnswer:
    """Resolve 1-based source numbers against the chunks the prompt listed.

    `chunks` must be exactly the sequence passed to the answerer, in the same
    order — the numbers mean nothing otherwise.
    """
    claims: list[GroundedClaim] = []
    unsupported: list[str] = []
    # Keyed by source number so one chunk cited by three claims is one source,
    # and so the collected list can be emitted in retrieval-rank order.
    used: dict[int, Citation] = {}

    for claim in generated.claims:
        resolved: list[Citation] = []
        seen: set[int] = set()

        for number in claim.sources:
            # 1-based, and out-of-range numbers are dropped rather than clamped:
            # a hallucinated "[9]" when eight sources were given is not evidence
            # for source 8.
            if not 1 <= number <= len(chunks) or number in seen:
                continue
            seen.add(number)

            citation = _citation_of(chunks[number - 1])
            resolved.append(citation)
            used.setdefault(number, citation)

        claims.append(GroundedClaim(text=claim.text, citations=resolved))
        if not resolved:
            unsupported.append(claim.text)

    return GroundedAnswer(
        answer=generated.answer,
        answerable=generated.answerable,
        claims=claims,
        citations=[used[number] for number in sorted(used)],
        unsupported_claims=unsupported,
    )
