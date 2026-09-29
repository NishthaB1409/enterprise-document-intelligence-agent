"""Turning the critic's "sources 2 and 4 disagree" into spans a reviewer can open.

The critic names conflicts by source number, the same 1-based numbering as the
generator's citations, so they get the same treatment: resolved against exactly
the chunks the answer was written from, with anything that does not resolve
dropped rather than repaired.

A conflict needs two sides. One that resolves to fewer than two distinct sources
— a hallucinated "[9]", or a source said to disagree with itself — is not
evidence that the documents disagree, and flagging it would send a reviewer
looking for a contradiction that is not there. It is logged instead, so a critic
that keeps doing it is visible.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.generation.citations import Citation, citation_of
from app.graph.nodes import SourceConflict
from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Contradiction:
    description: str
    # At least two, distinct, in the order the critic named them.
    citations: list[Citation]


def resolve_conflicts(
    conflicts: Sequence[SourceConflict], chunks: Sequence[ScoredChunk]
) -> list[Contradiction]:
    """`chunks` must be the list the critic was shown, in the same order."""
    resolved: list[Contradiction] = []
    for conflict in conflicts:
        numbers = list(
            dict.fromkeys(n for n in conflict.sources if 1 <= n <= len(chunks))
        )
        if len(numbers) < 2:
            logger.info(
                "Dropped a conflict naming sources %s of %d: %s",
                conflict.sources,
                len(chunks),
                conflict.description,
            )
            continue

        resolved.append(
            Contradiction(
                description=conflict.description,
                citations=[citation_of(chunks[n - 1]) for n in numbers],
            )
        )
    return resolved
