"""Turning "sources 2 and 4 disagree" into spans a reviewer can open.

Two things report conflicts, by source number, in the same 1-based numbering as
the generator's citations: the answerer, which runs on every pipeline, and the
critic, which runs only in the agentic graph. Both get the same treatment as a
citation: resolved against exactly the chunks the answer was written from, with
anything that does not resolve dropped rather than repaired.

A conflict needs two sides. One that resolves to fewer than two distinct sources
— a hallucinated "[9]", or a source said to disagree with itself — is not
evidence that the documents disagree, and flagging it would send a reviewer
looking for a contradiction that is not there. It is logged instead, so a model
that keeps doing it is visible.

The models describe conflicts the way they were shown the sources — "Source [1]
says thirty days, source [2] says forty-five" — which means nothing to a reviewer
who never saw the numbering. Descriptions are rewritten with the document and
page each number stood for.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.generation.answerer import GeneratedAnswer, SourceConflict
from app.generation.citations import Citation, citation_of
from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Contradiction:
    # Written for a reviewer: source numbers replaced by document and page.
    description: str
    # At least two, distinct, in the order the conflict named them.
    citations: list[Citation]


def resolve_conflicts(
    conflicts: Sequence[SourceConflict], chunks: Sequence[ScoredChunk]
) -> list[Contradiction]:
    """`chunks` must be the list the reporter was shown, in the same order."""
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
                description=readable(conflict.description, chunks),
                citations=[citation_of(chunks[n - 1]) for n in numbers],
            )
        )
    return resolved


def collect_contradictions(
    answer: GeneratedAnswer, critique_conflicts: Sequence[SourceConflict], chunks: Sequence[ScoredChunk]
) -> list[Contradiction]:
    """Every conflict either reporter found, each pair of sources once.

    The critic's version comes first when both report the same conflict: it is
    the independent read, where the answerer is describing evidence it has just
    chosen how to use. Two conflicts are the same when they name the same set of
    chunks, whatever words each reporter used.
    """
    merged: dict[frozenset[str], Contradiction] = {}
    for contradiction in resolve_conflicts(critique_conflicts, chunks) + resolve_conflicts(
        answer.conflicts, chunks
    ):
        key = frozenset(c.chunk_id for c in contradiction.citations)
        merged.setdefault(key, contradiction)
    return list(merged.values())


# "Source [2]", "source 2", "sources [1] and [3]", or a bare "[2]".
_SOURCE_REFERENCE = re.compile(r"\b[Ss]ources?\s+\[?(\d+)\]?|\[(\d+)\]")


def readable(description: str, chunks: Sequence[ScoredChunk]) -> str:
    """Replace each source number with the document and page it stood for.

    A number outside the list is left as written: guessing which document a
    hallucinated "[9]" meant would put a citation the model never made into a
    reviewer's hands.
    """

    def name(match: re.Match[str]) -> str:
        number = int(match.group(1) or match.group(2))
        if not 1 <= number <= len(chunks):
            return match.group(0)
        chunk = chunks[number - 1]
        return f"{chunk.source} p{chunk.chunk.page}"

    return _SOURCE_REFERENCE.sub(name, description)
