"""Retrieval metrics, and the relevance judgement they rest on.

Three numbers, because they fail differently and a single one hides the failure:

    hit@k   did *any* relevant chunk make the top k? This is the one that
            matters most here, because the generator is given all k chunks at
            once — a relevant chunk at position 4 is as usable as one at
            position 1. It is also the crudest: it cannot distinguish "barely
            scraped in" from "ranked first".

    MRR     one over the rank of the first relevant chunk. Rewards putting the
            answer at the top, which matters for cost (a smaller k becomes
            viable) and for the model's attention budget.

    nDCG@k  rank-weighted, and unlike MRR it keeps counting after the first
            relevant chunk. The one to watch when an answer needs to combine
            two passages, which contract questions frequently do.

Relevance is binary and comes from a verbatim span, not a graded human judgement.
That is a real limitation: a chunk containing most of the answer scores zero, the
same as one about a different document entirely. It is the right trade at this
size — graded relevance over 20 questions would be three annotators disagreeing
about the difference between a 2 and a 3, which is noise dressed as precision.
"""

import math
import re
from collections.abc import Sequence

from app.vectorstore.store import ScoredChunk

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace and case, so a gold span still matches a chunk whose
    line breaks the splitter reflowed."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def is_relevant(chunk: ScoredChunk, gold_span: str) -> bool:
    return normalize(gold_span) in normalize(chunk.chunk.text)


def relevance(hits: Sequence[ScoredChunk], gold_span: str) -> list[bool]:
    return [is_relevant(hit, gold_span) for hit in hits]


def hit_at_k(judgements: Sequence[bool], k: int) -> float:
    return 1.0 if any(judgements[:k]) else 0.0


def reciprocal_rank(judgements: Sequence[bool]) -> float:
    for rank, relevant in enumerate(judgements, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(judgements: Sequence[bool], k: int) -> float:
    """Binary-gain nDCG.

    The ideal ranking puts every relevant chunk first, so the denominator is
    computed from the number actually found rather than from a fixed assumption
    of one relevant chunk per question — a question whose answer is split across
    two chunks should not be capped below 1.0 for retrieving both.
    """
    top = list(judgements[:k])
    gain = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(top, start=1) if hit)

    ideal_count = sum(top)
    if not ideal_count:
        return 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return gain / ideal


# --- answer-level context metrics ---------------------------------------------
#
# The same two numbers Ragas calls context precision and context recall, computed
# from the gold spans instead of by asking a model which chunks look relevant.
# The golden set already records the passage that answers each question, so a
# judge here would be paying to approximate a fact that is written down.
#
# They differ from the retrieval metrics above in what they are applied to: the
# chunks the *generator was shown*, after grading, not what retrieval returned.
# That is the list the answer was written from, and the difference between the two
# is exactly what grading is supposed to change.


def context_precision(judgements: Sequence[bool]) -> float:
    """Ragas's rank-weighted precision: the mean of precision@k over every
    position k that holds a relevant chunk. 1.0 when everything relevant comes
    first; 0.0 when nothing shown is relevant."""
    found = 0
    total = 0.0
    for rank, relevant in enumerate(judgements, start=1):
        if relevant:
            found += 1
            total += found / rank
    return total / found if found else 0.0


def context_recall(judgements: Sequence[bool]) -> float:
    """Whether the passage that answers the question was shown at all. Binary,
    because each question has one gold span: there is no partial credit for
    showing half of a sentence."""
    return 1.0 if any(judgements) else 0.0
