"""Which answers are held for a human before the requester sees them.

The rules are deterministic and cost nothing: every signal they read was already
produced on the way to the answer. Grounding produced `unsupported_claims`; the
critic, when the agentic graph runs it, produced a confidence, a verdict, and
any conflicts between sources. The gate adds no model call of its own, which is
why it can run on the single-pass pipeline too — there it simply has fewer
signals to read.

What holds an answer, and why each one:

    an unsupported claim     an assertion no retrieved source backs. The
                             citation contract exists to make this detectable;
                             releasing it anyway would make the check decorative.

    answerable, no claims    the model said the documents answer the question
                             and then cited nothing. Either the claims were
                             dropped or the answer came from the model's memory.

    critic unavailable       a critic was configured and could not be reached.
                             Unreviewed is not the same as reviewed and clean,
                             so it is not released as though it were.

    critic: unsupported      the independent pass disagreed with the generator
                             about whether its own citations hold up.

    low critic confidence    below `review_min_confidence` (0.7). On the eval
                             cases conflicts never scored above 0.40 and clean
                             answers never below 0.80 — see `eval/REVIEW.md`.

    sources disagree         the documents contradict each other on something
                             the question turns on. The answer may have picked
                             the right side; a reader still needs to know there
                             were two.

What does not: an honest "no document covers this". Declining to answer is the
correct outcome when the evidence is not there, and holding it for review would
teach reviewers that most of the queue needs no attention.
"""

from collections.abc import Sequence

from app.generation.citations import GroundedAnswer
from app.graph.nodes import Critique
from app.review.contradictions import Contradiction


def review_reasons(
    answer: GroundedAnswer,
    *,
    critique: Critique | None,
    critique_failed: bool,
    contradictions: Sequence[Contradiction],
    min_confidence: float,
) -> list[str]:
    """Every reason this answer should be held, in a form a reviewer can act on.

    Empty means release it. All reasons are collected rather than stopping at the
    first, because the reviewer is about to read the answer anyway and a second
    problem found after approving the first is a second round trip.
    """
    reasons = [
        f"claim cites no retrieved source: {claim!r}" for claim in answer.unsupported_claims
    ]

    if answer.answerable and not answer.claims:
        reasons.append("answered from the documents but cited nothing")

    if critique_failed:
        reasons.append("the critic could not be reached; the answer is unreviewed")

    if critique is not None:
        if not critique.supported:
            reasons.append("the critic judged the answer not supported by its sources")
        if critique.confidence < min_confidence:
            reasons.append(
                f"critic confidence {critique.confidence:.2f} is below {min_confidence:.2f}"
            )

    reasons.extend(f"sources disagree: {c.description}" for c in contradictions)
    return reasons
