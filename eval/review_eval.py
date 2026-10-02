"""Measure whether the critic spots sources that contradict each other.

    python -m eval.review_eval [--trials 6]

Needs OPENAI_API_KEY and costs money: `trials` critique calls per case on
AGENT_MODEL, plus `trials` grading calls per conflict case, each a few hundred
tokens in and out. At the defaults that is 108 calls.

It checks the grader too, because the grader runs first and the critic only sees
what it keeps. On the first live run the grader dropped one side of a conflict
in 2 of 5 trials, giving "it contradicts the other source" as its reason. It had
settled the disagreement itself, and the critic, left with one source, had
nothing to flag. A critic that catches every conflict it is shown is worth
nothing if grading hides the conflict first.

What it measures. Contradiction detection rides on the critique call rather
than getting its own, so it costs nothing extra per query — but it also shares a
prompt with three other checks, on the cheaper model. Whether that is good
enough is an empirical question, and there are two ways to get it wrong:

    missed conflict     two sources disagree and the answer goes out citing one
                        of them. The failure the feature exists to prevent.

    false alarm         two sources say different things about *different*
                        terms and the answer is held anyway. Each one costs a
                        reviewer's time, and a queue full of them teaches
                        reviewers to approve without reading.

So the cases come in two halves. The conflicts are the same term stated twice,
differently — the realistic version being a main agreement and a schedule or an
old policy and a new one. The non-conflicts are the hard negatives: figures of
the same unit for different terms, and exceptions or pro-rata rules that refine a
term rather than contradict it. A detector that fires on any two differing
numbers would pass the first half and fail the second.

It also prints the critic's confidence per case, which is what
`REVIEW_MIN_CONFIDENCE` is set from. The first run put every conflict at or
below 0.40 and every clean case at or above 0.80 (eval/REVIEW.md).

Why it repeats each case: the same reason `routing_eval` does. One sample cannot
tell a 6/6 case from a 3/6 one, and it is the 3/6 ones that matter.
"""

import argparse
import logging
import statistics
from dataclasses import dataclass

from app.config import get_settings
from app.generation.answerer import LLMAnswerer
from app.graph.nodes import Critic, Grader
from app.ingest.chunking import Chunk
from app.review.contradictions import resolve_conflicts
from app.services import build_llm
from app.vectorstore.store import ScoredChunk


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    question: str
    # The answer being critiqued. Each one cites the first source, as a real
    # answer that picked one side of a conflict would.
    answer: str
    sources: tuple[tuple[str, str], ...]  # (document name, text)
    conflict: bool


CASES = [
    # --- the same term, stated differently -----------------------------------
    Case(
        "payment term",
        "How long do we have to pay an invoice?",
        "Invoices must be paid within thirty days of receipt.",
        (
            ("msa.pdf", "6.1 Invoices are payable within thirty (30) days of receipt."),
            ("schedule-b.pdf", "Payment. All fees invoiced under this Schedule are due within forty-five (45) days of the invoice date."),
        ),
        conflict=True,
    ),
    Case(
        "governing law",
        "Which law governs the agreement?",
        "The agreement is governed by the laws of England and Wales.",
        (
            ("msa.pdf", "18.1 This Agreement is governed by the laws of England and Wales."),
            ("order-form.pdf", "This Order Form and the Agreement it incorporates shall be construed in accordance with the laws of the State of New York."),
        ),
        conflict=True,
    ),
    Case(
        "leave entitlement",
        "How much annual leave do full-time staff get?",
        "Full-time employees receive 25 days of annual leave.",
        (
            ("handbook-2024.pdf", "Full-time employees are entitled to 25 days of paid annual leave per holiday year, plus public holidays."),
            ("leave-policy.pdf", "Annual leave for full-time staff is 20 days per year, exclusive of bank holidays."),
        ),
        conflict=True,
    ),
    Case(
        "commencement date",
        "When does the agreement start?",
        "The agreement commences on 1 March 2026.",
        (
            ("msa.pdf", "This Agreement commences on 1 March 2026 (the Commencement Date)."),
            ("msa.pdf", "Definitions. 'Effective Date' and 'Commencement Date' mean 1 April 2026."),
        ),
        conflict=True,
    ),
    Case(
        "insurance level",
        "How much professional indemnity insurance must the Provider carry?",
        "At least £2,000,000 of professional indemnity insurance.",
        (
            ("msa.pdf", "12.2 The Provider shall maintain professional indemnity insurance of not less than £2,000,000 per claim."),
            ("security-schedule.pdf", "The Provider must hold professional indemnity cover of at least £5,000,000 for each and every claim."),
        ),
        conflict=True,
    ),
    Case(
        "liability cap",
        "What is the cap on the supplier's liability?",
        "Liability is capped at the fees paid in the preceding twelve months.",
        (
            ("msa.pdf", "14.2 The Supplier's aggregate liability shall not exceed the fees paid in the twelve (12) months preceding the claim."),
            ("msa.pdf", "14.3 Notwithstanding anything else, the Supplier's total liability under this Agreement is limited to £1,000,000."),
        ),
        conflict=True,
    ),
    # --- different terms, or a refinement: no conflict -------------------------
    Case(
        "payment vs notice",
        "How long do we have to pay an invoice?",
        "Invoices must be paid within thirty days of receipt.",
        (
            ("msa.pdf", "6.1 Invoices are payable within thirty (30) days of receipt."),
            ("msa.pdf", "17.2 Either party may terminate this Agreement for convenience on ninety (90) days' written notice."),
        ),
        conflict=False,
    ),
    Case(
        "cap and carve-out",
        "What is the cap on the supplier's liability?",
        "Liability is capped at the fees paid in the preceding twelve months.",
        (
            ("msa.pdf", "14.2 The Supplier's aggregate liability shall not exceed the fees paid in the twelve (12) months preceding the claim."),
            ("msa.pdf", "14.4 The limit in clause 14.2 does not apply to death or personal injury caused by negligence."),
        ),
        conflict=False,
    ),
    Case(
        "leave pro rata",
        "How much annual leave do full-time staff get?",
        "Full-time employees receive 25 days of annual leave.",
        (
            ("handbook-2024.pdf", "Full-time employees are entitled to 25 days of paid annual leave per holiday year."),
            ("handbook-2024.pdf", "Part-time employees receive annual leave pro rata to their contracted hours."),
        ),
        conflict=False,
    ),
    Case(
        "law and jurisdiction",
        "Which law governs the agreement?",
        "The agreement is governed by the laws of England and Wales.",
        (
            ("msa.pdf", "18.1 This Agreement is governed by the laws of England and Wales."),
            ("msa.pdf", "18.2 The courts of England have exclusive jurisdiction over any dispute arising from it."),
        ),
        conflict=False,
    ),
    Case(
        "two insurance types",
        "How much professional indemnity insurance must the Provider carry?",
        "At least £2,000,000 of professional indemnity insurance.",
        (
            ("msa.pdf", "12.2 The Provider shall maintain professional indemnity insurance of not less than £2,000,000 per claim."),
            ("msa.pdf", "12.3 The Provider shall maintain public liability insurance of not less than £5,000,000 per occurrence."),
        ),
        conflict=False,
    ),
    Case(
        "rate and deadline",
        "What happens if we pay late?",
        "Late payments accrue interest at 4% a year above the Bank of England base rate.",
        (
            ("msa.pdf", "6.3 Overdue sums accrue interest at 4% per annum above the Bank of England base rate."),
            ("msa.pdf", "6.1 Invoices are payable within thirty (30) days of receipt."),
        ),
        conflict=False,
    ),
]


def _chunks(case: Case) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                id=f"{case.name}-{i}",
                doc_id=source,
                index=i,
                page=1,
                text=text,
                char_start=0,
                char_end=len(text),
            ),
            source=source,
            score=1.0,
        )
        for i, (source, text) in enumerate(case.sources)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=6,
        help="critique calls per case; more samples, more cost (default: 6)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.generation_configured:
        raise SystemExit(
            f"{settings.generation_key_variable} is not set — this eval calls the model."
        )

    logging.getLogger("langfuse").setLevel(logging.ERROR)

    llm = build_llm(settings, model=settings.agent_model)
    critic = Critic(llm)
    grader = Grader(llm)
    # The single-pass pipeline has no critic: there the answerer's own
    # `conflicts` field is the only signal, so it is measured on the same cases.
    answerer = LLMAnswerer(build_llm(settings), max_tokens=settings.answer_max_tokens)
    print(
        f"critic/grader: {settings.agent_model}  answerer: {settings.answer_model}  "
        f"cases: {len(CASES)}  trials each: {args.trials}\n"
        f"{'':4}{'critic':>9}  {'expect':8}{'conf (min/median)':>19}  {'graded':>7}"
        f"  {'answerer':>8}  case"
    )

    detected = alarms = failures = kept_both = 0
    answerer_detected = answerer_alarms = 0
    all_confidences: dict[bool, list[float]] = {True: [], False: []}
    for case in CASES:
        chunks = _chunks(case)
        flagged = 0
        confidences: list[float] = []
        for _ in range(args.trials):
            critique = critic.review(case.question, case.answer, chunks)
            if critique is None:
                failures += 1
                continue
            confidences.append(critique.confidence)
            # Scored after resolution, exactly as the gate sees it: a conflict
            # naming one real source does not count as a detection.
            flagged += bool(resolve_conflicts(critique.conflicts, chunks))

        if case.conflict:
            detected += flagged
        else:
            alarms += flagged
        all_confidences[case.conflict].extend(confidences)

        # Conflict cases only: whether both sides survive grading to reach the
        # critic at all. Clean cases are not graded, because dropping an
        # unrelated source there is the grader doing its job.
        graded = "-"
        survived = args.trials
        if case.conflict:
            survived = sum(
                len(grader.keep_relevant(case.question, chunks)[0]) == len(chunks)
                for _ in range(args.trials)
            )
            kept_both += survived
            graded = f"{survived}/{args.trials}"

        # The answerer sees the sources directly (the single pass has no
        # grading) and is scored after resolution, like the critic.
        answered = sum(
            bool(resolve_conflicts(answerer.answer(case.question, chunks).conflicts, chunks))
            for _ in range(args.trials)
        )
        if case.conflict:
            answerer_detected += answered
        else:
            answerer_alarms += answered

        correct = flagged if case.conflict else args.trials - flagged
        # A conflict case is only clean if the critic caught it *and* grading
        # let it through: in the pipeline both have to hold.
        correct = min(correct, survived)
        answerer_correct = answered if case.conflict else args.trials - answered
        mark = "    " if correct == answerer_correct == args.trials else "  ! "
        conf = (
            f"{min(confidences):.2f}/{statistics.median(confidences):.2f}"
            if confidences
            else "-"
        )
        expect = "conflict" if case.conflict else "none"
        answerer_col = f"{answered}/{args.trials}"
        print(
            f"{mark}{flagged:>2}/{args.trials:<6}  {expect:8}{conf:>19}  {graded:>7}"
            f"  {answerer_col:>8}  {case.name}"
        )

    positives = sum(c.conflict for c in CASES) * args.trials
    negatives = sum(not c.conflict for c in CASES) * args.trials
    print(
        f"\nconflicts detected: {detected}/{positives} ({detected / positives:.1%})"
        f"\nfalse alarms:       {alarms}/{negatives} ({alarms / negatives:.1%})"
        f"\ngrader kept both sides of a conflict: {kept_both}/{positives}"
        f" ({kept_both / positives:.1%})"
        f"\n\nanswerer (single pass), conflicts detected: {answerer_detected}/{positives}"
        f" ({answerer_detected / positives:.1%})"
        f"\nanswerer (single pass), false alarms:       {answerer_alarms}/{negatives}"
        f" ({answerer_alarms / negatives:.1%})"
    )
    for conflict, label in ((True, "conflict cases"), (False, "clean cases")):
        values = all_confidences[conflict]
        if values:
            print(
                f"critic confidence on {label}: median {statistics.median(values):.2f}, "
                f"min {min(values):.2f}"
            )
    if failures:
        print(f"\n{failures} critique call(s) failed and were not scored.")


if __name__ == "__main__":
    main()
