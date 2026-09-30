"""Measure how often the router sends an answerable question to retrieval.

    python -m eval.routing_eval [--trials 12]

Unlike `retrieval_eval`, this one costs money: it needs OPENAI_API_KEY and makes
`trials` calls per question on AGENT_MODEL. They are the cheapest calls in the
service — a boolean and a sentence, capped at 2000 tokens — but they are not
free, so the default trial count is small.

Why this exists at all. The router is the only node whose failure is silent and
terminal: routing a question to `direct` ends the graph before retrieval, and
the user gets "that does not appear to be about the documents" for a question
the corpus answers. Every other node fails open — a dead grader keeps all the
chunks, a dead critic returns None, a dead rewriter reuses the query — and a
reviewer reading the answer can tell something is missing. A wrong refusal looks
exactly like a correct one.

Why it repeats each question. The first version of this measurement asked once
per question and scored 11/11, which was luck: the router runs at the model's
default temperature, so a borderline question is not a fixed verdict but a coin
weighted somewhere. One sample cannot tell a 12/12 question from a 7/12 one, and
it is the 7/12 questions that make the feature unshippable. `--trials` is what
turns a pass/fail into a rate.

What the cases are chosen to stress. The refusals have to be *cheap* refusals —
greetings, trivia, arithmetic, meta-questions — because those are the only ones
where refusing is certainly right. The searches are deliberately colloquial and
name no document, because that is how people actually ask and it is the case the
router got wrong: a question phrased in everyday words about a topic the library
covers. Bridging the question's vocabulary to the document's is the rewriter's
job, downstream, and the router pre-empting it is the failure this measures.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Literal

from app.config import get_settings
from app.graph.nodes import Router
from app.services import build_llm


@dataclass(frozen=True, slots=True)
class Case:
    question: str
    # What the router should decide. `search` questions are answerable from a
    # contracts-and-policies library; `refuse` ones are answerable from no
    # document library at all.
    expected: Literal["search", "refuse"]
    why: str


CASES = [
    # Colloquial, document-free phrasing about topics the library governs. These
    # are the ones that regress: nothing in the wording says "contract".
    Case("How long do I get to pay a bill after they send it?", "search", "payment terms"),
    Case("Can I walk away from this early?", "search", "termination"),
    Case("What if I get hurt on the job?", "search", "liability and insurance"),
    Case("Do they have to warn me before they stop working?", "search", "notice period"),
    Case("Am I covered if something goes wrong?", "search", "indemnity"),
    Case("What happens if I pay late?", "search", "late payment"),
    Case("Can they put my prices up whenever they want?", "search", "fee variation"),
    Case("Who owns the stuff I make while working there?", "search", "IP assignment"),
    # Formal phrasing, for contrast: if these ever regress the problem is not
    # register, it is the router.
    Case("What is the notice period for termination for convenience?", "search", "explicit"),
    Case("What liability insurance must the Provider maintain?", "search", "explicit"),
    # Technical controls. Security policies and SLAs are documents too, but a
    # question about a protocol or a backup reads like general IT knowledge.
    # Added after the answer eval found "Which TLS version is used in transit?"
    # refused as "a technical specification ... not governed by enterprise
    # documents" — the security policy answers it in one sentence.
    Case("Which TLS version is used in transit?", "search", "security policy"),
    Case("How often are the backups tested?", "search", "business continuity"),
    Case("Do you use multi-factor authentication?", "search", "access control"),
    # No document library holds these, however it was filled.
    Case("Hello!", "refuse", "greeting"),
    Case("Who won the 2018 World Cup?", "refuse", "general knowledge"),
    Case("What is 17 times 23?", "refuse", "arithmetic"),
    Case("What can you do?", "refuse", "meta"),
    Case("Write me a poem about the sea.", "refuse", "unrelated generation"),
    Case("Thanks, that is all.", "refuse", "small talk"),
    # The boundary the technical cases need: what a technology *is* is general
    # knowledge; what *this organisation* does with it is policy.
    Case("What does TLS stand for?", "refuse", "general technical knowledge"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
        help="calls per question; more samples, more cost (default: 12)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.generation_configured:
        raise SystemExit(
            f"{settings.generation_key_variable} is not set — this eval calls the model."
        )

    logging.getLogger("langfuse").setLevel(logging.ERROR)

    router = Router(build_llm(settings, model=settings.agent_model))
    print(
        f"model: {settings.agent_model}  cases: {len(CASES)}  "
        f"trials each: {args.trials}\n"
    )

    total = 0
    failures: list[tuple[Case, int]] = []
    for case in CASES:
        searched = sum(router.route(case.question).needs_documents for _ in range(args.trials))
        correct = searched if case.expected == "search" else args.trials - searched
        total += correct
        # Anything short of unanimous is worth seeing. A question the router
        # only usually gets right is a question some users are refused.
        flag = "    " if correct == args.trials else "  ! "
        print(f"{flag}{correct:>2}/{args.trials}  {case.expected:6}  {case.question}")
        if correct < args.trials:
            failures.append((case, correct))

    denominator = len(CASES) * args.trials
    print(f"\n{total}/{denominator} correct ({total / denominator:.1%})")

    if failures:
        print("\n## Not unanimous\n")
        for case, correct in failures:
            print(f"  {correct}/{args.trials}  {case.question}  ({case.why})")


if __name__ == "__main__":
    main()
