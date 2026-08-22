"""The golden set: twenty questions with the passage that answers each.

Relevance is defined by a **gold span** — a verbatim phrase from the corpus —
rather than by a chunk id. Chunk ids move whenever the chunk size, the overlap,
or the splitter changes, which would silently invalidate the ground truth and
make a chunking change look like a retrieval regression. A span is stable: a
chunk is relevant if it contains the text that answers the question, however the
corpus happens to have been cut up that day.

`kind` records what each question is meant to stress, and the report breaks the
scores down by it. An aggregate number would show that hybrid helps without
showing *where*, and "where" is the part that generalises to someone else's
corpus:

    lexical   the answer hinges on a rare token the question repeats verbatim —
              a clause number, a certification, a figure. Dense retrieval has to
              represent that token in a 384-dimensional summary competing with
              every other word in the chunk; BM25 treats its rarity as the whole
              signal.

    semantic  the question and the answer share almost no vocabulary. BM25 has
              nothing to match on, and the embedding is doing the real work.

Several questions are deliberate near-misses of each other: four documents
mention a thirty-day period in four unrelated contexts, and two separate clauses
govern termination. Retrieving *something* about notice periods is not the same
as retrieving the right one, and a set without distractors would not tell the
difference.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Question:
    question: str
    # Verbatim from the corpus. A retrieved chunk counts as relevant if it
    # contains this, compared with whitespace collapsed and case ignored.
    gold_span: str
    kind: Literal["lexical", "semantic"]
    # Which document holds the answer. Not used for scoring — it is there so a
    # failure in the report can be read without going back to the corpus.
    doc_id: str


QUESTIONS: list[Question] = [
    # --- lexical: the question names a rare token the answer contains --------
    Question(
        question="What is the liability cap in Section 12.4?",
        gold_span="shall not exceed the total fees paid or payable by Customer in the twelve (12) months",
        kind="lexical",
        doc_id="msa",
    ),
    Question(
        question="Do you hold a SOC 2 Type II attestation?",
        gold_span="Provider maintains a SOC 2 Type II attestation",
        kind="lexical",
        doc_id="security",
    ),
    Question(
        question="What is the 99.95% availability commitment?",
        gold_span="available at least 99.95% of the time in each calendar month",
        kind="lexical",
        doc_id="sla",
    ),
    Question(
        question="Is data encrypted at rest with AES-256?",
        gold_span="at rest using AES-256",
        kind="lexical",
        doc_id="security",
    ),
    Question(
        question="What is a Permitted Sub-processor?",
        gold_span='"Permitted Sub-processor" means a third party approved in writing',
        kind="lexical",
        doc_id="msa",
    ),
    Question(
        question="What is the 72 hour breach notification requirement?",
        gold_span="within seventy-two (72) hours, after becoming aware of a Personal Data Breach",
        kind="lexical",
        doc_id="dpa",
    ),
    Question(
        question="Which TLS version is used in transit?",
        gold_span="encrypted in transit using TLS 1.3",
        kind="lexical",
        doc_id="security",
    ),
    Question(
        question="What is the recovery time objective?",
        gold_span="recovery time objective of four (4) hours",
        kind="lexical",
        doc_id="security",
    ),
    Question(
        question="What counts as an Excluded Claim?",
        gold_span='"Excluded Claims" means claims arising from indemnification obligations',
        kind="lexical",
        doc_id="msa",
    ),
    Question(
        question="What service credit applies below 95.0% availability?",
        gold_span="availability below 95.0% earns fifty percent (50%)",
        kind="lexical",
        doc_id="sla",
    ),
    # --- semantic: question and answer share almost no vocabulary -----------
    Question(
        question="How much warning must we give if we want to walk away from the contract early?",
        gold_span="terminate this Agreement for convenience upon ninety (90) days prior written notice",
        kind="semantic",
        doc_id="msa",
    ),
    Question(
        question="How long do staff have to file receipts to get their money back?",
        gold_span="Business expenses must be submitted within sixty (60) days",
        kind="semantic",
        doc_id="handbook",
    ),
    Question(
        question="Are people allowed to work from home?",
        gold_span="Employees may work remotely up to three (3) days per week",
        kind="semantic",
        doc_id="handbook",
    ),
    Question(
        question="If everything goes down, how fast will someone get back to us?",
        gold_span="acknowledge a Severity 1 incident within fifteen (15) minutes",
        kind="semantic",
        doc_id="sla",
    ),
    Question(
        question="How many holidays do employees get each year?",
        gold_span="accrue twenty-five (25) days of paid annual leave per year",
        kind="semantic",
        doc_id="handbook",
    ),
    Question(
        question="If the vendor breaks the deal badly, how long do they get to put it right?",
        gold_span="fails to cure that breach within thirty (30) days",
        kind="semantic",
        doc_id="msa",
    ),
    Question(
        question="What happens to our information once we stop being a customer?",
        gold_span="delete or return all Customer Personal Data within sixty (60) days",
        kind="semantic",
        doc_id="dpa",
    ),
    Question(
        question="How often does someone check who can get into the systems?",
        gold_span="Access rights are reviewed quarterly",
        kind="semantic",
        doc_id="security",
    ),
    Question(
        question="When would the platform normally be taken offline for work?",
        gold_span="on Sundays between 02:00 and 06:00 UTC",
        kind="semantic",
        doc_id="sla",
    ),
    Question(
        question="How quickly do bills have to be settled?",
        gold_span="pay all undisputed invoices within forty-five (45) days",
        kind="semantic",
        doc_id="msa",
    ),
]
