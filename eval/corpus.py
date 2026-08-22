"""A small enterprise corpus, held in the repo so the numbers are reproducible.

Five documents of the kind this service is aimed at: a services agreement, a
data processing addendum, a security policy, an SLA, and an employee handbook.
Written rather than sampled, for two reasons — real customer contracts cannot be
committed to a repo, and a synthetic corpus lets the hard cases be put there
deliberately.

What makes it a useful test of *hybrid* retrieval specifically is the presence
of tokens that carry meaning to a reader and almost none to an embedding model:
clause numbers (Section 12.4), certification names (SOC 2 Type II), figures
(99.95%), and defined terms (Permitted Sub-processor). Those are where dense
retrieval degrades and BM25 does not. The corpus also repeats itself on purpose
— four of the five documents mention a thirty-day notice period, in four
unrelated contexts — so that precision has something to go wrong on.

The corpus is small. A larger one would give tighter confidence intervals; it
would also make it impossible to read the thing being measured, which at this
stage matters more. The numbers in `eval/README.md` are quoted with that limit
stated rather than rounded off.
"""

from dataclasses import dataclass

from app.ingest.parser import Page, normalize


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    source: str
    pages: list[Page]


def _document(doc_id: str, source: str, pages: list[str]) -> Document:
    # Through `normalize` so this text matches what the PDF path would produce,
    # and so the gold spans in `questions.py` are matched against the same
    # string the store ends up holding.
    return Document(
        doc_id=doc_id,
        source=source,
        pages=[Page(number=i, text=normalize(p)) for i, p in enumerate(pages, start=1)],
    )


MSA = _document(
    "msa",
    "master-services-agreement.pdf",
    [
        """MASTER SERVICES AGREEMENT

Section 1. Definitions. "Services" means the software-as-a-service offering
described in each Order Form. "Confidential Information" means non-public
information disclosed by either party. "Permitted Sub-processor" means a third
party approved in writing by Customer to process Customer Data.

Section 2. Term. This Agreement begins on the Effective Date and continues for
an initial period of twenty-four (24) months, renewing automatically for
successive twelve (12) month periods unless either party gives notice of
non-renewal.

Section 3. Termination for Convenience. Either party may terminate this
Agreement for convenience upon ninety (90) days prior written notice to the
other party. Termination for convenience does not relieve Customer of fees
accrued before the effective date of termination.""",
        """Section 4. Termination for Cause. Either party may terminate this Agreement
immediately upon written notice if the other party commits a material breach and
fails to cure that breach within thirty (30) days of receiving notice of it.

Section 5. Fees and Payment. Customer shall pay all undisputed invoices within
forty-five (45) days of the invoice date. Late amounts accrue interest at one
and one-half percent (1.5%) per month.

Section 11. Warranties. Provider warrants that the Services will perform
materially in accordance with the Documentation. Provider does not warrant that
the Services will be uninterrupted or error-free.""",
        """Section 12. Limitation of Liability.

Section 12.1. Neither party shall be liable for indirect, incidental, special,
consequential, or punitive damages, or for lost profits or lost revenue.

Section 12.4. Except for the Excluded Claims, each party's total aggregate
liability arising out of this Agreement shall not exceed the total fees paid or
payable by Customer in the twelve (12) months immediately preceding the event
giving rise to the claim.

Section 12.5. "Excluded Claims" means claims arising from indemnification
obligations, breach of confidentiality, or a party's gross negligence or wilful
misconduct. The cap in Section 12.4 does not apply to Excluded Claims.

Section 13. Indemnification. Provider shall defend Customer against any claim
that the Services infringe a third party's intellectual property rights, and
shall pay any damages finally awarded against Customer for that claim.""",
    ],
)

DPA = _document(
    "dpa",
    "data-processing-addendum.pdf",
    [
        """DATA PROCESSING ADDENDUM

1. Roles. For the purposes of the General Data Protection Regulation, Customer
is the Controller and Provider is the Processor of Customer Personal Data.

2. Sub-processors. Provider may engage a Permitted Sub-processor provided that
Provider gives Customer at least thirty (30) days notice before that
sub-processor begins processing, and Customer does not reasonably object within
that period. A current list of sub-processors is maintained on the trust portal.

3. International Transfers. Where Customer Personal Data is transferred outside
the European Economic Area, the parties shall rely on the Standard Contractual
Clauses adopted by the European Commission.""",
        """4. Personal Data Breach. Provider shall notify Customer without undue delay,
and in any event within seventy-two (72) hours, after becoming aware of a
Personal Data Breach affecting Customer Personal Data. The notification shall
describe the nature of the breach, the categories and approximate number of data
subjects concerned, and the measures taken to address it.

5. Deletion and Return. Upon termination of the Agreement, Provider shall delete
or return all Customer Personal Data within sixty (60) days, except where
retention is required by applicable law.

6. Audit. Customer may audit Provider's compliance with this Addendum no more
than once per calendar year, on at least thirty (30) days written notice.""",
    ],
)

SECURITY = _document(
    "security",
    "information-security-policy.pdf",
    [
        """INFORMATION SECURITY POLICY

1. Certification. Provider maintains a SOC 2 Type II attestation covering the
Security, Availability, and Confidentiality trust services criteria. The report
is refreshed annually and made available to Customer under NDA.

2. Encryption. Customer Data is encrypted in transit using TLS 1.3 and at rest
using AES-256. Encryption keys are managed in a hardware security module and
rotated every twelve (12) months.

3. Access Control. Access to production systems follows the principle of least
privilege. Access rights are reviewed quarterly, and privileged access requires
multi-factor authentication and approval by the security team.""",
        """4. Vulnerability Management. Critical vulnerabilities are remediated within
seven (7) days of discovery; high-severity vulnerabilities within thirty (30)
days. Provider engages an independent firm to conduct penetration testing at
least annually.

5. Business Continuity. Production data is backed up continuously with a
recovery point objective of one (1) hour and a recovery time objective of four
(4) hours. Disaster recovery procedures are tested twice per year.

6. Personnel. All personnel complete security awareness training on hire and
annually thereafter. Background checks are performed where permitted by law.""",
    ],
)

SLA = _document(
    "sla",
    "service-level-agreement.pdf",
    [
        """SERVICE LEVEL AGREEMENT

1. Availability Commitment. Provider will make the Services available at least
99.95% of the time in each calendar month, excluding Scheduled Maintenance.

2. Service Credits. If monthly availability falls below the commitment, Customer
is entitled to a service credit against the following month's fees: availability
below 99.95% but at or above 99.0% earns a credit of ten percent (10%);
availability below 99.0% but at or above 95.0% earns twenty-five percent (25%);
availability below 95.0% earns fifty percent (50%).

3. Claiming Credits. Customer must request a service credit within thirty (30)
days of the end of the affected month. Service credits are the sole and
exclusive remedy for a failure to meet the Availability Commitment.""",
        """4. Support Severity Levels. A Severity 1 incident is a complete loss of
production service. A Severity 2 incident is a significant degradation affecting
a majority of users. A Severity 3 incident is a minor issue with a workaround
available.

5. Response Targets. Provider will acknowledge a Severity 1 incident within
fifteen (15) minutes, a Severity 2 incident within two (2) hours, and a Severity
3 incident within one (1) business day. Severity 1 incidents receive continuous
effort until resolved.

6. Scheduled Maintenance. Provider may perform Scheduled Maintenance during a
published window on Sundays between 02:00 and 06:00 UTC, with at least five (5)
days advance notice.""",
    ],
)

HANDBOOK = _document(
    "handbook",
    "employee-handbook.pdf",
    [
        """EMPLOYEE HANDBOOK

Paid Time Off. Full-time employees accrue twenty-five (25) days of paid annual
leave per year, accruing monthly. Up to five (5) unused days may be carried into
the following calendar year; anything above that is forfeited.

Remote Work. Employees may work remotely up to three (3) days per week with
manager approval. Fully remote arrangements require approval from a department
head and are reviewed annually.

Expenses. Business expenses must be submitted within sixty (60) days of being
incurred. Expenses over five hundred dollars ($500) require prior approval.""",
    ],
)

# The documents the golden set asks about. Every gold span lives in one of these.
GOLD_DOCUMENTS: list[Document] = [MSA, DPA, SECURITY, SLA, HANDBOOK]


def _corpus() -> list[Document]:
    # Imported here rather than at module scope: `distractors` imports `_document`
    # from this module, and a top-level import would be circular.
    from eval.distractors import DISTRACTORS

    return GOLD_DOCUMENTS + DISTRACTORS


# What actually gets indexed. The distractors carry no answers — they are there
# so that finding one is a mistake the metrics can see. See `eval/distractors.py`.
CORPUS: list[Document] = _corpus()
