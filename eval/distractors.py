"""Plausible wrong answers, generated so the golden set has to discriminate.

The five hand-written documents in `corpus.py` hold every answer the question
set asks for. On their own they came to ten chunks, which makes retrieving the
top five an act of returning half the corpus — at that size every configuration
scores near the ceiling, the differences between them are one or two questions
wide, and the measurement cannot support a claim in either direction.

So the corpus gets a second half: vendor agreements, policies, and runbooks that
are *about the same things* — notice periods, termination, encryption, uptime,
data deletion — with different parties, different numbers, and different clause
text. They are what a real corpus looks like around the document you actually
want, and they are the reason retrieval is hard at all. A question about a
ninety-day termination notice now has to beat eleven other documents that also
discuss termination notice, at thirty, sixty, and one hundred and eighty days.

Two rules make these safe to generate rather than write:

    Nothing here may contain a gold span. `tests/test_eval_corpus.py` asserts
    it, so a distractor that accidentally answers a question fails the build
    instead of silently corrupting the ground truth.

    Generation is deterministic — fixed lists, no RNG. The corpus has to be the
    same on every machine, or the numbers in `eval/README.md` are unreproducible.

They are deliberately not *only* near-duplicates of the gold documents. A corpus
of twelve minor variants of one contract would overstate BM25's value, because
exact-token matching is unusually decisive when everything else is identical.
The mix here is closer to what an enterprise index actually holds.
"""

from eval.corpus import Document, _document

# Parties, notice periods, and figures that deliberately collide with the gold
# documents' vocabulary while never reproducing their spans.
_VENDORS = [
    ("Northwind Logistics", 30, "99.5%", "60"),
    ("Globex Analytics", 60, "99.9%", "45"),
    ("Initech Payments", 180, "99.0%", "90"),
    ("Umbra Consulting", 45, "98.5%", "30"),
]


def _vendor_agreement(index: int, vendor: str, notice: int, uptime: str, cure: str) -> Document:
    return _document(
        f"vendor-{index}",
        f"{vendor.lower().replace(' ', '-')}-agreement.pdf",
        [
            f"""SERVICES AGREEMENT — {vendor.upper()}

Section 1. Engagement. {vendor} shall provide the professional services set out
in each Statement of Work agreed between the parties.

Section 2. Term and Renewal. This agreement runs for an initial term of thirty-six
(36) months and renews for successive six (6) month periods.

Section 3. Ending the Engagement. Either party may end this agreement for
convenience on {notice} days written notice to the other. Any Statement of Work
in progress at that date continues until completed unless the parties agree
otherwise in writing.""",
            f"""Section 7. Remedy Period. Where a party is in material default, the other
party shall give written particulars of the default and allow {cure} days for it
to be remedied before exercising any right to end the agreement.

Section 8. Service Availability. {vendor} targets availability of {uptime}
measured across each quarter. Availability is calculated excluding planned
downtime notified in advance.

Section 9. Charges. Invoices are payable thirty (30) days from receipt. Disputed
amounts must be raised within ten (10) business days of the invoice date.""",
            f"""Section 14. Liability. The aggregate liability of either party under this
agreement is limited to the charges paid in the six (6) months before the event
giving rise to the claim. Nothing limits liability for death, personal injury,
or fraud.

Section 15. Data. {vendor} shall return or destroy client data within ninety (90)
days of the end of the agreement and shall confirm destruction in writing on
request.

Section 16. Insurance. {vendor} shall maintain professional indemnity insurance
of not less than five million dollars ($5,000,000) for the duration of the
engagement.""",
        ],
    )


_LEGACY_SECURITY = _document(
    "security-2019",
    "information-security-policy-2019-superseded.pdf",
    [
        """INFORMATION SECURITY POLICY (2019 — SUPERSEDED)

1. Certification. The company holds an ISO/IEC 27001 certificate covering the
production environment. A SOC 2 Type I report was issued in 2018 and has not
been refreshed.

2. Encryption. Data in transit is protected using TLS 1.2. Data at rest is
protected using AES-128 on database volumes. Key rotation occurs every
twenty-four (24) months.

3. Access. Access to production is granted on request by a team lead. Access
reviews are performed annually as part of the audit cycle.""",
        """4. Patching. Security patches are applied within thirty (30) days for
critical issues and ninety (90) days otherwise. Penetration testing is performed
every two (2) years by a third party.

5. Continuity. Backups are taken nightly. The documented recovery point
objective is twenty-four (24) hours and the recovery time objective is
forty-eight (48) hours.

6. Training. Security training is delivered at induction. Refresher training is
not currently mandatory.""",
    ],
)

_SUPPORT_RUNBOOK = _document(
    "runbook",
    "support-escalation-runbook.pdf",
    [
        """SUPPORT ESCALATION RUNBOOK

Triage. The on-call engineer acknowledges every page and decides a severity
within ten (10) minutes. If the severity is unclear, treat it as the higher of
the two options until proven otherwise.

Escalation. A Priority A incident is escalated to the engineering manager after
thirty (30) minutes without a mitigation, and to the director after sixty (60)
minutes. Customer communications are sent every hour until resolution.

Maintenance. Routine maintenance is scheduled on Tuesdays between 01:00 and
03:00 local time and is announced on the status page seven (7) days ahead.""",
        """Postmortems. Every Priority A incident gets a written postmortem within five
(5) business days. Postmortems are blameless and are circulated to the whole
engineering organisation.

Paging. Engineers are paged through the on-call rota. An unacknowledged page
escalates to the secondary after five (5) minutes and to the whole team after
ten (10) minutes.

Status page. Incidents affecting more than five percent (5%) of requests are
posted publicly within fifteen (15) minutes of confirmation.""",
    ],
)

_PRIVACY_NOTICE = _document(
    "privacy",
    "customer-privacy-notice.pdf",
    [
        """CUSTOMER PRIVACY NOTICE

What we collect. We collect account details, usage data, and support
correspondence. We do not sell personal information to third parties.

Retention. Account records are retained for seven (7) years after the account
closes, in line with our statutory obligations. Support correspondence is
retained for twenty-four (24) months.

Your rights. You may request access to, correction of, or erasure of your
personal information by contacting the privacy team. We respond to requests
within one (1) month.""",
        """International transfers. Where we transfer personal information outside your
region we rely on approved transfer mechanisms and publish the current list of
recipients on our website.

Cookies. We use strictly necessary cookies by default. Analytics cookies are set
only where consent has been given and can be withdrawn at any time.

Contact. Questions about this notice should be sent to the privacy team, who
will respond within ten (10) business days.""",
    ],
)

_PROCUREMENT = _document(
    "procurement",
    "procurement-standard.pdf",
    [
        """PROCUREMENT STANDARD

Thresholds. Purchases up to ten thousand dollars ($10,000) may be approved by a
department head. Purchases above that require finance approval, and purchases
above one hundred thousand dollars ($100,000) require board approval.

Vendor due diligence. New vendors handling company data must complete a security
questionnaire and provide evidence of an independent audit before contracts are
signed. Evidence is refreshed every two (2) years.

Notice periods. Vendor contracts should be reviewed ninety (90) days before
their renewal date so that notice of non-renewal can be given in time.""",
        """Expenses. Travel booked through the corporate tool does not require
pre-approval. Travel booked outside the tool requires written approval from a
department head before the booking is made.

Payment terms. The company's standard payment terms are sixty (60) days from
invoice. Shorter terms require finance approval and are agreed case by case.

Records. Signed contracts are filed in the contract repository within five (5)
business days of execution.""",
    ],
)


DISTRACTORS: list[Document] = [
    _vendor_agreement(index, *vendor) for index, vendor in enumerate(_VENDORS, start=1)
] + [_LEGACY_SECURITY, _SUPPORT_RUNBOOK, _PRIVACY_NOTICE, _PROCUREMENT]
