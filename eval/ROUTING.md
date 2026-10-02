# Routing evaluation — the agent's one silent failure

The agentic graph's first live run against a real model refused a question the
corpus answered. This is the record of finding that, fixing it, and measuring
the fix.

```bash
python -m eval.routing_eval --trials 12
```

Unlike `retrieval_eval`, this one needs `OPENAI_API_KEY` and costs money —
`trials` calls per case on `AGENT_MODEL`. At the defaults (20 cases) that is 240 calls of a
few hundred tokens each.

## Why routing specifically

Every node in the graph except the router fails *open*. A dead grader keeps all
the chunks. A dead critic returns `None`, which the API renders as "unreviewed"
rather than "fine". A dead rewriter reuses the query it already had. In each
case the answer still gets produced, and a reader can tell that something was
skipped.

Routing is the exception. `needs_documents: false` ends the graph before
retrieval and returns a fixed apology. A wrong refusal is indistinguishable from
a right one — same response, no citations either way, nothing in the payload
saying "this decision might have been wrong." It is the one node that can lose
an answer without leaving a mark, which is why it is the one node with its own
eval.

## What was found

Against a one-page services agreement, with `AGENT_ENABLED=true`:

| question | agent off | agent on |
|---|---|---|
| What insurance must the Provider carry? | answered | answered |
| Who won the 2018 World Cup? | *(no chunks)* | refused — correct |
| How long do I get to pay a bill after they send it? | **answered** | **refused** |

The third row is a defect. Clause 5.2 of the document says fees are "payable
within thirty (30) days", and the single-pass pipeline answered it correctly.
The router's stated reason was *"about general billing practices and isn't
specific to any document."*

That reason is the bug. The old prompt asked whether a question was "about the
content, terms, obligations, dates, figures, or wording of documents" — and
almost no real question is, on its face. People ask about their situation, not
about the filing system. The prompt did say "when in doubt, say true", but the
model was not in doubt: it had confidently decided the question was about
billing in general.

## How badly

The first measurement asked each question once and scored 11/11, which was luck.
Repeated twelve times, the picture changed:

| | correct | note |
|---|---|---|
| original prompt | 181/192 (94.3%) | one case at **1/12** |
| revised prompt | 192/192 (100%) | no case below 12/12 |

Fifteen of sixteen cases were unanimous under both prompts. The whole of the
deficit was one question, wrong eleven times out of twelve — a systematic blind
spot rather than sampling noise. That distinction matters for the fix: a
near-deterministic wrong answer is a prompt problem, whereas a 7/12 case would
have pointed at the router running at the model's default temperature.

Both readings argue for repeating each case. One sample per question cannot tell
a 12/12 from a 1/12, and it is the low-scoring ones that make the feature
unshippable.

## The fix

The criterion changed from *"is this question about a document?"* to *"could a
library like this plausibly hold the answer?"*, with everyday phrasing named
explicitly as not being evidence against, and the false branch narrowed to
categories no document library could serve whatever it contained.

The first version of the revised prompt scored 192/192, but three of its
illustrative examples were near-duplicates of eval cases — teaching to the test.
They were replaced with topics the eval does not cover (leave entitlement,
equipment policy, risk of loss). The score held at **192/192**, with the
originally-failing question at 12/12 despite no longer appearing in the prompt.
That is the number worth trusting.

## What this does and doesn't establish

It establishes that the specific regression is fixed and that the fix survives
removing the answer from the prompt.

It does not establish that the router is now correct in general. Sixteen cases
is a small set, chosen by the same person who wrote the prompt, and the refusal
cases are all *easy* refusals — greetings, trivia, arithmetic, meta-questions —
because those are the only ones where refusing is unambiguously right. The
genuinely hard case for a router is a question that sounds in-domain but is not,
and there are none here. A corpus with real off-topic traffic would be a better
test than any list written from an armchair.

Two things also remain open:

- **Temperature was unpinned.** Fixed since: `LLM_TEMPERATURE` defaults to 0,
  and routing measured 120/120 at that setting (eval/ANSWERS.md, "Temperature 0").
- **The refusal is still silent.** Even a correct refusal returns no signal that
  a routing decision was made and could be revisited. Phase 4's human-review
  gate is where that becomes actionable.

## Second regression: technical policy questions

The caveat above was right about where the next failure would be. The phase-5
answer eval (`eval/answer_eval.py`) asked *"Which TLS version is used in
transit?"*, which the information security policy answers in one sentence. The
router refused it. Its reason:

> The question is about a technical specification regarding TLS versions, which
> is not governed by enterprise documents like contracts or policies.

This is the same failure as the first regression in a new place. The prompt's
list of subject matter was all legal and HR (terms, obligations, entitlements,
payments), so the model read a protocol question as general IT knowledge. Three
technical cases were added, plus a refusal on the other side of the line
(*"What does TLS stand for?"*), and measured before the prompt was touched:

| | correct | worst cases |
|---|---|---|
| prompt after the first fix | 217/240 (90.4%) | TLS version **0/12**, MFA **1/12** |
| + technical subjects named, with the "what this organisation does" test | 240/240 (100%) | none below 12/12 |

The fix names security policies and SLAs as part of the library, and draws the
line at *what this organisation does, uses, or commits to* (policy) versus *what
a technology is* (general knowledge). The boundary case still refuses 12/12, so
the router didn't simply learn to say yes to anything technical. The prompt's
examples (cloud region, patch timing) deliberately overlap none of the eval cases.

Both regressions were found by a different eval from the one that measures the
router. A routing eval can only test the categories its author thought of, so
the questions users actually ask, or an answer eval standing in for them, are
what find the next gap.

## Reproducing

```bash
python -m eval.routing_eval --trials 12     # ~240 calls on AGENT_MODEL
python -m eval.routing_eval --trials 1      # a cheap smoke test, proves nothing
```

Cases live in `eval/routing_eval.py`. Add your own before trusting the router on
your corpus: the questions that matter are the ones your users actually ask,
phrased the way they actually phrase them.
