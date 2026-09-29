# Review gate evaluation — contradiction detection

```bash
python -m eval.review_eval --trials 6
```

Needs `OPENAI_API_KEY` and costs money: `trials` critique calls per case, plus
`trials` grading calls per conflict case, on `AGENT_MODEL`. At the defaults that
is 108 calls of a few hundred tokens each.

**Status: the full runner has not been run yet.** The one measurement so far is
the grader finding below. Record the first full run's output here, including
anything that comes back below 100%.

## First finding: grading hid the conflict

The first live test used a two-page PDF. Page 1 said invoices were payable in
thirty days, and page 2 (a schedule) said forty-five. Nothing was held. The
response showed `grade:1/3` and `critique:1.00`, meaning the critic had been
given a single source and correctly found nothing wrong with it.

Running the grader alone on those two pages showed why:

| grader prompt | kept both sides |
|---|---|
| original | 3/5 |
| + "judge each source against the question, never against the other sources" | 12/12 |

The dropped side's stated reason was *"it contradicts the other source regarding
the specific payment terms"*. The grader's "be strict" instruction had turned it
into a tie-breaker: it chose one version, and the reader never learned there
were two. Grading runs before the critic, so no critic prompt could have fixed
this. That is why the runner now checks grading on every conflict case, in the
`graded` column.

The caveat is the same as everywhere else here: one case and seventeen samples.
It shows the failure was real and that the fix addresses it. It does not show the
grader never does this.

## What is being tested

Most of the review gate needs no evaluation. Its rules are deterministic and are
pinned in `tests/test_review.py`: an uncited claim holds the answer, a dead
critic holds it, and an honest "not covered" does not. Two parts depend on a
model's judgement, and those are what this runner measures:

1. **Does the critic notice sources that disagree?** Contradiction detection is
   an extra instruction in the critique prompt, not a call of its own. That
   makes it free per query, but it shares one prompt on the cheaper model with
   three other checks. Whether it still works there has to be measured.
2. **What confidence does the critic actually give?** `REVIEW_MIN_CONFIDENCE=0.7`
   is a guess. The runner prints the confidences, and the threshold should be
   set from those.

## Why the negatives matter as much as the positives

There are two failures, with different costs:

| failure | what happens | cost |
|---|---|---|
| missed conflict | the answer goes out citing one side of a disagreement | the failure the feature exists to prevent |
| false alarm | a consistent answer is held | a reviewer's time, and a queue that teaches reviewers to approve without reading |

The six conflict cases are one term stated twice, differently. The realistic
shape is a main agreement against a schedule or order form, or an old policy
against a new one. The six clean cases are **hard negatives**, built so that a
detector which fires on "two different numbers" fails them:

- figures in the same unit for *different* terms (a 30-day payment term beside a
  90-day notice period; £2m of indemnity cover beside £5m of public liability
  cover)
- a term plus an exception or a refinement of it (a liability cap and its
  carve-out for personal injury; full-time leave and a pro-rata rule for
  part-time staff)

Detections are scored after `resolve_conflicts`, exactly as the gate sees them.
A conflict that names only one real source is dropped there and does not count.

## What a result would and would not show

- Twelve hand-written cases, each with two short sources, is a smoke test of the
  prompt, not a benchmark. A clean score says the instruction works on clear
  cases. It says nothing about conflicts spread across long chunks, or across
  more than two sources.
- The cases are fixed and visible. Tuning the prompt against them is teaching to
  the test, the same trap noted in [`ROUTING.md`](ROUTING.md). If the prompt
  changes because of this runner, add cases the change was not written against.
- It does not measure the end-to-end rate at which answers get held on real
  traffic. The trace records `review_reasons` for every query, even with
  `REVIEW_ENABLED=false`, and that is the place to measure it.
