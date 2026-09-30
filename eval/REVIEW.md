# Review gate evaluation — contradiction detection

```bash
python -m eval.review_eval --trials 6
```

Needs `OPENAI_API_KEY` and costs money: `trials` critique calls per case, plus
`trials` grading calls per conflict case, on `AGENT_MODEL`. At the defaults that
is 108 calls of a few hundred tokens each.

**Status: measured.** First full run on 2026-09-30, `gpt-4o-mini`, 6 trials per case:

```
      flagged  expect    conf (min/median)   graded  case
     6/6       conflict          0.30/0.40      6/6  payment term
     6/6       conflict          0.20/0.25      6/6  governing law
     6/6       conflict          0.20/0.30      6/6  leave entitlement
     6/6       conflict          0.20/0.20      6/6  commencement date
     6/6       conflict          0.20/0.30      6/6  insurance level
  !  0/6       conflict          0.20/0.40      6/6  liability cap
     0/6       none              0.90/1.00        -  payment vs notice
     0/6       none              0.80/0.90        -  cap and carve-out
     0/6       none              0.90/1.00        -  leave pro rata
     0/6       none              1.00/1.00        -  law and jurisdiction
     0/6       none              0.90/0.90        -  two insurance types
     0/6       none              0.90/0.90        -  rate and deadline

conflicts detected: 30/36 (83.3%)
false alarms:       0/36 (0.0%)
grader kept both sides of a conflict: 36/36 (100.0%)
critic confidence on conflict cases: median 0.30, min 0.20
critic confidence on clean cases: median 0.90, min 0.80
```

What it shows:

- **No false alarms, including on all six hard negatives.** The critic
  distinguishes "two different figures" from "two figures for the same term",
  which the prompt was written to teach. That's the result that matters most for
  a review queue.
- **Every case is unanimous.** Five conflicts were caught 6/6, and one was missed
  6/6. That's a systematic blind spot, not sampling noise.
- **The miss is still held.** On the liability cap case the critic listed no
  conflict but gave confidence 0.20–0.40, well under the 0.7 threshold, so the
  gate holds the answer anyway, for a different stated reason. The case is also
  the most arguable of the six: its second clause opens *"Notwithstanding anything
  else…"*, which a reader can take as overriding the first rather than
  contradicting it. It is left as written rather than softened to make the score
  100%.
- **The 0.7 threshold now has evidence behind it.** Conflict cases never scored
  above 0.40, and clean cases never below 0.80. The threshold sits in a gap 0.40
  wide. It started as a guess, and on these cases it separates perfectly. That's
  12 cases, so it's a sanity check rather than a calibration, but a threshold
  anywhere from 0.45 to 0.75 would have held the same answers.
- **The grader fix held.** Both sides of every conflict survived grading, 36/36,
  against 3/5 on the single case measured before the fix.

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
