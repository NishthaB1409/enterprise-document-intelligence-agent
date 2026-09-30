# Answer quality — does the agent earn its cost?

```bash
uv sync --extra eval
python -m eval.answer_eval --json eval/answer_results.json   # both pipelines, ~440 calls; rewrites the record
python -m eval.answer_eval --pipelines simple --gate          # what CI would run; writes nothing
```

The phase-3 graph (route, grade, rewrite, critique) costs about four model calls
per question against the single pass's one. Until phase 5, the only part of it
that had been measured was routing. This runner answers the question the rest of
the design was waiting on: **does the agentic graph produce better answers than
the single pass?**

**Short answer: not on this corpus.** Its answers are no more correct, it
declines answerable questions the single pass gets right, and it costs 4× the
calls and 2.7× the tokens. What it does buy is review signals: a critic
confidence, and detected contradictions that hold an answer for a human. Those
are worth having in some deployments, but they aren't answer quality.
`AGENT_ENABLED` stays `false` by default.

## Setup

- 20 golden questions (10 exact-token, 10 paraphrased) from `eval/questions.py`,
  plus 6 unanswerable ones, over the 13-document corpus (5 gold documents,
  8 distractors). 30 chunks, dense retrieval, top-5: the shipped configuration.
- Both pipelines are built by `app.services.build_pipeline`, the same code the
  service runs, and scored on identical questions over the same index.
- Answer model, agent model, and judge are all `gpt-4o-mini`.
- One sample per question per run, at the API's default temperature.

## Results

Two full runs of both pipelines, plus a third run of the single pass alone (the
first gate run, on a different machine). Run 1's correctness column used a broken
metric (see findings 2 and 3), and run 1's agent had the router bug (finding 1).
So the comparison rests on **run 2**, with runs 1 and 3 there to show
run-to-run variance.

| | simple, run 1 | simple, run 2 | simple, run 3 | agent, run 1 | agent, run 2 | floor | target |
|---|---|---|---|---|---|---|---|
| answered (golden) | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | | |
| faithfulness | 0.867 | 0.805 | 0.868 | 0.901 | 0.894 | 0.75 | 0.90 |
| answer relevancy | 0.767 | 0.766 | 0.808 | 0.713 | 0.698 | 0.65 | 0.85 |
| answer correctness | *broken* | 0.675 | 0.750 | *broken* | 0.725 | 0.60 | |
| context precision | 0.702 | 0.702 | 0.702 | 0.725 | 0.775 | 0.65 | 0.80 |
| context recall | 0.850 | 0.850 | 0.850 | 0.750 | 0.850 | 0.75 | 0.80 |
| chunks shown | 5.00 | 5.00 | 5.00 | 1.50 | 1.65 | | |
| held for review | 0.000 | 0.000 | 0.000 | 0.077 | 0.154 | | |
| unanswerable declined | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | | |
| unanswerable released | 0 | 0 | 0 | 0 | 0 | | |
| model calls / question | 1.00 | 1.00 | 1.00 | 3.88 | 4.00 | | |
| tokens / question | 1222 | 1220 | 1221 | 3030 | 3282 | | |

Per-question rows for run 2, both pipelines, are in
[`answer_results.json`](answer_results.json). Run 3 was the first `--gate` run,
and it overwrote that file in the working copy, because the runner wrote results
on every run. It now writes only when given `--json`, so a gate run leaves the
record alone.

### Reading it

With 20 questions, one question is 0.05 on any per-question average. The
single pass's faithfulness moved by 0.06 across three runs of identical code,
its relevancy by 0.04, and its correctness by 0.075. So:

- **Correctness: no measured difference.** 0.725 against 0.675 is one question,
  and the single pass's own third run (0.750) scored above the agent.
- **Faithfulness: the agent is higher in both runs** (by 0.03 and 0.09). This is
  plausibly real, since an answer written from 1.6 chunks has less to
  misattribute than one written from 5. But it's inside the noise of a strict
  judge, so read it as "possibly slightly better".
- **Relevancy: the agent is lower in both runs** (by 0.05 and 0.07). Its two
  wrong declines score zero here, and its conflict-hedging answers ("the sources
  disagree…") read as less direct.
- **Unanswerable questions: no difference.** The single pass already declines
  all six near-miss questions. Routing and grading add nothing to that.
- **Cost: a consistent 4× calls, 2.7× tokens.**

The agent's one clear advantage is not in this table's quality rows. It held 15%
of its answers for review, including both questions where the corpus
deliberately contains a superseded 2019 policy that conflicts with the current
one. The single pass reported those conflicts in its prose on some runs and not
others, and was never held, because it has no critic to flag them.

## Findings

### 1. The router refused a security-policy question, 12 times out of 12

*"Which TLS version is used in transit?"* was routed away as *"a technical
specification… not governed by enterprise documents"*. The routing eval had
never asked a technical question. It measured 0/12 on that case and 1/12 on an MFA
question, then 12/12 on both after the prompt was fixed. The details are in
[`ROUTING.md`](ROUTING.md#second-regression-technical-policy-questions). This
was the whole of run 1's recall gap: run 2 has both pipelines at 0.850.

### 2. Ragas `AnswerAccuracy` can't grade against a fragment

It rated every correct answer 0.5. It expects a full reference answer, and our
gold spans are sentence fragments ("shall not exceed the total fees paid…"), so
the judge rated complete answers as partial matches.

### 3. Ragas `FactualCorrectness(mode="recall")` punishes extra detail

This was the replacement, and it was wrong too. *"Up to three (3) days per week
with manager approval"* scored **0.0** against *"up to three (3) days per week"*.
In recall mode its numerator is still the number of *answer* claims the
reference supports, so one extra true detail inside a claim makes the claim
unsupported and zeroes the score. An early sanity check had used only short
answers, which is why it passed.

`ReferenceRecall` in `answer_eval.py` scores in one direction only: break the
gold span into claims, then check each claim against the answer. It uses half the
calls. It was validated on the exact answers that broke the other two, twice
each: correct answers with extra detail score 1.0, and a wrong figure, a wrong
document and a decline score 0.0.

### 4. The grader drops gold passages the single pass would have used

In run 1, for *"What happens to our information once we stop being a
customer?"*, the grader discarded the DPA's 60-day deletion clause and kept two
retention-policy chunks. The answer was faithful to what it kept, and wrong. The
grader's "be strict" instruction trades recall for precision, and on a corpus
where dense top-5 already finds the gold passage 85% of the time, that trade
costs more than it buys.

### 5. The agent declines when it has only the exact passage

*"How many holidays do employees get each year?"* was declined by the agent in
both runs. Grading correctly kept only the handbook chunk ("twenty-five (25)
days of paid annual leave"). Given one chunk that says "annual leave" and never
"holidays", the generator marked the question unanswerable, while still quoting
the 25 days in its prose. The single pass, given five chunks, answered it both
times. It is reproducible (2/2), but the cause is a guess. The generator's
vocabulary bridging may simply be weaker with less context around the passage.

### 6. Faithfulness is strict about attribution

The single pass's recovery-time answer attributed a figure to "the 2023
information security policy". No source says 2023, so the model invented the
year, and faithfulness 0.0 is fair. "The current policy" (an inference, not
stated) was scored just as harshly. Single-question faithfulness is noisy, so
read the averages, not individual rows.

## The CI gate

`--gate` exits non-zero if a pipeline falls below any **floor** in
`THRESHOLDS`. The floors sit below run 2 by about the run-to-run noise seen
between the two runs. They are not the README's targets. The shipped single pass
misses three of those four (faithfulness, relevancy, precision), and a gate that
fails every build gets switched off rather than fixed. Raise a floor when a
change measurably lifts the baseline.

No CI workflow runs it yet. It needs `OPENAI_API_KEY` as a repository secret and
costs roughly $0.05 per run for the single pass alone, so it suits a nightly job
or a manual trigger better than every push.

## What this can't tell you

- **26 questions, one sample each, one judge.** Differences of a question or two
  are noise. The judge is the same model family as the generator, which is known
  to flatter it; `--judge-model` exists to check the ranking holds under another.
- **Synthetic corpus.** Written to contain hard cases (distractors, a superseded
  policy), so it's harder than some real corpora and easier than others.
- **Temperature is unpinned.** Every call runs at the API default, which is part
  of the run-to-run variance above. Pinning it is cheap and would tighten every
  number here. It's a product change, so it's left for its own measured commit.
- **The agent may earn its keep elsewhere.** On a corpus where dense retrieval
  misses more, grading and rewriting have more to fix. Re-run this against your
  own documents before deciding either way.
