# AI201 Project 4 — Provenance Guard

Backend system that classifies submitted creative text as likely AI-generated, likely human-written, or uncertain — with confidence scoring, a transparency label, an appeals workflow, rate limiting, and a structured audit log.

See `[planning.md](planning.md)` for the full design spec this README summarizes and reflects on.

## Table of contents

- [Setup](#setup)
- [Dev tools](#dev-tools)
- [Architecture](#architecture)
- [Detection signals — reasoning](#detection-signals--reasoning)
- [Confidence scoring — reasoning](#confidence-scoring--reasoning)
- [Transparency labels](#transparency-labels)
- [Label variants — typed](#label-variants--typed)
- [Rate limiting](#rate-limiting)
- [Appeals workflow](#appeals-workflow)
- [Audit log](#audit-log)
- [Known limitations](#known-limitations)
- [Spec reflection](#spec-reflection)
- [AI usage](#ai-usage)
- [Portfolio walkthrough](#portfolio-walkthrough)



## Setup

Dependencies are managed with `[uv](https://docs.astral.sh/uv/)`. `pyproject.toml` + `uv.lock` are the source of truth; `requirements.txt` is kept only for anyone not using `uv`.

```bash
uv sync                       # creates/updates .venv with runtime + dev deps
cp .env.example .env          # then fill in GROQ_API_KEY
uv run app.py                 # runs the Flask app using the project's own venv
```

Always invoke project scripts with `uv run ...` (not a bare `python`/`python3`) so they use `.venv` rather than whatever Python happens to be on your `PATH`.

## Dev tools

```bash
uv run tools/calibrate.py           # tune signal thresholds against fixtures/
uv run tools/planning_diagrams.py   # regenerate diagrams/*.png
uv run tools/check_determinism.py   # measure Signal 3's run-to-run stability (real Groq calls)
```



## Architecture

```
Client
  |
  |  POST /submit  { text, author_id? }
  v
Flask app (app.py) --- rate-limited: 10/min, 100/day (flask-limiter)
  |
  |-- signals.len_variation_score(text)  --> len_var_ai_score  (0-1 or null)
  |-- signals.lexical_score(text)        --> lexical_ai_score  (0-1 or null)
  |
  |-- signals.combine_scores(...)        --> ai_likeness_score, disagreement, low_signal_confidence
  |-- signals.get_band(...)              --> band (5 keys, ordered guard -> disagreement -> mean rule)
  |-- labels.generate_label(...)         --> label (exact user-facing text, planning.md §3)
  |
  |-- llm_signal.detect_ai_heuristics(text)  --> llm_heuristic (advisory only, Groq call, never
  |                                               feeds ai_likeness_score/disagreement/label)
  |
  |-- submissions.save(...)              --> in-memory store, keyed by submission_id
  |-- audit_log.append_log(...)          --> audit_log.jsonl (one JSON object per event)
  v
JSON response: { submission_id, ai_likeness_score, disagreement, band, label,
                 signals: {...}, llm_heuristic: {...}, status, timestamp }
```

Appeals reuse the same store and log rather than a separate pipeline:

```
POST /appeal {content_id, creator_reasoning}
  -> submissions.file_appeal(...)   status: final -> under_review
  -> audit_log event: appeal_filed  (carries the original score/label alongside the appeal)

GET /appeals
  -> submissions.list_pending_appeals()   reviewer queue: text + scores + signal breakdown + reason

POST /appeals/<id>/resolve {decision, notes}
  -> submissions.resolve_appeal(...)   status: under_review -> resolved-upheld | resolved-overturned
  -> audit_log event: appeal_resolved
```

**Module boundaries, and why:**

- `signals.py` is pure functions, no Flask, no network calls — this is deliberate so Signals 1 & 2 (the two signals that actually decide the label) can be unit-tested and calibrated (`tools/calibrate.py`) completely independently of the web layer.
- `llm_signal.py` is isolated in its own module specifically *because* it needs a network call (Groq) and therefore has looser guarantees (timeouts, caching, graceful degradation) than `signals.py` promises. Keeping it separate means a Groq outage can never take down `/submit`'s core scoring path — see the guard/error handling described below.
- `labels.py` only maps a `band` string onto label copy; it never re-derives the banding decision. This keeps the *wording* people read and the *logic* that decides the band from drifting apart when one gets edited without the other.
- `submissions.py` / `audit_log.py` are the two pieces of state in an otherwise stateless scoring pipeline — deliberately factored out so the appeals workflow doesn't need to touch `app.py`'s route logic to add a new stored field.



## Detection signals — reasoning

Two signals decide the label; a third is advisory only. Full formulas are in `planning.md` [§1](planning.md#1-detection-signals) — this section is about *why* these two and *why this combination*, not the arithmetic.

**Why sentence-length variation (Signal 1)?** It's the cheapest possible proxy for "burstiness" — the literature's term for perplexity variance across sentences — without needing model access. Humans tend to mix short and long sentences without thinking about it; a lot of LLM output (especially from older/base models) lands in a narrower band of medium-length, evenly-paced sentences. It's a weak signal against a careful modern model that's been told to vary its rhythm, but it's free, deterministic, and it's a real property of the text, not a proxy for something unrelated.

**Why lexical diversity + stock phrases (Signal 2)?** MATTR (not raw TTR) gives a length-stable read on vocabulary richness, and the stock-phrase list targets a genuinely observable habit of LLM output: leaning on a recognizable set of hedges and transition words. We picked **two signals, not one**, specifically so a `disagreement` axis could exist at all — a single signal has no way to flag "this looks confident but the underlying evidence conflicts," which is the whole point of the Uncertain band.

**Why *not* fold the third (LLM-read) signal into the score.** Signal 3 (`llm_signal.py`) asks Groq to find specific rhetorical patterns — overconfident assertions, hedge-then-assert, rule-of-three lists — which requires actual reading comprehension, not string matching. That's a genuinely different, arguably stronger kind of evidence than Signals 1/2's surface statistics. We kept it **advisory-only** anyway, for two reasons: (1) it would break the "deterministic, no model calls" guarantee that makes the primary score reproducible and testable, and (2) an LLM judging "is this an overconfident assertion" is itself a heuristic with its own biases, not ground truth — routing its output through a human reviewer as extra context is more honest than quietly laundering a third model's opinion into a number that looks purely statistical.

**What we'd change deploying this for real:** recalibrate `CV_REF`/`MATTR_REF`/`PHRASE_REF` against a much larger, more diverse fixture set than our 21 hand-picked samples (see [Known limitations](#known-limitations) — our current calibration data already produced one counter-intuitive, load-bearing result). We'd also want a mechanism to refresh the stock-phrase list on a cadence, since it's explicitly a snapshot of *current* LLM tells that will drift as models change (we already had to add a second wave of phrases mid-project after a real Gemini sample slipped past the first list — see [AI usage](#ai-usage)).

## Confidence scoring — reasoning

**Why a weighted mean *and* a separate disagreement term**, instead of just averaging two signals and calling it a score: Signals 1 and 2 are correlated, not independent — both tend to move together for "formal/simple/uniform" writing, human or not. If a human writes in a formulaic register (a five-paragraph essay, legal boilerplate, non-native English), both signals often drop together, and a naive average would report a falsely confident-looking number for what is actually a case where two correlated, and possibly wrong, signals agree with each other for the same underlying reason. Tracking `disagreement = abs(len_var_ai_score - lexical_ai_score)` as its own axis means the system can distinguish "(0.5, 0.5) two signals genuinely landing in the middle" from "(0.1, 0.9) two signals that flatly contradict each other" — those are different situations that should not produce the same confidence, even though they can average to the same mean.

**Why the ordered banding rule** (guard fired → disagreement too high → mean bands), rather than simple threshold on the mean alone: a confident label (AI or human) should require the two signals to *agree*, not just average to an extreme. Given they share a bias, agreement is the honest bar for confidence, not a nice-to-have.

**Two real examples from Milestone 4 testing, taken directly from the fixture set**, showing the scoring is not a constant:


|                     | `fixtures/human/human_09_thoreau.txt` | `fixtures/ai/ai_01.txt`       |
| ------------------- | ------------------------------------- | ----------------------------- |
| `len_var_ai_score`  | 0.281                                 | 0.796                         |
| `lexical_ai_score`  | 0.029                                 | 0.400                         |
| `ai_likeness_score` | **0.155**                             | **0.598**                     |
| `disagreement`      | 0.253                                 | 0.396                         |
| `band`              | Likely Human-Written                  | Uncertain — middling evidence |


The Thoreau excerpt lands near the human extreme with low disagreement — a genuinely **high-confidence** case, both signals pointing the same way and doing so strongly. The Groq-generated sample lands in the middle with meaningfully higher disagreement — a **lower-confidence** case where the system is honestly hedging rather than forcing a verdict off ambiguous evidence. (Reproduce with `uv run tools/try_signals.py --file fixtures/human/human_09_thoreau.txt` / `--file fixtures/ai/ai_01.txt`.)

Worth stating plainly: across all 11 AI fixtures, **none currently reaches the "Likely AI-Generated" band** (highest score observed is 0.598) — see [Known limitations](#known-limitations) for what that reveals about the current calibration.

## Transparency labels

`/submit`'s `label` field is always one of four fixed texts (planning.md §3), generated by `[labels.py](labels.py)` from the confidence score/disagreement/guard flags computed in `[signals.py](signals.py)`. It is **not** the same string regardless of score — the exact text returned depends on which band the submission lands in:


| Band (`signals.get_band`, also returned as `band`)                  | Label text (`label`)                                                                |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `Likely AI-Generated` (score > 0.65, signals agree)                 | ⚠️ **Likely AI-Generated.** ... If you wrote this yourself, you can file an appeal. |
| `Likely Human-Written` (score < 0.35, signals agree)                | **Likely Human-Written.** ... Contestable via appeal.                               |
| `Uncertain — middling evidence` *or* `Uncertain — signals conflict` | ❔ **Uncertain Provenance.** ... Treat this as inconclusive.                         |
| `Uncertain — too short / weak signal` (a signal guard fired)        | ❔ **Uncertain — Not Enough Text.** ... No provenance estimate is offered.           |


`band` is the short internal key; `label` is the full user-facing text. Verified by submitting real fixtures at three different confidence levels and confirming three distinct label texts came back:

```bash
# fixtures/human/human_09_thoreau.txt -> ai_likeness_score 0.155 -> "Likely Human-Written"
# a <50-word snippet                  -> low_signal_confidence   -> "Uncertain — Not Enough Text"
# fixtures/ai/ai_01.txt               -> ai_likeness_score 0.598 -> "Uncertain Provenance" (middling band)
```

`tests/test_labels.py` unit-tests all four label variants directly against `generate_label()`, and `tests/test_app.py::TestSubmitResponseShape::test_transparency_label_varies_by_confidence_level` confirms the live `/submit` route returns different label text for inputs at different confidence levels.

## Label variants — typed

The three variants requested for this milestone (plus the fourth, "too short," text the system actually returns), copied verbatim from `[labels.py](labels.py)`:

**High-confidence AI** (`Likely AI-Generated`, `ai_likeness_score > 0.65` with signals agreeing):

> ⚠️ **Likely AI-Generated.** This text shows stylistic patterns often associated with AI-generated writing (uniform sentence rhythm, low vocabulary variation). This is an estimate from heuristic signals — not proof of authorship, and not a claim about *how* the text was produced. If you wrote this yourself, you can file an appeal.

**High-confidence human** (`Likely Human-Written`, `ai_likeness_score < 0.35` with signals agreeing):

> **Likely Human-Written.** This text shows stylistic patterns more typical of human writing (varied sentence rhythm, diverse vocabulary). This is an estimate based on heuristics — **not a verification of authorship**, and it does not rule out AI assistance. Contestable via appeal.

**Uncertain** (middling score, or signals disagree by more than 0.4 — both routes to the same displayed text):

> ❔ **Uncertain Provenance.** Our signals were mixed or contradicted each other, so this text can't be confidently classified as AI- or human-written. Treat this as inconclusive.

**Uncertain — not enough text** (a signal guard fired, e.g. fewer than 3 sentences or fewer than 50 words — technically a fourth string, distinct from generic "Uncertain," because "we don't have enough to go on" is a different claim than "we looked and it's ambiguous"):

> ❔ **Uncertain — Not Enough Text.** This submission was too short or too structurally sparse for the signals to analyze reliably. No provenance estimate is offered.

Every one of these texts is deliberately hedged — "estimate," "heuristic signals," "not proof" — because none of the underlying signals can actually verify authorship. The human-written label is worded to avoid reading like a credential (no "Verified Human" badge) so it can't be farmed as a laundering stamp for AI-assisted text that a human lightly edited.

## Rate limiting

`/submit` is rate-limited via [Flask-Limiter](https://flask-limiter.readthedocs.io/) at `10 per minute; 100 per day` (see `[app.py](app.py)`), using in-memory storage (`storage_uri="memory://"`) — fine for local dev/grading, not for a multi-process production deployment.

**Reasoning:**

- **10/minute** comfortably covers a real writer submitting a draft, tweaking it, and resubmitting a few times in one sitting, while still stopping a tight retry/flood loop within seconds.
- **100/day** covers realistic heavy daily use (many drafts/revisions across a working session) while bounding total load from any single client per day.

**Evidence** — the milestone's exact 12-request loop against a clean rate-limit window (first 10 succeed, then `429`s):

```bash
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5001/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test submission for rate limit testing purposes only.", "creator_id": "ratelimit-test"}'
done
```

```
200
200
200
200
200
200
200
200
200
200
429
429
```

`tests/test_app.py::TestRateLimit::test_exceeding_the_limit_returns_429` covers this automatically (with rate limiting re-enabled just for that test; the shared `client` fixture disables it elsewhere so unrelated tests aren't flaky).

## Appeals workflow

Planning.md §4's full appeals workflow (file → reviewer queue → resolve) is implemented across three routes, backed by an in-memory submission store (`[submissions.py](submissions.py)`) that is **not persisted across restarts** and has **no auth** — anyone holding a `content_id`/`appeal_id` can act on it. Both are documented, known limitations (matching the project's existing "no auth system in scope" stance), not solved here.

`POST /appeal` — file an appeal against any submission. Updates that submission's status to `under_review` in the store and logs the appeal alongside the original classification decision:

```bash
curl -s -X POST http://localhost:5001/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "PASTE-CONTENT-ID-HERE", "creator_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."}' | python -m json.tool
```

```json
{
    "appeal_id": "63c4d2b7-26c8-4dff-8800-4535576056f3",
    "content_id": "b3effe74-2feb-4241-beff-1d9b813277c5",
    "message": "Appeal received. A human reviewer will evaluate this submission.",
    "status": "under_review",
    "submission_id": "b3effe74-2feb-4241-beff-1d9b813277c5"
}
```

Verify with `GET /log` — the `appeal_filed` entry shows `"status": "under_review"` and a populated `"appeal_reasoning"`, alongside the original score/label it's appealing.

`GET /appeals` — reviewer-facing queue of everything currently `under_review`, with enough context to judge without re-running the pipeline (original text, label, scores, signal breakdown, appeal reason):

```bash
curl -s http://localhost:5001/appeals | python -m json.tool
```

`POST /appeals/<appeal_id>/resolve` — a human reviewer's final call:

```bash
curl -s -X POST http://localhost:5001/appeals/63c4d2b7-26c8-4dff-8800-4535576056f3/resolve \
  -H "Content-Type: application/json" \
  -d '{"decision": "overturn", "notes": "Reviewer agrees this is human-authored formal writing."}' | python -m json.tool
```

```json
{
    "appeal_id": "63c4d2b7-26c8-4dff-8800-4535576056f3",
    "content_id": "b3effe74-2feb-4241-beff-1d9b813277c5",
    "status": "resolved-overturned"
}
```

Resolving removes the appeal from `GET /appeals` and logs an `appeal_resolved` event. `GET /submissions/<content_id>` returns the full stored record at any point, so status changes are directly verifiable in storage, not just inferred from the log.

`tests/test_app.py::TestAppeal` covers the full lifecycle (file → appears in queue → resolve → disappears from queue → both events in the log) plus 400/404 edge cases.

## Audit log

Every `/submit`, `/appeal`, and `/appeals/<id>/resolve` call appends one structured JSON object (newline-delimited, `[audit_log.jsonl](audit_log.jsonl)`) via `[audit_log.py](audit_log.py)` — never a bare `print()`. Each entry has an `event` field (`submission` / `appeal_filed` / `appeal_resolved`) plus, for submissions: timestamp, `content_id`, `attribution` (Signal 1), both individual signal scores (`len_var_ai_score`, `lexical_ai_score`), the combined `ai_likeness_score`, `disagreement`, `low_signal_confidence`, the short `band` and full `label` text, the advisory `llm_heuristic` signal, and whether an appeal has been filed (`appeal_filed`).

Three real entries generated during manual verification (`GET /log`), showing a submission → appeal → resolution chain for the same `content_id`:

```json
{
  "event": "submission",
  "timestamp": "2026-07-01T01:51:12.315395+00:00",
  "content_id": "b3effe74-2feb-4241-beff-1d9b813277c5",
  "attribution": 0.2814423010816618,
  "len_var_ai_score": 0.2814423010816618,
  "lexical_ai_score": 0.0288360450563204,
  "ai_likeness_score": 0.1551391730689911,
  "disagreement": 0.25260625602534137,
  "low_signal_confidence": false,
  "band": "Likely Human-Written",
  "label": "**Likely Human-Written.** This text shows stylistic patterns more typical of human writing (varied sentence rhythm, diverse vocabulary). This is an estimate based on heuristics — **not a verification of authorship**, and it does not rule out AI assistance. Contestable via appeal.",
  "status": "final",
  "appeal_filed": false
}
```

```json
{
  "event": "appeal_filed",
  "timestamp": "2026-07-01T01:51:16.587073+00:00",
  "appeal_id": "63c4d2b7-26c8-4dff-8800-4535576056f3",
  "content_id": "b3effe74-2feb-4241-beff-1d9b813277c5",
  "status": "under_review",
  "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical.",
  "ai_likeness_score": 0.1551391730689911,
  "disagreement": 0.25260625602534137,
  "low_signal_confidence": false,
  "band": "Likely Human-Written",
  "label": "**Likely Human-Written.** This text shows stylistic patterns more typical of human writing (varied sentence rhythm, diverse vocabulary). This is an estimate based on heuristics — **not a verification of authorship**, and it does not rule out AI assistance. Contestable via appeal."
}
```

```json
{
  "event": "appeal_resolved",
  "timestamp": "2026-07-01T01:51:27.197180+00:00",
  "appeal_id": "63c4d2b7-26c8-4dff-8800-4535576056f3",
  "content_id": "b3effe74-2feb-4241-beff-1d9b813277c5",
  "decision": "overturn",
  "notes": "Reviewer agrees this is human-authored formal writing.",
  "status": "resolved-overturned"
}
```



## Known limitations

1. **This system will likely under-flag confident, fluent AI writing from current-generation models — a false-negative bias, not just generic "needs more data."** This isn't speculation; it fell out of our own calibration run (`tools/calibrate.py`, `signals.py` lines 18-24) against `fixtures/`. Signal 2's core hypothesis is "AI text has lower lexical diversity (MATTR) than human text" — but on our 11 AI fixtures (mostly Groq/Gemini output from current instruction-tuned models) vs. 10 human fixtures, **MATTR came out slightly *higher* for the AI class** (0.847 mean) **than the human class** (0.809 mean) — the opposite of the spec's original assumption. We raised `MATTR_REF` to sit near the pooled mean so this component doesn't actively mislead in the wrong direction, but that also means Signal 2's repetition component now contributes close to zero separating power for either class — nearly all of Signal 2's remaining discriminative power rides on the stock-phrase list, and all of the separation that does exist is currently carried by Signal 1 (sentence-length uniformity). The practical consequence, confirmed by [Confidence scoring](#confidence-scoring--reasoning): **none of our 11 AI fixtures reaches the "Likely AI-Generated" band** (best case 0.598, still "Uncertain"). Any AI-generated submission that also varies its sentence length reasonably well — which current instruction-tuned models can do when even mildly prompted to — will likely land in "Uncertain" rather than "Likely AI-Generated," regardless of how much more fixture data we collect, because the signal that would need to catch it (lexical diversity) has already been shown, on real samples, to point the wrong way for modern models. Fixing this needs a different or additional signal for "AI-fluent" writing, not just more of the same two.

Other, smaller limitations, briefly:

- **No persistence.** `submissions.py`'s store is in-memory; a restart loses every submission and appeal. Fine for a grading/demo scope, not for anything real.
- **No auth.** Anyone holding a `content_id`/`appeal_id` can act on it (documented in `planning.md` §4 as out of scope, not solved here).
- **Naive sentence splitting.** The regex split on `.!?` over-counts sentences on abbreviations, decimals, and ellipses, adding noise to Signal 1 specifically for text with a lot of those (technical/legal writing, dialogue with "...").
- **21-sample fixture set.** Every threshold in `signals.py` (`CV_REF`, `MATTR_REF`, `PHRASE_REF`, the band edges) was tuned against 10 human + 11 AI hand-picked samples — enough to catch the MATTR surprise above, not enough to trust the exact numbers at scale.



## Spec reflection

**Where the spec helped:** `planning.md`'s ordered banding rule — guard fired → disagreement too high → mean bands, checked *in that order* — is the single most load-bearing design decision in the project, and it came straight from the spec rather than something we arrived at independently. Without that ordering being spelled out up front, the natural first implementation is "average the two signals, threshold the average," which silently treats a genuinely uncertain (0.1, 0.9) case the same as a confident (0.5, 0.5) case. Having the spec state the priority order explicitly meant `signals.get_band()` (and its test suite, `tests/test_signals.py::TestGetBand::test_high_disagreement_beats_the_mean`) was built against a real design decision from day one instead of retrofitted after noticing the bug in testing.

**Where we diverged, and why:** the spec's starting formula for Signal 2 explicitly weighted MATTR at 60% specifically because it expected the stock-phrase list to be the weaker, more gameable half of the pair (`planning.md` §1, "MATTR gets 60% weight because the stock-phrase list is really a formality detector"). Once real calibration data came in (see [Known limitations](#known-limitations)), it turned out MATTR itself doesn't cleanly separate the two classes on our fixtures — the opposite of what motivated giving it the majority weight. We kept the 0.6/0.4 split as-is (changing it further would be over-fitting a positive-sounding correlation on 21 samples) but diverged from the spec's own planned mitigation: rather than trusting the em-dash-rate and contrastive-construction-rate sub-metrics in Signal 3 (which the spec anticipated as usable positive-weight discriminators), we excluded both from the scored formula entirely after calibration showed they ran *backwards* on our data (em-dashes appeared almost exclusively in a human sample; the "not X but Y" construction fired more on 19th-century formal essays than on AI output). The spec's own philosophy — "calibrate before trusting the numbers" — is what justified diverging from its own initial formula once the numbers came back different than expected.

## AI usage

Two concrete instances where an AI coding tool's first output was revised or overridden after checking it against real data, rather than accepted as-is:

1. **Stock-phrase list for Signal 2.** Directed the AI tool to generate a list of phrases and words commonly over-used by LLMs ("moreover," "furthermore," "in conclusion," "delve into," etc.) for `STOCK_PHRASES` in `signals.py`. It produced a reasonable GPT-3.5-era list, and it worked on our first batch of Groq-generated AI fixtures — but scored `lexical_ai_score = 0.0` on a real Gemini-generated sample we added later (`fixtures/ai/ai_11_gemini_user_provided.txt`), missing it completely. We overrode/extended the list with a second wave of terms actually observed in that sample and other modern-model output ("leverage," "robust," "holistic," "seamless," "tapestry," "underscore," "paramount," etc. — see the comment block in `signals.py` lines 42-72), and added `tests/test_signals.py::TestLexicalScore::test_modern_ai_cliches_are_detected` as a regression test so a future model generation drifting away from this exact wording gets caught rather than silently missed again.
2. **Signal 3's sub-metric weighting.** Directed the AI tool to implement `em_dash_rate` and `contrastive_construction_rate` ("not only X but Y") as scored sub-metrics of the advisory LLM heuristic signal, per the spec's original plan that these were plausible AI tells. It produced working regex-based counters and a combined formula that included both with positive weights. Running `tools/calibrate.py` against the fixture set showed both ran **backwards**: em-dashes appeared almost exclusively in a *human* sample (Melville), and the contrastive construction fired more on formal 19th-century human essays (Emerson, Thoreau) than on the AI samples. We overrode the AI's initial weighting by excluding both from the scored `llm_heuristic_score` formula — they're still computed and returned in `evidence` as free, inspectable context — rather than accept a plausible-sounding but empirically wrong scoring rule; re-calibrating without them raised the human/AI mean gap on `llm_heuristic_score` from 0.245 to 0.448.

In both cases the AI tool's first draft was directionally reasonable and cheap to generate, but wrong or incomplete against actual calibration evidence — the review step (running `tools/calibrate.py` against real fixtures, not just eyeballing the code) is what caught it, not a second look at the code itself.

## Portfolio walkthrough

A short (2-3 minute), unpolished screen recording covering:

1. A `/submit` call with a clearly human sample (e.g. `fixtures/human/human_09_thoreau.txt`) and a clearly-formulaic AI sample (e.g. `fixtures/ai/ai_01.txt`), showing the different `ai_likeness_score`/`band`/`label` each produces end-to-end.
2. A quick tour of the design decisions in [Detection signals](#detection-signals--reasoning) and [Confidence scoring](#confidence-scoring--reasoning) above — why two correlated signals need a separate disagreement axis, and why the third (LLM) signal stays advisory-only.
3. One appeal filed against a submission (`POST /appeal` → `GET /appeals` → `POST /appeals/<id>/resolve`), and the resulting `GET /log` entries showing the full submission → appeal → resolution chain for one `content_id`.

*(Recording link/embed to be added here once captured — see the evidence already documented above (rate-limit 429 sequence, audit-log sample, all four label variants, appeal lifecycle) for the detailed backup to what the walkthrough shows live.)*