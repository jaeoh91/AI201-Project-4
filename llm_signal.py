"""Advisory Signal 3 — LLM-detected rhetorical AI heuristics.

Unlike signals.py (deterministic, zero network calls), this module makes a
single Groq call per submission to extract structured evidence of rhetorical
patterns that are hard to enumerate reliably via regex: overconfident
assertions of speculative claims, the "hedge, then confidently assert"
rhetorical pattern, and rule-of-three/parallelism. Two more sub-metrics
(em-dash rate, "not X but Y" contrastive-construction rate) ARE reliably
regex-countable and are computed directly here, with no LLM involved — but
see the note on W_EM_DASH/W_CONTRASTIVE below for why they're reported, not
scored.

Groq returns each judgment category as a **list of verbatim quote spans**
(not a bare integer) — this makes the evidence auditable by a human reviewer
(what exactly got flagged, not just how many things), and the count used in
the score formula is simply the validated list's length.

Advisory only — see planning.md §1 Signal 3. This never feeds into
combine_scores/get_band/ai_likeness_score; it is surfaced in /submit's
response and the audit log purely as extra context (e.g. a human reviewer's
tiebreaker), never as a scoring input. Never raises: any failure (missing
key, network error, timeout, malformed response) degrades to score=None + a
note, so /submit can never fail or hang because of this optional signal.
"""

import functools
import json
import os
import re

from dotenv import load_dotenv

# app.py never loads .env itself; without this, a GROQ_API_KEY defined only
# in .env (not exported in the shell) would be invisible to os.environ and
# this signal would always skip in real usage. Idempotent if already loaded.
load_dotenv()

WORD_RE = re.compile(r"[a-z']+")
EM_DASH_RE = re.compile(r"—|--")
CONTRASTIVE_RE = re.compile(r"\bnot\s+(?:only\s+|merely\s+|just\s+)?\S+(?:\s+\S+){0,8}\s+but\b", re.IGNORECASE)

MIN_WORDS = 50  # matches signals.lexical_score's guard
MODEL = "llama-3.3-70b-versatile"  # same model used in fixtures/generate_ai_samples.py
GROQ_TIMEOUT_S = 10  # /submit calls this synchronously on every request — bound it

MAX_SPANS_PER_CATEGORY = 25  # defensive cap against a pathological/malicious response

# Calibrated against fixtures/ (10 Project Gutenberg human excerpts vs. 11
# Groq-generated AI samples) with the span-based prompt — see
# tools/calibrate.py's llm_heuristic_score output for the reasoning behind
# each value. Re-running calibrate.py after moving to spans + dropping the
# em-dash/contrastive terms (below) produced human mean 0.221 vs. AI mean
# 0.669 — a 0.448 gap, nearly double the 0.245 gap measured before this pass.
W_CONFIDENCE = 1.0
# rule_of_three_instances was the standout discriminator: present in 10/11 AI
# samples but only 2/10 human samples (the two human hits were both formal
# 19th-century essays — Emerson, who leans heavily on rhetorical triads).
# Weighted above the starting guess to reflect that.
W_TRIPLET = 1.5
W_HEDGE = 1.0  # hedge_then_assert_pairs fired on only 1/21 fixtures (the hard
# Gemini sample) — too rare in this set to recalibrate with confidence either
# way, so its starting weight is kept as a reasonable prior.
# REF=2.75 means 3/11 AI fixtures saturate at score=1.0 (clamped) rather than
# their true, higher raw sums — deliberate: this bounded score is meant to
# read as "how strong is the evidence," and saturating on the strongest
# cases trades resolution at the extreme for a wider mean separation between
# classes overall (0.448 clamped vs. ~0.386 if REF were raised to avoid all
# clamping). The unclamped evidence rates in `evidence` are unaffected either way.
LLM_HEURISTIC_REF = 2.75

# em_dash_rate and contrastive_construction_rate are still COMPUTED and
# reported in `evidence` below (free, regex-based, no LLM needed) but no
# longer feed the score. Both ran BACKWARDS on the fixture set: em-dashes
# only appeared in a human sample (Melville), and the "not X but Y"
# contrastive construction fired mostly on formal 19th-century essays
# (Emerson, Thoreau) rather than the Groq-generated AI samples — both are
# markers of ornate rhetorical prose in general, not a modern-LLM tell.
# Rather than guess a negative weight (which would just be over-fitting a
# tiny anti-correlation on n=21), they're kept purely as inspectable context.

_PROMPT_INSTRUCTIONS = """You are analyzing a piece of text for specific rhetorical patterns often associated with AI-generated writing. For each category below, extract the EXACT, VERBATIM quote (a short phrase or sentence copied directly from the text) for every clear, unambiguous instance you find. Be conservative -- do not paraphrase, and skip anything borderline.

Categories:
1. high_confidence_assertions: sentences that confidently assert a speculative, subjective, or otherwise unverifiable claim as settled fact (e.g. using "undoubtedly", "it is clear that", "without question") about something that is not actually verifiable.
2. hedge_then_assert_pairs: instances of the rhetorical pattern "while some might argue/believe X ... it is important/clear that Y" -- a hedge immediately followed by a confident counter-assertion.
3. rule_of_three_instances: parallel three-item lists or triplets used for rhetorical effect (e.g. "fast, reliable, and scalable").

Respond with ONLY a JSON object with a list of verbatim quote strings for exactly these three keys (empty list if none found), nothing else:
{"high_confidence_assertions": ["<quote>", ...], "hedge_then_assert_pairs": ["<quote>", ...], "rule_of_three_instances": ["<quote>", ...]}

Text to analyze:
"""


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def _rate_per_100_words(count, word_count):
    if word_count == 0:
        return 0.0
    return (count / word_count) * 100


def _em_dash_rate(text, word_count):
    """Deterministic, regex-based -- no LLM needed for this one."""
    return _rate_per_100_words(len(EM_DASH_RE.findall(text)), word_count)


def _contrastive_construction_rate(text, word_count):
    """Deterministic, regex-based -- catches "not (only/merely/just) X but Y"."""
    return _rate_per_100_words(len(CONTRASTIVE_RE.findall(text)), word_count)


def _validate_spans(value):
    """Defensively coerce a parsed JSON value into a clean list[str].

    Guards against a malformed/adversarial Groq response feeding garbage
    into `evidence` or blowing up the score: non-list values become an
    empty list, non-string items are dropped, each quote is stripped and
    length-capped, and the list itself is capped at MAX_SPANS_PER_CATEGORY.
    """
    if not isinstance(value, list):
        raise ValueError(f"expected a list of quotes, got {type(value).__name__}")

    spans = []
    for item in value:
        if not isinstance(item, str):
            continue
        quote = item.strip()[:300]
        if quote:
            spans.append(quote)
    return spans[:MAX_SPANS_PER_CATEGORY]


def _build_prompt(text):
    return _PROMPT_INSTRUCTIONS + f'"""\n{text}\n"""'


def _get_client():
    from groq import Groq  # imported lazily so tests never need the package's side effects

    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _call_groq(text, client):
    """Make the Groq call and return the raw parsed JSON dict (unvalidated)."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": _build_prompt(text)}],
        timeout=GROQ_TIMEOUT_S,
    )
    content = response.choices[0].message.content
    return json.loads(content)


@functools.lru_cache(maxsize=256)
def _call_groq_cached(text):
    """Cached wrapper around _call_groq for the DEFAULT client path only.

    Only used when no client is injected (i.e. real /submit traffic, not
    tests or the determinism tool) — identical resubmissions of the same
    text skip the network call entirely. lru_cache does not cache raised
    exceptions, so a transient failure is retried on the next call rather
    than being "stuck" as a permanent failure.
    """
    return _call_groq(text, _get_client())


def detect_ai_heuristics(text, client=None):
    """Advisory Signal 3. Never raises.

    `client` is injectable (any object with a `.chat.completions.create(...)`
    method matching the Groq SDK's shape) so tests can supply a fake client
    without needing a real GROQ_API_KEY or network access. Passing a client
    explicitly also bypasses the default-client response cache (see
    _call_groq_cached) — useful for the determinism-measurement tool, which
    needs every call to actually hit the network.

    Returns {"score": float | None, "evidence": {...}, "note": str | None}.
    """
    word_count = len(WORD_RE.findall(text.lower()))

    em_dash_rate = _em_dash_rate(text, word_count)
    contrastive_rate = _contrastive_construction_rate(text, word_count)

    evidence = {
        # Computed and reported, but NOT part of the score — see the
        # W_EM_DASH/W_CONTRASTIVE comment above.
        "em_dash_rate_per_100_words": em_dash_rate,
        "contrastive_construction_rate_per_100_words": contrastive_rate,
        "high_confidence_assertion_rate_per_100_words": None,
        "hedge_then_assert_rate_per_100_words": None,
        "rule_of_three_rate_per_100_words": None,
        "high_confidence_assertions": None,
        "hedge_then_assert_pairs": None,
        "rule_of_three_instances": None,
        "high_confidence_assertion_spans": None,
        "hedge_then_assert_spans": None,
        "rule_of_three_spans": None,
    }

    if word_count < MIN_WORDS:
        return {"score": None, "evidence": evidence, "note": "skipped: text too short"}

    if client is None and not os.environ.get("GROQ_API_KEY"):
        return {"score": None, "evidence": evidence, "note": "skipped: GROQ_API_KEY not set"}

    try:
        if client is not None:
            parsed = _call_groq(text, client)
        else:
            parsed = _call_groq_cached(text)
        confidence_spans = _validate_spans(parsed["high_confidence_assertions"])
        hedge_spans = _validate_spans(parsed["hedge_then_assert_pairs"])
        triplet_spans = _validate_spans(parsed["rule_of_three_instances"])
    except Exception as exc:  # must never break /submit — degrade gracefully
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return {
            "score": None,
            "evidence": evidence,
            "note": f"llm_heuristic_unavailable: {type(exc).__name__}",
        }

    confidence_count = len(confidence_spans)
    hedge_count = len(hedge_spans)
    triplet_count = len(triplet_spans)

    confidence_rate = _rate_per_100_words(confidence_count, word_count)
    hedge_rate = _rate_per_100_words(hedge_count, word_count)
    triplet_rate = _rate_per_100_words(triplet_count, word_count)

    evidence.update(
        {
            "high_confidence_assertion_rate_per_100_words": confidence_rate,
            "hedge_then_assert_rate_per_100_words": hedge_rate,
            "rule_of_three_rate_per_100_words": triplet_rate,
            "high_confidence_assertions": confidence_count,
            "hedge_then_assert_pairs": hedge_count,
            "rule_of_three_instances": triplet_count,
            "high_confidence_assertion_spans": confidence_spans,
            "hedge_then_assert_spans": hedge_spans,
            "rule_of_three_spans": triplet_spans,
        }
    )

    weighted_sum = W_CONFIDENCE * confidence_rate + W_HEDGE * hedge_rate + W_TRIPLET * triplet_rate
    score = _clamp(weighted_sum / LLM_HEURISTIC_REF)

    return {"score": score, "evidence": evidence, "note": None}
