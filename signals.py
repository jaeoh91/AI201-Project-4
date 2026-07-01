"""Pure-function detection signals for Provenance Guard.

No Flask, no network calls — deterministic text statistics only, so this
module can be unit-tested standalone (see calibrate.py) before being wired
into the Flask app. See planning.md §1 for the full spec and caveats.
"""

import re
import statistics

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"[a-z']+")

# Calibrated against fixtures/ (10 Project Gutenberg human excerpts vs. 10
# Groq-generated samples) — see calibrate.py output / CALIBRATION.md for the
# reasoning behind each value.
CV_REF = 0.6
# MATTR ran 0.74-0.89 across BOTH classes in the fixture set, with the AI
# samples' mean (0.847) slightly ABOVE the human mean (0.809) — the opposite
# of the planning.md hypothesis, matching its own caveat that instruction-
# tuned models already produce fairly diverse vocabulary. Raising MATTR_REF
# near the pooled mean keeps repetition_component close to zero for both
# classes so it doesn't actively mislead; separation is instead carried by
# Signal 1 (CV) and the stock-phrase component below.
MATTR_REF = 0.85
MATTR_WINDOW = 50
# Human fixtures had phrase_rate = 0 in every sample; AI fixtures ranged
# 0-1.875. Lowering PHRASE_REF from the 3 starting guess makes this
# correctly-signed component more sensitive within that observed AI range.
PHRASE_REF = 1.5

STOCK_PHRASES = [
    "moreover",
    "furthermore",
    "in conclusion",
    "it is important to note",
    "delve into",
    "on the other hand",
    "in today's world",
    "in summary",
    "overall",
    "additionally",
    # Added after a real Gemini sample (fixtures/ai/ai_11_gemini_user_provided.txt)
    # scored lexical_ai_score=0.0 — the phrases above are GPT-3.5-era tells and
    # missed it entirely. These ran 0/10 in fixtures/human/ but appear across
    # several fixtures/ai/ samples (including that Gemini one), so they're a
    # meaningfully better-calibrated set for modern models, not just this list's
    # original guesses. Still a formality detector at heart (see the blind-spot
    # caveat above) — technical/business human writing can trip these too.
    "landscape",
    "realm",
    "underscore",
    "navigate",
    "leverage",
    "robust",
    "holistic",
    "seamless",
    "tapestry",
    "testament to",
    "harness",
    "unlock the potential",
    "myriad",
    "plethora",
    "meticulous",
    "pivotal",
    "paramount",
    "empower",
    "game changer",
    "in essence",
    "at its core",
    "ever-evolving",
]

# Banding rule constants (planning.md §2) — starting values, calibrate.
DISAGREEMENT_THRESHOLD = 0.4
BAND_LOW = 0.35
BAND_HIGH = 0.65


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def len_variation_score(text):
    """Signal 1 — sentence-length variation ("burstiness" proxy).

    Returns a 0-1 score (1 = AI-like / uniform, 0 = human-like / varied),
    or None if there are fewer than 3 sentences (guard).
    """
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) < 3:
        return None

    lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(lengths)
    if mean_len == 0:
        return None

    cv = statistics.stdev(lengths) / mean_len
    return 1 - _clamp(cv / CV_REF)


def _tokenize(text):
    return WORD_RE.findall(text.lower())


def _mattr(tokens, window=MATTR_WINDOW):
    """Moving-average type-token ratio: length-stable, unlike raw TTR."""
    if not tokens:
        return 0.0
    window = min(window, len(tokens))
    if window == 0:
        return 0.0

    ttrs = []
    for i in range(0, len(tokens) - window + 1):
        chunk = tokens[i : i + window]
        ttrs.append(len(set(chunk)) / len(chunk))
    return statistics.mean(ttrs) if ttrs else 0.0


def _phrase_rate(text, token_count):
    """Stock-phrase occurrences per 100 words."""
    if token_count == 0:
        return 0.0
    lowered = text.lower()
    count = sum(lowered.count(phrase) for phrase in STOCK_PHRASES)
    return (count / token_count) * 100


def lexical_score(text):
    """Signal 2 — lexical diversity (MATTR) + stock-phrase rate.

    Returns a 0-1 score (1 = AI-like), or None if fewer than 50 words (guard).
    """
    tokens = _tokenize(text)
    if len(tokens) < 50:
        return None

    mattr = _mattr(tokens)
    repetition_component = 1 - _clamp(mattr / MATTR_REF)

    phrase_rate = _phrase_rate(text, len(tokens))
    stock_phrase_component = _clamp(phrase_rate / PHRASE_REF)

    return 0.6 * repetition_component + 0.4 * stock_phrase_component


def combine_scores(len_var, lexical):
    """Combine the two signals into ai_likeness_score + disagreement.

    If one signal is None (a guard fired), its weight is redistributed to
    the other and low_signal_confidence is flagged; a single-signal score
    has no meaningful disagreement, so it's treated as maximum uncertainty.
    """
    if len_var is None and lexical is None:
        return {
            "ai_likeness_score": None,
            "disagreement": 1.0,
            "low_signal_confidence": True,
        }

    if len_var is None:
        return {
            "ai_likeness_score": lexical,
            "disagreement": 1.0,
            "low_signal_confidence": True,
        }

    if lexical is None:
        return {
            "ai_likeness_score": len_var,
            "disagreement": 1.0,
            "low_signal_confidence": True,
        }

    return {
        "ai_likeness_score": 0.5 * len_var + 0.5 * lexical,
        "disagreement": abs(len_var - lexical),
        "low_signal_confidence": False,
    }


def get_band(ai_likeness_score, disagreement, low_signal_confidence):
    """Ordered banding rule (planning.md §2) — checked in order, not a
    simple threshold: guard fired -> disagreement -> mean bands.
    """
    if low_signal_confidence:
        return "Uncertain — too short / weak signal"
    if disagreement > DISAGREEMENT_THRESHOLD:
        return "Uncertain — signals conflict"
    if ai_likeness_score < BAND_LOW:
        return "Likely Human-Written"
    if ai_likeness_score <= BAND_HIGH:
        return "Uncertain — middling evidence"
    return "Likely AI-Generated"
