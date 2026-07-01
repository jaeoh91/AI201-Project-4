"""Transparency label text (planning.md §3).

Maps the banding rule already implemented in signals.get_band onto the exact
label copy from the planning document. Kept as a separate module so the band
*logic* (signals.get_band) and the *text* a user actually reads don't drift
apart — this module only owns the text mapping, never re-derives the bands.
"""

from signals import get_band

LABEL_AI_GENERATED = (
    "\u26a0\ufe0f **Likely AI-Generated.** This text shows stylistic patterns often "
    "associated with AI-generated writing (uniform sentence rhythm, low "
    "vocabulary variation). This is an estimate from heuristic signals — "
    "not proof of authorship, and not a claim about *how* the text was "
    "produced. If you wrote this yourself, you can file an appeal."
)

LABEL_HUMAN_WRITTEN = (
    "**Likely Human-Written.** This text shows stylistic patterns more "
    "typical of human writing (varied sentence rhythm, diverse vocabulary). "
    "This is an estimate based on heuristics — **not a verification of "
    "authorship**, and it does not rule out AI assistance. Contestable via "
    "appeal."
)

LABEL_UNCERTAIN_MIXED = (
    "\u2754 **Uncertain Provenance.** Our signals were mixed or contradicted each "
    "other, so this text can't be confidently classified as AI- or "
    "human-written. Treat this as inconclusive."
)

LABEL_UNCERTAIN_TOO_SHORT = (
    "\u2754 **Uncertain — Not Enough Text.** This submission was too short or "
    "too structurally sparse for the signals to analyze reliably. No "
    "provenance estimate is offered."
)

# get_band() returns 5 distinct band strings; two of them ("middling
# evidence" and "signals conflict") both map to the same Uncertain label
# text per planning.md's "four label texts (three bands; Uncertain has two
# phrasings)".
BAND_TO_LABEL = {
    "Uncertain — too short / weak signal": LABEL_UNCERTAIN_TOO_SHORT,
    "Uncertain — signals conflict": LABEL_UNCERTAIN_MIXED,
    "Likely Human-Written": LABEL_HUMAN_WRITTEN,
    "Uncertain — middling evidence": LABEL_UNCERTAIN_MIXED,
    "Likely AI-Generated": LABEL_AI_GENERATED,
}


def generate_label(ai_likeness_score, disagreement, low_signal_confidence):
    """Return the full transparency label text for a scored submission.

    Delegates the actual banding decision to signals.get_band (ordered
    guard -> disagreement -> mean-band rule) and only maps the resulting
    band key onto its user-facing text.
    """
    band = get_band(ai_likeness_score, disagreement, low_signal_confidence)
    return BAND_TO_LABEL[band]
