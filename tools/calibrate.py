"""Calibration harness (Milestone 4, step 3; extended for Signal 3).

Runs Signals 1 & 2 (+ combine_scores/get_band) over fixtures/human/*.txt and
fixtures/ai/*.txt, prints a per-file table plus per-class means, so CV_REF /
MATTR_REF / PHRASE_REF (and the band edges) can be hand-tuned in signals.py
until the two classes separate.

Also runs the advisory Signal 3 (llm_signal.detect_ai_heuristics) over the
same fixtures to sanity-check/tune LLM_HEURISTIC_REF and its weights — this
makes real Groq API calls (one per fixture), which is acceptable here since
this is already a one-off dev tool, unlike the request-path in app.py where
it's advisory. Requires GROQ_API_KEY (see .env.example).

Not part of the runtime scoring path.
"""

import glob
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # signals.py / llm_signal.py live at the repo root, not in tools/

from llm_signal import detect_ai_heuristics
from signals import combine_scores, get_band, lexical_score, len_variation_score

FIXTURES_DIR = os.path.join(REPO_ROOT, "fixtures")


def load_class(name):
    paths = sorted(glob.glob(os.path.join(FIXTURES_DIR, name, "*.txt")))
    samples = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            samples.append((os.path.basename(path), f.read()))
    return samples


def score_class(label, samples):
    print(f"\n=== {label} (n={len(samples)}) ===")
    len_vars, lexicals, ai_scores, llm_scores = [], [], [], []
    for name, text in samples:
        lv = len_variation_score(text)
        lx = lexical_score(text)
        combined = combine_scores(lv, lx)
        band = get_band(
            combined["ai_likeness_score"], combined["disagreement"], combined["low_signal_confidence"]
        )
        llm = detect_ai_heuristics(text)
        print(
            f"  {name:24s} len_var={_fmt(lv)}  lexical={_fmt(lx)}  "
            f"ai_likeness={_fmt(combined['ai_likeness_score'])}  "
            f"disagreement={_fmt(combined['disagreement'])}  band={band}"
        )
        print(f"  {'':24s} llm_heuristic={_fmt(llm['score'])}  note={llm['note']}  evidence={llm['evidence']}")
        if lv is not None:
            len_vars.append(lv)
        if lx is not None:
            lexicals.append(lx)
        if combined["ai_likeness_score"] is not None:
            ai_scores.append(combined["ai_likeness_score"])
        if llm["score"] is not None:
            llm_scores.append(llm["score"])

    print(
        f"  -- means: len_var={_fmt(_mean(len_vars))}  lexical={_fmt(_mean(lexicals))}  "
        f"ai_likeness={_fmt(_mean(ai_scores))}  llm_heuristic={_fmt(_mean(llm_scores))}"
    )
    return ai_scores, llm_scores


def _mean(values):
    return statistics.mean(values) if values else None


def _fmt(value):
    return "None " if value is None else f"{value:.3f}"


def main():
    human = load_class("human")
    ai = load_class("ai")

    human_scores, human_llm_scores = score_class("HUMAN", human)
    ai_scores, ai_llm_scores = score_class("AI", ai)

    print("\n=== separation check (ai_likeness_score, Signals 1+2) ===")
    print(f"human mean ai_likeness_score: {_fmt(_mean(human_scores))}")
    print(f"ai mean ai_likeness_score:    {_fmt(_mean(ai_scores))}")
    if human_scores and ai_scores:
        gap = _mean(ai_scores) - _mean(human_scores)
        print(f"gap (ai - human): {gap:.3f} (want solidly positive, ideally > 0.2)")

    print("\n=== separation check (llm_heuristic_score, Signal 3 — advisory only) ===")
    print(f"human mean llm_heuristic_score: {_fmt(_mean(human_llm_scores))}")
    print(f"ai mean llm_heuristic_score:    {_fmt(_mean(ai_llm_scores))}")
    if human_llm_scores and ai_llm_scores:
        llm_gap = _mean(ai_llm_scores) - _mean(human_llm_scores)
        print(f"gap (ai - human): {llm_gap:.3f} (tune LLM_HEURISTIC_REF/weights in llm_signal.py toward this)")


if __name__ == "__main__":
    main()
