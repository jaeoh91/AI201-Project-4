"""Determinism-measurement harness for the advisory Signal 3.

llm_signal.py's docstring calls the score computation "deterministic" once
the LLM-provided counts come back — but `temperature=0` reduces, without
fully eliminating, run-to-run variation in what Groq actually returns for a
given text. This tool measures that variation directly instead of assuming
it away: call detect_ai_heuristics() N times on the same text (bypassing the
production response cache, since each run needs to actually hit the network)
and report how much each category's count wobbles.

Real Groq calls (N per invocation), so this is a one-off dev tool like
calibrate.py — not part of the runtime scoring path. Requires GROQ_API_KEY
(see .env.example).

Examples:
    uv run tools/check_determinism.py --file fixtures/ai/ai_11_gemini_user_provided.txt
    uv run tools/check_determinism.py --text "Some text to check." --runs 8
"""

import argparse
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # llm_signal.py lives at the repo root, not in tools/

from llm_signal import _get_client, detect_ai_heuristics

CATEGORIES = ("high_confidence_assertions", "hedge_then_assert_pairs", "rule_of_three_instances")

# A category is flagged "unstable" if its count varies by more than this
# many words-per-100 across runs — a starting threshold, not a hard science.
STABLE_STDEV_THRESHOLD = 0.5


def run_n_times(text, runs, client):
    """Call detect_ai_heuristics() `runs` times, passing `client` explicitly
    each time so the production cache (_call_groq_cached) is bypassed —
    otherwise every call after the first would just return the cached
    result and this tool would measure nothing.
    """
    results = []
    for i in range(runs):
        result = detect_ai_heuristics(text, client=client)
        if result["note"] is not None:
            print(f"  run {i + 1}: FAILED ({result['note']})", file=sys.stderr)
            continue
        results.append(result)
    return results


def summarize(results):
    print(f"\n{len(results)} successful run(s) out of the requested count.\n")
    any_unstable = False

    for category in CATEGORIES:
        counts = [r["evidence"][category] for r in results]
        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts) if len(counts) > 1 else 0.0
        verdict = "STABLE"
        if stdev > STABLE_STDEV_THRESHOLD:
            verdict = "UNSTABLE"
            any_unstable = True
        print(
            f"  {category:30s} counts={counts}  min={min(counts)}  max={max(counts)}  "
            f"mean={mean:.2f}  stdev={stdev:.2f}  [{verdict}]"
        )

    scores = [r["score"] for r in results if r["score"] is not None]
    if scores:
        print(f"\n  llm_heuristic_score across runs: {scores}")
        print(f"  score stdev: {statistics.stdev(scores) if len(scores) > 1 else 0.0:.4f}")

    print()
    if any_unstable:
        print(
            "Verdict: at least one category is NOT stable across repeated calls at "
            "temperature=0. The 'deterministic combination' language in llm_signal.py's "
            "docstring / planning.md should be softened to reflect this."
        )
    else:
        print("Verdict: all categories were stable across repeated calls for this text.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", help="Check determinism for this text.")
    parser.add_argument("--file", help="Check determinism for the contents of this file.")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to call Groq (default: 5).")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        parser.error("provide --text or --file")
        return

    print(f"Running detect_ai_heuristics() {args.runs} times on the same text (real Groq calls, cache bypassed)...")
    client = _get_client()
    results = run_n_times(text, args.runs, client)
    if not results:
        print("All runs failed — nothing to summarize.", file=sys.stderr)
        sys.exit(1)
    summarize(results)


if __name__ == "__main__":
    main()
