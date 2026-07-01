"""Manual testing playground for the detection signals.

Fire arbitrary text through len_variation_score / lexical_score /
combine_scores / get_band and see the full breakdown — either by calling the
functions directly (fast, no server needed) or by hitting the live
/submit endpoint with --live (exercises the real HTTP path + audit log).

Signal 3 (llm_signal.detect_ai_heuristics, advisory-only) is opt-in via --llm
in local mode, since unlike Signals 1/2 it's a real Groq network call with
cost/latency. --live mode always includes it, since app.py's /submit already
computes it for every request regardless of this script.

Examples:
    uv run tools/try_signals.py --text "Some text to score. Right here. Again."
    uv run tools/try_signals.py --file some_sample.txt
    uv run tools/try_signals.py                          # interactive prompt loop
    uv run tools/try_signals.py --llm --text "..."        # + Signal 3 (real Groq call)
    uv run tools/try_signals.py --live --text "..."       # POST to /submit instead
    uv run tools/try_signals.py --live --url http://localhost:5001
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # signals.py lives at the repo root, not in tools/

from llm_signal import detect_ai_heuristics
from signals import combine_scores, get_band, lexical_score, len_variation_score


def score_locally(text, include_llm=False):
    """Run the pure signal functions directly — no Flask, no network (unless
    include_llm is set, which makes one real Groq call for Signal 3).
    """
    word_count = len(text.split())
    len_var = len_variation_score(text)
    lexical = lexical_score(text)
    combined = combine_scores(len_var, lexical)
    label = get_band(combined["ai_likeness_score"], combined["disagreement"], combined["low_signal_confidence"])

    result = {
        "word_count": word_count,
        "len_var_ai_score": len_var,
        "lexical_ai_score": lexical,
        "ai_likeness_score": combined["ai_likeness_score"],
        "disagreement": combined["disagreement"],
        "low_signal_confidence": combined["low_signal_confidence"],
        "label": label,
    }

    if include_llm:
        # Advisory only — shown for inspection, never folded into the fields above.
        result["llm_heuristic"] = detect_ai_heuristics(text)

    return result


def score_via_live_server(text, author_id, url):
    """POST to the running Flask app's /submit route instead of calling the
    functions directly — exercises the full HTTP + audit-log path.
    """
    payload = json.dumps({"text": text, "author_id": author_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/submit",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "body": exc.read().decode("utf-8", errors="ignore")}
    except urllib.error.URLError as exc:
        return {"error": f"Could not reach {url} — is the app running? ({exc.reason})"}


def print_breakdown(result):
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_once(text, args):
    if not text or not text.strip():
        print("(empty input, skipped)")
        return
    if args.live:
        print_breakdown(score_via_live_server(text, args.author_id, args.url))
    else:
        print_breakdown(score_locally(text, include_llm=args.llm))


def interactive_loop(args):
    if args.live:
        mode = f"live @ {args.url}"
    else:
        mode = "local (direct function calls)" + (" + Signal 3 (Groq)" if args.llm else "")
    print(f"Provenance Guard signal tester — mode: {mode}")
    print("Paste a block of text and press Enter to score it. Type 'quit' or Ctrl+D to exit.\n")
    while True:
        try:
            text = input("text> ").strip()
        except EOFError:
            print()
            break
        if text.lower() in ("quit", "exit"):
            break
        run_once(text, args)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", help="Score this text directly and exit.")
    parser.add_argument("--file", help="Score the contents of this file and exit.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="POST to the running app's /submit route instead of calling signals.py directly.",
    )
    parser.add_argument("--url", default="http://localhost:5001", help="Base URL for --live mode.")
    parser.add_argument("--author-id", default="try-signals-script", dest="author_id", help="author_id for --live mode.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also compute the advisory Signal 3 breakdown in local mode (real Groq call; ignored with --live, which already includes it).",
    )
    args = parser.parse_args()

    if args.text:
        run_once(args.text, args)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            run_once(f.read(), args)
    else:
        interactive_loop(args)


if __name__ == "__main__":
    main()
