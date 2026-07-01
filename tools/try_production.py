"""Manual end-to-end tester for the Milestone 5 production layer.

Unlike tools/try_signals.py (which scores text via direct function calls or
a single /submit call), this script drives the *whole* running Flask app
over HTTP: submissions, the transparency label, the full appeals workflow
(file -> reviewer queue -> resolve), the audit log, and rate limiting.

Run it with no arguments for an interactive menu (easiest way to explore):

    python tools/try_production.py
    python try_production.py            # also works from inside tools/

Or use a subcommand directly for scripting:

    python tools/try_production.py submit --text "..." --creator-id me
    python tools/try_production.py appeal --content-id <id> --reason "I wrote this myself."
    python tools/try_production.py appeals
    python tools/try_production.py resolve --appeal-id <id> --decision overturn --notes "ok"
    python tools/try_production.py submission --content-id <id>
    python tools/try_production.py log
    python tools/try_production.py log --event appeal_filed
    python tools/try_production.py rate-limit-test
    python tools/try_production.py demo --file fixtures/human/human_09_thoreau.txt

Uses only the standard library (urllib), matching try_signals.py's style —
no extra HTTP client dependency needed for a manual testing script. Doesn't
import any repo-root module either, so it works regardless of your current
directory or whether you use `uv run` or a bare `python`.

Start the app first: uv run app.py   (serves on http://localhost:5001)
"""

import argparse
import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:5001"

DEMO_SAMPLE_TEXT = (
    "Okay so the hike was a disaster, in the best way. We took a wrong turn "
    "at the first fork, obviously, because the trail sign was basically just "
    "a stick pointing sideways. Ended up bushwhacking for like twenty "
    "minutes before finding the real path again. My boots are still wet. "
    "But we found this tiny waterfall nobody talks about and just sat there "
    "eating trail mix for an hour, not talking, just staring at it. Worth "
    "every soggy step, honestly, even the part where I almost stepped on a "
    "garter snake."
)


# --------------------------------------------------------------------------
# Thin HTTP layer
# --------------------------------------------------------------------------


def _request(method, url, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass
        return exc.code, body
    except urllib.error.URLError as exc:
        return None, {"error": f"Could not reach {url} — is the app running? ({exc.reason})"}


def _print(status, body):
    if status is not None:
        print(f"HTTP {status}")
    print(json.dumps(body, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# Core actions — one function per route, independent of argparse/menu.
# Each prints the result and returns the parsed JSON body.
# --------------------------------------------------------------------------


def do_submit(url, text, creator_id=None):
    payload = {"text": text}
    if creator_id:
        payload["creator_id"] = creator_id
    status, body = _request("POST", f"{url}/submit", payload)
    _print(status, body)
    return body


def do_appeal(url, content_id, reason, contact=None):
    payload = {"content_id": content_id, "creator_reasoning": reason}
    if contact:
        payload["contact"] = contact
    status, body = _request("POST", f"{url}/appeal", payload)
    _print(status, body)
    return body


def do_appeals(url):
    status, body = _request("GET", f"{url}/appeals")
    _print(status, body)
    return body


def do_resolve(url, appeal_id, decision, notes=None):
    payload = {"decision": decision}
    if notes:
        payload["notes"] = notes
    status, body = _request("POST", f"{url}/appeals/{appeal_id}/resolve", payload)
    _print(status, body)
    return body


def do_submission(url, content_id):
    status, body = _request("GET", f"{url}/submissions/{content_id}")
    _print(status, body)
    return body


def do_log(url, event=None, content_id=None, limit=20):
    status, body = _request("GET", f"{url}/log")
    entries = body.get("entries", []) if isinstance(body, dict) else []
    if event:
        entries = [e for e in entries if e.get("event") == event]
    if content_id:
        entries = [e for e in entries if e.get("content_id") == content_id]
    _print(status, {"entries": entries[:limit]})
    return entries


def do_rate_limit_test(url, count=12):
    """Fire `count` rapid /submit requests and report the status codes —
    exercises Flask-Limiter's "10 per minute;100 per day" rule on /submit.
    """
    payload = {
        "text": "This is a test submission for rate limit testing purposes only.",
        "creator_id": "ratelimit-test",
    }
    statuses = []
    for i in range(1, count + 1):
        status, _ = _request("POST", f"{url}/submit", payload)
        statuses.append(status)
        print(f"  request {i:2d}: {status}")
    ok = statuses.count(200)
    limited = statuses.count(429)
    print(f"\n{ok} succeeded (200), {limited} rate-limited (429), out of {len(statuses)} total.")
    if limited == 0:
        print("No 429s seen — try a larger count, or wait for a fresh rate-limit window.")


def do_demo(url, text):
    """Walk the entire production layer end-to-end in one shot: submit,
    inspect the label/band, file an appeal, see it in the reviewer queue,
    resolve it, and confirm everything landed in the audit log.
    """
    print("=== 1. POST /submit ===")
    submitted = do_submit(url, text, creator_id="demo-script")
    content_id = submitted.get("content_id")
    if not content_id:
        print("\nSubmission failed — stopping demo.")
        return
    print(f"\nband={submitted.get('band')!r}")
    print(f"label={submitted.get('label')!r}")

    print("\n=== 2. POST /appeal ===")
    appeal_resp = do_appeal(url, content_id, "I wrote this myself from personal experience.")
    appeal_id = appeal_resp.get("appeal_id")

    print("\n=== 3. GET /appeals (reviewer queue) ===")
    do_appeals(url)

    if appeal_id:
        print("\n=== 4. POST /appeals/<id>/resolve ===")
        do_resolve(url, appeal_id, "overturn", notes="Looks human-written.")

        print("\n=== 5. GET /appeals (should be empty again) ===")
        do_appeals(url)

    print("\n=== 6. GET /submissions/<content_id> (status verified in storage) ===")
    do_submission(url, content_id)

    print(f"\n=== 7. GET /log (filtered to content_id={content_id}) ===")
    do_log(url, content_id=content_id, limit=10)


# --------------------------------------------------------------------------
# Interactive menu — the friendly default when run with no arguments.
# Remembers the last content_id/appeal_id so you don't have to retype or
# copy-paste UUIDs between menu options.
# --------------------------------------------------------------------------

MENU = """
Provenance Guard — production layer tester   (server: {url})

  1) Submit text                 (POST /submit)
  2) File an appeal               (POST /appeal)
  3) View reviewer queue          (GET /appeals)
  4) Resolve an appeal            (POST /appeals/<id>/resolve)
  5) View a submission by ID      (GET /submissions/<id>)
  6) View audit log               (GET /log)
  7) Run rate-limit test          (12 rapid POST /submit)
  8) Run full demo                (submit -> appeal -> resolve -> log)
  9) Change server URL            (currently {url})
  0) Quit
"""


def _prompt(label, default=None):
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _prompt_multiline_text():
    """Read lines until a blank line (or EOF). A blank first line means
    "no text" (the caller falls back to the demo sample) — this does NOT
    support pasting text that itself contains blank lines (e.g. paragraph
    breaks); use --file/option 1's file path for that instead.
    """
    print("Paste/type the text to submit (single block, no blank lines within it). Finish with an empty line (or Ctrl+D).")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def _menu_submit(state):
    file_path = _prompt(
        "path to a text file to submit (blank to type/paste text instead)", default=""
    )
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = _prompt_multiline_text()
        if not text.strip():
            text = DEMO_SAMPLE_TEXT
            print("(using built-in demo sample text)")
    creator_id = _prompt("creator_id", default="try-production-script")
    body = do_submit(state["url"], text, creator_id=creator_id)
    if isinstance(body, dict) and body.get("content_id"):
        state["content_id"] = body["content_id"]


def _menu_appeal(state):
    content_id = _prompt("content_id", default=state.get("content_id"))
    if not content_id:
        print("No content_id available — submit something first (option 1).")
        return
    reason = _prompt(
        "creator_reasoning",
        default="I wrote this myself from personal experience.",
    )
    body = do_appeal(state["url"], content_id, reason)
    if isinstance(body, dict) and body.get("appeal_id"):
        state["appeal_id"] = body["appeal_id"]
        state["content_id"] = content_id


def _menu_appeals(state):
    do_appeals(state["url"])


def _menu_resolve(state):
    appeal_id = _prompt("appeal_id", default=state.get("appeal_id"))
    if not appeal_id:
        print("No appeal_id available — file an appeal first (option 2).")
        return
    decision = _prompt("decision (uphold/overturn)", default="overturn")
    if decision not in ("uphold", "overturn"):
        print("decision must be 'uphold' or 'overturn'.")
        return
    notes = _prompt("notes", default="Reviewed manually via try_production.py.")
    do_resolve(state["url"], appeal_id, decision, notes=notes)


def _menu_submission(state):
    content_id = _prompt("content_id", default=state.get("content_id"))
    if not content_id:
        print("No content_id available — submit something first (option 1).")
        return
    do_submission(state["url"], content_id)


def _menu_log(state):
    event = _prompt("filter by event (submission/appeal_filed/appeal_resolved, blank for all)", default="")
    content_id = _prompt("filter by content_id (blank for all)", default="")
    do_log(state["url"], event=event or None, content_id=content_id or None)


def _menu_rate_limit(state):
    count = _prompt("number of rapid requests", default="12")
    do_rate_limit_test(state["url"], count=int(count))


def _menu_demo(state):
    text = _prompt("text to submit (blank for a built-in sample)", default="")
    do_demo(state["url"], text or DEMO_SAMPLE_TEXT)


def _menu_change_url(state):
    state["url"] = _prompt("server base URL", default=state["url"])


MENU_ACTIONS = {
    "1": _menu_submit,
    "2": _menu_appeal,
    "3": _menu_appeals,
    "4": _menu_resolve,
    "5": _menu_submission,
    "6": _menu_log,
    "7": _menu_rate_limit,
    "8": _menu_demo,
    "9": _menu_change_url,
}


def interactive_menu(url):
    state = {"url": url, "content_id": None, "appeal_id": None}
    print("Make sure the app is running first: uv run app.py")
    while True:
        print(MENU.format(url=state["url"]))
        try:
            choice = input("Choose an option: ").strip()
        except EOFError:
            print("\nBye!")
            return
        if choice in ("0", "q", "quit", "exit"):
            print("Bye!")
            return
        action = MENU_ACTIONS.get(choice)
        if action is None:
            print(f"Unrecognized option: {choice!r}")
            continue
        try:
            action(state)
        except KeyboardInterrupt:
            print("\n(cancelled)")
        except EOFError:
            print("\nBye!")
            return
        except Exception as exc:  # keep the menu alive on any single-action failure
            print(f"Error: {exc}")
        try:
            input("\nPress Enter to continue...")
        except EOFError:
            print("\nBye!")
            return


# --------------------------------------------------------------------------
# argparse subcommands — for scripting/one-shot use
# --------------------------------------------------------------------------


def _read_text(args):
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    if args.text:
        return args.text
    return input("text> ")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base URL of the running app (default: {DEFAULT_URL}).")
    subparsers = parser.add_subparsers(dest="command")  # not required: no subcommand -> interactive menu

    p_submit = subparsers.add_parser("submit", help="POST /submit.")
    p_submit.add_argument("--text", help="Text to submit.")
    p_submit.add_argument("--file", help="Read text to submit from this file.")
    p_submit.add_argument("--creator-id", dest="creator_id", default="try-production-script")

    p_appeal = subparsers.add_parser("appeal", help="POST /appeal.")
    p_appeal.add_argument("--content-id", dest="content_id", required=True)
    p_appeal.add_argument("--reason", required=True, help="creator_reasoning for the appeal.")
    p_appeal.add_argument("--contact", default=None)

    subparsers.add_parser("appeals", help="GET /appeals (reviewer queue).")

    p_resolve = subparsers.add_parser("resolve", help="POST /appeals/<id>/resolve.")
    p_resolve.add_argument("--appeal-id", dest="appeal_id", required=True)
    p_resolve.add_argument("--decision", choices=["uphold", "overturn"], required=True)
    p_resolve.add_argument("--notes", default=None)

    p_submission = subparsers.add_parser("submission", help="GET /submissions/<id>.")
    p_submission.add_argument("--content-id", dest="content_id", required=True)

    p_log = subparsers.add_parser("log", help="GET /log, with optional filters.")
    p_log.add_argument("--event", choices=["submission", "appeal_filed", "appeal_resolved"], default=None)
    p_log.add_argument("--content-id", dest="content_id", default=None)
    p_log.add_argument("--limit", type=int, default=20)

    p_rate = subparsers.add_parser(
        "rate-limit-test", help="Fire rapid /submit requests to trigger 429s (default: 12)."
    )
    p_rate.add_argument("--count", type=int, default=12)

    p_demo = subparsers.add_parser(
        "demo", help="Run the full submit -> appeal -> resolve -> log lifecycle in one shot."
    )
    p_demo.add_argument("--text", help="Text to submit (defaults to a built-in sample).")
    p_demo.add_argument("--file", help="Read text to submit from this file.")

    return parser


def dispatch(args):
    if args.command == "submit":
        do_submit(args.url, _read_text(args), creator_id=args.creator_id)
    elif args.command == "appeal":
        do_appeal(args.url, args.content_id, args.reason, contact=args.contact)
    elif args.command == "appeals":
        do_appeals(args.url)
    elif args.command == "resolve":
        do_resolve(args.url, args.appeal_id, args.decision, notes=args.notes)
    elif args.command == "submission":
        do_submission(args.url, args.content_id)
    elif args.command == "log":
        do_log(args.url, event=args.event, content_id=args.content_id, limit=args.limit)
    elif args.command == "rate-limit-test":
        do_rate_limit_test(args.url, count=args.count)
    elif args.command == "demo":
        text = args.text or (open(args.file, encoding="utf-8").read() if args.file else DEMO_SAMPLE_TEXT)
        do_demo(args.url, text)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        interactive_menu(args.url)
    else:
        dispatch(args)


if __name__ == "__main__":
    main()
