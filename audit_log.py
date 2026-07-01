"""Structured audit log — JSON Lines file, no print statements.

Every /submit call (and later, appeal events) appends one JSON object per
line. Guarded by a lock since the Flask dev server is threaded by default.
"""

import json
import os
import threading

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.jsonl")
_lock = threading.Lock()


def append_log(entry):
    """Append a single structured entry to the audit log."""
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def get_log(limit=20):
    """Return the most recent `limit` audit log entries, most-recent-first."""
    if not os.path.exists(LOG_PATH):
        return []

    with _lock:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    entries.reverse()
    return entries[:limit]
