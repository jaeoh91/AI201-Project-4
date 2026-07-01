"""In-memory submission store (planning.md §4 — Appeals Workflow).

Backs the appeals lifecycle: a submission is created by /submit, optionally
moved to "under_review" by POST /appeal, and finally resolved by
POST /appeals/<id>/resolve. Deliberately NOT persisted across process
restarts and has NO auth — planning.md §4 explicitly documents "no auth
system in scope" as a known limitation rather than something this project
solves. Guarded by a lock since the Flask dev server is threaded by default
(mirrors audit_log.py's pattern).
"""

import threading

_lock = threading.Lock()
_submissions = {}  # submission_id -> record
_appeal_index = {}  # appeal_id -> submission_id


def save(submission_id, record):
    """Store (or overwrite) the full record for a submission."""
    with _lock:
        _submissions[submission_id] = dict(record)


def get(submission_id):
    """Return the stored record for a submission_id, or None."""
    with _lock:
        record = _submissions.get(submission_id)
        return dict(record) if record is not None else None


def get_by_appeal_id(appeal_id):
    """Return the stored record whose appeal_id matches, or None."""
    with _lock:
        submission_id = _appeal_index.get(appeal_id)
        if submission_id is None:
            return None
        record = _submissions.get(submission_id)
        return dict(record) if record is not None else None


def file_appeal(submission_id, appeal_id, appeal_reasoning, contact=None):
    """Flip a submission's status to "under_review" and attach appeal info.

    Returns the updated record, or None if submission_id is unknown.
    """
    with _lock:
        record = _submissions.get(submission_id)
        if record is None:
            return None
        record["status"] = "under_review"
        record["appeal_filed"] = True
        record["appeal_id"] = appeal_id
        record["appeal_reasoning"] = appeal_reasoning
        record["contact"] = contact
        _appeal_index[appeal_id] = submission_id
        return dict(record)


def resolve_appeal(appeal_id, decision, notes=None):
    """Resolve a previously filed appeal.

    `decision` must be "uphold" or "overturn". Returns the updated record,
    or None if appeal_id is unknown.
    """
    with _lock:
        submission_id = _appeal_index.get(appeal_id)
        if submission_id is None:
            return None
        record = _submissions.get(submission_id)
        if record is None:
            return None
        record["status"] = "resolved-upheld" if decision == "uphold" else "resolved-overturned"
        record["resolution_decision"] = decision
        record["resolution_notes"] = notes
        return dict(record)


def list_pending_appeals():
    """Return all records currently awaiting review (status == "under_review")."""
    with _lock:
        return [dict(record) for record in _submissions.values() if record.get("status") == "under_review"]


def clear():
    """Test helper — wipe the store between tests."""
    with _lock:
        _submissions.clear()
        _appeal_index.clear()
