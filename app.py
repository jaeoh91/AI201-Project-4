"""Provenance Guard — Flask app.

Milestone 3: /submit + Signal 1 (sentence-length variation).
Milestone 4: + Signal 2 (lexical/MATTR), combine_scores, get_band.
Milestone 5: + transparency label text (labels.py), the full appeals
workflow (POST /appeal, GET /appeals, POST /appeals/<id>/resolve,
GET /submissions/<id>), tightened rate limiting, and a complete audit log.
Signal 3 (llm_signal.detect_ai_heuristics) is advisory-only — see
planning.md §1 Signal 3 — and never feeds into ai_likeness_score/
disagreement/label below.
See planning.md (repo root) for the full spec.
"""

import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import llm_signal
import submissions
from audit_log import append_log, get_log
from labels import generate_label
from signals import combine_scores, get_band, lexical_score, len_variation_score

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    author_id = data.get("author_id") or data.get("creator_id")

    if not text or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' is required and must be a non-empty string"}), 400

    len_var_ai_score = len_variation_score(text)
    lexical_ai_score = lexical_score(text)
    combined = combine_scores(len_var_ai_score, lexical_ai_score)
    ai_likeness_score = combined["ai_likeness_score"]
    disagreement = combined["disagreement"]
    low_signal_confidence = combined["low_signal_confidence"]
    band = get_band(ai_likeness_score, disagreement, low_signal_confidence)
    label = generate_label(ai_likeness_score, disagreement, low_signal_confidence)

    # Advisory only — computed after and independently of the score/label
    # above, and never fed back into them. See llm_signal.py's docstring.
    llm_heuristic = llm_signal.detect_ai_heuristics(text)

    submission_id = str(uuid.uuid4())
    timestamp = _now_iso()

    # attribution/confidence are milestone-literal aliases for Signal 1's
    # result and the combined score, respectively — kept alongside
    # planning.md's own naming (submission_id/ai_likeness_score/label) so
    # both naming conventions are present in one response.
    attribution = len_var_ai_score
    confidence = ai_likeness_score

    response = {
        # planning.md naming (primary)
        "submission_id": submission_id,
        "author_id": author_id,
        "ai_likeness_score": ai_likeness_score,
        "disagreement": disagreement,
        "low_signal_confidence": low_signal_confidence,
        "signals": {"len_var_ai_score": len_var_ai_score, "lexical_ai_score": lexical_ai_score},
        "llm_heuristic": llm_heuristic,
        "band": band,
        "status": "final",
        "timestamp": timestamp,
        "appeal_filed": False,
        # milestone-literal naming (aliases)
        "content_id": submission_id,
        "attribution": attribution,
        "confidence": confidence,
        "label": label,
    }

    # Store the full record (including the original text) so the appeals
    # workflow below has something to validate/update and a reviewer has
    # enough context (planning.md §4 "What a human reviewer sees") without
    # re-running the pipeline.
    submissions.save(submission_id, {**response, "text": text})

    append_log(
        {
            "event": "submission",
            "timestamp": timestamp,
            "content_id": submission_id,
            "attribution": attribution,
            "len_var_ai_score": len_var_ai_score,
            "lexical_ai_score": lexical_ai_score,
            "ai_likeness_score": ai_likeness_score,
            "disagreement": disagreement,
            "low_signal_confidence": low_signal_confidence,
            "band": band,
            "label": label,
            "llm_heuristic": llm_heuristic,
            "status": "final",
            "appeal_filed": False,
        }
    )

    return jsonify(response)


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


@app.route("/submissions/<submission_id>", methods=["GET"])
def get_submission(submission_id):
    record = submissions.get(submission_id)
    if record is None:
        return jsonify({"error": f"no submission found for id '{submission_id}'"}), 404
    return jsonify(record)


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True) or {}
    # Accept both the milestone-literal field names and planning.md's names.
    submission_id = data.get("content_id") or data.get("submission_id")
    appeal_reasoning = data.get("creator_reasoning") or data.get("reason")
    contact = data.get("contact")

    if not submission_id or not isinstance(submission_id, str):
        return jsonify({"error": "'content_id' (or 'submission_id') is required"}), 400
    if not appeal_reasoning or not isinstance(appeal_reasoning, str) or not appeal_reasoning.strip():
        return jsonify({"error": "'creator_reasoning' (or 'reason') is required and must be non-empty"}), 400

    existing = submissions.get(submission_id)
    if existing is None:
        return jsonify({"error": f"no submission found for id '{submission_id}'"}), 404

    appeal_id = str(uuid.uuid4())
    timestamp = _now_iso()
    updated = submissions.file_appeal(submission_id, appeal_id, appeal_reasoning, contact=contact)

    # Logged alongside the original classification decision so a reviewer
    # can see both the appeal and what it's appealing in one entry.
    append_log(
        {
            "event": "appeal_filed",
            "timestamp": timestamp,
            "appeal_id": appeal_id,
            "content_id": submission_id,
            "status": "under_review",
            "appeal_reasoning": appeal_reasoning,
            "reason": appeal_reasoning,
            "contact": contact,
            "appeal_filed": True,
            # Original classification decision, carried alongside the appeal.
            "ai_likeness_score": existing.get("ai_likeness_score"),
            "disagreement": existing.get("disagreement"),
            "low_signal_confidence": existing.get("low_signal_confidence"),
            "band": existing.get("band"),
            "label": existing.get("label"),
        }
    )

    return jsonify(
        {
            "appeal_id": appeal_id,
            "content_id": submission_id,
            "submission_id": submission_id,
            "status": updated["status"],
            "message": "Appeal received. A human reviewer will evaluate this submission.",
        }
    )


@app.route("/appeals", methods=["GET"])
def list_appeals():
    """Reviewer-facing queue (planning.md §4 'What a human reviewer sees')."""
    pending = submissions.list_pending_appeals()
    queue = [
        {
            "appeal_id": record.get("appeal_id"),
            "content_id": record.get("submission_id"),
            "submission_id": record.get("submission_id"),
            "text": record.get("text"),
            "label": record.get("label"),
            "band": record.get("band"),
            "ai_likeness_score": record.get("ai_likeness_score"),
            "signals": record.get("signals"),
            "disagreement": record.get("disagreement"),
            "low_signal_confidence": record.get("low_signal_confidence"),
            "appeal_reasoning": record.get("appeal_reasoning"),
            "contact": record.get("contact"),
            "submitted_at": record.get("timestamp"),
            "status": record.get("status"),
        }
        for record in pending
    ]
    return jsonify({"appeals": queue})


@app.route("/appeals/<appeal_id>/resolve", methods=["POST"])
def resolve_appeal(appeal_id):
    data = request.get_json(silent=True) or {}
    decision = data.get("decision")
    notes = data.get("notes")

    if decision not in ("uphold", "overturn"):
        return jsonify({"error": "'decision' is required and must be 'uphold' or 'overturn'"}), 400

    existing = submissions.get_by_appeal_id(appeal_id)
    if existing is None:
        return jsonify({"error": f"no appeal found for id '{appeal_id}'"}), 404

    updated = submissions.resolve_appeal(appeal_id, decision, notes=notes)
    timestamp = _now_iso()

    append_log(
        {
            "event": "appeal_resolved",
            "timestamp": timestamp,
            "appeal_id": appeal_id,
            "content_id": updated["submission_id"],
            "decision": decision,
            "notes": notes,
            "status": updated["status"],
        }
    )

    return jsonify(
        {
            "appeal_id": appeal_id,
            "content_id": updated["submission_id"],
            "status": updated["status"],
        }
    )


if __name__ == "__main__":
    # Port 5000 is claimed by macOS AirPlay Receiver on many Macs; use 5001
    # to avoid needing curl's -4/127.0.0.1 workaround.
    app.run(debug=True, port=5001)
