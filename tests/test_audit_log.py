"""Unit tests for the JSON-lines audit log.

Run with: uv run pytest tests/test_audit_log.py -v
"""

import audit_log


def test_append_and_get_log_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")

    audit_log.append_log({"event": "first", "value": 1})
    audit_log.append_log({"event": "second", "value": 2})

    entries = audit_log.get_log()
    assert len(entries) == 2


def test_get_log_returns_most_recent_first(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")

    audit_log.append_log({"event": "oldest"})
    audit_log.append_log({"event": "middle"})
    audit_log.append_log({"event": "newest"})

    entries = audit_log.get_log()
    assert [e["event"] for e in entries] == ["newest", "middle", "oldest"]


def test_get_log_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")

    for i in range(5):
        audit_log.append_log({"event": f"entry-{i}"})

    entries = audit_log.get_log(limit=2)
    assert len(entries) == 2
    assert entries[0]["event"] == "entry-4"
    assert entries[1]["event"] == "entry-3"


def test_get_log_on_missing_file_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "does_not_exist.jsonl")
    assert audit_log.get_log() == []


def test_append_log_preserves_arbitrary_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")

    entry = {
        "timestamp": "2026-01-01T00:00:00Z",
        "content_id": "abc-123",
        "len_var_ai_score": 0.42,
        "lexical_ai_score": None,
        "low_signal_confidence": True,
    }
    audit_log.append_log(entry)

    (logged,) = audit_log.get_log()
    assert logged == entry
