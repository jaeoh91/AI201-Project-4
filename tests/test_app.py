"""Integration tests for the Flask app's /submit, /log, and appeals routes.

Run with: uv run pytest tests/test_app.py -v
"""

import app as app_module
from labels import LABEL_UNCERTAIN_TOO_SHORT

LONG_AI_STYLE_TEXT = (
    "Moreover, it is important to note that artificial intelligence is "
    "transforming many industries. Furthermore, businesses must adapt to "
    "remain competitive in this rapidly evolving landscape. In conclusion, "
    "organizations that delve into these emerging technologies will be "
    "better positioned for long-term success. Additionally, it is important "
    "to note that continuous learning is essential for every employee. "
    "Overall, the thoughtful integration of new tools requires careful "
    "planning across every department involved in the transition."
)

LONG_HUMAN_STYLE_TEXT = (
    "Okay so the hike was a disaster, in the best way. We took a wrong turn "
    "at the first fork, obviously, because the trail sign was basically just "
    "a stick pointing sideways. Ended up bushwhacking for like twenty "
    "minutes before finding the real path again. My boots are still wet. "
    "But we found this tiny waterfall nobody talks about and just sat there "
    "eating trail mix for an hour, not talking, just staring at it. Worth "
    "every soggy step, honestly, even the part where I almost stepped on a "
    "garter snake."
)


class TestSubmitValidation:
    def test_missing_text_returns_400(self, client):
        resp = client.post("/submit", json={"author_id": "someone"})
        assert resp.status_code == 400

    def test_blank_text_returns_400(self, client):
        resp = client.post("/submit", json={"text": "   "})
        assert resp.status_code == 400

    def test_non_string_text_returns_400(self, client):
        resp = client.post("/submit", json={"text": 12345})
        assert resp.status_code == 400


class TestSubmitResponseShape:
    def test_response_contains_both_naming_conventions(self, client):
        resp = client.post("/submit", json={"text": LONG_AI_STYLE_TEXT, "author_id": "u1"})
        assert resp.status_code == 200
        body = resp.get_json()

        # planning.md naming
        for key in ("submission_id", "ai_likeness_score", "disagreement", "signals", "status", "timestamp"):
            assert key in body

        # milestone-literal aliases
        for key in ("content_id", "attribution", "confidence", "label"):
            assert key in body

        assert body["content_id"] == body["submission_id"]
        assert body["confidence"] == body["ai_likeness_score"]
        assert set(body["signals"].keys()) == {"len_var_ai_score", "lexical_ai_score"}

        # "band" is the short get_band() key; "label" is the full
        # planning.md §3 transparency text derived from it (labels.py).
        assert "band" in body
        assert body["label"] != body["band"]
        assert body["appeal_filed"] is False

        # Signal 3 (advisory) — present, but see test_llm_heuristic_is_advisory_only
        # below for the part that actually matters: it can't move the score.
        assert "llm_heuristic" in body
        assert set(body["llm_heuristic"].keys()) == {"score", "evidence", "note"}

    def test_llm_heuristic_is_advisory_only(self, client, monkeypatch):
        """Regression guard: however Signal 3 scores, it must never change
        ai_likeness_score/disagreement/label — those come only from
        combine_scores/get_band over Signals 1 & 2 (see app.py).
        """
        import llm_signal

        monkeypatch.setattr(
            llm_signal,
            "detect_ai_heuristics",
            lambda text, client=None: {"score": 1.0, "evidence": {"fake": True}, "note": None},
        )
        resp_with_high_llm_score = client.post("/submit", json={"text": LONG_HUMAN_STYLE_TEXT}).get_json()

        monkeypatch.setattr(
            llm_signal,
            "detect_ai_heuristics",
            lambda text, client=None: {"score": 0.0, "evidence": {"fake": True}, "note": None},
        )
        resp_with_low_llm_score = client.post("/submit", json={"text": LONG_HUMAN_STYLE_TEXT}).get_json()

        assert resp_with_high_llm_score["ai_likeness_score"] == resp_with_low_llm_score["ai_likeness_score"]
        assert resp_with_high_llm_score["disagreement"] == resp_with_low_llm_score["disagreement"]
        assert resp_with_high_llm_score["label"] == resp_with_low_llm_score["label"]
        assert resp_with_high_llm_score["llm_heuristic"]["score"] == 1.0
        assert resp_with_low_llm_score["llm_heuristic"]["score"] == 0.0

    def test_creator_id_falls_back_to_author_id_field(self, client):
        resp = client.post("/submit", json={"text": LONG_AI_STYLE_TEXT, "creator_id": "legacy-name"})
        assert resp.get_json()["author_id"] == "legacy-name"

    def test_short_text_triggers_low_signal_confidence(self, client):
        resp = client.post("/submit", json={"text": "Too short to score."})
        body = resp.get_json()
        assert body["low_signal_confidence"] is True
        assert body["band"] == "Uncertain — too short / weak signal"
        assert body["label"] == LABEL_UNCERTAIN_TOO_SHORT

    def test_ai_style_text_scores_above_midpoint(self, client):
        resp = client.post("/submit", json={"text": LONG_AI_STYLE_TEXT})
        body = resp.get_json()
        assert body["ai_likeness_score"] > 0.5

    def test_human_style_text_scores_below_midpoint(self, client):
        resp = client.post("/submit", json={"text": LONG_HUMAN_STYLE_TEXT})
        body = resp.get_json()
        assert body["ai_likeness_score"] < 0.5

    def test_transparency_label_varies_by_confidence_level(self, client):
        """Milestone 5 checkpoint: the label text returned by /submit must
        actually change based on the confidence score — not be a fixed
        string regardless of score. Exercise all three reachable variants
        via inputs known to land in different bands.
        """
        ai_label = client.post("/submit", json={"text": LONG_AI_STYLE_TEXT}).get_json()["label"]
        human_label = client.post("/submit", json={"text": LONG_HUMAN_STYLE_TEXT}).get_json()["label"]
        short_label = client.post("/submit", json={"text": "Too short to score."}).get_json()["label"]

        assert len({ai_label, human_label, short_label}) == 3
        assert short_label == LABEL_UNCERTAIN_TOO_SHORT


class TestLog:
    def test_log_records_a_submission(self, client, monkeypatch, tmp_path):
        import audit_log

        monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")

        submit_resp = client.post("/submit", json={"text": LONG_AI_STYLE_TEXT, "author_id": "logtest"})
        submitted_id = submit_resp.get_json()["content_id"]

        log_resp = client.get("/log")
        assert log_resp.status_code == 200
        entries = log_resp.get_json()["entries"]
        assert any(e["content_id"] == submitted_id for e in entries)

    def test_log_entry_carries_both_signals_and_combined_score(self, client, monkeypatch, tmp_path):
        import audit_log

        monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")

        client.post("/submit", json={"text": LONG_AI_STYLE_TEXT})
        entries = client.get("/log").get_json()["entries"]

        newest = entries[0]
        for key in (
            "len_var_ai_score",
            "lexical_ai_score",
            "ai_likeness_score",
            "disagreement",
            "low_signal_confidence",
            "band",
            "label",
            "llm_heuristic",
            "appeal_filed",
        ):
            assert key in newest
        assert newest["appeal_filed"] is False


class TestAppeal:
    def _submit(self, client, text=LONG_AI_STYLE_TEXT):
        return client.post("/submit", json={"text": text}).get_json()

    def test_appeal_updates_status_and_returns_confirmation(self, client):
        submitted = self._submit(client)
        content_id = submitted["content_id"]

        resp = client.post(
            "/appeal",
            json={
                "content_id": content_id,
                "creator_reasoning": "I wrote this myself from personal experience.",
            },
        )
        body = resp.get_json()

        assert resp.status_code == 200
        assert body["content_id"] == content_id
        assert body["status"] == "under_review"
        assert "appeal_id" in body
        assert "message" in body

        # Status change is reflected in storage, not just the response.
        stored = client.get(f"/submissions/{content_id}").get_json()
        assert stored["status"] == "under_review"
        assert stored["appeal_reasoning"] == "I wrote this myself from personal experience."

    def test_appeal_is_logged_alongside_original_decision(self, client, monkeypatch, tmp_path):
        import audit_log

        monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")

        submitted = self._submit(client)
        content_id = submitted["content_id"]
        reasoning = (
            "I wrote this myself from personal experience. I am a non-native "
            "English speaker and my writing style may appear more formal than typical."
        )
        client.post("/appeal", json={"content_id": content_id, "creator_reasoning": reasoning})

        entries = client.get("/log").get_json()["entries"]
        appeal_entries = [e for e in entries if e.get("content_id") == content_id and e.get("event") == "appeal_filed"]
        assert len(appeal_entries) == 1

        entry = appeal_entries[0]
        assert entry["status"] == "under_review"
        assert entry["appeal_reasoning"] == reasoning
        # Original classification decision is present alongside the appeal.
        assert entry["ai_likeness_score"] == submitted["ai_likeness_score"]
        assert entry["label"] == submitted["label"]

    def test_appeal_missing_content_id_returns_400(self, client):
        resp = client.post("/appeal", json={"creator_reasoning": "no id provided"})
        assert resp.status_code == 400

    def test_appeal_missing_reasoning_returns_400(self, client):
        submitted = self._submit(client)
        resp = client.post("/appeal", json={"content_id": submitted["content_id"]})
        assert resp.status_code == 400

    def test_appeal_unknown_content_id_returns_404(self, client):
        resp = client.post(
            "/appeal", json={"content_id": "does-not-exist", "creator_reasoning": "whatever"}
        )
        assert resp.status_code == 404

    def test_full_appeal_lifecycle_via_appeals_queue_and_resolve(self, client):
        submitted = self._submit(client)
        content_id = submitted["content_id"]

        appeal_resp = client.post(
            "/appeal", json={"content_id": content_id, "creator_reasoning": "please review"}
        ).get_json()
        appeal_id = appeal_resp["appeal_id"]

        queue = client.get("/appeals").get_json()["appeals"]
        assert any(a["appeal_id"] == appeal_id for a in queue)
        pending = next(a for a in queue if a["appeal_id"] == appeal_id)
        assert pending["appeal_reasoning"] == "please review"
        assert pending["text"] == LONG_AI_STYLE_TEXT
        assert pending["status"] == "under_review"

        resolve_resp = client.post(f"/appeals/{appeal_id}/resolve", json={"decision": "overturn", "notes": "ok"})
        resolve_body = resolve_resp.get_json()
        assert resolve_resp.status_code == 200
        assert resolve_body["status"] == "resolved-overturned"

        queue_after = client.get("/appeals").get_json()["appeals"]
        assert not any(a["appeal_id"] == appeal_id for a in queue_after)

    def test_resolve_invalid_decision_returns_400(self, client):
        submitted = self._submit(client)
        appeal_resp = client.post(
            "/appeal", json={"content_id": submitted["content_id"], "creator_reasoning": "x"}
        ).get_json()
        resp = client.post(f"/appeals/{appeal_resp['appeal_id']}/resolve", json={"decision": "maybe"})
        assert resp.status_code == 400

    def test_resolve_unknown_appeal_id_returns_404(self, client):
        resp = client.post("/appeals/does-not-exist/resolve", json={"decision": "uphold"})
        assert resp.status_code == 404


class TestRateLimit:
    def test_exceeding_the_limit_returns_429(self):
        """Dedicated test with rate limiting actually enabled (the shared
        `client` fixture disables it — via `limiter.enabled`, see conftest.py
        — so other tests aren't flaky).
        """
        app_module.app.config["TESTING"] = True
        app_module.limiter.enabled = True
        app_module.limiter.reset()

        try:
            with app_module.app.test_client() as test_client:
                statuses = []
                for _ in range(11):  # the limit is 10/minute
                    resp = test_client.post("/submit", json={"text": LONG_AI_STYLE_TEXT})
                    statuses.append(resp.status_code)
                assert 429 in statuses
        finally:
            app_module.limiter.reset()
            app_module.limiter.enabled = False
