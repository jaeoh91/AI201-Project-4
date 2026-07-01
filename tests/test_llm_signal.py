"""Unit tests for the advisory Signal 3 (llm_signal.py).

Uses a fake Groq-shaped client throughout — no real network calls, no
GROQ_API_KEY needed. Imports detect_ai_heuristics directly (rather than
via the `llm_signal` module attribute) so these tests exercise the real
implementation, unaffected by conftest.py's autouse mock used elsewhere.

Run with: uv run pytest tests/test_llm_signal.py -v
"""

import json

import pytest

import llm_signal
from llm_signal import (
    LLM_HEURISTIC_REF,
    W_CONFIDENCE,
    W_HEDGE,
    W_TRIPLET,
    _call_groq_cached,
    _contrastive_construction_rate,
    _em_dash_rate,
    _validate_spans,
    detect_ai_heuristics,
)

LONG_ENOUGH_TEXT = "word " * 100  # 100 words, clears the MIN_WORDS=50 guard


class FakeClient:
    """Minimal stand-in for groq.Groq matching .chat.completions.create(...)."""

    def __init__(self, content=None, exception=None):
        self._content = content
        self._exception = exception
        self.last_kwargs = None
        self.call_count = 0

    def __getattr__(self, name):
        # supports `client.chat.completions.create(...)` chaining
        if name == "chat":
            return self
        if name == "completions":
            return self
        raise AttributeError(name)

    def create(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        if self._exception is not None:
            raise self._exception
        return _FakeResponse(self._content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def _spans(n, label):
    return [f"{label} quote {i}" for i in range(n)]


def fake_client_with_counts(confidence=0, hedge=0, triplet=0):
    """Build a fake client whose response has `confidence`/`hedge`/`triplet`
    dummy quote spans per category (the new list-of-quotes contract).
    """
    payload = json.dumps(
        {
            "high_confidence_assertions": _spans(confidence, "confidence"),
            "hedge_then_assert_pairs": _spans(hedge, "hedge"),
            "rule_of_three_instances": _spans(triplet, "triplet"),
        }
    )
    return FakeClient(content=payload)


class TestGuards:
    def test_short_text_skips_without_calling_client(self):
        client = fake_client_with_counts(confidence=5)  # would blow up the score if called
        result = detect_ai_heuristics("Too short.", client=client)
        assert result["score"] is None
        assert result["note"] == "skipped: text too short"
        assert client.last_kwargs is None  # never invoked

    def test_missing_api_key_and_no_client_skips(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=None)
        assert result["score"] is None
        assert result["note"] == "skipped: GROQ_API_KEY not set"

    def test_injected_client_bypasses_the_api_key_guard(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        client = fake_client_with_counts(confidence=1)
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert result["note"] is None
        assert result["score"] is not None

    def test_guard_still_returns_the_free_regex_submetrics(self):
        text = "Too short but has — an em dash."
        result = detect_ai_heuristics(text, client=fake_client_with_counts())
        assert result["evidence"]["em_dash_rate_per_100_words"] > 0


class TestRegexSubmetrics:
    def test_em_dash_rate_counts_em_dashes_and_double_hyphens(self):
        text = "Some things move quickly here — and that matters — a lot, or so it seems -- somewhat."
        word_count = len(text.split())
        rate = _em_dash_rate(text, word_count)
        assert rate == pytest.approx((3 / word_count) * 100)

    def test_em_dash_rate_is_zero_with_no_dashes(self):
        assert _em_dash_rate("Plain text with no dashes at all here.", 8) == 0.0

    def test_contrastive_construction_rate_detects_not_x_but_y(self):
        text = "This is not just fast but also reliable, and it is not merely useful but essential too."
        word_count = len(text.split())
        rate = _contrastive_construction_rate(text, word_count)
        assert rate == pytest.approx((2 / word_count) * 100)

    def test_contrastive_construction_rate_is_zero_without_the_pattern(self):
        assert _contrastive_construction_rate("A perfectly ordinary sentence with no contrast.", 8) == 0.0

    def test_em_dash_and_contrastive_are_reported_but_not_scored(self):
        """Regression guard for the sign-fix: both regex sub-metrics must
        still appear in evidence, but must NOT move the score — calibration
        found both ran backwards (more common in the human fixtures) on the
        existing fixture set. Two texts with identical LLM-judged spans but
        very different em-dash/contrastive rates must score identically.
        """
        client_a = fake_client_with_counts(confidence=1, hedge=0, triplet=1)
        client_b = fake_client_with_counts(confidence=1, hedge=0, triplet=1)

        # Identical alpha-word tokens in both texts (dashes replace spaces
        # rather than adding/removing words), so word_count -- and therefore
        # every rate that divides by it -- is exactly equal. Any remaining
        # score difference can only come from the em-dash rate itself.
        tail_words = "a plain passage with no unusual punctuation right here at"
        text_without_dashes = "word " * 100 + tail_words
        text_with_dashes = "word " * 100 + tail_words.replace(" ", "—")

        result_a = detect_ai_heuristics(text_with_dashes, client=client_a)
        result_b = detect_ai_heuristics(text_without_dashes, client=client_b)

        assert result_a["evidence"]["em_dash_rate_per_100_words"] > 0
        assert result_b["evidence"]["em_dash_rate_per_100_words"] == 0.0
        # weighted_sum no longer includes em_dash_rate/contrastive_rate, so
        # with equal word counts the scores must match exactly.
        assert result_a["score"] == pytest.approx(result_b["score"])


class TestSpanValidation:
    def test_validate_spans_keeps_only_strings(self):
        assert _validate_spans(["a real quote", 42, None, "another quote"]) == ["a real quote", "another quote"]

    def test_validate_spans_strips_and_caps_length(self):
        padded = "   spaced out quote   "
        assert _validate_spans([padded]) == [padded.strip()]
        very_long = "x" * 500
        assert len(_validate_spans([very_long])[0]) == 300

    def test_validate_spans_drops_blank_strings(self):
        assert _validate_spans(["", "   ", "real"]) == ["real"]

    def test_validate_spans_caps_list_length(self):
        many = [f"quote {i}" for i in range(100)]
        assert len(_validate_spans(many)) == llm_signal.MAX_SPANS_PER_CATEGORY

    def test_validate_spans_rejects_non_list(self):
        with pytest.raises(ValueError):
            _validate_spans("not a list")

    def test_non_list_category_in_response_degrades_gracefully(self):
        """A malformed response (int instead of list, the OLD contract's
        shape) must degrade via the normal error path, not crash /submit.
        """
        payload = json.dumps(
            {"high_confidence_assertions": 2, "hedge_then_assert_pairs": [], "rule_of_three_instances": []}
        )
        client = FakeClient(content=payload)
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert result["score"] is None
        assert "llm_heuristic_unavailable" in result["note"]


class TestScoreFromStructuredEvidence:
    def test_happy_path_computes_score_from_evidence_not_from_llm_judgment(self):
        client = fake_client_with_counts(confidence=2, hedge=1, triplet=3)
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)

        assert result["note"] is None
        evidence = result["evidence"]
        assert evidence["high_confidence_assertions"] == 2
        assert evidence["hedge_then_assert_pairs"] == 1
        assert evidence["rule_of_three_instances"] == 3
        assert evidence["high_confidence_assertion_spans"] == _spans(2, "confidence")
        assert evidence["hedge_then_assert_spans"] == _spans(1, "hedge")
        assert evidence["rule_of_three_spans"] == _spans(3, "triplet")

        word_count = len(LONG_ENOUGH_TEXT.split())
        expected_weighted_sum = (
            W_CONFIDENCE * (2 / word_count * 100) + W_HEDGE * (1 / word_count * 100) + W_TRIPLET * (3 / word_count * 100)
        )
        expected_score = min(1.0, max(0.0, expected_weighted_sum / LLM_HEURISTIC_REF))
        assert result["score"] == pytest.approx(expected_score)

    def test_zero_counts_scores_zero(self):
        client = fake_client_with_counts(confidence=0, hedge=0, triplet=0)
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert result["score"] == 0.0
        assert result["evidence"]["high_confidence_assertion_spans"] == []

    def test_score_is_clamped_to_one(self):
        client = fake_client_with_counts(confidence=50, hedge=50, triplet=50)
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert result["score"] == 1.0


class TestErrorHandling:
    def test_malformed_json_degrades_to_none_with_note(self):
        client = FakeClient(content="this is not json")
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert result["score"] is None
        assert "llm_heuristic_unavailable" in result["note"]
        assert "JSONDecodeError" in result["note"]

    def test_missing_expected_keys_degrades_to_none_with_note(self):
        client = FakeClient(content=json.dumps({"unexpected": "shape"}))
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert result["score"] is None
        assert "llm_heuristic_unavailable" in result["note"]

    def test_client_exception_degrades_to_none_with_note_and_never_raises(self):
        client = FakeClient(exception=TimeoutError("groq took too long"))
        result = detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)  # must not raise
        assert result["score"] is None
        assert "llm_heuristic_unavailable: TimeoutError" in result["note"]

    def test_error_still_returns_the_free_regex_submetrics(self):
        text = "This has an em dash — right here. " + "word " * 50
        client = FakeClient(exception=RuntimeError("boom"))
        result = detect_ai_heuristics(text, client=client)
        assert result["evidence"]["em_dash_rate_per_100_words"] > 0


class TestRequestPathRobustness:
    def test_groq_call_is_given_an_explicit_timeout(self):
        client = fake_client_with_counts(confidence=1)
        detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert client.last_kwargs["timeout"] == llm_signal.GROQ_TIMEOUT_S

    def test_injected_client_path_is_never_cached(self):
        """Passing a client explicitly (tests, the determinism tool) must
        hit .create() every time — caching only applies to the default,
        no-client production path (_call_groq_cached).
        """
        client = fake_client_with_counts(confidence=1)
        detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        detect_ai_heuristics(LONG_ENOUGH_TEXT, client=client)
        assert client.call_count == 2

    def test_default_client_path_is_cached_across_identical_text(self, monkeypatch):
        _call_groq_cached.cache_clear()
        client = fake_client_with_counts(confidence=1)
        monkeypatch.setattr(llm_signal, "_get_client", lambda: client)
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

        detect_ai_heuristics(LONG_ENOUGH_TEXT, client=None)
        detect_ai_heuristics(LONG_ENOUGH_TEXT, client=None)

        assert client.call_count == 1  # second call served from cache
        _call_groq_cached.cache_clear()
