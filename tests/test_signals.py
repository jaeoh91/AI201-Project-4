"""Unit tests for the pure detection-signal functions (planning.md §1).

Run with: uv run pytest tests/test_signals.py -v
"""

import pytest

from signals import combine_scores, get_band, lexical_score, len_variation_score

UNIFORM_TEXT = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
VARIED_TEXT = (
    "Cats. "
    "The quick brown fox jumped over the lazy dog and kept running for a while. "
    "Why did it run so fast though, nobody really knows for sure, or maybe they do "
    "but just are not telling anyone about it. "
    "Rain."
)
TOO_SHORT_TEXT = "Just one sentence here, no period trouble."  # 1 sentence


class TestLenVariationScore:
    def test_uniform_sentences_score_high(self):
        score = len_variation_score(UNIFORM_TEXT)
        assert score is not None
        assert score > 0.7  # low CV -> close to 1 (AI-like / uniform)

    def test_varied_sentences_score_low(self):
        score = len_variation_score(VARIED_TEXT)
        assert score is not None
        assert score < 0.3  # high CV -> close to 0 (human-like / varied)

    def test_fewer_than_three_sentences_returns_none(self):
        assert len_variation_score(TOO_SHORT_TEXT) is None
        assert len_variation_score("Two sentences. Right here.") is None
        assert len_variation_score("") is None

    def test_exactly_three_sentences_does_not_guard(self):
        assert len_variation_score("One. Two. Three.") is not None

    def test_score_is_within_unit_range(self):
        score = len_variation_score(UNIFORM_TEXT)
        assert 0.0 <= score <= 1.0


class TestLexicalScore:
    REPETITIVE_TEXT = "The cat sat on the mat. " * 20  # ~120 words, low diversity
    STOCK_PHRASE_TEXT = (
        "Moreover, it is important to note that furthermore we must delve into "
        "this topic in great depth. "
    ) * 6  # ~120 words, dense in stock phrases
    DIVERSE_TEXT = (
        "Quantum entanglement puzzled early physicists who debated whether particles "
        "truly influenced each other instantaneously across vast distances. Einstein "
        "derided this behavior as spooky action at a distance, preferring hidden "
        "variable theories that preserved locality. Decades later, experimental tests "
        "by Aspect and others closed loopholes and confirmed quantum predictions, "
        "overturning many classical intuitions. Entangled photons now underpin "
        "emerging technologies like quantum key distribution, promising theoretically "
        "unbreakable encryption schemes for sensitive communications worldwide."
    )

    def test_under_fifty_words_returns_none(self):
        assert lexical_score("Too short to score reliably at all here.") is None

    def test_exactly_at_boundary_does_not_guard(self):
        fifty_words = " ".join(["word"] * 50)
        assert lexical_score(fifty_words) is not None

    def test_repetitive_text_scores_higher_than_diverse_text(self):
        repetitive = lexical_score(self.REPETITIVE_TEXT)
        diverse = lexical_score(self.DIVERSE_TEXT)
        assert repetitive is not None and diverse is not None
        assert repetitive > diverse

    def test_stock_phrase_heavy_text_scores_high(self):
        score = lexical_score(self.STOCK_PHRASE_TEXT)
        assert score is not None
        assert score > 0.5

    def test_score_is_within_unit_range(self):
        score = lexical_score(self.DIVERSE_TEXT)
        assert 0.0 <= score <= 1.0

    def test_modern_ai_cliches_are_detected(self):
        # Regression test: the original STOCK_PHRASES list (moreover,
        # furthermore, delve into, ...) missed this real Gemini sample
        # entirely (scored 0.0) because it only covers GPT-3.5-era tells.
        text = (
            "The rapid evolution of artificial intelligence is altering the "
            "landscape of digital information. In the realm of cyber threat "
            "intelligence, analysts increasingly harness these tools. "
            "Developing robust frameworks for AI governance is not just a "
            "theoretical exercise, it is an urgent societal imperative, and "
            "we must act to ensure artificial intelligence empowers rather "
            "than deceives the public it is meant to serve today."
        )
        assert lexical_score(text) > 0.0


class TestCombineScores:
    def test_both_signals_present_averages_and_diffs(self):
        result = combine_scores(0.2, 0.4)
        assert result["ai_likeness_score"] == pytest.approx(0.3)
        assert result["disagreement"] == pytest.approx(0.2)
        assert result["low_signal_confidence"] is False

    def test_agreeing_signals_low_disagreement(self):
        result = combine_scores(0.82, 0.80)
        assert result["disagreement"] == pytest.approx(0.02)
        assert result["low_signal_confidence"] is False

    def test_len_var_none_redistributes_to_lexical(self):
        result = combine_scores(None, 0.7)
        assert result["ai_likeness_score"] == 0.7
        assert result["disagreement"] == 1.0
        assert result["low_signal_confidence"] is True

    def test_lexical_none_redistributes_to_len_var(self):
        result = combine_scores(0.6, None)
        assert result["ai_likeness_score"] == 0.6
        assert result["disagreement"] == 1.0
        assert result["low_signal_confidence"] is True

    def test_both_none(self):
        result = combine_scores(None, None)
        assert result["ai_likeness_score"] is None
        assert result["disagreement"] == 1.0
        assert result["low_signal_confidence"] is True


class TestGetBand:
    def test_guard_fired_beats_everything_else(self):
        # Even a score that would otherwise read as confidently AI-like
        # must be banded as "too short" once low_signal_confidence is set.
        band = get_band(0.95, 0.0, low_signal_confidence=True)
        assert band == "Uncertain — too short / weak signal"

    def test_high_disagreement_beats_the_mean(self):
        band = get_band(0.5, 0.8, low_signal_confidence=False)
        assert band == "Uncertain — signals conflict"

    def test_low_score_low_disagreement_is_likely_human(self):
        band = get_band(0.1, 0.05, low_signal_confidence=False)
        assert band == "Likely Human-Written"

    def test_middling_score_low_disagreement_is_uncertain(self):
        band = get_band(0.5, 0.05, low_signal_confidence=False)
        assert band == "Uncertain — middling evidence"

    def test_high_score_low_disagreement_is_likely_ai(self):
        band = get_band(0.9, 0.05, low_signal_confidence=False)
        assert band == "Likely AI-Generated"

    @pytest.mark.parametrize("score", [0.35, 0.65])
    def test_band_edges_are_inclusive_to_uncertain(self, score):
        assert get_band(score, 0.0, low_signal_confidence=False) == "Uncertain — middling evidence"
