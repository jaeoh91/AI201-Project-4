"""Unit tests for the transparency label text (planning.md §3).

Run with: uv run pytest tests/test_labels.py -v
"""

from labels import (
    LABEL_AI_GENERATED,
    LABEL_HUMAN_WRITTEN,
    LABEL_UNCERTAIN_MIXED,
    LABEL_UNCERTAIN_TOO_SHORT,
    generate_label,
)


class TestGenerateLabel:
    def test_guard_fired_returns_too_short_label(self):
        # Even a score that would otherwise read as confidently AI-like
        # must still get the "too short" text once a guard fired.
        label = generate_label(0.95, 0.0, low_signal_confidence=True)
        assert label == LABEL_UNCERTAIN_TOO_SHORT

    def test_high_disagreement_returns_uncertain_mixed_label(self):
        label = generate_label(0.5, 0.8, low_signal_confidence=False)
        assert label == LABEL_UNCERTAIN_MIXED

    def test_low_score_returns_human_written_label(self):
        label = generate_label(0.2, 0.05, low_signal_confidence=False)
        assert label == LABEL_HUMAN_WRITTEN

    def test_middling_score_returns_uncertain_mixed_label(self):
        label = generate_label(0.5, 0.05, low_signal_confidence=False)
        assert label == LABEL_UNCERTAIN_MIXED

    def test_high_score_returns_ai_generated_label(self):
        label = generate_label(0.8, 0.05, low_signal_confidence=False)
        assert label == LABEL_AI_GENERATED

    def test_all_four_label_texts_are_reachable_and_distinct(self):
        reached = {
            generate_label(0.2, 0.05, low_signal_confidence=False),
            generate_label(0.5, 0.05, low_signal_confidence=False),
            generate_label(0.8, 0.05, low_signal_confidence=False),
            generate_label(0.5, 0.8, low_signal_confidence=False),
            generate_label(0.95, 0.0, low_signal_confidence=True),
        }
        assert reached == {
            LABEL_HUMAN_WRITTEN,
            LABEL_UNCERTAIN_MIXED,
            LABEL_AI_GENERATED,
            LABEL_UNCERTAIN_TOO_SHORT,
        }

    def test_ai_generated_label_mentions_appeal(self):
        assert "appeal" in LABEL_AI_GENERATED.lower()

    def test_human_written_label_does_not_claim_verification(self):
        assert "not a verification of authorship" in LABEL_HUMAN_WRITTEN
