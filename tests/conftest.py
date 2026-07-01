import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # app.py / signals.py / audit_log.py live at the repo root

import pytest

import app as app_module
import audit_log
import llm_signal
import submissions

STUB_LLM_HEURISTIC = {
    "score": None,
    "evidence": {
        "em_dash_rate_per_100_words": 0.0,
        "contrastive_construction_rate_per_100_words": 0.0,
        "high_confidence_assertion_rate_per_100_words": None,
        "hedge_then_assert_rate_per_100_words": None,
        "rule_of_three_rate_per_100_words": None,
        "high_confidence_assertions": None,
        "hedge_then_assert_pairs": None,
        "rule_of_three_instances": None,
        "high_confidence_assertion_spans": None,
        "hedge_then_assert_spans": None,
        "rule_of_three_spans": None,
    },
    "note": "stubbed for tests",
}


@pytest.fixture(autouse=True)
def mock_llm_signal(monkeypatch):
    """Signal 3 makes a real Groq network call. Autouse + patched at the
    `llm_signal` module level (not just via the `client` fixture below) so
    EVERY test — including tests/test_app.py::TestRateLimit, which builds
    its own test client directly and would otherwise fire off real Groq
    calls on every rate-limit iteration — stays offline, fast, and free.
    """
    monkeypatch.setattr(llm_signal, "detect_ai_heuristics", lambda text, client=None: dict(STUB_LLM_HEURISTIC))


@pytest.fixture(autouse=True)
def isolate_audit_log(monkeypatch, tmp_path):
    """Every test that hits /submit, /appeal, etc. appends to audit_log.py's
    module-level LOG_PATH. Without this, running the suite pollutes the
    real audit_log.jsonl at the repo root with test data on every run.
    Individual tests that already monkeypatch LOG_PATH themselves (e.g. to
    assert on log contents) simply override this default afterward.
    """
    monkeypatch.setattr(audit_log, "LOG_PATH", tmp_path / "audit_log.jsonl")


@pytest.fixture(autouse=True)
def clear_submissions_store():
    """The in-memory submissions store (submissions.py) is module-level
    state shared across the whole test session — clear it before every
    test so appeal-lifecycle tests don't see leftovers from prior tests.
    """
    submissions.clear()
    yield
    submissions.clear()


@pytest.fixture
def client():
    """Flask test client with rate limiting disabled (most tests don't care
    about it and hammering /submit repeatedly across the test session would
    otherwise trip the real 10/minute limit and make unrelated tests flaky).

    NOTE: flask-limiter reads the `RATELIMIT_ENABLED` config key exactly once,
    via `config.setdefault(...)`, at extension-init time (when `Limiter(...)`
    is constructed in app.py) and caches the result on `limiter.enabled` —
    setting `app.config["RATELIMIT_ENABLED"]` afterward (e.g. per-test) is a
    no-op. The actual switch is the `limiter.enabled` attribute itself.
    """
    app_module.app.config["TESTING"] = True
    app_module.limiter.enabled = False
    app_module.limiter.reset()
    try:
        with app_module.app.test_client() as test_client:
            yield test_client
    finally:
        app_module.limiter.enabled = True
