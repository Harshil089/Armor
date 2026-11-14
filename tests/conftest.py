import os

import pytest

from armor.observability import log_store


@pytest.fixture(autouse=True)
def _isolated_log_db(tmp_path, monkeypatch):
    """Every test that exercises /v1/inspect or /proxy writes a log entry
    (armor/observability/log_store.py). Without this, the whole suite
    would write to the real armor/observability/logs.db on every run.
    Module-qualified attribute (log_store.DB_PATH), not a copied name —
    the hook sites read it through the module each call so this actually
    takes effect (same lesson as the providers.json isolation fixtures)."""
    monkeypatch.setattr(log_store, "DB_PATH", tmp_path / "logs.db")


@pytest.fixture(autouse=True)
def _no_real_semantic_judge(monkeypatch):
    """Several existing tests mock get_armor_base() with scores that land in
    the semantic judge's gate band (armor/api/main.py's SEMANTIC_JUDGE_LOW
    to BANDIT_FLOOR) without knowing it — e.g. FakeArmorBase's 0.9. Without
    this, those "fast" tests would download and run the real ~0.5B-param
    model (armor/models/semantic_judge.py), turning a unit-test run into a
    multi-minute one. Skipped for the dedicated slow test
    (ARMOR_RUN_SLOW_TESTS=1), which needs the real model on purpose."""
    if os.environ.get("ARMOR_RUN_SLOW_TESTS") == "1":
        return
    import armor.models.semantic_judge as semantic_judge

    class _StubJudge:
        def analyze(self, text: str):
            return None  # same as a real judge failure -> caller fills neutral defaults

    monkeypatch.setattr(semantic_judge, "get_semantic_judge", lambda: _StubJudge())
