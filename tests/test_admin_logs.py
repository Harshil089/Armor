import json

import httpx
import pytest
from starlette.testclient import TestClient

import armor.proxy.app as proxy_app
import armor.models.bandit as bandit_module
import armor.proxy.config as proxy_config
from armor.api.main import app
from armor.observability import log_store


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Isolate both the providers registry and the log DB so tests never
    touch the real seeded files (same lesson as tests/test_admin.py's
    isolated_providers: monkeypatch the module attribute, not a copied
    name, so every module reading it dynamically actually sees the patch)."""
    providers_path = tmp_path / "providers.json"
    providers_path.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "openai", "base_url": "https://api.openai.com", "adapter": "openai"},
                ]
            }
        )
    )
    monkeypatch.setattr(proxy_config, "PROVIDERS_PATH", providers_path)
    original_upstreams = proxy_config.DEFAULT_CONFIG.upstreams
    proxy_config.DEFAULT_CONFIG.upstreams = proxy_config._load_providers(providers_path)

    log_db_path = tmp_path / "logs.db"
    monkeypatch.setattr(log_store, "DB_PATH", log_db_path)
    monkeypatch.setattr(bandit_module, "BANDIT_STATE_PATH", tmp_path / "bandit.npz")
    monkeypatch.setattr(bandit_module, "_bandit", None)

    yield
    proxy_config.DEFAULT_CONFIG.upstreams = original_upstreams


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_direct_inspect_call_is_logged():
    client = TestClient(app)
    client.post("/v1/inspect", json={"direction": "prompt", "content": "hello there"})

    resp = client.get("/admin/logs")
    logs = resp.json()["logs"]
    assert any(l["source"] == "inspect" for l in logs)


def test_proxy_call_is_logged_with_provider_and_detail_visible():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    proxy_app._client = _mock_client(handler)
    client = TestClient(app)
    client.post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "a very specific test prompt"}]},
    )

    logs = client.get("/admin/logs").json()["logs"]
    proxy_logs = [l for l in logs if l["source"] == "proxy" and l["provider"] == "openai"]
    assert proxy_logs

    detail = client.get(f"/admin/logs/{proxy_logs[0]['id']}").json()
    assert "content" in detail
    assert "findings" in detail


def test_log_detail_404_for_unknown_id():
    resp = TestClient(app).get("/admin/logs/999999")
    assert resp.status_code == 404


def test_reviews_endpoint_lists_uncorrected_review_rows():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="review", content="ambiguous request",
    )
    resp = TestClient(app).get("/admin/reviews")
    assert resp.status_code == 200
    ids = {r["id"] for r in resp.json()["reviews"]}
    assert entry_id in ids


def test_reviews_endpoint_excludes_decision_corrected_rows():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="review", content="already handled",
    )
    log_store.record_feedback(
        log_store.DB_PATH, entry_id, "allow", "decision", "reviewer_consensus", "2026-01-01T00:00:00Z"
    )

    ids = {r["id"] for r in TestClient(app).get("/admin/reviews").json()["reviews"]}
    assert entry_id not in ids


def test_reviews_endpoint_excludes_shadow_suggestions_and_proxy_summaries():
    shadow_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="allow", policy_action="review",
        content="shadow recommendation only",
    )
    proxy_id = log_store.log_entry(
        log_store.DB_PATH, source="proxy", decision="review", content="duplicate summary",
    )
    ids = {r["id"] for r in TestClient(app).get("/admin/reviews").json()["reviews"]}
    assert shadow_id not in ids
    assert proxy_id not in ids


def test_correction_defaults_to_human_label_source():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="flag", content="test",
        bandit_action="flag", bandit_context=json.dumps([0.0] * 12),
    )
    TestClient(app).post(f"/admin/logs/{entry_id}/correct", json={"correction": "allow"})
    assert log_store.get_entry(log_store.DB_PATH, entry_id)["label_source"] == "human"


def test_correction_respects_explicit_pseudo_model_label_source():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="flag", content="test",
        bandit_action="flag", bandit_context=json.dumps([0.0] * 12),
    )
    TestClient(app).post(
        f"/admin/logs/{entry_id}/correct", json={"correction": "allow", "label_source": "pseudo_model"}
    )
    assert log_store.get_entry(log_store.DB_PATH, entry_id)["label_source"] == "pseudo_model"
    row = log_store.get_entry(log_store.DB_PATH, entry_id)
    assert row["bandit_correction"] == "allow"
    assert row["bandit_reward_applied"] == 0


def test_decision_feedback_does_not_update_bandit():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="block", content="test",
        bandit_action="flag", bandit_context=json.dumps([0.0] * 12),
    )
    response = TestClient(app).post(
        f"/admin/logs/{entry_id}/correct",
        json={"correction": "allow", "target": "decision", "label_source": "reviewer_consensus"},
    )
    assert response.status_code == 200
    assert response.json()["reward"] is None
    row = log_store.get_entry(log_store.DB_PATH, entry_id)
    assert row["decision_correction"] == "allow"
    assert row["decision_label_source"] == "reviewer_consensus"
    assert row["bandit_correction"] is None


def test_ai_bandit_feedback_is_scored_but_not_applied_until_consensus():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="block", content="test",
        bandit_action="block", bandit_context=json.dumps([0.1] * 12),
    )
    client = TestClient(app)
    provisional = client.post(
        f"/admin/logs/{entry_id}/correct",
        json={"correction": "allow", "target": "both", "label_source": "pseudo_model"},
    )
    assert provisional.status_code == 200
    assert provisional.json()["reward"] == -1.0
    assert provisional.json()["reward_applied"] is False

    consensus = client.post(
        f"/admin/logs/{entry_id}/correct",
        json={"correction": "allow", "target": "both", "label_source": "reviewer_consensus"},
    )
    assert consensus.status_code == 200
    assert consensus.json()["reward_applied"] is True
    assert log_store.get_entry(log_store.DB_PATH, entry_id)["bandit_reward_applied"] == 1

    duplicate = client.post(
        f"/admin/logs/{entry_id}/correct",
        json={"correction": "allow", "target": "both", "label_source": "reviewer_consensus"},
    )
    assert duplicate.status_code == 409


def test_ai_reviewer_consensus_records_reward_without_live_bandit_update():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="allow", content="AI-only eval",
        bandit_action="flag", bandit_context=json.dumps([0.1] * 12),
    )
    response = TestClient(app).post(
        f"/admin/logs/{entry_id}/correct",
        json={
            "correction": "allow",
            "target": "both",
            "label_source": "reviewer_ai_consensus",
        },
    )
    assert response.status_code == 200
    assert response.json()["reward"] == -0.3
    assert response.json()["reward_applied"] is False
    row = log_store.get_entry(log_store.DB_PATH, entry_id)
    assert row["decision_label_source"] == "reviewer_ai_consensus"
    assert row["bandit_label_source"] == "reviewer_ai_consensus"
    assert row["bandit_reward_applied"] == 0


def test_stale_bandit_context_is_saved_without_weight_update():
    entry_id = log_store.log_entry(
        log_store.DB_PATH, source="inspect", decision="allow", content="old context",
        bandit_action="flag", bandit_context=json.dumps([0.1] * 7),
    )
    response = TestClient(app).post(
        f"/admin/logs/{entry_id}/correct",
        json={"correction": "allow", "target": "bandit", "label_source": "human"},
    )
    assert response.status_code == 200
    assert response.json()["reward_applied"] is False
    assert "context shape" in response.json()["update_reason"]
    assert log_store.get_entry(log_store.DB_PATH, entry_id)["bandit_correction"] == "allow"
