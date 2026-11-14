import json

import httpx
import pytest
from starlette.testclient import TestClient

import armor.proxy.app as proxy_app
import armor.proxy.config as proxy_config
from armor.api.main import app


@pytest.fixture(autouse=True)
def isolated_providers(tmp_path, monkeypatch):
    """Never mutate the real seeded configs/proxy/providers.json in tests."""
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "openai", "base_url": "https://api.openai.com", "adapter": "openai"},
                    {"name": "anthropic", "base_url": "https://api.anthropic.com", "adapter": "anthropic"},
                ]
            }
        )
    )
    monkeypatch.setattr(proxy_config, "PROVIDERS_PATH", path)
    original_upstreams = proxy_config.DEFAULT_CONFIG.upstreams
    proxy_config.DEFAULT_CONFIG.upstreams = proxy_config._load_providers(path)
    yield path
    proxy_config.DEFAULT_CONFIG.upstreams = original_upstreams


def test_list_providers_returns_seeded_and_adapters():
    resp = TestClient(app).get("/admin/providers")
    assert resp.status_code == 200
    data = resp.json()
    names = {p["name"] for p in data["providers"]}
    assert {"openai", "anthropic"}.issubset(names)
    assert "openai" in data["adapters"] and "anthropic" in data["adapters"]


def test_list_providers_reports_verified_env_var_overrides_only():
    resp = TestClient(app).get("/admin/providers")
    by_name = {p["name"]: p for p in resp.json()["providers"]}
    assert by_name["openai"]["env_var"] == "OPENAI_BASE_URL"
    assert by_name["anthropic"]["env_var"] == "ANTHROPIC_BASE_URL"


def test_unverified_provider_reports_no_env_var():
    client = TestClient(app)
    client.post(
        "/admin/providers",
        json={"name": "groq", "base_url": "https://api.groq.com/openai", "adapter": "openai"},
    )
    resp = client.get("/admin/providers")
    by_name = {p["name"]: p for p in resp.json()["providers"]}
    assert by_name["groq"]["env_var"] is None


def test_add_provider_is_immediately_usable():
    client = TestClient(app)
    resp = client.post(
        "/admin/providers",
        json={"name": "groq", "base_url": "https://api.groq.com/openai", "adapter": "openai"},
    )
    assert resp.status_code == 200

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    proxy_app._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    proxy_resp = client.post(
        "/proxy/groq/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert proxy_resp.status_code == 200
    assert proxy_resp.headers["x-armor-decision"] == "allow"


def test_add_provider_rejects_unknown_adapter():
    resp = TestClient(app).post(
        "/admin/providers",
        json={"name": "bogus", "base_url": "https://example.com", "adapter": "not_a_real_adapter"},
    )
    assert resp.status_code == 400
    assert "not_a_real_adapter" in resp.json()["detail"]


def test_add_provider_rejects_bad_base_url():
    resp = TestClient(app).post(
        "/admin/providers",
        json={"name": "bad", "base_url": "ftp://example.com", "adapter": "openai"},
    )
    assert resp.status_code == 400


def test_delete_provider_removes_it_and_404s_after():
    client = TestClient(app)
    client.post(
        "/admin/providers",
        json={"name": "temp", "base_url": "https://example.com", "adapter": "openai"},
    )
    del_resp = client.delete("/admin/providers/temp")
    assert del_resp.status_code == 200

    proxy_resp = client.post("/proxy/temp/v1/chat/completions", json={"messages": []})
    assert proxy_resp.status_code == 404


def test_delete_unknown_provider_404s():
    resp = TestClient(app).delete("/admin/providers/does-not-exist")
    assert resp.status_code == 404


def test_admin_ui_page_loads():
    resp = TestClient(app).get("/admin/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_logs_ui_page_loads():
    resp = TestClient(app).get("/admin/logs-page")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
