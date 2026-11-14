from starlette.testclient import TestClient

import armor.api.main as main
from armor.detectors.registry import default_registry

ZA_ID = "8001015009087"
FR_NIR = "185077511400154"


def test_no_filter_returns_every_locale():
    registry = default_registry()
    findings = registry.scan(f"{ZA_ID} {FR_NIR}", locales=None)
    entity_ids = {f["entity_id"] for f in findings}
    assert {"za_id", "fr_nir"} <= entity_ids


def test_filter_scopes_to_requested_locale_plus_global():
    registry = default_registry()
    findings = registry.scan(f"{ZA_ID} {FR_NIR} contact admin@example.com", locales={"FR"})
    entity_ids = {f["entity_id"] for f in findings}
    assert "fr_nir" in entity_ids
    assert "za_id" not in entity_ids
    assert "email" in entity_ids  # global-locale entities always pass through


def test_api_respects_active_locales_env(monkeypatch):
    monkeypatch.setattr(main, "ACTIVE_LOCALES", {"FR"})
    client = TestClient(main.app)

    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": ZA_ID})
    assert resp.json()["decision"] == "allow"  # ZA scoped out under an FR-only deployment

    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": FR_NIR})
    assert resp.json()["decision"] == "block"  # FR still active


def test_api_default_active_locales_is_unfiltered():
    assert main.ACTIVE_LOCALES is None
