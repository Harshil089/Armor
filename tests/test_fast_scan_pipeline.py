from starlette.testclient import TestClient

from armor.api.main import app

client = TestClient(app)


def test_high_severity_entity_blocks_without_model():
    resp = client.post(
        "/v1/inspect",
        json={"direction": "prompt", "content": "here is the key AKIAABCDEFGHIJKLMNOP for prod"},
    )
    body = resp.json()
    assert body["decision"] == "block"
    assert body["route"] == "fast"
    assert body["findings"][0]["label"] == "sensitive_data_exposure"
    assert "AKIAABCDEFGHIJKLMNOP" not in body["findings"][0]["evidence"]


def test_medium_severity_entity_flags():
    resp = client.post(
        "/v1/inspect",
        json={"direction": "prompt", "content": "contact admin@example.com about the outage"},
    )
    body = resp.json()
    assert body["decision"] == "flag"
    assert body["route"] == "fast"


def test_low_severity_only_falls_through_to_stub():
    resp = client.post(
        "/v1/inspect",
        json={"direction": "prompt", "content": "see https://example.com/docs for more"},
    )
    body = resp.json()
    assert body["decision"] == "allow"
    assert body["model_version"] == "stub-0.1"


def test_no_entities_falls_through_to_stub():
    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": "hello there"})
    body = resp.json()
    assert body["decision"] == "allow"
    assert body["route"] == "fast"
    assert body["model_version"] == "stub-0.1"
