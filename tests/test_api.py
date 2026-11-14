from fastapi.testclient import TestClient

from armor.api.main import app

client = TestClient(app)


def test_inspect_allow_stub():
    resp = client.post(
        "/v1/inspect",
        json={"direction": "prompt", "content": "hello"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "allow"
    assert body["route"] == "fast"
