from starlette.testclient import TestClient

import armor.api.main as main
import armor.models.armor_base as armor_base


class FakeArmorBaseWeakSignal:
    checkpoint_loaded = True

    def predict(self, text: str) -> dict[str, float]:
        return {"sensitive_data_exposure": 0.4}  # inside the escalate band (0.3-0.5)


class FakeArmorBaseNoSignal:
    checkpoint_loaded = True

    def predict(self, text: str) -> dict[str, float]:
        return {"sensitive_data_exposure": 0.1}  # below ESCALATE_LOW


def test_escalates_to_ner_on_weak_signal_and_finds_a_name(monkeypatch):
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(main, "PII_NER_ENABLED", True)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: FakeArmorBaseWeakSignal())

    client = TestClient(main.app)
    resp = client.post(
        "/v1/inspect",
        json={"direction": "prompt", "content": "Please contact John Smith about this."},
    )
    body = resp.json()
    assert body["decision"] == "flag"
    assert body["route"] == "escalated"
    assert any(f["label"] == "sensitive_data_exposure" for f in body["findings"])
    assert "pii-ner" in body["model_version"]


def test_no_escalation_below_escalate_low(monkeypatch):
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(main, "PII_NER_ENABLED", True)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: FakeArmorBaseNoSignal())

    client = TestClient(main.app)
    resp = client.post(
        "/v1/inspect",
        json={"direction": "prompt", "content": "Please contact John Smith about this."},
    )
    body = resp.json()
    assert body["decision"] == "allow"
    assert body["route"] == "base"


def test_no_escalation_when_ner_disabled(monkeypatch):
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(main, "PII_NER_ENABLED", False)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: FakeArmorBaseWeakSignal())

    client = TestClient(main.app)
    resp = client.post(
        "/v1/inspect",
        json={"direction": "prompt", "content": "Please contact John Smith about this."},
    )
    body = resp.json()
    assert body["decision"] == "allow"
    assert body["route"] == "base"
