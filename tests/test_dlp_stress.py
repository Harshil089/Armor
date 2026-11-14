import re
import string
import time

import pytest
from starlette.testclient import TestClient

from armor.api.main import app
from armor.detectors.registry import default_registry

client = TestClient(app)


# ---------------------------------------------------------------------------
# ReDoS / pathological-input timing per compiled pattern
# ---------------------------------------------------------------------------

ADVERSARIAL_INPUTS = [
    "1" * 50_000,
    ("1" * 19 + " ") * 3000,  # near-miss digit groups, forces lots of \b boundary attempts
    "A" * 50_000,
    ("AB" + "1" * 6 + "C ") * 3000,  # near-miss NINO-shaped noise
    "1234-5678-" * 5000,
]


@pytest.mark.parametrize("text", ADVERSARIAL_INPUTS)
def test_no_catastrophic_backtracking_per_entity(text):
    registry = default_registry()
    for entry in registry.entries:
        start = time.perf_counter()
        entry.pattern.findall(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"{entry.entity_id} took {elapsed:.2f}s on adversarial input"


# ---------------------------------------------------------------------------
# Large-document throughput
# ---------------------------------------------------------------------------

def test_large_document_scan_stays_fast():
    filler = "the quick brown fox jumps over the lazy dog. " * 20000  # ~940KB benign
    planted = " contact admin@example.com and card 4111 1111 1111 1111 end "
    text = filler + planted + filler

    start = time.perf_counter()
    findings = default_registry().scan(text)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"full scan took {elapsed:.2f}s on ~1MB input"
    entity_ids = {f["entity_id"] for f in findings}
    assert "email" in entity_ids
    assert "payment_card" in entity_ids


def test_high_match_volume_no_crash_and_all_masked():
    text = " ".join(f"admin{i}@example.com" for i in range(5000))
    findings = default_registry().scan(text)
    assert len(findings) == 5000
    for f in findings:
        assert "admin" not in f["evidence"] or f["evidence"].startswith("*")


# ---------------------------------------------------------------------------
# False-positive corpus: benign prose/code shouldn't fire high/medium severity
# ---------------------------------------------------------------------------

BENIGN_SAMPLES = [
    "The quarterly report shows revenue increased by twelve percent.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "git status && git add -A && git commit -m 'fix bug'",
    "Please schedule the meeting for next Tuesday at 3pm.",
    "The recipe calls for 2 cups of flour and 350 degrees for 45 minutes.",
    "Chapter 12, page 4451, paragraph 3 discusses the treaty of 1848.",
]


@pytest.mark.parametrize("text", BENIGN_SAMPLES)
def test_benign_prose_triggers_no_decisive_finding(text):
    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": text})
    body = resp.json()
    assert body["decision"] == "allow", f"false positive on: {text!r} -> {body['findings']}"


# ---------------------------------------------------------------------------
# Cross-entity collision: multiple locales active at once, no exceptions,
# every finding still well-formed (mask applied, entity_id/severity present)
# ---------------------------------------------------------------------------

def test_multi_entity_document_all_findings_well_formed():
    text = (
        "PAN ABCPE1234F, Aadhaar 234567890124, SSN 219-09-9999, "
        "NINO AB123456C, SIN 046-454-286, CPF 111.444.777-35, "
        "CN ID 11010519491231002X, Steuer-ID 86095742719, "
        "TFN 123456782, ZA ID 8001015009087, card 4111 1111 1111 1111, "
        "email admin@example.com, IP 10.0.0.1, key AKIAABCDEFGHIJKLMNOP"
    )
    findings = default_registry().scan(text)
    assert len(findings) >= 13  # at least one hit per planted entity
    for f in findings:
        assert set(f) == {"entity_id", "severity", "confidence_tier", "evidence"}
        assert f["severity"] in {"low", "medium", "high"}
        assert f["confidence_tier"] in {"structured", "validated"}


# ---------------------------------------------------------------------------
# Concurrency: shared singletons (registry, module-level _fast_scan) must be
# safe under concurrent requests
# ---------------------------------------------------------------------------

def test_concurrent_requests_no_crash():
    import concurrent.futures

    def hit(i):
        content = "hello" if i % 2 == 0 else "key AKIAABCDEFGHIJKLMNOP here"
        return client.post("/v1/inspect", json={"direction": "prompt", "content": content})

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(hit, range(64)))

    for i, resp in enumerate(results):
        assert resp.status_code == 200
        expected = "block" if i % 2 else "allow"
        assert resp.json()["decision"] == expected


# ---------------------------------------------------------------------------
# Malformed / edge-case inputs
# ---------------------------------------------------------------------------

def test_empty_content_does_not_crash():
    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": ""})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "allow"


def test_unicode_and_control_characters_do_not_crash():
    weird = "😀" * 500 + "\x00\x01\x02" + "п" * 500 + " admin@example.com "
    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": weird})
    assert resp.status_code == 200


def test_whitespace_padding_evasion_between_card_digit_groups():
    # extra spaces beyond the entity's own separator tolerance should not
    # match — confirms the regex isn't accidentally over-permissive
    registry = default_registry()
    findings = registry.scan("4111    1111    1111    1111")
    cc = [f for f in findings if f["entity_id"] == "payment_card"]
    assert cc == []  # 4+ spaces between groups is not a real card format
