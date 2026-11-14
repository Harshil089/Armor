from armor.detectors.registry import default_registry

SAMPLES = {
    "email": "reach me at admin@example.com for details",
    "url": "see https://example.com/path?x=1 for docs",
    "ipv4": "server lives at 10.0.0.42 in the VPC",
    "ipv6": "bound to fe80::1a2b:3c4d:5e6f:7a8b on the link",
    "aws_access_key": "key is AKIAABCDEFGHIJKLMNOP for the bucket",
    "github_token": "token: ghp_" + "a" * 36,
    "stripe_key": "use sk_live_" + "b" * 24 + " in prod",
    "slack_token": "bot token xoxb-1234567890-abcdef",
    "generic_jwt": "auth: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PYYuNneY",
}


def test_default_registry_loads_global_pack():
    registry = default_registry()
    entity_ids = {e.entity_id for e in registry.entries}
    assert set(SAMPLES) <= entity_ids  # other packs (DLP-3+) add entries, don't replace these


def test_each_sample_is_detected_as_its_own_entity():
    registry = default_registry()
    for entity_id, text in SAMPLES.items():
        findings = registry.scan(text)
        matched_ids = {f["entity_id"] for f in findings}
        assert entity_id in matched_ids, f"{entity_id} not detected in: {text}"
        for finding in findings:
            if finding["entity_id"] == entity_id:
                assert text not in finding["evidence"]


def test_plain_sentence_has_no_matches():
    registry = default_registry()
    assert registry.scan("what is the capital of France") == []
