import json

from armor.proxy.config import UpstreamConfig, _load_providers


def test_load_providers_reads_seeded_json():
    from armor.proxy.config import PROVIDERS_PATH

    providers = _load_providers(PROVIDERS_PATH)

    assert "openai" in providers and "anthropic" in providers
    assert providers["openai"].adapter == "openai"
    assert providers["anthropic"].adapter == "anthropic"
    # generic OpenAI-compatible providers, not hardcoded anywhere but here
    assert providers["groq"].adapter == "openai"
    assert providers["groq"].base_url.startswith("https://")


def test_load_providers_defaults_adapter_to_openai(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"providers": [{"name": "custom", "base_url": "https://example.com"}]}))

    providers = _load_providers(path)

    assert providers["custom"] == UpstreamConfig(base_url="https://example.com", adapter="openai")
