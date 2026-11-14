import json
import os

from armor.proxy import config as proxy_config
from armor.proxy.config import ProxyConfig, UpstreamConfig, _load_providers, sync_providers, upsert_provider


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


def test_sync_providers_reloads_when_mtime_changes(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"providers": [{"name": "openai", "base_url": "https://api.openai.com"}]}))
    monkeypatch.setattr(proxy_config, "PROVIDERS_PATH", path)

    cfg = ProxyConfig()
    assert "groq" not in cfg.upstreams

    upsert_provider(path, "groq", "https://api.groq.com/openai", "openai")
    # Force a distinct mtime_ns even on coarse filesystems
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    sync_providers(cfg)
    assert "groq" in cfg.upstreams
    assert cfg.upstreams["groq"].base_url == "https://api.groq.com/openai"
