import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ponytail: mirrors armor/detectors/registry.py's pattern — provider list is
# data (JSON config), not code, so adding a provider that speaks an already-
# supported wire format (the vast majority — see providers.json's comment)
# never needs a code change. Overridable like ARMOR_ACTIVE_LOCALES elsewhere
# in this codebase.
_DEFAULT_PROVIDERS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "proxy" / "providers.json"
)
PROVIDERS_PATH = Path(os.environ.get("ARMOR_PROXY_PROVIDERS_PATH", _DEFAULT_PROVIDERS_PATH))


@dataclass
class UpstreamConfig:
    base_url: str
    adapter: str = "openai"
    timeout_s: float = 30.0


def _load_providers(path: Path) -> dict[str, UpstreamConfig]:
    raw = json.loads(path.read_text())
    return {
        entry["name"]: UpstreamConfig(base_url=entry["base_url"], adapter=entry.get("adapter", "openai"))
        for entry in raw["providers"]
    }


def upsert_provider(path: Path, name: str, base_url: str, adapter: str) -> None:
    """Add or replace one provider entry in providers.json, by name."""
    raw = json.loads(path.read_text()) if path.exists() else {"providers": []}
    entries = [e for e in raw["providers"] if e["name"] != name]
    entries.append({"name": name, "base_url": base_url, "adapter": adapter})
    raw["providers"] = entries
    path.write_text(json.dumps(raw, indent=2) + "\n")


def delete_provider(path: Path, name: str) -> bool:
    """Remove one provider entry by name. Returns False if it wasn't there."""
    raw = json.loads(path.read_text())
    before = len(raw["providers"])
    raw["providers"] = [e for e in raw["providers"] if e["name"] != name]
    if len(raw["providers"]) == before:
        return False
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return True


@dataclass
class ProxyConfig:
    # ponytail: fail-open on the prompt leg (favor availability), fail-closed on
    # the response leg (favor safety) per proxy design doc section 6. Override
    # per deployment via configs/, not code.
    fail_open_prompt: bool = True
    fail_open_response: bool = False
    upstreams: dict[str, UpstreamConfig] = field(default_factory=lambda: _load_providers(PROVIDERS_PATH))


DEFAULT_CONFIG = ProxyConfig()
