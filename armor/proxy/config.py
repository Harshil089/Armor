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
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(raw, indent=2) + "\n")
    os.replace(tmp_path, path)


def delete_provider(path: Path, name: str) -> bool:
    """Remove one provider entry by name. Returns False if it wasn't there."""
    raw = json.loads(path.read_text())
    before = len(raw["providers"])
    raw["providers"] = [e for e in raw["providers"] if e["name"] != name]
    if len(raw["providers"]) == before:
        return False
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(raw, indent=2) + "\n")
    os.replace(tmp_path, path)
    return True


@dataclass
class ProxyConfig:
    # ponytail: fail-open on the prompt leg (favor availability), fail-closed on
    # the response leg (favor safety) per proxy design doc section 6. Override
    # per deployment via configs/, not code.
    fail_open_prompt: bool = True
    fail_open_response: bool = False
    upstreams: dict[str, UpstreamConfig] = field(default_factory=lambda: _load_providers(PROVIDERS_PATH))
    _providers_path: Path | None = field(default=None, init=False, repr=False, compare=False)
    _providers_mtime_ns: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            self._providers_path = PROVIDERS_PATH
            self._providers_mtime_ns = PROVIDERS_PATH.stat().st_mtime_ns
        except OSError:
            pass


DEFAULT_CONFIG = ProxyConfig()


def sync_providers(config: ProxyConfig | None = None) -> dict[str, UpstreamConfig]:
    """Reload provider upstreams if PROVIDERS_PATH changed or its mtime updated on disk."""
    target = config or DEFAULT_CONFIG
    path = PROVIDERS_PATH
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return target.upstreams
    if target._providers_path != path or target._providers_mtime_ns != mtime_ns:
        target.upstreams = _load_providers(path)
        target._providers_path = path
        target._providers_mtime_ns = mtime_ns
    return target.upstreams
