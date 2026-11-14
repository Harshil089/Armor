import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from armor.detectors.masks import MASKS
from armor.detectors.validators import VALIDATORS


@dataclass(frozen=True)
class DetectorEntry:
    entity_id: str
    locale: str  # "global" or a locale code, e.g. "IN", "US" (framework doc section 5)
    pattern: re.Pattern
    severity: str
    validator: Callable[[str], bool] | None = None
    mask_fn: Callable[[str], str] = MASKS["last4"]

    @property
    def confidence_tier(self) -> str:
        return "validated" if self.validator is not None else "structured"


def _load_entry(raw: dict) -> DetectorEntry:
    validator_name = raw.get("validator")
    return DetectorEntry(
        entity_id=raw["entity_id"],
        locale=raw.get("locale", "global"),
        pattern=re.compile(raw["pattern"]),
        severity=raw.get("severity", "medium"),
        validator=VALIDATORS.get(validator_name) if validator_name else None,
        mask_fn=MASKS.get(raw.get("mask", "last4"), MASKS["last4"]),
    )


class DetectorRegistry:
    """sensitive_data_exposure entity registry (DLP-1, framework doc section 4).

    Entities are data (JSON config), not code: validator/mask are looked up
    by name against the built-in libraries in validators.py / masks.py, so
    adding an entity is a config change, not a code change (section 5).
    Real entity packs (email/IP/URL, card/IBAN, PAN/Aadhaar) are DLP-2
    through DLP-4 — this registry has no built-in entities of its own.
    """

    def __init__(self, entries: list[DetectorEntry] | None = None):
        self.entries = list(entries or [])

    @classmethod
    def load(cls, config_path: str | Path) -> "DetectorRegistry":
        raw = json.loads(Path(config_path).read_text())
        return cls([_load_entry(entry) for entry in raw["entities"]])

    @classmethod
    def load_dir(cls, config_dir: str | Path) -> "DetectorRegistry":
        """Load every *.json entity pack in a directory (e.g. configs/detectors/)."""
        entries: list[DetectorEntry] = []
        for path in sorted(Path(config_dir).glob("*.json")):
            raw = json.loads(path.read_text())
            entries.extend(_load_entry(entry) for entry in raw["entities"])
        return cls(entries)

    def active(self, locales: set[str] | None = None) -> list[DetectorEntry]:
        if locales is None:
            return list(self.entries)
        return [e for e in self.entries if e.locale == "global" or e.locale in locales]

    def scan(self, text: str, locales: set[str] | None = None) -> list[dict]:
        findings = []
        for entry in self.active(locales):
            for match in entry.pattern.finditer(text):
                value = match.group(0)
                if entry.validator is not None and not entry.validator(value):
                    continue
                findings.append(
                    {
                        "entity_id": entry.entity_id,
                        "severity": entry.severity,
                        "confidence_tier": entry.confidence_tier,
                        "evidence": entry.mask_fn(value),
                    }
                )
        return findings


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "detectors"


def default_registry() -> DetectorRegistry:
    return DetectorRegistry.load_dir(DEFAULT_CONFIG_DIR)
