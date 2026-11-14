import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from armor.models.bandit import compute_reward, get_bandit, save_bandit
from armor.observability import log_store
from armor.proxy import config as proxy_config
from armor.proxy.adapters import ADAPTERS
from armor.proxy.config import DEFAULT_CONFIG, delete_provider, upsert_provider
from armor.proxy.env_overrides import KNOWN_ENV_OVERRIDES

# ponytail: module-qualified access (proxy_config.PROVIDERS_PATH), not a
# copied `from ... import PROVIDERS_PATH` name — tests monkeypatch that
# attribute on the config module to redirect writes to a tmp file, which
# only works if this reads it through the module each call, not a value
# bound once at import time.

# ponytail: no auth — matches Armor's current trust model, nothing else in
# this app has auth either. This is a localhost/trusted-network dev tool;
# do not expose it on a public interface. Add a shared-secret header check
# here first if that assumption ever stops holding.
router = APIRouter(prefix="/admin")

_UI_PATH = Path(__file__).with_name("admin_ui.html")
_LOGS_UI_PATH = Path(__file__).with_name("admin_logs_ui.html")


class ProviderCreate(BaseModel):
    name: str
    base_url: str
    adapter: str = "openai"


@router.get("/")
def admin_ui() -> HTMLResponse:
    return HTMLResponse(_UI_PATH.read_text())


@router.get("/logs-page")
def logs_ui() -> HTMLResponse:
    return HTMLResponse(_LOGS_UI_PATH.read_text())


@router.get("/providers")
def list_providers() -> dict:
    return {
        "providers": [
            {
                "name": name,
                "base_url": u.base_url,
                "adapter": u.adapter,
                "env_var": KNOWN_ENV_OVERRIDES.get(name),
            }
            for name, u in DEFAULT_CONFIG.upstreams.items()
        ],
        "adapters": sorted(ADAPTERS),
    }


@router.post("/providers")
def add_provider(provider: ProviderCreate) -> dict:
    if not provider.name.strip():
        raise HTTPException(400, "name must not be empty")
    if not provider.base_url.startswith(("http://", "https://")):
        raise HTTPException(400, "base_url must start with http:// or https://")
    if provider.adapter not in ADAPTERS:
        raise HTTPException(400, f"unknown adapter '{provider.adapter}', valid: {sorted(ADAPTERS)}")

    upsert_provider(proxy_config.PROVIDERS_PATH, provider.name, provider.base_url, provider.adapter)
    DEFAULT_CONFIG.upstreams = proxy_config._load_providers(proxy_config.PROVIDERS_PATH)
    return {"name": provider.name, "base_url": provider.base_url, "adapter": provider.adapter}


@router.delete("/providers/{name}")
def remove_provider(name: str) -> dict:
    if not delete_provider(proxy_config.PROVIDERS_PATH, name):
        raise HTTPException(404, f"unknown provider: {name}")
    DEFAULT_CONFIG.upstreams = proxy_config._load_providers(proxy_config.PROVIDERS_PATH)
    return {"deleted": name}


@router.get("/logs")
def list_logs(limit: int = 200) -> dict:
    return {"logs": log_store.list_entries(log_store.DB_PATH, limit=limit)}


@router.get("/logs/{entry_id}")
def get_log(entry_id: int) -> dict:
    entry = log_store.get_entry(log_store.DB_PATH, entry_id)
    if entry is None:
        raise HTTPException(404, f"no log entry with id {entry_id}")
    return entry


class LogCorrection(BaseModel):
    correction: Literal["allow", "flag", "block", "review"]
    # Compatibility: existing clients corrected the bandit recommendation.
    target: Literal["decision", "bandit", "both"] = "bandit"
    label_source: Literal[
        "human", "pseudo_model", "reviewer_ai_a", "reviewer_ai_b",
        "reviewer_ai_consensus", "reviewer_consensus",
    ] = "human"


@router.post("/logs/{entry_id}/correct")
def correct_log(entry_id: int, body: LogCorrection) -> dict:
    entry = log_store.get_entry(log_store.DB_PATH, entry_id)
    if entry is None:
        raise HTTPException(404, f"no log entry with id {entry_id}")
    reviewed_at = datetime.now(timezone.utc).isoformat()
    reward = None
    if body.target in {"bandit", "both"}:
        if not entry.get("bandit_action"):
            raise HTTPException(400, "no bandit recommendation on this log entry")
        reward = compute_reward(entry["bandit_action"], body.correction)

    try:
        log_store.record_feedback(
            log_store.DB_PATH, entry_id, body.correction, body.target,
            body.label_source, reviewed_at, bandit_reward=reward,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    reward_applied = False
    update_reason = None
    if reward is not None and body.label_source in {"human", "reviewer_consensus"}:
        try:
            context = np.asarray(json.loads(entry["bandit_context"]), dtype=float)
            bandit = get_bandit()
            if context.shape != (bandit.n_features,):
                update_reason = f"context shape {context.shape} does not match current bandit ({bandit.n_features},)"
            else:
                bandit.update(context, entry["bandit_action"], reward, feedback_id=f"logs:{entry_id}")
                save_bandit()
                log_store.mark_bandit_reward_applied(log_store.DB_PATH, entry_id, reward)
                reward_applied = True
        except Exception as exc:
            # Preserve the labeled feedback; the unapplied state is visible and
            # can be retried without double-counting (feedback IDs are persisted).
            update_reason = str(exc)
    elif reward is not None:
        update_reason = "provisional source; reward recorded but not applied"

    return {
        "id": entry_id,
        "target": body.target,
        "reward": reward,
        "reward_applied": reward_applied,
        "update_reason": update_reason,
    }


@router.get("/reviews")
def list_reviews(limit: int = 200) -> dict:
    """Unresolved production review decisions; shadow recommendations and
    duplicate proxy summary rows are excluded."""
    return {"reviews": log_store.list_pending_reviews(log_store.DB_PATH, limit=limit)}
