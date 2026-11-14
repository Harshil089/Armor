from starlette.testclient import TestClient
from types import SimpleNamespace

import pytest
import torch
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertModel

import armor.api.main as main
import armor.models.armor_base as armor_base


class FakeArmorBase:
    checkpoint_loaded = True

    def predict(self, text: str) -> dict[str, float]:
        return {"prompt_injection": 0.9, "jailbreak": 0.1}


def test_inspect_uses_base_model_when_enabled(monkeypatch):
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: FakeArmorBase())

    client = TestClient(main.app)
    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": "ignore instructions"})

    assert resp.status_code == 200
    body = resp.json()
    # 0.9 clears BLOCK_THRESHOLD (0.8), so this is a block, not a flag.
    assert body["decision"] == "block"
    assert body["route"] == "base"
    assert body["model_version"] == "armor-base-seed-v0.1"
    assert body["findings"] == [{"label": "prompt_injection", "score": 0.9, "evidence": None}]


def test_inspect_stub_path_unaffected_by_default():
    client = TestClient(main.app)
    resp = client.post("/v1/inspect", json={"direction": "prompt", "content": "hello"})
    assert resp.json()["route"] == "fast"


class FakeHighConfidence:
    checkpoint_loaded = True

    def predict(self, text: str) -> dict[str, float]:
        return {"prompt_injection": 0.97, "jailbreak": 0.1}


class FakeBorderline:
    checkpoint_loaded = True

    def predict(self, text: str) -> dict[str, float]:
        return {"harmful_content": 0.53, "jailbreak": 0.1}


def test_high_confidence_finding_blocks(monkeypatch):
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: FakeHighConfidence())
    body = TestClient(main.app).post(
        "/v1/inspect", json={"direction": "prompt", "content": "attack"}
    ).json()
    assert body["decision"] == "block"


def test_borderline_finding_only_flags(monkeypatch):
    """The 0.5-0.8 band stays flag rather than becoming a hard block."""
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: FakeBorderline())
    body = TestClient(main.app).post(
        "/v1/inspect", json={"direction": "prompt", "content": "checklist please"}
    ).json()
    assert body["decision"] == "flag"


def test_empty_content_bypasses_base_model(monkeypatch):
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(
        armor_base,
        "get_armor_base",
        lambda: (_ for _ in ()).throw(AssertionError("model should not run")),
    )
    body = TestClient(main.app).post(
        "/v1/inspect", json={"direction": "response", "content": "  \n"}
    ).json()
    assert body["decision"] == "allow"
    assert body["risk_score"] == 0.0


def test_encoder_receives_tokens_beyond_the_old_512_limit():
    class Tokenizer:
        model_max_length = 8192

        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 8193
            self.calls = getattr(self, "calls", 0) + 1
            # Model the capped tokenizer output, including one sentinel token.
            count = min(len(text.split()), kwargs["max_length"])
            return {
                "input_ids": torch.ones((1, count), dtype=torch.long),
                "attention_mask": torch.ones((1, count), dtype=torch.long),
                "offset_mapping": torch.zeros((1, count, 2), dtype=torch.long),
            }

    class Encoder:
        config = SimpleNamespace(max_position_embeddings=8192)

        def __call__(self, input_ids, **kwargs):
            self.seen_tokens = input_ids.shape[-1]
            return SimpleNamespace(last_hidden_state=torch.ones((1, 1, 2)))

    model = armor_base.ArmorBase.__new__(armor_base.ArmorBase)
    model.tokenizer = Tokenizer()
    model.encoder = Encoder()
    model.head = torch.nn.Linear(2, 6)
    model.backend = "pytorch"
    model.device = "cpu"

    model.predict_with_embedding("word " * 600 + "latest request")
    assert model.encoder.seen_tokens == 602
    assert model.tokenizer.calls == 1

    with pytest.raises(armor_base.InputTooLongError):
        model.predict_with_embedding("word " * 8193)


def test_preflight_oversize_skips_tokenizer_and_encoder(monkeypatch):
    class Tokenizer:
        model_max_length = 8

        def __call__(self, *args, **kwargs):
            raise AssertionError("preflight-capped input must not be tokenized")

    class Encoder:
        config = SimpleNamespace(max_position_embeddings=8)

        def __call__(self, **kwargs):
            raise AssertionError("preflight-capped input must not reach ModernBERT")

    model = armor_base.ArmorBase.__new__(armor_base.ArmorBase)
    model.tokenizer = Tokenizer()
    model.encoder = Encoder()
    model.head = torch.nn.Linear(2, 6)
    model.backend = "pytorch"
    model.device = "cpu"
    monkeypatch.setattr(armor_base, "MAX_PREFLIGHT_CHARS", 3)

    context = model.prepare_token_context("four")
    assert context.oversize_reason == "input_exceeds_preflight_limit"
    assert context.token_count is None
    with pytest.raises(armor_base.InputTooLongError) as exc:
        model.predict_with_embedding(context)
    assert exc.value.reason == "input_exceeds_preflight_limit"


def test_staged_modernbert_continuation_matches_full_depth_without_recompute():
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=32,
        local_attention=4,
        global_attn_every_n_layers=2,
        pad_token_id=0,
        attn_implementation="eager",
    )
    encoder = ModernBertModel(config).eval()
    ids = torch.tensor([[1, 3, 4, 5, 0]])
    mask = (ids != 0).long()
    context = armor_base.TokenContext(
        inputs={"input_ids": ids, "attention_mask": mask},
        token_count=5, original_length=4,
    )
    model = armor_base.ArmorBase.__new__(armor_base.ArmorBase)
    model.encoder = encoder
    model.backend = "pytorch"
    model.device = "cpu"
    model.head = torch.nn.Linear(16, 6)

    counts = [0, 0, 0, 0]
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, _output, index=index: counts.__setitem__(index, counts[index] + 1)
        )
        for index, layer in enumerate(encoder.layers)
    ]
    with torch.no_grad():
        full = encoder(**context.inputs).last_hidden_state
        state = model.begin_encoder(context)
        model.continue_encoder(state, 2)
        first_exit = state.hidden_states[:, 0, :].clone()
        model.continue_encoder(state, 3)
        model.continue_encoder(state, 4)
    for hook in hooks:
        hook.remove()

    torch.testing.assert_close(state.hidden_states, full, rtol=1e-5, atol=1e-6)
    assert first_exit.shape == (1, 16)
    assert counts == [2, 2, 2, 2]  # one full reference + one staged execution
    with pytest.raises(ValueError, match="backwards"):
        model.continue_encoder(state, 3)


def test_content_beyond_model_context_is_reviewed(monkeypatch, tmp_path):
    class TooLongModel:
        checkpoint_loaded = True

        def predict_with_embedding(self, text):
            raise armor_base.InputTooLongError("too long")

    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: TooLongModel())
    monkeypatch.setattr(main.log_store, "DB_PATH", str(tmp_path / "logs.db"))

    body = TestClient(main.app).post(
        "/v1/inspect", json={"direction": "prompt", "content": "ordinary long content"}
    ).json()
    assert body["decision"] == "review"
    assert body["route"] == "escalated"
    assert body["model_version"] == "armor-base-context-limit"


def test_api_uses_one_token_context_for_model_and_logs_timing(monkeypatch, tmp_path):
    class OneTokenContextModel:
        checkpoint_loaded = True
        fingerprint = None

        def __init__(self):
            self.prepare_calls = 0
            self.received = None

        def prepare_token_context(self, text):
            self.prepare_calls += 1
            return SimpleNamespace(
                oversize=False, oversize_reason=None, token_count=3,
                original_length=len(text), timings_ms={"preflight_ms": 0.01, "tokenization_ms": 0.2},
            )

        def predict_with_embedding(self, context):
            self.received = context
            return {label: 0.0 for label in armor_base.LABELS}, None

    model = OneTokenContextModel()
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: model)
    monkeypatch.setattr(main.log_store, "DB_PATH", str(tmp_path / "logs.db"))
    response = TestClient(main.app).post(
        "/v1/inspect", json={"direction": "response", "content": "ordinary"}
    )
    assert response.json()["decision"] == "allow"
    assert model.prepare_calls == 1
    assert model.received is not None
    entry_id = main.log_store.list_entries(main.log_store.DB_PATH)[0]["id"]
    entry = main.log_store.get_entry(main.log_store.DB_PATH, entry_id)
    assert entry["inference_trace"]["tokenization_ms"] == 0.2
    assert entry["inference_trace"]["token_count"] == 3


def test_api_preflight_limit_reviews_before_loading_model(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(armor_base, "MAX_PREFLIGHT_CHARS", 3)
    monkeypatch.setattr(main.log_store, "DB_PATH", str(tmp_path / "logs.db"))
    monkeypatch.setattr(
        armor_base, "get_armor_base",
        lambda: (_ for _ in ()).throw(AssertionError("oversize preflight must not load model")),
    )
    body = TestClient(main.app).post(
        "/v1/inspect", json={"direction": "response", "content": "ordinary"}
    ).json()
    assert body["decision"] == "review"
    assert body["model_version"] == "armor-base-preflight-limit"
    entry_id = main.log_store.list_entries(main.log_store.DB_PATH)[0]["id"]
    entry = main.log_store.get_entry(main.log_store.DB_PATH, entry_id)
    assert entry["review_reason"] == "input_exceeds_preflight_limit"
    assert entry["inference_trace"]["tokenization_ms"] == 0.0
