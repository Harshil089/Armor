import hashlib
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from armor.models.labels import LABELS

MODEL_NAME = os.environ.get("ARMOR_BASE_MODEL", "answerdotai/ModernBERT-base")
DEFAULT_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "head.pt")
CHECKPOINT_PATH = os.environ.get("ARMOR_BASE_HEAD_PATH", DEFAULT_CHECKPOINT_PATH)

# ponytail: opt-in ONNX Runtime backend (~21% faster encoder forward pass on
# CPU, verified bit-for-bit-equivalent output — architecture spec section 7's
# "benchmark native PyTorch first, then evaluate ONNX Runtime"). Needs a
# one-time export: python -m armor.models.export_onnx (needs the "onnx" extra).
BACKEND = os.environ.get("ARMOR_BASE_BACKEND", "pytorch")
ONNX_DIR = os.environ.get("ARMOR_BASE_ONNX_DIR", os.path.join(os.path.dirname(__file__), "onnx_export"))
ONNX_FINGERPRINT_PATH = os.path.join(ONNX_DIR, "encoder_fingerprint.txt")
MAX_PREFLIGHT_CHARS = int(os.environ.get("ARMOR_MAX_PREFLIGHT_CHARS", "65536"))
if MAX_PREFLIGHT_CHARS <= 0:
    raise ValueError("ARMOR_MAX_PREFLIGHT_CHARS must be positive")


class InputTooLongError(ValueError):
    """The encoder cannot inspect all content within configured limits."""

    def __init__(self, message: str, *, reason: str = "input_exceeds_model_context"):
        super().__init__(message)
        self.reason = reason


@dataclass
class TokenContext:
    """One bounded tokenization shared by inspection stages."""

    inputs: dict[str, torch.Tensor]
    token_count: int | None
    original_length: int
    offsets: torch.Tensor | None = None
    oversize_reason: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def oversize(self) -> bool:
        return self.oversize_reason is not None


@dataclass
class EncoderState:
    """A resumable state inside the single ModernBERT encoder stack."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    sliding_window_mask: torch.Tensor
    position_ids: torch.Tensor
    next_layer: int
    context: TokenContext
    timings_ms: dict[str, float] = field(default_factory=dict)


def encoder_fingerprint(checkpoint: dict | None) -> str:
    """Content hash of a checkpoint's fine-tuned encoder deltas, or "base"
    when the checkpoint has none. Used to detect a stale ONNX export instead
    of a blunt presence-only check (a fresh re-export must be recognized as
    fresh, not just "has fine-tuned weights so always block")."""
    encoder_state = checkpoint.get("encoder_state_dict") if checkpoint else None
    if not encoder_state:
        return "base"
    digest = hashlib.sha256()
    for key in sorted(encoder_state):
        digest.update(key.encode())
        digest.update(encoder_state[key].numpy().tobytes())
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint: dict | None) -> str:
    """Content hash of the FULL checkpoint (head + encoder deltas). Unlike
    encoder_fingerprint (which only tracks encoder changes, deliberately, for
    the ONNX staleness check above), calibration (armor/models/calibration.py)
    needs to know when the HEAD changed too — most retrains here are
    head-only (fine_tune_layers=0), so encoder_fingerprint alone would keep
    reusing a stale calibration table across two checkpoints that differ only
    in their head weights."""
    digest = hashlib.sha256()
    if checkpoint is None:
        return "untrained"
    for section in ("state_dict", "encoder_state_dict"):
        state = checkpoint.get(section) or {}
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].numpy().tobytes())
    return digest.hexdigest()


class ArmorBase:
    """Dense multi-label security classifier (architecture spec section 5.4).

    If a checkpoint exists at CHECKPOINT_PATH, its head and optional partially
    fine-tuned encoder weights are loaded. Otherwise scores are meaningless.
    Check `checkpoint_loaded` to tell which case you're in.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None, backend: str | None = None):
        self.backend = backend or BACKEND
        # Production-safe default is CPU. A deployment may explicitly select
        # an accelerator; never infer production hardware from the dev host.
        configured_device = device or os.environ.get("ARMOR_BASE_DEVICE", "cpu")
        self.device = configured_device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        checkpoint = (
            torch.load(CHECKPOINT_PATH, map_location="cpu")
            if os.path.exists(CHECKPOINT_PATH)
            else None
        )

        if self.backend == "onnx":
            if not os.path.isdir(ONNX_DIR):
                raise RuntimeError(
                    f"ARMOR_BASE_BACKEND=onnx but no export found at {ONNX_DIR}. "
                    "Run: python -m armor.models.export_onnx (needs pip install -e '.[onnx]')"
                )
            expected = encoder_fingerprint(checkpoint)
            actual = (
                open(ONNX_FINGERPRINT_PATH).read().strip()
                if os.path.exists(ONNX_FINGERPRINT_PATH)
                else "base"
            )
            if expected != actual:
                raise RuntimeError(
                    "ONNX export predates the fine-tuned encoder. Re-export before using ONNX."
                )
            from optimum.onnxruntime import ORTModelForFeatureExtraction

            self.encoder = ORTModelForFeatureExtraction.from_pretrained(ONNX_DIR)
            hidden_size = AutoConfig.from_pretrained(model_name).hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
            if checkpoint and checkpoint.get("encoder_state_dict"):
                self.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
            self.encoder.eval()
            hidden_size = self.encoder.config.hidden_size

        self.head = torch.nn.Linear(hidden_size, len(LABELS)).to(self.device)
        self.checkpoint_loaded = False
        if checkpoint is not None:
            self.head.load_state_dict(checkpoint["state_dict"])
            self.checkpoint_loaded = True
        self.head.eval()
        # Keys the calibration table (armor/models/calibration.py) so a new
        # checkpoint gets a fresh calibration fit instead of reusing a stale
        # one — same staleness problem the ONNX fingerprint check above
        # already solves, same fix, but over the full checkpoint (head
        # included) since most retrains here only change the head.
        self.fingerprint = checkpoint_fingerprint(checkpoint)
        self.encoder_fingerprint = encoder_fingerprint(checkpoint)

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, float]:
        return self.predict_with_embedding(text)[0]

    def prepare_token_context(self, text: str) -> TokenContext:
        """Bound input work and tokenize once for the encoder and policy guards.

        The raw-character limit is a separate resource guard: exceeding it
        routes to REVIEW without claiming the exact model token count. Below
        that limit, tokenizer output is capped at max_tokens + 1 so oversize
        requests retain only the evidence needed to distinguish the limit.
        """
        preflight_started = time.perf_counter()
        if len(text) > MAX_PREFLIGHT_CHARS:
            return TokenContext(
                inputs={}, token_count=None, original_length=len(text),
                oversize_reason="input_exceeds_preflight_limit",
                timings_ms={
                    "preflight_ms": (time.perf_counter() - preflight_started) * 1000,
                    "tokenization_ms": 0.0,
                },
            )
        preflight_ms = (time.perf_counter() - preflight_started) * 1000

        max_tokens = min(
            self.tokenizer.model_max_length,
            getattr(self.encoder.config, "max_position_embeddings", self.tokenizer.model_max_length),
        )
        tokenization_started = time.perf_counter()
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_tokens + 1,
        )
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
        offsets = encoded.pop("offset_mapping", None)
        token_count = int(encoded["input_ids"].shape[-1])
        oversize_reason = "input_exceeds_model_context" if token_count > max_tokens else None
        return TokenContext(
            inputs=encoded,
            token_count=token_count,
            original_length=len(text),
            offsets=offsets,
            oversize_reason=oversize_reason,
            timings_ms={"preflight_ms": preflight_ms, "tokenization_ms": tokenization_ms},
        )

    @torch.no_grad()
    def begin_encoder(self, context: TokenContext) -> EncoderState:
        """Run ModernBERT's embedding stage once and prepare reusable masks.

        This staged interface intentionally supports the native PyTorch model
        only. The current ONNX backend has no public intermediate-state API;
        its existing full-depth path remains unchanged.
        """
        if self.backend == "onnx":
            raise RuntimeError("staged inference requires the native PyTorch encoder")
        if context.oversize:
            raise InputTooLongError(
                context.oversize_reason or "input exceeds inspection limits",
                reason=context.oversize_reason or "input_exceeds_model_context",
            )
        encoder = self.encoder
        if not hasattr(encoder, "layers") or not hasattr(encoder, "embeddings"):
            raise TypeError("encoder does not expose ModernBERT layers")
        if getattr(encoder.config, "_attn_implementation", "") == "flash_attention_2":
            raise RuntimeError("staged inference does not support Flash Attention unpadding")

        inputs = {key: value.to(self.device) for key, value in context.inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        if hasattr(encoder, "warn_if_padding_and_no_attention_mask"):
            encoder.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        global_mask, sliding_mask = encoder._update_attention_mask(
            attention_mask, output_attentions=False
        )
        started = time.perf_counter()
        hidden_states = encoder.embeddings(input_ids=input_ids)
        elapsed = (time.perf_counter() - started) * 1000
        return EncoderState(
            hidden_states=hidden_states,
            attention_mask=global_mask,
            sliding_window_mask=sliding_mask,
            position_ids=position_ids,
            next_layer=1,
            context=context,
            timings_ms={"embeddings_ms": elapsed},
        )

    @torch.no_grad()
    def continue_encoder(self, state: EncoderState, end_layer: int) -> EncoderState:
        """Continue the SAME hidden tensor from its next layer to end_layer.

        Layer numbers are one-based and inclusive. Calling this method with a
        lower/equal checkpoint is rejected, preventing accidental recompute.
        The final norm is applied only after the actual final encoder layer,
        matching ModernBertModel.forward pooling semantics.
        """
        layers = self.encoder.layers
        if not 1 <= end_layer <= len(layers):
            raise ValueError(f"end_layer must be between 1 and {len(layers)}")
        if end_layer < state.next_layer:
            raise ValueError("cannot continue an encoder state backwards")
        start_layer = state.next_layer
        started = time.perf_counter()
        for layer_index in range(state.next_layer, end_layer + 1):
            state.hidden_states = layers[layer_index - 1](
                state.hidden_states,
                attention_mask=state.attention_mask,
                sliding_window_mask=state.sliding_window_mask,
                position_ids=state.position_ids,
                output_attentions=False,
            )[0]
        state.next_layer = end_layer + 1
        if end_layer == len(layers):
            state.hidden_states = self.encoder.final_norm(state.hidden_states)
        state.timings_ms[f"layers_{start_layer}_{end_layer}_ms"] = (
            time.perf_counter() - started
        ) * 1000
        return state

    @torch.no_grad()
    def predict_from_encoder_state(
        self, state: EncoderState
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Run the existing heads from a completed full-depth encoder state."""
        if state.next_layer != len(self.encoder.layers) + 1:
            raise ValueError("the existing policy heads require full-depth encoder state")
        pooled = state.hidden_states[:, 0, :]
        scores = torch.sigmoid(self.head(pooled)).squeeze(0).tolist()
        return dict(zip(LABELS, scores)), pooled.squeeze(0)

    @torch.no_grad()
    def predict_with_adaptive_shadow(
        self, context: TokenContext
    ) -> tuple[dict[str, float], torch.Tensor, dict[str, object]]:
        """Run staged inference but preserve full-depth outputs in shadow mode.

        If no fitted early-exit heads exist, this falls back to the established
        one-call path. With heads present, it records whether the configured
        one-checkpoint or patience policy would have stopped at layer 6/12,
        then continues the same hidden state through layer 22.
        """
        from armor.models.early_exit import candidate_from_logits, load_exit_heads

        def early_candidate(logits, depth):
            raw = dict(zip(LABELS, torch.sigmoid(logits[0]).detach().cpu().tolist()))
            if self.fingerprint is not None:
                from armor.models.calibration import calibrate_scores

                calibrated = {
                    label: value["score"]
                    for label, value in calibrate_scores(raw, self.fingerprint).items()
                }
            else:
                calibrated = raw
            return candidate_from_logits(
                logits, depth=depth, calibrated_threat_scores=calibrated
            )

        if self.backend == "onnx":
            scores, pooled = self.predict_with_embedding(context)
            return scores, pooled, {"adaptive_shadow_status": "unsupported_backend"}
        hidden_size = self.encoder.config.hidden_size
        heads = load_exit_heads(
            hidden_size, expected_encoder_fingerprint=self.encoder_fingerprint
        )
        if heads is None:
            scores, pooled = self.predict_with_embedding(context)
            return scores, pooled, {"adaptive_shadow_status": "exit_heads_unavailable"}

        state = self.begin_encoder(context)
        self.continue_encoder(state, 6)
        started = time.perf_counter()
        out6 = heads.layer6(state.hidden_states)
        head6_ms = (time.perf_counter() - started) * 1000
        candidate6 = early_candidate(out6, 6)

        self.continue_encoder(state, 12)
        started = time.perf_counter()
        out12 = heads.layer12(state.hidden_states)
        head12_ms = (time.perf_counter() - started) * 1000
        candidate12 = early_candidate(out12, 12)

        exit_policy = os.environ.get("ARMOR_EXIT_POLICY", "patience").lower()
        if exit_policy not in {"single", "patience"}:
            raise ValueError("ARMOR_EXIT_POLICY must be 'single' or 'patience'")
        selected = None
        if exit_policy == "single":
            selected = candidate6 or candidate12
        elif candidate6 is not None and candidate12 is not None:
            selected = candidate12

        self.continue_encoder(state, len(self.encoder.layers))
        scores, pooled = self.predict_from_encoder_state(state)
        context.timings_ms.update(state.timings_ms)
        context.timings_ms.update({"exit_head6_ms": head6_ms, "exit_head12_ms": head12_ms})
        if "layers_1_6_ms" in state.timings_ms:
            context.timings_ms["layer6_ms"] = state.timings_ms["layers_1_6_ms"] + head6_ms
        if "layers_7_12_ms" in state.timings_ms:
            context.timings_ms["layer12_ms"] = state.timings_ms["layers_7_12_ms"] + head12_ms
        if "layers_13_22_ms" in state.timings_ms:
            context.timings_ms["layer22_ms"] = state.timings_ms["layers_13_22_ms"]
        shadow = {
            "adaptive_shadow_status": "evaluated",
            "approximate_compute_path": "staged_full_depth_shadow",
            "shadow_exit_depth": selected.depth if selected else None,
            "shadow_decision": selected.decision if selected else None,
            "shadow_confidence": selected.confidence if selected else None,
            "shadow_policy": exit_policy,
        }
        if context.token_count and context.token_count > int(os.environ.get("ARMOR_WINDOW_SIZE", "512")):
            from armor.inference.adaptive import run_adaptive_windows

            window_run = run_adaptive_windows(
                self, context, self.tokenizer,
                window_size=int(os.environ.get("ARMOR_WINDOW_SIZE", "512")),
                overlap=int(os.environ.get("ARMOR_WINDOW_OVERLAP", "96")),
                max_windows=int(os.environ.get("ARMOR_MAX_WINDOWS", "24")),
                max_full_depth_windows=int(os.environ.get("ARMOR_MAX_FULL_DEPTH_WINDOWS", "4")),
                max_context_expansions=int(os.environ.get("ARMOR_MAX_CONTEXT_EXPANSIONS", "2")),
                max_expanded_tokens=int(os.environ.get("ARMOR_EXPANDED_CONTEXT_TOKENS", "768")),
                max_inspection_time_ms=float(os.environ.get("ARMOR_MAX_INSPECTION_TIME_MS", "3000")),
                exit_policy=exit_policy,
            )
            shadow["windowed_shadow_decision"] = window_run.decision
            shadow["windowed_shadow_review_reason"] = window_run.review_reason
            shadow["windowed_shadow_status"] = window_run.trace.get("approximate_compute_path")
            for key, value in window_run.trace.items():
                shadow[f"windowed_{key}"] = value
        return scores, pooled, shadow

    @torch.no_grad()
    def window_scores(
        self,
        context: TokenContext,
        window: int = 512,
        overlap: int = 96,
        batch_size: int = 8,
        early_stop_threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-depth sigmoid scores [n_windows, n_labels] and pooled CLS [n_windows, hidden].

        Attention cost is quadratic in length, so N windows of `window` tokens
        beat one N*window pass, and 512 matches the training collator.
        """
        if context.oversize:
            raise InputTooLongError(
                context.oversize_reason or "input exceeds inspection limits",
                reason=context.oversize_reason or "input_exceeds_model_context",
            )
        ids = context.inputs["input_ids"][0]
        body = ids[1:-1]
        span = window - 2
        if len(body) <= span:
            scores, pooled = self.predict_with_embedding(context)
            return torch.tensor([list(scores.values())]), pooled[None]
        starts = list(range(0, max(1, len(body) - overlap), span - overlap))
        max_windows = int(os.environ.get("ARMOR_MAX_WINDOWS", "24"))
        if max_windows > 0 and len(starts) > max_windows:
            starts = starts[:max_windows]
        batch_size = int(os.environ.get("ARMOR_WINDOW_BATCH_SIZE", str(batch_size)))
        cls_id, sep_id, pad_id = (
            self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id,
        )
        rows = [
            torch.cat([ids.new_tensor([cls_id]), body[s:s + span], ids.new_tensor([sep_id])])
            for s in starts
        ]
        if early_stop_threshold is None:
            env_stop = os.environ.get("ARMOR_WINDOW_EARLY_STOP_THRESHOLD")
            if env_stop:
                early_stop_threshold = float(env_stop)
        pi_idx = LABELS.index("prompt_injection")
        scores, pooled_all = [], []
        started = time.perf_counter()
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            width = max(len(r) for r in chunk)
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, r in enumerate(chunk):
                input_ids[j, :len(r)] = r
                mask[j, :len(r)] = 1
            pooled = self.encoder(
                input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device)
            ).last_hidden_state[:, 0, :].clone()
            scores.append(torch.sigmoid(self.head(pooled.to(self.head.weight.device))))
            pooled_all.append(pooled)
            if early_stop_threshold is not None and early_stop_threshold > 0:
                cat_scores = torch.cat(scores)
                if (
                    float(cat_scores[:, pi_idx].max().item()) >= early_stop_threshold
                    or int((cat_scores >= early_stop_threshold).sum(dim=0).max().item()) >= 2
                ):
                    break
        context.timings_ms["encoder_ms"] = (time.perf_counter() - started) * 1000
        return torch.cat(scores), torch.cat(pooled_all)

    def predict_windowed(self, context: TokenContext, **kwargs) -> tuple[dict[str, float], torch.Tensor]:
        """Per-label max over windows (any window that flags flags the request)."""
        scores, pooled = self.window_scores(context, **kwargs)
        return dict(zip(LABELS, scores.max(dim=0).values.tolist())), pooled.mean(dim=0)

    @torch.no_grad()
    def predict_with_embedding(self, text: str | TokenContext) -> tuple[dict[str, float], torch.Tensor]:
        """Same forward pass as predict(), but also returns the pooled
        embedding instead of discarding it. armor/models/policy_head.py
        consumes this directly -- no second tokenize/encoder call. This is
        the fix for the representation bottleneck documented in
        queue/QUEUE-02: every downstream decision-maker (bandit, semantic
        judge) was trying to reconstruct semantic nuance from 6 scalar
        scores alone, after the actual semantic representation had already
        been computed and thrown away right here."""
        # ModernBERT supports 8192 tokens. The old 512-token cap silently
        # discarded the final request in long conversations; never inspect
        # only a prefix while reporting a decision for the entire input.
        context = text if isinstance(text, TokenContext) else self.prepare_token_context(text)
        if context.oversize:
            raise InputTooLongError(
                context.oversize_reason or "input exceeds inspection limits",
                reason=context.oversize_reason or "input_exceeds_model_context",
            )
        inputs = context.inputs
        if self.backend != "onnx":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        encoder_started = time.perf_counter()
        pooled = self.encoder(**inputs).last_hidden_state[:, 0, :]
        context.timings_ms["encoder_ms"] = (time.perf_counter() - encoder_started) * 1000
        if pooled.device != self.head.weight.device:
            pooled = pooled.to(self.head.weight.device)
        head_started = time.perf_counter()
        scores = torch.sigmoid(self.head(pooled)).squeeze(0).tolist()
        context.timings_ms["classifier_head_ms"] = (time.perf_counter() - head_started) * 1000
        return dict(zip(LABELS, scores)), pooled.squeeze(0)


@lru_cache(maxsize=1)
def get_armor_base() -> ArmorBase:
    return ArmorBase()
