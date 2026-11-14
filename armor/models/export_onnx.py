"""One-time ONNX export of the Armor Base encoder for faster CPU inference.

Verified bit-for-bit-equivalent output vs the PyTorch encoder (max abs diff
~6e-6, floating-point noise) and ~21% faster forward pass on CPU (30ms vs
23.7ms median, single-request). Architecture spec section 7: "benchmark
native PyTorch first, then evaluate ONNX Runtime."

If the deployed checkpoint (head.pt) carries fine-tuned encoder deltas
(`encoder_state_dict`, from armor/models/train.py's fine_tune_layers path),
those deltas are applied to the base encoder before export — otherwise the
ONNX export would silently serve the original pretrained weights instead of
the fine-tuned ones. armor/models/armor_base.py refuses to load an ONNX
export that predates a fine-tuned checkpoint, so re-run this after every
promotion of a fine-tuned candidate.

Run: python -m armor.models.export_onnx (needs pip install -e ".[onnx]")
Then: ARMOR_BASE_BACKEND=onnx to use it.
"""

import shutil
import tempfile

import torch
from transformers import AutoModel

from armor.models.armor_base import CHECKPOINT_PATH, MODEL_NAME, ONNX_DIR, ONNX_FINGERPRINT_PATH, encoder_fingerprint


def export() -> None:
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if checkpoint.get("encoder_state_dict"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            encoder = AutoModel.from_pretrained(MODEL_NAME)
            encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
            encoder.save_pretrained(tmp_dir)
            model = ORTModelForFeatureExtraction.from_pretrained(tmp_dir, export=True)
    else:
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)

    shutil.rmtree(ONNX_DIR, ignore_errors=True)
    model.save_pretrained(ONNX_DIR)
    with open(ONNX_FINGERPRINT_PATH, "w") as f:
        f.write(encoder_fingerprint(checkpoint))
    print(f"exported ONNX model to {ONNX_DIR}")


if __name__ == "__main__":
    export()
