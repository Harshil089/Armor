from armor.data.synthetic_pii import (
    _gen_aadhaar,
    _gen_credit_card,
    _gen_iban,
    generate_examples,
)
from armor.detectors.validators import VALIDATORS
from armor.models.labels import LABELS

_SDE_INDEX = LABELS.index("sensitive_data_exposure")


def test_generated_card_passes_real_validator():
    assert VALIDATORS["payment_card"](_gen_credit_card())


def test_generated_iban_passes_real_validator():
    assert VALIDATORS["iban_mod97"](_gen_iban())


def test_generated_aadhaar_passes_real_validator():
    assert VALIDATORS["verhoeff"](_gen_aadhaar())


def test_generate_examples_shape_and_labels():
    examples = generate_examples(n_positive=20, n_negative=20)
    assert len(examples) == 40

    positives = [e for e in examples if e[1][_SDE_INDEX] == 1.0]
    negatives = [e for e in examples if e[1][_SDE_INDEX] == 0.0]
    assert len(positives) == 20
    assert len(negatives) == 20

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert mask_vec[_SDE_INDEX] == 1.0  # this label is always supervised
        assert sum(mask_vec) == 1.0  # every other label stays unknown
