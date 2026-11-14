from functools import lru_cache

import spacy

MODEL_NAME = "en_core_web_sm"

# DLP-6 (framework doc section 3): unstructured PII has no fixed format,
# so it needs a learned specialist rather than regex. Maps spaCy's own
# entity labels onto the two unstructured-PII classes the taxonomy doc cares
# about; everything else (ORG, DATE, MONEY, ...) is ignored.
_ENTITY_LABEL_MAP = {
    "PERSON": "person_name",
    "GPE": "address",
    "LOC": "address",
    "FAC": "address",
}


class PiiNerSpecialist:
    def __init__(self, model_name: str = MODEL_NAME):
        self.nlp = spacy.load(model_name)

    def predict(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        findings = []
        for ent in doc.ents:
            entity_id = _ENTITY_LABEL_MAP.get(ent.label_)
            if entity_id is None:
                continue
            findings.append({"entity_id": entity_id, "evidence": ent.text})
        return findings


@lru_cache(maxsize=1)
def get_pii_ner() -> PiiNerSpecialist:
    return PiiNerSpecialist()
