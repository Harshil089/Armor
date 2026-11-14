import torch
import sqlite3

import armor.models.train_policy as train_policy_module
from armor.models.train_policy import _hierarchical_labels, load_training_rows, train_hierarchical
from armor.observability import log_store


def test_hierarchical_labels_allow():
    assert _hierarchical_labels("allow") == (0, 0, False)


def test_hierarchical_labels_flag():
    assert _hierarchical_labels("flag") == (1, 0, True)


def test_hierarchical_labels_block():
    assert _hierarchical_labels("block") == (1, 1, True)


def test_hierarchical_labels_unresolved_review_excluded():
    assert _hierarchical_labels("review") is None


def test_training_rows_default_to_trusted_valid_labels(tmp_path):
    db = tmp_path / "logs.db"
    log_store.init_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO logs (content, analyst_correction, label_source) VALUES ('legacy trusted', 'allow', 'human')"
    )
    conn.execute(
        "INSERT INTO logs (content, analyst_correction, label_source) VALUES ('single model', 'flag', 'pseudo_model')"
    )
    conn.execute(
        "INSERT INTO logs (content, analyst_correction, label_source) VALUES ('dirty marker', 'stale-6dim', 'pseudo_model')"
    )
    conn.execute(
        "INSERT INTO logs (content, decision_correction, decision_label_source) "
        "VALUES ('AI consensus', 'review', 'reviewer_consensus')"
    )
    conn.execute(
        "INSERT INTO logs (content, decision_correction, decision_label_source) "
        "VALUES ('Jev review', 'block', 'reviewer_ai_b')"
    )
    conn.commit()
    conn.close()

    assert set(load_training_rows(db)) == {
        ("legacy trusted", "allow", "human"),
        ("AI consensus", "review", "reviewer_consensus"),
    }
    assert set(load_training_rows(db, include_pseudo_model=True)) == {
        ("legacy trusted", "allow", "human"),
        ("single model", "flag", "pseudo_model"),
        ("AI consensus", "review", "reviewer_consensus"),
        ("Jev review", "block", "reviewer_ai_b"),
    }
    assert set(load_training_rows(db, include_jev_reviews=True)) == {
        ("legacy trusted", "allow", "human"),
        ("AI consensus", "review", "reviewer_consensus"),
        ("Jev review", "block", "reviewer_ai_b"),
    }


def test_ai_reviewer_consensus_is_provisional_for_training(tmp_path):
    db = tmp_path / "logs.db"
    log_store.log_entry(
        db, source="inspect", content="AI-only consensus", decision="allow",
        policy_action="flag", bandit_action="review",
    )
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE logs SET decision_correction='flag', decision_label_source='reviewer_ai_consensus'"
    )
    conn.commit()
    conn.close()

    assert load_training_rows(db) == []
    assert load_training_rows(db, include_pseudo_model=True) == [
        ("AI-only consensus", "flag", "reviewer_ai_consensus")
    ]


class _FakeModel:
    def predict_with_embedding(self, text: str):
        # Deterministic per-text embedding so allow/flag/block map to
        # separable regions -- proves the training loop actually learns
        # something, not just that it runs without erroring.
        seed = {"safe text": 0.0, "mild text": 1.0, "bad text": 2.0}[text]
        embedding = torch.full((8,), seed)
        scores = {"prompt_injection": 0.0, "jailbreak": 0.0, "sensitive_data_exposure": 0.0,
                   "dangerous_tool_action": 0.0, "harmful_content": 0.0, "obfuscation_encoding": 0.0}
        return scores, embedding


def test_train_hierarchical_excludes_review_rows_from_training(monkeypatch, capsys):
    monkeypatch.setattr(train_policy_module, "get_armor_base", lambda: _FakeModel())
    monkeypatch.setattr(
        train_policy_module,
        "load_training_rows",
        lambda **_kwargs: [
            ("safe text", "allow", "pseudo_model"),
            ("mild text", "flag", "pseudo_model"),
            ("bad text", "block", "pseudo_model"),
            ("safe text", "review", "pseudo_model"),  # must be excluded, not trained on
        ],
    )
    train_hierarchical(epochs=1)
    captured = capsys.readouterr()
    assert "1 unresolved 'review' rows excluded" in captured.out
    assert "hierarchical training rows: 3" in captured.out
