from armor.observability import review_store


def test_add_sample_and_record_decisions(tmp_path):
    db = tmp_path / "review.db"
    review_store.add_sample(
        db, sample_id="s1", content="test content", existing_label="flag",
        label_source_at_audit="pseudo_model", rubric_version="v1",
    )
    review_store.record_reviewer_decision(db, "s1", "a", "flag", "dual_use_low_risk")
    samples = review_store.all_samples(db)
    assert len(samples) == 1
    assert samples[0]["reviewer_a_decision"] == "flag"
    assert samples[0]["reviewer_a_id"] is None
    assert samples[0]["agreement"] is None  # b not recorded yet


def test_reviewer_identity_is_stored_with_decision(tmp_path):
    db = tmp_path / "review.db"
    review_store.add_sample(
        db, sample_id="s1", content="test", existing_label="allow",
        label_source_at_audit="pseudo_model", rubric_version="v1.2",
        reviewer_a_id="reviewer_ai_jev", reviewer_b_id="reviewer_ai_blind_b",
    )
    review_store.record_reviewer_decision(
        db, "s1", "a", "flag", reviewer_id="reviewer_ai_jev"
    )
    review_store.record_reviewer_decision(
        db, "s1", "b", "allow", reviewer_id="reviewer_ai_blind_b"
    )
    row = review_store.all_samples(db)[0]
    assert row["reviewer_a_id"] == "reviewer_ai_jev"
    assert row["reviewer_b_id"] == "reviewer_ai_blind_b"
    assert row["agreement"] == 0

    # Retrying the same write is safe; replacing a stored reviewer output is not.
    review_store.record_reviewer_decision(
        db, "s1", "a", "flag", reviewer_id="reviewer_ai_jev"
    )
    try:
        review_store.record_reviewer_decision(db, "s1", "a", "allow")
        assert False, "stored reviewer decisions must be immutable"
    except ValueError:
        pass


def test_init_db_expands_legacy_schema_without_losing_decisions(tmp_path):
    import sqlite3

    db = tmp_path / "legacy.db"
    legacy_schema = review_store._SCHEMA.replace("    reviewer_a_id TEXT,\n", "").replace(
        "    reviewer_b_id TEXT,\n", ""
    )
    with sqlite3.connect(db) as conn:
        conn.execute(legacy_schema)
        conn.execute(
            "INSERT INTO review_audit (sample_id, reviewer_a_decision, reviewer_b_decision, agreement) "
            "VALUES ('legacy-1', 'allow', 'flag', 0)"
        )

    review_store.init_db(db)
    row = review_store.all_samples(db)[0]
    assert row["reviewer_a_decision"] == "allow"
    assert row["reviewer_b_decision"] == "flag"
    assert row["reviewer_a_id"] is None
    assert row["reviewer_b_id"] is None


def test_agreement_computed_once_both_reviewers_present(tmp_path):
    db = tmp_path / "review.db"
    review_store.add_sample(
        db, sample_id="s1", content="c", existing_label="flag",
        label_source_at_audit="pseudo_model", rubric_version="v1",
    )
    review_store.record_reviewer_decision(db, "s1", "a", "flag")
    review_store.record_reviewer_decision(db, "s1", "b", "block")
    samples = review_store.all_samples(db)
    assert samples[0]["agreement"] == 0


def test_agreement_true_when_reviewers_match(tmp_path):
    db = tmp_path / "review.db"
    review_store.add_sample(
        db, sample_id="s1", content="c", existing_label="allow",
        label_source_at_audit="pseudo_model", rubric_version="v1",
    )
    review_store.record_reviewer_decision(db, "s1", "a", "allow")
    review_store.record_reviewer_decision(db, "s1", "b", "allow")
    samples = review_store.all_samples(db)
    assert samples[0]["agreement"] == 1


def test_invalid_reviewer_raises(tmp_path):
    db = tmp_path / "review.db"
    review_store.add_sample(
        db, sample_id="s1", content="c", existing_label="allow",
        label_source_at_audit="pseudo_model", rubric_version="v1",
    )
    try:
        review_store.record_reviewer_decision(db, "s1", "c", "allow")
        assert False, "should have raised"
    except ValueError:
        pass


def test_list_disagreements_excludes_adjudicated(tmp_path):
    db = tmp_path / "review.db"
    review_store.add_sample(
        db, sample_id="s1", content="c", existing_label="flag",
        label_source_at_audit="pseudo_model", rubric_version="v1",
    )
    review_store.record_reviewer_decision(db, "s1", "a", "flag")
    review_store.record_reviewer_decision(db, "s1", "b", "block")
    assert len(review_store.list_disagreements(db)) == 1

    review_store.record_adjudication(db, "s1", "flag")
    assert len(review_store.list_disagreements(db)) == 0


def test_adjudication_does_not_overwrite_reviewer_decisions(tmp_path):
    db = tmp_path / "review.db"
    review_store.add_sample(
        db, sample_id="s1", content="c", existing_label="flag",
        label_source_at_audit="pseudo_model", rubric_version="v1",
    )
    review_store.record_reviewer_decision(db, "s1", "a", "flag")
    review_store.record_reviewer_decision(db, "s1", "b", "block")
    review_store.record_adjudication(db, "s1", "block")

    samples = review_store.all_samples(db)
    assert samples[0]["reviewer_a_decision"] == "flag"
    assert samples[0]["reviewer_b_decision"] == "block"
    assert samples[0]["adjudicated_label"] == "block"
