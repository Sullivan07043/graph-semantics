import inspect

import numpy as np
import pytest

from v6.baselines import clip_dissect_e5 as td


def _toy_encoder(texts):
    vectors = {
        "very outgoing": (1.0, 0.0),
        "social and talkative": (0.9, 0.1),
        "quiet and reflective": (-0.9, 0.1),
        "very reserved": (-1.0, 0.0),
        "extraversion": (1.0, 0.0),
        "introversion": (-1.0, 0.0),
        "abstract reasoning": (0.0, 1.0),
        "an invalid construct name with five words": (0.7, 0.7),
    }
    return np.asarray([vectors[text] for text in texts], dtype=np.float64)


def _toy_config():
    return td.TextDissectConfig(
        top_k=3,
        soft_top_k=2,
        rank_top_fraction=0.5,
        concept_batch_size=2,
        rank_seed=17,
    )


def test_names_both_activation_poles_with_injected_encoder_deterministically():
    concepts = [
        "extraversion",
        "introversion",
        "abstract reasoning",
        "an invalid construct name with five words",
    ]
    bank = td.build_concept_bank(concepts, encoder=_toy_encoder, batch_size=2)
    profiles = [
        "very outgoing",
        "social and talkative",
        "quiet and reflective",
        "very reserved",
    ]
    activations = {"L0": np.array([3.0, 2.0, -2.0, -3.0])}

    first = td.text_dissect(
        profiles, activations, bank, encoder=_toy_encoder, config=_toy_config()
    )
    second = td.run_text_dissect(
        profiles, activations, bank, encoder=_toy_encoder, config=_toy_config()
    )

    assert [result.to_dict() for result in first] == [result.to_dict() for result in second]
    assert first[0].construct_name == "extraversion"
    assert first[0].negative_pole.construct_name == "introversion"
    assert all(len(item.name.split()) <= 4 for item in first[0].top_concepts)
    assert first[0].positive_pole.soft_profile_indices == (0, 1)
    assert first[0].negative_pole.soft_profile_indices == (3, 2)
    assert "gold" not in inspect.signature(td.text_dissect).parameters


def test_accepts_existing_dictionary_tuple_and_preencoded_profiles():
    names = ("extraversion", "introversion", "abstract reasoning")
    embeddings = _toy_encoder(names)
    profile_embeddings = _toy_encoder(
        ["very outgoing", "social and talkative", "quiet and reflective", "very reserved"]
    )
    results = td.text_dissect(
        None,
        np.array([3.0, 2.0, -2.0, -3.0]),
        (embeddings, names),
        profile_embeddings=profile_embeddings,
        latent_ids=["latent-node-7"],
        config=_toy_config(),
    )
    payload = results[0].to_dict()
    assert payload["latent_id"] == "latent-node-7"
    assert payload["construct_name"] == "extraversion"
    assert payload["negative_construct_name"] == "introversion"
    assert set(payload["native_diagnostics"]).issuperset(
        {"positive_soft_wpmi", "positive_rank_reorder", "positive_rank_agreement_at_k"}
    )


def test_npz_loader_checks_schema_and_encoder_metadata(tmp_path):
    good = tmp_path / "bank.npz"
    np.savez(
        good,
        emb=np.eye(2, dtype=np.float32),
        names=np.asarray(["alpha", "beta"], dtype=object),
        encoder=np.asarray("intfloat/e5-large-v2"),
    )
    bank = td.load_concept_bank(good, expected_encoder="e5-large-v2")
    assert bank.names == ("alpha", "beta")
    assert bank.metadata["encoder"] == "intfloat/e5-large-v2"

    with pytest.raises(ValueError, match="encoder mismatch"):
        td.load_concept_bank(good, expected_encoder="gte-large")

    adapted = tmp_path / "adapted.npz"
    np.savez(
        adapted,
        emb=np.eye(2, dtype=np.float32),
        names=np.asarray(["alpha", "beta"], dtype=object),
        encoder=np.asarray("intfloat/e5-large-v2"),
        lora_checkpoint_sha256=np.asarray("deadbeef"),
    )
    with pytest.raises(ValueError, match="LoRA checkpoint"):
        td.load_concept_bank(adapted, expected_encoder="e5-large-v2")
    assert td.load_concept_bank(adapted, allow_adapted_encoder=True).names == ("alpha", "beta")

    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, emb=np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="missing required arrays.*names"):
        td.load_concept_bank(malformed)


def test_errors_are_explicit_for_incompatible_inputs(tmp_path):
    bank = td.ConceptBank(np.eye(2), ("alpha", "beta"))
    with pytest.raises(FileNotFoundError, match="concept bank not found"):
        td.load_concept_bank(tmp_path / "missing.npz")
    with pytest.raises(ValueError, match="constant activations"):
        td.text_dissect(
            None,
            np.ones(3),
            bank,
            profile_embeddings=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float),
        )
    with pytest.raises(ValueError, match="dimension mismatch"):
        td.text_dissect(
            None,
            np.asarray([1.0, -1.0]),
            bank,
            profile_embeddings=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float),
        )


def test_rank_stability_supports_objects_and_cached_dicts():
    bank = td.build_concept_bank(
        ["extraversion", "introversion", "abstract reasoning"], encoder=_toy_encoder
    )
    profiles = [
        "very outgoing",
        "social and talkative",
        "quiet and reflective",
        "very reserved",
    ]
    result = td.text_dissect(
        profiles,
        [3.0, 2.0, -2.0, -3.0],
        bank,
        encoder=_toy_encoder,
        config=_toy_config(),
    )[0]
    assert td.rank_stability([result, result], k=2) == 1.0
    assert td.rank_stability([result.to_dict(), result.to_dict()], pole="negative", k=2) == 1.0


def _pole_from_named_scores(names, soft_by_name, reorder_by_name, *, top_k=3):
    soft = np.asarray([[soft_by_name[name] for name in names]], dtype=float)
    reorder = np.asarray([[reorder_by_name[name] for name in names]], dtype=float)
    return td._make_pole_result(
        "positive",
        0,
        names,
        np.arange(len(names)),
        soft,
        reorder,
        [np.asarray([0, 1])],
        [np.asarray([0, 1])],
        td.TextDissectConfig(top_k=top_k, soft_top_k=2, rank_top_fraction=0.5),
    )


def test_all_equal_rank_reorder_scores_have_equal_midrank_contribution():
    quality, ranks = td._rank_quality(np.zeros(4))
    np.testing.assert_allclose(quality, np.full(4, 0.5))
    np.testing.assert_allclose(ranks, np.full(4, 2.5))

    quality, ranks = td._rank_quality(np.asarray([3.0, 3.0, 1.0]))
    np.testing.assert_allclose(ranks, np.asarray([1.5, 1.5, 3.0]))
    assert quality[0] == quality[1] > quality[2]


def test_tied_rank_reorder_is_bank_permutation_invariant_and_threshold_neutral():
    soft = {"zeta": 0.1, "alpha": 0.9, "beta": 0.5}
    tied_reorder = {name: 0.0 for name in soft}
    first = _pole_from_named_scores(
        ["zeta", "alpha", "beta"], soft, tied_reorder, top_k=1
    )
    second = _pole_from_named_scores(
        ["beta", "zeta", "alpha"], soft, tied_reorder, top_k=1
    )

    assert first.construct_name == second.construct_name == "alpha"
    assert first.top_concepts[0].combined_score == second.top_concepts[0].combined_score
    assert first.top_concepts[0].rank_reorder_rank == 2.0
    # The rank scorer's top-1 threshold contains the complete three-way tie;
    # it does not arbitrarily select concept-bank row zero.
    assert first.rank_agreement_at_k == second.rank_agreement_at_k == 1 / 3


def test_soft_wpmi_ties_and_complete_metric_ties_do_not_leak_bank_order():
    names_a = ["zeta", "alpha", "beta"]
    names_b = ["beta", "zeta", "alpha"]
    tied_soft = {name: 0.0 for name in names_a}
    reorder = {"zeta": -0.8, "alpha": -0.1, "beta": -0.4}

    first = _pole_from_named_scores(names_a, tied_soft, reorder)
    second = _pole_from_named_scores(names_b, tied_soft, reorder)
    assert [item.name for item in first.top_concepts] == [
        item.name for item in second.top_concepts
    ] == ["alpha", "beta", "zeta"]
    assert all(item.soft_wpmi_rank == 2.0 for item in first.top_concepts)

    tied_reorder = {name: 0.0 for name in names_a}
    all_tied_a = _pole_from_named_scores(names_a, tied_soft, tied_reorder)
    all_tied_b = _pole_from_named_scores(names_b, tied_soft, tied_reorder)
    assert [item.name for item in all_tied_a.top_concepts] == [
        item.name for item in all_tied_b.top_concepts
    ] == ["alpha", "beta", "zeta"]
