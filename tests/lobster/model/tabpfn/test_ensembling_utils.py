import pytest
import numpy as np

from lobster.model.tabpfn._ensembling_utils import (
    check_pretraining_limits,
    validate_limits,
    create_feature_subsets,
    create_sample_chunks,
)
from lobster.constants import MAX_FEATURES_PER_MODEL


def test_check_pretraining_limits_within():
    exceeds_feat, exceeds_samp = check_pretraining_limits(100, 400)
    assert not exceeds_feat
    assert not exceeds_samp


def test_check_pretraining_limits_exceeds_features():
    exceeds_feat, exceeds_samp = check_pretraining_limits(100, 600)
    assert exceeds_feat
    assert not exceeds_samp


def test_check_pretraining_limits_exceeds_samples():
    exceeds_feat, exceeds_samp = check_pretraining_limits(60000, 400)
    assert not exceeds_feat
    assert exceeds_samp


def test_check_pretraining_limits_exceeds_both():
    exceeds_feat, exceeds_samp = check_pretraining_limits(60000, 600)
    assert exceeds_feat
    assert exceeds_samp


def test_validate_limits_no_exceeding():
    needs_ensemble = validate_limits(100, 400, True, True)
    assert not needs_ensemble


def test_validate_limits_with_ensembling_enabled():
    needs_ensemble = validate_limits(60000, 600, True, True)
    assert needs_ensemble


def test_validate_limits_exceeds_without_ensembling_raises():
    with pytest.raises(ValueError, match="Dataset exceeds TabPFN pretraining limits"):
        validate_limits(60000, 600, False, False)


def test_validate_limits_exceeds_without_ensembling_but_ignore():
    needs_ensemble = validate_limits(60000, 600, False, True)
    assert not needs_ensemble


def test_create_feature_subsets_within_limit():
    subsets = create_feature_subsets(400, seed=42)
    assert len(subsets) == 1
    assert len(subsets[0]) == 400


def test_create_feature_subsets_exceeds_limit():
    subsets = create_feature_subsets(1200, seed=42)
    assert len(subsets) == 3
    for subset in subsets:
        assert len(subset) == MAX_FEATURES_PER_MODEL


def test_create_feature_subsets_deterministic():
    subsets1 = create_feature_subsets(800, seed=42)
    subsets2 = create_feature_subsets(800, seed=42)

    assert len(subsets1) == len(subsets2)
    for s1, s2 in zip(subsets1, subsets2):
        assert np.array_equal(s1, s2)


def test_create_feature_subsets_different_seeds():
    subsets1 = create_feature_subsets(800, seed=42)
    subsets2 = create_feature_subsets(800, seed=99)

    assert len(subsets1) == len(subsets2)
    assert not np.array_equal(subsets1[0], subsets2[0])


def test_create_sample_chunks_within_limit():
    chunks = create_sample_chunks(30000)
    assert len(chunks) == 1
    assert chunks[0] == (0, 30000)


def test_create_sample_chunks_exceeds_limit():
    chunks = create_sample_chunks(120000)
    assert len(chunks) == 3
    assert chunks[0] == (0, 50000)
    assert chunks[1] == (50000, 100000)
    assert chunks[2] == (100000, 120000)


def test_create_sample_chunks_exact_limit():
    chunks = create_sample_chunks(100000)
    assert len(chunks) == 2
    assert chunks[0] == (0, 50000)
    assert chunks[1] == (50000, 100000)
