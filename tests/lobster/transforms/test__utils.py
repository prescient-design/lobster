import importlib.resources
from typing import Dict, List

import pytest
from lobster.transforms import (
    invert_residue_to_codon_mapping,
    json_load,
    sample_list_with_probs,
    uniform_sample,
)


@pytest.fixture(scope="class")
def residue_to_codon() -> Dict[str, List[str]]:
    path = importlib.resources.files("lobster") / "assets" / "codon_tables" / "codon_table.json"
    data = json_load(path)
    return data


class TestUtils:
    def test_uniform_sample(self):
        vals = ["abd", "cbe", "kfi", "ahf", "foh"]
        sampled = uniform_sample(vals)
        assert (sampled in vals) and (isinstance(sampled, str))
        sampled_vals = []
        for _ in range(1000):
            sampled_vals.append(uniform_sample(vals))
        assert len(set(sampled_vals)) > 1

    def test_sample_list_with_probs(self):
        vals = ["abd", "cbe", "kfi", "ahf", "foh"]
        probs = [0.1, 0.35, 0.3, 0.1, 0.15]
        sampled = sample_list_with_probs(vals, probs)
        assert (sampled in vals) and (isinstance(sampled, str))
        sampled_vals = []
        for _ in range(1000):
            sampled_vals.append(uniform_sample(vals))
        assert len(set(sampled_vals)) > 1

    def test_invert_dict(self, residue_to_codon):
        codon_to_residue = invert_residue_to_codon_mapping(residue_to_codon)
        assert isinstance(residue_to_codon, dict)
        all_codons = [i for v in residue_to_codon.values() for i in v]
        assert sorted(codon_to_residue.keys()) == sorted(all_codons)
        assert sorted(set(codon_to_residue.values())) == sorted(residue_to_codon.keys())
