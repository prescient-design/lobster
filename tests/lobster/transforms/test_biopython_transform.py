from torch import Tensor
import torch
from lobster.transforms import ProteinToBioPythonFeaturesTransform
import pytest


class TestProteinToBioPythonFeaturesTransform:
    def test_initialization(self):
        transform = ProteinToBioPythonFeaturesTransform()
        assert transform is not None

    def test_transform(self):
        transform = ProteinToBioPythonFeaturesTransform()
        out = transform("MALWMRLLPLLALLALWGPDPAAA")
        assert out is not None
        assert isinstance(out, Tensor)
        assert out.shape == (13,)
        assert torch.allclose(
            out,
            torch.tensor(
                [
                    2.4000e01,
                    2.6042e03,
                    8.3333e-02,
                    4.9771e01,
                    5.5878e00,
                    6.6667e-01,
                    2.0833e-01,
                    4.1667e-01,
                    1.1000e04,
                    1.1000e04,
                    1.2500e00,
                    -7.6279e-02,
                    -4.9853e-01,
                ]
            ),
            atol=1e-4,
        )

    def test_transform_order(self):
        transform = ProteinToBioPythonFeaturesTransform(feature_list=["molecular_weight", "sequence_length"])
        out = transform("MALWMRLLPLLALLALWGPDPAAA")
        assert out is not None
        assert isinstance(out, Tensor)
        assert out.shape == (2,)
        assert torch.allclose(out, torch.tensor([2604.2253, 24.0]), atol=1e-4)

    def test_transform_with_return_dict(self):
        transform = ProteinToBioPythonFeaturesTransform(return_dict=True)
        out = transform("MALWMRLLPLLALLALWGPDPAAA")
        assert isinstance(out, dict)
        assert len(out) == 13
        assert out == {
            "sequence_length": 24.0,
            "molecular_weight": 2604.2252999999996,
            "aromaticity_index": 0.08333333333333334,
            "instability_index": 49.77083333333333,
            "isoelectric_point": 5.587750434875488,
            "alpha_helix_fraction": 0.6666666666666667,
            "turn_structure_fraction": 0.20833333333333334,
            "beta_sheet_fraction": 0.41666666666666674,
            "molar_extinction_coefficient_reduced_cysteines": 11000.0,
            "molar_extinction_coefficient_oxidized_cysteines": 11000.0,
            "grand_average_hydropathy_index": 1.25,
            "net_charge_at_ph_6": -0.07627881289313532,
            "net_charge_at_ph_7": -0.49853455141881153,
        }

    def test_transform_with_return_dict_complex(self):
        transform = ProteinToBioPythonFeaturesTransform(return_dict=True, standardize=False)
        out = transform("MALWMRLLPLL.ALLALWGPDPAAA")
        assert isinstance(out, dict)
        assert len(out) == 13
        assert out == {
            "sequence_length": 24.0,
            # MW for the complex is slightly different than for 1 chain (the test above)
            # this is because when you have separate chains, each chain has its own N-terminus (NH₂) and C-terminus (COOH)
            # when chains are connected in a single peptide, the internal peptide bonds eliminate water
            # The difference = ~ weight of 1 water molecule
            "molecular_weight": 2622.2406,
            "aromaticity_index": 0.08391608391608392,
            "instability_index": 51.251398601398606,
            "isoelectric_point": 6.7750528335571305,
            "alpha_helix_fraction": 0.6713286713286712,
            "turn_structure_fraction": 0.19930069930069932,
            "beta_sheet_fraction": 0.42657342657342656,
            "molar_extinction_coefficient_reduced_cysteines": 11000.0,
            "molar_extinction_coefficient_oxidized_cysteines": 11000.0,
            "grand_average_hydropathy_index": 1.2769230769230768,
            "net_charge_at_ph_6": -0.09780304498386949,
            "net_charge_at_ph_7": -0.7026599635647247,
        }

    def test_transform_with_standardization(self):
        transform = ProteinToBioPythonFeaturesTransform(standardize=True)
        out = transform("MALWMRLLPLLALLALWGPDPAAA")
        assert out is not None
        assert isinstance(out, Tensor)

        assert torch.allclose(
            out,
            torch.tensor(
                [
                    0.9070,
                    0.8782,
                    0.1171,
                    0.1854,
                    -0.2906,
                    2.4211,
                    -0.7331,
                    0.7475,
                    3.6618,
                    3.6573,
                    2.3006,
                    0.2066,
                    0.2249,
                ]
            ),
            atol=1e-4,
        )

    def test_transform_with_standardization_and_peptide_threshold(self):
        transform = ProteinToBioPythonFeaturesTransform(standardize=True, peptide_threshold=2)
        out = transform("MALWMRLLPLLALLALWGPDPAAA")
        assert out is not None
        assert isinstance(out, Tensor)
        assert torch.allclose(
            out,
            torch.tensor(
                [
                    -1.2239,
                    -1.2341,
                    0.0874,
                    1.0910,
                    -0.6812,
                    6.6324,
                    -1.4228,
                    0.8423,
                    -0.7527,
                    -0.7554,
                    3.1770,
                    -0.0387,
                    0.2204,
                ]
            ),
            atol=1e-4,
        )

    def test_invalid_input(self):
        transform = ProteinToBioPythonFeaturesTransform()
        out = transform("@#$Z")
        assert out is None

    def test_transform_invalid_feature(self):
        with pytest.raises(ValueError):
            ProteinToBioPythonFeaturesTransform(standardize=True, feature_list=["invalid_feature"])
