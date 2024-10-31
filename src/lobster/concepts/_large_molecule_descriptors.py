from dataclasses import dataclass, field
from typing import Optional

from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData

from ._descriptors import Descriptors


def _get_avg_quantity(sequence: str, lookup: dict[str, float]) -> float:
    return sum(lookup.get(aa, 0.0) for aa in sequence) / len(sequence)


@dataclass
class LargeMoleculeDescriptors(Descriptors):
    sequence: str = field(metadata={"not_a_descriptor": True})
    _protein_analysis: ProteinAnalysis = field(init=False, metadata={"not_a_descriptor": True})
    length: int = field(init=False)

    molecular_weight: float = field(init=False)
    aromaticity: float = field(init=False)
    instability_index: float = field(init=False)
    isoelectric_point: float = field(init=False)
    gravy: float = field(init=False)
    charge_at_pH6: float = field(init=False)
    charge_at_pH7: float = field(init=False)

    helix_fraction: float = field(init=False)
    turn_structure_fraction: float = field(init=False)
    sheet_structure_fraction: float = field(init=False)

    molar_extinction_coefficient_reduced: float = field(init=False)
    molar_extinction_coefficient_oxidized: float = field(init=False)

    avg_hydrophilicity: float = field(init=False)
    avg_surface_accessibility: float = field(init=False)

    def __post_init__(self):
        self.length = len(self.sequence)
        self._protein_analysis = ProteinAnalysis(self.sequence)
        self.molecular_weight = self._protein_analysis.molecular_weight()
        self.aromaticity = self._protein_analysis.aromaticity()
        self.instability_index = self._protein_analysis.instability_index()
        self.isoelectric_point = self._protein_analysis.isoelectric_point()

        self.gravy = self._protein_analysis.gravy()
        self.charge_at_pH6 = self._protein_analysis.charge_at_pH(6)
        self.charge_at_pH7 = self._protein_analysis.charge_at_pH(7)
        (
            self.helix_fraction,
            self.turn_structure_fraction,
            self.sheet_structure_fraction,
        ) = self._protein_analysis.secondary_structure_fraction()
        (
            self.molar_extinction_coefficient_reduced,
            self.molar_extinction_coefficient_oxidized,
        ) = self._protein_analysis.molar_extinction_coefficient()

        self.avg_hydrophilicity = _get_avg_quantity(self.sequence, ProtParamData.hw)
        self.avg_surface_accessibility = _get_avg_quantity(self.sequence, ProtParamData.em)

    @classmethod
    def from_sequence(cls, sequence: str) -> Optional["LargeMoleculeDescriptors"]:
        if len(sequence) > 0:
            return LargeMoleculeDescriptors(sequence)
        else:
            return None
