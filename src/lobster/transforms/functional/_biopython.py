from Bio.SeqUtils.ProtParam import ProteinAnalysis


def get_biopython_features(seq) -> dict:
    """
    input: str
        Amino acid sequence.
    output: dict of biopython features
    """
    chain = ProteinAnalysis(seq)
    mec = chain.molar_extinction_coefficient()
    ssf = chain.secondary_structure_fraction()

    feats = {
        "length": len(seq),
        "molecular_weight": chain.molecular_weight(),
        "aromaticity": chain.aromaticity(),
        "instability_index": chain.instability_index(),
        "isoelectric_point": chain.isoelectric_point(),
        "helix_fraction": ssf[0],
        "turn_structure_fraction": ssf[1],
        "sheet_structure_fraction": ssf[2],
        "molar_extinction_coefficient_reduced": mec[0],
        "molar_extinction_coefficient_oxidized": mec[1],
        "gravy": chain.gravy(),
        "charge_at_pH6": chain.charge_at_pH(6),
        "charge_at_pH7": chain.charge_at_pH(7),
    }

    return feats
