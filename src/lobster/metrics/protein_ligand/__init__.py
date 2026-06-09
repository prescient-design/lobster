"""Protein-ligand evaluators for the LeFlur model family.

This subpackage groups the evaluators, ablations, and baselines that operate on
protein-ligand complexes. The three main evaluator classes are used by
:mod:`lobster.cmdline._ligand_conditioned_runner` (and the Hydra entry point
:mod:`lobster.cmdline.generate`) to drive the three Tier-1 ligand-conditioned
modes:

* :class:`LigandConditionedProteinGenerationEvaluator` --
  de-novo protein generation conditioned on a ligand pocket.
* :class:`ProteinLigandForwardFoldingEvaluator` --
  sequence + ligand -> structure prediction (forward folding).
* :class:`ProteinLigandInverseFoldingEvaluator` --
  structure + ligand -> sequence design (inverse folding).

Two ablation scripts (``ablation_*_on_protein_only.py``) and one competitor
baseline (``baseline_ligandmpnn.py``) live alongside the evaluators but are
invoked as standalone ``python -m`` scripts and are not auto-imported.
"""

from .forward_folding import ProteinLigandForwardFoldingEvaluator
from .inverse_folding import ProteinLigandInverseFoldingEvaluator
from .ligand_conditioned_generation import LigandConditionedProteinGenerationEvaluator
from .baseline_ligandmpnn import LigandMPNNInverseFoldingBaselineEvaluator

__all__ = [
    "LigandConditionedProteinGenerationEvaluator",
    "ProteinLigandForwardFoldingEvaluator",
    "ProteinLigandInverseFoldingEvaluator",
    "LigandMPNNInverseFoldingBaselineEvaluator",
]
