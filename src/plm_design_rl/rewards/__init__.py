"""Reward terms for protein-design RL post-training.

This subpackage collects the reward oracles and shaping terms used by the GRPO
trainer, kept separate from the trainer loop and the policy/adapter utilities. The
final reward is a sum of weighted, per-metric-clipped terms:

* :mod:`._protenix_reward` — Protenix co-folding confidence reward client + queue
  protocol (weighted linear combo of ptm/iptm/abag_iptm/plddt/gpde/pae metrics →
  scalar reward and the binder pass criterion),
* :mod:`._structure_reward` — self-consistency TM-scores (Kabsch + TM-score) of the
  generated backbone vs the Protenix-predicted backbone (binder + complex),
* :mod:`._diversity_reward` — within-group k-mer-Jaccard novelty and pairwise-Hamming
  distance (between-design) on the AA sequence and the 3Di structural-token string, plus
  the per-design linguistic-complexity term (within-sequence anti-degeneracy), available
  both as a penalty floor (:func:`lc_floor_penalty`) and as a saturating reward
  (:func:`lc_saturating_reward`).
* :mod:`._distribution_reward` — interface AA + 3Di histogram distance (TV/JS) of a
  design to a per-target reference distribution (Proteina-Complexa binders), a dense
  Protenix-free shaping signal.
* :mod:`._clash_reward` — smooth steric-clash + interface-contact geometry term
  (soft-core over backbone+Cβ atom pairs × soft-count of contacting residues), a
  dense Protenix-free shaping signal that penalizes both overlapping and floating
  binders.
* :mod:`._chainbreak_reward` — backbone chain-break (peptide-bond integrity) term
  (per-bond ``C–N`` realism × catastrophic-break gate) over the generated binder
  backbone, a dense Protenix-free **realism regularizer** that keeps the decoded
  coordinates physically valid so the other structure-track rewards (clash / shape /
  3Di-dist), all measured on that backbone, stay trustworthy.
* :mod:`._sc_clash_reward` — **all-atom** side-chain steric-clash penalty over the
  *whole binder* (binder self-clash + binder↔antigen overlap on LigandMPNN-repacked
  side chains; interface tracked as diagnostic only), a biophysically-grounded
  Protenix-free constraint. Scored on the repacked atom14 clouds served by the CPU
  repack worker pool.
* :mod:`._aar_reward` — LigandMPNN/ProteinMPNN amino-acid-recovery (AAR) over the
  *whole binder* (seq↔structure designability of the design's own backbone; interface
  AAR + C_mpnn tracked as diagnostics only), a Protenix-free consistency signal. The
  ProteinMPNN forward runs in the CPU scoring worker; this module only reduces/maps the
  per-residue results. Offline pass-correlation is weak/anti-predictive — kept opt-in
  (default weight 0), documented rather than gated.
* :mod:`._shape_reward` — order-20 3D-Zernike interface shape-complementarity (SC)
  of ΔSASA contact patches. The full-atom entry point
  (:func:`shape_complementarity_reward_atoms`) is scored on LigandMPNN-repacked
  side-chain atom clouds served by a worker pool (see
  :mod:`plm_design_rl.pool.queue` and ``scripts/ligandmpnn_repack_server.py``); the
  packing step (openfold) stays isolated in the worker so importing this module stays
  pure numpy/scipy.

Each module is pure (no ``trl`` dependency) so the policy side can import the
reward terms without pulling in the reward oracle's heavy deps.
"""

from ._protenix_reward import (
    DEFAULT_CONF_WEIGHTS,
    ProtenixRewardClient,
    confidence_components,
    continuous_ip,
    passes,
    reward_from_confidence,
)
from ._structure_reward import kabsch, structure_terms, tm_score
from ._diversity_reward import (
    coverage,
    hamming_novelty_group,
    jaccard_novelty_group,
    kmer_jaccard,
    lc_floor_penalty,
    lc_saturating_reward,
    linguistic_complexity,
)
from ._distribution_reward import (
    combined_distribution_terms,
    design_hists_scoped,
    design_interface_hists,
    distribution_terms,
    js,
    load_reference_table,
    reference_for,
    tv,
)
from ._clash_reward import clash_contact_reward
from ._chainbreak_reward import chainbreak_reward
from ._sc_clash_reward import (
    binder_clash_terms,
    binder_selfclash,
    cloud_from_atom14,
    interface_allatom_clash,
    sc_clash_reward,
)
from ._aar_reward import (
    SFT_IGNORE_INDEX,
    aar_terms,
    binder_letters_to_aa33,
    interface_residue_mask,
    reward_from_aar,
)
from ._shape_reward import (
    shape_complementarity_reward,
    shape_complementarity_reward_atoms,
)
from ._protenix_structure_expert import (
    INVALID_3DI_STATE,
    assemble_structure_expert,
    build_struct_sft_targets,
    derive_3di_tokens,
    derive_structure_tokens,
    structure_expert_from_cif,
)

__all__ = [
    "DEFAULT_CONF_WEIGHTS",
    "ProtenixRewardClient",
    "confidence_components",
    "continuous_ip",
    "passes",
    "reward_from_confidence",
    "kabsch",
    "structure_terms",
    "tm_score",
    "coverage",
    "hamming_novelty_group",
    "jaccard_novelty_group",
    "kmer_jaccard",
    "lc_floor_penalty",
    "lc_saturating_reward",
    "linguistic_complexity",
    "combined_distribution_terms",
    "design_hists_scoped",
    "design_interface_hists",
    "distribution_terms",
    "js",
    "load_reference_table",
    "reference_for",
    "tv",
    "clash_contact_reward",
    "chainbreak_reward",
    "binder_clash_terms",
    "binder_selfclash",
    "cloud_from_atom14",
    "interface_allatom_clash",
    "sc_clash_reward",
    "aar_terms",
    "binder_letters_to_aa33",
    "interface_residue_mask",
    "reward_from_aar",
    "SFT_IGNORE_INDEX",
    "shape_complementarity_reward",
    "shape_complementarity_reward_atoms",
    "INVALID_3DI_STATE",
    "assemble_structure_expert",
    "build_struct_sft_targets",
    "derive_3di_tokens",
    "derive_structure_tokens",
    "structure_expert_from_cif",
]
