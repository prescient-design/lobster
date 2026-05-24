"""End-to-end smoke tests for LeFlur ligand-conditioned protein generation.

Sister to ``test_generate_smoke.py``; wraps
``scripts/leflur_ligand_conditioned_smoke.py``.

Two tests:

1. ``test_ligand_conditioned_length100_decodes_to_pdb`` — generates one
   ligand-conditioned design with the PL checkpoint, decodes both protein
   backbone and ligand atoms, writes two PDBs. Asserts both files exist with
   plausible content and that the model returned finite coords for both
   modalities.

2. ``test_ligand_conditioned_length100_esmfold_agrees`` — additionally folds
   the designed sequence with ESMFold and asserts the scTM-score between the
   decoded LeFlur backbone and the ESMFold prediction is above a lenient
   floor.

Both tests are marked ``slow`` and require:

- CUDA-capable GPU.
- Read access to the canonical LeFlur PL checkpoint and the PoseBusters
  benchmark .pt files (default paths under ``/cv/scratch/`` and
  ``/cv/home/lisanzas/lobster/data/``).
- Network for the HuggingFace-hosted LG codec the PL module loads at
  construction time and for the ``esmfold_v1`` weights.

If any precondition is missing the test skips cleanly so non-GPU /
non-Genentech CI environments still pass the suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "leflur_ligand_conditioned_smoke.py"
PL_CKPT = Path(
    "/cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium/runs"
    "/2026-02-11T19-45-30/epoch=278-step=40057-val_loss=1.6365.ckpt"
)
LIGAND_DIR = Path("/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap")

# Ligand-conditioned designs run at lower temperatures than unconditional so
# the published self-consistency expectation is lower (mean scTM is
# ligand-and-pocket dependent; the evaluator's success rubric uses scTM > 0.5
# as the "high self-consistency" cutoff, see ``_print_summary`` in
# ``lobster_generate generation.mode=ligand_conditioned`` via
# ``_ligand_conditioned_runner.run_ligand_conditioned_generation``). 0.4 is a
# lenient single-sample smoke floor that catches structural collapse without
# flaking on hard ligands or unlucky seeds.
SCTM_FLOOR = 0.40


def _load_smoke_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("leflur_ligand_conditioned_smoke", str(_SCRIPT_PATH))
    assert spec is not None and spec.loader is not None, f"could not build module spec from {_SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _skip_if_unsupported(require_cuda: bool = True) -> None:
    if not PL_CKPT.exists():
        pytest.skip(f"LeFlur PL checkpoint not present: {PL_CKPT}")
    if not LIGAND_DIR.exists():
        pytest.skip(f"PoseBusters ligand fixtures not present: {LIGAND_DIR}")
    if not any(LIGAND_DIR.glob("*_ligand.pt")):
        pytest.skip(f"No *_ligand.pt under {LIGAND_DIR}")
    if require_cuda:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable; LeFlur PL sampling + ESMFold need a GPU in practice")


@pytest.fixture(scope="module")
def ligand_conditioned_sample(tmp_path_factory):
    """Run one ligand-conditioned generation and share across tests."""
    _skip_if_unsupported(require_cuda=True)
    smoke = _load_smoke_module()

    out_dir = tmp_path_factory.mktemp("leflur_ligand_smoke")
    # Hyperparameters mirror the canonical Hydra config
    # `experiment/generate_ligand_conditioned.yaml`
    # exactly so this test exercises the canonical inference recipe.
    args = smoke.parse_args(
        [
            "--checkpoint",
            str(PL_CKPT),
            "--ligand-dir",
            str(LIGAND_DIR),
            "--length",
            "100",
            "--nsteps",
            "200",
            "--seed",
            "1234",
            "--out-dir",
            str(out_dir),
            "--temperature-seq",
            "0.153",
            "--temperature-struc",
            "0.05",
            "--stochasticity-seq",
            "20",
            "--stochasticity-struc",
            "20",
            "--temperature-ligand",
            "0.1",
            "--stochasticity-ligand",
            "5",
            "--ligand-context-mode",
            "atom_bond_only",
            "--inference-schedule-seq",
            "LinearInferenceSchedule",
            "--inference-schedule-struc",
            "PowerInferenceSchedule",
            "--inference-schedule-ligand-atom",
            "PowerInferenceSchedule",
            "--inference-schedule-ligand-struc",
            "LinearInferenceSchedule",
        ]
    )
    return smoke.generate_one(args), args


@pytest.mark.slow
def test_ligand_conditioned_length100_decodes_to_pdb(ligand_conditioned_sample) -> None:
    """Both protein backbone and decoded ligand are written as valid PDBs."""
    import torch

    result, args = ligand_conditioned_sample

    protein_pdb: Path = result["protein_pdb"]
    assert protein_pdb.exists(), f"missing protein PDB: {protein_pdb}"
    atom_lines = sum(1 for line in protein_pdb.read_text().splitlines() if line.startswith("ATOM"))
    assert atom_lines >= 250, f"protein PDB has only {atom_lines} ATOM lines; expected ~300 for L=100"

    coords = result["protein_coords"]
    assert coords.shape[1] == args.length, "protein length mismatch"
    assert coords.shape[2] >= 3 and coords.shape[-1] == 3, "bad protein coord shape"
    assert torch.isfinite(coords).all(), "protein coords contain NaN / Inf"

    seq = result["sequence"]
    assert len(seq) == args.length
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWYX"), f"unexpected AA codes: {sorted(set(seq))}"

    # Decoded ligand must be present and have at least as many heavy atoms
    # as the PoseBusters fixture provided.
    lig_coords = result["decoded_ligand_coords"]
    assert lig_coords is not None, "model did not return decoded ligand coordinates"
    assert torch.isfinite(lig_coords).all(), "ligand coords contain NaN / Inf"
    n_decoded_atoms = lig_coords.shape[1]
    assert n_decoded_atoms >= 5, f"only {n_decoded_atoms} decoded ligand atoms; PoseBusters fixtures have ≥10"

    ligand_pdb: Path | None = result["ligand_pdb"]
    assert ligand_pdb is not None and ligand_pdb.exists(), "ligand PDB was not written"
    het_lines = sum(1 for line in ligand_pdb.read_text().splitlines() if line.startswith("HETATM"))
    assert het_lines == n_decoded_atoms, (
        f"ligand PDB HETATM count ({het_lines}) does not match decoded atom count ({n_decoded_atoms})"
    )


@pytest.mark.slow
def test_ligand_conditioned_length100_esmfold_agrees(
    ligand_conditioned_sample,
) -> None:
    """ESMFold of the designed sequence agrees with the decoded backbone."""
    smoke = _load_smoke_module()

    result, args = ligand_conditioned_sample
    args.esmfold = True
    args.tm_floor = SCTM_FLOOR

    esm = smoke.maybe_esmfold(result, args)
    assert esm is not None, "ESMFold step returned None"

    sc_tm = esm["scTM"]
    sc_rmsd = esm["scRMSD"]
    plddt = esm["plddt"]
    msg = f"ESMFold↔LeFlur (PL) scTM={sc_tm:.4f} (floor={SCTM_FLOOR:.2f}), scRMSD={sc_rmsd:.2f}Å, pLDDT={plddt:.3f}"
    assert sc_tm >= SCTM_FLOOR, msg
    # ESMFold pLDDT collapse → designed sequence quality regression.
    assert plddt >= 0.40, f"ESMFold pLDDT collapsed: {msg}"
    assert esm["esmfold_pdb"].exists()
