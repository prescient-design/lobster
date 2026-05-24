"""End-to-end smoke tests for LeFlur unconditional generation.

Two tests:

1. ``test_unconditional_length100_decodes_to_pdb`` — generates one length-100
   backbone, decodes it through the latent generator, and writes a PDB. Asserts
   the file exists, is non-trivial, and that the decoded coordinates have the
   expected ``[L, 3, 3]`` (N / CA / C atoms) shape and are finite. Requires
   CUDA — sampling 200 timesteps from a 750 M-parameter model on CPU takes far
   longer than a reasonable smoke test budget.

2. ``test_unconditional_length100_esmfold_agrees`` — additionally folds the
   designed sequence with ESMFold and asserts the TM-score between the LeFlur
   backbone and the ESMFold prediction is above a lenient floor. At L=100 the
   TED checkpoint achieves a mean ESMFold TM of ~0.93 (conference benchmark
   plan, Table 1), so anything < 0.50 indicates the rename has corrupted
   structure decoding rather than ordinary sampling variance.

Both tests are marked ``slow`` and require:

- CUDA-capable GPU (ESMFold + LeFlur sampling are GPU-bound in practice).
- Read access to the canonical TED checkpoint file referenced from
  ``.cursor/plans/conference_benchmark_comparison_9b71ca71.plan.md``.
- Network access for the HuggingFace-hosted ``LG full attention`` codec the
  LeFlur module loads at construction time and for the ``esmfold_v1`` weights.

If any of those preconditions is missing the test skips cleanly so non-GPU /
non-Genentech CI environments still pass the suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[4] / "scripts" / "leflur_generate_smoke.py"
)


def _load_smoke_module() -> ModuleType:
    """Load the standalone ``leflur_generate_smoke`` script as a module.

    ``scripts/`` is not a Python package in this repo, so a plain ``from
    scripts import leflur_generate_smoke`` would not work. Loading via
    ``importlib`` keeps the CLI script and this test consistent without
    forcing a package layout on ``scripts/``.
    """
    spec = importlib.util.spec_from_file_location(
        "leflur_generate_smoke", str(_SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None, (
        f"could not build module spec from {_SCRIPT_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# The conference benchmark plan refers to this as
# ``gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59``. The
# suffix is the *eval launch* timestamp; the underlying ckpt file is the
# ``last.ckpt`` symlink in the 2026-03-14T15-41-36 training run.
TED_CKPT = Path(
    "/cv/scratch/u/lisanzas/gen_ume_denovo_ted_cath/runs"
    "/2026-03-14T15-41-36/last.ckpt"
)

# Below this TM-score we are confident the structure/sequence agreement
# has collapsed (TED mean TM at L=100 is ~0.934 per the benchmark plan, so
# 0.50 is a wide safety margin around the expected operating point that
# catches structural-decoding regressions without flaking on sampling noise).
TM_FLOOR = 0.50


def _skip_if_unsupported(require_cuda: bool = True) -> None:
    if not TED_CKPT.exists():
        pytest.skip(f"LeFlur TED checkpoint not present: {TED_CKPT}")
    if require_cuda:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable; LeFlur sampling + ESMFold need a GPU in practice")


@pytest.fixture(scope="module")
def generated_sample(tmp_path_factory):
    """Run unconditional generation once and share the result across tests."""
    _skip_if_unsupported(require_cuda=True)

    # Import via the script we ship alongside this test so the two flows
    # cannot drift out of sync.
    leflur_generate_smoke = _load_smoke_module()

    out_dir = tmp_path_factory.mktemp("leflur_smoke")
    # Generation hyperparameters match the canonical
    # ``generate_unconditional_denovo.yaml`` Hydra config — the exact recipe
    # used for the conference benchmark ``GenUME-TED`` baseline cell (Table 1
    # L=100, Pass% 99, mean TM 0.934, mean pLDDT 0.79). Anything below
    # TM_FLOOR with these knobs indicates a real structural-decoding
    # regression from the rename.
    args = leflur_generate_smoke.parse_args(
        [
            "--checkpoint",
            str(TED_CKPT),
            "--length",
            "100",
            "--nsteps",
            "400",
            "--seed",
            "12345",
            "--out-dir",
            str(out_dir),
            "--temperature-seq",
            "0.27315634404739075",
            "--temperature-struc",
            "0.31640411575109995",
            "--stochasticity-seq",
            "20",
            "--stochasticity-struc",
            "60",
            "--inference-schedule-seq",
            "LogInferenceSchedule",
            "--inference-schedule-struc",
            "PowerInferenceSchedule",
        ]
    )
    return leflur_generate_smoke.generate_one(args), args


@pytest.mark.slow
def test_unconditional_length100_decodes_to_pdb(generated_sample) -> None:
    """LeFlur produces a length-100 backbone PDB that loads as a real file."""
    import torch

    result, args = generated_sample

    pdb_path: Path = result["pdb_path"]
    assert pdb_path.exists(), f"expected PDB at {pdb_path}"
    contents = pdb_path.read_text()
    n_atom_lines = sum(1 for line in contents.splitlines() if line.startswith("ATOM"))
    # 3 backbone atoms (N, CA, C) × 100 residues = 300 ATOM lines, give or
    # take whatever ``writepdb`` adds. Anything below ~250 means the PDB
    # writer silently dropped residues.
    assert n_atom_lines >= 250, (
        f"backbone PDB has too few ATOM lines ({n_atom_lines}); "
        f"expected ~300 for length=100"
    )

    coords = result["coords"]
    assert coords.shape[0] == 1, "expected one sample"
    assert coords.shape[1] == args.length, "decoded length mismatch"
    # Backbone tensor must have at least 3 atom positions (N, CA, C) per residue.
    assert coords.shape[2] >= 3, "decoded coords missing backbone atoms"
    assert coords.shape[-1] == 3, "coordinate tensor is not 3D"
    assert torch.isfinite(coords).all(), "decoded coords contain NaN / Inf"

    # The designed sequence must contain only valid one-letter codes
    # (20 canonical AAs + X for unknown).
    seq = result["sequence"]
    assert len(seq) == args.length, "sequence length mismatch"
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWYX"), (
        f"designed sequence contains unexpected codes: {sorted(set(seq))}"
    )


@pytest.mark.slow
def test_unconditional_length100_esmfold_agrees(generated_sample) -> None:
    """ESMFold of the designed sequence agrees with the decoded backbone."""
    leflur_generate_smoke = _load_smoke_module()

    result, args = generated_sample
    # Force ESMFold on for this test even though the fixture didn't pass it.
    args.esmfold = True
    args.tm_floor = TM_FLOOR

    esm = leflur_generate_smoke.maybe_esmfold(result, args)
    assert esm is not None, "ESMFold step returned None"

    # Diagnostics we want surfaced in pytest output when something fails.
    tm = esm["tm_score"]
    rmsd = esm["rmsd"]
    plddt = esm["plddt"]
    msg = (
        f"ESMFold↔LeFlur TM={tm:.4f} (floor={TM_FLOOR:.2f}), "
        f"RMSD={rmsd:.2f}Å, pLDDT={plddt:.3f}"
    )
    assert tm >= TM_FLOOR, msg
    # ESMFold should not be returning garbage pLDDT either — anything below
    # 0.5 means we are folding an obviously unfoldable sequence (designed
    # sequence quality regression). The conference benchmark cell shows mean
    # pLDDT 0.79 at L=100 for TED.
    assert plddt >= 0.50, f"ESMFold pLDDT collapsed: {msg}"

    # Confirm the ESMFold PDB is on disk so the user can visualise both
    # backbones side by side.
    assert esm["esmfold_pdb"].exists()
