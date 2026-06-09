"""End-to-end tests for the ``score_pll`` generation mode.

Exercises :func:`lobster.cmdline.generate_modes._score_pll._score_pll` with
a synthetic input CSV and a stub Lightning module. Asserts:

1. The output CSV is written, contains every original column verbatim plus
   one ``pll_<variant>`` column per requested variant.
2. Sample-level scores are finite floats.
3. When ``rank_within`` is set, per-group ``rank_<variant>`` columns are
   present and contain valid 1-based ranks.

These tests run on CPU in well under a second and do not require any LeFlur
checkpoint or GPU — they patch the model class check so the stub passes the
``isinstance`` gate inside the score_pll dispatcher.
"""

from __future__ import annotations

import csv
import math
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from lobster.cmdline.generate_modes._score_pll import _score_pll as score_pll_fn


@contextmanager
def _patched_leflur_classes(stub_cls):
    """Patch the two LeFlur isinstance gates in ``_score_pll`` to accept ``stub_cls``.

    The mode does ``from lobster.model.leflur import ...`` inside the
    function body, so patching needs to target the source module
    ``lobster.model.leflur``, not the call site.
    """
    import lobster.model.leflur as leflur_mod

    real_protein = leflur_mod.LeFlurSequenceStructureEncoderLightningModule
    real_pl = leflur_mod.LeFlurProteinLigandLightningModule
    with (
        patch.object(leflur_mod, "LeFlurSequenceStructureEncoderLightningModule", stub_cls),
        patch.object(leflur_mod, "LeFlurProteinLigandLightningModule", real_pl),
    ):
        yield
    leflur_mod.LeFlurSequenceStructureEncoderLightningModule = real_protein
    leflur_mod.LeFlurProteinLigandLightningModule = real_pl


class _StubProteinModule:
    """Bare-minimum LeFlur protein-only Lightning module surrogate.

    Returns uniform logits so every CE evaluates to log(V) — exact, predictable.
    """

    def __init__(self, vocab_seq: int = 24, vocab_struc: int = 512):
        self.vocab_seq = vocab_seq
        self.vocab_struc = vocab_struc
        self.mask_token_id = vocab_seq - 1
        self.mask_index_struc_tokens = vocab_struc - 2
        self.num_struc_classes = vocab_struc
        self.vocab_size = vocab_seq
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    @torch.no_grad()
    def score_pll(self, **kwargs):
        from lobster.model.leflur._pll_scoring import (
            PROTEIN_VARIANTS,
            score_protein_pll,
        )

        variants = kwargs.pop("variants", None)
        return score_protein_pll(
            self,
            **{k: v for k, v in kwargs.items() if k != "variants"},
            variants=variants if variants is not None else PROTEIN_VARIANTS,
        )

    def forward(self, x_t, mask, residue_index, conditioning_tensor, timesteps=None):
        K, L = x_t["sequence_tokens"].shape
        device = x_t["sequence_tokens"].device
        return {
            "sequence_logits": torch.zeros(K, L, self.vocab_seq, device=device),
            "structure_logits": torch.zeros(K, L, self.vocab_struc, device=device),
        }


def _write_candidates_csv(path: Path, n_rows: int = 4, length: int = 128) -> None:
    """Write a synthetic candidates CSV.

    Default ``length=128`` keeps the canonical zero-mask-draw bias well below
    1e-4 (see test_score_protein_pll_uniform_logits_gives_log_V) so the
    log(V) exactness check passes under K=4 strata.
    """
    rows = []
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"  # 20 valid AAs
    for i in range(n_rows):
        seq_str = (aa_alphabet * (length // len(aa_alphabet) + 1))[:length]
        struc_tokens = ",".join(str((i * 7 + j) % 64) for j in range(length))
        rows.append(
            {
                "iteration": i // 2,
                "sample_idx": i,
                "input_structure": f"target_{i % 2}.pdb",
                "sequence": seq_str,
                "latent_generator_tokens": struc_tokens,
            }
        )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_cfg(
    candidates_csv: Path,
    output_dir: Path,
    *,
    variants: list[str] | None = None,
    rank_within: str | None = None,
    K: int = 4,
):
    cfg_dict = {
        "generation": {
            "mode": "score_pll",
            "candidates_csv": str(candidates_csv),
            "output_csv": None,
            "K": K,
            "eps": 0.02,
            "seed": 0,
            "variants": variants,
            "rank_within": rank_within,
            "max_length": 512,
            "max_samples": None,
            "log_every": 10,
            "use_esmfold": False,
            "save_csv_metrics": False,
            "create_plots": False,
        },
    }
    return OmegaConf.create(cfg_dict)


def test_score_pll_writes_augmented_csv(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    _write_candidates_csv(candidates_csv, n_rows=4, length=128)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = _build_cfg(candidates_csv, output_dir, variants=["seq", "struc", "joint_protein"], K=8)

    stub = _StubProteinModule()

    with _patched_leflur_classes(type(stub)):
        score_pll_fn(
            stub,
            cfg,
            device=torch.device("cpu"),
            output_dir=output_dir,
        )

    output_csvs = list(output_dir.glob("pll_scores_*.csv"))
    assert len(output_csvs) == 1, f"expected one output CSV, found {output_csvs}"
    output_csv = output_csvs[0]

    with output_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        cols = reader.fieldnames or []

    # Original columns preserved (minus internal `_` columns).
    assert {"iteration", "sample_idx", "input_structure", "sequence", "latent_generator_tokens"}.issubset(cols)
    # Each requested variant + diagnostic *_score_arllh shows up as pll_<key>.
    expected_pll = {"pll_seq", "pll_struc", "pll_joint_protein", "pll_seq_score_arllh", "pll_struc_score_arllh"}
    assert expected_pll.issubset(cols), f"missing pll columns; cols={cols}"

    assert len(rows) == 4
    expected_seq_nll = math.log(stub.vocab_seq)
    expected_struc_nll = math.log(stub.vocab_struc)
    for row in rows:
        assert float(row["pll_seq"]) == pytest.approx(expected_seq_nll, abs=1e-3)
        assert float(row["pll_struc"]) == pytest.approx(expected_struc_nll, abs=1e-3)
        assert float(row["pll_joint_protein"]) == pytest.approx(expected_seq_nll + expected_struc_nll, abs=1e-3)


def test_score_pll_rank_within_assigns_per_group_ranks(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    _write_candidates_csv(candidates_csv, n_rows=4, length=64)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = _build_cfg(
        candidates_csv,
        output_dir,
        variants=["joint_protein"],
        rank_within="input_structure",
        K=4,
    )

    stub = _StubProteinModule()
    with _patched_leflur_classes(type(stub)):
        score_pll_fn(
            stub,
            cfg,
            device=torch.device("cpu"),
            output_dir=output_dir,
        )

    output_csv = next(output_dir.glob("pll_scores_*.csv"))
    with output_csv.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert "rank_joint_protein" in rows[0]
    by_target: dict[str, list[int]] = {}
    for row in rows:
        by_target.setdefault(row["input_structure"], []).append(int(row["rank_joint_protein"]))
    for target, ranks in by_target.items():
        assert sorted(ranks) == list(range(1, len(ranks) + 1)), (
            f"target={target} ranks={ranks} should be a permutation of 1..N"
        )


def test_score_pll_missing_candidates_csv_raises(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "does_not_exist.csv"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = _build_cfg(candidates_csv, output_dir)

    stub = _StubProteinModule()
    with _patched_leflur_classes(type(stub)):
        with pytest.raises(FileNotFoundError, match="candidates_csv"):
            score_pll_fn(
                stub,
                cfg,
                device=torch.device("cpu"),
                output_dir=output_dir,
            )


def test_score_pll_rejects_unknown_variant(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    _write_candidates_csv(candidates_csv, n_rows=2, length=32)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = _build_cfg(candidates_csv, output_dir, variants=["seq", "totally_made_up"])

    stub = _StubProteinModule()
    with _patched_leflur_classes(type(stub)):
        with pytest.raises(ValueError, match="Unknown PLL variant"):
            score_pll_fn(
                stub,
                cfg,
                device=torch.device("cpu"),
                output_dir=output_dir,
            )
