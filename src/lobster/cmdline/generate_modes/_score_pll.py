"""LeFlur PLL (pseudo-log-likelihood) scoring mode.

Reads a CSV of candidate (sequence, structure-token) pairs — optionally with
ligand tokens for the protein-ligand checkpoint — and writes a copy of that
CSV augmented with PLL columns from ``model.score_pll(...)``.

This is the centralised, framework-level version of the ad-hoc
``scripts/score_leflur_*pll.py`` research scripts that drove the conference
ranking studies. See :mod:`lobster.model.leflur._pll_scoring` for the
underlying Monte-Carlo stratified-``t`` pseudo-NLL estimator and the full
list of supported variants.

Called from :func:`lobster.cmdline.generate.generate` when
``cfg.generation.mode == "score_pll"``.

Expected input CSV
------------------
Columns ``sequence`` (or ``original_sequence``) and ``latent_generator_tokens``
(comma-separated structure-token ints, length-matched to the sequence) are
required. For the protein-ligand checkpoint, the additional columns
``ligand_atom_tokens`` and ``ligand_structure_tokens`` are required.

Output CSV
----------
Same columns as the input plus per-variant PLL columns (one ``pll_<variant>``
per requested variant) and, when ``rank_within=...`` is set, a per-group
``rank_<variant>`` integer rank column.

Typical invocations
-------------------
Protein-only (LeFlur-base / LeFlur-TED)::

    lobster_generate --config-name experiment/score_pll \\
        model.ckpt_path=leflur-ted \\
        generation.candidates_csv=/path/to/sequences_unconditional_*.csv

Protein-ligand (LeFlur-PL)::

    lobster_generate --config-name experiment/score_pll \\
        model.ckpt_path=leflur-pl \\
        generation.candidates_csv=/path/to/sequences_ligand_conditioned_*.csv \\
        generation.variants='[joint_protein, joint_ligand, joint_all, joint_true_4]'
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig, ListConfig

from lobster.model.leflur._pll_scoring import (
    PROTEIN_LIGAND_VARIANTS,
    PROTEIN_VARIANTS,
)


_SEQ_COL_CANDIDATES: tuple[str, ...] = ("sequence", "original_sequence", "designed_sequence")
_STRUC_COL: str = "latent_generator_tokens"
_LIG_ATOM_COL: str = "ligand_atom_tokens"
_LIG_STRUC_COL: str = "ligand_structure_tokens"


def _resolve_seq_col(fieldnames: list[str]) -> str:
    for cand in _SEQ_COL_CANDIDATES:
        if cand in fieldnames:
            return cand
    raise ValueError(f"candidates_csv must contain one of {_SEQ_COL_CANDIDATES}; got columns: {fieldnames}")


def _parse_token_list(value: str) -> list[int]:
    return [int(t) for t in value.strip().split(",") if t.strip() != ""]


def _tokenize_aa_sequence(aa_transform, seq_str: str) -> torch.Tensor:
    """Tokenize an AA string identically to how :mod:`generate.py` feeds the model.

    Inverts the residue_constants letter→PDB-int mapping then runs the standard
    ``AminoAcidTokenizerTransform``. The resulting 1-D long tensor is
    bit-identical to what other generation modes build for ``sequence``.
    """
    from lobster.model.latent_generator.utils.residue_constants import (
        restype_order_with_x,
    )

    pdb_ints = torch.tensor([restype_order_with_x.get(c, 20) for c in seq_str], dtype=torch.long)
    out = aa_transform({"sequence": pdb_ints})
    return out["sequence"]


def _coerce_variants(value, *, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return allowed
    if isinstance(value, (list, tuple, ListConfig)):
        variants = tuple(str(v) for v in value)
    elif isinstance(value, str):
        variants = tuple(v.strip() for v in value.split(",") if v.strip())
    else:
        raise TypeError(
            f"generation.variants must be a list/tuple or comma-separated string, got {type(value).__name__}"
        )
    unknown = set(variants) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown PLL variant(s): {sorted(unknown)}. Allowed for this model: {list(allowed)}")
    return variants


def _score_pll(  # public name used by the dispatcher
    model,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path,
    plm_fold=None,
    csv_writer=None,
    plotter=None,
) -> None:
    """Hydra entry point for ``generation.mode=score_pll``.

    All knobs live under ``cfg.generation``:

    * ``candidates_csv`` (str, required) — CSV with the candidate columns
      described in the module docstring.
    * ``output_csv`` (str, optional) — output path. Defaults to
      ``<output_dir>/pll_scores_<timestamp>.csv``.
    * ``K`` (int, default 32) — Monte-Carlo draws per modality.
    * ``eps`` (float, default 0.02) — stratified-t endpoint margin.
    * ``seed`` (int, default 0) — base scoring seed (per-row uses ``seed + i``).
    * ``variants`` (list[str], optional) — subset of the per-model variant
      tuple; defaults to all variants for the model class.
    * ``rank_within`` (str, optional) — column name to group by before
      ranking each variant (e.g. ``"input_structure"`` for inverse folding,
      ``"target_id"`` for ligand-conditioned). When set, the output CSV gets
      one ``rank_<variant>`` integer rank column per scored variant.
    * ``max_length`` (int, default 512) — skip rows longer than this.
    * ``max_samples`` (int, optional) — score only the first N rows.
    """
    from lobster.model.leflur import (
        LeFlurProteinLigandLightningModule,
        LeFlurSequenceStructureEncoderLightningModule,
    )
    from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

    gen_cfg = cfg.generation
    candidates_csv = gen_cfg.get("candidates_csv")
    if candidates_csv is None:
        raise ValueError("generation.candidates_csv is required for mode=score_pll")
    candidates_path = Path(candidates_csv).expanduser().resolve()
    if not candidates_path.is_file():
        raise FileNotFoundError(f"candidates_csv not found: {candidates_path}")

    is_pl = isinstance(model, LeFlurProteinLigandLightningModule)
    is_protein_only = isinstance(model, LeFlurSequenceStructureEncoderLightningModule)
    if not (is_pl or is_protein_only):
        raise TypeError(f"generation.mode=score_pll requires a LeFlur Lightning module, got {type(model).__name__}")
    allowed = PROTEIN_LIGAND_VARIANTS if is_pl else PROTEIN_VARIANTS

    variants = _coerce_variants(gen_cfg.get("variants"), allowed=allowed)
    K = int(gen_cfg.get("K", 32))
    eps = float(gen_cfg.get("eps", 0.02))
    seed = int(gen_cfg.get("seed", 0))
    rank_within = gen_cfg.get("rank_within")
    max_length = int(gen_cfg.get("max_length", 512))
    max_samples = gen_cfg.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)

    output_csv = gen_cfg.get("output_csv")
    if output_csv is None:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_csv = output_dir / f"pll_scores_{ts}.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "PLL scoring: model=%s candidates=%s K=%d variants=%s output=%s",
        type(model).__name__,
        candidates_path,
        K,
        variants,
        output_csv,
    )

    aa_transform = AminoAcidTokenizerTransform(max_length=max_length)
    rows: list[dict] = []
    seq_col: str | None = None
    fieldnames: list[str] = []
    with candidates_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        seq_col = _resolve_seq_col(fieldnames)
        if _STRUC_COL not in fieldnames:
            raise ValueError(f"candidates_csv missing required column {_STRUC_COL!r}. Columns present: {fieldnames}")
        if is_pl:
            for need in (_LIG_ATOM_COL, _LIG_STRUC_COL):
                if need not in fieldnames:
                    raise ValueError(
                        f"protein-ligand candidates_csv missing column {need!r}. Columns present: {fieldnames}"
                    )
        for row in reader:
            seq_str = (row.get(seq_col) or "").strip()
            tokens_str = (row.get(_STRUC_COL) or "").strip()
            if not seq_str or not tokens_str:
                continue
            try:
                struc_tokens = _parse_token_list(tokens_str)
            except ValueError:
                continue
            if len(struc_tokens) != len(seq_str):
                logger.debug(
                    "Length mismatch (seq=%d struc=%d); skipping row",
                    len(seq_str),
                    len(struc_tokens),
                )
                continue
            if len(struc_tokens) > max_length:
                continue

            lig_atom: list[int] | None = None
            lig_struc: list[int] | None = None
            if is_pl:
                try:
                    lig_atom = _parse_token_list(row.get(_LIG_ATOM_COL) or "")
                    lig_struc = _parse_token_list(row.get(_LIG_STRUC_COL) or "")
                except ValueError:
                    continue
                if not lig_atom or not lig_struc:
                    continue
                if len(lig_atom) != len(lig_struc):
                    logger.debug(
                        "Ligand atom/structure length mismatch (%d vs %d); skipping",
                        len(lig_atom),
                        len(lig_struc),
                    )
                    continue

            row["_seq_str"] = seq_str
            row["_struc_tokens"] = struc_tokens
            row["_lig_atom"] = lig_atom
            row["_lig_struc"] = lig_struc
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise RuntimeError(f"No scoreable rows in {candidates_path}")
    logger.info("Loaded %d scoreable rows from %s", len(rows), candidates_path.name)

    score_keys: list[str] = []
    n_done = 0
    n_skipped = 0
    per_row_scores: list[dict[str, float]] = []
    for row_idx, row in enumerate(rows):
        try:
            seq_tensor = _tokenize_aa_sequence(aa_transform, row["_seq_str"]).to(device)
        except Exception as e:
            logger.warning("Row %d tokenize failed: %s", row_idx, e)
            n_skipped += 1
            per_row_scores.append({})
            continue

        L = int(seq_tensor.shape[0])
        seq_clean = seq_tensor.unsqueeze(0)
        struc_clean = torch.tensor(row["_struc_tokens"], dtype=torch.long, device=device).unsqueeze(0)
        protein_mask = torch.ones(1, L, device=device)

        try:
            if is_pl:
                lig_atom = torch.tensor(row["_lig_atom"], dtype=torch.long, device=device).unsqueeze(0)
                lig_struc = torch.tensor(row["_lig_struc"], dtype=torch.long, device=device).unsqueeze(0)
                lig_mask = torch.ones(1, lig_atom.shape[1], device=device)
                scores = model.score_pll(
                    sequence_tokens=seq_clean,
                    structure_tokens=struc_clean,
                    protein_mask=protein_mask,
                    ligand_atom_tokens=lig_atom,
                    ligand_structure_tokens=lig_struc,
                    ligand_mask=lig_mask,
                    K=K,
                    eps=eps,
                    seed=seed + row_idx,
                    variants=variants,
                )
            else:
                scores = model.score_pll(
                    sequence_tokens=seq_clean,
                    structure_tokens=struc_clean,
                    mask=protein_mask,
                    K=K,
                    eps=eps,
                    seed=seed + row_idx,
                    variants=variants,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM on row %d (L=%d); skipping", row_idx, L)
            n_skipped += 1
            per_row_scores.append({})
            continue

        per_row = {k: float(v.item()) for k, v in scores.items()}
        per_row_scores.append(per_row)
        if not score_keys:
            score_keys = sorted(per_row.keys())
        n_done += 1
        if n_done == 1 or n_done % max(1, int(gen_cfg.get("log_every", 25))) == 0:
            logger.info(
                "[%4d/%d] L=%d  %s",
                n_done,
                len(rows),
                L,
                "  ".join(f"{k}={per_row[k]:.4f}" for k in score_keys[:4]),
            )

    if n_done == 0:
        raise RuntimeError("PLL scoring produced zero rows (all skipped)")

    pll_cols = [f"pll_{k}" for k in score_keys]
    rank_cols: list[str] = []
    if rank_within is not None:
        rank_cols = [f"rank_{k}" for k in score_keys]
        groups: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            if not per_row_scores[i]:
                continue
            key = row.get(rank_within, "") if rank_within in fieldnames else ""
            groups.setdefault(str(key), []).append(i)
        per_row_ranks: list[dict[str, int]] = [{} for _ in rows]
        for variant in score_keys:
            for group_rows in groups.values():
                ordered = sorted(group_rows, key=lambda i: per_row_scores[i][variant])
                for rank, i in enumerate(ordered, start=1):
                    per_row_ranks[i][variant] = rank
    else:
        per_row_ranks = [{} for _ in rows]

    out_fields = [c for c in fieldnames if not c.startswith("_")] + pll_cols + rank_cols
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            sanitized = {k: v for k, v in row.items() if not k.startswith("_")}
            for variant, val in per_row_scores[i].items():
                if math.isnan(val) or math.isinf(val):
                    sanitized[f"pll_{variant}"] = ""
                else:
                    sanitized[f"pll_{variant}"] = f"{val:.6f}"
            for variant, rank in per_row_ranks[i].items():
                sanitized[f"rank_{variant}"] = rank
            writer.writerow(sanitized)

    logger.info(
        "PLL scoring complete: %d scored / %d skipped → %s",
        n_done,
        n_skipped,
        output_csv,
    )
