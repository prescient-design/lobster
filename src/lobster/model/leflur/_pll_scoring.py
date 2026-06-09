"""Pseudo-likelihood (PLL) scoring for LeFlur.

Centralizes the Monte-Carlo stratified-`t` pseudo-NLL estimator first developed
in the conference-supplement studies (see :doc:`docs/leflur/benchmarks.md`).
The canonical algorithm is::

    for each modality m in MODS:
        sample K stratified t-values t_1, ..., t_K from (eps, 1 - eps)
        for each draw k:
            mask every clean token of m independently with prob (1 - t_k)
            hold every other modality clean (timestep = 1)
            forward pass: get logits[m]
            per_pos_ce  = -log_softmax(logits[m]).gather(target[m])
            avg_ce[k]   = sum_masked(per_pos_ce) / n_masked
        score_m = mean_k avg_ce[k]                              # *_score
        score_arllh_m = mean_k (sum_ce[k] / ((1 - t_k) * L))   # *_score_arllh

The composite variants are then:

* ``joint_protein`` = ``seq + struc``                        (additive)
* ``joint_ligand``  = ``lig_atom + lig_struc``               (additive, PL only)
* ``joint_all``     = ``seq + struc + lig_atom + lig_struc`` (additive, PL only)
* ``joint_true_2``  = one forward with **both** protein modalities masked at
  the same ``t_k`` (true AO-ARM estimator over the unified 2L stream)
* ``joint_true_4``  = analogous true AO-ARM on the unified 2(L + M) stream
  (PL only)

Lower NLL means higher likelihood, so picking by ``argmin`` selects the
best-ranked candidate.

This module is a pure helper — it does **not** import the Lightning modules.
``LeFlurSequenceStructureEncoderLightningModule.score_pll`` and
``LeFlurProteinLigandLightningModule.score_pll`` call into these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

if TYPE_CHECKING:
    pass

PROTEIN_VARIANTS: Final[tuple[str, ...]] = (
    "seq",
    "struc",
    "joint_protein",
    "joint_true_2",
)
PROTEIN_LIGAND_VARIANTS: Final[tuple[str, ...]] = (
    "seq",
    "struc",
    "lig_atom",
    "lig_struc",
    "joint_protein",
    "joint_ligand",
    "joint_all",
    "joint_true_4",
)

_STRATIFIED_EPS_DEFAULT: Final[float] = 0.02


def stratified_t_samples(
    K: int,
    *,
    eps: float = _STRATIFIED_EPS_DEFAULT,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return ``K`` stratified samples of ``t`` in ``(eps, 1 - eps)``.

    Stratified sampling eliminates the dominant source of cross-seed variance
    at small ``K``. With ``eps`` close to 0 and 1 the strata still cover most of
    the unit interval while avoiding the degenerate endpoints.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    edges = torch.linspace(eps, 1.0 - eps, steps=K + 1, device=device)
    u = torch.empty(K, device=device).uniform_(0.0, 1.0, generator=generator)
    return edges[:-1] + u * (edges[1:] - edges[:-1])


def absorbing_corrupt(
    clean: torch.Tensor,
    t_values: torch.Tensor,
    mask_id: int,
    valid_mask: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask tokens of ``clean`` with probability ``(1 - t_k)`` per position.

    Parameters
    ----------
    clean : Tensor, shape ``(K, L)`` (or broader)
        Clean tokens replicated across the K MC draws.
    t_values : Tensor, shape ``(K,)``
        Per-draw fraction of tokens kept clean.
    mask_id : int
        Mask token id for this modality.
    valid_mask : Tensor, shape ``(K, L)``
        Boolean mask of positions that are scoreable (1) vs padding (0).
    generator : torch.Generator
        RNG.

    Returns
    -------
    x_t : Tensor
        Corrupted tokens, same shape as ``clean``.
    masked_positions : Tensor
        Boolean tensor (same shape as ``clean``) of positions actually masked
        (intersection of "mask die rolled" AND "position valid").
    """
    rand = torch.rand(clean.shape, generator=generator, device=clean.device)
    mask_pos = rand > t_values.view(-1, *([1] * (clean.dim() - 1)))
    mask_pos = mask_pos & valid_mask.bool()
    x_t = torch.where(mask_pos, torch.full_like(clean, mask_id), clean)
    return x_t, mask_pos


def ce_on_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    masked_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute average / summed CE restricted to ``masked_pos``.

    Parameters
    ----------
    logits : Tensor ``(K, N, V)``
    target : Tensor ``(K, N)`` of token indices
    masked_pos : Tensor ``(K, N)`` bool

    Returns
    -------
    avg_ce, sum_ce, n_masked : each ``(K,)`` tensor of floats.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    per_pos_ce = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    n_masked = masked_pos.sum(dim=1).to(torch.float32)
    sum_ce = (per_pos_ce * masked_pos.float()).sum(dim=1)
    avg_ce = sum_ce / n_masked.clamp(min=1.0)
    return avg_ce, sum_ce, n_masked


# ---------------------------------------------------------------------------
# Protein-only scoring (LeFlurSequenceStructureEncoderLightningModule)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _score_one_protein_sample(
    module,
    *,
    seq_clean: torch.Tensor,
    struc_clean: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor,
    K: int,
    eps: float,
    seed: int,
    variants: frozenset[str],
) -> dict[str, float]:
    """Score one ``(seq, struc)`` sample.

    ``seq_clean`` / ``struc_clean`` / ``mask`` / ``residue_index`` are
    ``(1, L)``-shaped (single sample). Internally expanded to ``(K, L)``.
    """
    device = seq_clean.device
    seq_mask_id = int(module.mask_token_id)
    struc_mask_id = int(module.mask_index_struc_tokens)
    L = int(seq_clean.shape[1])

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    t_values = stratified_t_samples(K, eps=eps, device=device, generator=generator)

    seq_K = seq_clean.expand(K, -1).contiguous()
    struc_K = struc_clean.expand(K, -1).contiguous()
    mask_K = mask.expand(K, -1).contiguous()
    residue_K = residue_index.expand(K, -1).contiguous()
    conditioning = torch.zeros(K, L, 1, device=device)

    out: dict[str, float] = {}

    if "seq" in variants or "joint_protein" in variants:
        seq_x, seq_mp = absorbing_corrupt(seq_K, t_values, seq_mask_id, mask_K, generator)
        timesteps = {
            "sequence_tokens": t_values,
            "structure_tokens": torch.ones(K, device=device),
        }
        forward_out = module.forward(
            {"sequence_tokens": seq_x, "structure_tokens": struc_K},
            mask_K,
            residue_K,
            conditioning,
            timesteps=timesteps,
        )
        avg_ce, sum_ce, _ = ce_on_masked(forward_out["sequence_logits"], seq_K, seq_mp)
        out["seq"] = avg_ce.mean().item()
        out["seq_score_arllh"] = (sum_ce / ((1.0 - t_values) * L).clamp(min=1e-6)).mean().item()

    if "struc" in variants or "joint_protein" in variants:
        struc_x, struc_mp = absorbing_corrupt(struc_K, t_values, struc_mask_id, mask_K, generator)
        timesteps = {
            "sequence_tokens": torch.ones(K, device=device),
            "structure_tokens": t_values,
        }
        forward_out = module.forward(
            {"sequence_tokens": seq_K, "structure_tokens": struc_x},
            mask_K,
            residue_K,
            conditioning,
            timesteps=timesteps,
        )
        avg_ce, sum_ce, _ = ce_on_masked(forward_out["structure_logits"], struc_K, struc_mp)
        out["struc"] = avg_ce.mean().item()
        out["struc_score_arllh"] = (sum_ce / ((1.0 - t_values) * L).clamp(min=1e-6)).mean().item()

    if "joint_protein" in variants:
        out["joint_protein"] = out["seq"] + out["struc"]

    if "joint_true_2" in variants:
        seq_x, seq_mp = absorbing_corrupt(seq_K, t_values, seq_mask_id, mask_K, generator)
        struc_x, struc_mp = absorbing_corrupt(struc_K, t_values, struc_mask_id, mask_K, generator)
        timesteps = {
            "sequence_tokens": t_values,
            "structure_tokens": t_values,
        }
        forward_out = module.forward(
            {"sequence_tokens": seq_x, "structure_tokens": struc_x},
            mask_K,
            residue_K,
            conditioning,
            timesteps=timesteps,
        )
        _, sum_ce_s, n_s = ce_on_masked(forward_out["sequence_logits"], seq_K, seq_mp)
        _, sum_ce_st, n_st = ce_on_masked(forward_out["structure_logits"], struc_K, struc_mp)
        n_total = n_s + n_st
        sum_total = sum_ce_s + sum_ce_st
        out["joint_true_2"] = (sum_total / n_total.clamp(min=1.0)).mean().item()

    return out


@torch.no_grad()
def score_protein_pll(
    module,
    *,
    sequence_tokens: torch.Tensor,
    structure_tokens: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor | None = None,
    K: int = 32,
    eps: float = _STRATIFIED_EPS_DEFAULT,
    seed: int = 0,
    variants: tuple[str, ...] = PROTEIN_VARIANTS,
) -> dict[str, torch.Tensor]:
    """Batch wrapper around :func:`_score_one_protein_sample`.

    Parameters
    ----------
    module : LeFlurSequenceStructureEncoderLightningModule
    sequence_tokens : Tensor ``(B, L)``
    structure_tokens : Tensor ``(B, L)``
    mask : Tensor ``(B, L)``
    residue_index : Tensor ``(B, L)`` or ``None``
        If ``None``, ``torch.arange(L)`` is broadcast across the batch.

    Returns
    -------
    dict mapping each requested variant name to a ``(B,)`` float tensor of
    per-sample mean-over-K negative log likelihoods (lower = higher
    likelihood). The dict also contains the diagnostic ``seq_score_arllh`` /
    ``struc_score_arllh`` weighted variants when ``seq`` / ``struc`` are
    requested.
    """
    _validate_variants(variants, allowed=PROTEIN_VARIANTS, kind="protein-only")
    variant_set = frozenset(variants)

    B, L = sequence_tokens.shape
    if structure_tokens.shape != sequence_tokens.shape:
        raise ValueError(
            f"sequence_tokens shape {tuple(sequence_tokens.shape)} != structure_tokens shape "
            f"{tuple(structure_tokens.shape)}"
        )
    if mask.shape != sequence_tokens.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != sequence_tokens shape {tuple(sequence_tokens.shape)}")
    device = sequence_tokens.device
    if residue_index is None:
        residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    per_sample: list[dict[str, float]] = []
    for b in range(B):
        result = _score_one_protein_sample(
            module,
            seq_clean=sequence_tokens[b : b + 1],
            struc_clean=structure_tokens[b : b + 1],
            mask=mask[b : b + 1],
            residue_index=residue_index[b : b + 1],
            K=K,
            eps=eps,
            seed=seed + b,
            variants=variant_set,
        )
        per_sample.append(result)

    keys = sorted(per_sample[0].keys())
    return {k: torch.tensor([r[k] for r in per_sample], device=device) for k in keys}


# ---------------------------------------------------------------------------
# Protein-ligand scoring (LeFlurProteinLigandLightningModule)
# ---------------------------------------------------------------------------


def _kbatch_forward_pl(
    module,
    *,
    seq_tokens: torch.Tensor,
    struc_tokens: torch.Tensor,
    lig_atom_tokens: torch.Tensor,
    lig_struc_tokens: torch.Tensor,
    protein_mask: torch.Tensor,
    ligand_mask: torch.Tensor,
    residue_index: torch.Tensor,
    bond_matrix: torch.Tensor | None,
    timesteps: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    K, L = seq_tokens.shape
    conditioning = torch.zeros(K, L, 1, device=seq_tokens.device)
    bond_K = bond_matrix.expand(K, *bond_matrix.shape[1:]).contiguous() if bond_matrix is not None else None
    return module.forward(
        x_t={
            "sequence_tokens": seq_tokens,
            "structure_tokens": struc_tokens,
            "ligand_atom_tokens": lig_atom_tokens,
            "ligand_structure_tokens": lig_struc_tokens,
        },
        mask=protein_mask,
        residue_index=residue_index,
        conditioning_tensor=conditioning,
        timesteps=timesteps,
        ligand_mask=ligand_mask,
        bond_matrix=bond_K,
    )


@torch.no_grad()
def _score_one_protein_ligand_sample(
    module,
    *,
    seq_clean: torch.Tensor,  # (1, L)
    struc_clean: torch.Tensor,  # (1, L)
    lig_atom_clean: torch.Tensor,  # (1, M)
    lig_struc_clean: torch.Tensor,  # (1, M)
    protein_mask: torch.Tensor,  # (1, L)
    ligand_mask: torch.Tensor,  # (1, M)
    residue_index: torch.Tensor,  # (1, L)
    bond_matrix: torch.Tensor | None,
    K: int,
    eps: float,
    seed: int,
    variants: frozenset[str],
) -> dict[str, float]:
    device = seq_clean.device
    seq_mask_id = int(module.mask_token_id)
    struc_mask_id = int(module.mask_index_struc_tokens)
    lig_atom_mask_id = int(module.ligand_mask_token_id)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    t_values = stratified_t_samples(K, eps=eps, device=device, generator=generator)
    t_clean = torch.ones(K, device=device)

    seq_K = seq_clean.expand(K, -1).contiguous()
    struc_K = struc_clean.expand(K, -1).contiguous()
    lig_atom_K = lig_atom_clean.expand(K, -1).contiguous()
    lig_struc_K = lig_struc_clean.expand(K, -1).contiguous()
    pmask_K = protein_mask.expand(K, -1).contiguous()
    lmask_K = ligand_mask.expand(K, -1).contiguous()
    residue_K = residue_index.expand(K, -1).contiguous()

    needs_seq = "seq" in variants or "joint_protein" in variants or "joint_all" in variants
    needs_struc = "struc" in variants or "joint_protein" in variants or "joint_all" in variants
    needs_la = "lig_atom" in variants or "joint_ligand" in variants or "joint_all" in variants
    needs_ls = "lig_struc" in variants or "joint_ligand" in variants or "joint_all" in variants

    out: dict[str, float] = {}

    def _score_single_modality(
        modality: str,
        target_K: torch.Tensor,
        mask_id: int,
        valid_K: torch.Tensor,
        logits_key: str,
    ) -> float:
        x_t, mask_pos = absorbing_corrupt(target_K, t_values, mask_id, valid_K, generator)
        kwargs = {
            "seq_tokens": seq_K,
            "struc_tokens": struc_K,
            "lig_atom_tokens": lig_atom_K,
            "lig_struc_tokens": lig_struc_K,
        }
        kwargs[
            {
                "seq": "seq_tokens",
                "struc": "struc_tokens",
                "lig_atom": "lig_atom_tokens",
                "lig_struc": "lig_struc_tokens",
            }[modality]
        ] = x_t
        timesteps = {
            "sequence_tokens": t_clean,
            "structure_tokens": t_clean,
            "ligand_atom_tokens": t_clean,
            "ligand_structure_tokens": t_clean,
        }
        timesteps[
            {
                "seq": "sequence_tokens",
                "struc": "structure_tokens",
                "lig_atom": "ligand_atom_tokens",
                "lig_struc": "ligand_structure_tokens",
            }[modality]
        ] = t_values
        forward_out = _kbatch_forward_pl(
            module,
            protein_mask=pmask_K,
            ligand_mask=lmask_K,
            residue_index=residue_K,
            bond_matrix=bond_matrix,
            timesteps=timesteps,
            **kwargs,
        )
        avg_ce, _, _ = ce_on_masked(forward_out[logits_key], target_K, mask_pos)
        return float(avg_ce.mean().item())

    if needs_seq:
        out["seq"] = _score_single_modality("seq", seq_K, seq_mask_id, pmask_K, "sequence_logits")
    if needs_struc:
        out["struc"] = _score_single_modality("struc", struc_K, struc_mask_id, pmask_K, "structure_logits")
    if needs_la:
        out["lig_atom"] = _score_single_modality(
            "lig_atom", lig_atom_K, lig_atom_mask_id, lmask_K, "ligand_atom_logits"
        )
    if needs_ls:
        out["lig_struc"] = _score_single_modality(
            "lig_struc", lig_struc_K, struc_mask_id, lmask_K, "ligand_structure_logits"
        )

    if "joint_protein" in variants:
        out["joint_protein"] = out["seq"] + out["struc"]
    if "joint_ligand" in variants:
        out["joint_ligand"] = out["lig_atom"] + out["lig_struc"]
    if "joint_all" in variants:
        out["joint_all"] = out["seq"] + out["struc"] + out["lig_atom"] + out["lig_struc"]

    if "joint_true_4" in variants:
        seq_x, seq_mp = absorbing_corrupt(seq_K, t_values, seq_mask_id, pmask_K, generator)
        struc_x, struc_mp = absorbing_corrupt(struc_K, t_values, struc_mask_id, pmask_K, generator)
        la_x, la_mp = absorbing_corrupt(lig_atom_K, t_values, lig_atom_mask_id, lmask_K, generator)
        ls_x, ls_mp = absorbing_corrupt(lig_struc_K, t_values, struc_mask_id, lmask_K, generator)
        timesteps = {
            "sequence_tokens": t_values,
            "structure_tokens": t_values,
            "ligand_atom_tokens": t_values,
            "ligand_structure_tokens": t_values,
        }
        forward_out = _kbatch_forward_pl(
            module,
            seq_tokens=seq_x,
            struc_tokens=struc_x,
            lig_atom_tokens=la_x,
            lig_struc_tokens=ls_x,
            protein_mask=pmask_K,
            ligand_mask=lmask_K,
            residue_index=residue_K,
            bond_matrix=bond_matrix,
            timesteps=timesteps,
        )
        _, sum_s, n_s = ce_on_masked(forward_out["sequence_logits"], seq_K, seq_mp)
        _, sum_st, n_st = ce_on_masked(forward_out["structure_logits"], struc_K, struc_mp)
        _, sum_la, n_la = ce_on_masked(forward_out["ligand_atom_logits"], lig_atom_K, la_mp)
        _, sum_ls, n_ls = ce_on_masked(forward_out["ligand_structure_logits"], lig_struc_K, ls_mp)
        n_total = n_s + n_st + n_la + n_ls
        sum_total = sum_s + sum_st + sum_la + sum_ls
        out["joint_true_4"] = (sum_total / n_total.clamp(min=1.0)).mean().item()

    # Drop modality scalars that weren't explicitly requested but were computed
    # only as additive-joint dependencies.
    for k in ("seq", "struc", "lig_atom", "lig_struc"):
        if k not in variants and k in out:
            out.pop(k)

    return out


@torch.no_grad()
def score_protein_ligand_pll(
    module,
    *,
    sequence_tokens: torch.Tensor,
    structure_tokens: torch.Tensor,
    protein_mask: torch.Tensor,
    ligand_atom_tokens: torch.Tensor,
    ligand_structure_tokens: torch.Tensor,
    ligand_mask: torch.Tensor,
    residue_index: torch.Tensor | None = None,
    bond_matrix: torch.Tensor | None = None,
    K: int = 32,
    eps: float = _STRATIFIED_EPS_DEFAULT,
    seed: int = 0,
    variants: tuple[str, ...] = PROTEIN_LIGAND_VARIANTS,
) -> dict[str, torch.Tensor]:
    """4-modality PLL for the protein-ligand LeFlur checkpoint.

    Parameters
    ----------
    sequence_tokens, structure_tokens : Tensor ``(B, L)``
    ligand_atom_tokens, ligand_structure_tokens : Tensor ``(B, M)``
    protein_mask, ligand_mask : Tensor of same shape as the respective tokens
    residue_index : Tensor ``(B, L)`` or ``None``
    bond_matrix : optional bond-matrix tensor consumed by the encoder
    K : Monte-Carlo draws per modality
    eps : stratified-t margin (default 0.02)
    seed : per-sample base seed (``seed + b`` is used for sample ``b``)
    variants : subset of :data:`PROTEIN_LIGAND_VARIANTS`

    Returns
    -------
    dict mapping each requested variant name to a ``(B,)`` float tensor.
    """
    _validate_variants(variants, allowed=PROTEIN_LIGAND_VARIANTS, kind="protein-ligand")
    variant_set = frozenset(variants)

    B, L = sequence_tokens.shape
    device = sequence_tokens.device

    if structure_tokens.shape != sequence_tokens.shape:
        raise ValueError("structure_tokens shape must match sequence_tokens shape")
    if protein_mask.shape != sequence_tokens.shape:
        raise ValueError("protein_mask shape must match sequence_tokens shape")
    if ligand_structure_tokens.shape != ligand_atom_tokens.shape:
        raise ValueError("ligand_structure_tokens shape must match ligand_atom_tokens shape")
    if ligand_mask.shape != ligand_atom_tokens.shape:
        raise ValueError("ligand_mask shape must match ligand_atom_tokens shape")
    if residue_index is None:
        residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    per_sample: list[dict[str, float]] = []
    for b in range(B):
        bond_b = bond_matrix[b : b + 1] if bond_matrix is not None else None
        result = _score_one_protein_ligand_sample(
            module,
            seq_clean=sequence_tokens[b : b + 1],
            struc_clean=structure_tokens[b : b + 1],
            lig_atom_clean=ligand_atom_tokens[b : b + 1],
            lig_struc_clean=ligand_structure_tokens[b : b + 1],
            protein_mask=protein_mask[b : b + 1],
            ligand_mask=ligand_mask[b : b + 1],
            residue_index=residue_index[b : b + 1],
            bond_matrix=bond_b,
            K=K,
            eps=eps,
            seed=seed + b,
            variants=variant_set,
        )
        per_sample.append(result)

    keys = sorted(per_sample[0].keys())
    return {k: torch.tensor([r[k] for r in per_sample], device=device) for k in keys}


def _validate_variants(
    variants: tuple[str, ...],
    *,
    allowed: tuple[str, ...],
    kind: str,
) -> None:
    if not variants:
        raise ValueError(f"variants must be non-empty for {kind} scoring")
    unknown = set(variants) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown PLL variant(s) for {kind} scoring: {sorted(unknown)}. Allowed: {list(allowed)}")
