"""GPU overfit smoke for :class:`Tokenizer3diInput`.

Marked ``@pytest.mark.slow`` — runs only when explicitly requested
(``pytest -m slow``). It loads up to 16 random Step-A ``.pt`` files
from ``swissprot/structures_v6/``, runs ``Structure3diTransform`` to
materialise the 3Di states, then trains the 3Di-input variant to
overfit them. The final RMSD-to-input must be ≤ 4 Å — a generous bar
chosen to catch wiring regressions, not to grade reconstruction quality.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
from pathlib import Path

import pytest
import torch

_OPTIONAL_DEPS = ("torch_geometric", "rotary_embedding_torch", "mini3di")
_MISSING = [m for m in _OPTIONAL_DEPS if importlib.util.find_spec(m) is None]
if _MISSING:
    pytest.skip(
        f"Skipping Tokenizer3diInput overfit smoke: missing {_MISSING}",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    pytest.skip("Tokenizer3diInput overfit smoke requires a GPU", allow_module_level=True)

STRUCTURES_DIR = Path(
    os.environ.get(
        "SWISSPROT_STRUCTURES_DIR",
        "/cv/data/ai4dd/data/prescient-modal/datasets/swissprot/structures_v6",
    )
)

if not STRUCTURES_DIR.is_dir():
    pytest.skip(
        f"Step-A structures dir {STRUCTURES_DIR} not present; skipping Tokenizer3diInput overfit smoke.",
        allow_module_level=True,
    )


pytestmark = pytest.mark.slow


def _sample_pt_paths(n: int) -> list[Path]:
    paths = list(STRUCTURES_DIR.glob("AF-*-F1-model_v6.pt"))
    if not paths:
        pytest.skip(f"No .pt files in {STRUCTURES_DIR}")
    random.seed(0)
    random.shuffle(paths)
    return paths[:n]


def _load_batch(paths: list[Path], max_len: int = 256, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Right-pad a small homogeneous batch of `.pt` items with 3Di states."""
    from lobster.transforms._structure_transforms import Structure3diTransform

    tr = Structure3diTransform()
    items = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        L = int(d["coords_res"].shape[0])
        if L > max_len:
            d["coords_res"] = d["coords_res"][:max_len]
            d["sequence"] = d["sequence"][:max_len]
            d["mask"] = d["mask"][:max_len]
            d["indices"] = d["indices"][:max_len]
            d["chains_ids"] = d["chains_ids"][:max_len]
        d = tr(d)
        items.append(d)

    lengths = [int(it["coords_res"].shape[0]) for it in items]
    B = len(items)
    L = max(lengths)
    coords = torch.zeros(B, L, 3, 3, dtype=torch.float32)
    mask = torch.zeros(B, L, dtype=torch.float32)
    indices = torch.zeros(B, L, dtype=torch.long)
    states = torch.full((B, L), 20, dtype=torch.long)  # pad row
    for i, it in enumerate(items):
        n = lengths[i]
        coords[i, :n] = it["coords_res"]
        mask[i, :n] = 1.0
        indices[i, :n] = it["indices"].to(torch.long)
        states[i, :n] = it["3di_states"].to(torch.long)

    return {
        "3di_states": states.to(device),
        "mask": mask.to(device),
        "indices": indices.to(device),
        "coords_res": coords.to(device),
    }


def _rmsd_per_residue(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean per-residue Cα RMSD (Å), masked. No SVD alignment — the L2 loss
    is itself align-invariant via the LG `L2Loss`, but this smoke check just
    needs a sane upper bound that catches "model returns garbage" regressions.
    """
    d = pred[..., 1, :] - gt[..., 1, :]  # CA difference
    sq = (d**2).sum(dim=-1)  # (B, L)
    sq = sq * mask
    n = mask.sum().clamp(min=1.0)
    return math.sqrt((sq.sum() / n).item())


def test_overfit_swissprot_subset() -> None:
    from lobster.model.latent_generator.structure_decoder import DecoderFactory, ViTDecoder
    from lobster.model.latent_generator.tokenizer import (
        L2Loss,
        LossFactory,
        PairWiseL2Loss,
        Tokenizer3diInput,
    )

    paths = _sample_pt_paths(16)
    batch = _load_batch(paths, max_len=256)

    L_max = int(batch["mask"].shape[1])
    decoder = ViTDecoder(
        struc_token_codebook_size=21,  # 20 3Di classes + 1 pad row
        indexed=True,
        struc_token_dim=256,
        data_fixed_size=max(L_max, 256),
        n_atoms=3,
        uvit_n_layers=4,
        uvit_n_heads=8,
        uvit_dim_head=32,
        uvit_position_embedding_type="rotary",
        encode_ligand=False,
    )
    decoder_factory = DecoderFactory.from_mapping(
        decoder_mapping={"vit_decoder": decoder},
        decoder2loss_dict={"vit_decoder": ["l2_loss", "pairwise_l2_loss"]},
    )
    loss_factory = LossFactory.from_mapping(
        loss_mapping={"l2_loss": L2Loss(), "pairwise_l2_loss": PairWiseL2Loss()},
        weight_dict={"l2_loss": 0.01, "pairwise_l2_loss": 1.0},
    )

    model = Tokenizer3diInput(
        decoder_factory=decoder_factory,
        loss_factory=loss_factory,
        optim=lambda params, **kw: torch.optim.Adam(params, lr=3e-4),
        lr_scheduler=lambda optimizer, **kw: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
        num_3di_classes=20,
    ).cuda()

    opt = model.configure_optimizers()["optimizer"]
    for _ in range(200):
        opt.zero_grad()
        out = model._single_step(batch, split="train")
        out["loss"].backward()
        opt.step()

    with torch.no_grad():
        pred = model(batch)
    rmsd = _rmsd_per_residue(pred, batch["coords_res"], batch["mask"])
    # 4 Å is a generous bar — actual overfit should be << 1 Å, but this
    # test exists only to catch wiring regressions, not to grade quality.
    assert rmsd < 4.0, f"final CA RMSD {rmsd:.2f} Å exceeds 4 Å — wiring regression?"
