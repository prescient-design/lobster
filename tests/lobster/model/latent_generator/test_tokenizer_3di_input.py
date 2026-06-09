"""CPU unit test for :class:`Tokenizer3diInput`.

Builds a tiny instance, pushes a synthetic stub batch through it, and
verifies that two training steps decrease the pairwise-L2 loss. No GPU
or processed dataset required — designed to run in any developer
environment in under a few seconds.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

# These are LG-internal dependencies that some developer environments will
# not have installed (`torch_geometric`, `rotary_embedding_torch`,
# `cpdb`, etc.). Skip cleanly if any are missing.
_OPTIONAL_DEPS = ("torch_geometric", "rotary_embedding_torch")
_MISSING_DEPS = [m for m in _OPTIONAL_DEPS if importlib.util.find_spec(m) is None]
if _MISSING_DEPS:
    pytest.skip(
        f"Skipping Tokenizer3diInput tests: missing {_MISSING_DEPS}",
        allow_module_level=True,
    )

from lobster.model.latent_generator.structure_decoder import DecoderFactory, ViTDecoder  # noqa: E402
from lobster.model.latent_generator.tokenizer import (  # noqa: E402
    L2Loss,
    LossFactory,
    PairWiseL2Loss,
    Tokenizer3diInput,
)


def _build_tiny_model(struc_token_dim: int = 64, data_fixed_size: int = 64) -> Tokenizer3diInput:
    """Construct a tiny `Tokenizer3diInput` that fits in CPU memory."""
    decoder = ViTDecoder(
        # 20 3Di classes + 1 pad row consumed via the decoder's `indexed=True` embedding.
        struc_token_codebook_size=21,
        indexed=True,
        struc_token_dim=struc_token_dim,
        data_fixed_size=data_fixed_size,
        n_atoms=3,
        uvit_n_layers=2,
        uvit_n_heads=4,
        uvit_dim_head=8,
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

    def optim_fn(params, **kw):
        return torch.optim.Adam(params, lr=1e-3)

    def sched_fn(optimizer, **kw):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    return Tokenizer3diInput(
        decoder_factory=decoder_factory,
        loss_factory=loss_factory,
        optim=optim_fn,
        lr_scheduler=sched_fn,
        num_3di_classes=20,
        num_warmup_steps=10,
        num_training_steps=100,
    )


def _build_stub_batch(B: int = 2, L: int = 32) -> dict[str, torch.Tensor]:
    """Random 3Di states + a smooth synthetic backbone target."""
    torch.manual_seed(0)
    states = torch.randint(0, 20, (B, L), dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.float32)
    indices = torch.arange(L, dtype=torch.long)[None].repeat(B, 1)
    # Synthetic helix-ish backbone: smooth alpha-like Cα drift along Z, with
    # offset N and C atoms relative to Cα.
    z = torch.arange(L, dtype=torch.float32).unsqueeze(-1) * 1.5  # (L, 1)
    ca = torch.cat([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)  # (L, 3)
    n = ca + torch.tensor([0.0, 1.0, 0.0])
    c = ca + torch.tensor([1.0, 0.0, 0.5])
    coords = torch.stack([n, ca, c], dim=1)  # (L, 3, 3) i.e. (L, atoms=3, xyz=3)
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)
    return {
        "3di_states": states,
        "mask": mask,
        "indices": indices,
        "coords_res": coords,
    }


def test_forward_shape() -> None:
    model = _build_tiny_model()
    batch = _build_stub_batch()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == batch["coords_res"].shape, (
        f"expected {tuple(batch['coords_res'].shape)}, got {tuple(out.shape)}"
    )


def test_loss_decreases_two_steps() -> None:
    model = _build_tiny_model()
    batch = _build_stub_batch()

    optim_cfg = model.configure_optimizers()
    optimizer = optim_cfg["optimizer"]

    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad()
        out = model._single_step(batch, split="train")
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    # Two SGD steps should drive the loss down from the initial value.
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
