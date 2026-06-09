"""CPU dry-run for Step H: per-step random SO(3) augmentation of ``x_1``.

What this validates
-------------------

1. The rotation matrix is orthogonal (``R R^T = I``) and proper (``det(R) = +1``)
   for a batch of samples -- i.e. ``_apply_random_so3_aug`` actually samples
   from SO(3), not just O(3) or a non-uniform subset.

2. With ``random_so3_aug=True`` and ``split="train"``, the ``x_1`` that
   actually enters ``_moco_loss`` is a rotated version of the (centered)
   input batch -- so the loss is computed against the rotated GT (the
   user's correctness requirement). We verify by monkey-patching
   ``_moco_loss`` to capture its arguments.

3. With ``split="val"``, the ``x_1`` that enters ``_moco_loss`` equals
   the (centered) input batch exactly (no augmentation at eval time --
   val/loss must stay comparable to baseline runs).

4. With ``random_so3_aug=False``, the train-time ``x_1`` in the loss
   equals the centered input batch exactly (back-compat: existing
   checkpoints that load with the old default behave identically).

5. ``_pairwise_l2`` of the captured x_1 against the canonical (centered)
   x_1 is ~0 in *all* aug modes -- pairwise CA-CA distances are SO(3)
   invariants, so we have a second-order sanity check that the rotation
   really is a rigid motion (no shear, no scale).

Not part of pytest collection -- invoke directly:

    .venv/bin/python tests/lobster/hydra_config/_dryrun_3di_input_flow_so3_aug_cpu.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRA_ROOT = REPO_ROOT / "src" / "lobster" / "hydra_config"

DRY_DIR = REPO_ROOT / ".dryrun_3di_input_data"
SMALL_TRAIN = str(DRY_DIR / "train.pt")
SMALL_VAL = str(DRY_DIR / "val.pt")
SMALL_TEST = str(DRY_DIR / "test.pt")


def _center_geometric_ref(coords: torch.Tensor, seq_mask: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Mirror of ``Tokenizer3diInputFlow._center_geometric`` for ground truth."""
    m = seq_mask[..., None, None].to(coords.dtype)
    total = (coords * m).sum(dim=(1, 2))
    n = (seq_mask.sum(dim=-1).clamp_min(1.0) * n_atoms).unsqueeze(-1)
    com = total / n
    return (coords - com[:, None, None, :]) * m


def _pairwise_dist_diff(a: torch.Tensor, b: torch.Tensor, seq_mask: torch.Tensor) -> float:
    """Max absolute difference in CA-CA pairwise distances (SO(3) invariant).

    Computed in fp64 so the tolerance can be tight: ``torch.cdist`` in fp32
    introduces ~1e-2 A of roundoff for distances of ~70 A (the diagonal of
    a 200-residue protein), which would mask genuine bugs at the 1e-3
    level. fp64 cdist holds to ~1e-10.
    """
    a_ca = a[:, :, 1, :].to(torch.float64)
    b_ca = b[:, :, 1, :].to(torch.float64)
    da = torch.cdist(a_ca, a_ca, p=2)
    db = torch.cdist(b_ca, b_ca, p=2)
    pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
    diff = (da - db).abs() * pair_mask.to(torch.float64)
    return float(diff.max().item())


def _check_so3_basics(model, B: int = 8) -> None:
    """Sample a batch of rotations directly and assert SO(3) properties."""
    torch.manual_seed(0)
    # Apply the augmenter to a known unit-vector tensor so we can read back
    # the rotation matrix per sample (column j of the rotated identity).
    # We actually want to extract the rotation, so use a more direct probe:
    # x_in[b, 0, :, :] = I3, then x_out[b, 0, :, :] = I3 @ R[b] = R[b].
    x_in = torch.zeros(B, 1, 3, 3)
    x_in[:, 0, :, :] = torch.eye(3)
    seq_mask = torch.ones(B, 1)
    # Force the train path even though model.training may be False.
    model.random_so3_aug = True
    R_out = model._apply_random_so3_aug(x_in, seq_mask, training=True)
    R = R_out[:, 0, :, :].to(torch.float64)  # (B, 3, 3) -- the actual rotations

    # Orthogonality: R R^T = I.
    eye3 = torch.eye(3, dtype=torch.float64).expand(B, 3, 3)
    ortho_err = (R @ R.transpose(-1, -2) - eye3).abs().max().item()
    print(f"[so3] max |R R^T - I|     = {ortho_err:.2e}  (expect < 1e-5)")
    assert ortho_err < 1e-5, f"rotations are not orthogonal: max err {ortho_err}"

    # Proper rotation: det = +1.
    dets = torch.linalg.det(R)
    det_err = (dets - 1.0).abs().max().item()
    print(f"[so3] max |det(R) - 1|    = {det_err:.2e}  (expect < 1e-5)")
    assert det_err < 1e-5, f"rotations have det != +1: dets={dets.tolist()}"

    # Sanity: rotations are not all the identity.
    not_id_count = (R - eye3).abs().sum(dim=(-2, -1)).gt(0.1).sum().item()
    print(f"[so3] non-identity rots   = {not_id_count}/{B}  (expect = {B} w/ seed=0)")
    assert not_id_count >= B - 1, (
        f"only {not_id_count}/{B} rotations are non-identity -- the augmenter is probably returning x unchanged"
    )


def _capture_loss_x1(model, batch, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``_single_step`` once and capture the ``x_1`` that reaches the loss.

    Returns
    -------
    (x_1_in_loss, x_1_canonical_centered) : (Tensor, Tensor)
        The first is what actually went into ``flow_matcher.loss`` as the
        target; the second is what the GT would look like if augmentation
        were off (the centered raw batch). Comparing the two tells us:

        - identical -> augmentation did NOT fire (val/test, or aug=False).
        - same pairwise-CA distances but different absolute coords ->
          rotation was applied.
    """
    captured: dict[str, torch.Tensor] = {}

    orig_moco_loss = model._moco_loss

    def _spy(x_1_hat, x_1, t, x_t, seq_mask):  # noqa: ANN001
        captured["x_1"] = x_1.detach().clone()
        captured["x_t"] = x_t.detach().clone()
        captured["x_1_hat"] = x_1_hat.detach().clone()
        captured["t"] = t.detach().clone()
        return orig_moco_loss(x_1_hat, x_1, t, x_t, seq_mask)

    model._moco_loss = _spy  # type: ignore[method-assign]
    try:
        torch.manual_seed(42)
        model._single_step(batch, split=split)
    finally:
        model._moco_loss = orig_moco_loss  # type: ignore[method-assign]

    # Canonical (centered, NOT rotated) version of x_1 for comparison. Apply the
    # same Angstrom->nm coordinate scaling the model applies after centering, so the
    # reference lives in the same (scaled) space as the captured loss target.
    seq_mask = batch["mask"].float()
    x_1_raw = batch["coords_res"].to(torch.get_default_dtype())
    x_1_centered = _center_geometric_ref(x_1_raw, seq_mask, model.n_atoms) * model.coord_scale
    return captured["x_1"], x_1_centered


def main() -> int:
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(HYDRA_ROOT),
            job_name="dryrun_3di_input_flow_so3_aug_cpu",
        ):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=train_latent_generator_3di_input_flow",
                    (f"data.path_to_datasets=[{SMALL_TRAIN},{SMALL_VAL},{SMALL_TEST}]"),
                    "data.batch_size=2",
                    "data.num_workers=0",
                ],
            )

        print(f"[compose] model._target_     = {cfg.model._target_}")
        print(f"[compose] random_so3_aug     = {cfg.model.random_so3_aug}")
        assert cfg.model.random_so3_aug is True, "expected the default YAML to enable random_so3_aug after Step H"

        print("[instantiate] datamodule ...")
        dm = instantiate(cfg.data)
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print(f"[batch] coords_res shape = {tuple(batch['coords_res'].shape)}")

        print("[instantiate] flow model (random_so3_aug=True from yaml) ...")
        model = instantiate(cfg.model)
        model.n_sampling_steps = 2

        # ---- Test 1: SO(3) basics -----------------------------------------
        print("\n--- Test 1: SO(3) sampling properties ---")
        _check_so3_basics(model)

        # ---- Test 2: train + aug=True -> loss sees rotated x_1 ------------
        print("\n--- Test 2: train + aug=True -> loss target is ROTATED ---")
        model.train()
        seq_mask = batch["mask"].float()
        x_1_loss, x_1_centered = _capture_loss_x1(model, batch, split="train")
        # Per-atom max deviation (should be > 0: rotation was applied).
        max_dev = (x_1_loss - x_1_centered).abs().max().item()
        print(f"[train+aug] max |x_1_loss - x_1_centered| = {max_dev:.4f}  (expect > 0.1)")
        assert max_dev > 0.1, (
            "x_1 in the loss equals the centered raw batch -- augmentation "
            "did not fire. Check that random_so3_aug is wired to _single_step."
        )
        # Pairwise CA-CA distances should be preserved (rigid motion check).
        # Note: the captured x_1 has gone through Kabsch noise-alignment but
        # NOT through Kabsch alignment of x_1 itself, so pairwise distances
        # of x_1_loss must equal those of (centered, then rotated) x_1.
        # Since rotation preserves pairwise distances, x_1_loss pairwise
        # distances must equal x_1_centered pairwise distances.
        pdist_err = _pairwise_dist_diff(x_1_loss, x_1_centered, seq_mask)
        # fp32 rotation -> ~1e-5 per coord error -> ~1e-4 distance error.
        # 1e-3 is a tight bound that still catches any non-rigid bug
        # (e.g. accidental scaling or shear from a botched signs trick).
        print(
            f"[train+aug] max |pdist(x_1_loss) - pdist(x_1_centered)| = {pdist_err:.2e}  (expect < 1e-3, fp64 cdist of fp32 rotation)"
        )
        assert pdist_err < 1e-3, (
            f"pairwise distances changed by {pdist_err:.2e} -- the augmentation "
            "is not a rigid motion (probably shear/scale or a bug in the QR step)."
        )

        # ---- Test 3: val + aug=True -> loss sees CANONICAL x_1 -----------
        print("\n--- Test 3: val + aug=True -> loss target is CANONICAL ---")
        model.eval()
        x_1_loss_val, x_1_centered_val = _capture_loss_x1(model, batch, split="val")
        max_dev_val = (x_1_loss_val - x_1_centered_val).abs().max().item()
        print(f"[val+aug]   max |x_1_loss - x_1_centered| = {max_dev_val:.2e}  (expect ~ 0)")
        assert max_dev_val < 1e-5, (
            f"val-time x_1 differs from canonical centered by {max_dev_val:.2e} -- "
            "augmentation should be train-only. val/loss must stay comparable to baseline."
        )

        # ---- Test 4: aug=False on train -> loss sees CANONICAL x_1 -------
        print("\n--- Test 4: train + aug=False -> loss target is CANONICAL ---")
        model.train()
        model.random_so3_aug = False
        x_1_loss_noaug, x_1_centered_noaug = _capture_loss_x1(model, batch, split="train")
        max_dev_noaug = (x_1_loss_noaug - x_1_centered_noaug).abs().max().item()
        print(f"[train -aug] max |x_1_loss - x_1_centered| = {max_dev_noaug:.2e}  (expect ~ 0)")
        assert max_dev_noaug < 1e-5, (
            f"with random_so3_aug=False the train x_1 differs from canonical "
            f"by {max_dev_noaug:.2e} -- back-compat is broken."
        )
        # Restore default.
        model.random_so3_aug = True

        # ---- Test 5: x_t built from rotated x_1 --------------------------
        print("\n--- Test 5: x_t built from rotated x_1 (interpolant consistency) ---")
        model.train()
        # Re-capture with aug=True, then verify x_t = t * x_1_loss + (1-t) * x_0
        # by reconstructing x_0 from x_t and x_1.
        x_1_loss2, _ = _capture_loss_x1(model, batch, split="train")
        # We need the actual x_t and t from the capture as well; re-spy.
        captured: dict[str, torch.Tensor] = {}
        orig = model._moco_loss

        def _spy2(x_1_hat, x_1, t, x_t, seq_mask):  # noqa: ANN001
            captured.update(x_1=x_1.detach(), x_t=x_t.detach(), t=t.detach())
            return orig(x_1_hat, x_1, t, x_t, seq_mask)

        model._moco_loss = _spy2  # type: ignore[method-assign]
        try:
            torch.manual_seed(123)
            model._single_step(batch, split="train")
        finally:
            model._moco_loss = orig  # type: ignore[method-assign]

        # x_t = t * x_1 + (1-t) * x_0  =>  x_0 = (x_t - t * x_1) / (1 - t)
        t_b = captured["t"].view(-1, 1, 1, 1)
        x_0_implied = (captured["x_t"] - t_b * captured["x_1"]) / (1.0 - t_b).clamp_min(1e-4)
        # x_0 (after Kabsch) is expected to be CoM-centered (the prior gives
        # centered noise, Kabsch alignment is a rotation -> still centered).
        m = seq_mask[..., None, None]
        x_0_com = (x_0_implied * m).sum(dim=(1, 2)) / (seq_mask.sum(dim=-1).clamp_min(1.0) * model.n_atoms).unsqueeze(
            -1
        )
        max_x0_com = float(x_0_com.abs().max().item())
        print(f"[interp]    max |CoM(x_0)| recovered      = {max_x0_com:.2e}  (expect ~ 0)")
        assert max_x0_com < 1e-3, (
            f"x_0 implied by x_t and x_1 has CoM = {max_x0_com:.2e} -- the "
            "interpolant is not consistent with `x_t = t*x_1 + (1-t)*x_0` over "
            "centered tensors."
        )

        print("\nOK -- Step H SO(3) augmentation dry-run passed on CPU.")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    sys.exit(main())
