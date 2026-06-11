"""Strip optimizer/scheduler state from a Lightning checkpoint.

When you `Trainer.fit(ckpt_path=ckpt)`, Lightning restores model weights
AND `optimizer_states` AND `lr_schedulers` AND step counters from the
ckpt. The optimizer state restore overwrites any LR you tried to set via
`+model.optim.lr=...` on the Hydra command line: param_groups[0]['lr']
comes from the ckpt, not from cfg.

To force a new LR on resume, write a stripped ckpt that omits
`optimizer_states` and `lr_schedulers`. Lightning then rebuilds them
fresh from `configure_optimizers()` at the cfg's LR.

Run:

    uv run python scripts/_strip_ckpt_optim.py \
        --src /path/to/epoch=N-step=M-val_loss=L.ckpt \
        --dst /path/to/epoch=N-step=M-val_loss=L_resume_NEWLR.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--reset-step", action="store_true",
                        help="Also drop epoch/global_step (fresh wandb timeline).")
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(args.src)
    print(f"loading {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)

    keys_before = sorted(ckpt.keys())
    print(f"keys before: {keys_before}")

    for k in ("optimizer_states", "lr_schedulers"):
        if k in ckpt:
            print(f"  dropping `{k}` ({len(ckpt[k])} entry(ies))")
            del ckpt[k]
    if args.reset_step:
        for k in ("epoch", "global_step"):
            if k in ckpt:
                print(f"  resetting `{k}` (was {ckpt[k]})")
                del ckpt[k]

    print(f"keys after:  {sorted(ckpt.keys())}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"wrote {args.dst}")
    print(f"  src size {args.src.stat().st_size / 1e9:.2f} GB")
    print(f"  dst size {args.dst.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
