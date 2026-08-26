"""Out-of-process reward worker-pool clients + queue protocol.

Heavy reward oracles (Protenix co-folding, LigandMPNN side-chain repack + 3D-Zernike
shape complementarity) run in separate worker processes and communicate over a
filesystem job/result queue, so the policy process stays free of their (GPU / openfold)
dependencies. This subpackage holds the client side of that protocol.

* :mod:`.queue` — the shared filesystem-queue client (:class:`ShapeRewardClient` for the
  LigandMPNN-repack shape-complementarity pool) plus its reduce helper
  (:func:`reward_from_shape`). The queue primitives (:func:`atomic_write_json`,
  :func:`ensure_queue`) and the Protenix confidence client
  (:class:`ProtenixRewardClient`) still live in
  :mod:`plm_design_rl.rewards._protenix_reward` and are re-exported here for convenience.
"""

from plm_design_rl.rewards._protenix_reward import (
    ProtenixRewardClient,
    atomic_write_json,
    ensure_queue,
)

from .queue import ShapeRewardClient, reward_from_shape

__all__ = [
    "ShapeRewardClient",
    "reward_from_shape",
    "ProtenixRewardClient",
    "atomic_write_json",
    "ensure_queue",
]
