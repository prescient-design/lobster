"""Back-compat shim: moved to :mod:`plm_design_rl.pool.queue`.

The LigandMPNN-repack shape-complementarity worker-pool client now lives in the
standalone ``plm_design_rl`` package. This module aliases the old import path to the
moved module so existing imports
(``from lobster.rl_training.rewards._shape_reward_pool import ShapeRewardClient`` etc.)
keep resolving to the exact same objects.
"""

import sys

from plm_design_rl.pool import queue as _m

sys.modules[__name__] = _m
