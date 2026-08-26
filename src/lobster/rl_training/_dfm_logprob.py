"""Back-compat shim: moved to :mod:`plm_design_rl.logprob.dfm`.

The DFM log-probability kernels now live in the standalone ``plm_design_rl`` package.
This module aliases the old import path to the moved module so existing imports
(``from lobster.rl_training._dfm_logprob import dfm_step_logprob`` etc.) keep resolving
to the exact same objects.
"""

import sys

from plm_design_rl.logprob import dfm as _m

sys.modules[__name__] = _m
