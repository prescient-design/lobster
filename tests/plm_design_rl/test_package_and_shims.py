"""Import + shim-parity tests for the extracted ``plm_design_rl`` package.

Step 1 of the refactor moved the (pure) reward terms, the DFM log-prob kernels, and the
worker-pool queue client out of ``lobster.rl_training`` into the standalone
``plm_design_rl`` package, leaving transparent back-compat shims at every old import
path. These tests assert both that the new package imports cleanly and that the old paths
resolve to the *exact same* module/function objects (so no consumer changed behavior).
"""

from __future__ import annotations

import importlib
import sys

import pytest

# The 10 reward-term submodules moved verbatim into ``plm_design_rl.rewards``.
_REWARD_SUBMODULES = [
    "_aar_reward",
    "_chainbreak_reward",
    "_clash_reward",
    "_distribution_reward",
    "_diversity_reward",
    "_protenix_reward",
    "_protenix_structure_expert",
    "_sc_clash_reward",
    "_shape_reward",
    "_structure_reward",
]

# A representative public symbol per submodule, to assert function-identity parity.
_REWARD_SYMBOLS = {
    "_aar_reward": "aar_terms",
    "_chainbreak_reward": "chainbreak_reward",
    "_clash_reward": "clash_contact_reward",
    "_distribution_reward": "distribution_terms",
    "_diversity_reward": "linguistic_complexity",
    "_protenix_reward": "ProtenixRewardClient",
    "_protenix_structure_expert": "assemble_structure_expert",
    "_sc_clash_reward": "sc_clash_reward",
    "_shape_reward": "shape_complementarity_reward_atoms",
    "_structure_reward": "structure_terms",
}


def test_top_level_package_imports() -> None:
    pkg = importlib.import_module("plm_design_rl")
    for sub in ("rewards", "logprob", "pool"):
        assert hasattr(pkg, sub)
        importlib.import_module(f"plm_design_rl.{sub}")


def test_new_reward_submodules_import() -> None:
    for sub in _REWARD_SUBMODULES:
        importlib.import_module(f"plm_design_rl.rewards.{sub}")


@pytest.mark.parametrize("sub", _REWARD_SUBMODULES)
def test_reward_submodule_shim_is_same_object(sub: str) -> None:
    """The old ``lobster.rl_training.rewards._X`` path IS the moved module object."""
    old = importlib.import_module(f"lobster.rl_training.rewards.{sub}")
    new = importlib.import_module(f"plm_design_rl.rewards.{sub}")
    assert old is new
    assert sys.modules[f"lobster.rl_training.rewards.{sub}"] is new


@pytest.mark.parametrize("sub,symbol", sorted(_REWARD_SYMBOLS.items()))
def test_reward_symbol_identity(sub: str, symbol: str) -> None:
    old = importlib.import_module(f"lobster.rl_training.rewards.{sub}")
    new = importlib.import_module(f"plm_design_rl.rewards.{sub}")
    assert getattr(old, symbol) is getattr(new, symbol)


def test_dfm_logprob_shim() -> None:
    old = importlib.import_module("lobster.rl_training._dfm_logprob")
    new = importlib.import_module("plm_design_rl.logprob.dfm")
    assert old is new
    assert sys.modules["lobster.rl_training._dfm_logprob"] is new
    for fn in ("dfm_step_prob", "dfm_step_logprob", "dfm_step_kl"):
        assert getattr(old, fn) is getattr(new, fn)
    # also reachable from the logprob subpackage namespace
    logprob = importlib.import_module("plm_design_rl.logprob")
    assert logprob.dfm_step_logprob is new.dfm_step_logprob


def test_shape_reward_pool_shim() -> None:
    old = importlib.import_module("lobster.rl_training.rewards._shape_reward_pool")
    new = importlib.import_module("plm_design_rl.pool.queue")
    assert old is new
    assert sys.modules["lobster.rl_training.rewards._shape_reward_pool"] is new
    for sym in ("ShapeRewardClient", "reward_from_shape"):
        assert getattr(old, sym) is getattr(new, sym)


def test_rewards_package_namespace_parity() -> None:
    """The old ``rewards`` package re-exports the identical public surface."""
    old = importlib.import_module("lobster.rl_training.rewards")
    new_rewards = importlib.import_module("plm_design_rl.rewards")
    new_pool = importlib.import_module("plm_design_rl.pool")

    # Everything the new rewards package exports is identical at the old path.
    for name in new_rewards.__all__:
        assert hasattr(old, name), name
        assert getattr(old, name) is getattr(new_rewards, name), name

    # The pool client surface preserved at the historical rewards namespace.
    for name in ("ShapeRewardClient", "reward_from_shape"):
        assert getattr(old, name) is getattr(new_pool, name), name


def test_pool_reexports_protenix_client() -> None:
    pool = importlib.import_module("plm_design_rl.pool")
    protenix = importlib.import_module("plm_design_rl.rewards._protenix_reward")
    for name in ("ProtenixRewardClient", "atomic_write_json", "ensure_queue"):
        assert getattr(pool, name) is getattr(protenix, name), name


def test_rl_training_top_level_unchanged() -> None:
    """``lobster.rl_training`` still exposes its public API through the shims."""
    r = importlib.import_module("lobster.rl_training")
    for name in ("dfm_step_logprob", "dfm_step_kl", "dfm_step_prob"):
        assert hasattr(r, name), name
    assert r.dfm_step_logprob is importlib.import_module("plm_design_rl.logprob.dfm").dfm_step_logprob
