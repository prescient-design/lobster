"""Tests for the full-atom 3DZD interface shape-complementarity (SC) reward.

Three layers, mirroring ``test_clash_reward.py``:

* the pure numpy reward math (:mod:`lobster.rl_training.rewards._shape_reward`),
* the policy-side pool client + queue protocol
  (:mod:`lobster.rl_training.rewards._shape_reward_pool`) — exercised end-to-end
  against an in-test fake worker that drains the filesystem queue,
* the trainer wiring (``LeFlurGRPOTrainer._shape_terms_for_group`` + the ``w_shape``
  gate in ``_compute_rewards``), asserting the SC term is weighted, averaged into the
  metrics, and — critically — **byte-identical / inert when ``w_shape == 0``**.

The trainer tests live here (not in ``test_trainers.py``) so they avoid importing
``lobster.rl_training.trainers`` (which pulls in ``trl``, absent in this env).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lobster.rl_training import LeFlurGRPOTrainer
from lobster.rl_training.rewards import (
    ShapeRewardClient,
    reward_from_shape,
    shape_complementarity_reward,
    shape_complementarity_reward_atoms,
)
from lobster.rl_training.rewards._shape_reward import radii_from_elements
from lobster.rl_training.rewards._shape_reward_pool import _design_key


# --------------------------------------------------------------- reward math
def _slab(nx: int, ny: int, z: float, spacing: float = 1.8) -> np.ndarray:
    """A flat rectangular sheet of atoms in the z=const plane."""
    xs = np.arange(nx) * spacing
    ys = np.arange(ny) * spacing
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


class TestShapeComplementarityRewardAtoms:
    def test_bounded_unit_interval_and_diag_keys(self):
        a = _slab(8, 8, z=0.0)
        b = _slab(8, 8, z=3.0)  # facing sheet in contact across the gap
        ar = radii_from_elements(["C"] * len(a))
        br = radii_from_elements(["C"] * len(b))
        term, diag = shape_complementarity_reward_atoms(a, ar, b, br, nsphere=48)
        assert 0.0 <= term <= 1.0
        assert set(diag) >= {"sc", "n_patch_a", "n_patch_b"}
        assert diag["n_patch_a"] > 0 and diag["n_patch_b"] > 0

    def test_deterministic(self):
        a = _slab(6, 6, z=0.0)
        b = _slab(6, 6, z=3.0)
        ar = radii_from_elements(["C"] * len(a))
        br = radii_from_elements(["C"] * len(b))
        t1, _ = shape_complementarity_reward_atoms(a, ar, b, br, nsphere=48)
        t2, _ = shape_complementarity_reward_atoms(a, ar, b, br, nsphere=48)
        assert t1 == t2

    def test_clipped_from_raw_pearson(self):
        """term = clip(sc, 0, 1); an anti-correlated patch (sc<0) floors at 0."""
        a = _slab(8, 8, z=0.0)
        b = _slab(8, 8, z=3.0)
        ar = radii_from_elements(["C"] * len(a))
        br = radii_from_elements(["C"] * len(b))
        term, diag = shape_complementarity_reward_atoms(a, ar, b, br, nsphere=48)
        if np.isfinite(diag["sc"]) and diag["sc"] < 0:
            assert term == 0.0
        else:
            assert term == pytest.approx(float(np.clip(diag["sc"], 0.0, 1.0)))

    def test_radii_from_elements(self):
        r = radii_from_elements(["C", "N", "O", "S"])
        assert r.shape == (4,)
        assert np.all(r > 0)


class TestShapeComplementarityRewardBackbone:
    def test_backbone_entry_bounded(self):
        rng = np.random.default_rng(0)
        ag = rng.normal(size=(30, 3, 3)) * 3.0
        bd = ag + np.array([10.0, 0.0, 0.0])
        coords = np.concatenate([ag, bd], axis=0)
        valid = np.ones(60, dtype=bool)
        binder = np.array([False] * 30 + [True] * 30)
        term, diag = shape_complementarity_reward(coords, valid, binder, nsphere=48)
        assert 0.0 <= term <= 1.0

    def test_empty_chain_is_zero(self):
        coords = np.zeros((10, 3, 3))
        valid = np.ones(10, dtype=bool)
        binder = np.zeros(10, dtype=bool)  # no binder residues
        term, diag = shape_complementarity_reward(coords, valid, binder)
        assert term == 0.0
        assert diag["n_patch_a"] == 0 and diag["n_patch_b"] == 0


# --------------------------------------------------------------- client helpers
class TestRewardFromShape:
    def test_none_is_floor(self):
        assert reward_from_shape(None) == 0.0

    def test_missing_term_is_floor(self):
        assert reward_from_shape({"sc": 0.5}) == 0.0
        assert reward_from_shape({"term": None}) == 0.0

    def test_term_passthrough(self):
        assert reward_from_shape({"term": 0.73}) == pytest.approx(0.73)


class TestDesignKey:
    def _d(self):
        ag = np.arange(9, dtype=np.float32).reshape(3, 3)
        bd = np.arange(9, dtype=np.float32).reshape(3, 3) + 1.0
        return ag, bd

    def test_stable(self):
        ag, bd = self._d()
        k1 = _design_key("t", ag, "AA", bd, "CC")
        k2 = _design_key("t", ag.copy(), "AA", bd.copy(), "CC")
        assert k1 == k2

    def test_sensitive_to_coords_seq_target(self):
        ag, bd = self._d()
        base = _design_key("t", ag, "AA", bd, "CC")
        assert _design_key("t", ag + 0.01, "AA", bd, "CC") != base  # coords
        assert _design_key("t", ag, "AW", bd, "CC") != base  # antigen seq
        assert _design_key("t", ag, "AA", bd, "CW") != base  # binder seq
        assert _design_key("u", ag, "AA", bd, "CC") != base  # target


# ------------------------------------------------- client <-> fake-worker round trip
def _fake_worker(queue_dir, stop_evt: threading.Event, score_fn=None):
    """Drain ``new/`` jobs -> ``done/`` results, imitating the repack server.

    ``score_fn(design_id, ag_bb, bd_bb, ag_seq, bd_seq) -> dict`` produces each design's
    result; defaults to a deterministic stub keyed on the coord sum.
    """
    from lobster.rl_training.rewards._protenix_reward import atomic_write_json

    def _default_score(did, ag, bd, aseq, bseq):
        return {
            "term": float(np.clip(np.tanh(ag.mean() + bd.mean()), 0, 1)),
            "sc": float(np.tanh(ag.mean() + bd.mean())),
            "n_patch_a": int(ag.shape[0]),
            "n_patch_b": int(bd.shape[0]),
        }

    score_fn = score_fn or _default_score
    new_dir = queue_dir / "new"
    while not stop_evt.is_set():
        for jf in sorted(new_dir.glob("*.json")):
            try:
                # claim by rename (mirrors the server), tolerate races
                claimed = queue_dir / "claimed" / jf.name
                jf.rename(claimed)
            except OSError:
                continue
            job = json.loads(claimed.read_text())
            npz = np.load(queue_dir / job["npz"])
            results = {}
            for d in job["designs"]:
                did = d["design_id"]
                results[did] = score_fn(did, npz[f"{did}__ag"], npz[f"{did}__bd"], d["ag_seq"], d["bd_seq"])
            atomic_write_json(
                queue_dir / "done" / f"{job['job_id']}.json",
                {"job_id": job["job_id"], "results": results},
            )
        time.sleep(0.01)


def _mk_designs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        na, nb = 5, 4
        out.append(
            {
                "ag_bb": rng.normal(size=(na, 3, 3)).astype(np.float32),
                "ag_seq": "A" * na,
                "bd_bb": rng.normal(size=(nb, 3, 3)).astype(np.float32),
                "bd_seq": "C" * nb,
            }
        )
    return out


class TestShapeRewardClientRoundTrip:
    def _run_with_worker(self, client, target, designs, score_fn=None):
        stop = threading.Event()
        t = threading.Thread(target=_fake_worker, args=(client.queue, stop, score_fn), daemon=True)
        t.start()
        try:
            return client.score_group(target, designs)
        finally:
            stop.set()
            t.join(timeout=2.0)

    def test_end_to_end_maps_results(self, tmp_path):
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02)
        designs = _mk_designs(3)
        res = self._run_with_worker(client, "t0", designs)
        assert len(res) == 3
        assert all(r is not None and "term" in r for r in res)

    def test_data_dir_and_npz_created(self, tmp_path):
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02)
        assert (client.queue / "data").is_dir()
        self._run_with_worker(client, "t0", _mk_designs(2))
        # at least one npz sidecar was written into data/
        assert list((client.queue / "data").glob("*.npz"))

    def test_sharding_splits_jobs(self, tmp_path):
        """n_shards>1 fans the group across several jobs (verified by counting new/)."""
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02, n_shards=3)
        designs = _mk_designs(6)
        # write only (no worker): count the queued jobs, then drain manually
        seen = {"n": 0}

        def counting_await(job_ids):
            seen["n"] = len(job_ids)
            return {jid: None for jid in job_ids}  # floor everything

        client._await_many = counting_await
        res = client.score_group("t0", designs)
        assert seen["n"] == 3  # 6 designs / ceil(6/3)=2 per shard -> 3 jobs
        assert res == [None] * 6

    def test_cache_reuses_scalar(self, tmp_path):
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02, cache=True)
        designs = _mk_designs(2, seed=1)
        r1 = self._run_with_worker(client, "t0", designs)
        # second call for the SAME designs must not hit the queue at all
        client._await_many = Mock(side_effect=AssertionError("cache miss: queue was hit"))
        r2 = client.score_group("t0", designs)
        assert r2 == r1

    def test_rewards_for_group_floors_failures(self, tmp_path):
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02)
        designs = _mk_designs(2)
        client._await_many = lambda job_ids: {jid: None for jid in job_ids}
        rewards, diags = client.rewards_for_group("t0", designs)
        assert rewards == [0.0, 0.0]
        assert diags == [None, None]

    def test_timeout_returns_none(self, tmp_path):
        """No worker + short timeout -> all designs floor to None, no crash."""
        client = ShapeRewardClient(tmp_path / "q", timeout_s=0.15, poll_s=0.02)
        res = client.score_group("t0", _mk_designs(2))
        assert res == [None, None]


# --------------------------------- submit_group / collect_group split (lever A pipelining)
#
# score_group is now collect_group(submit_group(...)); the split lets the trainer enqueue
# several groups' repack jobs up front and collect them later, so the a10g pool scores
# earlier targets while the b200 rolls out later ones. These tests pin that the split is
# byte-identical to the blocking path: same result mapping, same cache behaviour, same
# failure→None flooring — and that overlapping submits before any collect is safe.
class TestSubmitCollectSplit:
    def test_submit_then_collect_matches_score_group(self, tmp_path):
        """collect_group(submit_group(...)) end-to-end == the blocking score_group result."""
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02)
        designs = _mk_designs(5, seed=7)
        stop = threading.Event()
        t = threading.Thread(target=_fake_worker, args=(client.queue, stop), daemon=True)
        t.start()
        try:
            handle = client.submit_group("t0", designs)
            res = client.collect_group(handle)
        finally:
            stop.set()
            t.join(timeout=2.0)
        assert len(res) == 5 and all(r is not None and "term" in r for r in res)

    def test_two_submits_overlap_then_collect(self, tmp_path):
        """Enqueue two groups BEFORE collecting either — the pipelining scenario.

        Each group's results must map back to its own designs (no cross-talk), matching
        what sequential score_group calls would return.
        """
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02)
        g0, g1 = _mk_designs(3, seed=1), _mk_designs(4, seed=2)
        stop = threading.Event()
        t = threading.Thread(target=_fake_worker, args=(client.queue, stop), daemon=True)
        t.start()
        try:
            # submit both up front (b200 would be rolling out g1 while a10g scores g0) ...
            h0 = client.submit_group("t0", g0)
            h1 = client.submit_group("t1", g1)
            # ... then collect in order.
            r0 = client.collect_group(h0)
            r1 = client.collect_group(h1)
        finally:
            stop.set()
            t.join(timeout=2.0)
        assert len(r0) == 3 and len(r1) == 4
        # Reference: the deterministic stub scores each design by its coord means, so the
        # overlapped results must equal per-design direct scoring of the same coords.
        for grp, res in ((g0, r0), (g1, r1)):
            for d, r in zip(grp, res):
                exp = float(np.clip(np.tanh(d["ag_bb"].mean() + d["bd_bb"].mean()), 0, 1))
                assert r is not None and abs(r["term"] - exp) < 1e-6

    def test_submit_prefills_cache_hits_without_shards(self, tmp_path):
        """A fully-cached group needs no shards; collect returns the cached scalars, no queue."""
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02, cache=True)
        designs = _mk_designs(2, seed=3)
        stop = threading.Event()
        t = threading.Thread(target=_fake_worker, args=(client.queue, stop), daemon=True)
        t.start()
        try:
            first = client.score_group("t0", designs)  # warms the cache
        finally:
            stop.set()
            t.join(timeout=2.0)
        handle = client.submit_group("t0", designs)
        assert handle["shards"] == []  # every design was a cache hit -> nothing enqueued
        assert handle["results"] == first  # cache hits pre-filled at submit time
        client._await_many = Mock(side_effect=AssertionError("cache hit must not await the queue"))
        assert client.collect_group(handle) == first

    def test_collect_floors_failed_shards_to_none(self, tmp_path):
        """A failed/timed-out shard maps its designs to None (matching score_group flooring)."""
        client = ShapeRewardClient(tmp_path / "q", timeout_s=10.0, poll_s=0.02)
        designs = _mk_designs(3)
        handle = client.submit_group("t0", designs)
        assert handle["shards"]  # jobs were enqueued (no worker will drain them)
        client._await_many = lambda job_ids: {jid: None for jid in job_ids}
        assert client.collect_group(handle) == [None, None, None]


# ------------------------------------------------------------- trainer wiring
def _shape_cfg(**overrides):
    cfg = dict(w_shape=1.0)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _fake_model(G: int, L: int):
    """A stand-in model exposing decode_endpoint_aa -> (G, L) AA ids."""
    m = Mock()
    m.decode_endpoint_aa = Mock(return_value=torch.zeros(G, L, dtype=torch.long))
    return m


class TestShapeTermsForGroup:
    def test_weights_and_aggregates(self):
        G, L = 3, 6
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _shape_cfg(w_shape=2.0)
        trainer.model = _fake_model(G, L)
        gen_bb = np.zeros((G, L, 3, 3), dtype=np.float32)
        comp = {
            "mask": torch.ones(1, L),
            "binder_positions": torch.tensor([0, 0, 0, 1, 1, 1]),
        }
        rewards = [0.5, 0.2, 0.0]
        diags = [
            {"term": 0.5, "sc": 0.5, "n_patch_a": 10, "n_patch_b": 8},
            {"term": 0.2, "sc": 0.2, "n_patch_a": 12, "n_patch_b": 9},
            None,  # failed design
        ]
        trainer._shape_client = Mock()
        trainer._shape_client.rewards_for_group = Mock(return_value=(rewards, diags))

        weighted, metrics = trainer._shape_terms_for_group(
            target_id="t0", trajectory={}, comp=comp, seqs=["CCC", "CCC", "CCC"], gen_bb=gen_bb
        )

        assert weighted == pytest.approx([1.0, 0.4, 0.0])  # 2.0 * term
        assert metrics["reward/shape_term_mean"] == pytest.approx((1.0 + 0.4 + 0.0) / 3)
        assert metrics["shape/sc_mean"] == pytest.approx((0.5 + 0.2) / 2)  # None excluded
        assert metrics["shape/n_patch_a_mean"] == pytest.approx((10 + 12) / 2)
        assert metrics["shape/n_patch_b_mean"] == pytest.approx((8 + 9) / 2)
        assert metrics["shape/scored_frac"] == pytest.approx(2 / 3)

    def test_splits_antigen_and_binder(self):
        """The design dicts carry binder coords for binder positions and antigen for the rest."""
        G, L = 1, 5
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _shape_cfg()
        trainer.model = _fake_model(G, L)
        gen_bb = np.arange(G * L * 3 * 3, dtype=np.float32).reshape(G, L, 3, 3)
        comp = {"mask": torch.ones(1, L), "binder_positions": torch.tensor([0, 0, 0, 1, 1])}
        captured = {}

        def _capture(target_id, designs):
            captured["designs"] = designs
            return [0.0], [None]

        trainer._shape_client = Mock()
        trainer._shape_client.rewards_for_group = _capture
        trainer._shape_terms_for_group(target_id="t0", trajectory={}, comp=comp, seqs=["KK"], gen_bb=gen_bb)
        d = captured["designs"][0]
        assert d["ag_bb"].shape == (3, 3, 3)  # 3 antigen residues
        assert d["bd_bb"].shape == (2, 3, 3)  # 2 binder residues
        assert d["bd_seq"] == "KK"
        assert len(d["ag_seq"]) == 3

    def test_uses_shared_gen_bb_without_decoding(self):
        G, L = 2, 5
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _shape_cfg()
        trainer.model = _fake_model(G, L)
        trainer._decode_backbone_coords = Mock(side_effect=AssertionError("must not decode"))
        comp = {"mask": torch.ones(1, L), "binder_positions": torch.tensor([0, 0, 0, 1, 1])}
        trainer._shape_client = Mock()
        trainer._shape_client.rewards_for_group = Mock(return_value=([0.0, 0.0], [None, None]))
        gen_bb = np.zeros((G, L, 3, 3), dtype=np.float32)
        weighted, _ = trainer._shape_terms_for_group(
            target_id="t0", trajectory={}, comp=comp, seqs=["CC", "CC"], gen_bb=gen_bb
        )
        assert len(weighted) == 2
        trainer._decode_backbone_coords.assert_not_called()


class TestComputeRewardsShapeGate:
    def _cfg(self, **overrides):
        cfg = dict(
            w_iptm=0.0,
            w_ptm=0.0,
            w_abag_iptm=0.0,
            w_plddt=0.0,
            w_gpde=0.0,
            w_pae_global=0.0,
            w_pae_interface=0.0,
            w_sctm_binder=0.0,
            w_sctm_complex=0.0,
            log_struct_diagnostic=False,
            log_dist_diagnostic=False,
            per_token_clash=False,
            per_token_chainbreak=False,
            w_seq_diversity=0.0,
            w_struct_diversity=0.0,
            w_aa_dist=0.0,
            w_3di_dist=0.0,
            w_clash_contact=0.0,
            w_chainbreak=0.0,
            w_rog=0.0,
            w_shape=0.0,
            w_sc_clash=0.0,
            w_aar=0.0,
        )
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def test_w_shape_zero_is_inert(self):
        """w_shape=0 ⇒ shape helper never called, no decode, no shape metrics, reward 0."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = self._cfg(w_shape=0.0)
        trainer.reward_client = Mock()
        trainer.reward_client.score_group.side_effect = AssertionError("score_group must not be called")
        trainer._decode_backbone_coords = Mock(side_effect=AssertionError("must not decode"))
        trainer._shape_terms_for_group = Mock(side_effect=AssertionError("shape helper must not be called"))

        rewards, metrics, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["ACDE", "FGHI", "KLMN"], tri_seqs=None, trajectory={}, comp={}
        )

        trainer._shape_terms_for_group.assert_not_called()
        trainer._decode_backbone_coords.assert_not_called()
        assert rewards.tolist() == pytest.approx([0.0, 0.0, 0.0])
        assert not any(k.startswith("shape/") for k in metrics)
        assert "reward/shape_term_mean" not in metrics

    def test_shape_summed_and_protenix_free(self):
        """Shape-only reward ⇒ score_group untouched, reward = shape term, metrics flow."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = self._cfg(w_shape=1.0)
        trainer.reward_client = Mock()
        trainer.reward_client.score_group.side_effect = AssertionError("score_group must not be called")
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 4, 3, 3))
        shape_terms = [0.2, 0.6, 0.9]
        trainer._shape_terms_for_group = Mock(
            return_value=(shape_terms, {"reward/shape_term_mean": 0.5666, "shape/sc_mean": 0.4})
        )

        rewards, metrics, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["ACDE", "FGHI", "KLMN"], tri_seqs=None, trajectory={}, comp={}
        )

        trainer.reward_client.score_group.assert_not_called()
        trainer._decode_backbone_coords.assert_called_once()  # decoded once, shared
        _, kwargs = trainer._shape_terms_for_group.call_args
        assert kwargs["gen_bb"] is not None
        assert rewards.tolist() == pytest.approx(shape_terms)
        assert metrics["shape/sc_mean"] == pytest.approx(0.4)
        assert metrics["reward/confidence_term_mean"] == 0.0
