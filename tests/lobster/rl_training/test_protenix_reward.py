"""Round-trip + reward-rule tests for the Protenix reward queue.

The real reward oracle (Protenix-v2) needs a GPU + its own venv, so these tests
exercise the *queue protocol* end-to-end against a lightweight mock worker that
uses the real server's atomic-claim logic (``protenix_reward_server.claim_one``)
but returns deterministic fake confidences instead of running Protenix. They
assert:

1. ``ProtenixRewardClient.score_group`` submits a job, the worker claims + answers
   it, and the client reads back the per-design confidences in input order.
2. The ``(target_id, seq)`` cache avoids re-submitting already-scored sequences.
3. A timeout (no worker) yields ``None`` -> floor reward, not a crash.
4. The reward rule matches the eval: ``ip = abag_iptm if not-null else iptm``;
   PASS iff ``ptm > 0.80`` and ``ip > 0.70``.
5. The confidence reward is a flat weighted-linear combo; the default weights
   recover the shipped ``abag_iptm + 0.5*ptm`` behaviour.
6. ``return_coords=True`` threads the request into the job and the worker's
   predicted CA coordinates ride back inside each confidence dict.
"""

from __future__ import annotations

import csv
import importlib.util
import threading
import time
from pathlib import Path

import pytest

from lobster.rl_training.rewards._protenix_reward import (
    ProtenixRewardClient,
    continuous_ip,
    passes,
    reward_from_confidence,
)

# Import the standalone server script by path (scripts/ is not a package).
_SERVER_PATH = Path(__file__).resolve().parents[3] / "scripts" / "protenix_reward_server.py"
_spec = importlib.util.spec_from_file_location("protenix_reward_server", _SERVER_PATH)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


def _write_targets_csv(path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target_id", "antigen_seq", "antigen_a3m"])
        w.writerow(["01_PD1", "MKAIVLLL", "/dev/null"])


def _fake_conf_for(seq: str) -> dict:
    """Deterministic fake confidence: longer binders score 'better' (monotone in len)."""
    ip = min(0.95, 0.4 + 0.02 * len(seq))
    return {"ptm": 0.85, "iptm": ip, "plddt": 90.0, "abag_iptm": round(ip, 4)}


class _MockWorker(threading.Thread):
    """Claims jobs via the real server logic; answers with fake confidences."""

    def __init__(self, queue: Path) -> None:
        super().__init__(daemon=True)
        self.queue = queue
        self._stop = threading.Event()
        self.jobs_served = 0

    def run(self) -> None:
        while not self._stop.is_set():
            claimed = server.claim_one(self.queue, worker_id="mock")
            if claimed is None:
                time.sleep(0.02)
                continue
            job_id, job = claimed
            self.jobs_served += 1
            results = {d["design_id"]: _fake_conf_for(d["binder_seq"]) for d in job["designs"]}
            server._atomic_write_json(self.queue / "done" / f"{job_id}.json", {"job_id": job_id, "results": results})

    def stop(self) -> None:
        self._stop.set()


@pytest.fixture()
def queue_and_targets(tmp_path: Path):
    from lobster.rl_training.rewards._protenix_reward import ensure_queue

    queue = ensure_queue(tmp_path / "queue")
    targets = tmp_path / "targets.csv"
    _write_targets_csv(targets)
    return queue, targets


def test_score_group_roundtrip(queue_and_targets) -> None:
    queue, targets = queue_and_targets
    worker = _MockWorker(queue)
    worker.start()
    try:
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=False)
        seqs = ["AAAA", "AAAAAAAA", "AAAAAAAAAAAA"]
        confs = client.score_group("01_PD1", seqs)
    finally:
        worker.stop()

    assert len(confs) == len(seqs)
    assert [c["abag_iptm"] for c in confs] == [_fake_conf_for(s)["abag_iptm"] for s in seqs]


def test_rewards_for_group_and_ordering(queue_and_targets) -> None:
    queue, targets = queue_and_targets
    worker = _MockWorker(queue)
    worker.start()
    try:
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=False)
        seqs = ["AAAA", "AAAAAAAAAAAA"]
        rewards, confs = client.rewards_for_group("01_PD1", seqs)
    finally:
        worker.stop()

    # Reward == clipped continuous ip; monotone in length for the fake oracle.
    assert rewards == [reward_from_confidence(c) for c in confs]
    assert rewards[1] > rewards[0]


def test_cache_avoids_resubmission(queue_and_targets) -> None:
    queue, targets = queue_and_targets
    worker = _MockWorker(queue)
    worker.start()
    try:
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=True)
        seqs = ["AAAA", "CCCC"]
        first = client.score_group("01_PD1", seqs)
        served_after_first = worker.jobs_served
        # All cached now -> second call must not submit a new job.
        second = client.score_group("01_PD1", seqs)
    finally:
        worker.stop()

    assert [c["abag_iptm"] for c in first] == [c["abag_iptm"] for c in second]
    assert worker.jobs_served == served_after_first  # no new job for fully-cached group


def test_score_group_sharding_roundtrip(queue_and_targets) -> None:
    """n_shards splits a group into parallel sub-jobs; results reassemble in order."""
    queue, targets = queue_and_targets
    workers = [_MockWorker(queue) for _ in range(3)]
    for w in workers:
        w.start()
    try:
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=False, n_shards=4)
        seqs = ["A" * (n + 1) for n in range(8)]  # 8 distinct-length seqs
        confs = client.score_group("01_PD1", seqs)
    finally:
        for w in workers:
            w.stop()

    # 8 designs / 4 shards -> ceil(8/4)=2 per shard -> 4 sub-jobs across the pool.
    assert sum(w.jobs_served for w in workers) == 4
    assert len(confs) == len(seqs)
    assert [c["abag_iptm"] for c in confs] == [_fake_conf_for(s)["abag_iptm"] for s in seqs]


def test_sharding_cache_bypasses_resubmission(queue_and_targets) -> None:
    """A fully-cached group submits no shard jobs on the second call."""
    queue, targets = queue_and_targets
    worker = _MockWorker(queue)
    worker.start()
    try:
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=True, n_shards=4)
        seqs = ["A" * (n + 1) for n in range(8)]
        first = client.score_group("01_PD1", seqs)
        served_after_first = worker.jobs_served
        second = client.score_group("01_PD1", seqs)
    finally:
        worker.stop()

    assert [c["abag_iptm"] for c in first] == [c["abag_iptm"] for c in second]
    assert worker.jobs_served == served_after_first  # no new shard jobs for a cached group


def test_sharding_failure_isolation(queue_and_targets) -> None:
    """A failed shard floors only its own designs; sibling shards are unaffected."""
    queue, targets = queue_and_targets

    class _FailingWorker(_MockWorker):
        """Fails any shard containing the sentinel seq; answers the rest normally."""

        def run(self) -> None:
            while not self._stop.is_set():
                claimed = server.claim_one(self.queue, worker_id="mock")
                if claimed is None:
                    time.sleep(0.02)
                    continue
                job_id, job = claimed
                self.jobs_served += 1
                if any(d["binder_seq"] == "FAIL" for d in job["designs"]):
                    server._atomic_write_json(
                        self.queue / "failed" / f"{job_id}.json", {"job_id": job_id, "error": "boom"}
                    )
                else:
                    results = {d["design_id"]: _fake_conf_for(d["binder_seq"]) for d in job["designs"]}
                    server._atomic_write_json(
                        self.queue / "done" / f"{job_id}.json", {"job_id": job_id, "results": results}
                    )

    worker = _FailingWorker(queue)
    worker.start()
    try:
        # n_shards >= n_designs -> each design is its own shard, so FAIL is isolated.
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=False, n_shards=8)
        confs = client.score_group("01_PD1", ["AAAA", "FAIL", "CCCCCC"])
    finally:
        worker.stop()

    assert confs[0] is not None and confs[2] is not None
    assert confs[1] is None  # only the failed shard's design is floored


def test_timeout_returns_none_floor_reward(queue_and_targets) -> None:
    queue, targets = queue_and_targets  # no worker running
    client = ProtenixRewardClient(queue, targets, timeout_s=0.3, poll_s=0.05, cache=False)
    confs = client.score_group("01_PD1", ["AAAA", "CCCC"])
    assert confs == [None, None]
    assert reward_from_confidence(confs[0]) == 0.0


def test_unknown_target_raises(queue_and_targets) -> None:
    queue, targets = queue_and_targets
    client = ProtenixRewardClient(queue, targets, cache=False)
    with pytest.raises(KeyError, match="not in manifest"):
        client.score_group("NOPE", ["AAAA"])


@pytest.mark.parametrize(
    "conf,exp_ip,exp_pass",
    [
        ({"ptm": 0.85, "iptm": 0.60, "abag_iptm": 0.75}, 0.75, True),  # abag used, passes
        ({"ptm": 0.85, "iptm": 0.75, "abag_iptm": None}, 0.75, True),  # falls back to iptm
        ({"ptm": 0.79, "iptm": 0.90, "abag_iptm": 0.90}, 0.90, False),  # ptm below thr
        ({"ptm": 0.90, "iptm": 0.65, "abag_iptm": 0.65}, 0.65, False),  # ip below thr
        (None, None, False),  # missing score
    ],
)
def test_reward_rule(conf, exp_ip, exp_pass) -> None:
    assert continuous_ip(conf) == exp_ip
    assert passes(conf) is exp_pass


def test_confidence_flat_linear_combo() -> None:
    """Default weights recover ``abag_iptm + 0.5*ptm``; extra metrics add oriented terms."""
    from lobster.rl_training.rewards._protenix_reward import (
        DEFAULT_CONF_WEIGHTS,
        confidence_components,
    )

    conf = {"ptm": 0.82, "abag_iptm": 0.60, "iptm": 0.55, "plddt": 90.0, "gpde": 1.0}
    # Default: only abag_iptm (w=1) and ptm (w=0.5) are active.
    assert reward_from_confidence(conf, DEFAULT_CONF_WEIGHTS) == pytest.approx(0.60 + 0.5 * 0.82)
    comps = confidence_components(conf, DEFAULT_CONF_WEIGHTS)
    assert set(comps) == {"w_abag_iptm", "w_ptm"}

    # A custom weight set orients + clips the extra metrics: plddt/100, 1 - gpde/2.
    w = {"w_abag_iptm": 1.0, "w_plddt": 1.0, "w_gpde": 1.0}
    r = reward_from_confidence(conf, w)
    assert r == pytest.approx(0.60 + 0.90 + (1.0 - 1.0 / 2.0))
    # An absent field contributes 0 (no pae_* in conf here).
    assert reward_from_confidence(conf, {"w_pae_global": 1.0}) == 0.0


def test_ipsae_reward_term_oriented_as_is() -> None:
    """``w_ipsae`` orients ipSAE as-is (already higher-is-better in [0,1]) and floors when absent."""
    from lobster.rl_training.rewards._protenix_reward import confidence_components

    conf = {"ptm": 0.82, "abag_iptm": 0.60, "ipsae": 0.73}
    # ipSAE is taken as-is: no orientation flip, no ceiling divide.
    assert reward_from_confidence(conf, {"w_ipsae": 1.0}) == pytest.approx(0.73)
    assert reward_from_confidence(conf, {"w_ipsae": 2.0}) == pytest.approx(1.46)
    comps = confidence_components(conf, {"w_abag_iptm": 1.0, "w_ipsae": 0.5})
    assert comps == pytest.approx({"w_abag_iptm": 0.60, "w_ipsae": 0.5 * 0.73})
    # Clipped to [0,1] before weighting.
    assert reward_from_confidence({"ipsae": 1.5}, {"w_ipsae": 1.0}) == pytest.approx(1.0)
    # Absent ipSAE (worker ran without per-token PAE dump) contributes 0.
    assert reward_from_confidence({"ptm": 0.82}, {"w_ipsae": 1.0}) == 0.0


def test_worker_ipsae_scalar_matches_reference() -> None:
    """The worker's vendored ``_ipsae_scalar`` reproduces scripts/_grpo_ipsae_metrics.py exactly."""
    import importlib.util as _ilu

    import numpy as np

    ref_path = _SERVER_PATH.parent / "_grpo_ipsae_metrics.py"
    ref_spec = _ilu.spec_from_file_location("_grpo_ipsae_metrics", ref_path)
    ref = _ilu.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref)

    rng = np.random.default_rng(0)
    n_a, n_b = 12, 9  # antigen (chain 0) then binder (chain 1), first-appearance order
    n = n_a + n_b
    pae = rng.uniform(0.0, 20.0, size=(n, n)).astype(np.float32)
    asym = np.array([0] * n_a + [1] * n_b)

    got = server._ipsae_scalar(pae, asym, server.IPSAE_PAE_CUTOFF)
    want = ref.ipsae_from_full_data(
        {"token_pair_pae": pae.tolist(), "token_asym_id": asym.tolist()},
        server.IPSAE_PAE_CUTOFF,
    )
    assert want is not None
    assert got == pytest.approx(want[0])
    # Worker default cutoff equals the offline metric's default.
    assert server.IPSAE_PAE_CUTOFF == ref.PAE_CUTOFF
    # Malformed / single-chain inputs return None (design dropped, term floors to 0).
    assert server._ipsae_scalar(pae, np.zeros(n), server.IPSAE_PAE_CUTOFF) is None
    assert server._ipsae_scalar(pae[:-1], asym, server.IPSAE_PAE_CUTOFF) is None


def test_worker_pertoken_conf_from_full_data(tmp_path) -> None:
    """Per-token confidence vectors have binder length, correct per-residue values + orientation."""
    import json

    import numpy as np

    n_a, n_b = 3, 4  # antigen (chain 0) then binder (chain 1), first-appearance order
    n = n_a + n_b
    rng = np.random.default_rng(1)
    pae = rng.uniform(0.0, 12.0, size=(n, n)).astype(np.float64)
    asym = np.array([0] * n_a + [1] * n_b)
    # One atom per token so per-token pLDDT aggregation is the identity on the binder slice.
    atom_plddt = np.linspace(0.30, 0.95, n)
    atom_to_token_idx = np.arange(n)

    # Lay out dump/name/**/full_data as parse_one expects.
    dump = tmp_path / "dump"
    name = "design_0"
    pred = dump / name / "seed_1" / "predictions"
    pred.mkdir(parents=True)
    (pred / f"{name}_full_data_sample_0.json").write_text(
        json.dumps(
            {
                "token_pair_pae": pae.tolist(),
                "token_asym_id": asym.tolist(),
                "atom_plddt": atom_plddt.tolist(),
                "atom_to_token_idx": atom_to_token_idx.tolist(),
            }
        )
    )

    res = server._pertoken_conf_from_full_data(dump, name)
    assert res is not None
    assert set(res) == {"plddt_res", "pae_interface_res", "pae_global_res", "ipsae_res"}
    for v in res.values():
        assert len(v) == n_b  # one value per binder residue

    b_idx = np.arange(n_a, n)
    a_idx = np.arange(0, n_a)
    # pLDDT: one atom per token -> the binder-token atom_plddt values (higher = better, un-negated).
    assert res["plddt_res"] == pytest.approx([round(x, 4) for x in atom_plddt[b_idx].tolist()])
    # interface PAE: mean over antigen columns; global PAE: mean over all columns (lower = better).
    assert res["pae_interface_res"] == pytest.approx(
        [round(x, 4) for x in pae[np.ix_(b_idx, a_idx)].mean(axis=1).tolist()]
    )
    assert res["pae_global_res"] == pytest.approx([round(x, 4) for x in pae[b_idx, :].mean(axis=1).tolist()])
    # ipSAE per binder residue (B->A); its max is the scalar asymmetric B->A ipSAE.
    mask_a, mask_b = server._two_chain_masks(asym)
    assert max(res["ipsae_res"]) == pytest.approx(
        round(server._ipsae_asym(pae, mask_b, mask_a, server.IPSAE_PAE_CUTOFF), 4)
    )
    # Single-chain / missing full-data -> None (design contributes no per-token conf credit).
    solo = tmp_path / "solo"
    (solo / "d1" / "predictions").mkdir(parents=True)
    (solo / "d1" / "predictions" / "d1_full_data_sample_0.json").write_text(
        json.dumps({"token_pair_pae": pae.tolist(), "token_asym_id": [0] * n})
    )
    assert server._pertoken_conf_from_full_data(solo, "d1") is None
    assert server._pertoken_conf_from_full_data(tmp_path, "does_not_exist") is None


class _CoordWorker(_MockWorker):
    """Answers with fake confidences plus predicted CA coords when asked."""

    def run(self) -> None:
        while not self._stop.is_set():
            claimed = server.claim_one(self.queue, worker_id="mock")
            if claimed is None:
                time.sleep(0.02)
                continue
            job_id, job = claimed
            self.jobs_served += 1
            want = bool(job.get("return_coords", False))
            results = {}
            for d in job["designs"]:
                conf = _fake_conf_for(d["binder_seq"])
                if want:
                    n = len(d["binder_seq"])
                    conf["binder_xyz"] = [[float(i), 0.0, 0.0] for i in range(n)]
                    conf["antigen_xyz"] = [[0.0, float(i), 0.0] for i in range(4)]
                results[d["design_id"]] = conf
            server._atomic_write_json(self.queue / "done" / f"{job_id}.json", {"job_id": job_id, "results": results})


def test_return_coords_roundtrip(queue_and_targets) -> None:
    """``return_coords=True`` threads through and coords ride back in each conf dict."""
    queue, targets = queue_and_targets
    worker = _CoordWorker(queue)
    worker.start()
    try:
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=False)
        seqs = ["AAAA", "AAAAAAAA"]
        without = client.score_group("01_PD1", seqs, return_coords=False)
        with_coords = client.score_group("01_PD1", seqs, return_coords=True)
    finally:
        worker.stop()

    assert all("binder_xyz" not in c for c in without)
    for seq, conf in zip(seqs, with_coords):
        assert len(conf["binder_xyz"]) == len(seq)
        assert len(conf["antigen_xyz"]) == 4


class _PertokenConfWorker(_MockWorker):
    """Answers with fake confidences plus per-binder-residue vectors when asked."""

    def run(self) -> None:
        while not self._stop.is_set():
            claimed = server.claim_one(self.queue, worker_id="mock")
            if claimed is None:
                time.sleep(0.02)
                continue
            job_id, job = claimed
            self.jobs_served += 1
            want = bool(job.get("return_pertoken_conf", False))
            results = {}
            for d in job["designs"]:
                conf = _fake_conf_for(d["binder_seq"])
                if want:
                    n = len(d["binder_seq"])
                    conf["pertoken_conf"] = {
                        "plddt_res": [0.7] * n,
                        "pae_interface_res": [8.0] * n,
                        "pae_global_res": [9.0] * n,
                        "ipsae_res": [0.5] * n,
                    }
                results[d["design_id"]] = conf
            server._atomic_write_json(self.queue / "done" / f"{job_id}.json", {"job_id": job_id, "results": results})


def test_return_pertoken_conf_roundtrip(queue_and_targets) -> None:
    """``return_pertoken_conf=True`` threads through and per-residue vectors ride back per design."""
    queue, targets = queue_and_targets
    worker = _PertokenConfWorker(queue)
    worker.start()
    try:
        client = ProtenixRewardClient(queue, targets, timeout_s=10, poll_s=0.02, cache=False)
        seqs = ["AAAA", "AAAAAAAA"]
        without = client.score_group("01_PD1", seqs, return_pertoken_conf=False)
        with_pt = client.score_group("01_PD1", seqs, return_pertoken_conf=True)
    finally:
        worker.stop()

    assert all("pertoken_conf" not in c for c in without)
    for seq, conf in zip(seqs, with_pt):
        ptc = conf["pertoken_conf"]
        assert set(ptc) == {"plddt_res", "pae_interface_res", "pae_global_res", "ipsae_res"}
        assert all(len(v) == len(seq) for v in ptc.values())
