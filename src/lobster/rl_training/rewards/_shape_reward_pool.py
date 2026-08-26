"""Client + shared protocol for the persistent LigandMPNN-repack shape-complementarity pool.

The 3DZD interface shape-complementarity (SC) reward (:mod:`._shape_reward`) needs
**full-atom** side chains to carry signal (backbone-only SC is chance, AUROC 0.533;
LigandMPNN-repacked full-atom is 0.650, memory ``zernike-sc-discriminates-pass``). At
GRPO reward time LeFlur decodes only the N/CA/C backbone, so each design must be
side-chain-repacked before it can be scored. Repacking (LigandMPNN, openfold on GPU)
plus the SASA+Zernike SC (scipy on CPU) is served by a **persistent worker pool** over
a shared-filesystem queue, exactly like the Protenix confidence pool
(:mod:`._protenix_reward`) — this module is the *policy-side* (lobster venv) client; the
worker is ``scripts/ligandmpnn_repack_server.py`` (blessed venv, openfold isolated).

Unlike the Protenix pool (whose reward is a pure function of *sequence*, so it caches
and transports tiny JSON), SC is a function of the generated *backbone coordinates*,
which differ every rollout. So the job carries the per-design N/CA/C clouds in an
``.npz`` sidecar (heavy arrays, not JSON), and the result cache keys on a content hash
of the coordinates + sequence (a true duplicate design reuses its scalar; distinct
rollouts always recompute).

Queue protocol (JSON control files via atomic ``os.replace``; coords via ``.npz``)::

    $QUEUE/data/<job_id>.npz       # client -> pool: per-design N/CA/C clouds (written first)
    $QUEUE/new/<job_id>.json       # client -> pool: which designs, sequences, npz path
    $QUEUE/claimed/<job_id>.json   # a worker atomically renames new/ -> claimed/ to own it
    $QUEUE/done/<job_id>.json      # worker -> client: per-design SC scalars
    $QUEUE/failed/<job_id>.json    # worker -> client: unrecoverable error

Job schema (``new/``)::

    {"job_id", "target_id", "npz": "data/<job_id>.npz",
     "designs": [{"design_id", "ag_seq", "bd_seq"}, ...]}
    # npz keys: "<design_id>__ag" (Na,3,3), "<design_id>__bd" (Nb,3,3), [N,CA,C] order

Result schema (``done/``)::

    {"job_id", "results": {design_id: {"term", "sc", "n_patch_a", "n_patch_b"} | null}}

``term`` ∈ [0, 1] is the reward (clipped Pearson of the two interface 3DZD descriptors);
a missing/failed design is ``None`` → floor reward 0.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid

import numpy as np

from ._protenix_reward import atomic_write_json, ensure_queue

logger = logging.getLogger(__name__)


def reward_from_shape(res: dict | None) -> float:
    """Scalar SC reward from a worker result dict: ``term`` ∈ [0,1], or 0 if missing/failed."""
    if res is None:
        return 0.0
    term = res.get("term")
    return 0.0 if term is None else float(term)


def _design_key(
    target_id: str,
    ag_bb: np.ndarray,
    ag_seq: str,
    bd_bb: np.ndarray,
    bd_seq: str,
    want: tuple[str, ...] = ("sc",),
    return_seq: bool = False,
) -> str:
    """Content-address a design by (target, antigen coords+seq, binder coords+seq, metric set).

    SC depends on the exact backbone geometry, so the key hashes the float32 coordinate
    bytes (not just the sequence). Identical designs (exact within-group duplicates)
    collide and reuse the cached scalar; distinct rollouts always miss. ``want`` (the set
    of requested metrics — ``sc`` / ``clash`` / ``aar``) is folded in so an SC-only cached
    result is never reused for a job that also asked for clash/aar (different result shape).
    """
    h = hashlib.sha1()
    h.update(target_id.encode())
    h.update(b"\x00")
    h.update(np.ascontiguousarray(ag_bb, dtype=np.float32).tobytes())
    h.update(ag_seq.encode())
    h.update(b"\x00")
    h.update(np.ascontiguousarray(bd_bb, dtype=np.float32).tobytes())
    h.update(bd_seq.encode())
    h.update(b"\x00")
    h.update(",".join(want).encode())
    if return_seq:
        # SFT jobs carry the designed sequence in the payload -> distinct cache entry so a
        # plain-AAR cached result (no seq_design) is never reused for an SFT request.
        h.update(b"\x00seq")
    return h.hexdigest()


class ShapeRewardClient:
    """Blocking client for the persistent LigandMPNN-repack shape-complementarity pool.

    Submits a GRPO group of designs (each = antigen N/CA/C + sequence, binder N/CA/C +
    sequence) as queue jobs and blocks until the pool returns per-design SC scalars.
    Results are cached by a coordinate+sequence content hash. Timeouts / worker failures
    yield ``None`` for the affected designs (→ floor reward 0), so a stuck worker
    degrades the reward rather than crashing training.

    Parameters
    ----------
    queue_dir : str | os.PathLike
        Shared-filesystem queue root (created if absent; adds a ``data/`` subdir for the
        coordinate sidecars).
    timeout_s : float, optional
        Max seconds to wait for a group's results. Defaults to 1800.
    poll_s : float, optional
        Filesystem poll interval while waiting. Defaults to 2.0.
    cache : bool, optional
        Enable the content-hash result cache. Defaults to ``True`` (correct: exact
        duplicate designs are rare across rollouts, but free to reuse when they occur).
    n_shards : int, optional
        Split each group's to-score designs into this many independent sub-jobs so idle
        workers repack them in parallel. ``1`` (default) is one job per group; set
        ``= M_SHAPE`` (the worker pool size) to fan a whole group across the pool.
    """

    def __init__(
        self,
        queue_dir: str | os.PathLike,
        timeout_s: float = 1800.0,
        poll_s: float = 2.0,
        cache: bool = True,
        n_shards: int = 1,
    ) -> None:
        self.queue = ensure_queue(queue_dir)
        (self.queue / "data").mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.n_shards = max(1, int(n_shards))
        self._cache: dict[str, dict] | None = {} if cache else None

    def _write_npz(self, job_id: str, arrays: dict[str, np.ndarray]) -> str:
        """Atomically write the per-design coordinate sidecar; return its queue-relative path."""
        data_dir = self.queue / "data"
        tmp = data_dir / f".{job_id}.npz.tmp.{uuid.uuid4().hex}"
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, data_dir / f"{job_id}.npz")
        return f"data/{job_id}.npz"

    def score_group(
        self,
        target_id: str,
        designs: list[dict],
        want: tuple[str, ...] = ("sc",),
        return_seq: bool = False,
    ) -> list[dict | None]:
        """Repack + score a group of designs against one target; block for results.

        Thin blocking convenience over the non-blocking :meth:`submit_group` /
        :meth:`collect_group` split — ``collect_group(submit_group(...))`` — kept for callers
        that do not need to overlap the round-trip with other GPU work. Byte-identical to the
        pre-split behaviour.

        Parameters
        ----------
        target_id : str
            Target identifier (recorded in the job; also part of the cache key).
        designs : list[dict]
            One dict per design with keys ``ag_bb`` ``(Na, 3, 3)`` antigen N/CA/C,
            ``ag_seq`` antigen 1-letter AA string (len ``Na``), ``bd_bb`` ``(Nb, 3, 3)``
            binder N/CA/C, ``bd_seq`` binder AA string (len ``Nb``).
        want : tuple[str, ...], optional
            Which metrics the worker should compute for each design: any of ``"sc"``
            (3DZD shape-complementarity), ``"clash"`` (all-atom side-chain clash), ``"aar"``
            (ProteinMPNN amino-acid recovery). Defaults to ``("sc",)`` — the historical
            SC-only path. When ``want == ("sc",)`` the job JSON is written **without** a
            ``want`` key, so SC-only jobs stay byte-identical to the pre-generalization
            protocol; the worker then returns the flat ``{"term","sc",...}`` dict. For any
            other set the result per design is nested ``{"sc":..,"clash":..,"aar":..}`` for
            exactly the requested metrics (see ``scripts/ligandmpnn_repack_server.py``).

        Returns
        -------
        list[dict | None]
            One result per input design (flat ``{"term","sc","n_patch_a","n_patch_b"}`` for
            SC-only, else nested by requested metric), or ``None`` where scoring was a
            cache-miss and then failed / timed out.
        """
        return self.collect_group(self.submit_group(target_id, designs, want=want, return_seq=return_seq))

    def submit_group(
        self,
        target_id: str,
        designs: list[dict],
        want: tuple[str, ...] = ("sc",),
        return_seq: bool = False,
    ) -> dict:
        """Cache-check + shard + enqueue a group's designs WITHOUT blocking for results.

        Returns a *handle* dict carrying the state :meth:`collect_group` needs to await and
        map the results. Cache hits are pre-filled into ``results``; only cache-miss designs
        become queued shards. Splitting submit from :meth:`collect_group` lets the caller
        enqueue several groups' jobs (they run concurrently across the worker pool) before
        blocking on any of them — e.g. overlapping the a10g repack round-trip with the next
        target's GPU rollout.

        Parameters
        ----------
        target_id, designs, want, return_seq
            As in :meth:`score_group`.

        Returns
        -------
        dict
            Handle with keys ``target_id``, ``want`` (canonicalized), ``return_seq``,
            ``results`` (list pre-filled with cache hits, ``None`` elsewhere), ``keys``
            (per-design cache keys), and ``shards`` (list of ``(job_id, job_designs,
            subset)`` for the enqueued cache-miss jobs).
        """
        want = tuple(sorted(set(want)))  # canonical order for the cache key + worker checks
        results: list[dict | None] = [None] * len(designs)
        keys: list[str | None] = [None] * len(designs)
        to_score: list[int] = []
        for i, d in enumerate(designs):
            key = None
            if self._cache is not None:
                key = _design_key(target_id, d["ag_bb"], d["ag_seq"], d["bd_bb"], d["bd_seq"], want, return_seq)
                keys[i] = key
                if key in self._cache:
                    results[i] = self._cache[key]
                    continue
            to_score.append(i)

        shards: list[tuple[str, list[dict], list[int]]] = []
        if to_score:
            n = len(to_score)
            n_shards = max(1, min(self.n_shards, n))
            chunk = -(-n // n_shards)  # ceil division
            for start in range(0, n, chunk):
                subset = to_score[start : start + chunk]
                job_id = uuid.uuid4().hex
                job_designs: list[dict] = []
                arrays: dict[str, np.ndarray] = {}
                for i in subset:
                    d = designs[i]
                    did = f"{job_id}_{i}"
                    job_designs.append({"design_id": did, "ag_seq": d["ag_seq"], "bd_seq": d["bd_seq"]})
                    arrays[f"{did}__ag"] = np.ascontiguousarray(d["ag_bb"], dtype=np.float32)
                    arrays[f"{did}__bd"] = np.ascontiguousarray(d["bd_bb"], dtype=np.float32)
                # NPZ first (so the coords are on disk before the job is claimable), then json.
                npz_rel = self._write_npz(job_id, arrays)
                job = {"job_id": job_id, "target_id": target_id, "npz": npz_rel, "designs": job_designs}
                if want != ("sc",):
                    # Only emit `want` for multi-metric jobs; SC-only jobs stay byte-identical.
                    job["want"] = list(want)
                if return_seq:
                    # CHORD SFT: ask the AAR scorer to also emit the designed binder sequence.
                    job["aar_return_seq"] = True
                atomic_write_json(self.queue / "new" / f"{job_id}.json", job)
                shards.append((job_id, job_designs, subset))

        return {
            "target_id": target_id,
            "want": want,
            "return_seq": return_seq,
            "results": results,
            "keys": keys,
            "shards": shards,
        }

    def collect_group(self, handle: dict) -> list[dict | None]:
        """Block for the shards enqueued by :meth:`submit_group` and return per-design results.

        Awaits every shard job under one shared deadline (:meth:`_await_many`), maps each
        worker result back onto its design slot, and populates the content-hash cache. Cache
        hits recorded at submit time are already in ``handle["results"]`` and pass through
        untouched.

        Parameters
        ----------
        handle : dict
            The handle returned by :meth:`submit_group`.

        Returns
        -------
        list[dict | None]
            One result per input design, ``None`` where scoring was a cache-miss and then
            failed / timed out.
        """
        results: list[dict | None] = handle["results"]
        keys: list[str | None] = handle["keys"]
        shards: list[tuple[str, list[dict], list[int]]] = handle["shards"]
        if shards:
            all_res = self._await_many([s[0] for s in shards])
            for job_id, job_designs, subset in shards:
                res = all_res.get(job_id)
                for i, jd in zip(subset, job_designs):
                    out = None if res is None else res.get(jd["design_id"])
                    results[i] = out
                    if self._cache is not None and out is not None and keys[i] is not None:
                        self._cache[keys[i]] = out
        return results

    def rewards_for_group(self, target_id: str, designs: list[dict]) -> tuple[list[float], list[dict | None]]:
        """Convenience: :meth:`score_group` mapped through :func:`reward_from_shape`.

        Returns ``(rewards, diagnostics)`` — the scalar SC rewards (floor 0 on
        missing/failed) and the raw per-design result dicts (for logging patch sizes / SC).
        """
        res = self.score_group(target_id, designs)
        return [reward_from_shape(r) for r in res], res

    def _await_many(self, job_ids: list[str]) -> dict[str, dict | None]:
        """Poll for several jobs' ``done``/``failed`` files until all resolve or timeout.

        Waits on all shard jobs of a group concurrently under one shared deadline, so the
        group's wall-clock is the *slowest* shard, not their sum. Each job resolves to its
        ``results`` dict (success), or ``None`` (failure/timeout) — a stuck shard floors
        only its own designs, never the whole group.
        """
        pending = set(job_ids)
        out: dict[str, dict | None] = {}
        deadline = time.time() + self.timeout_s
        while pending and time.time() < deadline:
            for jid in list(pending):
                done = self.queue / "done" / f"{jid}.json"
                failed = self.queue / "failed" / f"{jid}.json"
                if done.exists():
                    try:
                        out[jid] = json.loads(done.read_text()).get("results", {})
                        pending.discard(jid)
                    except (json.JSONDecodeError, OSError):
                        pass  # reader raced the writer's replace; retry next poll
                elif failed.exists():
                    logger.warning("Shape-reward job %s reported failure", jid)
                    out[jid] = None
                    pending.discard(jid)
            if pending:
                time.sleep(self.poll_s)
        for jid in pending:
            logger.warning("Shape-reward job %s timed out after %.0fs", jid, self.timeout_s)
            out[jid] = None
        return out
