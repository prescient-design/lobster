"""Client + shared protocol for the persistent Protenix co-folding reward pool.

GRPO on the LeFlur binder policy uses Protenix-v2 co-folding ipTM as the reward
oracle. Protenix is slow (~80 s weight load + ~25 s/design on an A10G) and lives in
its own py3.11 venv, so it is served by a **persistent worker pool** over a
shared-filesystem queue rather than called inline. This module is the *policy-side*
(lobster venv) client; the worker is ``scripts/protenix_reward_server.py`` (runs in
the Protenix venv, standalone).

Queue protocol (all files are JSON, written via atomic ``os.replace``)::

    $QUEUE/new/<job_id>.json       # client -> pool: one GRPO group to score
    $QUEUE/claimed/<job_id>.json   # a worker atomically renames new/ -> claimed/ to own it
    $QUEUE/done/<job_id>.json      # worker -> client: per-design confidences
    $QUEUE/failed/<job_id>.json    # worker -> client: unrecoverable error

Job schema (``new/``)::

    {"job_id", "target_id", "antigen_seq", "antigen_a3m",
     "designs": [{"design_id", "binder_seq"}, ...]}

Result schema (``done/``)::

    {"job_id", "results": {design_id: {"ptm", "iptm", "plddt", "abag_iptm",
                                       "gpde", "pae_global", "pae_interface",
                                       # only when the job set return_coords=true:
                                       "antigen_xyz", "binder_xyz"}, ...}}

The confidence reward is a **flat, weighted linear combination** of these metrics —
each oriented onto a higher-is-better axis and clipped to ``[0,1]`` (see
:func:`reward_from_confidence` and ``README.md``).

The pass rule matches the eval (``scripts/_complexa_pertarget.py``):
``ip = abag_iptm if not-null else iptm``; a design PASSES if ``ptm > 0.80`` and
``ip > 0.70``. The pass gate is fixed and does **not** depend on the reward weights.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

QUEUE_SUBDIRS = ("new", "claimed", "done", "failed")

# Reward / pass thresholds — kept in sync with scripts/_complexa_pertarget.py.
PTM_PASS_THR = 0.80
IP_PASS_THR = 0.70

# Per-metric ceilings for the lower-is-better error metrics (Å-like). ``gpde`` is
# a global distance error in the ~0-2 Å range; PAE is a predicted aligned error
# whose Protenix bin ceiling is ~31.75 Å. Oriented as ``1 - value/ceiling`` so
# "small error" maps to "near 1".
_GPDE_CEIL = 2.0
_PAE_CEIL = 32.0

# Default confidence weights: recover the shipped M22 behaviour
# (reward = clip(abag_iptm) + 0.5*clip(ptm)); every other metric weight defaults 0.
DEFAULT_CONF_WEIGHTS: dict[str, float] = {"w_abag_iptm": 1.0, "w_ptm": 0.5}


def _clip01(x: float) -> float:
    """Clip ``x`` to ``[0, 1]``."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _oriented_metric(conf: dict, weight_key: str) -> float | None:
    """Oriented, [0,1]-clipped value for the metric that ``weight_key`` scales.

    Maps each raw Protenix field onto a higher-is-better axis (``*tm`` as-is,
    ``plddt/100``, ``1 - gpde/2``, ``1 - pae/32``) then clips to ``[0,1]``.
    Returns ``None`` when the underlying field is absent (→ 0 contribution).
    """
    if weight_key == "w_iptm":
        v = conf.get("iptm")
        return None if v is None else _clip01(float(v))
    if weight_key == "w_ptm":
        v = conf.get("ptm")
        return None if v is None else _clip01(float(v))
    if weight_key == "w_abag_iptm":
        v = conf.get("abag_iptm")
        return None if v is None else _clip01(float(v))
    if weight_key == "w_plddt":
        v = conf.get("plddt")
        return None if v is None else _clip01(float(v) / 100.0)
    if weight_key == "w_gpde":
        v = conf.get("gpde")
        return None if v is None else _clip01(1.0 - float(v) / _GPDE_CEIL)
    if weight_key == "w_pae_global":
        v = conf.get("pae_global")
        return None if v is None else _clip01(1.0 - float(v) / _PAE_CEIL)
    if weight_key == "w_pae_interface":
        v = conf.get("pae_interface")
        return None if v is None else _clip01(1.0 - float(v) / _PAE_CEIL)
    return None


def ensure_queue(queue_dir: str | os.PathLike) -> Path:
    """Create the queue directory tree (``new/claimed/done/failed``) if absent.

    Parameters
    ----------
    queue_dir : str | os.PathLike
        Root directory of the shared-filesystem queue.

    Returns
    -------
    Path
        The resolved queue root.
    """
    root = Path(queue_dir)
    for sub in QUEUE_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def atomic_write_json(path: str | os.PathLike, obj: dict) -> None:
    """Write ``obj`` as JSON to ``path`` atomically (tmp file + ``os.replace``).

    The temp file is created in the destination directory so the rename stays on
    one filesystem (a genuinely atomic operation), which lets a reader never
    observe a partially written job/result file.

    Parameters
    ----------
    path : str | os.PathLike
        Destination path.
    obj : dict
        JSON-serializable payload.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def continuous_ip(conf: dict | None) -> float | None:
    """The continuous interface-pTM used as reward: ``abag_iptm`` else ``iptm``.

    Parameters
    ----------
    conf : dict | None
        A per-design confidence dict, or ``None`` (missing / failed score).

    Returns
    -------
    float | None
        The interface pTM, or ``None`` if unavailable.
    """
    if conf is None:
        return None
    abag = conf.get("abag_iptm")
    if abag is not None:
        return float(abag)
    iptm = conf.get("iptm")
    return None if iptm is None else float(iptm)


def confidence_components(conf: dict | None, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Per-metric confidence-reward contributions ``w_m * clip(orient_m(conf[m]), 0, 1)``.

    Only metrics with a non-zero weight *and* a present underlying field appear in
    the returned dict; a missing field simply omits that metric (0 contribution).
    Keyed by weight name (``w_ptm``, ``w_abag_iptm``, …) for direct wandb logging.

    Parameters
    ----------
    conf : dict | None
        Per-design confidence dict, or ``None`` (→ empty dict).
    weights : dict[str, float] | None, optional
        Metric weights; defaults to :data:`DEFAULT_CONF_WEIGHTS`.

    Returns
    -------
    dict[str, float]
        ``{weight_key: contribution}`` for each active, present metric.
    """
    if conf is None:
        return {}
    w = DEFAULT_CONF_WEIGHTS if weights is None else weights
    out: dict[str, float] = {}
    for key, weight in w.items():
        if not weight:
            continue
        oriented = _oriented_metric(conf, key)
        if oriented is not None:
            out[key] = float(weight) * oriented
    return out


def reward_from_confidence(conf: dict | None, weights: dict[str, float] | None = None) -> float:
    """Flat, per-metric-clipped weighted linear combination of confidence metrics.

    ``reward = Σ_m w_m · clip(orient_m(conf[m]), 0, 1)`` where ``orient_m`` maps each
    raw Protenix field onto a higher-is-better axis (``*tm`` as-is, ``plddt/100``,
    ``1 - gpde/2``, ``1 - pae/32``). A missing field (or ``conf is None``)
    contributes 0, so a failed/timed-out design floors to 0.

    Parameters
    ----------
    conf : dict | None
        Per-design confidence dict, or ``None`` (missing/failed → 0 reward).
    weights : dict[str, float] | None, optional
        Metric weights keyed by ``w_<metric>`` (see :data:`DEFAULT_CONF_WEIGHTS`).
        Defaults to the shipped ``{w_abag_iptm: 1.0, w_ptm: 0.5}``.

    Returns
    -------
    float
        The summed weighted reward (``0`` when scores are missing).
    """
    if conf is None:
        return 0.0
    return float(sum(confidence_components(conf, weights).values()))


def passes(conf: dict | None, ptm_thr: float = PTM_PASS_THR, ip_thr: float = IP_PASS_THR) -> bool:
    """Whether a design passes the binder criterion ``ptm > ptm_thr`` and ``ip > ip_thr``.

    Parameters
    ----------
    conf : dict | None
        Per-design confidence dict, or ``None``.
    ptm_thr, ip_thr : float, optional
        Thresholds (defaults match the eval: 0.80 / 0.70).

    Returns
    -------
    bool
        ``True`` iff both thresholds are strictly exceeded.
    """
    if conf is None:
        return False
    ptm = conf.get("ptm")
    ip = continuous_ip(conf)
    return bool(ptm is not None and ip is not None and ptm > ptm_thr and ip > ip_thr)


class ProtenixRewardClient:
    """Blocking client for the persistent Protenix reward pool.

    Submits a GRPO group of binder sequences as one queue job and blocks until the
    pool returns per-design confidences. Results are LRU-cached by
    ``(target_id, binder_seq)`` so repeated / degenerate designs are scored once.
    Timeouts and worker failures yield ``None`` for the affected designs (the
    caller maps ``None`` to the floor reward via :func:`reward_from_confidence`),
    so a stuck worker degrades the reward rather than crashing training.

    Parameters
    ----------
    queue_dir : str | os.PathLike
        Shared-filesystem queue root (created if absent).
    targets_csv : str | os.PathLike
        Manifest with ``target_id, antigen_seq, antigen_a3m`` columns (e.g.
        ``complexa_score_targets.csv``); supplies the static per-target reward
        inputs.
    timeout_s : float, optional
        Max seconds to wait for a group's results. Defaults to 1800.
    poll_s : float, optional
        Filesystem poll interval while waiting. Defaults to 2.0.
    cache : bool, optional
        Enable the ``(target_id, seq)`` result cache. Defaults to ``True``.
    n_shards : int, optional
        Split each group's to-score designs into this many independent sub-jobs so
        idle workers score them in parallel (throughput lever B). ``1`` (default)
        preserves the legacy one-job-per-group behaviour. Set ``= N_WORKERS`` to
        keep the whole A10G pool busy on a single group. A shard's timeout/failure
        floors only its own designs (the rest of the group is unaffected).

    Raises
    ------
    KeyError
        If ``score_group`` is called with a ``target_id`` absent from the manifest.
    """

    def __init__(
        self,
        queue_dir: str | os.PathLike,
        targets_csv: str | os.PathLike,
        timeout_s: float = 1800.0,
        poll_s: float = 2.0,
        cache: bool = True,
        n_shards: int = 1,
    ) -> None:
        self.queue = ensure_queue(queue_dir)
        self.targets = self._load_targets(targets_csv)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.n_shards = max(1, int(n_shards))
        self._cache: dict[tuple[str, str], dict] | None = {} if cache else None

    @staticmethod
    def _load_targets(targets_csv: str | os.PathLike) -> dict[str, dict]:
        with open(targets_csv, newline="") as fh:
            return {r["target_id"]: r for r in csv.DictReader(fh)}

    def score_group(self, target_id: str, binder_seqs: list[str], return_coords: bool = False) -> list[dict | None]:
        """Score a group of binder sequences against one target; block for results.

        Parameters
        ----------
        target_id : str
            Target key into the manifest.
        binder_seqs : list[str]
            Binder amino-acid strings (one per design in the GRPO group).
        return_coords : bool, optional
            When ``True``, ask the worker to also return the Protenix-predicted CA
            coordinates (``antigen_xyz`` chain A, ``binder_xyz`` chain B) inside each
            confidence dict — needed by the structure self-consistency term. Off by
            default to keep result JSON lean when structure weights are 0. Note the
            result cache keys only on ``(target_id, seq)``: a cached hit returns
            whatever fields were fetched the first time, so a run that needs coords
            should keep this ``True`` for every call.

        Returns
        -------
        list[dict | None]
            One confidence dict per input sequence (``{"ptm","iptm","plddt",
            "abag_iptm","gpde","pae_global","pae_interface"}`` plus
            ``"antigen_xyz"``/``"binder_xyz"`` when ``return_coords``), or ``None``
            where scoring was cached-miss + failed / timed out.
        """
        if target_id not in self.targets:
            raise KeyError(f"target_id {target_id!r} not in manifest ({len(self.targets)} targets)")

        results: list[dict | None] = [None] * len(binder_seqs)
        to_score: list[tuple[int, str]] = []
        for i, seq in enumerate(binder_seqs):
            if self._cache is not None and (target_id, seq) in self._cache:
                results[i] = self._cache[(target_id, seq)]
            else:
                to_score.append((i, seq))

        if to_score:
            row = self.targets[target_id]
            # Shard the uncached designs across up to ``n_shards`` sub-jobs so idle
            # workers score them in parallel (throughput lever B). Contiguous chunks
            # of ``ceil(n / n_shards)`` keep each shard balanced; with n_shards=1 this
            # is exactly the legacy single-job path.
            n = len(to_score)
            n_shards = max(1, min(self.n_shards, n))
            chunk = -(-n // n_shards)  # ceil division
            shards: list[tuple[str, list[dict], list[tuple[int, str]]]] = []
            for start in range(0, n, chunk):
                subset = to_score[start : start + chunk]
                job_id = uuid.uuid4().hex
                designs = [{"design_id": f"{job_id}_{i}", "binder_seq": seq} for i, seq in subset]
                job = {
                    "job_id": job_id,
                    "target_id": target_id,
                    "antigen_seq": row["antigen_seq"],
                    "antigen_a3m": row["antigen_a3m"],
                    "designs": designs,
                    "return_coords": bool(return_coords),
                }
                atomic_write_json(self.queue / "new" / f"{job_id}.json", job)
                shards.append((job_id, designs, subset))

            all_res = self._await_many([s[0] for s in shards])
            for job_id, designs, subset in shards:
                res = all_res.get(job_id)
                for (i, seq), d in zip(subset, designs):
                    conf = None if res is None else res.get(d["design_id"])
                    results[i] = conf
                    if self._cache is not None and conf is not None:
                        self._cache[(target_id, seq)] = conf

        return results

    def rewards_for_group(
        self, target_id: str, binder_seqs: list[str], weights: dict[str, float] | None = None
    ) -> tuple[list[float], list[dict | None]]:
        """Convenience: :meth:`score_group` mapped through :func:`reward_from_confidence`.

        Parameters
        ----------
        target_id : str
            Target key into the manifest.
        binder_seqs : list[str]
            Binder amino-acid strings.
        weights : dict[str, float] | None, optional
            Confidence metric weights (defaults to :data:`DEFAULT_CONF_WEIGHTS`).

        Returns
        -------
        tuple[list[float], list[dict | None]]
            ``(rewards, confidences)`` — the scalar confidence rewards and the raw
            confidence dicts (the latter for logging pass rate / diagnostics).
        """
        confs = self.score_group(target_id, binder_seqs)
        rewards = [reward_from_confidence(c, weights) for c in confs]
        return rewards, confs

    def _await_many(self, job_ids: list[str]) -> dict[str, dict | None]:
        """Poll for several jobs' ``done``/``failed`` files until all resolve or timeout.

        Waits on all shard jobs of a group concurrently under one shared deadline, so
        the group's wall-clock is the *slowest* shard, not their sum. Each job resolves
        to its ``results`` dict (success), or ``None`` (failure or timeout) — a stuck
        shard floors only its own designs, never the whole group.

        Parameters
        ----------
        job_ids : list[str]
            Job ids to await.

        Returns
        -------
        dict[str, dict | None]
            ``{job_id: results_dict_or_None}`` for every requested job.
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
                        # Reader raced the writer's replace; retry on the next poll.
                        pass
                elif failed.exists():
                    logger.warning("Protenix job %s reported failure", jid)
                    out[jid] = None
                    pending.discard(jid)
            if pending:
                time.sleep(self.poll_s)
        for jid in pending:
            logger.warning("Protenix job %s timed out after %.0fs", jid, self.timeout_s)
            out[jid] = None
        return out
