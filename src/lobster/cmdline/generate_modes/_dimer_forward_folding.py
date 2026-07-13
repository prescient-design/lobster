"""LeFlur DIMER forward-folding mode.

Predicts a multi-chain (complex) structure given an input multi-chain
sequence. Same backbone as the monomer ``_forward_folding`` mode --
sequence held fixed, structure stream sampled, decoded via the ViT
decoder -- but the inputs carry per-residue chain identity so the
encoder's chain-embedding fires and the model produces a complex pose,
not two collapsed monomers.

Scoring is reported at TWO granularities so we can decompose
per-chain folding quality from inter-chain pose quality:

  * **per-chain TM/RMSD** -- each predicted chain aligned INDEPENDENTLY
    to its GT chain (TM-align per chain, Kabsch per chain). Aggregate:
    mean over all chains in all dimers. Directly comparable to the
    monomer fwd-fold eval since the alignment unit is one chain.

  * **complex TM/RMSD** -- the WHOLE predicted dimer aligned to the
    whole GT dimer in a SINGLE rigid-body alignment. TM-align is run
    once over the concatenated CA atoms (full sequence string passed in
    so its length normalization uses the full dimer L). Captures both
    per-chain shape AND inter-chain placement; degrades sharply when
    the relative pose is wrong even if both chains fold individually.

Called from :func:`lobster.cmdline.generate.generate` when
``cfg.generation.mode == "dimer_forward_folding"``.

Reads from ``cfg.generation``:
    input_structures        glob to Pinder-style multi-chain .pt files
    nsteps, temperature_*,  flow-matching sampling knobs (defaults match
        stochasticity_*           the monomer config)
    max_length              dimers with total L > this are dropped
    use_chain_ids           if true, pass per-residue chain_ids to
                            generate_sample (default true)
    use_epitope_conditioning  if true, pass GT interface residues as
                            ``conditioning_tensor_override`` (oracle
                            hint; default false)
    hotspot_cutoff_a        CA-CA inter-chain distance threshold for
                            "interface residue" (default 10.0)
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig, ListConfig
from tmtools import tm_align

from lobster.metrics import align_and_compute_rmsd
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils import apply_random_se3_batched
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.transforms._structure_transforms import (
    AminoAcidTokenizerTransform,
    Atom14ToBackboneTransform,
)


def _dockq_score(model_pdb: str, native_pdb: str, dockq_python: str, dockq_script: str) -> float | None:
    """Run DockQ (isolated venv, subprocess) on a predicted vs native complex PDB.
    Returns the best-interface DockQ score, or None on failure."""
    try:
        out = subprocess.run(
            [dockq_python, dockq_script, model_pdb, native_pdb], capture_output=True, text=True, timeout=300
        )
        lines = out.stdout.strip().splitlines()
        d = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else {}
        return float(d["DockQ"]) if "DockQ" in d else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"DockQ failed: {e}")
        return None


def _write_pdb_multichain(
    filename: str,
    coords: torch.Tensor,
    seq_openfold: torch.Tensor,
    chain_ids_per_res: torch.Tensor,
    bfactors: torch.Tensor | None = None,
) -> None:
    """Write a multi-chain PDB by stitching together one ``writepdb`` call
    per chain. ``writepdb`` hard-codes chain ``A`` at column 21 -- we
    rewrite that column to ``A, B, C, ...`` in the order chain ids appear,
    and bump the residue numbers so chains don't share residue indices.

    ``coords``           (L, 3, 3) backbone N/CA/C atoms
    ``seq_openfold``     (L,) openfold AA indices (0-19, 20 = UNK)
    ``chain_ids_per_res`` (L,) integer chain id per residue (any
                          convention; sorted uniques become A/B/...)
    """
    import os as _os
    import tempfile as _tempfile

    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_ids_per_res = chain_ids_per_res.cpu()
    uniques = sorted(int(c) for c in chain_ids_per_res.unique().tolist())
    blocks: list[str] = []
    res_offset = 0
    for chain_idx, raw_id in enumerate(uniques):
        if chain_idx >= len(ALPHA):
            break
        chain_letter = ALPHA[chain_idx]
        sel = (chain_ids_per_res == raw_id).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        sub_coords = coords[sel]
        sub_seq = seq_openfold[sel]
        sub_bf = bfactors[sel] if bfactors is not None else None
        with _tempfile.NamedTemporaryFile(mode="r+", suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        writepdb(tmp_path, sub_coords, sub_seq, bfacts=sub_bf, add_cb_o=True)
        with open(tmp_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    orig_resnum = int(line[22:26])
                    new_resnum = orig_resnum + res_offset
                    line = line[:21] + chain_letter + f"{new_resnum:>4}" + line[26:]
                    blocks.append(line)
            blocks.append("TER\n")
        _os.unlink(tmp_path)
        res_offset += int(sub_seq.shape[0])
    with open(filename, "w") as f:
        f.writelines(blocks)
        f.write("END\n")


def _load_dimer_pt(path: str, structure_transform, tokenizer_transform, max_length: int = 512) -> dict | None:
    """Load a Pinder-style multi-chain .pt and normalize it to the
    monomer-like schema expected by the rest of this pipeline.

    Returns dict with keys: coords_res (L, 3, 3), sequence (L,) tokens,
    sequence_str (str), chains_ids (L,), indices (L,), mask (L,), name.
    Returns None if the file can't be read, lacks required fields, OR
    if the raw coords length exceeds ``max_length`` (this check has to
    happen BEFORE tokenization because the AA tokenizer silently
    truncates long sequences to ``max_length - 2``, which would otherwise
    let an oversized dimer through in a corrupted form).
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d is None:
        return None
    # Pinder files: atom14_coords (L, 14, 3) + atom14_mask. CAMEO files:
    # coords_res (L, 3, 3) already. Handle both.
    if "atom14_coords" in d and "coords_res" not in d:
        d = structure_transform(d)
    if "coords_res" not in d or "sequence" not in d:
        logger.warning(f"{path}: missing coords_res or sequence -- skip")
        return None
    # Skip oversized dimers BEFORE tokenization (the tokenizer truncates
    # to max_length-2, so doing this check after would let a 654-residue
    # dimer through as a corrupted 510-residue one).
    if d["coords_res"].shape[0] > max_length:
        return None
    seq = d["sequence"]
    if seq.dim() > 1:
        seq = seq.squeeze()
    d["sequence"] = tokenizer_transform({"sequence": seq})["sequence"]
    # Off-by-one safety guard: in rare cases the raw Pinder sequence and
    # coords arrays differ by exactly 1 residue (likely a non-standard
    # residue that survives in sequence but not in atom14). Skip those
    # rather than truncate (truncation would silently corrupt the data).
    L_seq = int(d["sequence"].shape[0])
    L_coords = int(d["coords_res"].shape[0])
    if L_seq != L_coords:
        logger.warning(f"{Path(path).stem}: length mismatch seq={L_seq} coords={L_coords} -- skip")
        return None
    if "mask" not in d:
        d["mask"] = torch.ones(L_coords, dtype=torch.float32)
    if "chains_ids" not in d:
        d["chains_ids"] = torch.zeros(L_coords, dtype=torch.long)
    if "indices" not in d:
        d["indices"] = torch.arange(L_coords, dtype=torch.long)
    d["name"] = Path(path).stem
    return d


def _build_chain_ids_for_embedding(chains_ids: torch.Tensor, max_num_chains: int = 2) -> torch.Tensor:
    """Remap raw chain ids in any range to dense 1..max_num_chains
    (0 reserved for padding). Mirrors the training-time
    ``RemapChainIdsForEmbedding`` transform without importing it.
    """
    unique = sorted(int(c) for c in chains_ids.unique().tolist())
    mapping = {c: (idx + 1) for idx, c in enumerate(unique[:max_num_chains])}
    out = torch.zeros_like(chains_ids)
    for raw, dense in mapping.items():
        out[chains_ids == raw] = dense
    return out


def _interface_residues_local(coords_res: torch.Tensor, chains_ids: torch.Tensor, cutoff_a: float) -> torch.Tensor:
    """Boolean (L,) marking inter-chain interface residues by CA-CA
    pairwise distance < ``cutoff_a``. Used to build the oracle
    conditioning tensor when ``use_epitope_conditioning`` is set."""
    ca = coords_res[:, 1, :]
    diff = ca[:, None, :] - ca[None, :, :]
    dist = torch.sqrt((diff**2).sum(dim=-1))
    diff_chain = chains_ids[:, None] != chains_ids[None, :]
    inter = torch.where(diff_chain, dist, torch.full_like(dist, float("inf")))
    return inter.min(dim=-1).values < cutoff_a


def _is_homodimer(seq_str: str, chains_ids: torch.Tensor, min_id: float = 0.90) -> bool:
    """True if the two chains share >= ``min_id`` sequence identity (homodimer).

    Standard homo/hetero split for dimer forward-folding reporting: homodimers have a
    symmetric interface and are the easier docking case, so DockQ/TM are reported split
    by this flag. ``seq_str`` = per-residue AA string (len L), ``chains_ids`` = (L,).
    """
    uch = sorted(int(c) for c in chains_ids.unique().tolist())
    if len(uch) < 2:
        return False
    a = [seq_str[i] for i in range(min(len(seq_str), chains_ids.shape[0])) if int(chains_ids[i]) == uch[0]]
    b = [seq_str[i] for i in range(min(len(seq_str), chains_ids.shape[0])) if int(chains_ids[i]) == uch[1]]
    if not a or not b:
        return False
    L = min(len(a), len(b))
    same = sum(1 for i in range(L) if a[i] == b[i])
    return same / max(len(a), len(b)) >= min_id


def _generate_dimer_forward_folding(
    model,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path,
    plm_fold=None,
    csv_writer=None,
    plotter=None,
) -> None:
    """Generate complex structures from multi-chain input sequences."""
    logger.info("Starting DIMER forward folding generation...")

    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for dimer_forward_folding mode")

    # Resolve glob / list / file / dir to a flat list of paths (same
    # logic as the monomer mode -- copied verbatim to keep the modes
    # self-contained).
    structure_paths: list[str] = []
    if isinstance(input_structures, str):
        if "*" in input_structures or "?" in input_structures:
            structure_paths = sorted(glob.glob(input_structures))
        else:
            path = Path(input_structures)
            if path.is_file():
                structure_paths = [str(path)]
            elif path.is_dir():
                structure_paths = sorted(glob.glob(str(path / "*.pt")))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, tuple, ListConfig)):
        for path_str in input_structures:
            if Path(path_str).is_file():
                structure_paths.append(str(path_str))
            else:
                logger.warning(f"Skipping non-existent file: {path_str}")
    else:
        raise ValueError(f"Invalid input_structures format: {type(input_structures)}")
    if not structure_paths:
        raise ValueError("No valid structure files found in input_structures")
    logger.info(f"Found {len(structure_paths)} dimer files to process")

    gen_cfg = cfg.generation
    # Multi-GPU sharding: each array task processes a disjoint stride of the
    # structures (structure_paths[shard::num_shards]); merge the per-shard CSVs
    # afterward. Default num_shards=1 -> single-process (unchanged).
    shard = int(gen_cfg.get("shard", 0))
    num_shards = int(gen_cfg.get("num_shards", 1))
    if num_shards > 1:
        structure_paths = structure_paths[shard::num_shards]
        logger.info(f"Shard {shard}/{num_shards}: processing {len(structure_paths)} of the dimers")
    max_length = gen_cfg.get("max_length", 512)
    nsteps = gen_cfg.get("nsteps", 200)
    use_chain_ids = gen_cfg.get("use_chain_ids", True)
    use_epitope_conditioning = gen_cfg.get("use_epitope_conditioning", False)
    hotspot_cutoff_a = gen_cfg.get("hotspot_cutoff_a", 10.0)
    # Template conditioning (per-chain, isolated, INDEPENDENT random SE(3) frame per chain --
    # matches training's build_template_tokens / ForwardFoldingCallback). Requires a
    # template-trained checkpoint (model.template_percentage>0 -> model.no_template_idx set).
    use_template = gen_cfg.get("use_template", False)
    cfg_weight = float(gen_cfg.get("cfg_weight", 1.0))  # classifier-free guidance on epitope conditioning
    # Structure inference schedule: default Linear (uniform t). Set inference_schedule_struc +
    # schedule_exponent to front-load steps at low t (PowerInferenceSchedule exponent>1 -> more
    # pose-forming-window steps). Callable is instantiated with nsteps inside generate_sample.
    import functools as _functools

    from lobster.cmdline.generate_modes._shared import _get_inference_schedule_class
    from lobster.model.leflur._leflur_sequence_structure_encoder_lightning_module import LinearInferenceSchedule

    _sched_name = gen_cfg.get("inference_schedule_struc", None)
    _sched_exp = gen_cfg.get("schedule_exponent", None)
    struct_schedule = LinearInferenceSchedule
    if _sched_name:
        _cls = _get_inference_schedule_class(_sched_name)
        struct_schedule = _functools.partial(_cls, exponent=float(_sched_exp)) if _sched_exp is not None else _cls
    logger.info(f"structure inference schedule: {_sched_name or 'Linear'} exponent={_sched_exp}")
    # DockQ interface scoring (isolated venv via subprocess; numpy<2). Off by default.
    use_dockq = gen_cfg.get("use_dockq", False)
    dockq_python = gen_cfg.get("dockq_python", None) or os.environ.get(
        "DOCKQ_PYTHON", "/cv/scratch/u/lisanzas/.dockq_venv/bin/python"
    )
    dockq_script = gen_cfg.get("dockq_script", None) or os.environ.get(
        "DOCKQ_SCRIPT", "/cv/home/lisanzas/lobster/scripts/_dockq_pair.py"
    )

    # Two transforms: atom14 -> backbone (for Pinder schema) and AA
    # tokenizer (to match the encoder's expected vocab).
    atom14_to_backbone = Atom14ToBackboneTransform()
    tokenizer_transform = AminoAcidTokenizerTransform(max_length=max_length)

    all_per_chain_tm: list[float] = []
    all_per_chain_rmsd: list[float] = []
    all_complex_tm: list[float] = []
    all_complex_rmsd: list[float] = []
    all_complex_dockq: list[float] = []
    # homo/hetero split (standard reporting): DockQ + complex-TM keyed by dimer type
    dockq_by_type: dict[str, list[float]] = {"homo": [], "hetero": []}
    cplx_tm_by_type: dict[str, list[float]] = {"homo": [], "hetero": []}

    with torch.no_grad():
        for i, structure_path in enumerate(structure_paths):
            logger.info(f"[{i + 1}/{len(structure_paths)}] {Path(structure_path).name}")
            d = _load_dimer_pt(structure_path, atom14_to_backbone, tokenizer_transform, max_length=max_length)
            if d is None:
                continue
            L = d["coords_res"].shape[0]
            if L < 30:
                logger.info(f"  L={L} < 30 -- skip")
                continue
            if L > max_length:
                logger.info(f"  L={L} > max_length={max_length} -- skip")
                continue
            n_unique = len(d["chains_ids"].unique())
            if n_unique < 2:
                logger.info(f"  only {n_unique} chain in file -- skip (use monomer mode)")
                continue

            # Generate at the NATIVE dimer length L, not a fixed padded
            # canvas. The earlier "pad to max_length for batch
            # consistency" comment was wrong: the production
            # ForwardFoldingCallback samples at
            # ``max(coords_res.shape[0] for batch)`` (tight to the real
            # structures), and padding a short complex out to 512 lets the
            # flow-matching velocity field act over hundreds of empty
            # positions and blows the geometry apart. Measured on the
            # monomer path: native length gives CAMEO TM 0.67 vs 0.18 when
            # padded to 512 (same ckpt, same temperature). max_length stays
            # the *cap* (oversized dimers were already skipped above); the
            # sample length is L. With L positions there is no padding, so
            # the mask is all ones.
            input_seq = torch.zeros(1, L, dtype=torch.long, device=device)
            input_seq[0, :L] = d["sequence"].to(device)
            mask = torch.ones(1, L, device=device)
            indices = torch.arange(L, device=device).unsqueeze(0)
            chain_ids_arg = None
            if use_chain_ids:
                cid = torch.zeros(1, L, dtype=torch.long, device=device)
                cid[0, :L] = _build_chain_ids_for_embedding(d["chains_ids"]).to(device)
                chain_ids_arg = cid
            cond_arg = None
            if use_epitope_conditioning:
                # Match TRAINING's epitope_tensor exactly: interface residues (CA-CA < cutoff,
                # bilateral) restricted to ONE chain (the "epitope" side). Training picks a
                # random chain as paratope and conditions on the OTHER chain's interface only;
                # feeding BOTH sides would be out-of-distribution. We pick the last chain id
                # deterministically as the epitope side for reproducibility.
                inter = _interface_residues_local(d["coords_res"], d["chains_ids"], cutoff_a=hotspot_cutoff_a)
                chains_local = d["chains_ids"]
                epi_chain = sorted(int(c) for c in chains_local.unique().tolist())[-1]
                ep_local = inter & (chains_local == epi_chain)
                cond_tensor = torch.zeros(1, L, 1, device=device)
                cond_tensor[0, :L, 0] = ep_local.float().to(device)
                cond_arg = cond_tensor

            # Per-chain template tokens: encode each chain in ISOLATION (partner masked out ->
            # no interface/pose leak) under its OWN random SE(3) frame, argmax to structure
            # tokens, assign only to that chain's residues. Gives the model each chain's fold as
            # a template while withholding the relative pose (the docking target).
            template_arg = None
            if use_template and getattr(model, "no_template_idx", None) is not None and chain_ids_arg is not None:
                coords_b = d["coords_res"].unsqueeze(0).to(device)  # (1, L, 3, 3)
                template_tokens = torch.full((1, L), model.no_template_idx, dtype=torch.long, device=device)
                for c in [int(x) for x in chain_ids_arg.unique().tolist() if int(x) != 0]:
                    mask_c = mask * (chain_ids_arg == c).float()
                    if mask_c.sum() == 0:
                        continue
                    coords_c = apply_random_se3_batched(coords_b.clone())  # independent frame per chain
                    xq_c, _, _ = model.encode_structure(coords_c, mask_c, indices)
                    tok_c = xq_c.argmax(dim=-1)  # (1, L)
                    sel = chain_ids_arg == c
                    template_tokens[sel] = tok_c[sel]
                template_arg = template_tokens

            gen = model.generate_sample(
                length=L,
                num_samples=1,
                nsteps=nsteps,
                forward_folding=True,
                input_sequence_tokens=input_seq,
                input_mask=mask,
                input_indices=indices,
                temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                temperature_struc=gen_cfg.get("temperature_struc", 0.2195534567490864),
                stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                chain_ids=chain_ids_arg,
                conditioning_tensor_override=cond_arg,
                template_structure_tokens=template_arg,
                cfg_weight=cfg_weight,
                inference_schedule_struc=struct_schedule,
                asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
            )
            decoded = model.decode_structure(gen, mask)
            x_recon = decoded.get("vit_decoder")
            if x_recon is None:
                logger.warning("  no vit_decoder output -- skip")
                continue
            pred_xyz = x_recon[0, :L].detach().cpu()  # (L, 3, 3)
            gt_xyz = d["coords_res"][:L].cpu()  # (L, 3, 3)
            pred_ca = pred_xyz[:, 1, :].numpy()  # (L, 3)
            gt_ca = gt_xyz[:, 1, :].numpy()

            # Reconstruct sequence string from generated logits (same as
            # monomer mode) for TM-align.
            if gen["sequence_logits"].shape[-1] == 33:
                seq = convert_lobster_aa_tokenization_to_standard_aa(gen["sequence_logits"], device=device)[0, :L]
            else:
                seq = gen["sequence_logits"][0, :L].argmax(dim=-1)
                seq[seq > 21] = 20
            seq_full_str = "".join(restype_order_with_x_inv[int(t)] for t in seq.tolist())

            # ------------- per-chain TM/RMSD (independent alignments) -------------
            chains_local = d["chains_ids"][:L]
            per_chain_tm: list[float] = []
            per_chain_rmsd: list[float] = []
            per_chain_str: list[str] = []
            unique_chains = sorted(int(c) for c in chains_local.unique().tolist())
            for cid_raw in unique_chains:
                sel = (chains_local == cid_raw).nonzero(as_tuple=True)[0]
                if sel.numel() < 5:
                    continue
                seq_chain = "".join(seq_full_str[int(j)] for j in sel.tolist())
                tm_out_chain = tm_align(pred_ca[sel.numpy()], gt_ca[sel.numpy()], seq_chain, seq_chain)
                rmsd_chain = align_and_compute_rmsd(
                    coords1=pred_xyz[sel].to(device),
                    coords2=gt_xyz[sel].to(device),
                    mask=None,
                    return_aligned=False,
                    device=device,
                )
                per_chain_tm.append(tm_out_chain.tm_norm_chain1)
                per_chain_rmsd.append(rmsd_chain)
                per_chain_str.append(
                    f"chain{cid_raw}: TM={tm_out_chain.tm_norm_chain1:.3f} RMSD={rmsd_chain:.2f}Å L={sel.numel()}"
                )

            # ------------- complex TM/RMSD (single rigid-body alignment) -------------
            tm_out_complex = tm_align(pred_ca, gt_ca, seq_full_str, seq_full_str)
            pred_xyz_aligned, rmsd_complex = align_and_compute_rmsd(
                coords1=pred_xyz.to(device),
                coords2=gt_xyz.to(device),
                mask=None,
                return_aligned=True,
                device=device,
            )
            pred_xyz_aligned = pred_xyz_aligned.detach().cpu()

            mean_chain_tm = float(sum(per_chain_tm) / len(per_chain_tm)) if per_chain_tm else float("nan")
            mean_chain_rmsd = float(sum(per_chain_rmsd) / len(per_chain_rmsd)) if per_chain_rmsd else float("nan")

            all_per_chain_tm.extend(per_chain_tm)
            all_per_chain_rmsd.extend(per_chain_rmsd)
            all_complex_tm.append(float(tm_out_complex.tm_norm_chain1))
            all_complex_rmsd.append(float(rmsd_complex))

            logger.info(
                f"  per-chain: {' | '.join(per_chain_str)} | "
                f"mean_chain_TM={mean_chain_tm:.3f} mean_chain_RMSD={mean_chain_rmsd:.2f}Å | "
                f"COMPLEX_TM={tm_out_complex.tm_norm_chain1:.3f} COMPLEX_RMSD={rmsd_complex:.2f}Å"
            )

            # ------------- write multi-chain dimer PDBs -------------
            # Single PDB per dimer with both chains labelled A/B/...
            # ``_generated.pdb`` carries the predicted complex Kabsch-
            # aligned to the GT complex (single rigid-body transform over
            # all CA atoms of both chains together -- same alignment that
            # produced ``complex_rmsd``). Loading the generated + original
            # PDBs together in PyMOL puts them in the same frame so you
            # see the prediction errors at the right scale; no manual
            # ``align`` command needed.
            name = d["name"]
            gen_pdb = str(output_dir / f"dimer_forward_folding_{name}_generated.pdb")
            orig_pdb = str(output_dir / f"dimer_forward_folding_{name}_original.pdb")
            # Hotspot coloring: B-factor = 100 at the epitope-conditioned residues (cond_arg), 0 else,
            # so gen/native PDBs can be colored by B-factor. None when no epitope conditioning.
            bfac = (100.0 * cond_arg[0, :L, 0].detach().cpu()) if cond_arg is not None else None
            try:
                _write_pdb_multichain(gen_pdb, pred_xyz_aligned, seq, chains_local, bfactors=bfac)
                _write_pdb_multichain(orig_pdb, gt_xyz, seq, chains_local, bfactors=bfac)
            except Exception as e:
                logger.warning(f"PDB dump failed for {name}: {type(e).__name__}: {e}")

            # ------------- homo vs hetero dimer (standard reporting) -------------
            dimer_type = "homo" if _is_homodimer(seq_full_str, d["chains_ids"][:L]) else "hetero"
            cplx_tm_by_type[dimer_type].append(float(tm_out_complex.tm_norm_chain1))

            # ------------- DockQ interface quality (predicted vs native) -------------
            dockq_val = None
            if use_dockq and len(unique_chains) >= 2:
                dockq_val = _dockq_score(gen_pdb, orig_pdb, dockq_python, dockq_script)
                if dockq_val is not None:
                    all_complex_dockq.append(dockq_val)
                    dockq_by_type[dimer_type].append(dockq_val)
                    logger.info(f"  DockQ={dockq_val:.3f} ({dimer_type})")

            # CSV row per dimer.
            if csv_writer is not None:
                csv_writer.write_batch_metrics(
                    {
                        "mean_chain_tm_score": mean_chain_tm,
                        "mean_chain_rmsd": mean_chain_rmsd,
                        "complex_tm_score": float(tm_out_complex.tm_norm_chain1),
                        "complex_rmsd": float(rmsd_complex),
                        "complex_dockq": dockq_val if dockq_val is not None else float("nan"),
                        "is_homodimer": int(dimer_type == "homo"),
                    },
                    run_id=f"dimer_forward_folding_{name}",
                    sequence_length=L,
                    input_file=name,
                )

    # Aggregate
    logger.info("=" * 80)
    logger.info("DIMER FORWARD FOLDING AGGREGATE STATISTICS")
    logger.info("=" * 80)

    def _agg(label: str, vals: list[float], pass_threshold: float | None = None):
        if not vals:
            logger.warning(f"{label}: no data")
            return
        vals_clean = [v for v in vals if v != float("inf")]
        n = len(vals_clean)
        mean = sum(vals_clean) / n
        logger.info(f"Average {label}: {mean:.4f} (n={n})")
        if pass_threshold is not None:
            pass_count = sum(1 for v in vals_clean if v < pass_threshold)
            logger.info(f"{label} pass rate (< {pass_threshold}): {pass_count}/{n} ({100 * pass_count / n:.1f}%)")

    _agg("per-chain TM-Score", all_per_chain_tm)
    _agg("per-chain RMSD (Å)", all_per_chain_rmsd, pass_threshold=2.0)
    _agg("COMPLEX TM-Score", all_complex_tm)
    _agg("COMPLEX RMSD (Å)", all_complex_rmsd, pass_threshold=5.0)
    if all_complex_dockq:
        _agg("COMPLEX DockQ", all_complex_dockq)
        n = len(all_complex_dockq)
        acc = sum(1 for q in all_complex_dockq if q >= 0.23)
        logger.info(f"COMPLEX DockQ acceptable (>= 0.23): {acc}/{n} ({100 * acc / n:.1f}%)")

    # ------------- homo vs hetero breakdown (standard reporting) -------------
    logger.info("-" * 80)
    logger.info("HOMO vs HETERO dimer breakdown (homodimer = two chains >=90% seq id)")
    n_homo = len(cplx_tm_by_type["homo"])
    n_het = len(cplx_tm_by_type["hetero"])
    n_tot = n_homo + n_het
    if n_tot:
        logger.info(
            f"  composition: homo={n_homo} ({100 * n_homo / n_tot:.1f}%)  hetero={n_het} ({100 * n_het / n_tot:.1f}%)"
        )
    for t in ("homo", "hetero"):
        tm = cplx_tm_by_type[t]
        dq = dockq_by_type[t]
        if tm:
            logger.info(f"  [{t}] COMPLEX TM: mean={sum(tm) / len(tm):.3f} (n={len(tm)})")
        if dq:
            acc_t = sum(1 for q in dq if q >= 0.23)
            logger.info(
                f"  [{t}] COMPLEX DockQ: mean={sum(dq) / len(dq):.3f}  "
                f"acceptable(>=0.23)={acc_t}/{len(dq)} ({100 * acc_t / len(dq):.1f}%)"
            )
