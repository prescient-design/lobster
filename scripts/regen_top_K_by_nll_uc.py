"""Replay the top-K-by-NLL UC candidates and compute conference-style metrics.

The original `unconditional_bestofN_pll.py` run produced PLL scores + ESMFold
metrics for 1500 candidates (5 lengths × 10 slots × 30 best-of-N draws) but did
NOT save sequences or PDBs. To answer "what would Pass% / TM / H-S-C / Clusters
be if we sub-selected the top-K by NLL per length?", we need the actual
predicted structures.

This script:
  1. Reads the existing candidates CSV.
  2. For each length, ranks candidates by `--score-col` (default
     `struc_score_unif`) and keeps the top K (default 100).
  3. Replays generation deterministically with the original seed for each
     selected candidate.
  4. Computes ESMFold sc-TM/RMSD/pLDDT and P-SEA secondary-structure (H/E/C).
  5. Saves the ESMFold PDB to `<length>/esmfold_{idx}.pdb`.
  6. After all lengths, runs `foldseek easy-cluster` (TM ≥ 0.5) per length on
     the designable subset (RMSD < 2 Å) and counts clusters.
  7. Writes a conference-style markdown + LaTeX table.

Usage:
    uv run python scripts/regen_top_K_by_nll_uc.py \\
        --candidates /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional/bestofN_uc_candidates_20260503T020756.csv \\
        --ckpt /cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt \\
        --output-dir /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional/topK_by_struc_pll \\
        --score-col struc_score_unif --K 100
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regen_topK_uc")

# Pull helpers from the original best-of-N script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unconditional_bestofN_pll import (  # noqa: E402
    _build_logit_bias,
    _esmfold_one_candidate,
    _generate_one_candidate,
    _get_inference_schedule_class,
)
from analyze_secondary_structure import ss_content_from_coords  # noqa: E402

FOLDSEEK_BIN = "/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin"
TM_CLUSTER = 0.5
RMSD_DESIGNABLE = 2.0


def _setup_foldseek_path():
    try:
        from lobster.metrics.cal_foldseek_clusters import setup_foldseek_path
        setup_foldseek_path(FOLDSEEK_BIN)
    except Exception as e:
        logger.warning("setup_foldseek_path import failed (%s); assuming foldseek on PATH", e)


def _run_foldseek(pdb_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "foldseek", "easy-cluster",
        str(pdb_dir), str(output_dir / "res"), str(output_dir / "tmp"),
        "--alignment-type", "1",
        "--cov-mode", "0",
        "--min-seq-id", "0",
        "--tmscore-threshold", str(TM_CLUSTER),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logger.error("foldseek failed (%d): %s", p.returncode, p.stderr[:500])
        return 0
    rep = output_dir / "res_rep_seq.fasta"
    if rep.exists():
        return sum(1 for line in open(rep) if line.startswith(">"))
    return 0


def _writepdb_local(path: Path, coords: torch.Tensor, seq_std: torch.Tensor):
    """Minimal CA/N/C-only PDB writer (ESMFold returns N,CA,C)."""
    from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
    coords_np = coords.detach().cpu().numpy()
    seq = [restype_order_with_x_inv.get(int(t), "X") for t in seq_std.tolist()]
    AA3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU","G":"GLY",
           "H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER",
           "T":"THR","W":"TRP","Y":"TYR","V":"VAL","X":"GLY"}
    with path.open("w") as f:
        atom_idx = 1
        for i, aa in enumerate(seq):
            res3 = AA3.get(aa, "GLY")
            for atom_idx_local, atom_name in enumerate(["N", "CA", "C"]):
                x, y, z = coords_np[i, atom_idx_local]
                f.write(
                    f"ATOM  {atom_idx:>5d} {atom_name:>3s}  {res3} A{i+1:>4d}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {atom_name[0]}\n"
                )
                atom_idx += 1
        f.write("END\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--score-col", default="struc_score_unif")
    p.add_argument("--K", type=int, default=100, help="Top-K per length to replay")
    p.add_argument("--lengths", default="100,200,300,400,500")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # LEFLUR-P-VAL hyperparameters (must match original run)
    p.add_argument("--nsteps", type=int, default=400)
    p.add_argument("--temperature-seq", type=float, default=0.27315634404739075)
    p.add_argument("--temperature-struc", type=float, default=0.31640411575109995)
    p.add_argument("--stochasticity-seq", type=int, default=20)
    p.add_argument("--stochasticity-struc", type=int, default=60)
    p.add_argument("--inference-schedule-seq", default="LogInferenceSchedule")
    p.add_argument("--inference-schedule-struc", default="PowerInferenceSchedule")
    p.add_argument("--bias-V", type=float, default=1.0)
    p.add_argument("--bias-steps", type=int, default=25)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    cdf = pd.read_csv(args.candidates)
    cdf = cdf[cdf.length > 0].copy()

    selected = []
    for L in lengths:
        sub = cdf[cdf.length == L].nsmallest(args.K, args.score_col).copy()
        sub["pick_rank"] = np.arange(len(sub))
        selected.append(sub)
    sel = pd.concat(selected, ignore_index=True)
    logger.info("Selected %d top-K candidates across %d lengths", len(sel), len(lengths))

    from lobster.model import LobsterPLMFold
    from lobster.model.gen_ume import UMESequenceStructureEncoderLightningModule

    logger.info("Loading checkpoint: %s", args.ckpt)
    t0 = time.time()
    model = UMESequenceStructureEncoderLightningModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval(); model.to(device)
    model.interpolant_seq.device = device
    model.interpolant_struc.device = device
    logger.info("Model loaded in %.1fs", time.time() - t0)

    logger.info("Loading ESMFold...")
    t0 = time.time()
    plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=max(lengths))
    plm_fold.to(device); plm_fold.model.eval()
    logger.info("ESMFold loaded in %.1fs", time.time() - t0)

    sched_seq_cls = _get_inference_schedule_class(args.inference_schedule_seq)
    sched_struc_cls = _get_inference_schedule_class(args.inference_schedule_struc)
    bias_cfg = {"V": float(args.bias_V)} if args.bias_V is not None else None
    bias = _build_logit_bias(bias_cfg, device)
    gen_kwargs = dict(
        nsteps=args.nsteps,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        stochasticity_seq=args.stochasticity_seq,
        stochasticity_struc=args.stochasticity_struc,
        inference_schedule_seq=sched_seq_cls,
        inference_schedule_struc=sched_struc_cls,
        asynchronous_sampling=False,
        sequence_logit_bias=bias,
        sequence_logit_bias_steps=args.bias_steps,
    )

    out_csv = args.output_dir / f"topK_replay_{ts}.csv"
    fh = out_csv.open("w", newline="")
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "length", "slot", "candidate_idx", "seed", "pick_rank",
            "score_col", "score_value",
            "esmfold_plddt", "esmfold_tm_score", "esmfold_rmsd", "esmfold_pae",
            "ss_H", "ss_E", "ss_C", "ss_category",
            "designable", "pdb_path",
        ],
    )
    writer.writeheader()

    for L in lengths:
        (args.output_dir / f"L{L}").mkdir(parents=True, exist_ok=True)

    n_done = 0
    t_start = time.time()
    for _, row in sel.iterrows():
        L = int(row.length); slot = int(row.slot); cidx = int(row.candidate_idx)
        seed = int(row.seed); pick_rank = int(row.pick_rank)
        score_value = float(row[args.score_col])

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        try:
            seq_std, _seq_lob, _struc_arg, x_recon_xyz, _mask = _generate_one_candidate(model, L, gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM on L=%d slot=%d cidx=%d, skipping", L, slot, cidx)
            continue

        try:
            plddt, tm, rmsd, pae = _esmfold_one_candidate(plm_fold, seq_std, x_recon_xyz, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            plddt = tm = rmsd = pae = float("nan")

        # ESMFold predicted backbone (replay - run again for the PDB)
        from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
        seq_chars = "".join(restype_order_with_x_inv.get(int(t), "X") for t in seq_std[0].tolist()).replace("X", "A")
        tokenized = plm_fold.tokenizer.encode_plus(
            seq_chars, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        with torch.no_grad():
            outputs = plm_fold.model(tokenized)
        pred_coords = outputs["positions"][-1, 0, :, :3, :]  # [L, 3, 3]

        ss = ss_content_from_coords(pred_coords)
        from analyze_secondary_structure import assign_ss_category
        ss_cat = assign_ss_category(ss["H"], ss["E"])

        designable = (rmsd == rmsd) and (rmsd < RMSD_DESIGNABLE)
        pdb_path = args.output_dir / f"L{L}" / f"esmfold_slot{slot:03d}_cidx{cidx:02d}_rank{pick_rank:03d}.pdb"
        try:
            _writepdb_local(pdb_path, pred_coords, seq_std[0])
        except Exception as e:
            logger.warning("writepdb failed: %s", e)
            pdb_path = ""

        writer.writerow(
            {
                "length": L, "slot": slot, "candidate_idx": cidx, "seed": seed, "pick_rank": pick_rank,
                "score_col": args.score_col, "score_value": score_value,
                "esmfold_plddt": plddt, "esmfold_tm_score": tm, "esmfold_rmsd": rmsd, "esmfold_pae": pae,
                "ss_H": ss["H"], "ss_E": ss["E"], "ss_C": ss["C"], "ss_category": ss_cat,
                "designable": int(designable), "pdb_path": str(pdb_path),
            }
        )
        fh.flush()
        n_done += 1
        if n_done % 25 == 0:
            elapsed = time.time() - t_start
            logger.info(
                "[%4d/%d] L=%d slot=%d rank=%d  TM=%.3f RMSD=%.2f H=%.0f%% E=%.0f%% C=%.0f%% (%s)  %.1fs/cand",
                n_done, len(sel), L, slot, pick_rank, tm, rmsd, ss["H"]*100, ss["E"]*100, ss["C"]*100, ss_cat,
                elapsed/n_done,
            )

    fh.close()
    logger.info("Wrote %s (%d rows)", out_csv, n_done)

    # ---- Foldseek clustering on designables per length ----
    _setup_foldseek_path()
    rdf = pd.read_csv(out_csv)
    summary_rows = []
    for L in lengths:
        sub = rdf[rdf.length == L]
        N = len(sub)
        if N == 0:
            continue
        des = sub[sub.designable == 1]
        n_des = len(des)
        des_dir = args.output_dir / f"L{L}_designable_pdbs"
        des_dir.mkdir(parents=True, exist_ok=True)
        # Re-collect PDBs (they're already at sub.pdb_path)
        for pp in (Path(p) for p in des.pdb_path if p):
            if pp.exists():
                target = des_dir / pp.name
                if not target.exists():
                    target.write_bytes(pp.read_bytes())
        n_clust = 0
        if n_des > 0:
            fs_out = args.output_dir / f"foldseek_L{L}"
            n_clust = _run_foldseek(des_dir, fs_out)
        summary_rows.append(
            {
                "length": L,
                "N": N,
                "designable": n_des,
                "pass_pct": float(sub.designable.mean()) * 100,
                "tm_mean": float(sub.esmfold_tm_score.mean()),
                "rmsd_mean": float(sub.esmfold_rmsd.mean()),
                "plddt_mean": float(sub.esmfold_plddt.mean()),
                "ss_H_pct": float(sub.ss_H.mean()) * 100,
                "ss_E_pct": float(sub.ss_E.mean()) * 100,
                "ss_C_pct": float(sub.ss_C.mean()) * 100,
                "all_helical_pct": float((sub.ss_category == "all_helical").mean()) * 100,
                "all_beta_pct": float((sub.ss_category == "all_beta").mean()) * 100,
                "mixed_pct": float((sub.ss_category == "mixed").mean()) * 100,
                "n_clusters": n_clust,
            }
        )

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(args.output_dir / f"topK_summary_{ts}.csv", index=False)

    md = ["# Top-K-by-NLL UC replay (LEFLUR-P-VAL, GenUME-TED)", ""]
    md.append(f"Source candidates : `{args.candidates}`")
    md.append(f"Score column      : `{args.score_col}`")
    md.append(f"K per length      : {args.K}")
    md.append(f"Designable threshold: RMSD < {RMSD_DESIGNABLE} Å")
    md.append(f"Foldseek cluster  : easy-cluster, TM ≥ {TM_CLUSTER}")
    md.append("")
    md.append("| Length | N | Pass% | pLDDT | TM | RMSD (Å) | H% | E% | C% | Clusters |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in sdf.iterrows():
        md.append(
            f"| {int(r.length)} | {int(r.N)} | {r.pass_pct:.1f} | {r.plddt_mean:.3f} | {r.tm_mean:.3f} | "
            f"{r.rmsd_mean:.2f} | {r.ss_H_pct:.1f} | {r.ss_E_pct:.1f} | {r.ss_C_pct:.1f} | {int(r.n_clusters)} |"
        )

    md.append("")
    md.append("## Pooled (sum / weighted-mean across lengths)")
    md.append("")
    n_total = sdf.N.sum()
    md.append(f"- N total: {int(n_total)}")
    md.append(f"- Pass%: {(sdf.designable.sum() / n_total) * 100:.1f}")
    md.append(f"- TM (mean of mean): {sdf.tm_mean.mean():.3f}")
    md.append(f"- pLDDT (mean of mean): {sdf.plddt_mean.mean():.3f}")
    md.append(f"- H/E/C (mean of mean): {sdf.ss_H_pct.mean():.1f}% / {sdf.ss_E_pct.mean():.1f}% / {sdf.ss_C_pct.mean():.1f}%")
    md.append(f"- Total clusters: {int(sdf.n_clusters.sum())}")

    md.append("")
    md.append("## LaTeX")
    md.append("")
    md.append("```latex")
    md.append("\\begin{tabular}{rrrrrrrrr}")
    md.append("\\toprule")
    md.append("Length & $N$ & Pass\\% & pLDDT & TM & RMSD (\\AA) & H/S/C (\\%) & Clusters \\\\")
    md.append("\\midrule")
    for _, r in sdf.iterrows():
        md.append(
            f"{int(r.length)} & {int(r.N)} & {r.pass_pct:.1f} & {r.plddt_mean:.3f} & {r.tm_mean:.3f} & "
            f"{r.rmsd_mean:.2f} & {r.ss_H_pct:.0f} / {r.ss_E_pct:.0f} / {r.ss_C_pct:.0f} & {int(r.n_clusters)} \\\\"
        )
    md.append("\\bottomrule")
    md.append("\\end{tabular}")
    md.append("```")

    (args.output_dir / "topK_report.md").write_text("\n".join(md))
    logger.info("Wrote %s", args.output_dir / "topK_report.md")


if __name__ == "__main__":
    main()
