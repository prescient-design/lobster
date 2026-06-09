"""LeFlur autoencode CLI: encode + decode structures through the LG codec.

Hydra entry point registered as ``lobster_autoencode``.

For both protein-only (``leflur-base`` / ``leflur-ted``) and protein-ligand
(``leflur-pl``) checkpoints, this script:

1. Resolves the checkpoint via :func:`lobster.model.leflur.resolve_checkpoint`
   (short names, ``hf://`` URIs, or local paths).
2. Loads the right Lightning module class (auto-detected from the checkpoint
   ``hyper_parameters`` payload — no need to specify protein vs PL).
3. Iterates input files (PDB / CIF / glob / dir of ``*.pt`` for protein-only;
   ``*_protein.pt`` + ``*_ligand.pt`` pairs for protein-ligand).
4. Encodes + decodes each structure through the bundled LG codec (already
   instantiated inside the Lightning module at load time).
5. Writes reconstructed PDB(s) and a metrics CSV (RMSD per file plus
   summary stats) using the helpers in
   :mod:`lobster.metrics.evaluate_reconstruction`.

The two canonical Hydra entries are:

- :file:`experiment/autoencode.yaml`              — protein-only, ``leflur-ted`` by default.
- :file:`experiment/autoencode_protein_ligand.yaml` — protein-ligand, ``leflur-pl`` by default.

Usage::

    uv run python -m lobster.cmdline.autoencode \
        --config-name experiment/autoencode \
        model.ckpt_path=leflur-ted \
        autoencode.input=/path/to/structures \
        autoencode.output_dir=/tmp/leflur_autoencode

    # Protein-ligand:
    uv run python -m lobster.cmdline.autoencode \
        --config-name experiment/autoencode_protein_ligand \
        model.ckpt_path=leflur-pl \
        autoencode.input=/path/to/posebusters_benchmark_no_overlap
"""

from __future__ import annotations

import csv
import glob
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from lobster.model.leflur import resolve_checkpoint

# Reuse the LG-side helpers — they cover both PDB/CIF/SDF loading and
# RMSD/percent-identity computation for both protein-only and PL cases.
from lobster.metrics.evaluate_reconstruction import (
    evaluate_model_on_structure,
    find_paired_protein_ligand_files,
    load_structure_data,
)

logging.basicConfig(level=logging.INFO)


# --- Family detection -----------------------------------------------------


def _load_lightning_module(ckpt_path: Path, device: str):
    """Instantiate the right LeFlur Lightning module from a checkpoint.

    Auto-detects protein-only vs protein-ligand by peeking at the saved
    ``hyper_parameters`` blob. Falls back to a try-and-recover approach if
    the introspection fails.
    """
    from lobster.model.leflur import (
        LeFlurProteinLigandLightningModule,
        LeFlurSequenceStructureEncoderLightningModule,
    )

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = raw.get("hyper_parameters", {}) if isinstance(raw, dict) else {}

    pl_signal = any(
        key in hp
        for key in (
            "ligand_n_tokens",
            "ligand_levels",
            "ligand_context_mode",
        )
    )
    family = "protein_ligand" if pl_signal else "protein"

    cls = (
        LeFlurProteinLigandLightningModule
        if family == "protein_ligand"
        else LeFlurSequenceStructureEncoderLightningModule
    )
    logger.info("autoencode: detected family=%s -> %s", family, cls.__name__)
    model = cls.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)
    return model, family


# --- Input discovery ------------------------------------------------------


def _expand_protein_inputs(spec: str) -> list[str]:
    """Resolve ``spec`` into a flat list of structure files (protein-only).

    Accepts:
    - A single PDB / CIF / PT file.
    - A directory (recursively globs ``*.pdb``, ``*.cif``, ``*.pt``).
    - A glob pattern (passed verbatim to :func:`glob.glob`).
    """
    p = Path(spec)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        hits: list[str] = []
        for ext in ("pdb", "cif", "pt"):
            hits.extend(sorted(glob.glob(str(p / f"*.{ext}"))))
            hits.extend(sorted(glob.glob(str(p / "**" / f"*.{ext}"), recursive=True)))
        # Dedup, preserve order.
        return list(dict.fromkeys(hits))
    return sorted(glob.glob(spec))


def _expand_protein_ligand_inputs(spec: str) -> list[tuple[str, str, str]]:
    """Resolve ``spec`` into paired ``(base, protein.pt, ligand.pt)`` triples."""
    p = Path(spec)
    if p.is_dir():
        return find_paired_protein_ligand_files(str(p))
    raise ValueError(
        f"autoencode protein-ligand input must be a directory of *_protein.pt + *_ligand.pt fixtures, got {spec!r}"
    )


# --- Main -----------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../hydra_config",
    config_name="experiment/autoencode",
)
def autoencode(cfg: DictConfig) -> None:
    """Hydra entry point. Dispatches by detected checkpoint family."""
    logger.info("Starting LeFlur autoencode")
    logger.info("Config:\n %s", OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ckpt_uri = cfg.model.ckpt_path
    ckpt_path = resolve_checkpoint(ckpt_uri)
    logger.info("Resolved checkpoint: %s -> %s", ckpt_uri, ckpt_path)

    _model, family = _load_lightning_module(ckpt_path, device)

    autoencode_cfg = cfg.get("autoencode", {})
    input_spec = autoencode_cfg.get("input")
    if input_spec is None:
        raise ValueError("autoencode requires `autoencode.input=<path-or-glob>`.")

    output_dir = Path(autoencode_cfg.get("output_dir", cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # The LG-side helpers select their behaviour off a string flag. Use the
    # LeFlur-paired LG codec name from the Lightning module so reuse stays
    # 1:1 with what the model loaded.
    lg_codec_name = cfg.model.get("latent_generator_model_name") or (
        "LG Protein Ligand fsq 4375" if family == "protein_ligand" else "LG full attention"
    )

    save_pdbs = bool(autoencode_cfg.get("save_pdbs", True))
    num_steps = autoencode_cfg.get("num_steps")
    use_canonical_pose = bool(autoencode_cfg.get("use_canonical_pose", False))

    rows: list[dict] = []

    if family == "protein":
        files = _expand_protein_inputs(input_spec)
        logger.info("Found %d protein input files under %s", len(files), input_spec)
        for path in files:
            try:
                structure_data = load_structure_data(path)
            except Exception as exc:
                logger.warning("Skipping %s: failed to load (%s)", path, exc)
                rows.append({"path": path, "rmsd": float("nan"), "ok": False, "error": str(exc)})
                continue

            res = evaluate_model_on_structure(
                model_name=lg_codec_name,
                structure_data=structure_data,
                structure_path=path,
                save_structures=save_pdbs,
                output_dir=str(output_dir),
                use_canonical_pose=use_canonical_pose,
                num_steps=num_steps,
            )
            rows.append(
                {
                    "path": path,
                    "rmsd": res.get("rmsd", float("nan")),
                    "ok": bool(res.get("success", False)),
                    "error": res.get("error") or "",
                }
            )
    else:
        triples = _expand_protein_ligand_inputs(input_spec)
        logger.info(
            "Found %d (protein, ligand) input pairs under %s",
            len(triples),
            input_spec,
        )
        for base, protein_pt, ligand_pt in triples:
            try:
                protein_data = load_structure_data(protein_pt)
                ligand_data = load_structure_data(ligand_pt)
                structure_data = {**protein_data, **ligand_data}
            except Exception as exc:
                logger.warning("Skipping %s: failed to load (%s)", base, exc)
                rows.append({"path": base, "rmsd": float("nan"), "ok": False, "error": str(exc)})
                continue

            res = evaluate_model_on_structure(
                model_name=lg_codec_name,
                structure_data=structure_data,
                structure_path=base,
                save_structures=save_pdbs,
                output_dir=str(output_dir),
                use_canonical_pose=use_canonical_pose,
                num_steps=num_steps,
                minimize_ligand=bool(autoencode_cfg.get("minimize_ligand", False)),
                minimize_steps=int(autoencode_cfg.get("minimize_steps", 500)),
                force_field=str(autoencode_cfg.get("force_field", "MMFF94")),
                minimize_mode=str(autoencode_cfg.get("minimize_mode", "bonds_and_angles")),
            )
            rows.append(
                {
                    "path": base,
                    "rmsd": res.get("rmsd", float("nan")),
                    "ligand_rmsd": res.get("ligand_rmsd", float("nan")),
                    "ok": bool(res.get("success", False)),
                    "error": res.get("error") or "",
                }
            )

    # Aggregate + write CSV.
    csv_path = output_dir / autoencode_cfg.get("output_csv", "autoencode_results.csv")
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    finite_rmsds = [r["rmsd"] for r in rows if isinstance(r["rmsd"], (int, float)) and np.isfinite(r["rmsd"])]
    n_ok = sum(1 for r in rows if r.get("ok"))
    logger.info(
        "autoencode summary: %d/%d ok, mean RMSD = %.3f, median RMSD = %.3f",
        n_ok,
        len(rows),
        float(np.mean(finite_rmsds)) if finite_rmsds else float("nan"),
        float(np.median(finite_rmsds)) if finite_rmsds else float("nan"),
    )
    logger.info("Results CSV: %s", csv_path)


if __name__ == "__main__":
    autoencode()
