"""Run a single local protenix or boltz co-folding prediction.

This is a standalone worker script meant to be called from a SLURM array job.
Each invocation processes one input JSON and writes one output JSON.

Protenix and boltz are installed in isolated venvs to avoid dependency conflicts
with lobster. The venv paths are configured via CLI args or environment variables.

Usage:
    python -m lobster.cmdline.run_cofold_local \
        --input_json /path/to/input_0.json \
        --output_json /path/to/output_0.json \
        --backend protenix

Input JSON format:
    {"id": "sample_id", "sequence": "MKWVT...", "smiles": "CCO"}

Output JSON format:
    {"id": "sample_id", "confidence": {...}, "structure": "data_...", "error": null}
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROTENIX_VENV = os.environ.get("PROTENIX_VENV", "/cv/scratch/u/lisanzas/uv_envs/protenix/.venv")
BOLTZ_VENV = os.environ.get("BOLTZ_VENV", "/cv/scratch/u/lisanzas/uv_envs/boltz/.venv")
PROTENIX_ROOT_DIR = os.environ.get("PROTENIX_ROOT_DIR", "/cv/scratch/u/lisanzas/protenix_data")


def run_protenix(
    sequence: str,
    smiles: str | None,
    work_dir: Path,
    model_name: str = "protenix_base_default_v1.0.0",
    n_cycle: int = 4,
    n_step: int = 20,
    n_sample: int = 1,
    seed: int = 101,
) -> dict:
    """Run protenix CLI and return {confidence, structure}."""
    sequences = [{"proteinChain": {"sequence": sequence, "count": 1}}]
    if smiles is not None:
        sequences.append({"ligand": {"ligand": smiles, "count": 1}})

    json_data = [
        {
            "name": "prediction",
            "sequences": sequences,
            "covalent_bonds": [],
        }
    ]

    input_json = work_dir / "input.json"
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json, "w") as f:
        json.dump(json_data, f, indent=2)

    protenix_bin = os.path.join(PROTENIX_VENV, "bin", "protenix")
    cmd = [
        protenix_bin,
        "pred",
        "-i",
        str(input_json),
        "-o",
        str(output_dir),
        "-s",
        str(seed),
        "-n",
        model_name,
        "-c",
        str(n_cycle),
        "-p",
        str(n_step),
        "-e",
        str(n_sample),
        "--use_default_params",
        "true",
    ]

    if "v1.0.0" in model_name:
        cmd.extend(["--use_msa", "false", "--use_template", "false", "--use_rna_msa", "false"])

    env = os.environ.copy()
    env["PATH"] = os.path.join(PROTENIX_VENV, "bin") + ":" + env.get("PATH", "")
    env["PROTENIX_ROOT_DIR"] = PROTENIX_ROOT_DIR

    logger.info("Running: %s", " ".join(cmd))
    logger.info("PROTENIX_ROOT_DIR=%s", PROTENIX_ROOT_DIR)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Protenix failed (rc={result.returncode}): {result.stderr[-2000:]}")

    err_dir = output_dir / "ERR"
    if err_dir.exists():
        err_files = list(err_dir.glob("*.txt"))
        if err_files:
            err_text = err_files[0].read_text().strip()
            if err_text:
                raise RuntimeError(f"Protenix error: {err_text[:1000]}")

    return _collect_protenix_output(output_dir)


def _collect_protenix_output(output_dir: Path) -> dict:
    """Parse protenix output directory into {confidence, structure}."""
    result = {}

    confidence_files = sorted(output_dir.rglob("*summary_confidence_sample_*.json"))
    if confidence_files:
        with open(confidence_files[0]) as f:
            result["confidence"] = json.load(f)

    for ext in ("*.cif", "*.mmcif", "*.pdb"):
        candidates = sorted(output_dir.rglob(ext))
        if candidates:
            result["structure"] = candidates[0].read_text()
            break
    else:
        result["structure"] = None

    return result


def run_boltz(
    sequence: str,
    smiles: str | None,
    work_dir: Path,
    model: str = "boltz2",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
) -> dict:
    """Run boltz CLI and return {confidence, structure}."""
    import yaml

    yaml_data: dict = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}},
        ],
    }
    if smiles is not None:
        yaml_data["sequences"].append({"ligand": {"id": "B", "smiles": smiles}})

    input_yaml = work_dir / "input.yaml"
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_yaml, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    boltz_bin = os.path.join(BOLTZ_VENV, "bin", "boltz")
    cmd = [
        boltz_bin,
        "predict",
        str(input_yaml),
        "--out_dir",
        str(output_dir),
        "--recycling_steps",
        str(recycling_steps),
        "--sampling_steps",
        str(sampling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--accelerator",
        "gpu",
        "--output_format",
        "mmcif",
        "--model",
        model,
        "--override",
    ]

    env = os.environ.copy()
    env["PATH"] = os.path.join(BOLTZ_VENV, "bin") + ":" + env.get("PATH", "")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Boltz failed (rc={result.returncode}): {result.stderr[-2000:]}")

    return _collect_boltz_output(output_dir)


def _collect_boltz_output(output_dir: Path) -> dict:
    """Parse boltz output directory into {confidence, structure}.

    Boltz2 creates nested structure: output_dir/boltz_results_*/predictions/<name>/
    """
    result = {}

    # Boltz2 nests under boltz_results_*/predictions/<name>/
    search_dir = output_dir
    boltz_results = sorted(output_dir.glob("boltz_results_*/predictions"))
    if boltz_results:
        pred_dir = boltz_results[0]
        subdirs = [d for d in pred_dir.iterdir() if d.is_dir()]
        search_dir = subdirs[0] if subdirs else pred_dir
    elif (output_dir / "predictions").exists():
        pred_dir = output_dir / "predictions"
        subdirs = [d for d in pred_dir.iterdir() if d.is_dir()]
        search_dir = subdirs[0] if subdirs else pred_dir

    confidence_files = sorted(search_dir.glob("confidence_*_model_0.json"))
    if not confidence_files:
        confidence_files = sorted(search_dir.glob("confidence_*.json"))
    if confidence_files:
        with open(confidence_files[0]) as f:
            result["confidence"] = json.load(f)

    for ext in ("*.cif", "*.mmcif", "*.pdb"):
        candidates = sorted(search_dir.rglob(ext))
        if candidates:
            result["structure"] = candidates[0].read_text()
            break
    else:
        result["structure"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a single local cofold prediction")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--output_json", type=str, required=True, help="Path to write output JSON")
    parser.add_argument("--backend", type=str, default="protenix", choices=["protenix", "boltz"])
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory (default: auto temp)")
    parser.add_argument("--keep_work_dir", action="store_true", help="Don't delete work directory after")
    parser.add_argument("--protenix_model", type=str, default="protenix_base_default_v1.0.0")
    parser.add_argument("--boltz_model", type=str, default="boltz2")
    parser.add_argument("--protenix_venv", type=str, default=None, help="Override PROTENIX_VENV path")
    parser.add_argument("--boltz_venv", type=str, default=None, help="Override BOLTZ_VENV path")
    args = parser.parse_args()

    global PROTENIX_VENV, BOLTZ_VENV
    if args.protenix_venv:
        PROTENIX_VENV = args.protenix_venv
    if args.boltz_venv:
        BOLTZ_VENV = args.boltz_venv

    with open(args.input_json) as f:
        inp = json.load(f)

    sample_id = inp["id"]
    sequence = inp["sequence"]
    smiles = inp.get("smiles")

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"cofold_{sample_id}_"))
        cleanup = not args.keep_work_dir

    output = {"id": sample_id, "confidence": {}, "structure": None, "error": None}

    try:
        if args.backend == "protenix":
            result = run_protenix(sequence, smiles, work_dir, model_name=args.protenix_model)
        else:
            result = run_boltz(sequence, smiles, work_dir, model=args.boltz_model)

        output["confidence"] = result.get("confidence", {})
        output["structure"] = result.get("structure")
        logger.info("Success for %s: confidence keys = %s", sample_id, list(output["confidence"].keys()))

    except Exception as e:
        output["error"] = str(e)
        logger.error("Failed for %s: %s", sample_id, e)

    finally:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)

        if cleanup and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
