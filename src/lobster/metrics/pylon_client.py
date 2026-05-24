"""HTTP client for Pylon-deployed structure prediction and inverse folding endpoints.

Provides lightweight wrappers around the Pylon REST API for:
- Protenix (protein-ligand co-folding)
- LigandMPNN (inverse folding / sequence design)
- ESMFold (single-sequence structure prediction)

Endpoints are deployed at https://pylon.dev.lightship.gene.com/<service>/predict
"""

import logging
import os
import time
import urllib3
from typing import Any

import requests
import torch
from torch import Tensor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

PYLON_BASE_URL = "https://pylon.dev.lightship.gene.com"
DEFAULT_TIMEOUT = 600  # 10 minutes
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds


def _post_with_retry(
    url: str,
    payload: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """POST to a Pylon endpoint with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
                verify=False,
            )
            if resp.status_code >= 400:
                logger.error(
                    "Pylon %s returned %d: %s",
                    url, resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                logger.warning(
                    "Pylon request to %s failed (attempt %d/%d): %s. Retrying in %ds...",
                    url, attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Pylon request to %s failed after %d attempts: %s", url, max_retries, e)
    raise RuntimeError(f"Pylon request failed after {max_retries} attempts: {last_error}")


def call_protenix(
    sequence: str,
    ligand_smiles: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call Protenix co-folding endpoint.

    Parameters
    ----------
    sequence : str
        Protein amino acid sequence.
    ligand_smiles : str, optional
        SMILES string for the ligand. If provided, runs protein-ligand co-folding.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict with keys:
        - confidence: dict with ptm, iptm, plddt, ranking_score, chain_pair_iptm, has_clash
        - structure: str (mmCIF content)
    """
    sequences = [{"proteinChain": {"sequence": sequence, "count": 1}}]
    if ligand_smiles is not None:
        sequences.append({"ligand": {"ligand": ligand_smiles, "count": 1}})

    record = [
        {
            "name": "prediction",
            "sequences": sequences,
            "covalent_bonds": [],
        }
    ]

    url = f"{PYLON_BASE_URL}/protenix/predict"
    payload = {"records": [record], "tasks": None}
    data = _post_with_retry(url, payload, timeout=timeout)

    records = data.get("records", [])
    if not records:
        raise RuntimeError("Protenix returned no records")
    return records[0]


def call_boltz(
    sequence: str,
    ligand_smiles: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call Boltz-2 co-folding endpoint.

    Parameters
    ----------
    sequence : str
        Protein amino acid sequence.
    ligand_smiles : str, optional
        SMILES string for the ligand. If provided, runs protein-ligand co-folding.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict with keys:
        - confidence: dict with ptm, iptm, ligand_iptm, protein_iptm,
          complex_plddt, complex_iplddt, confidence_score, etc.
        - structure: str (mmCIF content)
    """
    sequences: list[dict[str, Any]] = [
        {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}}
    ]
    if ligand_smiles is not None:
        sequences.append({"ligand": {"id": "B", "smiles": ligand_smiles}})

    record = {"version": 1, "sequences": sequences}

    url = f"{PYLON_BASE_URL}/boltz-str-prd/predict"
    payload = {"records": [record], "tasks": None}
    data = _post_with_retry(url, payload, timeout=timeout)

    records = data.get("records", [])
    if not records:
        raise RuntimeError("Boltz returned no records")
    return records[0]


def call_cofold(
    sequence: str,
    ligand_smiles: str | None = None,
    backend: str = "protenix",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Unified co-folding dispatch: calls Protenix or Boltz.

    Parameters
    ----------
    sequence : str
        Protein amino acid sequence.
    ligand_smiles : str, optional
        SMILES string for the ligand.
    backend : str
        One of "protenix" or "boltz".
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict with keys: confidence (dict), structure (str).
    Confidence keys vary by backend:
        - Protenix: ptm, iptm, plddt, ranking_score, chain_pair_iptm, has_clash
        - Boltz: ptm, iptm, ligand_iptm, protein_iptm, complex_plddt, confidence_score, ...
    """
    if backend == "protenix":
        return call_protenix(sequence, ligand_smiles=ligand_smiles, timeout=timeout)
    elif backend == "boltz":
        return call_boltz(sequence, ligand_smiles=ligand_smiles, timeout=timeout)
    else:
        raise ValueError(f"Unknown co-folding backend: {backend!r}. Use 'protenix' or 'boltz'.")


def call_ligandmpnn(
    structure: str,
    batch_size: int = 10,
    number_of_batches: int = 1,
    temperature: float = 0.1,
    seed: int = 42,
    model_type: str = "ligand_mpnn",
    **kwargs: Any,
) -> list[str]:
    """Call LigandMPNN inverse folding endpoint.

    Parameters
    ----------
    structure : str
        PDB/mmCIF file content (protein-ligand complex).
    batch_size : int
        Number of sequences per batch.
    number_of_batches : int
        Number of batches.
    temperature : float
        Sampling temperature.
    seed : int
        Random seed.
    model_type : str
        LigandMPNN model type.

    Returns
    -------
    list[str]
        Designed protein sequences.
    """
    record: dict[str, Any] = {
        "structure": structure,
        "batch_size": batch_size,
        "number_of_batches": number_of_batches,
        "temperature": temperature,
        "seed": seed,
        "model_type": model_type,
    }
    record.update(kwargs)

    url = f"{PYLON_BASE_URL}/ligandmpnn/predict"
    payload = {"records": [record], "tasks": None}
    data = _post_with_retry(url, payload)

    records = data.get("records", [])
    if not records:
        raise RuntimeError("LigandMPNN returned no records")
    return records[0].get("sequences", [])


LIGANDMPNN_DEFAULT_PATH = "/cv/home/lisanzas/LigandMPNN"


def call_ligandmpnn_local(
    structure: str,
    batch_size: int = 10,
    number_of_batches: int = 1,
    temperature: float = 0.1,
    seed: int = 42,
    model_type: str = "ligand_mpnn",
    ligandmpnn_path: str = LIGANDMPNN_DEFAULT_PATH,
) -> list[str]:
    """Run LigandMPNN locally via subprocess.

    Same interface as call_ligandmpnn but runs the model locally instead of
    calling a Pylon endpoint. Requires the LigandMPNN repo with model weights
    at ``ligandmpnn_path``.

    Parameters
    ----------
    structure : str
        PDB file content (protein-ligand complex).
    batch_size : int
        Number of sequences per batch.
    number_of_batches : int
        Number of batches.
    temperature : float
        Sampling temperature.
    seed : int
        Random seed.
    model_type : str
        LigandMPNN model type (``ligand_mpnn``, ``protein_mpnn``, etc.).
    ligandmpnn_path : str
        Path to the LigandMPNN repo root (must contain ``run.py`` and
        ``model_params/``).

    Returns
    -------
    list[str]
        Designed protein sequences. Index 0 is the wild-type sequence,
        indices 1..N are the designs (matching the Pylon endpoint convention).
    """
    import shutil
    import subprocess
    import tempfile

    run_script = os.path.join(ligandmpnn_path, "run.py")
    if not os.path.exists(run_script):
        raise FileNotFoundError(f"LigandMPNN run.py not found at {run_script}")

    work_dir = tempfile.mkdtemp(prefix="ligandmpnn_")
    try:
        pdb_path = os.path.join(work_dir, "input.pdb")
        with open(pdb_path, "w") as f:
            f.write(structure)

        out_folder = os.path.join(work_dir, "output")
        os.makedirs(out_folder, exist_ok=True)

        cmd = [
            "python", run_script,
            "--model_type", model_type,
            "--pdb_path", pdb_path,
            "--out_folder", out_folder,
            "--batch_size", str(batch_size),
            "--number_of_batches", str(number_of_batches),
            "--temperature", str(temperature),
            "--seed", str(seed),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = ligandmpnn_path + ":" + env.get("PYTHONPATH", "")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env,
            cwd=ligandmpnn_path,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LigandMPNN failed (rc={result.returncode}): {result.stderr[-2000:]}"
            )

        return _parse_ligandmpnn_fasta(out_folder)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _parse_ligandmpnn_fasta(out_folder: str) -> list[str]:
    """Parse designed sequences from LigandMPNN FASTA output.

    LigandMPNN writes ``<out_folder>/seqs/<name>.fa`` with entries:
      >name, id=0, ... (wild-type)
      SEQUENCE
      >name, id=1, ... (design 1)
      SEQUENCE
      ...

    Returns list[str] with WT at index 0 followed by designs.
    """
    import glob as glob_mod

    seqs_dir = os.path.join(out_folder, "seqs")
    fasta_files = sorted(glob_mod.glob(os.path.join(seqs_dir, "*.fa")))
    if not fasta_files:
        raise RuntimeError(f"No FASTA files found in {seqs_dir}")

    sequences: list[str] = []
    with open(fasta_files[0]) as f:
        current_seq_lines: list[str] = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_seq_lines:
                    sequences.append("".join(current_seq_lines))
                    current_seq_lines = []
            else:
                current_seq_lines.append(line)
        if current_seq_lines:
            sequences.append("".join(current_seq_lines))

    if not sequences:
        raise RuntimeError(f"No sequences parsed from {fasta_files[0]}")

    return sequences


def call_esmfold(
    sequence: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call ESMFold structure prediction endpoint.

    Parameters
    ----------
    sequence : str
        Protein amino acid sequence.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict with keys:
        - confidence: dict with plddt, ptm
        - structure: str (PDB content)
    """
    record = {"sequence": sequence}
    url = f"{PYLON_BASE_URL}/esmfold/predict"
    payload = {"records": [record], "tasks": None}
    data = _post_with_retry(url, payload, timeout=timeout)

    records = data.get("records", [])
    if not records:
        raise RuntimeError("ESMFold returned no records")
    return records[0]


def parse_pdb_to_backbone_coords(pdb_text: str) -> Tensor:
    """Parse PDB text to backbone (N, CA, C) coordinates.

    Returns
    -------
    Tensor of shape [L, 3, 3] where dim 1 is (N, CA, C) and dim 2 is (x, y, z).
    """
    atom_coords: dict[int, dict[str, list[float]]] = {}

    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue
        chain_id = line[21]  # noqa: F841 -- keep for future multi-chain support
        res_seq = int(line[22:26].strip())
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        if res_seq not in atom_coords:
            atom_coords[res_seq] = {}
        atom_coords[res_seq][atom_name] = [x, y, z]

    backbone_order = ["N", "CA", "C"]
    coords = []
    for res_seq in sorted(atom_coords.keys()):
        res_atoms = atom_coords[res_seq]
        if all(a in res_atoms for a in backbone_order):
            coords.append([res_atoms[a] for a in backbone_order])

    if not coords:
        raise ValueError("No backbone atoms found in PDB text")
    return torch.tensor(coords, dtype=torch.float32)


def parse_mmcif_to_backbone_coords(mmcif_text: str) -> Tensor:
    """Parse mmCIF text to backbone (N, CA, C) coordinates.

    Handles Protenix output format where _atom_site columns are used.

    Returns
    -------
    Tensor of shape [L, 3, 3] where dim 1 is (N, CA, C) and dim 2 is (x, y, z).
    """
    try:
        from gemmi import cif
    except ImportError:
        return _parse_mmcif_simple(mmcif_text)

    doc = cif.read_string(mmcif_text)
    block = doc.sole_block()
    atom_site = block.find(["_atom_site."], [
        "label_atom_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
        "label_asym_id", "group_PDB",
    ])

    atom_coords: dict[int, dict[str, list[float]]] = {}
    for row in atom_site:
        group = row[6] if len(row) > 6 else "ATOM"
        if group != "ATOM":
            continue
        atom_name = row[0]
        if atom_name not in ("N", "CA", "C"):
            continue
        res_seq = int(row[1])
        x, y, z = float(row[2]), float(row[3]), float(row[4])
        if res_seq not in atom_coords:
            atom_coords[res_seq] = {}
        atom_coords[res_seq][atom_name] = [x, y, z]

    backbone_order = ["N", "CA", "C"]
    coords = []
    for res_seq in sorted(atom_coords.keys()):
        res_atoms = atom_coords[res_seq]
        if all(a in res_atoms for a in backbone_order):
            coords.append([res_atoms[a] for a in backbone_order])

    if not coords:
        raise ValueError("No backbone atoms found in mmCIF text")
    return torch.tensor(coords, dtype=torch.float32)


def _parse_mmcif_simple(mmcif_text: str) -> Tensor:
    """Fallback mmCIF parser without gemmi -- handles Protenix _atom_site loop."""
    in_atom_site = False
    col_names: list[str] = []
    rows: list[list[str]] = []

    for line in mmcif_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            in_atom_site = True
            col_names.append(stripped.split(".")[1])
            continue
        if in_atom_site:
            if stripped.startswith("_") or stripped.startswith("#") or stripped == "":
                if rows:
                    break
                continue
            if stripped.startswith("loop_"):
                if rows:
                    break
                continue
            rows.append(stripped.split())

    if not col_names or not rows:
        raise ValueError("Could not parse _atom_site loop from mmCIF")

    col_idx = {name: i for i, name in enumerate(col_names)}
    atom_id_col = col_idx.get("label_atom_id", col_idx.get("auth_atom_id"))
    seq_id_col = col_idx.get("label_seq_id", col_idx.get("auth_seq_id"))
    x_col = col_idx["Cartn_x"]
    y_col = col_idx["Cartn_y"]
    z_col = col_idx["Cartn_z"]
    group_col = col_idx.get("group_PDB")

    atom_coords: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        if group_col is not None and row[group_col] != "ATOM":
            continue
        atom_name = row[atom_id_col]
        if atom_name not in ("N", "CA", "C"):
            continue
        res_seq = int(row[seq_id_col])
        x, y, z = float(row[x_col]), float(row[y_col]), float(row[z_col])
        if res_seq not in atom_coords:
            atom_coords[res_seq] = {}
        atom_coords[res_seq][atom_name] = [x, y, z]

    backbone_order = ["N", "CA", "C"]
    coords = []
    for res_seq in sorted(atom_coords.keys()):
        res_atoms = atom_coords[res_seq]
        if all(a in res_atoms for a in backbone_order):
            coords.append([res_atoms[a] for a in backbone_order])

    if not coords:
        raise ValueError("No backbone atoms found in mmCIF text")
    return torch.tensor(coords, dtype=torch.float32)


def parse_mmcif_ligand_coords(mmcif_text: str) -> Tensor:
    """Parse ligand (HETATM) coordinates from mmCIF text.

    Returns
    -------
    Tensor of shape [N_atoms, 3] with ligand heavy atom coordinates.
    """
    in_atom_site = False
    col_names: list[str] = []
    rows: list[list[str]] = []

    for line in mmcif_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            in_atom_site = True
            col_names.append(stripped.split(".")[1])
            continue
        if in_atom_site:
            if stripped.startswith("_") or stripped.startswith("#") or stripped == "":
                if rows:
                    break
                continue
            if stripped.startswith("loop_"):
                if rows:
                    break
                continue
            rows.append(stripped.split())

    if not col_names or not rows:
        raise ValueError("Could not parse _atom_site loop from mmCIF")

    col_idx = {name: i for i, name in enumerate(col_names)}
    x_col = col_idx["Cartn_x"]
    y_col = col_idx["Cartn_y"]
    z_col = col_idx["Cartn_z"]
    group_col = col_idx.get("group_PDB")
    element_col = col_idx.get("type_symbol")

    coords = []
    for row in rows:
        if group_col is not None and row[group_col] != "HETATM":
            continue
        if element_col is not None and row[element_col] == "H":
            continue
        x, y, z = float(row[x_col]), float(row[y_col]), float(row[z_col])
        coords.append([x, y, z])

    if not coords:
        raise ValueError("No ligand (HETATM) atoms found in mmCIF text")
    return torch.tensor(coords, dtype=torch.float32)


def parse_structure_to_coords(structure_text: str) -> Tensor:
    """Parse a structure (PDB or mmCIF) to backbone coordinates.

    Auto-detects format based on content.

    Returns
    -------
    Tensor of shape [L, 3, 3] where dim 1 is (N, CA, C).
    """
    if structure_text.strip().startswith("data_"):
        return parse_mmcif_to_backbone_coords(structure_text)
    return parse_pdb_to_backbone_coords(structure_text)


def ligand_data_to_smiles(
    atom_names: list[str],
    bond_matrix: Tensor,
    coords: Tensor | None = None,
) -> str:
    """Convert ligand atom names + bond matrix to SMILES using RDKit.

    Parameters
    ----------
    atom_names : list[str]
        Element symbols or atom names (e.g. ['C', 'N', 'O', 'S']).
    bond_matrix : Tensor
        [N, N] integer bond matrix (1=single, 2=double, 3=triple, 4=aromatic).
    coords : Tensor, optional
        [N, 3] atom coordinates. If provided, sets 3D conformer.

    Returns
    -------
    str
        SMILES string.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    BOND_TYPE_MAP = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    mol = Chem.RWMol()
    for name in atom_names:
        if len(name) >= 2 and name[:2] in ("Cl", "Br", "Si", "Se", "Fe", "Zn", "Mg", "Ca", "Mn", "Cu", "Co", "Ni", "Bi"):
            elem = name[:2]
        else:
            elem = name[0].upper()
        atom = Chem.Atom(elem)
        mol.AddAtom(atom)

    bm = bond_matrix.cpu().numpy() if isinstance(bond_matrix, Tensor) else bond_matrix
    n_atoms = len(atom_names)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            bt = int(bm[i, j])
            if bt > 0:
                rd_bt = BOND_TYPE_MAP.get(bt, Chem.BondType.SINGLE)
                mol.AddBond(i, j, rd_bt)

    if coords is not None:
        conf = Chem.Conformer(n_atoms)
        c = coords.cpu().numpy() if isinstance(coords, Tensor) else coords
        for i in range(n_atoms):
            conf.SetAtomPosition(i, (float(c[i, 0]), float(c[i, 1]), float(c[i, 2])))
        mol.AddConformer(conf, assignId=True)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        logger.warning("RDKit sanitization failed; returning raw SMILES")

    return Chem.MolToSmiles(mol)


def ligand_sdf_to_smiles(sdf_path: str) -> str:
    """Read SMILES from an SDF file.

    Parameters
    ----------
    sdf_path : str
        Path to SDF file.

    Returns
    -------
    str
        SMILES string.
    """
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = next(iter(suppl), None)
    if mol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")
    return Chem.MolToSmiles(mol)
