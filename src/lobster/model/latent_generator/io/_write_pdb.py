import logging
import math

import torch

py_logger = logging.getLogger(__name__)

# Ideal geometry constants for backbone atoms
CA_CB_BOND = 1.521  # Å - standard CA-CB bond length
C_O_BOND = 1.231  # Å - carbonyl C=O bond length
N_CA_CB_ANGLE = math.radians(110.5)  # tetrahedral angle
CA_C_O_ANGLE = math.radians(120.5)  # sp2 carbonyl angle

# Glycine index in num2aa (GLY has no CB)
GLY_INDEX = 7


def _normalize(v):
    """Normalize a vector, handling zero-length vectors."""
    norm = torch.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def calculate_idealized_cb(n_pos, ca_pos, c_pos):
    """Calculate CB position using tetrahedral geometry.

    Places CB in the standard L-amino acid position using the
    tetrahedral geometry around the CA atom.

    Args:
        n_pos: N atom coordinates (torch.Tensor, shape [3])
        ca_pos: CA atom coordinates (torch.Tensor, shape [3])
        c_pos: C atom coordinates (torch.Tensor, shape [3])

    Returns:
        CB position as torch.Tensor of shape [3]
    """
    # Vectors from CA to N and C
    n_vec = n_pos - ca_pos
    c_vec = c_pos - ca_pos

    # Normalize
    n_unit = _normalize(n_vec)
    c_unit = _normalize(c_vec)

    # Calculate the N-CA-C plane normal
    plane_normal = torch.linalg.cross(n_unit, c_unit)
    plane_normal_norm = torch.linalg.norm(plane_normal)

    if plane_normal_norm > 1e-6:
        plane_normal = plane_normal / plane_normal_norm
    else:
        # Fallback for collinear atoms
        plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=n_pos.dtype, device=n_pos.device)

    # CB direction: solve for position that makes correct angles with N and C
    # For tetrahedral geometry, CB should make ~110.5° with both N and C
    cos_target = math.cos(N_CA_CB_ANGLE)  # cos(110.5°) ≈ -0.35
    cos_ncc = torch.dot(n_unit, c_unit).item()  # cos of N-CA-C angle

    # From the constraint equations:
    # CB_dir · n_unit = cos(110.5°)
    # CB_dir · c_unit = cos(110.5°)
    # Solving: a = b = cos_target / (1 + cos_ncc)
    denom = 1 + cos_ncc
    if abs(denom) < 1e-6:
        denom = 1e-6
    a = cos_target / denom

    # c² = 1 - 2*a²*(1 + cos_ncc)
    c_sq = 1 - 2 * a * a * (1 + cos_ncc)
    if c_sq < 0:
        c_sq = 0.01  # Handle numerical issues

    # For L-amino acids, CB is on the positive side of the plane
    c_coeff = math.sqrt(c_sq)

    cb_dir = a * n_unit + a * c_unit + c_coeff * plane_normal
    cb_dir = _normalize(cb_dir)

    cb_pos = ca_pos + CA_CB_BOND * cb_dir
    return cb_pos


def calculate_idealized_o(ca_pos, c_pos, next_n_pos=None):
    """Calculate carbonyl O position.

    Places O in the peptide plane, trans to the next residue's N
    (if available) or using simple geometry.

    Args:
        ca_pos: CA atom coordinates (torch.Tensor, shape [3])
        c_pos: C atom coordinates (torch.Tensor, shape [3])
        next_n_pos: Next residue's N atom coordinates (optional)

    Returns:
        O position as torch.Tensor of shape [3]
    """
    c_to_ca = ca_pos - c_pos
    c_to_ca = _normalize(c_to_ca)

    if next_n_pos is not None:
        # O is roughly trans to N across the C-CA axis
        c_to_n = next_n_pos - c_pos
        c_to_n = _normalize(c_to_n)

        # Calculate plane normal
        plane_normal = torch.linalg.cross(c_to_ca, c_to_n)
        plane_normal_norm = torch.linalg.norm(plane_normal)

        if plane_normal_norm > 1e-6:
            plane_normal = plane_normal / plane_normal_norm

            # O direction is in the plane, roughly opposite to N
            # Use the CA-C-N angle to place O correctly
            # O should be at ~120° from both CA and N (sp2 geometry)
            o_dir = torch.linalg.cross(plane_normal, c_to_n)
            o_dir = _normalize(o_dir)

            # Blend to get correct angle
            # O is at ~120° from N, so mix -c_to_n and perpendicular
            o_dir = -c_to_n * 0.5 + o_dir * 0.866  # cos(120°), sin(120°)
            o_dir = _normalize(o_dir)
        else:
            # Fallback: place O perpendicular to CA direction
            o_dir = _get_perpendicular(c_to_ca)
    else:
        # No next N available (terminal residue)
        # Place O roughly perpendicular to CA-C bond
        o_dir = _get_perpendicular(c_to_ca)

    o_pos = c_pos + C_O_BOND * o_dir
    return o_pos


def _get_perpendicular(v):
    """Get a unit vector perpendicular to v."""
    # Choose reference axis that's not parallel to v
    if abs(v[2]) < 0.9:
        ref = torch.tensor([0.0, 0.0, 1.0], dtype=v.dtype, device=v.device)
    else:
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=v.dtype, device=v.device)

    perp = torch.linalg.cross(v, ref)
    return _normalize(perp)


num2aa = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "MAS",
]

# full sc atom representation (Nx14)
aa2long = [
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "3HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # ala
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " NE ",
        " CZ ",
        " NH1",
        " NH2",
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HD ",
        "2HD ",
        " HE ",
        "1HH1",
        "2HH1",
        "1HH2",
        "2HH2",
    ),  # arg
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " OD1",
        " ND2",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HD2",
        "2HD2",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # asn
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " OD1",
        " OD2",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # asp
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " SG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # cys
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " OE1",
        " NE2",
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HE2",
        "2HE2",
        None,
        None,
        None,
        None,
        None,
    ),  # gln
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " OE1",
        " OE2",
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # glu
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        "1HA ",
        "2HA ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # gly
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " ND1",
        " CD2",
        " CE1",
        " NE2",
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD2",
        " HE1",
        " HE2",
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # his
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG1",
        " CG2",
        " CD1",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        " HB ",
        "1HG2",
        "2HG2",
        "3HG2",
        "1HG1",
        "2HG1",
        "1HD1",
        "2HD1",
        "3HD1",
        None,
        None,
    ),  # ile
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HG ",
        "1HD1",
        "2HD1",
        "3HD1",
        "1HD2",
        "2HD2",
        "3HD2",
        None,
        None,
    ),  # leu
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " CE ",
        " NZ ",
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HD ",
        "2HD ",
        "1HE ",
        "2HE ",
        "1HZ ",
        "2HZ ",
        "3HZ ",
    ),  # lys
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " SD ",
        " CE ",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HE ",
        "2HE ",
        "3HE ",
        None,
        None,
        None,
        None,
    ),  # met
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        " CE1",
        " CE2",
        " CZ ",
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD1",
        " HD2",
        " HE1",
        " HE2",
        " HZ ",
        None,
        None,
        None,
        None,
    ),  # phe
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HD ",
        "2HD ",
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # pro
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " OG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HG ",
        " HA ",
        "1HB ",
        "2HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # ser
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " OG1",
        " CG2",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HG1",
        " HA ",
        " HB ",
        "1HG2",
        "2HG2",
        "3HG2",
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # thr
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        " NE1",
        " CE2",
        " CE3",
        " CZ2",
        " CZ3",
        " CH2",
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD1",
        " HE1",
        " HZ2",
        " HH2",
        " HZ3",
        " HE3",
        None,
        None,
        None,
    ),  # trp
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        " CE1",
        " CE2",
        " CZ ",
        " OH ",
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD1",
        " HE1",
        " HE2",
        " HD2",
        " HH ",
        None,
        None,
        None,
        None,
    ),  # tyr
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG1",
        " CG2",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        " HB ",
        "1HG1",
        "2HG1",
        "3HG1",
        "1HG2",
        "2HG2",
        "3HG2",
        None,
        None,
        None,
        None,
    ),  # val
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "3HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # unk
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "3HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # mask
]


# writepdb
def writepdb(filename, atoms, seq, idx_pdb=None, bfacts=None, add_cb_o=True, chains=None):
    """Write protein structure to a PDB file.

    Args:
        filename: Output PDB filename
        atoms: Tensor of atom coordinates. Shape can be:
            - [num_residues, 3] for CA-only
            - [num_residues, 3, 3] for backbone (N, CA, C)
            - [num_residues, 14, 3] or [num_residues, 27, 3] for full atoms
        seq: Tensor of residue type indices (into num2aa)
        idx_pdb: Optional tensor of residue numbers (default: 1-indexed sequential)
        bfacts: Optional tensor of B-factors (default: zeros)
        add_cb_o: If True and atoms has shape [N, 3, 3] (backbone only), add
            idealized O and CB atoms. CB is not added for glycine. Default: True
        chains: Optional per-residue chain identifiers (tensor or list, length
            num_residues). Distinct values are mapped to chain letters A, B, C, …
            in order of first appearance, and a TER record is written at each
            chain boundary. Default None -> everything on chain "A" (legacy).
    """
    f = open(filename, "w")
    ctr = 1
    scpu = seq.cpu().squeeze()
    atomscpu = atoms.cpu().squeeze()
    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp(bfacts.cpu(), 0, 1)
    num_residues = len(scpu)

    # Per-residue chain letters (map distinct chain ids -> A, B, C, ... by first appearance).
    if chains is not None:
        _ch = chains.cpu().squeeze().tolist() if hasattr(chains, "cpu") else list(chains)
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        _seen: dict = {}
        chain_letters = []
        for _c in _ch:
            if _c not in _seen:
                _seen[_c] = _letters[len(_seen) % 26]
            chain_letters.append(_seen[_c])
    else:
        chain_letters = ["A"] * num_residues

    for i, s in enumerate(scpu):
        if i > 0 and chain_letters[i] != chain_letters[i - 1]:
            f.write("TER\n")
        if len(atomscpu.shape) == 2:
            f.write(
                f"{'ATOM':<6}{ctr:>5} {'CA':>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {atomscpu[i, 0]:8.3f}{atomscpu[i, 1]:8.3f}{atomscpu[i, 2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
            )
            ctr += 1
        elif atomscpu.shape[1] == 3:
            if add_cb_o:
                # Write N, CA, C, O, CB (CB only for non-glycine)
                n_pos = atomscpu[i, 0]
                ca_pos = atomscpu[i, 1]
                c_pos = atomscpu[i, 2]

                # Write N
                f.write(
                    f"{'ATOM':<6}{ctr:>5} {' N  ':>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {n_pos[0]:8.3f}{n_pos[1]:8.3f}{n_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                )
                ctr += 1

                # Write CA
                f.write(
                    f"{'ATOM':<6}{ctr:>5} {' CA ':>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {ca_pos[0]:8.3f}{ca_pos[1]:8.3f}{ca_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                )
                ctr += 1

                # Write C
                f.write(
                    f"{'ATOM':<6}{ctr:>5} {' C  ':>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {c_pos[0]:8.3f}{c_pos[1]:8.3f}{c_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                )
                ctr += 1

                # Write O (carbonyl oxygen)
                next_n_pos = atomscpu[i + 1, 0] if i < num_residues - 1 else None
                o_pos = calculate_idealized_o(ca_pos, c_pos, next_n_pos)
                f.write(
                    f"{'ATOM':<6}{ctr:>5} {' O  ':>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {o_pos[0]:8.3f}{o_pos[1]:8.3f}{o_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                )
                ctr += 1

                # Write CB (skip for glycine)
                if s != GLY_INDEX:
                    cb_pos = calculate_idealized_cb(n_pos, ca_pos, c_pos)
                    f.write(
                        f"{'ATOM':<6}{ctr:>5} {' CB ':>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {cb_pos[0]:8.3f}{cb_pos[1]:8.3f}{cb_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                    )
                    ctr += 1
            else:
                # Original behavior: just N, CA, C
                for j, atm_j in enumerate([" N  ", " CA ", " C  "]):
                    f.write(
                        f"{'ATOM':<6}{ctr:>5} {atm_j:>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {atomscpu[i, j, 0]:8.3f}{atomscpu[i, j, 1]:8.3f}{atomscpu[i, j, 2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                    )
                    ctr += 1
        else:
            natoms = atomscpu.shape[1]
            if natoms != 14 and natoms != 27:
                print("bad size!", atoms.shape)
                raise AssertionError(f"Unexpected number of atoms: {natoms}, expected 14 or 27")
            atms = aa2long[s]
            # his prot hack
            if s == 8 and torch.linalg.norm(atomscpu[i, 9, :] - atomscpu[i, 5, :]) < 1.7:
                atms = (
                    " N  ",
                    " CA ",
                    " C  ",
                    " O  ",
                    " CB ",
                    " CG ",
                    " NE2",
                    " CD2",
                    " CE1",
                    " ND1",
                    None,
                    None,
                    None,
                    None,
                    " H  ",
                    " HA ",
                    "1HB ",
                    "2HB ",
                    " HD2",
                    " HE1",
                    " HD1",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )  # his_d

            for j, atm_j in enumerate(atms):
                if j < natoms and atm_j is not None:  # and not torch.isnan(atomscpu[i,j,:]).any()):
                    f.write(
                        f"{'ATOM':<6}{ctr:>5} {atm_j:>4} {num2aa[s]:>3} {chain_letters[i]}{idx_pdb[i]:>4}    {atomscpu[i, j, 0]:8.3f}{atomscpu[i, j, 1]:8.3f}{atomscpu[i, j, 2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                    )
                    ctr += 1


def writepdb_ligand_complex(
    filename,
    protein_atoms=None,
    protein_seq=None,
    protein_idx=None,
    protein_bfacts=None,
    protein_chain="A",
    ligand_atoms=None,
    ligand_atom_names=None,
    ligand_idx=None,
    ligand_bfacts=None,
    ligand_chain="L",
    ligand_resname="LIG",
    ligand_bond_matrix=None,
    add_cb_o=True,
):
    """Write protein and ligand atoms to a PDB file.

    Args:
        filename: Output PDB filename
        protein_atoms: Tensor of protein atom coordinates [num_residues, num_atoms_per_residue, 3]
        protein_seq: Tensor of protein residue types
        protein_idx: Optional tensor of protein residue indices (default: sequential)
        protein_bfacts: Optional tensor of protein B-factors (default: zeros)
        protein_chain: Chain ID for protein (default: "A")
        ligand_atoms: Tensor of ligand atom coordinates [num_atoms, 3]
        ligand_atom_names: List of ligand atom names (e.g. ["C1", "N2", "O3", ...])
        ligand_idx: Optional tensor of ligand residue indices (default: all atoms in residue 1)
        ligand_bfacts: Optional tensor of ligand B-factors (default: zeros)
        ligand_chain: Chain ID for ligand (default: "L")
        ligand_resname: Residue name for ligand atoms (default: "LIG")
        ligand_bond_matrix: Optional bond matrix [num_atoms, num_atoms] where non-zero values
            indicate bonds. Used to write CONECT records for proper bond visualization.
        add_cb_o: If True and protein_atoms has shape [N, 3, 3] (backbone only), add
            idealized O and CB atoms. CB is not added for glycine. Default: True

    """
    # Check if protein_atoms and ligand_atoms are provided
    if protein_atoms is None and ligand_atoms is None:
        raise ValueError("Either protein_atoms or ligand_atoms must be provided.")

    with open(filename, "w") as f:
        atom_counter = 1

        # Write protein atoms if provided
        if protein_atoms is not None and protein_seq is not None:
            scpu = protein_seq.cpu().squeeze()
            atomscpu = protein_atoms.cpu().squeeze()

            if protein_bfacts is None:
                protein_bfacts = torch.zeros(atomscpu.shape[0])
            if protein_idx is None:
                protein_idx = 1 + torch.arange(atomscpu.shape[0])

            Bfacts = torch.clamp(protein_bfacts.cpu(), 0, 1)
            num_residues = len(scpu)

            for i, s in enumerate(scpu):
                if len(atomscpu.shape) == 2:
                    # Single atom per residue (CA only)
                    f.write(
                        f"{'ATOM':<6}{atom_counter:>5} {' CA ':>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {atomscpu[i, 0]:8.3f}{atomscpu[i, 1]:8.3f}{atomscpu[i, 2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                    )
                    atom_counter += 1

                elif atomscpu.shape[1] == 3:
                    if add_cb_o:
                        # Write N, CA, C, O, CB (CB only for non-glycine)
                        n_pos = atomscpu[i, 0]
                        ca_pos = atomscpu[i, 1]
                        c_pos = atomscpu[i, 2]

                        # Write N
                        f.write(
                            f"{'ATOM':<6}{atom_counter:>5} {' N  ':>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {n_pos[0]:8.3f}{n_pos[1]:8.3f}{n_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                        )
                        atom_counter += 1

                        # Write CA
                        f.write(
                            f"{'ATOM':<6}{atom_counter:>5} {' CA ':>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {ca_pos[0]:8.3f}{ca_pos[1]:8.3f}{ca_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                        )
                        atom_counter += 1

                        # Write C
                        f.write(
                            f"{'ATOM':<6}{atom_counter:>5} {' C  ':>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {c_pos[0]:8.3f}{c_pos[1]:8.3f}{c_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                        )
                        atom_counter += 1

                        # Write O (carbonyl oxygen)
                        next_n_pos = atomscpu[i + 1, 0] if i < num_residues - 1 else None
                        o_pos = calculate_idealized_o(ca_pos, c_pos, next_n_pos)
                        f.write(
                            f"{'ATOM':<6}{atom_counter:>5} {' O  ':>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {o_pos[0]:8.3f}{o_pos[1]:8.3f}{o_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                        )
                        atom_counter += 1

                        # Write CB (skip for glycine)
                        if s != GLY_INDEX:
                            cb_pos = calculate_idealized_cb(n_pos, ca_pos, c_pos)
                            f.write(
                                f"{'ATOM':<6}{atom_counter:>5} {' CB ':>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {cb_pos[0]:8.3f}{cb_pos[1]:8.3f}{cb_pos[2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                            )
                            atom_counter += 1
                    else:
                        # Original behavior: just N, CA, C
                        for j, atm_j in enumerate([" N  ", " CA ", " C  "]):
                            f.write(
                                f"{'ATOM':<6}{atom_counter:>5} {atm_j:>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {atomscpu[i, j, 0]:8.3f}{atomscpu[i, j, 1]:8.3f}{atomscpu[i, j, 2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                            )
                            atom_counter += 1

                else:
                    # Full atom representation
                    natoms = atomscpu.shape[1]
                    if natoms != 14 and natoms != 27:
                        print("Bad size!", atomscpu.shape)
                        raise AssertionError(f"Unexpected number of atoms: {natoms}, expected 14 or 27")

                    atms = aa2long[s]
                    # His protonation state hack
                    if s == 8 and torch.linalg.norm(atomscpu[i, 9, :] - atomscpu[i, 5, :]) < 1.7:
                        atms = (
                            " N  ",
                            " CA ",
                            " C  ",
                            " O  ",
                            " CB ",
                            " CG ",
                            " NE2",
                            " CD2",
                            " CE1",
                            " ND1",
                            None,
                            None,
                            None,
                            None,
                            " H  ",
                            " HA ",
                            "1HB ",
                            "2HB ",
                            " HD2",
                            " HE1",
                            " HD1",
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )  # his_d

                    for j, atm_j in enumerate(atms):
                        if j < natoms and atm_j is not None:  # and not torch.isnan(atomscpu[i, j, :]).any()):
                            f.write(
                                f"{'ATOM':<6}{atom_counter:>5} {atm_j:>4} {num2aa[s]:>3} {protein_chain}{protein_idx[i]:>4}    {atomscpu[i, j, 0]:8.3f}{atomscpu[i, j, 1]:8.3f}{atomscpu[i, j, 2]:8.3f}{1.0:6.2f}{Bfacts[i]:6.2f}\n"
                            )
                            atom_counter += 1

        # Write ligand atoms if provided
        if ligand_atoms is not None:
            latoms = ligand_atoms.cpu().squeeze()

            # Ensure ligand_atoms has the right shape [num_atoms, 3]
            if len(latoms.shape) == 3 and latoms.shape[0] == 1:
                latoms = latoms.squeeze(0)  # Remove batch dimension

            if ligand_bfacts is None:
                ligand_bfacts = torch.zeros(latoms.shape[0])
            if ligand_idx is None:
                ligand_idx = torch.ones(latoms.shape[0], dtype=torch.int)  # All atoms in residue 1

            lBfacts = torch.clamp(ligand_bfacts.cpu(), 0, 1)

            # Generate generic atom names if not provided
            if ligand_atom_names is None:
                # Make all atoms carbon by default
                py_logger.warning("Ligand atom names not provided. Using default names and setting all to carbon.")
                atom_names = []
                num_atoms = latoms.shape[0]
                for i in range(num_atoms):
                    atom_name = f" C{i + 1} "
                    atom_names.append(atom_name)
            else:
                # Use provided atom names, ensuring they are formatted correctly for PDB
                atom_names = []
                for name in ligand_atom_names:
                    # Format atom name to 4 characters, right-justified if starts with a letter
                    if name[0].isalpha():
                        formatted_name = name.ljust(4)
                    else:
                        formatted_name = name.rjust(4)
                    atom_names.append(formatted_name)

            # Write ligand atoms
            for i in range(latoms.shape[0]):
                # Get atom name (ensure it's exactly 4 characters)
                atom_name = atom_names[i] if i < len(atom_names) else f" X{i + 1} "

                # Format atom name to fit PDB standard (4 characters)
                if len(atom_name) < 4:
                    atom_name = atom_name.ljust(4)
                elif len(atom_name) > 4:
                    atom_name = atom_name[:4]

                # Get residue index
                res_idx = int(ligand_idx[i]) if isinstance(ligand_idx, torch.Tensor) else ligand_idx

                f.write(
                    f"{'HETATM':<6}{atom_counter:>5} {atom_name:>4} {ligand_resname:>3} {ligand_chain}{res_idx:>4}    {latoms[i, 0]:8.3f}{latoms[i, 1]:8.3f}{latoms[i, 2]:8.3f}{1.0:6.2f}{lBfacts[i]:6.2f}\n"
                )
                atom_counter += 1

            # Write CONECT records for ligand bonds if bond matrix provided
            if ligand_bond_matrix is not None:
                bond_mat = (
                    ligand_bond_matrix.cpu().numpy()
                    if isinstance(ligand_bond_matrix, torch.Tensor)
                    else ligand_bond_matrix
                )
                # ligand_start_atom is the first atom serial number for ligand atoms
                ligand_start_atom = atom_counter - latoms.shape[0]

                for i in range(latoms.shape[0]):
                    # Find all atoms bonded to atom i
                    bonded_atoms = []
                    for j in range(latoms.shape[0]):
                        if i != j and bond_mat[i, j] > 0:
                            bonded_atoms.append(ligand_start_atom + j)

                    if bonded_atoms:
                        # Write CONECT record: atom serial number followed by bonded atoms
                        atom_serial = ligand_start_atom + i
                        # PDB CONECT format: up to 4 bonded atoms per line
                        conect_line = f"CONECT{atom_serial:5d}"
                        for bonded in bonded_atoms[:4]:
                            conect_line += f"{bonded:5d}"
                        f.write(conect_line + "\n")

                        # If more than 4 bonds, write continuation lines
                        if len(bonded_atoms) > 4:
                            for batch_start in range(4, len(bonded_atoms), 4):
                                batch = bonded_atoms[batch_start : batch_start + 4]
                                conect_line = f"CONECT{atom_serial:5d}"
                                for bonded in batch:
                                    conect_line += f"{bonded:5d}"
                                f.write(conect_line + "\n")

        # Write TER record to indicate end of chains
        f.write("TER\nEND\n")
