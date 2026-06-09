"""Open Babel ligand minimization utilities.

This module provides functions for energy minimization of ligand structures
using Open Babel force fields. It can be used as a post-processing step
after structure generation to improve ligand geometry.

Functions
---------
minimize_ligand_structure : Minimize ligand coordinates using force field optimization
get_ligand_energy : Calculate the potential energy of a ligand structure
"""

import logging

import torch

py_logger = logging.getLogger(__name__)


def minimize_ligand_structure(
    coords: torch.Tensor,
    atom_types: list[str],
    bond_matrix: torch.Tensor | None = None,
    steps: int = 500,
    force_field: str = "MMFF94",
    method: str = "cg",
    mode: str = "full",
) -> torch.Tensor:
    """Minimize ligand structure using Open Babel force field optimization.

    This function performs energy minimization on ligand coordinates to improve
    geometry (bond lengths, angles, torsions). It can use provided bond connectivity
    or infer bonds from atomic distances.

    Parameters
    ----------
    coords : torch.Tensor
        Ligand coordinates with shape (num_atoms, 3) or (batch, num_atoms, 3).
        Coordinates should be in Angstroms.
    atom_types : list[str]
        Element symbols for each atom (e.g., ["C", "N", "O", "C", ...]).
        Length must match num_atoms.
    bond_matrix : torch.Tensor, optional
        Bond connectivity matrix with shape (num_atoms, num_atoms).
        Values: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic.
        If None, bonds will be inferred from coordinates using Open Babel.
    steps : int, default=500
        Maximum number of minimization steps. Ignored if mode="bonds_only".
    force_field : str, default="MMFF94"
        Force field to use. Options: "MMFF94", "MMFF94s", "UFF", "GAFF", "Ghemical".
        - MMFF94: Merck Molecular Force Field (recommended for drug-like molecules)
        - MMFF94s: MMFF94 with modified torsion parameters for planar groups
        - UFF: Universal Force Field (good fallback, works for all elements)
        - GAFF: General AMBER Force Field (good for organic molecules)
        - Ghemical: Ghemical force field
    method : str, default="cg"
        Optimization method. Options: "cg" (conjugate gradients), "sd" (steepest descent).
        Conjugate gradients is generally faster and recommended.
    mode : str, default="full"
        Minimization mode:
        - "full": Full energy minimization (default, may change conformation)
        - "local": Short minimization (50 steps) to fix bond lengths/angles only
        - "bonds_only": Correct bond lengths to ideal values without minimization
        - "bonds_and_angles": Correct both bond lengths and angles to ideal values

    Returns
    -------
    torch.Tensor
        Minimized coordinates with same shape as input.

    Raises
    ------
    ImportError
        If openbabel is not installed.
    ValueError
        If force field setup fails or coordinates are invalid.

    Examples
    --------
    >>> coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.3, 1.2, 0.0]])
    >>> atom_types = ["C", "C", "O"]
    >>> minimized = minimize_ligand_structure(coords, atom_types, mode="local")

    Notes
    -----
    - If bond_matrix is not provided, Open Babel will infer bonds based on
      atomic distances and element types. This works well for most organic molecules.
    - The minimization preserves the overall molecular topology and only adjusts
      atomic positions to lower the potential energy.
    - For best results with drug-like molecules, use MMFF94 force field.
    - UFF is recommended as fallback since it supports all elements.
    - Use mode="local" or mode="bonds_only" to preserve overall conformation
      while fixing local geometry issues.
    """
    try:
        from openbabel import openbabel as ob
    except ImportError as e:
        raise ImportError(
            "Open Babel is required for ligand minimization. Install with: pip install openbabel-wheel"
        ) from e

    # Handle batch dimension
    had_batch_dim = coords.dim() == 3
    if had_batch_dim:
        if coords.shape[0] != 1:
            raise ValueError(f"Batch minimization not supported. Got batch size {coords.shape[0]}, expected 1.")
        coords = coords.squeeze(0)

    # Validate inputs
    num_atoms = coords.shape[0]
    if len(atom_types) != num_atoms:
        raise ValueError(f"Number of atom types ({len(atom_types)}) must match number of atoms ({num_atoms})")

    # Convert to numpy for Open Babel
    coords_np = coords.detach().cpu().numpy()

    # Create Open Babel molecule
    mol = ob.OBMol()

    # Add atoms
    for i, (coord, atom_type) in enumerate(zip(coords_np, atom_types)):
        atom = mol.NewAtom()
        # Handle element lookup - strip any numbers from atom names (e.g., "C1" -> "C")
        element = "".join(c for c in atom_type if c.isalpha())
        atomic_num = ob.GetAtomicNum(element)
        if atomic_num == 0:
            py_logger.warning(f"Unknown element '{element}', defaulting to Carbon")
            atomic_num = 6  # Default to Carbon
        atom.SetAtomicNum(atomic_num)
        atom.SetVector(float(coord[0]), float(coord[1]), float(coord[2]))

    # Add bonds from bond_matrix if provided, otherwise let Open Babel infer
    if bond_matrix is not None:
        bond_matrix_np = bond_matrix.detach().cpu().numpy()
        # Map our bond types to Open Babel bond orders
        bond_order_map = {
            1: 1,  # Single
            2: 2,  # Double
            3: 3,  # Triple
            4: 5,  # Aromatic (Open Babel uses 5 for aromatic)
            5: 1,  # Other -> Single
        }
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                bond_val = int(bond_matrix_np[i, j])
                if bond_val > 0:
                    order = bond_order_map.get(bond_val, 1)
                    mol.AddBond(i + 1, j + 1, order)  # OB uses 1-based indexing
    else:
        # Infer bonds from coordinates
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

    # Helper function to correct bond lengths
    def _correct_bond_lengths(molecule):
        """Correct bond lengths to ideal values."""
        for bond in ob.OBMolBondIter(molecule):
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            # Get ideal bond length from Open Babel's tables
            ideal_length = ob.GetCovalentRad(atom1.GetAtomicNum()) + ob.GetCovalentRad(atom2.GetAtomicNum())
            if bond.GetBondOrder() == 2:
                ideal_length *= 0.87  # Double bonds are ~13% shorter
            elif bond.GetBondOrder() == 3:
                ideal_length *= 0.78  # Triple bonds are ~22% shorter
            elif bond.IsAromatic():
                ideal_length *= 0.91  # Aromatic bonds are ~9% shorter

            # Get current bond vector
            v1 = atom1.GetVector()
            v2 = atom2.GetVector()
            current_length = v1.distSq(v2) ** 0.5

            if current_length > 0.01:  # Avoid division by zero
                # Scale factor to achieve ideal length
                scale = ideal_length / current_length
                # Move atoms toward/away from each other equally
                midpoint_x = (v1.GetX() + v2.GetX()) / 2
                midpoint_y = (v1.GetY() + v2.GetY()) / 2
                midpoint_z = (v1.GetZ() + v2.GetZ()) / 2

                # New positions scaled from midpoint
                new_x1 = midpoint_x + (v1.GetX() - midpoint_x) * scale
                new_y1 = midpoint_y + (v1.GetY() - midpoint_y) * scale
                new_z1 = midpoint_z + (v1.GetZ() - midpoint_z) * scale
                new_x2 = midpoint_x + (v2.GetX() - midpoint_x) * scale
                new_y2 = midpoint_y + (v2.GetY() - midpoint_y) * scale
                new_z2 = midpoint_z + (v2.GetZ() - midpoint_z) * scale

                atom1.SetVector(new_x1, new_y1, new_z1)
                atom2.SetVector(new_x2, new_y2, new_z2)

    # Helper function to get ideal bond angle based on hybridization
    def _get_ideal_angle(central_atom):
        """Get ideal bond angle for a central atom based on its hybridization."""
        hyb = central_atom.GetHyb()
        if hyb == 1:  # sp - linear
            return 180.0
        elif hyb == 2:  # sp2 - trigonal planar
            return 120.0
        elif hyb == 3:  # sp3 - tetrahedral
            return 109.47
        else:
            # Default to sp3 if unknown
            return 109.47

    # Helper function to correct bond angles
    def _correct_bond_angles(molecule, num_iterations=3):
        """Correct bond angles to ideal values based on hybridization."""
        import math

        for _ in range(num_iterations):
            for angle in ob.OBMolAngleIter(molecule):
                # angle is a tuple (vertex_idx, atom1_idx, atom2_idx) - 0-based
                vertex_idx, idx1, idx2 = angle

                # Get atoms (OBMol uses 1-based indexing)
                central_atom = molecule.GetAtom(vertex_idx + 1)
                atom1 = molecule.GetAtom(idx1 + 1)
                atom2 = molecule.GetAtom(idx2 + 1)

                if central_atom is None or atom1 is None or atom2 is None:
                    continue

                # Get ideal angle for this central atom
                ideal_angle = _get_ideal_angle(central_atom)
                ideal_rad = math.radians(ideal_angle)

                # Get current positions
                vc = central_atom.GetVector()
                v1 = atom1.GetVector()
                v2 = atom2.GetVector()

                # Calculate vectors from central atom
                vec1_x = v1.GetX() - vc.GetX()
                vec1_y = v1.GetY() - vc.GetY()
                vec1_z = v1.GetZ() - vc.GetZ()

                vec2_x = v2.GetX() - vc.GetX()
                vec2_y = v2.GetY() - vc.GetY()
                vec2_z = v2.GetZ() - vc.GetZ()

                # Calculate current angle
                len1 = math.sqrt(vec1_x**2 + vec1_y**2 + vec1_z**2)
                len2 = math.sqrt(vec2_x**2 + vec2_y**2 + vec2_z**2)

                if len1 < 0.01 or len2 < 0.01:
                    continue

                dot = vec1_x * vec2_x + vec1_y * vec2_y + vec1_z * vec2_z
                cos_angle = max(-1.0, min(1.0, dot / (len1 * len2)))
                current_rad = math.acos(cos_angle)

                # Calculate angle difference
                angle_diff = ideal_rad - current_rad

                # Skip if angle is already close to ideal (within 5 degrees)
                if abs(angle_diff) < math.radians(5.0):
                    continue

                # Calculate rotation axis (perpendicular to the plane of the angle)
                cross_x = vec1_y * vec2_z - vec1_z * vec2_y
                cross_y = vec1_z * vec2_x - vec1_x * vec2_z
                cross_z = vec1_x * vec2_y - vec1_y * vec2_x
                cross_len = math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

                if cross_len < 0.001:
                    continue  # Vectors are parallel, can't define rotation axis

                # Normalize rotation axis
                axis_x = cross_x / cross_len
                axis_y = cross_y / cross_len
                axis_z = cross_z / cross_len

                # Rotate atom2 around axis by half the angle difference
                # (and atom1 in opposite direction by half)
                half_diff = angle_diff / 2.0

                # Rodrigues rotation formula for atom2
                cos_rot = math.cos(half_diff)
                sin_rot = math.sin(half_diff)

                # Rotate vec2
                dot_axis_vec2 = axis_x * vec2_x + axis_y * vec2_y + axis_z * vec2_z
                cross2_x = axis_y * vec2_z - axis_z * vec2_y
                cross2_y = axis_z * vec2_x - axis_x * vec2_z
                cross2_z = axis_x * vec2_y - axis_y * vec2_x

                new_vec2_x = vec2_x * cos_rot + cross2_x * sin_rot + axis_x * dot_axis_vec2 * (1 - cos_rot)
                new_vec2_y = vec2_y * cos_rot + cross2_y * sin_rot + axis_y * dot_axis_vec2 * (1 - cos_rot)
                new_vec2_z = vec2_z * cos_rot + cross2_z * sin_rot + axis_z * dot_axis_vec2 * (1 - cos_rot)

                # Rotate vec1 in opposite direction
                cos_rot_neg = math.cos(-half_diff)
                sin_rot_neg = math.sin(-half_diff)

                dot_axis_vec1 = axis_x * vec1_x + axis_y * vec1_y + axis_z * vec1_z
                cross1_x = axis_y * vec1_z - axis_z * vec1_y
                cross1_y = axis_z * vec1_x - axis_x * vec1_z
                cross1_z = axis_x * vec1_y - axis_y * vec1_x

                new_vec1_x = vec1_x * cos_rot_neg + cross1_x * sin_rot_neg + axis_x * dot_axis_vec1 * (1 - cos_rot_neg)
                new_vec1_y = vec1_y * cos_rot_neg + cross1_y * sin_rot_neg + axis_y * dot_axis_vec1 * (1 - cos_rot_neg)
                new_vec1_z = vec1_z * cos_rot_neg + cross1_z * sin_rot_neg + axis_z * dot_axis_vec1 * (1 - cos_rot_neg)

                # Update positions
                atom1.SetVector(vc.GetX() + new_vec1_x, vc.GetY() + new_vec1_y, vc.GetZ() + new_vec1_z)
                atom2.SetVector(vc.GetX() + new_vec2_x, vc.GetY() + new_vec2_y, vc.GetZ() + new_vec2_z)

    # Handle bonds_only mode - correct bond lengths without energy minimization
    if mode == "bonds_only":
        builder = ob.OBBuilder()
        builder.CorrectStereoAtoms(mol)
        _correct_bond_lengths(mol)
    elif mode == "bonds_and_angles":
        # Use constrained force field minimization with ideal bond lengths and angles

        # Set up constraints for ideal geometry
        constraints = ob.OBFFConstraints()
        constraints.SetFactor(10000.0)  # High weight to enforce constraints

        # Add distance constraints for all bonds at ideal lengths
        for bond in ob.OBMolBondIter(mol):
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            # Calculate ideal bond length based on atom types and bond order
            ideal_length = ob.GetCovalentRad(atom1.GetAtomicNum()) + ob.GetCovalentRad(atom2.GetAtomicNum())
            if bond.GetBondOrder() == 2:
                ideal_length *= 0.87
            elif bond.GetBondOrder() == 3:
                ideal_length *= 0.78
            elif bond.IsAromatic():
                ideal_length *= 0.91

            constraints.AddDistanceConstraint(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), ideal_length)

        # Add angle constraints for all angles at ideal values based on hybridization
        for angle in ob.OBMolAngleIter(mol):
            vertex_idx, idx1, idx2 = angle
            central_atom = mol.GetAtom(vertex_idx + 1)

            # Determine ideal angle based on hybridization
            hyb = central_atom.GetHyb()
            if hyb == 1:  # sp - linear
                ideal_angle = 180.0
            elif hyb == 2:  # sp2 - trigonal planar
                ideal_angle = 120.0
            else:  # sp3 - tetrahedral (default)
                ideal_angle = 109.47

            # OBFFConstraints uses 1-based indexing
            constraints.AddAngleConstraint(idx1 + 1, vertex_idx + 1, idx2 + 1, ideal_angle)

        # Run constrained minimization
        ff = ob.OBForceField.FindForceField(force_field)
        if ff is None:
            ff = ob.OBForceField.FindForceField("UFF")

        if ff is not None and ff.Setup(mol, constraints):
            ff.ConjugateGradients(min(steps, 500))
            ff.GetCoordinates(mol)
        else:
            py_logger.warning("Constrained minimization failed, falling back to bonds_only")
            _correct_bond_lengths(mol)
    else:
        # Force field minimization modes
        # Determine actual steps based on mode
        actual_steps = steps
        if mode == "local":
            actual_steps = min(50, steps)  # Cap at 50 for local mode
        elif mode != "full":
            raise ValueError(f"Unknown mode: {mode}. Use 'full', 'local', 'bonds_only', or 'bonds_and_angles'.")

        # Set up force field
        ff = ob.OBForceField.FindForceField(force_field)
        if ff is None:
            # Try fallback to UFF
            py_logger.warning(f"Force field '{force_field}' not available, falling back to UFF")
            ff = ob.OBForceField.FindForceField("UFF")
            if ff is None:
                raise ValueError("No force field available for minimization")

        # Initialize force field with molecule
        if not ff.Setup(mol):
            py_logger.warning(f"Force field setup failed with {force_field}, trying UFF as fallback")
            ff = ob.OBForceField.FindForceField("UFF")
            if ff is None or not ff.Setup(mol):
                py_logger.warning("Force field setup failed, returning original coordinates")
                if had_batch_dim:
                    return coords.unsqueeze(0)
                return coords

        # Run minimization
        if method == "cg":
            ff.ConjugateGradients(actual_steps)
        elif method == "sd":
            ff.SteepestDescent(actual_steps)
        else:
            raise ValueError(f"Unknown optimization method: {method}. Use 'cg' or 'sd'.")

        # Update coordinates in molecule
        ff.GetCoordinates(mol)

    # Extract minimized coordinates
    minimized_coords = torch.zeros_like(coords)
    for i in range(num_atoms):
        atom = mol.GetAtom(i + 1)  # OB uses 1-based indexing
        minimized_coords[i, 0] = atom.GetX()
        minimized_coords[i, 1] = atom.GetY()
        minimized_coords[i, 2] = atom.GetZ()

    # Restore batch dimension if needed
    if had_batch_dim:
        minimized_coords = minimized_coords.unsqueeze(0)

    return minimized_coords


def get_ligand_energy(
    coords: torch.Tensor,
    atom_types: list[str],
    bond_matrix: torch.Tensor | None = None,
    force_field: str = "MMFF94",
) -> float:
    """Calculate the potential energy of a ligand structure.

    This function computes the force field energy of a ligand, which can be
    used to compare structures before and after minimization.

    Parameters
    ----------
    coords : torch.Tensor
        Ligand coordinates with shape (num_atoms, 3) or (batch, num_atoms, 3).
    atom_types : list[str]
        Element symbols for each atom.
    bond_matrix : torch.Tensor, optional
        Bond connectivity matrix. If None, bonds will be inferred.
    force_field : str, default="MMFF94"
        Force field to use for energy calculation.

    Returns
    -------
    float
        Potential energy in kcal/mol. Returns float('inf') if calculation fails.

    Examples
    --------
    >>> coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    >>> atom_types = ["C", "C"]
    >>> energy = get_ligand_energy(coords, atom_types)
    """
    try:
        from openbabel import openbabel as ob
    except ImportError:
        py_logger.warning("Open Babel not available for energy calculation")
        return float("inf")

    # Handle batch dimension
    if coords.dim() == 3:
        coords = coords.squeeze(0)

    num_atoms = coords.shape[0]
    coords_np = coords.detach().cpu().numpy()

    # Create molecule
    mol = ob.OBMol()
    for i, (coord, atom_type) in enumerate(zip(coords_np, atom_types)):
        atom = mol.NewAtom()
        element = "".join(c for c in atom_type if c.isalpha())
        atomic_num = ob.GetAtomicNum(element)
        if atomic_num == 0:
            atomic_num = 6
        atom.SetAtomicNum(atomic_num)
        atom.SetVector(float(coord[0]), float(coord[1]), float(coord[2]))

    # Add bonds
    if bond_matrix is not None:
        bond_matrix_np = bond_matrix.detach().cpu().numpy()
        bond_order_map = {1: 1, 2: 2, 3: 3, 4: 5, 5: 1}
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                bond_val = int(bond_matrix_np[i, j])
                if bond_val > 0:
                    order = bond_order_map.get(bond_val, 1)
                    mol.AddBond(i + 1, j + 1, order)
    else:
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

    # Calculate energy
    ff = ob.OBForceField.FindForceField(force_field)
    if ff is None:
        ff = ob.OBForceField.FindForceField("UFF")

    if ff is None or not ff.Setup(mol):
        return float("inf")

    return ff.Energy()
