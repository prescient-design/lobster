import rdkit
from atomic_datasets import QM9
from rdkit import Chem
from rdkit.Chem import rdchem, rdFMCS
import selfies as sf
import numpy as np
import py3Dmol
import pickle
import os
from tqdm import tqdm
from atomic_datasets.utils.rdkit import is_molecule_sane
from typing import Optional

import itertools
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import polars as pl
#from rdkit.Chem.rdmolfiles import SDMolSupplier

import time
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
import pandas as pd

def is_valency_ok(mol):
    pt = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()

        # Current valence: from the atom itself (new API)
        try:
            current_valence = atom.GetTotalValence() #atom.GetValence(which=rdchem.AtomValence.ALL)
        except Exception as e:
            print(f"Warning: cannot compute valence for atom {atomic_num}: {e}")
            continue

        # Allowed max valence
        try:
            max_valence = max([v for v in pt.GetValenceList(atomic_num)]) #pt.GetValence(atomic_num, which=Chem.GetPeriodicTable().ALL)
        except Exception as e:
            print(f"Warning: cannot get max valence for atom {atomic_num}: {e}")
            continue

        if current_valence > max_valence:
            # get partial or formal charge. if current valence > max_valence + current charge, then it's not ok
            print(f'valence for atom {pt.GetElementSymbol(atomic_num)} is {current_valence} but max is {max_valence}')
            return False
    return True

def visualize_mol(mol):
    view = py3Dmol.view(data=Chem.MolToMolBlock(mol),style={'stick':{},'sphere':{'scale':0.3}})
    view.zoomTo()
    view.show()

def human_readable_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

# mol = dataset[0]['properties']['rdkit_mol']
# mol.GetNumConformers()
# np.linalg.norm(mol.GetConformer().GetPositions() - dataset[0]['nodes']['positions'])

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

def my_mols_to_3d_grid(mol_pairs, rows=2, cols=4, spacing=5.0, col_titles=None, width=800, height=600):
    """
    Show a grid of molecules with their 3D conformers using py3Dmol.
    Each molecule is translated to a unique grid location in 3D.

    Args:   
        mol_pairs: list of (mol_i, mol_j) tuples (RDKit Mol objects)
        rows: number of rows in the grid
        cols: number of columns in the grid
        spacing: spacing between molecules in the grid
        col_titles: list of strings for column titles (length should be >= cols)
        width: width of the view in pixels (default: 800)
        height: height of the view in pixels (default: 600)
    """
    view = py3Dmol.view(width=width, height=height)
    count = 0

    for row in range(rows):
        for col in range(cols):
            if count >= len(mol_pairs)*2:
                break
            mol = mol_pairs[col][row]
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol)

            # Translate molecule to grid position
            conf = mol.GetConformer()
            #dx, dy = j * spacing, -i * spacing

            # Check if molecule is centered (zero center of mass) and center it if needed
            positions = []
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                positions.append([pos.x, pos.y, pos.z])
            
            positions = np.array(positions)
            center_of_mass = np.mean(positions, axis=0)
            
            # Check if center of mass is close to zero (within tolerance)
            tolerance = 1e-6
            if np.any(np.abs(center_of_mass) > tolerance):
                
                # Center the molecule by subtracting the center of mass from all positions
                for k in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(k)
                    centered_pos = pos - Chem.rdGeometry.Point3D(*center_of_mass)
                    conf.SetAtomPosition(k, centered_pos)

            offset = np.array([col * spacing, -row * spacing, 0.0])
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                conf.SetAtomPosition(k, pos + Chem.rdGeometry.Point3D(*offset)) #(pos.x + dx, pos.y + dy, pos.z))

            # Add to viewer
            mb = Chem.MolToMolBlock(mol)
            view.addModel(mb, 'mol')
            view.setStyle({'model': count}, {'stick': {}, 'sphere': {'scale': 0.3}})
            count += 1

    # Add column titles as text labels if provided
    if col_titles and len(col_titles) >= cols:
        title_height = spacing * 0.8  # Height above the grid for titles (proportional to spacing)
        for col in range(cols):
            if col < len(col_titles):
                title = col_titles[col]
                # Position title above the center of each column
                title_x = col * spacing 
                title_y = title_height
                title_z = 0
                
                # Add text label
                view.addLabel(
                    title,
                    {
                        'position': {'x': title_x, 'y': title_y, 'z': title_z},
                        'fontSize': 16,
                        'fontColor': 'black',
                        'backgroundColor': 'white',
                        'borderThickness': 1,
                        'borderColor': 'black'
                    }
                )
    
    view.zoomTo()
    view.show()  # Auto-display the view
    return view

def visualize_mol_pairs_grid(
    mol_pairs,
    mols_per_row=2,
    legends=None,
    col_titles=None,
    size=(300, 300),
    title_fontsize=24
):
    """
    Visualizes a grid of 3D conformers from molecule pairs, with optional column titles.
    
    Args:
        mol_pairs: list of (mol_i, mol_j) tuples (RDKit Mol objects)
        mols_per_row: number of molecules per row (should be 2 for pairs)
        legends: list of string legends, length should be 2 * len(mol_pairs)
        col_titles: list of column titles, e.g., ["Mol A", "Mol B"]
        size: size of each sub-image (w, h)
        title_fontsize: font size for the column titles
    """
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import MolsToGridImage

    # Flatten mols
    mols = [mol for pair in mol_pairs for mol in pair]

    # Compute 2D coords if needed
    # for mol in mols:
    #     rdDepictor.Compute2DCoords(mol)

    # Generate placeholder legends if not provided
    if legends is None:
        legends = [f"Mol {i}" for i in range(len(mols))]

    # Generate main image
    img = MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=size,
        legends=legends,
        useMolBlock=False,
    )

    # Add column titles above image (optional)
    if col_titles:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", title_fontsize)
        except IOError:
            font = ImageFont.load_default()

        spacing = 10
        title_y = spacing
        for i, title in enumerate(col_titles):
            x = i * size[0] + size[0] // 2
            text_width, _ = draw.textsize(title, font=font)
            draw.text((x - text_width // 2, title_y), title, fill="black", font=font)

    return img

def display_mols_3d_grid(mol_pairs, rows=2, cols=4, spacing=5.0, col_titles=None, width=800, height=600):
    """
    Display a grid of molecules with their 3D conformers using py3Dmol.
    This function automatically shows the visualization.
    
    Args:   
        mol_pairs: list of (mol_i, mol_j) tuples (RDKit Mol objects)
        rows: number of rows in the grid
        cols: number of columns in the grid
        spacing: spacing between molecules in the grid
        col_titles: list of strings for column titles (length should be >= cols)
        width: width of the view in pixels (default: 800)
        height: height of the view in pixels (default: 600)
    """
    view = my_mols_to_3d_grid(mol_pairs, rows, cols, spacing, col_titles, width, height)
    view.show()
    return view

def get_shape_tanimoto(molA, molB, return_extra=True, verbose=True):
    
    start_time = time.time()

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrder
    mcs_result = rdFMCS.FindMCS([molA, molB], parameters=mcs_params)
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    # Find matching atoms
    matchA = molA.GetSubstructMatch(mcs_mol)
    matchB = molB.GetSubstructMatch(mcs_mol)
    matching_time = time.time() - start_time

    try:
        # Align molecules
        align_start = time.time()
        rmsd = AllChem.AlignMol(molB, molA, atomMap=list(zip(matchB, matchA)))
        align_time = time.time() - align_start

        # Calculate shape Tanimoto
        tani_start = time.time() 
        shape_tani_dist = ShapeTanimotoDist(molA, molB)
        tani_time = time.time() - tani_start
        
        if verbose:
            elapsed = time.time() - start_time
            if elapsed > 10:
                print(f"Mol A atoms: {molA.GetNumAtoms()} Mol B atoms: {molB.GetNumAtoms()} | Find matching atoms: {matching_time:.3f}s | Alignment: {align_time:.3f}s | Shape tani: {tani_time:.3f}s | Total: {time.time() - start_time:.3f}s")
                print(f" rmsd: {rmsd:.3f} | shape tani: {shape_tani_dist:.3f}")
            
        if return_extra:
            return shape_tani_dist, matchA, matchB, rmsd, matching_time
        else:
            return shape_tani_dist
    except Exception as e:
        if verbose:
            elapsed = time.time() - start_time
            print(f"    (threw exception) Elapsed time: {elapsed:.2f} seconds")
            
        if return_extra:
            return 100, [], []
        else:
            return 100


def visualize_mol_grid(mols, titles=None, num_rows=None, num_cols=None, spacing=5.0, width=800, height=600, title_height=None):
    """
    Visualize a grid of molecules using py3Dmol.
    
    Args:
        mols: list of RDKit Mol objects to visualize
        titles: optional list of strings for titles above each molecule
        num_rows: optional number of rows (auto-calculated if None)
        num_cols: optional number of columns (auto-calculated if None)
        spacing: spacing between molecules in the grid
        width: width of the view in pixels
        height: height of the view in pixels
        title_height: optional float, controls vertical space for titles/row spacing (default: spacing*0.8 if titles, else 0)
    
    Returns:
        py3Dmol view object
    """
    if not mols:
        raise ValueError("mols list cannot be empty")
    
    # Auto-calculate grid dimensions if not provided
    if num_rows is None and num_cols is None:
        n_mols = len(mols)
        num_cols = int(np.ceil(np.sqrt(n_mols)))
        num_rows = int(np.ceil(n_mols / num_cols))
        if num_rows > num_cols:
            num_cols, num_rows = num_rows, num_cols
    elif num_rows is None:
        num_rows = int(np.ceil(len(mols) / num_cols))
    elif num_cols is None:
        num_cols = int(np.ceil(len(mols) / num_rows))
    
    view = py3Dmol.view(width=width, height=height)
    count = 0
    
    # Calculate title spacing if titles are provided
    if title_height is not None:
        row_title_spacing = title_height if titles else 0
    else:
        row_title_spacing = spacing * 0.8 if titles else 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            mol_idx = row * num_cols + col
            if mol_idx >= len(mols):
                break
            mol = mols[mol_idx]
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol)
            conf = mol.GetConformer()
            positions = []
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                positions.append([pos.x, pos.y, pos.z])
            positions = np.array(positions)
            center_of_mass = np.mean(positions, axis=0)
            tolerance = 1e-6
            if np.any(np.abs(center_of_mass) > tolerance):
                for k in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(k)
                    centered_pos = pos - Chem.rdGeometry.Point3D(*center_of_mass)
                    conf.SetAtomPosition(k, centered_pos)
            # Position molecule in grid
            offset = np.array([col * spacing, -(row * spacing + row_title_spacing), 0.0])
            for k in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(k)
                conf.SetAtomPosition(k, pos + Chem.rdGeometry.Point3D(*offset))
            mb = Chem.MolToMolBlock(mol)
            view.addModel(mb, 'mol')
            view.setStyle({'model': count}, {'stick': {}, 'sphere': {'scale': 0.3}})
            count += 1
    # Add titles if provided
    if titles:
        for row in range(num_rows):
            for col in range(num_cols):
                mol_idx = row * num_cols + col
                if mol_idx >= len(mols):
                    break
                if mol_idx < len(titles) and titles[mol_idx]:
                    title = titles[mol_idx]
                    title_x = col * spacing
                    title_y = -(row * spacing + row_title_spacing * 0.5)  # Position above molecule
                    view.addLabel(
                        title,
                        {
                            'position': {'x': title_x, 'y': title_y, 'z': 0},
                            'fontSize': 14,
                            'fontColor': 'black',
                            'backgroundColor': 'white',
                            'borderThickness': 1,
                            'borderColor': 'black'
                        }
                    )
    view.zoomTo()
    view.show()
    return view

def get_tanimoto_distance(smiles1: str, smiles2: str, verbose: bool = True) -> float:
    """
    Compute Tanimoto distance between two SMILES strings using Morgan fingerprints.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        float: Tanimoto distance (1.0 - similarity), where 0.0 = identical, 1.0 = completely different
    """
    try:
    
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            if verbose:
                print(f"Invalid molecule: {smiles1} or {smiles2}")
            return 1.0  # Maximum distance for invalid molecules
            
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = gen.GetFingerprint(mol1)
        fp2 = gen.GetFingerprint(mol2)
        
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return 1.0 - similarity  # Convert to distance
        
    except Exception:
        if verbose:
            print(f"Error computing Tanimoto distance: {e}")
        return 1.0  # Maximum distance on error


def get_shape_tanimoto_distance(smiles1: str, smiles2: str, verbose: bool = True) -> float:
    """
    Compute shape Tanimoto distance between two SMILES strings.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        verbose: Whether to print verbose output
        
    Returns:
        float: Shape Tanimoto distance (lower = more similar shapes)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 100.0  # Maximum distance for invalid molecules
            
        # Generate 3D conformers if needed
        if mol1.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol1)
        if mol2.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol2)
            
        # Use the existing get_shape_tanimoto function
        shape_dist = get_shape_tanimoto(mol1, mol2, return_extra=False, verbose=verbose)
        return shape_dist
        
    except Exception as e:
        if verbose:
            print(f"Error computing shape Tanimoto distance: {e}")
        return 100.0  # Maximum distance on error

def filter_top_pairs_per_molecule(df: pd.DataFrame, property_key: Optional[str] = 'shape_tanimoto_distance', percentile: Optional[float] = None, num_pairs: Optional[int] = None) -> pd.DataFrame:
    """
    For each unique molecule_a_idx, keep only the top pairs with the lowest shape_tanimoto_distance.
    Can filter by either percentile or concrete number of pairs per molecule.

    Args:
        df: DataFrame with columns ['molecule_a_idx', 'molecule_b_idx', 'shape_tanimoto_distance', ...]
        percentile: float in (0, 100), e.g., 5.0 for top 5% (closest pairs). If provided, num_pairs is ignored.
        num_pairs: int > 0, e.g., 10 for top 10 closest pairs per molecule. Used only if percentile is None.

    Returns:
        Filtered DataFrame with only the top pairs per molecule_a_idx.
    """
    if percentile is not None and num_pairs is not None:
        raise ValueError("Cannot specify both percentile and num_pairs. Use one or the other.")
    
    if percentile is None and num_pairs is None:
        raise ValueError("Must specify either percentile or num_pairs.")
    
    if percentile is not None:
        if percentile <= 0 or percentile > 100:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
        
        def filter_group_percentile(group):
            cutoff = group[property_key].quantile(percentile / 100.0)
            return group[group[property_key] <= cutoff]
        
        filtered = df.groupby('molecule_a_idx', group_keys=False).apply(filter_group_percentile)
    
    else:  # num_pairs is not None
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")
        
        def filter_group_num_pairs(group):
            # Sort by property_key, e.g. shape_tanimoto_distance, and take top num_pairs
            return group.nsmallest(num_pairs, property_key)
        
        filtered = df.groupby('molecule_a_idx', group_keys=False).apply(filter_group_num_pairs)
    
    return filtered.reset_index(drop=True)