import torch
from loguru import logger as py_logger

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    ]

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1"," HE2",  None,  None,  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HD2"," HE1"," HE2"," HZ ",  None,  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HE2"," HD2"," HH ",  None,  None,  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
]

# writepdb
def writepdb(filename, atoms, seq, idx_pdb=None, bfacts=None):
    f = open(filename,"w")
    ctr = 1
    scpu = seq.cpu().squeeze()
    atomscpu = atoms.cpu().squeeze()
    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    for i,s in enumerate(scpu):
        if (len(atomscpu.shape)==2):
            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, " CA ", num2aa[s],
                    "A", idx_pdb[i], atomscpu[i,0], atomscpu[i,1], atomscpu[i,2],
                    1.0, Bfacts[i] ) )
            ctr += 1
        elif atomscpu.shape[1]==3:
            for j,atm_j in enumerate([" N  "," CA "," C  "]):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        "A", idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                ctr += 1
        else:
            natoms = atomscpu.shape[1]
            if (natoms!=14 and natoms!=27):
                print ('bad size!', atoms.shape)
                assert(False)
            atms = aa2long[s]
            # his prot hack
            if (s==8 and torch.linalg.norm( atomscpu[i,9,:]-atomscpu[i,5,:] ) < 1.7):
                atms = (
                    " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                      None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                    " HD1",  None,  None,  None,  None,  None,  None) # his_d

            for j,atm_j in enumerate(atms):
                if (j<natoms and atm_j is not None): # and not torch.isnan(atomscpu[i,j,:]).any()):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        "A", idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                    ctr += 1

def writepdb_ligand_complex(filename,
                     protein_atoms=None, protein_seq=None, protein_idx=None, protein_bfacts=None, protein_chain="A",
                     ligand_atoms=None, ligand_atom_names=None, ligand_idx=None, ligand_bfacts=None, ligand_chain="L",
                     ligand_resname="LIG"):
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

            for i, s in enumerate(scpu):
                if len(atomscpu.shape) == 2:
                    # Single atom per residue (CA only)
                    f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % (
                        "ATOM", atom_counter, " CA ", num2aa[s],
                        protein_chain, protein_idx[i], atomscpu[i, 0], atomscpu[i, 1], atomscpu[i, 2],
                        1.0, Bfacts[i]))
                    atom_counter += 1

                elif atomscpu.shape[1] == 3:
                    # Backbone atoms (N, CA, C)
                    for j, atm_j in enumerate([" N  ", " CA ", " C  "]):
                        f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % (
                            "ATOM", atom_counter, atm_j, num2aa[s],
                            protein_chain, protein_idx[i], atomscpu[i, j, 0], atomscpu[i, j, 1], atomscpu[i, j, 2],
                            1.0, Bfacts[i]))
                        atom_counter += 1

                else:
                    # Full atom representation
                    natoms = atomscpu.shape[1]
                    if natoms != 14 and natoms != 27:
                        print('Bad size!', atomscpu.shape)
                        assert(False)

                    atms = aa2long[s]
                    # His protonation state hack
                    if s == 8 and torch.linalg.norm(atomscpu[i, 9, :] - atomscpu[i, 5, :]) < 1.7:
                        atms = (
                            " N  ", " CA ", " C  ", " O  ", " CB ", " CG ", " NE2", " CD2", " CE1", " ND1",
                            None, None, None, None, " H  ", " HA ", "1HB ", "2HB ", " HD2", " HE1",
                            " HD1", None, None, None, None, None, None)  # his_d

                    for j, atm_j in enumerate(atms):
                        if j < natoms and atm_j is not None:  # and not torch.isnan(atomscpu[i, j, :]).any()):
                            f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % (
                                "ATOM", atom_counter, atm_j, num2aa[s],
                                protein_chain, protein_idx[i], atomscpu[i, j, 0], atomscpu[i, j, 1], atomscpu[i, j, 2],
                                1.0, Bfacts[i]))
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
                    atom_name = f" C{i+1} "
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
                atom_name = atom_names[i] if i < len(atom_names) else f" X{i+1} "

                # Format atom name to fit PDB standard (4 characters)
                if len(atom_name) < 4:
                    atom_name = atom_name.ljust(4)
                elif len(atom_name) > 4:
                    atom_name = atom_name[:4]

                # Get residue index
                res_idx = int(ligand_idx[i]) if isinstance(ligand_idx, torch.Tensor) else ligand_idx

                f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % (
                    "HETATM", atom_counter, atom_name, ligand_resname,
                    ligand_chain, res_idx, latoms[i, 0], latoms[i, 1], latoms[i, 2],
                    1.0, lBfacts[i]))
                atom_counter += 1

        # Write TER record to indicate end of chains
        f.write("TER\nEND\n")
