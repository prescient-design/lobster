# code to modify input for training

import logging

import numpy as np
import torch
from rdkit import Chem

from lobster.model.latent_generator.utils import apply_random_se3_batched, residue_constants
from lobster.model.latent_generator.utils._kinematics import c6d_to_bins, xyz_to_c6d
from lobster.model.latent_generator.utils.mini3di import Encoder, calculate_cb

# Import ESM for embeddings
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
except ImportError:
    ESMC, ESMProtein, LogitsConfig = None, None, None
try:
    from torch_geometric.transforms import BaseTransform
except ImportError:
    BaseTransform = None

logger = logging.getLogger(__name__)


class ESMEmbeddingTransform(BaseTransform):
    """Transform that extracts ESM-C embeddings from sequences."""

    def __init__(self, model_name: str = "esmc_300m", device: str = "auto", **kwargs):
        """Initialize the ESM embedding transform.

        Args:
            model_name: Name of the ESM model to use
            device: Device to load the model on ("auto", "cuda", "cpu")
            **kwargs: Additional arguments
        """
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")
        lobster.ensure_package("esm", group="lg-gpu (or --extra lg-cpu)")

        super().__init__(**kwargs)

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading ESM model {model_name} on device {self.device}")

        # Load the ESM model
        self.model = ESMC.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode

        logger.info("ESM model loaded successfully")

    def sequence_to_esm_embeddings(self, sequence: torch.Tensor, chains: torch.Tensor = None) -> torch.Tensor:
        """Convert sequence tensor to ESM embeddings with glycine linker for multiple chains.

        Args:
            sequence: Tensor of shape (seq_len,) containing amino acid indices
            chains: Tensor of shape (seq_len,) containing chain IDs (optional)

        Returns:
            Tensor of shape (1, seq_len, embed_dim) containing ESM embeddings
        """
        # Convert indices to amino acid strings
        if sequence.dim() == 1:
            # Single sequence
            seq_indices = sequence.cpu().numpy()

            # Handle multiple chains with glycine linker
            if chains is not None:
                chains_array = chains.cpu().numpy()
                unique_chains = np.unique(chains_array)

                if len(unique_chains) > 1:
                    # Multiple chains detected - apply glycine linker logic

                    # Get the first two unique chains (or duplicate if only one)
                    if len(unique_chains) == 1:
                        unique_chains = [unique_chains[0], unique_chains[0]]
                        logger.warning("Only one chain found, using the same chain for both chains")

                    # Create boolean masks for each chain
                    chain_1_mask = chains_array == unique_chains[0]
                    chain_2_mask = chains_array == unique_chains[1]

                    # Get coordinates for each chain (assuming coords_res is available)
                    # For now, we'll work with sequence indices
                    chain_1_indices = seq_indices[chain_1_mask]
                    chain_2_indices = seq_indices[chain_2_mask]
                    index_to_add = chain_1_indices.shape[0]

                    # Add glycine linker (40 glycines)
                    glycine_idx = residue_constants.restype_order_with_x["G"]
                    glycine_linker = [glycine_idx] * 40

                    # Combine: chain1 + glycine_linker + chain2
                    combined_indices = np.concatenate([chain_1_indices, glycine_linker, chain_2_indices])

                    # Convert to amino acid sequence
                    aa_sequence = [residue_constants.restype_order_with_x_inv[aa] for aa in combined_indices]
                    aa_sequence = "".join(aa_sequence)

                    # Create ESM protein and get embeddings
                    protein = ESMProtein(sequence=aa_sequence)
                    protein_tensor = self.model.encode(protein)
                    embeddings = self.model.logits(
                        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                    ).embeddings

                    # Remove first and last token (CLS and EOS tokens)
                    embeddings = embeddings[:, 1:-1]

                    # Extract only the original sequence embeddings (remove glycine linker)
                    # Create masks for the combined sequence
                    combined_chain_1_mask = np.concatenate(
                        [chain_1_mask[:index_to_add], np.zeros(40, dtype=bool), chain_1_mask[index_to_add:]]
                    )
                    combined_chain_2_mask = np.concatenate(
                        [chain_2_mask[:index_to_add], np.zeros(40, dtype=bool), chain_2_mask[index_to_add:]]
                    )

                    # Extract embeddings for original positions
                    original_mask = combined_chain_1_mask | combined_chain_2_mask
                    embeddings = embeddings[0][original_mask][None]

                    if embeddings.shape[1] != len(sequence):
                        raise ValueError(
                            f"Embeddings shape {embeddings.shape} does not match sequence length {len(sequence)}"
                        )

                    return embeddings.to(sequence.device)

            # Single chain or no chain info - use original logic
            aa_sequence = [residue_constants.restype_order_with_x_inv[aa] for aa in seq_indices]
            aa_sequence = "".join(aa_sequence)

            # Create ESM protein and get embeddings
            protein = ESMProtein(sequence=aa_sequence)
            protein_tensor = self.model.encode(protein)
            embeddings = self.model.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            ).embeddings

            # Remove first and last token (CLS and EOS tokens)
            embeddings = embeddings[:, 1:-1]

            return embeddings.to(sequence.device)

        else:
            # Batch of sequences - handle each sequence individually
            batch_embeddings = []
            for i in range(sequence.shape[0]):
                seq_embeddings = self.sequence_to_esm_embeddings(sequence[i], chains[i] if chains is not None else None)
                batch_embeddings.append(seq_embeddings)

            return torch.cat(batch_embeddings, dim=0)

    def __call__(self, x: dict) -> dict:
        """Apply ESM embedding transformation.

        Args:
            x: Dictionary containing 'sequence' key with amino acid indices
               and optionally 'chains' key with chain IDs

        Returns:
            Dictionary with added 'esm_c_embeddings' key
        """
        if "sequence" not in x:
            logger.warning("No 'sequence' key found in batch, skipping ESM embedding generation")
            return x

        with torch.no_grad():  # Disable gradients for inference
            try:
                # Get chains if available
                chains = x.get("chains", None)

                # Generate ESM embeddings
                esm_embeddings = self.sequence_to_esm_embeddings(x["sequence"], chains)

                # Add to batch
                x["esm_c_embeddings"] = esm_embeddings

                logger.debug(f"Generated ESM embeddings of shape {esm_embeddings.shape}")

            except Exception as e:
                logger.error(f"Error generating ESM embeddings: {e}")
                # Add empty embeddings as fallback
                seq_len = x["sequence"].shape[-1] if x["sequence"].dim() > 0 else len(x["sequence"])
                x["esm_c_embeddings"] = torch.zeros((1, seq_len, 960), device=x["sequence"].device)

        return x


class StructureBackboneTransform(BaseTransform):
    def __init__(self, max_length=512, **kwargs):
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        logger.info("StructureBackboneTransform")
        self.max_length = max_length
        self.unk_idx = residue_constants.restype_order_with_x["X"]

    def __call__(self, x: dict) -> dict:
        """Rename keys in the input dictionary to match the expected input of the backbone model and crops.

        Args:
            x (dict): Input dictionary containing the data to be transformed.

        Returns:
            dict: Transformed dictionary with renamed keys.

        """
        x["mask"] = torch.ones_like(x["indices"])
        if "chains_ids" in x:
            x["chains"] = x["chains_ids"]
            del x["chains_ids"]
            # make sequence long instead of int
            x["sequence"] = x["sequence"].long()

        chains = x["chains"].tolist()
        if len(x["indices"]) > self.max_length:
            # get random chain
            set_chains = set(chains)
            chain = torch.randint(0, len(set_chains), (1,)).item()
            chain = list(set_chains)[chain]

            # get all indices in chain
            # chain_indices = [i for i in range(len(chains)) if chains[i] == chain]
            chains_array = np.array(chains)
            chain_indices = np.where(chains_array == chain)[0].tolist()

            if len(chain_indices) > self.max_length:
                # get random start
                start = np.random.randint(0, len(chain_indices) - self.max_length)
                x["coords_res"] = x["coords_res"][chain_indices[start : start + self.max_length]]
                x["indices"] = x["indices"][chain_indices[start : start + self.max_length]]
                x["mask"] = x["mask"][chain_indices[start : start + self.max_length]]
                x["sequence"] = x["sequence"][chain_indices[start : start + self.max_length]]
                x["chains"] = x["chains"][chain_indices[start : start + self.max_length]]
                if "esm_c_embeddings" in x:
                    x["plm_embeddings"] = x["esm_c_embeddings"][0][chain_indices[start : start + self.max_length]]
                    x["plm_embeddings"] = torch.tensor(
                        x["plm_embeddings"], dtype=x["coords_res"].dtype, device=x["coords_res"].device
                    )
            else:
                x["coords_res"] = x["coords_res"][chain_indices]
                x["indices"] = x["indices"][chain_indices]
                x["mask"] = x["mask"][chain_indices]
                x["sequence"] = x["sequence"][chain_indices]
                x["chains"] = x["chains"][chain_indices]
                if "esm_c_embeddings" in x:
                    x["plm_embeddings"] = x["esm_c_embeddings"][0][chain_indices]
                    x["plm_embeddings"] = torch.tensor(
                        x["plm_embeddings"], dtype=x["coords_res"].dtype, device=x["coords_res"].device
                    )
        else:
            if "esm_c_embeddings" in x:
                x["plm_embeddings"] = x["esm_c_embeddings"][0]
                x["plm_embeddings"] = torch.tensor(
                    x["plm_embeddings"], dtype=x["coords_res"].dtype, device=x["coords_res"].device
                )

        return x


class StructureTemplateTransform(BaseTransform):
    def __init__(self, template_percentage: float = 0.5, mask_percentage: float = 0.3, **kwargs):
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        super().__init__(**kwargs)
        self.template_percentage = template_percentage
        self.mask_percentage = mask_percentage

    def __call__(self, x: dict) -> dict:
        # apply this as the last transform
        chains = x["chains"].unique()
        if len(chains) > 1 and torch.rand(1) < self.template_percentage:
            # mask a random chain
            x["template_coords"] = x["coords_res"].clone()
            chain_to_mask = chains[torch.randint(0, len(chains), (1,)).item()]
            chain_indices = torch.nonzero(x["chains"] == chain_to_mask, as_tuple=True)[0]

            # Create template mask more efficiently
            template_mask = x["mask"].clone()
            template_mask[chain_indices] = 0

            # Calculate additional residues to mask
            remaining_mask = template_mask.bool()
            num_to_mask = int(remaining_mask.sum() * self.mask_percentage)

            if num_to_mask > 0:
                # Get indices of remaining residues and randomly select subset
                remaining_indices = torch.nonzero(remaining_mask, as_tuple=True)[0]
                indices_to_mask = remaining_indices[torch.randperm(remaining_indices.shape[0])[:num_to_mask]]

                # Combine all indices to mask in one operation
                all_indices_to_mask = torch.cat([chain_indices, indices_to_mask])
                x["template_coords"][all_indices_to_mask] = 0
                template_mask[all_indices_to_mask] = 0
            else:
                # Only mask chain indices
                x["template_coords"][chain_indices] = 0
                template_mask[chain_indices] = 0

            x["template_mask"] = template_mask
        else:
            x["template_coords"] = torch.zeros_like(x["coords_res"], device=x["coords_res"].device)
            x["template_mask"] = torch.zeros_like(x["mask"], device=x["mask"].device)
        return x


class StructureLigandTransform(BaseTransform):
    def __init__(self, max_length=512, rand_permute_ligand=False, **kwargs):
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        logger.info("StructureLigandTransform")
        self.max_length = max_length
        self.rand_permute_ligand = rand_permute_ligand
        self.periodic_table = Chem.GetPeriodicTable()

    def __call__(self, x: dict) -> dict:
        # Convert atom names to element indices using our vocabulary
        if "atom_names" in x:
            element_indices = torch.tensor(
                [
                    residue_constants.ELEMENT_TO_IDX[atom_name]  # Will raise KeyError if element not in vocab
                    for atom_name in x["atom_names"]
                ],
                dtype=torch.long,
            )
            x["element_indices"] = element_indices

        if self.rand_permute_ligand:
            random_order = torch.randperm(x["atom_coords"].shape[0])
            x["atom_coords"] = x["atom_coords"][random_order]
            x["mask"] = x["mask"][random_order]
            random_order_list = random_order.tolist()
            x["atom_names"] = [x["atom_names"][i] for i in random_order_list]
            if "element_indices" in x:
                x["element_indices"] = x["element_indices"][random_order]

        return x


class Structure3diTransform(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        self.encoder = Encoder()

    def __call__(self, x: dict) -> dict:
        Ca, Cb, N, C = calculate_cb(x)
        out = self.encoder.encode_atoms(Ca, Cb, N, C)
        sequence_3di = self.encoder.build_sequence(out["states"])
        x["3di_states"] = out["states"].data
        x["3di_descriptors"] = out["descriptors"].data
        x["3di_sequence"] = sequence_3di
        # turm to tensors from numpy arrays
        x["3di_states"] = torch.tensor(x["3di_states"], device=x["coords_res"].device)
        x["3di_descriptors"] = torch.tensor(x["3di_descriptors"], device=x["coords_res"].device)

        return x


class StructureC6DTransform(BaseTransform):
    def __init__(self, dist_cutoff: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        self.dist_cutoff = dist_cutoff

    def __call__(self, x: dict) -> dict:
        x["c6d"] = xyz_to_c6d(x["coords_res"][None])[0]
        # calc ca distance matrix
        ca_dist_matrix = torch.cdist(x["coords_res"][:, 1, :], x["coords_res"][:, 1, :], p=2)
        # set to mask if any element is less than dist_cutoff
        c6d_mask = ca_dist_matrix < self.dist_cutoff
        x["c6d_mask"] = c6d_mask
        x["c6d_binned"] = c6d_to_bins(x["c6d"])
        return x


class BinderTargetTransform(BaseTransform):
    def __init__(self, translation_scale: float = 1.0, **kwargs):
        """Initialize the BinderTargetTransform.

        Args:
            translation_scale (float): Scale for random SE(3) transformations.
            **kwargs: Additional arguments.

        """
        super().__init__(**kwargs)
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        # logger.info("BinderTargetTransform")
        self.translation_scale = translation_scale

    def make_conditioning_res_tensor_mask(self, chains, mask):
        """Create a mask that masks one of the unique chains entirely at random for each batch index.

        Args:
            chains (torch.Tensor): A tensor of shape (batch_size, seq_length) where each unique value represents a chain, and -1 indicates padding.
            mask (torch.Tensor): A tensor of the same shape as `chains` indicating valid positions (1 for valid, 0 for invalid).

        Returns:
            torch.Tensor: A tensor mask of the same shape as the input chains tensor.

        """
        conditioning_mask = torch.zeros_like(chains, dtype=torch.float32)  # Start with all ones (unmasked)
        conditioning_mask_2 = torch.zeros_like(chains, dtype=torch.float32)  # Start with all ones (unmasked)
        batch_size, seq_length = chains.shape

        for i in range(batch_size):
            # Get unique chain IDs for the current batch index, excluding -1 (padding)
            unique_chains = torch.unique(chains[i])
            unique_chains = unique_chains[unique_chains != -1]

            if len(unique_chains) > 1:
                # Randomly select one chain to mask
                chain_to_mask = unique_chains[torch.randint(0, len(unique_chains), (1,)).item()]

                # Get indices corresponding to the selected chain and mask them
                chain_indices = torch.nonzero(chains[i] == chain_to_mask, as_tuple=True)[0]
                # Create a boolean mask for all positions
                all_indices = torch.arange(chains.size(1), device=chains.device)
                non_chain_indices = torch.isin(all_indices, chain_indices, invert=True)
                conditioning_mask[i, chain_indices] = 1
                conditioning_mask_2[i, non_chain_indices] = 1

            else:
                # If there's only one unique chain, mask all positions
                conditioning_mask[i] = 0
                conditioning_mask_2[i] = 0
        conditioning_mask[mask == 0] = 0  # Set padding positions to zero
        conditioning_mask_2[mask == 0] = 0  # Set padding positions to zero

        return conditioning_mask, conditioning_mask_2

    def __call__(self, batch: dict) -> dict:
        """Apply the transformation to the input batch.

        Args:
            batch (dict): Input batch containing keys:
                - "coords_res": Coordinates tensor of shape (batch_size, seq_length, 3).
                - "indices": Residue indices tensor of shape (batch_size, seq_length).
                - "mask": Mask tensor of shape (batch_size, seq_length).
                - "chains": Chains tensor of shape (batch_size, seq_length).
                - "sequence": Sequence tensor.

        Returns:
            dict: Transformed feature dictionary.

        """
        # Extract inputs from the batch
        coords = batch["coords_res"].clone()
        coords = apply_random_se3_batched(coords, translation_scale=self.translation_scale)
        residue_index = batch["indices"].clone()
        mask = batch["mask"].clone()
        chains = batch["chains"].clone()

        # Generate conditioning mask
        conditioning_res_tensor_mask, conditioning_res_tensor_mask_2 = self.make_conditioning_res_tensor_mask(
            chains, mask
        )

        # Apply random SE(3) transformation
        coords_1 = apply_random_se3_batched(coords.clone(), translation_scale=self.translation_scale)
        cond_coords = coords_1 * conditioning_res_tensor_mask[:, :, None, None]
        # Set masked regions to -1
        cond_residue_index = residue_index * conditioning_res_tensor_mask
        cond_residue_index[conditioning_res_tensor_mask == 0] = -1
        cond_seq_mask = mask * conditioning_res_tensor_mask

        sequence = batch["sequence"].clone()

        # Generate binder mask and apply random SE(3) transformation
        coords_2 = apply_random_se3_batched(coords.clone(), translation_scale=self.translation_scale)
        cond_coords_2 = coords_2 * conditioning_res_tensor_mask_2[:, :, None, None]
        # Set masked regions to -1
        cond_residue_index_2 = residue_index * conditioning_res_tensor_mask_2
        cond_residue_index_2[conditioning_res_tensor_mask_2 == 0] = -1
        cond_seq_mask_2 = mask * conditioning_res_tensor_mask_2

        # Create feature dictionary
        feat_dict = {
            "mask": mask,
            "indices": residue_index,
            "chains": chains,
            "coords_res": coords,
            "input": (coords, mask, residue_index),
            "cond_target": (cond_coords, cond_seq_mask, cond_residue_index),
            "cond_binder": (cond_coords_2, cond_seq_mask_2, cond_residue_index_2),
            "sequence": sequence,
        }

        return feat_dict


class StructureResidueTransform(BaseTransform):
    def __init__(self, atom14=True, crop=None):
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        self.atom14 = atom14
        self.crop = crop
        logger.info(f"StructureResidueTransform: atom14={atom14}, crop={crop}")
        # make dictionary of expected num_atoms per residue type
        self.residue_atom_dict = {}
        for aa in residue_constants.restype_order_with_x:
            # convert to 3 letter code
            if aa == "X" or aa == "." or aa == "-":
                continue
            aa_3 = residue_constants.restype_1to3[aa]
            n_atoms = residue_constants.restype_name_to_atom_thin_names[aa_3]
            # count number of non blank string in n_atoms
            n_atoms = [i for i in n_atoms if i and i != "HH" and "HG" not in i]
            self.residue_atom_dict[aa] = n_atoms

    def __call__(self, x: dict) -> dict:
        """Transform coordinates from being n_atoms, 3 to being num_residues, 14, 3.

        Args:
            x (dict): A dictionary containing the coordinates.

        Returns:
            dict: A dictionary containing the transformed coordinates.

        """
        coords_atom14 = []
        coords_atom14_mask = []
        for i in range(len(x["sequence"])):
            residue_indices = x["residue2atom_dict"][i]

            residue = x["coords"][residue_indices]

            if "atoms_channel_ltr" not in x:
                x["atoms_channel_ltr"] = np.array(
                    [
                        residue_constants.HASH_2_ELEMENT_FUNCBIND[int(x["atoms_channel"][i])]
                        for i in range(len(x["atoms_channel"]))
                    ]
                )

            atoms = x["atoms_channel_ltr"][residue_indices]
            residue_length = residue.shape[0]
            if residue_length != len(self.residue_atom_dict[x["sequence"][i]]):
                min_residue_length = min(residue_length, len(self.residue_atom_dict[x["sequence"][i]]))
                # note atoms is of form element type ["N","C","C","O",...] while self.residue_atom_dict[x["sequence"][i]] is of form ["N","CA","C","O",...] so will need to convert to element type by just using the first letter
                matched_atoms = []
                matched_residues = []
                # for k, atom in enumerate(atoms):
                for k in range(min_residue_length):
                    atom = atoms[k]
                    # note for some reason, the cif files list N C C C O instead of N C C O C so i need to investigate this but if we use biopandas for pdb it doesnt pop up (might be specific to my parser); this might be becaus eof the cif data from the af3 dataset i was using
                    if atom == self.residue_atom_dict[x["sequence"][i]][k][0]:
                        matched_atoms.append(atom)
                        matched_residues.append(residue[k])

                residue = np.array(matched_residues)
                residue_length = residue.shape[0]

            # fill to 14 atoms
            if len(residue.shape) == 1:
                residue_mask = np.zeros(14)
                residue_fill = np.zeros((14, 3))
            else:
                residue_mask = np.zeros(14)
                residue_mask[:residue_length] = 1
                residue_fill = np.zeros((14, 3))
                residue_fill[:residue_length] = residue

            coords_atom14.append(residue_fill)
            coords_atom14_mask.append(residue_mask)
        coords_atom14 = np.array(coords_atom14)
        coords_atom14_mask = np.array(coords_atom14_mask)
        x["coords_atom14"] = torch.tensor(coords_atom14, dtype=torch.float32)
        x["coords_atom14_mask"] = torch.tensor(coords_atom14_mask, dtype=torch.float32)

        if self.crop is not None:
            ##DO NOT USE THIS WITH INTERFACES AND CROPPING FROM STRUCTURE TRANMSFORM
            # ensure it is an integer
            self.crop = int(self.crop)
            if len(x["indices"]) > self.crop:
                if max(0, len(x["indices"]) - self.crop) > 0:
                    start = torch.randint(0, len(x["indices"]) - self.crop, (1,)).item()
                else:
                    start = 0
                end = start + self.crop
                x["coords_atom14"] = x["coords_atom14"][start:end]
                x["coords_atom14_mask"] = x["coords_atom14_mask"][start:end]
                x["indices"] = x["indices"][start:end]
                x["epitope_tensor"] = x["epitope_tensor"][start:end]
                x["seq_int"] = x["seq_int"][start:end]
                x["sequence_index"] = x["sequence_index"][start:end]
                x["sequence"] = x["sequence"][start:end]
                x["sequence_chain_ids"] = x["sequence_chain_ids"][start:end]
                conditioning_res_tensor_mask = torch.zeros(len(x["indices"]))
                x["conditioning_res_tensor_mask"] = conditioning_res_tensor_mask

        return x


class RandomChainTransform(BaseTransform):
    """Transform that picks a random chain and filters data to only include that chain."""

    def __init__(self, **kwargs):
        import lobster

        lobster.ensure_package("torch_geometric", group="lg-gpu (or --extra lg-cpu)")

        """Initialize the RandomChainTransform.

        Args:
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        logger.info("RandomChainTransform")

    def __call__(self, x: dict) -> dict:
        """Pick a random chain and filter data to only include that chain.

        This transform assumes it's run after StructureBackboneTransform, so it only
        handles the keys that StructureBackboneTransform produces.

        Args:
            x (dict): Input dictionary containing the data to be transformed.
                Expected keys: 'chains', 'coords_res', 'indices', 'mask', 'sequence'
                Optional keys: 'plm_embeddings'

        Returns:
            dict: Transformed dictionary with only the selected chain's data.
        """
        # Get unique chains
        chains = x["chains"]
        unique_chains = torch.unique(chains)

        # Remove any padding values (-1) from unique chains
        unique_chains = unique_chains[unique_chains != -1]

        if len(unique_chains) == 0:
            logger.warning("No valid chains found, returning original data")
            return x

        # if there is only one chain, return the original data
        if len(unique_chains) == 1:
            return x

        # Pick a random chain
        chain_idx = torch.randint(0, len(unique_chains), (1,)).item()
        selected_chain = unique_chains[chain_idx]

        logger.debug(f"Selected chain: {selected_chain.item()}")

        # Create mask for the selected chain
        chain_mask = chains == selected_chain

        # Filter all relevant tensors to only include the selected chain
        x["coords_res"] = x["coords_res"][chain_mask]
        x["indices"] = x["indices"][chain_mask]
        x["mask"] = x["mask"][chain_mask]
        x["sequence"] = x["sequence"][chain_mask]
        x["chains"] = x["chains"][chain_mask]

        # Handle optional plm_embeddings (produced by StructureBackboneTransform)
        if "plm_embeddings" in x:
            x["plm_embeddings"] = x["plm_embeddings"][chain_mask]

        logger.debug(f"Filtered data to chain {selected_chain.item()}, new sequence length: {len(x['sequence'])}")

        return x
