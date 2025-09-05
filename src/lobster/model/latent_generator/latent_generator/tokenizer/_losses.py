"""losses for training"""
import abc

import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from einops import rearrange

from lobster.model.latent_generator.latent_generator.utils import kabsch_torch_batched


class TokenizerLoss(nn.Module, abc.ABC):
    def __init__(self):
        super(TokenizerLoss, self).__init__()

    @abc.abstractmethod
    def forward(self, ground_truth, predictions, mask, eps=1e-5,  **kwargs):
        """Implement loss function for training.

        Args:
            ground_truth (torch.Tensor): Ground truth tensor with shape (B, L, ...)
            predictions (torch.Tensor): Predictions tensor with shape (B, L, ...)
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)


        Returns:
            torch.Tensor: Loss value

        """
        ...

class L2Loss(TokenizerLoss):
    def __init__(self, clamp=50, ligand_weight=5.0, permute_chains=False):
        super(L2Loss, self).__init__()
        self.clamp = clamp
        self.permute_chains = permute_chains
        self.ligand_weight = ligand_weight

    def forward_ligand(self, ground_truth, predictions, mask, eps=1e-5,  **kwargs):
        #note that we do not consider relative reconstruction for the ligand and the protein
        predicted_protein = predictions["protein_coords"]
        if predicted_protein is not None:
            B, L, n_atoms, _ = predicted_protein.shape
            predicted_protein = predicted_protein[:, :, :3, :]
            ground_truth_protein = ground_truth['coords_res']
            ground_truth_protein = ground_truth_protein[:, :, :3, :]
            mask_protein = mask["protein_mask"]
        predicted_ligand = predictions["ligand_coords"]
        ground_truth_ligand = ground_truth["ligand_coords"]
        mask_ligand = mask["ligand_mask"]

        #align ground truth to predictions
        with torch.no_grad():
            with torch.autocast(enabled=False, device_type=predicted_ligand.device.type):
                if predicted_protein is not None:
                    mask_protein_expanded = mask_protein.unsqueeze(-1).repeat(1, 1, 3)
                    ground_truth_protein = kabsch_torch_batched(ground_truth_protein.reshape(B,-1,3), predicted_protein.reshape(B,-1,3), mask_protein_expanded.reshape(B,-1))
                    ground_truth_protein = ground_truth_protein.reshape(B,L,3,3)
                ground_truth_ligand = kabsch_torch_batched(ground_truth_ligand, predicted_ligand, mask_ligand)


        #calculate loss
        if predicted_protein is not None:
            loss_protein = nn.MSELoss(reduction='none')(predicted_protein, ground_truth_protein)
            loss_protein = loss_protein * mask_protein[:, :, None, None]
            loss_protein = loss_protein.sum() / (mask_protein.sum() + eps)
        else:
            loss_protein = 0

        loss_ligand = nn.MSELoss(reduction='none')(predicted_ligand, ground_truth_ligand)
        loss_ligand = loss_ligand * mask_ligand[:, :, None]
        loss_ligand = loss_ligand.sum() / (mask_ligand.sum() + eps)
        loss = loss_protein + self.ligand_weight * loss_ligand
        return loss

    def forward(self, ground_truth_, predictions, mask, eps=1e-5,  keep_batch_dim: bool = False, **kwargs):
        """Implement L2 loss function for training for structure reconstruction.

        Args:
            ground_truth (dict): Ground truth dict with keys "coords_res" shape (B, L, 3, 3)
            predictions (torch.Tensor): Predictions tensor with shape (B, L, 3, 3)
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)
            permute_chains (bool): Whether to switch the order of the chains score both and take the min for homodimers
            keep_batch_dim (bool): Whether to keep the batch dimension in the loss (default: False)
        """
        if isinstance(predictions, dict) and "protein_coords_refinement" not in predictions:
            return self.forward_ligand(ground_truth_, predictions, mask, eps, **kwargs)
        
        ground_truth= ground_truth_["coords_res"]
        B, L = ground_truth.shape[:2]
        ground_truth = ground_truth[:, :, :3, :]
        if isinstance(predictions, dict):
            predictions = predictions["protein_coords"]
        predictions = predictions[:, :, :3, :]

        #align predictions to ground truth
        with torch.no_grad():
            with torch.autocast(enabled=False, device_type=predictions.device.type):
                mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 3)
                ground_truth = kabsch_torch_batched(ground_truth.reshape(B,-1,3), predictions.reshape(B,-1,3), mask_expanded.reshape(B,-1))
                ground_truth = ground_truth.reshape(B,L,3,3)

        #use MSE loss
        if self.permute_chains:
            #the goal is to calculate the loss with one ordering of the chains (as it is without permuting)
            #and then permute the chains and calculate the loss again and take the min
            #this is to avoid the problem of the loss overpenalizing symmetric structures

            #step 1: get the chain indices for chains 1 and 2 per batch dimension
            chains_ids = ground_truth_["chains"]  # Shape: (B, L)
            unique_chains_per_batch = []
            for b in range(B):
                #make sure chains are in the same order as they appear in the ground truth
                unique_chains = chains_ids[b].unique_consecutive()
                #remove -1
                unique_chains = unique_chains[unique_chains != -1]
                unique_chains_per_batch.append(unique_chains)
            
            #step 2: calculate the loss with one ordering of the chains (Current ordering)
            loss = nn.MSELoss(reduction='none')(predictions, ground_truth)
            loss = loss * mask[:, :, None, None]
            loss = loss.sum(dim=(1,2,3))/mask.sum(dim=1)

            #step 2a: permute the chains and calculate the loss again and take the min
            with torch.no_grad():
                ground_truth_permuted = ground_truth.clone()
                for b in range(B):
                    unique_chains = unique_chains_per_batch[b]
                    if len(unique_chains) == 2:
                        chain_1_mask = chains_ids[b] == unique_chains[0]
                        chain_2_mask = chains_ids[b] == unique_chains[1]
                        chain_1_coords = ground_truth[b, chain_1_mask, :, :]
                        chain_2_coords = ground_truth[b, chain_2_mask, :, :]
                        permuted_coords = torch.cat([chain_2_coords, chain_1_coords], dim=0)
                        ground_truth_permuted[b, :permuted_coords.shape[0], :, :] = permuted_coords
                #step 2b: realign the permuted chains to the predictions
                with torch.autocast(enabled=False, device_type=predictions.device.type):
                    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 3)
                    ground_truth_permuted = kabsch_torch_batched(ground_truth_permuted.reshape(B,-1,3), predictions.reshape(B,-1,3), mask_expanded.reshape(B,-1))
                    ground_truth_permuted = ground_truth_permuted.reshape(B,L,3,3)

            loss_permuted = nn.MSELoss(reduction='none')(predictions, ground_truth_permuted)
            loss_permuted = loss_permuted * mask[:, :, None, None]
            loss_permuted = loss_permuted.sum(dim=(1,2,3))/mask.sum(dim=1)

            #step 3: take the min of the two losses per batch dimension
            loss_cat = torch.cat([loss[:, None], loss_permuted[:, None]], dim=1)
            loss = torch.min(loss_cat, dim=1)[0]
            loss = loss.sum() / B

        else:
            loss = nn.MSELoss(reduction='none')(predictions, ground_truth)
            loss = loss * mask[:, :, None, None]
            if keep_batch_dim:
                loss = loss.sum(dim=(1,2,3))/mask.sum(dim=1)
            else:
                loss = loss.sum() / (mask.sum() + eps)

        return loss
    
class L2RefinementLoss(TokenizerLoss):
    def __init__(self, clamp=50):
        super(L2RefinementLoss, self).__init__()
        self.clamp = clamp


    def forward(self, ground_truth_, predictions, mask, eps=1e-5,  keep_batch_dim: bool = False, **kwargs):
        """Implement L2 loss function for training for structure reconstruction.

        Args:
            ground_truth (dict): Ground truth dict with keys "coords_res" shape (B, L, 3, 3)
            predictions (dict): Predictions dict with keys "protein_coords_refinement" shape (B, L, 3, 3)
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)
            keep_batch_dim (bool): Whether to keep the batch dimension in the loss (default: False)
        """
        
        ground_truth= ground_truth_["coords_res"]
        B, L = ground_truth.shape[:2]
        ground_truth = ground_truth[:, :, :3, :]
        predictions = predictions["protein_coords_refinement"][:, :, :3, :]

        #align predictions to ground truth
        with torch.no_grad():
            with torch.autocast(enabled=False, device_type=predictions.device.type):
                mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 3)
                ground_truth = kabsch_torch_batched(ground_truth.reshape(B,-1,3), predictions.reshape(B,-1,3), mask_expanded.reshape(B,-1))
                ground_truth = ground_truth.reshape(B,L,3,3)

        loss = nn.MSELoss(reduction='none')(predictions, ground_truth)
        loss = loss * mask[:, :, None, None]
        if keep_batch_dim:
            loss = loss.sum(dim=(1,2,3))/mask.sum(dim=1)
        else:
            loss = loss.sum() / (mask.sum() + eps)

        return loss

class LigandL2Loss(TokenizerLoss):
    def __init__(self, ligand_weight=1.0):
        super(LigandL2Loss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.ligand_weight = ligand_weight

    def forward(self, ground_truth, predictions, mask, eps=1e-5,  **kwargs):
        predicted_ligand = self.ligand_weight * predictions["ligand_coords"]
        ground_truth_ligand = self.ligand_weight * ground_truth["ligand_coords"]
        mask_ligand = mask["ligand_mask"]

        #align ground truth to predictions
        with torch.no_grad():
            with torch.autocast(enabled=False, device_type=predicted_ligand.device.type):
                ground_truth_ligand = kabsch_torch_batched(ground_truth_ligand, predicted_ligand, mask_ligand)

        loss_ligand = nn.MSELoss(reduction='none')(predicted_ligand, ground_truth_ligand)
        loss_ligand = loss_ligand * mask_ligand[:, :, None]
        loss_ligand = loss_ligand.sum() / (mask_ligand.sum() + eps)
        return loss_ligand

class LigandPairWiseL2Loss(TokenizerLoss):
    def __init__(self, ligand_weight=1.0):
        super(LigandPairWiseL2Loss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.ligand_weight = ligand_weight 

    def forward(self, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        predicted_ligand = self.ligand_weight * predictions["ligand_coords"]
        ground_truth_ligand = self.ligand_weight * ground_truth["ligand_coords"]
        mask_ligand = mask["ligand_mask"]

        Dpred = torch.cdist(predicted_ligand, predicted_ligand, p=2)
        Dpred = torch.clamp(Dpred, max=20)
        D = torch.cdist(ground_truth_ligand, ground_truth_ligand, p=2)
        D = torch.clamp(D, max=20)
        E = (Dpred - D) ** 2
        E = torch.clamp(E, max=25)
        mask = mask_ligand[:, :, None] * mask_ligand[:, None, :]
        E = E * mask
        l = E.sum() / (mask.sum() + eps)
        return l


class ScalarRegressionLoss(TokenizerLoss):
    """Base class for scalar regression losses with optional z-score normalization.
    
    This base class provides normalized MSE loss computation for scalar properties.
    To handle scale mismatches between different scalar properties and coordinate
    reconstruction losses, this supports z-score normalization:
    
    normalized_value = (value - mean) / std
    
    This transforms scalar values to have approximately mean=0 and std=1, making the loss
    magnitude comparable to other tasks in multi-task learning. The normalization
    parameters should be computed from your training dataset statistics.
    
    Args:
        mean: Mean value for normalization (default: 0.0, no normalization)  
        std: Standard deviation for normalization (default: 1.0, no normalization)
    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean = mean  # Dataset-specific mean value
        self.std = std    # Dataset-specific std value
    
    def get_ground_truth_key(self):
        """Return the key to extract ground truth from the batch dict.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_ground_truth_key()")
    
    def forward(self, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        """
        Args:
            ground_truth (dict): Contains the target scalar values
            predictions (torch.Tensor): Predicted scalar values with shape (B,)
            mask: Not used for global properties like scalars
        """
        gt_key = self.get_ground_truth_key()
        gt_values = ground_truth[gt_key]  # Shape: (B,)
        pred_values = predictions  # Shape: (B,)
        
        # Normalize ground truth and predictions using dataset statistics
        gt_values_norm = (gt_values - self.mean) / self.std
        pred_values_norm = (pred_values - self.mean) / self.std
        
        # MSE loss on normalized values
        loss = torch.nn.functional.mse_loss(pred_values_norm, gt_values_norm)
        return loss


class RgLoss(ScalarRegressionLoss):
    """MSE loss for radius of gyration regression with optional normalization.
    
    Inherits from ScalarRegressionLoss to provide normalized MSE loss computation
    specifically for radius of gyration values.
    
    Args:
        mean: Mean Rg value for normalization (default: 0.0, no normalization)  
        std: Std Rg value for normalization (default: 1.0, no normalization)
    """
    
    def get_ground_truth_key(self):
        """Return the key for radius of gyration in the ground truth dict."""
        return "radius_of_gyration"


class SasaLoss(ScalarRegressionLoss):
    """MSE loss for solvent accessible surface area regression with optional normalization.
    
    Inherits from ScalarRegressionLoss to provide normalized MSE loss computation
    specifically for SASA values.
    
    Args:
        mean: Mean SASA value for normalization (default: 0.0, no normalization)  
        std: Std SASA value for normalization (default: 1.0, no normalization)
    """
    
    def get_ground_truth_key(self):
        """Return the key for SASA in the ground truth dict."""
        return "solvent_accessible_surface_area"


class PairWiseL2Loss(TokenizerLoss):
    def __init__(self):
        super(PairWiseL2Loss, self).__init__()

    def forward(self, ground_truth, predictions, mask, eps=1e-5, keep_batch_dim: bool = False,**kwargs):
        """Implement Pairwise L2 loss function for training for structure reconstruction.

        Args:
            ground_truth (dict): Ground truth dict with keys "coords_res" shape (B, L, n_atoms, 3)
            predictions (torch.Tensor): Predictions tensor with shape (B, L, n_atoms, 3)
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)
            keep_batch_dim (bool): Whether to keep the batch dimension in the loss (default: False)

        """
        if isinstance(predictions, dict) and "protein_coords_refinement" not in predictions:
            ligand_present = True
        else:
            ligand_present = False
        if isinstance(predictions, dict) and "protein_coords_refinement" in predictions:
            predictions = predictions["protein_coords"]

        if ligand_present:
            #get the protein and ligand predictions
            predicted_protein = predictions["protein_coords"]
            if predicted_protein is not None:
                B, L, n_atoms, _ = predicted_protein.shape
                predicted_protein = predicted_protein[:, :, :3, :]
                ground_truth_protein = ground_truth['coords_res']
                ground_truth_protein = ground_truth_protein[:, :, :3, :]
                mask_protein = mask["protein_mask"]
                #Step 1: Flatten the predictions and ground truth by just taking CA
                Z_hat_protein = predicted_protein.reshape(B, -1, 3)
                Z_protein = ground_truth_protein.reshape(B, -1, 3)
                mask_protein = mask_protein.unsqueeze(-1).repeat(1, 1, n_atoms).view(B, -1)
            ground_truth_ligand = ground_truth["ligand_coords"]
            predicted_ligand = predictions["ligand_coords"]
            mask_ligand = mask["ligand_mask"]
            #concatenate the protein and ligand predictions and ground truth
            if predicted_protein is not None:
                Z_hat = torch.cat([Z_hat_protein, predicted_ligand], dim=1)
                Z = torch.cat([Z_protein, ground_truth_ligand], dim=1)
                mask = torch.cat([mask_protein, mask_ligand], dim=1)
            else:
                Z_hat = predicted_ligand
                Z = ground_truth_ligand
                mask = mask_ligand
                n_atoms = 3

        else:
            ground_truth= ground_truth["coords_res"]
            B, L, n_atoms, _ = predictions.shape
            ground_truth = ground_truth[:, :, :n_atoms, :]

            # Step 1: Flatten predictions and ground_truth
            Z_hat = predictions.reshape(predictions.size(0), -1, 3) # (B, L*n_atoms,3)
            Z = ground_truth.reshape(ground_truth.size(0), -1, 3) # (B, L*n_atoms,3)

        # Step 2: Compute Dpred
        Dpred = torch.cdist(Z_hat, Z_hat, p=2) # (B, L*n_atoms, L*n_atoms)
        Dpred = torch.clamp(Dpred, max=20)


        # Step 3: Compute D
        D = torch.cdist(Z, Z, p=2) # (B, L*n_atoms, L*n_atoms)

        # Step 4: Compute E
        E = (Dpred - D) ** 2

        # Step 5: Clip E at 25
        E = torch.clamp(E, max=25)

        #step 5a: mask missing residues
        if n_atoms > 3:
            raise NotImplementedError("nonbackbone PairWiseL2Loss is not implemented correctly yet")
        else:
            if not ligand_present:
                mask = mask.unsqueeze(-1).repeat(1, 1, n_atoms).view(B, -1)
            mask = mask[:, None, :] * mask[:,:,None] # (B, L*n_atoms, L*n_atoms)
            E = E * mask
            if keep_batch_dim:
                l = E.sum(dim=(1,2))/mask.sum(dim=1)
            else:
                l = E.sum() / (mask.sum() + eps)

        return l

class PairWiseL2RefinementLoss(TokenizerLoss):
    def __init__(self):
        super(PairWiseL2RefinementLoss, self).__init__()

    def forward(self, ground_truth, predictions, mask, eps=1e-5, keep_batch_dim: bool = False,**kwargs):
        """Implement Pairwise L2 loss function for training for structure reconstruction.

        Args:
            ground_truth (dict): Ground truth dict with keys "coords_res" shape (B, L, n_atoms, 3)
            predictions (dict): Predictions dict with keys "protein_coords_refinement" shape (B, L, n_atoms, 3)
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)
            keep_batch_dim (bool): Whether to keep the batch dimension in the loss (default: False)

        """
        ground_truth= ground_truth["coords_res"]
        predictions = predictions["protein_coords_refinement"]
        B, L, n_atoms, _ = predictions.shape
        ground_truth = ground_truth[:, :, :n_atoms, :]

        # Step 1: Flatten predictions and ground_truth
        Z_hat = predictions.reshape(predictions.size(0), -1, 3) # (B, L*n_atoms,3)
        Z = ground_truth.reshape(ground_truth.size(0), -1, 3) # (B, L*n_atoms,3)

        # Step 2: Compute Dpred
        Dpred = torch.cdist(Z_hat, Z_hat, p=2) # (B, L*n_atoms, L*n_atoms)
        Dpred = torch.clamp(Dpred, max=20)


        # Step 3: Compute D
        D = torch.cdist(Z, Z, p=2) # (B, L*n_atoms, L*n_atoms)

        # Step 4: Compute E
        E = (Dpred - D) ** 2

        # Step 5: Clip E at 25
        E = torch.clamp(E, max=25)

        #step 5a: mask missing residues
        if n_atoms > 3:
            raise NotImplementedError("nonbackbone PairWiseL2RefinementLoss is not implemented correctly yet")
        else:
            mask = mask.unsqueeze(-1).repeat(1, 1, n_atoms).view(B, -1)
            mask = mask[:, None, :] * mask[:,:,None] # (B, L*n_atoms, L*n_atoms)
            E = E * mask
            if keep_batch_dim:
                l = E.sum(dim=(1,2))/mask.sum(dim=1)
            else:
                l = E.sum() / (mask.sum() + eps)

        return l

class CCELoss(TokenizerLoss):
    def __init__(self, label_smoothing=0.1):
        super(CCELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        """Categorical cross entropy loss function for training logits.

        Args:
            ground_truth (dict): Ground truth dict with keys "3di_states" shape (B, L)
            predictions (torch.Tensor): Predictions tensor with shape (B, L, n_tokens)
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)

        """
        if isinstance(mask,dict):
            mask = mask["protein_mask"]

        ground_truth= ground_truth["3di_states"].long()
        B, L, n_tokens = predictions.shape
        predictions = rearrange(predictions, "b n c -> b c n")

        loss = self.criterion(predictions, ground_truth)
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + eps)

        return loss

class SequenceCCELoss(TokenizerLoss):
    def __init__(self, label_smoothing=0.1):
        super(SequenceCCELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        """Categorical cross entropy loss function for training logits.

        Args:
            ground_truth (dict): Ground truth dict with keys "sequence" shape (B, L)
            predictions (torch.Tensor): Predictions tensor with shape (B, L, n_tokens)
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)

        """
        if isinstance(mask,dict):
            mask = mask["protein_mask"]
        ground_truth= ground_truth["sequence"].long()
        B, L, n_tokens = predictions.shape
        predictions = rearrange(predictions, "b n c -> b c n")

        loss = self.criterion(predictions, ground_truth)
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + eps)

        return loss
    
class ElementCCELoss(TokenizerLoss):
    def __init__(self, label_smoothing=0.1):
        super(ElementCCELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, ground_truth, predictions, mask, eps=1e-5):
        """Categorical cross entropy loss function for ligand element type prediction.

        Args:
            ground_truth (dict): Ground truth dict with keys "ligand_element_indices" shape (B, L)
            predictions (torch.Tensor): Predictions tensor with shape (B, L, 14)
            mask (torch.Tensor): Mask tensor with shape (B, L) or dict with "ligand_mask"
            eps (float): Small value for numerical stability (default: 1e-5)

        """
        if isinstance(mask, dict):
            mask = mask["ligand_mask"]

        ground_truth_elements = ground_truth["ligand_element_indices"].long()
        predictions = rearrange(predictions, "b n c -> b c n")

        loss = self.criterion(predictions, ground_truth_elements)
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + eps)

        return loss
    
class C6DLoss(TokenizerLoss):
    def __init__(self, label_smoothing=0.0):
        super(C6DLoss, self).__init__()
        #self.criterion = torch.nn.MSELoss(reduction='none')
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        """Mean squared error loss function for outputed distance metric.

        Args:
            ground_truth (dict): Ground truth dict with keys "c6d" shape (B, L, L, 4)
            predictions (torch.Tensor): Predictions tensor with shape (B, L, L, 4)
            mask (torch.Tensor): Mask tensor with shape (B, L, L)
            eps (float): Small value for numerical stability (default: 1e-5)

        """
        if isinstance(mask,dict):
            mask = mask["protein_mask"]
        ground_truth_ = ground_truth["c6d_binned"]
        mask = ground_truth["c6d_mask"]
        losses = 0
        for i in range(4):
            predictions_ = predictions[i]
            ground_truth__ = ground_truth_[...,i]
            loss = self.criterion(predictions_, ground_truth__)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + eps)
            losses += loss


        return losses
    
class MSELatentAlignLoss(TokenizerLoss):
    def __init__(self):
        super(MSELatentAlignLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        """Mean squared error loss function for outputed distance metric.

        Args:
            ground_truth (dict): Not used
            predictions (dict): Predictions dict with keys "pred_dist_metric" shape (B, L), "gt_dist_metric" shape (B, L), "mask" shape (B, L)
            mask (torch.Tensor): Not used
            eps (float): Small value for numerical stability (default: 1e-5)

        """
        ground_truth = predictions["gt_dist_metric"]
        prediction = predictions["pred_dist_metric"]
        mask = predictions["mask"]
        B, L = ground_truth.shape
        loss = self.criterion(prediction, ground_truth)

        loss = loss * mask
        loss = loss.sum() / (mask.sum() + eps)

        return loss

class BackboneDirectionLoss(TokenizerLoss):
    def __init__(self, max_error=20.0):
        super(BackboneDirectionLoss, self).__init__()
        self.max_error = max_error

    def forward(self, ground_truth, predictions, mask, eps=1e-5, **kwargs):
        """Backbone direction loss that computes six direction vectors and their pairwise dot products.
        
        Args:
            ground_truth (torch.Tensor): Ground truth tensor with shape (B, L, 3, 3) where last dim is N, CA, C
            predictions (torch.Tensor): Predictions tensor with shape (B, L, 3, 3) where last dim is N, CA, C
            mask (torch.Tensor): Mask tensor with shape (B, L)
            eps (float): Small value for numerical stability (default: 1e-5)
            
        Returns:
            torch.Tensor: Loss value
        """
        ground_truth= ground_truth["coords_res"]
        B, L, _, _ = predictions.shape
        
        # Extract N, CA, C coordinates
        N_pred = predictions[:, :, 0, :]  # (B, L, 3)
        CA_pred = predictions[:, :, 1, :]  # (B, L, 3)
        C_pred = predictions[:, :, 2, :]  # (B, L, 3)
        
        N_gt = ground_truth[:, :, 0, :]  # (B, L, 3)
        CA_gt = ground_truth[:, :, 1, :]  # (B, L, 3)
        C_gt = ground_truth[:, :, 2, :]  # (B, L, 3)
        
        # Compute the six direction vectors for predictions
        # a. CA → C*
        v1_pred = C_pred - CA_pred  # (B, L, 3)
        
        # b. C* → CA
        v2_pred = CA_pred - C_pred  # (B, L, 3)
        
        # c. CA → C_next (shift by 1)
        C_next_pred = torch.roll(C_pred, shifts=-1, dims=1)
        v3_pred = C_next_pred - CA_pred  # (B, L, 3)
        
        # d. v4 = -(CA → C*) × (C* → CA) = -(v1 × v2)
        v4_pred = -torch.cross(v1_pred, v2_pred, dim=-1)  # (B, L, 3)
        
        # e. v5 = N_prev → C × (CA → C*)
        N_prev_pred = torch.roll(N_pred, shifts=1, dims=1)
        v5_pred = torch.cross(N_prev_pred - C_pred, v1_pred, dim=-1)  # (B, L, 3)
        
        # f. v6 = (C* → CA) × (CA → C_next) = v2 × v3
        v6_pred = torch.cross(v2_pred, v3_pred, dim=-1)  # (B, L, 3)
        
        # Stack all vectors for predictions: (B, L, 6, 3)
        vectors_pred = torch.stack([v1_pred, v2_pred, v3_pred, v4_pred, v5_pred, v6_pred], dim=2)
        
        # Compute the six direction vectors for ground truth
        # a. CA → C*
        v1_gt = C_gt - CA_gt  # (B, L, 3)
        
        # b. C* → CA
        v2_gt = CA_gt - C_gt  # (B, L, 3)
        
        # c. CA → C_next (shift by 1)
        C_next_gt = torch.roll(C_gt, shifts=-1, dims=1)
        v3_gt = C_next_gt - CA_gt  # (B, L, 3)
        
        # d. v4 = -(CA → C*) × (C* → CA) = -(v1 × v2)
        v4_gt = -torch.cross(v1_gt, v2_gt, dim=-1)  # (B, L, 3)
        
        # e. v5 = N_prev → C × (CA → C*)
        N_prev_gt = torch.roll(N_gt, shifts=1, dims=1)
        v5_gt = torch.cross(N_prev_gt - C_gt, v1_gt, dim=-1)  # (B, L, 3)
        
        # f. v6 = (C* → CA) × (CA → C_next) = v2 × v3
        v6_gt = torch.cross(v2_gt, v3_gt, dim=-1)  # (B, L, 3)
        
        # Stack all vectors for ground truth: (B, L, 6, 3)
        vectors_gt = torch.stack([v1_gt, v2_gt, v3_gt, v4_gt, v5_gt, v6_gt], dim=2)
        
        # Normalize vectors to unit length
        vectors_pred_norm = vectors_pred #torch.nn.functional.normalize(vectors_pred, p=2, dim=-1)
        vectors_gt_norm = vectors_gt #torch.nn.functional.normalize(vectors_gt, p=2, dim=-1)
        
        # Compute pairwise dot products for predictions: (B, L, 6, 6)
        # This computes dot product between each pair of vectors
        dot_products_pred = torch.einsum('blik,bljk->blij', vectors_pred_norm, vectors_pred_norm)
        
        # Compute pairwise dot products for ground truth: (B, L, 6, 6)
        dot_products_gt = torch.einsum('blik,bljk->blij', vectors_gt_norm, vectors_gt_norm)
        
        # Compute the difference and clamp
        diff = (dot_products_pred - dot_products_gt)**2
        diff = torch.clamp(diff, max=self.max_error)
        
        # Apply mask and compute mean
        # Expand mask to match the shape of diff: (B, L, 6, 6)
        mask_expanded = mask[:, :, None, None].expand(-1, -1, 6, 6)
        
        # Compute loss
        loss = (diff * mask_expanded).sum() / (mask_expanded.sum() + eps)
        
        return loss
