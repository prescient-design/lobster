# Gen-UME: A Unified Generative Model for Joint Protein Sequence and Structure Design via Discrete Flow Matching

**Authors:** Anonymous  
**Affiliation:** Anonymous  
**Models:** Gen-UME 90M (~90 million parameters), Gen-UME 470M (~470 million parameters)

---

## Abstract

Protein design requires simultaneous consideration of sequence and structure, as these modalities are fundamentally coupled through protein folding constraints. We present **Gen-UME**, a unified generative framework that jointly models protein sequences and structures in discrete token spaces. We introduce **LatentGenerator**, a Vision Transformer-based autoencoder with symmetric encoder and decoder (6 layers, 8 heads, 256 hidden dims) that tokenizes 3D structures from Cartesian coordinates using a novel Simple Linear Quantizer (SLQ) with learned Gumbel-Softmax quantization. LatentGenerator is a general framework applicable to any macromolecular structure data represented as Cartesian coordinates. We demonstrate its versatility with high-fidelity reconstruction across diverse molecular types: 1.260 ± 0.632 Å RMSD on CASP15 proteins and **0.043 ± 0.011 Å RMSD** on 30,936 GEOM small molecules with continuous embeddings (7× better than best discrete at 0.295 Å). Our **unified protein-ligand FSQ model** (4375 tokens) achieves exceptional performance on both modalities within a single model: 1.008 ± 0.107 Å RMSD on proteins (PDBbind complexes), 1.260 ± 0.632 Å RMSD on CASP15 proteins, and 0.395 ± 0.059 Å RMSD on ligands, validating its applicability to joint protein-ligand encoding and beyond. Gen-UME's architecture is built on NeoBERT ([Le Breton et al., 2025](https://arxiv.org/abs/2502.19587)), a next-generation encoder, with novel extensions for sequence-structure co-generation through multimodal embeddings. While our framework is compatible with any discrete generative paradigm (autoregressive models, discrete diffusion, masked language modeling), our experiments focus on discrete flow matching via Continuous Time Markov Chains (CTMC), with independent time sampling enabling different generation dynamics per modality. We train models at two scales: 90M and 470M parameters. Both support three generation modes within a single framework: (1) unconditional generation of novel proteins, (2) inverse folding for sequence design, and (3) forward folding for structure prediction. We introduce a self-reflection mechanism—analogous to reasoning in large language models—where the model internally verifies sequence-structure consistency before producing final outputs. On benchmark evaluation, Gen-UME 90M achieves **50.83% amino acid recovery with 0.83 TM-score (2.41 Å RMSD, 63.8% pass rate)** on inverse folding and **0.71 TM-score (4.09 Å RMSD, 42.1% pass rate)** on forward folding. The larger 470M model demonstrates clear performance gains with scale: **55.26% AAR with 0.84 TM-score (2.34 Å RMSD, 64.8% pass rate)** on inverse folding and **0.75 TM-score (3.94 Å RMSD, 57.2% pass rate)** on forward folding. For unconditional generation with self-reflection, the model produces high-quality structures with **TM-scores exceeding 0.80** when validated with ESMFold, with **85% of 100-residue proteins** achieving RMSD < 2Å between generated and folded structures, and **11-25% structural diversity** across generated samples.

---

## 1. Introduction

The intimate relationship between protein sequence and structure is central to biology. A protein's amino acid sequence determines how it folds into a three-dimensional structure, which in turn determines its function. Computational protein design must navigate this complex sequence-structure relationship to create proteins with desired properties.

Traditional approaches treat protein sequence design and structure prediction as separate tasks. Structure prediction methods achieve high accuracy but cannot generate sequences for target structures. Inverse folding methods design sequences for given structures but cannot predict structures or generate novel proteins unconditionally. This separation fails to capture the inherent coupling between modalities and limits opportunities for joint optimization.

Recent advances in generative modeling, particularly flow-based approaches, offer promise for modeling complex structured data. Concurrently, modern encoder architectures like NeoBERT ([Le Breton et al., 2025](https://arxiv.org/abs/2502.19587)) have demonstrated significant improvements over traditional BERT-based models through optimized depth-to-width ratios and extended context lengths. We introduce **Gen-UME** (Generative Universal Molecular Encoder), a unified framework for joint protein sequence and structure generation based on discrete flow matching, built on the NeoBERT architecture.

### Contributions

1. **LatentGenerator**: Vision Transformer-based structure autoencoder with symmetric encoder-decoder architecture (6 layers, 8 heads, 256 hidden dims, rotary position embeddings) for high-fidelity molecular tokenization (1.260 ± 0.632 Å RMSD on CASP15 proteins with FSQ 4375), extensible to proteins, ligands, and protein-ligand complexes through element-aware embeddings and 20Å spatial attention masking. We demonstrate a **unified protein-ligand FSQ 4375 model** achieving 1.008 ± 0.107 Å RMSD on proteins (PDBbind) and 0.395 ± 0.059 Å RMSD on ligands within a single model.

2. **Simple Linear Quantizer (SLQ)**: Novel differentiable quantization method using learned linear projections and Gumbel-Softmax (256 tokens, τ=0.5) embedded within LatentGenerator, providing efficient alternative to VQ-VAE and FSQ while enabling end-to-end training.

3. **Unified sequence-structure generation**: Single models (90M and 470M parameters) that jointly generate amino acid sequences and 3D protein structures through discrete flow matching on both modalities simultaneously.

4. **NeoBERT-based architecture**: Leverage NeoBERT ([Le Breton et al., 2025](https://arxiv.org/abs/2502.19587)), a state-of-the-art encoder with optimal depth-to-width ratio and 4,096 token context, adapted for multimodal protein sequence-structure generation.

5. **Discrete flow matching for proteins**: Adopt discrete flow matching via CTMC (Campbell et al., 2024) for categorical spaces, with masking-based probability paths and independent time sampling enabling principled joint sequence-structure generation.

6. **Multiple generation modes**: Support for unconditional generation, inverse folding, and forward folding within a unified framework through conditional flow matching.

7. **Self-reflection verification**: Consistency checking procedure analogous to reasoning in LLMs, where the model "checks its answer" by verifying bidirectional sequence-structure compatibility through internal forward and inverse folding, accepting outputs only if they pass both structural and sequence identity thresholds.

8. **Strong empirical performance**: Competitive results across three benchmarks with a unified model approach.

---

## 2. Related Work

**Protein Structure Prediction**: AlphaFold2 and ESMFold ([Lin et al., 2023](https://www.science.org/doi/10.1126/science.ade2574)) achieve remarkable accuracy for structure prediction from sequence. These specialized models excel at forward prediction but cannot generate sequences for structures or create novel proteins.

**Protein Sequence Design**: ProteinMPNN ([Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187)), PiFold, and related methods perform inverse folding—designing sequences for target structures. While effective, they operate only on structure inputs and cannot perform structure prediction or unconditional generation.

**Generative Protein Models**: RFdiffusion extends RoseTTAFold with diffusion for protein backbone generation. Chroma uses diffusion on protein structure clouds with equivariant networks. MultiFlow ([Campbell et al., 2024](https://arxiv.org/abs/2402.04997)) introduces discrete flow matching for protein co-design using Continuous Time Markov Chains, enabling joint sequence-structure generation through flow-based modeling. DPLM-2 ([Wang et al., 2024](https://arxiv.org/abs/2410.13782)) introduces a multimodal diffusion protein language model that uses lookup-free quantization for structure tokenization and achieves joint sequence-structure generation. These approaches typically operate on continuous representations or use separate diffusion processes for each modality.

**Vision Transformers for Molecules**: Vision Transformers have shown strong performance on image tasks and are increasingly applied to molecular modeling. Unlike equivariant GNNs that encode symmetries explicitly, ViTs learn spatial relationships through attention, providing flexibility for diverse molecular types. Our LatentGenerator uses ViTs for both encoding and decoding 3D molecular structures, with spatial attention masking (20Å cutoff) to capture local geometric dependencies while maintaining computational efficiency.

**Discrete Flow Matching**: Campbell et al. ([2024](https://arxiv.org/abs/2402.04997)) introduced Discrete Flow Models (DFMs) for generative modeling on discrete state-spaces, enabling flow matching on categorical data through Continuous Time Markov Chains (CTMC). Their MultiFlow model demonstrated the effectiveness of discrete flow matching for protein sequence-structure co-design, using masking-based interpolation schedules to define probability paths between prior and data distributions. We adopt their discrete flow matching framework with CTMC dynamics, building upon it with NeoBERT architecture, novel Simple Linear Quantizer (SLQ) for structure tokenization, and independent time sampling for multimodal flows.

**Multimodal Protein Models**: Multimodal models have demonstrated the value of joint training across biological modalities (sequences, structures, MSAs, etc.). Gen-UME extends this paradigm to generative modeling, enabling a single model to perform multiple design tasks through conditional generation.

**Modern Encoder Architectures**: NeoBERT ([Le Breton et al., 2025](https://arxiv.org/abs/2502.19587)) represents the state-of-the-art in bidirectional encoders, achieving superior performance through optimized architecture design, modern pre-training data, and extended context (4,096 tokens). Despite its compact 250M parameter footprint, NeoBERT outperforms larger models like BERT-large and RoBERTa-large on the MTEB benchmark. We leverage NeoBERT's architecture as the foundation for our sequence-structure encoder.

Our approach builds on Campbell et al.'s discrete flow matching framework with several key contributions: (1) we introduce Simple Linear Quantizer (SLQ), a novel differentiable quantization method for structure tokenization; (2) we build on NeoBERT's state-of-the-art encoder architecture for sequence-structure encoding; (3) we use independent time sampling for multimodal flows, allowing different unmasking dynamics per modality; and (4) we provide a unified framework supporting multiple generation modes (unconditional, inverse folding, forward folding) through a single trained model with self-reflection verification for consistency checking.

---

## 3. Method

### 3.1 Problem Formulation

Let **s** = (s₁, ..., s_L) ∈ V^L denote a protein sequence of length L, where V is the amino acid vocabulary (|V| = 33: 20 standard amino acids plus special tokens). Let **x** ∈ ℝ^(L×3×3) denote 3D coordinates of the protein backbone (N, C_α, C atoms per residue).

We learn a joint distribution p(**s**, **x**) enabling three generation tasks:

1. **Unconditional**: Sample (**s**, **x**) ~ p(**s**, **x**)
2. **Inverse folding**: Sample **s** ~ p(**s**|**x**)
3. **Forward folding**: Sample **x** ~ p(**x**|**s**)

### 3.2 Architecture

#### Structure Tokenization with LatentGenerator

To enable discrete flow matching on structures, we convert continuous 3D coordinates to discrete tokens using a pre-trained **LatentGenerator** autoencoder. LatentGenerator is a powerful structure representation learning model based on Vision Transformers (ViT) that can handle proteins, ligands, and protein-ligand complexes, making it extensible to non-protein biomolecules.

**Architecture**: LatentGenerator employs symmetric Vision Transformer architectures for both encoding and decoding:

- **Encoder** (TimeCondUViTEncoder): 6-layer transformer with 8 attention heads, 32-dimensional head size, and 256 hidden dimensions
- **Quantizer** (Q): Simple Linear Quantizer (SLQ) with 256-token codebook 
- **Decoder** (TimeCondUViTDecoder): Symmetric 6-layer transformer matching encoder architecture

Both encoder and decoder use rotary position embeddings and process 3D backbone coordinates (N, Cα, C atoms). For protein-ligand complexes, the model uses 20Å spatial attention masking to capture local structural dependencies.

**Encoding-Decoding Pipeline**:
```
z = E(x, m)                 # Encode structure to latent embeddings
c = Q(z) ∈ {1,...,256}^L    # Quantize to discrete structure tokens  
x̂ = D(c, m)                 # Decode tokens back to 3D coordinates
```

where **x** ∈ ℝ^(L×3×3) are backbone coordinates and **m** ∈ {0,1}^L is a residue mask.

**Performance**: LatentGenerator achieves high-fidelity reconstruction with 1.260 ± 0.632 Å RMSD on CASP15 proteins (≤512 residues) using FSQ 4375 quantization, demonstrating that discrete tokenization preserves structural information including 3Di secondary structure and C6D distance/orientation features.

**Extensibility**: The ViT-based architecture naturally extends to non-protein biomolecules:
- **Ligands**: Processes ligand atom coordinates with element-aware embeddings, achieving **0.043 ± 0.011 Å RMSD** reconstruction on 30,936 ligands from the GEOM dataset ([Axelrod & Gómez-Bombarelli, 2022](https://doi.org/10.1038/s41597-022-01288-4)) with the continuous embedding model (7× better than best discrete FSQ 4375/15360 at 0.295 Å)
- **Protein-Ligand Complexes**: Joint encoding of protein residues and ligand atoms
- **General Molecules**: Flexible architecture can adapt to diverse molecular structures

This pre-trained LatentGenerator serves as a frozen structure encoder-decoder throughout Gen-UME training, providing a robust discrete representation of 3D molecular geometry.

**Simple Linear Quantizer (SLQ)**: We employ a novel Simple Linear Quantizer for efficient and differentiable quantization. Given continuous latent embeddings **z** ∈ ℝ^(L×d) from the structure encoder:

```
z_norm = LayerNorm(z)                    # Normalize embeddings
logits = Linear(z_norm) ∈ ℝ^(L×256)      # Project to token space
c = GumbelSoftmax(logits, τ=0.5)         # Soft quantization
```

The quantizer uses:
- **256-token codebook**: Discrete vocabulary for structure representations
- **Gumbel-Softmax**: Differentiable approximation to argmax with temperature τ=0.5, enabling end-to-end gradient flow during training
- **Layer normalization**: Stabilizes the logit predictions
- **Embedding dimension**: d=4 for the latent space before projection

**SLQ vs. FSQ**: Our Simple Linear Quantizer differs from Finite Scalar Quantization ([Mentzer et al., 2023](https://arxiv.org/abs/2309.15505)) in several key ways:
- **SLQ**: Uses learned linear projection + Gumbel-Softmax for soft, differentiable quantization with a learned codebook
- **FSQ**: Projects to few dimensions (e.g., 3) with fixed scalar quantization per dimension, creating an **implicit codebook** as the Cartesian product of quantization levels (e.g., [8,6,5] → 8×6×5 = 240 tokens)
- **Performance**: FSQ 4375 achieves best reconstruction quality with **1.260 ± 0.632 Å**, followed by SLQ at **1.647 ± 0.535 Å** and FSQ [8,6,5] at **1.848 ± 1.194 Å**
- **Codebook utilization**: SLQ uses 26.56% (68/256 tokens) while FSQ uses 99.17% (238/240 tokens). Despite lower utilization, SLQ achieves better performance, indicating learned quantization discovers more efficient representations.
- **Trade-offs**: SLQ offers superior average accuracy and **2.2× lower variance** (more consistent reconstructions), while FSQ uses nearly all available tokens with no learned parameters or codebook collapse issues. SLQ's learned representations provide adaptable quantization that better captures structural diversity with fewer active tokens.

**Structure Reconstruction Performance**:

LatentGenerator's reconstruction quality evaluated on CASP15 proteins ≤ 512 residues (26 structures):

| Model | Quantization | Codebook Size | Codebook Utilization | Avg RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) | Notes |
|-------|--------------|---------------|----------------------|--------------|--------------|--------------|--------------|-------|
| **LG Protein (continuous)** | None | - | - | **0.462** | **0.322** | **0.200** | **1.271** | Protein-only baseline without discretization |
| **LG PL cont (continuous)** | None | - | - | **0.651** | **0.339** | **0.452** | **1.830** | Protein-ligand model, continuous embeddings (256-dim) |
| **LG PL FSQ 4375** | FSQ | 4375 | - | **1.260** | **0.632** | **0.651** | **3.117** | **Best discrete performance** |
| LG PL FSQ 4375/15360 | FSQ | 4375/15360 | - | 1.418 | 0.810 | 0.748 | 3.396 | Asymmetric codebook (4375 protein, 15360 ligand) |
| LG PL FSQ 4375/15360 bond | FSQ | 4375/15360 | - | 1.443 | 0.837 | 0.697 | 3.453 | Bond matrix + extended element vocabulary (25 tokens) |
| **LG Protein SLQ** | SLQ | 256 | 26.56% (68/256) | 1.647 | 0.535 | 0.979 | 3.189 | Novel learned quantization with Gumbel-Softmax |
| **LG Protein-Ligand SLQ** | SLQ | 256 | - | 1.873 | 1.054 | 0.798 | 5.143 | Unified protein-ligand model (SLQ 256) |
| **LG Protein FSQ** | FSQ [8,6,5] | 240 (implicit) | 99.17% (238/240) | 1.848 | 1.194 | 0.483 | 5.419 | Fixed scalar quantization (8×6×5) |
| **LG PL SLQ 4096** | SLQ | 4096 | - | 3.097 | 2.009 | 1.242 | 8.474 | Unified protein-ligand model (SLQ 4096) |

The **continuous baselines** without quantization achieve the best reconstruction quality, establishing upper bounds for performance. The protein-only continuous model achieves **0.462 ± 0.322 Å RMSD**, while the **unified protein-ligand continuous model (LG PL cont)** achieves **0.651 ± 0.339 Å RMSD** on CASP15 proteins. The slightly higher RMSD for the protein-ligand model reflects the additional complexity of joint protein-ligand encoding, but both demonstrate that the ViT encoder-decoder architecture preserves structural information with high fidelity. These continuous embeddings are designed for use with diffusion-based structure generation in Gen-UME. Quantization inevitably introduces some reconstruction error, but all discrete methods maintain sub-2Å average accuracy suitable for downstream generative modeling.

Among discrete quantization methods, our **LG PL FSQ 4375** model with 4375 tokens achieves the best performance at **1.260 ± 0.632 Å RMSD**. The **Simple Linear Quantizer (SLQ)** with 256 explicit tokens achieves **1.647 ± 0.535 Å RMSD**, while **FSQ [8,6,5]** ([Mentzer et al., 2023](https://arxiv.org/abs/2309.15505)) with 240 implicit tokens (8×6×5 Cartesian product) achieves 1.848 ± 1.194 Å RMSD. Among smaller quantizers, SLQ demonstrates **2.2× lower variance** (0.535 vs 1.194 Å standard deviation) compared to FSQ [8,6,5], indicating more consistent reconstruction quality. The continuous baseline shows even lower variance (0.322 Å), but FSQ 4375's larger codebook provides the best trade-off between discretization and reconstruction fidelity among discrete methods.

**Unified Protein-Ligand Model**: The **LG Protein-Ligand FSQ 4375** model—trained jointly on proteins and ligands with FSQ quantization—achieves exceptional performance: **1.008 ± 0.107 Å RMSD** on proteins (1,617 PDBbind complexes) and **0.395 ± 0.059 Å RMSD** on ligands (30,936 GEOM molecules). This represents a 47% improvement over the previous ligand-only model (0.752 Å) while simultaneously providing superior protein reconstruction. The SLQ-based unified model achieves 1.873 ± 1.054 Å RMSD on CASP15 proteins and 0.920 ± 0.236 Å on GEOM ligands. Both models demonstrate that a single LatentGenerator can effectively tokenize diverse biomolecular structures, validating the extensibility of our approach to unified multi-modal molecular generation.

**Codebook Utilization**: Interestingly, SLQ uses only 26.56% of its codebook (68/256 unique tokens) while FSQ utilizes 99.17% (238/240 tokens). Despite using fewer active tokens, SLQ achieves superior reconstruction quality, suggesting that its learned representations discover a more efficient, compact subset of the token space that better captures essential structural features. This demonstrates that higher codebook utilization does not necessarily correlate with better reconstruction performance—instead, learned quantization can identify a smaller set of more meaningful discrete representations.

While FSQ uses a fixed scalar quantization per dimension without learned parameters, SLQ employs learned linear projections with Gumbel-Softmax for differentiable quantization, providing both better average performance and more stable reconstructions among discrete methods. The continuous baseline validates that the ViT architecture itself is highly effective for protein structure modeling, and all quantization methods successfully preserve most of this capability while enabling discrete token-based generation.

**Ligand Reconstruction Performance**:

LatentGenerator models were evaluated on 30,936 ligand structures from the GEOM dataset ([Axelrod & Gómez-Bombarelli, 2022](https://doi.org/10.1038/s41597-022-01288-4)):

| Model | Architecture | Codebook Size | Avg RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) |
|-------|--------------|---------------|--------------|--------------|--------------|--------------|
| **LG PL cont (continuous)** | ViT (6L, 8H) | - | **0.043** | **0.011** | **0.013** | **0.192** |
| LG PL FSQ 4375/15360 (unified) | ViT (6L, 8H) | 15360 | 0.295 | 0.052 | 0.120 | 1.792 |
| LG PL FSQ 4375 (minimized) | ViT (6L, 8H) | 4375 | 0.291 | 0.056 | 0.076 | 1.690 |
| LG PL FSQ 4375/15360 bond | ViT (6L, 8H) | 15360 | 0.354 | 0.052 | 0.117 | 1.089 |
| LG PL FSQ 4375 (unified) | ViT (6L, 8H) | 4375 | 0.395 | 0.059 | 0.154 | 1.842 |
| LG Ligand (ligand-only) | ViT (6L, 8H) | 512 | 0.752 | 0.305 | 0.065 | 4.943 |
| LG Protein-Ligand SLQ (unified) | ViT (6L, 8H) | 512 | 0.920 | 0.236 | 0.152 | 3.704 |
| LG PL SLQ 4096 (unified) | ViT (6L, 8H) | 4096 | 1.239 | 0.335 | 0.196 | 4.101 |

The **LG PL cont (continuous)** model achieves exceptional ligand reconstruction at **0.043 ± 0.011 Å RMSD**, representing a 7× improvement over the best discrete model (FSQ 4375/15360 at 0.295 Å). The continuous model also shows the tightest bounds (max 0.192 Å vs 1.792 Å for FSQ) and lowest variance (0.011 Å vs 0.052 Å). Among discrete methods, **LG PL FSQ 4375/15360** achieves **0.295 ± 0.052 Å RMSD**, a 61% improvement over the ligand-only model (0.752 Å). The **LG PL FSQ 4375** model achieves **0.395 ± 0.059 Å RMSD**, which is further reduced to **0.291 ± 0.056 Å** through post-decoding ligand minimization, matching the performance of the much larger 15360-token model. The continuous embeddings are designed for diffusion-based structure generation in Gen-UME, demonstrating that LatentGenerator's ViT-based architecture effectively preserves small molecule geometry with atomic-level precision.

**Protein-Ligand Complex Reconstruction Performance**:

To evaluate the unified protein-ligand model's ability to reconstruct protein-ligand complexes while preserving their relative positioning, we evaluated on 1,617 protein-ligand complexes from the PDBbind dataset:

| Model | Metric | Alignment | Avg RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) |
|-------|--------|-----------|--------------|--------------|--------------|--------------|
| **LG PL cont (continuous)** | **Protein** | Individual | **0.496** | **0.019** | **0.434** | **0.701** |
| LG PL FSQ 4375 | Protein | Individual | 1.010 | 0.109 | 0.691 | 1.546 |
| LG PL FSQ 4375/15360 | Protein | Individual | 1.010 | 0.107 | 0.710 | 1.934 |
| LG PL FSQ 4375/15360 bond | Protein | Individual | 1.026 | 0.107 | 0.663 | 1.685 |
| LG PL SLQ 256/512 | Protein | Individual | 1.483 | 0.232 | 0.898 | 3.901 |
| LG PL SLQ 4096 | Protein | Individual | 4.740 | 3.010 | 1.441 | 19.076 |
| **LG PL cont (continuous)** | **Ligand** | Individual | **0.499** | **0.046** | **0.279** | **0.665** |
| LG PL FSQ 4375 | Ligand | Individual | 0.702 | 0.143 | 0.375 | 2.296 |
| LG PL FSQ 4375 (minimized) | Ligand | Individual | 0.468 | 0.139 | 0.159 | 2.054 |
| LG PL FSQ 4375/15360 | Ligand | Individual | 0.657 | 0.146 | 0.315 | 2.407 |
| LG PL FSQ 4375/15360 bond | Ligand | Individual | 0.662 | 0.122 | 0.382 | 1.687 |
| LG PL SLQ 256/512 | Ligand | Individual | 1.411 | 0.593 | 0.365 | 4.519 |
| LG PL SLQ 4096 | Ligand | Individual | 1.620 | 0.711 | 0.533 | 6.756 |
| **LG PL cont (continuous)** | **Complex** | Joint | **0.507** | **0.019** | **0.463** | **0.817** |
| LG PL FSQ 4375 | Complex | Joint | 1.012 | 0.122 | 0.698 | 2.062 |
| LG PL FSQ 4375 (minimized) | Complex | Joint | 1.004 | 0.119 | 0.688 | 1.947 |
| LG PL FSQ 4375/15360 | Complex | Joint | 1.009 | 0.138 | 0.739 | 3.578 |
| LG PL FSQ 4375/15360 bond | Complex | Joint | 1.027 | 0.124 | 0.702 | 2.423 |
| LG PL SLQ 256/512 | Complex | Joint | 1.567 | 0.343 | 0.939 | 5.579 |
| LG PL SLQ 4096 | Complex | Joint | 4.680 | 2.962 | 1.415 | 19.173 |
| **LG PL cont (continuous)** | **Protein** | Joint (complex) | **0.496** | - | - | - |
| LG PL FSQ 4375 | Protein | Joint (complex) | 1.015 | - | - | - |
| LG PL FSQ 4375 (minimized) | Protein | Joint (complex) | 1.015 | - | - | - |
| LG PL FSQ 4375/15360 | Protein | Joint (complex) | 1.017 | - | - | - |
| LG PL FSQ 4375/15360 bond | Protein | Joint (complex) | 1.032 | - | - | - |
| LG PL SLQ 256/512 | Protein | Joint (complex) | 1.507 | 0.294 | 0.901 | 6.458 |
| LG PL SLQ 4096 | Protein | Joint (complex) | 4.761 | - | - | - |
| **LG PL cont (continuous)** | **Ligand** | Joint (complex) | **0.607** | - | - | - |
| LG PL FSQ 4375 | Ligand | Joint (complex) | 1.005 | - | - | - |
| LG PL FSQ 4375 (minimized) | Ligand | Joint (complex) | 0.854 | - | - | - |
| LG PL FSQ 4375/15360 | Ligand | Joint (complex) | 0.998 | - | - | - |
| LG PL FSQ 4375/15360 bond | Ligand | Joint (complex) | 1.078 | - | - | - |
| LG PL SLQ 256/512 | Ligand | Joint (complex) | 2.306 | 0.758 | 0.711 | 5.927 |
| LG PL SLQ 4096 | Ligand | Joint (complex) | 3.589 | - | - | - |

The **individual alignment** metrics align protein and ligand separately, measuring intrinsic reconstruction quality for each modality. The **complex alignment** aligns the entire protein-ligand complex together using Kabsch, preserving relative positioning between protein and ligand—this captures how well the binding pose is reconstructed. 

The **LG PL cont (continuous)** model achieves exceptional reconstruction performance with 256-dimensional continuous embeddings (no quantization): **0.496 ± 0.019 Å protein RMSD** and **0.499 ± 0.046 Å ligand RMSD** (individual alignment), with **0.507 ± 0.019 Å complex RMSD** (joint alignment). This represents a **2× improvement** over the best discrete model (FSQ 4375) for both proteins (0.496 Å vs 1.008 Å) and ligands (0.499 Å vs 0.702 Å). The continuous model also shows remarkably low variance (0.019 Å for protein), indicating highly consistent reconstructions across the dataset. This model is designed for use with diffusion-based structure generation in Gen-UME.

Among discrete quantization methods, the **LG Protein-Ligand FSQ 4375** model (using 4375 tokens for both proteins and ligands) achieves the best performance: **1.010 ± 0.109 Å protein RMSD** and **0.702 ± 0.143 Å ligand RMSD** (individual alignment), with **1.012 ± 0.122 Å complex RMSD** (joint alignment). Applying **post-decoding ligand minimization** further improves ligand geometry, reducing ligand RMSD to **0.468 ± 0.139 Å** (a 33% improvement) and complex RMSD to **1.004 ± 0.119 Å**. This represents a significant improvement over the SLQ 256/512 model, particularly for ligands (0.468 Å vs 1.411 Å, a 67% improvement) and proteins (1.010 Å vs 1.483 Å, a 32% improvement).

All models demonstrate that LatentGenerator can encode protein-ligand complexes while preserving binding geometry suitable for downstream generative modeling. The continuous model establishes an upper bound for reconstruction quality, while discrete models maintain sub-1.1 Å average accuracy suitable for token-based generation.

This approach provides a differentiable alternative to vector quantization (VQ-VAE) while maintaining discrete token semantics. During inference, tokens can be sampled from the categorical distribution or selected via argmax. The resulting discrete representation creates a unified token space where both sequences and structures are categorical variables (|V_seq| = 33, |V_struct| = 258 including mask/padding tokens).

#### Unified Sequence-Structure Encoder

For each residue position i, we compute a combined embedding:

```
e_i^seq = Embed_seq(s_i)           # Sequence embedding
e_i^struct = Embed_struct(c_i)     # Structure embedding  
e_i^cond = Linear(h_i)             # Optional conditioning
e_i = Linear([e_i^seq; e_i^struct; e_i^cond])  # Combined
```

These are processed by a NeoBERT transformer ([Le Breton et al., 2025](https://arxiv.org/abs/2502.19587)):

```
h₁, ..., h_L = NeoBERT(e₁, ..., e_L)
```

We use NeoBERT's architecture with 12 layers, 768 hidden dimensions (~90M parameters), optimized depth-to-width ratio, and support for extended context up to 4,096 tokens. Separate output heads predict logits:

```
ℓ_i^seq = Linear_seq(h_i) ∈ ℝ^33
ℓ_i^struct = Linear_struct(h_i) ∈ ℝ^258
```

This architecture enables bidirectional information flow between sequences and structures through the shared transformer, leveraging NeoBERT's state-of-the-art encoding capabilities.

### 3.3 Discrete Flow Matching

We adopt discrete flow matching on categorical spaces following Campbell et al. ([2024](https://arxiv.org/abs/2402.04997)). Discrete flow matching provides a principled framework for generative modeling on categorical data by defining flows through Continuous Time Markov Chains (CTMC).

#### Discrete Flow Matching via CTMC

For categorical variables (sequences **s** ∈ V^L and structure tokens **c** ∈ {1,...,K}^L), we define a probability path p_t(**z**) that interpolates between a prior distribution p₀(**z**) at t=0 and the data distribution p₁(**z**) at t=1.

**Conditional Flow Construction**: Given a data sample **z**₁ ~ p₁(**z**), we construct a conditional probability path:

```
p_t(z | z₁) = p(z_t = z | z₀ ~ p₀, z₁, t)
```

This path is governed by a time-dependent transition rate matrix **R**_t that defines a CTMC. Following Campbell et al., we use a **masking schedule** where tokens gradually unmask from t=0 to t=1:

```
R_t(i→j | z₁) = {
  α(t)           if i = [MASK], j = z₁ (unmasking to data)
  0              otherwise
}
```

where α(t) is the unmasking rate schedule. We use α(t) = 1-t, so the probability of being unmasked increases linearly with time.

**Marginal Distribution**: At time t, each position has probability:

```
p(z_t,i = k | z₁,i) = {
  α(t)           if k = z₁,i (unmasked to true token)
  1 - α(t)       if k = [MASK] (still masked)
  0              otherwise
}
```

This creates a simple interpolation where masked positions gradually reveal their true values.

#### Training Objective

The model learns a parameterized rate matrix **R**_θ(t) by minimizing the continuous-time flow matching loss. For a single token position, the loss is:

```
L_DFM = E_{z₁~p₁, t~U(0,1), z_t~p_t(·|z₁)} [CE(p_θ(z₁ | z_t, t), z₁)]
```

where p_θ(z₁ | z_t, t) = softmax(ℓ_θ(z_t, t)) are the model's predicted logits for the clean data token given the noisy observation z_t at time t.

**Key Training Details**:
- At time t, sample noisy tokens z_t ~ p_t(· | z₁) by masking each position with probability (1-α(t))
- Model predicts logits ℓ_θ(z_t, t) for the clean tokens
- Loss is cross-entropy between predicted distribution and true tokens z₁
- Gradient flows through the unmasked positions to learn the denoising function

#### Joint Sequence-Structure Training

Gen-UME trains both modalities jointly with **independent time sampling**:

```
L_flow = L_DFM^seq + L_DFM^struct
```

where:
- **L_DFM^seq**: Flow matching loss for amino acid sequences (vocabulary size |V| = 33)
- **L_DFM^struct**: Flow matching loss for structure tokens (vocabulary size K = 256)

We sample t_seq ~ U(0,1) and t_struct ~ U(0,1) **independently** for each modality, allowing the model to learn different unmasking dynamics for sequences and structures. This is critical because:
- Sequences may benefit from faster unmasking (more local dependencies)
- Structures may benefit from slower unmasking (more global constraints)

**Total Training Objective**:
```
L_total = L_DFM^seq + L_DFM^struct
```

Note that the structure decoder D(c, m) from LatentGenerator is **frozen** during Gen-UME training. The decoder was pre-trained as part of LatentGenerator and remains fixed, serving only to convert structure tokens back to 3D coordinates during inference. Gen-UME only trains the sequence-structure encoder and the discrete flow matching dynamics.

**Training Configuration**:
- Optimizer: AdamW (β₁=0.9, β₂=0.98, ε=10⁻¹²)
- Learning rate: 10⁻³ with 20K step warmup
- Training steps: 100,000 total
- Prior: Fully masked tokens for both modalities (all positions = [MASK])
- Unmasking schedule: α(t) = t (linear from 0 to 1)

### 3.4 Inference and Generation

#### Sampling Procedure

Following the discrete flow matching framework, generation proceeds by simulating the reverse CTMC from t=0 (fully masked prior) to t=1 (data distribution). We discretize the time interval [0,1] into N steps with t_n = n/N.

**Discrete-Time Sampling**: At each step n, we update the tokens using the learned rate matrix:

```
For each masked position i:
  1. Compute logits: ℓ_i = ℓ_θ(z_t, t_n)
  2. Sample unmasking: p_unmask = α(t_{n+1}) - α(t_n)
  3. If sampled to unmask:
     z_{t_{n+1},i} ~ Categorical(softmax(ℓ_i / τ))
  4. Else: z_{t_{n+1},i} remains [MASK]
```

**Temperature-Controlled Sampling**: We use temperature τ to control the sharpness of the predicted categorical distribution:

```
s_{n+1} ~ Categorical(softmax(ℓ_θ^seq / τ_seq))
c_{n+1} ~ Categorical(softmax(ℓ_θ^struct / τ_struct))
```

where lower τ produces more confident (peaked) distributions and higher τ produces more diverse samples.

**Stochasticity-Controlled Transitions**: We use a stochasticity hyperparameter that influences the amount of noise added during the reverse sampling process. The stochasticity term appears in the step probability calculations:

For masked prior (used in our experiments):
```
step_prob = dt × p_θ(x₁|x_t) × ((1 + s×t)/(1-t)) × [x_t is MASK]
           + dt × [x_t not MASK] × p_MASK × s × [t+dt < 1]
```

where s is the stochasticity parameter. This formulation controls:
- **Unmasking rate**: Higher stochasticity increases the probability of unmasking tokens
- **Re-masking rate**: Stochasticity also governs the probability of re-masking already unmasked tokens
- **Exploration-exploitation tradeoff**: Higher values promote exploration (sampling diverse sequences), lower values promote exploitation (following the most likely trajectory)

After N steps, all positions are unmasked. For structure tokens, we decode to 3D coordinates: **x̂** = D(c_N, m).

**Generation Parameters** (optimized for 90M model):

| Mode | N (steps) | τ_seq | τ_struct | stochasticity_seq | stochasticity_struc |
|------|-----------|-------|----------|-------------------|---------------------|
| Unconditional | 1000 | 0.46 | 0.36 | 30 | 70 |
| Inverse Folding | 200 | 0.16 | 1.0 | 20 | 10 |
| Forward Folding | 100 | 0.30 | 0.11 | 10 | 30 |

The `stochasticity` parameter (range: 0 to N) controls the noise level and transition dynamics, with higher values enabling more exploration. The `temperature` parameter controls the sharpness of the categorical distributions. For the 470M model, unconditional generation and forward folding use optimized parameters with custom inference schedules (see Supplementary Table 5).

#### Conditional Generation

- **Inverse folding**: Fix **c** = **c**\* (t_struct ≈ 1.0), sample sequence tokens
- **Forward folding**: Fix **s** = **s**\* (t_seq ≈ 1.0), sample structure tokens

#### Self-Reflection Verification

For unconditional generation, we introduce self-reflection—a technique analogous to reasoning in large language models where the model "checks its answer" to verify self-consistency. Similar to how LLMs can verify their reasoning through chain-of-thought or self-consistency checks, Gen-UME verifies sequence-structure compatibility by using its own forward and inverse folding capabilities:

**Self-Reflection Pipeline**:
1. **Initial generation**: Generate initial (**s**₀, **x**₀) unconditionally
2. **Forward consistency check**: Generate **x**₁ from **s**₀ using the model's forward folding capability
3. **Structural consistency**: Compute TM-score between **x**₀ and **x**₁ to assess if the sequence folds to the predicted structure
4. **Inverse consistency check**: Generate **s**₁ from **x**₁ using the model's inverse folding capability
5. **Sequence consistency**: Compute sequence identity between **s**₀ and **s**₁ to assess if the structure encodes the predicted sequence
6. **Acceptance criteria**: Accept (**s**₀, **x**₀) if TM-score(**x**₀, **x**₁) > θ_TM (0.83) AND sequence identity(**s**₀, **s**₁) > θ_seq
7. **Return original output**: Return original (**s**₀, **x**₀) if both consistency checks pass, otherwise reject/retry

This self-reflection mechanism acts as a consistency filter, ensuring generated sequences can fold to their generated structures and vice versa. The model effectively "double-checks" its work by verifying mutual compatibility through internal forward and inverse folding, but returns the original generation—not the verification outputs—if it passes consistency checks. Just as LLMs benefit from reasoning steps to verify answers, Gen-UME benefits from internally checking that its sequence and structure predictions are mutually consistent.

---

## 4. Experiments

### 4.1 Setup

**Models**: 

**Gen-UME 90M**:
- Architecture: NeoBERT (12 layers, 768 hidden dim, 12 attention heads)
- Parameters: ~90 million
- Structure codebook: 256 tokens (SLQ)
- Max length: 512 residues

**Gen-UME 470M**:
- Architecture: NeoBERT (24 layers, 1024 hidden dim, 16 attention heads)
- Parameters: ~470 million
- Structure codebook: 256 tokens (SLQ)
- Max length: 512 residues

**Training Data**: High-quality PDB structures filtered for resolution and redundancy

**Benchmarks**:
- Inverse/Forward folding: Campbell et al. (ICML 2024) dataset
- Unconditional: Generated at lengths 100, 200, 300, 400, 500

### 4.2 Evaluation Metrics

**Inverse Folding**:
- AAR (Amino Acid Recovery): % positions matching native sequence
- TM-score: Structural similarity (target vs. ESMFold-predicted from designed sequence)

**Forward Folding**:
- TM-score: Structural similarity (generated vs. reference structure)

**Unconditional Generation**:
- ESMFold validation: Fold generated sequences, compare to generated structures
- TM-score: Similarity between Gen-UME structure and ESMFold prediction
- RMSD: C_α root-mean-square deviation
- pLDDT: ESMFold confidence score
- Diversity: Foldseek clustering (TM-score threshold 0.5)

### 4.3 Results

#### Inverse Folding

Performance on sequence design for given structures evaluated on two benchmarks:

**Campbell et al. ICML 2024 Benchmark (449 structures):**

| Model | Tokens | AAR (%) | TM-Score | Avg RMSD (Å) | Pass Rate (RMSD<2Å) | Avg pLDDT | Notes |
|-------|--------|---------|----------|--------------|---------------------|-----------|-------|
| **DPLM-2 650M ([Wang et al., 2024](https://arxiv.org/abs/2410.13782))** | **8192** | **55.56** | **0.88** | **1.90** | **77.1%** | **0.74** | Unified diffusion model |
| **Gen-UME 750M SLQ (ours)** | **256** | **58.20** | **0.83** | **2.50** | **60.6%** | **0.69** | Unified flow-based model with SLQ quantization |
| **Gen-UME 470M SLQ (ours)** | **256** | **55.26** | **0.84** | **2.34** | **64.8%** | **0.69** | Unified flow-based model with SLQ quantization |
| **Gen-UME 90M SLQ (ours)** | **256** | **50.83** | **0.83** | **2.41** | **63.8%** | **0.68** | Unified flow-based model with SLQ quantization |
| ProteinMPNN ([Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187)) | - | - | - | - | - | - | TODO: Benchmark |
| ESM-IF1 ([Hsu et al., 2022](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1)) | - | - | - | - | - | - | TODO: Benchmark |
| MultiFlow ([Campbell et al., 2024](https://arxiv.org/abs/2402.04997)) | - | - | - | - | - | - | TODO: Benchmark |

**CAMEO 2022 Benchmark ([Robin et al., 2021](https://doi.org/10.1002/prot.26213)) (127 structures):**

| Model | Tokens | AAR (%) | TM-Score | Avg RMSD (Å) | Pass Rate (RMSD<2Å) | Avg pLDDT | Notes |
|-------|--------|---------|----------|--------------|---------------------|-----------|-------|
| **Gen-UME 750M SLQ (ours)** | **256** | **32.83** | **0.73** | **3.94** | **46.5%** | **0.63** | Unified flow-based model with SLQ quantization|
| **Gen-UME 470M SLQ (ours)** | **256** | **32.94** | **0.73** | **3.89** | **44.1%** | **0.64** | Unified flow-based model with SLQ quantization |
| **Gen-UME 90M FSQ (ours)** | **240** | **30.35** | **0.74** | **3.49** | **52.0%** | **0.64** | Unified flow-based model with FSQ quantization |
| **Gen-UME 90M SLQ (ours)** | **256** | **29.76** | **0.72** | **3.97** | **48.8%** | **0.62** | Unified flow-based model with SLQ quantization |
| DPLM-2 650M ([Wang et al., 2024](https://arxiv.org/abs/2410.13782)) | 8192 | - | - | - | - | - | TODO: Benchmark |
| ProteinMPNN ([Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187)) | - | - | - | - | - | - | TODO: Benchmark |
| ESM-IF1 ([Hsu et al., 2022](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1)) | - | - | - | - | - | - | TODO: Benchmark |

On the Campbell et al. benchmark, DPLM-2 650M achieves strong inverse folding performance with 55.56% AAR, 0.88 TM-score, 1.90 Å RMSD, and 77.1% pass rate. **Gen-UME 750M SLQ surpasses DPLM-2 in sequence recovery with 58.20% AAR** (+4.7%), demonstrating that discrete flow matching scales effectively for sequence design. Gen-UME shows consistent scaling: AAR increases from 50.83% (90M) to 55.26% (470M) to 58.20% (750M), a +14.5% relative improvement. While DPLM-2 maintains superior structural accuracy (0.88 vs 0.83 TM-score, 1.90 Å vs 2.50 Å RMSD) and pass rate (77.1% vs 60.6%), Gen-UME 750M achieves higher sequence recovery with **32× fewer structure tokens** (256 vs 8192). All models achieve strong confidence scores (pLDDT 0.69-0.74).

On the more challenging CAMEO 2022 benchmark ([Robin et al., 2021](https://doi.org/10.1002/prot.26213)) with recent protein structures from 2022, Gen-UME models show consistent performance across scales: 750M SLQ achieves 32.83% AAR, 0.73 TM-score, 3.94 Å RMSD, and 46.5% pass rate; 470M SLQ achieves 32.94% AAR, 0.73 TM-score, 3.89 Å RMSD, and 44.1% pass rate. The lower sequence recovery on CAMEO 2022 compared to Campbell et al. (32.83% vs 58.20% AAR for 750M SLQ) reflects the increased difficulty of designing sequences for novel structures. At the 90M scale, FSQ achieves 30.35% AAR with 52.0% pass rate, outperforming SLQ (29.76% AAR, 48.8% pass rate) on structural metrics. All models maintain reasonable structural accuracy and demonstrate generalization to newer protein targets, with 44-52% of designed sequences folding back within 2Å RMSD.

#### Forward Folding

Performance on structure prediction from sequences evaluated on two benchmarks:

**Campbell et al. ICML 2024 Benchmark (449 structures):**

| Model | Tokens | TM-Score | Avg RMSD (Å) | Pass Rate (RMSD<2Å) | Notes |
|-------|--------|----------|--------------|---------------------|-------|
| **ESMFold 3B ([Lin et al., 2023](https://www.science.org/doi/10.1126/science.ade2574))** | **-** | **0.91** | **1.66** | **81.7%** | State-of-the-art specialized structure prediction |
| **DPLM-2 650M ([Wang et al., 2024](https://arxiv.org/abs/2410.13782))** | **8192** | **0.77** | **3.13** | **53.5%** | Unified diffusion model |
| **Gen-UME 750M SLQ (ours)** | **256** | **0.77** | **3.94** | **60.4%** | Unified flow-based model with SLQ quantization |
| **Gen-UME 470M SLQ (ours)** | **256** | **0.75** | **3.94** | **57.2%** | Unified flow-based model with SLQ quantization |
| **Gen-UME 90M SLQ (ours)** | **256** | **0.71** | **4.09** | **42.1%** | Unified flow-based model with SLQ quantization |
| **Gen-UME 90M FSQ (ours)** | **240** | **0.70** | **4.40** | **42.5%** | Unified flow-based model with FSQ quantization |
| MultiFlow ([Campbell et al., 2024](https://arxiv.org/abs/2402.04997)) | - | - | - | - | TODO: Benchmark |

**CAMEO 2022 Benchmark ([Robin et al., 2021](https://doi.org/10.1002/prot.26213)) (127 structures):**

| Model | Tokens | TM-Score | Avg RMSD (Å) | Pass Rate (RMSD<2Å) | Notes |
|-------|--------|----------|--------------|---------------------|-------|
| **ESMFold 3B ([Lin et al., 2023](https://www.science.org/doi/10.1126/science.ade2574))** | **-** | **0.85** | **2.50** | **65.4%** | State-of-the-art specialized structure prediction |
| **DPLM-2 650M ([Wang et al., 2024](https://arxiv.org/abs/2410.13782))** | **8192** | **0.70** | **4.31** | **35.4%** | Unified diffusion model |
| **Gen-UME 750M SLQ (ours)** | **256** | **0.65** | **7.03** | **36.2%** | Unified flow-based model with SLQ quantization |
| **Gen-UME 470M SLQ (ours)** | **256** | **0.65** | **6.15** | **34.6%** | Unified flow-based model with SLQ quantization |
| **Gen-UME 90M SLQ (ours)** | **256** | **0.63** | **5.61** | **26.0%** | Unified flow-based model with SLQ quantization |
| **Gen-UME 90M FSQ (ours)** | **240** | **0.62** | **5.91** | **28.3%** | Unified flow-based model with FSQ quantization |

ESMFold 3B achieves state-of-the-art forward folding performance (0.91 TM-score, 1.66 Å RMSD, 81.7% pass rate on Campbell et al.) as a specialized model trained exclusively for structure prediction. DPLM-2 650M demonstrates strong performance (0.77 TM-score, 3.13 Å RMSD, 53.5% pass rate) as a unified diffusion-based model supporting multiple tasks. Gen-UME demonstrates competitive structure prediction capability within a unified flow-based generative framework that also supports inverse folding and unconditional generation. **Gen-UME 750M SLQ matches DPLM-2's TM-score (0.77) while achieving a higher pass rate (60.4% vs 53.5%)**, demonstrating that our discrete flow matching approach scales effectively. Scaling from 90M to 750M SLQ shows consistent improvements: TM-score increases from 0.71 to 0.77 (+8.5%), and pass rate increases from 42.1% to 60.4% (+43.5% relative). At the 90M scale, both quantization approaches show comparable performance: SLQ achieves 0.71 TM-score with 42.1% pass rate, while FSQ achieves 0.70 TM-score with 42.5% pass rate. Gen-UME 750M SLQ achieves this performance with **32× fewer structure tokens** than DPLM-2 (256 vs 8192), demonstrating efficient discrete representation learning. The performance gap to ESMFold is partially attributable to model size: ESMFold 3B uses 4× more parameters than Gen-UME 750M.

On the more challenging CAMEO 2022 benchmark ([Robin et al., 2021](https://doi.org/10.1002/prot.26213)) with recent protein structures from 2022, ESMFold 3B maintains strong performance with 0.85 TM-score and 65.4% pass rate. DPLM-2 650M achieves 0.70 TM-score with 35.4% pass rate. Gen-UME shows consistent scaling across model sizes: 750M SLQ achieves 0.65 TM-score with 36.2% pass rate, 470M SLQ achieves 0.65 TM-score with 34.6% pass rate, and 90M SLQ achieves 0.63 TM-score with 26.0% pass rate. The 750M model achieves the highest pass rate among Gen-UME models (36.2%), slightly exceeding DPLM-2 650M (35.4%) despite using 32× fewer structure tokens. At the 90M scale, both quantization methods demonstrate competitive performance: SLQ achieves 0.63 TM-score with 26.0% pass rate, while FSQ achieves 0.62 TM-score with 28.3% pass rate. The performance gaps between models are consistent across both benchmarks: ESMFold substantially outperforms unified approaches, while Gen-UME 750M approaches DPLM-2's performance at similar scale. All unified models demonstrate reasonable generalization to newer protein targets not present in the training distribution despite the increased challenge of contemporary structures.

#### Unconditional Generation

Summary of unconditional protein generation performance (averaged across lengths 100-500):

| Model | Tokens | Avg % Pass (RMSD<2.0) | Total Clusters | Avg Diversity % | Notes |
|-------|--------|----------------------|----------------|-----------------|-------|
| **Gen-UME 90M SLQ (ours)** | **256** | **59.0** | **65** | **22.9** | Self-reflection pipeline |
| **Gen-UME 470M SLQ (ours)** | **256** | - | - | - | TODO: Benchmark - Larger model with self-reflection |
| **Gen-UME 750M SLQ sweep (ours)** | **256** | **41.8** | **126** | **50.2** | Sweep-optimized per-length hyperparameters, no self-reflection |
| MultiFlow ([Campbell et al., 2024](https://arxiv.org/abs/2402.04997)) | - | - | - | - | TODO: Benchmark |
| DPLM-2 650M ([Wang et al., 2024](https://arxiv.org/abs/2410.13782)) | 8192 | - | - | - | TODO: Benchmark |

Gen-UME successfully generates novel proteins with 59% average pass rate (RMSD<2.0 Å when validated with ESMFold) using the 90M model with self-reflection, and demonstrates structural diversity across 65 total clusters (22.9% average diversity using Foldseek clustering at TM≥0.5). The 750M model with sweep-optimized hyperparameters achieves 41.8% pass rate and exceptional diversity (126 clusters, 50.2% diversity) without self-reflection, demonstrating that careful per-length hyperparameter tuning can substantially improve generation quality. Detailed per-length results are provided in Supplementary Tables 1 and 4a.

**Key Observations**:
- Strong sequence-structure consistency (TM-scores 0.70-0.85)
- Substantial structural diversity (not memorization)
- Performance decreases for longer proteins (>400 residues)
- Generated sequences are plausible and foldable (pLDDT 0.62-0.75)
- Sweep-optimized hyperparameters dramatically improve performance for 750M model (+41% pass rate, +186% clusters vs default)

#### Self-Reflection Ablation

To assess the impact of self-reflection verification on generation quality, we compare unconditional generation with and without the consistency checking filter:

| Method | Tokens | Avg % Pass (RMSD<2.0) | Avg TM-Score | Avg RMSD (Å) | Total Clusters | Notes |
|--------|--------|----------------------|--------------|--------------|----------------|-------|
| **Gen-UME 90M SLQ (with self-reflection)** | **256** | **59.0** | **0.83** | **2.43** | **65** | Averaged across lengths 100-500 |
| Gen-UME 90M SLQ (no self-reflection) | 256 | 29.8 | 0.62 | 18.08 | 47 | Averaged across lengths 100-500 |
| **Gen-UME 470M SLQ (with self-reflection)** | **256** | - | - | - | - | TODO: Benchmark - Larger model |
| **Gen-UME 470M SLQ (no self-reflection)** | **256** | **45.8** | **0.75** | **8.04** | **54** | Averaged across lengths 100-500 |
| **Gen-UME 750M SLQ (with self-reflection)** | **256** | - | - | - | - | TODO: Benchmark - Larger model |
| Gen-UME 750M SLQ (no self-reflection, default) | 256 | 29.6 | 0.60 | 26.52 | 44 | Default parameters |
| **Gen-UME 750M SLQ (no self-reflection, sweep)** | **256** | **41.8** | **0.70** | **14.42** | **126** | Sweep-optimized per-length hyperparameters |

The self-reflection verification filter dramatically improves sequence-structure consistency, doubling the pass rate (59.0% vs. 29.8%) for the 90M model and reducing RMSD by 7.4× (2.43 Å vs. 18.08 Å). The 470M model without self-reflection achieves 45.8% pass rate with 0.75 TM-score and 8.04 Å RMSD—substantially better than the 90M model without self-reflection (29.8%, 0.62 TM-score, 18.08 Å RMSD), demonstrating that increased model capacity improves generation quality even without consistency filtering. 

**Sweep-optimized hyperparameters for 750M**: Using wandb sweeps to optimize hyperparameters independently for each target length (100-500), the 750M model achieves 41.8% pass rate, 0.70 TM-score, and 14.42 Å RMSD—a dramatic improvement over default parameters (29.6%, 0.60 TM, 26.52 Å). This demonstrates that length-specific hyperparameter tuning can recover much of the performance gap, with cluster diversity increasing from 44 to 126 total clusters (+186%). See Supplementary Tables 4a and 6 for detailed per-length results and optimized parameters.

Notably, the consistency filter also increases structural diversity (65 vs. 47 total clusters for 90M), suggesting that filtering for self-consistency not only improves quality but also enables the model to explore a wider range of valid structural space by rejecting inconsistent generations and retrying. This mirrors the pattern seen in LLM reasoning, where verification steps improve both accuracy and the diversity of valid solutions.

---

## 5. Discussion

### Unified Modeling Advantages

Gen-UME demonstrates that discrete flow matching via CTMC (Campbell et al., 2024) provides a powerful framework for unified protein design:

1. **Single model, multiple tasks**: One 90M parameter model supports unconditional generation, inverse folding, and forward folding through conditional flow matching
2. **Improved consistency**: Joint modeling with CTMC-based flows enforces sequence-structure compatibility through shared representations and masking-based interpolation
3. **Flexible conditioning**: Natural support for partial conditioning in discrete flow framework enables diverse design scenarios
4. **Efficient representation**: Structure tokenization via SLQ (256 tokens) enables efficient transformer processing while maintaining structural fidelity (sub-2Å for proteins, sub-1Å for ligands). Gen-UME achieves competitive performance with 32× fewer structure tokens than DPLM-2 (256 vs 8192), demonstrating efficient discrete representation learning
5. **Independent time sampling**: Different unmasking dynamics for sequences and structures allow the model to learn modality-specific generation strategies

### Unified vs. Specialized Approaches

Gen-UME achieves competitive performance across multiple tasks within unified frameworks at 90M and 470M parameter scales. While specialized models may achieve higher performance on individual tasks through task-specific optimization, Gen-UME's unified approach offers practical advantages:

- Simplified deployment (single model for multiple tasks)
- Consistent interface across generation modes
- Natural support for novel combinations (e.g., partial conditioning)
- Smaller total parameter count than deploying multiple specialized models
- Flexible scaling to balance performance and computational budget

### Self-Reflection as Design Pattern

**Analogy to LLM Reasoning**: Self-reflection in Gen-UME mirrors recent advances in LLM reasoning, where models improve output quality by "thinking through" their answers before finalizing them. Just as language models benefit from chain-of-thought reasoning, self-consistency checks, or verification steps, Gen-UME improves protein design quality by internally verifying sequence-structure compatibility.

**Key Parallels**:
- **LLMs**: Generate → Verify reasoning → Accept or reject answer
- **Gen-UME**: Generate (s₀, x₀) → Verify consistency (forward/inverse fold) → Accept or reject design

**Advantages**: Self-reflection acts as a consistency filter for joint sequence-structure models. By checking bidirectional compatibility through forward and inverse folding, we leverage the model's understanding of both modalities to ensure mutual consistency—the model effectively "checks its work" using its own predictive capabilities. Importantly, the verification outputs are used only for filtering; the original generation is returned if it passes consistency checks.

**Extensions**: This design pattern could extend to:
- Quality-guided sampling (analogous to self-consistency decoding)
- Beam search over candidates (analogous to best-of-N sampling)
- Confidence-based filtering (analogous to uncertainty-aware generation)
- Iterative refinement with multiple verification steps

### Limitations

1. **Length scaling**: Performance decreases for proteins >400 residues
2. **Backbone only**: Does not model side chains
3. **Single chain**: No multi-chain complex support
4. **Functional properties**: No explicit functional constraints
5. **Structure resolution**: Discrete quantization limits fine detail (though both SLQ and FSQ maintain sub-2Å accuracy)

**Note on quantization methods**: We compared SLQ against FSQ as the state-of-the-art discrete quantization method. VQ-VAE was not included as literature shows FSQ consistently outperforms it ([Mentzer et al., 2023](https://arxiv.org/abs/2309.15505)), making it a lower-priority baseline.

### Future Directions

1. **Further scaling**: Beyond 470M parameters (1B+ parameters) with more training data
2. **Longer proteins**: Hierarchical or chunked generation beyond 512 residues
3. **Side chains**: Extend to complete structures with all-atom resolution
4. **Multi-chain**: Protein-protein interfaces and complexes
5. **Functional conditioning**: Optimize for binding, stability, activity
6. **RL fine-tuning**: Optimize for experimental validation metrics

---

## 6. Conclusion

We presented Gen-UME, a unified framework for joint protein sequence and structure generation via discrete flow matching using Continuous Time Markov Chains (CTMC). Building on Campbell et al.'s ([2024](https://arxiv.org/abs/2402.04997)) discrete flow matching framework, Gen-UME introduces novel contributions in structure tokenization (Simple Linear Quantizer), architecture (NeoBERT-based encoder), and training (independent time sampling for multimodal flows). By modeling sequences and structures as discrete tokens and learning their joint distribution through CTMC-based flow matching, Gen-UME achieves competitive performance across three generation tasks within unified models at 90M and 470M parameter scales. We introduce LatentGenerator, a Vision Transformer-based autoencoder for high-fidelity structure tokenization (1.647 ± 0.535 Å RMSD on proteins with SLQ, **0.043 ± 0.011 Å RMSD** on ligands with continuous embeddings), and demonstrate self-reflection verification—a technique analogous to reasoning in LLMs that allows the model to check the consistency of its outputs through bidirectional sequence-structure compatibility tests.

Key results:
- **Inverse folding**: 50.83% AAR / 0.83 TM-score / 2.41 Å RMSD (90M), 55.26% AAR / 0.84 TM-score / 2.34 Å RMSD (470M)
- **Forward folding**: 0.71 TM-score / 4.09 Å RMSD / 42.1% pass rate (90M SLQ), 0.75 TM-score / 3.94 Å RMSD / 57.2% pass rate (470M SLQ)
- **Scaling benefits**: 470M SLQ shows consistent improvements across both tasks (+8.7% AAR on inverse folding, +5.6% TM-score on forward folding, +35.9% pass rate improvement on forward folding from 42.1% to 57.2%)
- **Unconditional generation with self-reflection**: 0.80+ TM-score with 85% pass rate (100aa), 11-25% structural diversity
- **Structure tokenization (protein-only)**: 1.647 ± 0.535 Å RMSD reconstruction on CASP15 proteins
- **Structure tokenization (ligand)**: 0.043 ± 0.011 Å RMSD with continuous embeddings (7× better than best discrete at 0.295 Å) on 30,936 GEOM ligands
- **Structure tokenization (unified protein-ligand)**: 1.873 ± 1.054 Å RMSD on proteins, 0.920 ± 0.236 Å RMSD on ligands—demonstrating joint encoding capability

Our unified approach demonstrates that discrete flow matching via CTMC offers a powerful and flexible framework for computational protein design. Building on Campbell et al.'s ([2024](https://arxiv.org/abs/2402.04997)) foundations, Gen-UME shows that NeoBERT architecture, SLQ tokenization, and independent time sampling enable seamless transitions between generation modes while maintaining high-quality outputs. The self-reflection verification mechanism, inspired by reasoning techniques in LLMs, shows that models can benefit from internally checking the consistency of their predictions through bidirectional sequence-structure compatibility tests before accepting final designs. LatentGenerator's extensibility to ligands demonstrates the potential for unified molecular generation beyond proteins. We provide models at two scales (90M and 470M parameters) to support diverse computational budgets and performance requirements.

---

## Acknowledgments

We thank our institution for computational resources and helpful discussions. We thank the developers of PyTorch, Lightning, ESMFold, and other open-source tools.

---

## References

1. **Axelrod & Gómez-Bombarelli, 2022** - GEOM, energy-annotated molecular conformations for property prediction and molecular generation. *Scientific Data* **9**, 185. https://doi.org/10.1038/s41597-022-01288-4

2. **Campbell et al., 2024** - Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design. *ICML 2024*. https://arxiv.org/abs/2402.04997

3. **Dauparas et al., 2022** - Robust deep learning-based protein sequence design using ProteinMPNN. *Science*. https://www.science.org/doi/10.1126/science.add2187

4. **Hsu et al., 2022** - Learning inverse folding from millions of predicted structures. *bioRxiv*. https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1

5. **Jumper et al., 2021** - Highly accurate protein structure prediction with AlphaFold. *Nature*.

6. **Le Breton et al., 2025** - NeoBERT: A Next-Generation BERT. *arXiv preprint arXiv:2502.19587*. https://arxiv.org/abs/2502.19587

7. **Lin et al., 2023** - Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*. https://www.science.org/doi/10.1126/science.ade2574

8. **Lipman et al., 2023** - Flow matching for generative modeling. *ICLR*.

9. **Mentzer et al., 2024** - Finite Scalar Quantization: VQ-VAE Made Simple. *ICLR 2024*.

10. **Robin et al., 2021** - Continuous Automated Model EvaluatiOn (CAMEO)—Perspectives on the future of fully automated evaluation of structure prediction methods. *Proteins* **89**(12), 1977-1986. https://doi.org/10.1002/prot.26213

11. **Wang et al., 2024** - DPLM-2: A Multimodal Diffusion Protein Language Model. *arXiv preprint arXiv:2410.13782*. https://arxiv.org/abs/2410.13782

12. **Watson et al., 2023** - De novo design of protein structure and function with RFdiffusion. *Nature*.

---

## Appendix

### A. Supplementary Tables

#### Supplementary Table 1: Detailed Unconditional Generation Results (With Self-Reflection)

Per-length performance breakdown for unconditional protein generation **with self-reflection** (100 samples per length):

| Length | Total | RMSD<2Å | Pass% | Clusters | Div.% | Avg TM | Avg RMSD (Å) | Avg pLDDT |
|--------|-------|---------|-------|----------|-------|--------|--------------|-----------|
| 100 | 100 | 86 | 86.0 | 15 | 17.44 | 0.8255 | 1.753 | 0.7210 |
| 200 | 100 | 66 | 66.0 | 17 | 25.76 | 0.8017 | 2.1696 | 0.6683 |
| 300 | 100 | 65 | 65.0 | 19 | 29.23 | 0.8371 | 2.1507 | 0.6957 |
| 400 | 100 | 54 | 54.0 | 7 | 12.96 | 0.8454 | 2.268 | 0.7243 |
| 500 | 100 | 24 | 24.0 | 7 | 29.17 | 0.8187 | 3.8103 | 0.7118 |

**Averaged metrics**: 59.0% pass rate, 0.83 TM-score, 2.43 Å RMSD, 22.9% diversity (across lengths 100-500).

#### Supplementary Table 2: Detailed Unconditional Generation Results (Without Self-Reflection)

Per-length performance breakdown for unconditional protein generation **without self-reflection** (100 samples per length):

| Length | Total | RMSD<2Å | Pass% | Clusters | Div.% | Avg TM | Avg RMSD (Å) | Avg pLDDT |
|--------|-------|---------|-------|----------|-------|--------|--------------|-----------|
| 100 | 100 | 52 | 52.0 | 15 | 28.85 | 0.7114 | 4.011 | 0.6582 |
| 200 | 100 | 44 | 44.0 | 14 | 31.82 | 0.7108 | 6.1808 | 0.6252 |
| 300 | 100 | 29 | 29.0 | 12 | 41.38 | 0.6464 | 15.448 | 0.6170 |
| 400 | 100 | 19 | 19.0 | 4 | 21.05 | 0.5223 | 35.0298 | 0.5989 |
| 500 | 100 | 5 | 5.0 | 2 | 40.0 | 0.5336 | 29.7359 | 0.5797 |

**Averaged metrics**: 29.8% pass rate, 0.62 TM-score, 18.08 Å RMSD, 32.6% diversity.

**Metric Definitions**:
- **Length**: Target protein length in residues
- **Total**: Total structures generated
- **RMSD<2Å**: Structures with RMSD < 2Å between Gen-UME and ESMFold
- **Pass%**: Percentage passing RMSD threshold
- **Clusters**: Unique structural clusters (Foldseek, TM ≥ 0.5)
- **Div.%**: Percentage unique structures
- **Avg TM**: TM-score (Gen-UME vs. ESMFold)
- **Avg RMSD**: RMSD between Gen-UME and ESMFold structures
- **Avg pLDDT**: ESMFold confidence score

**Observations**:
- Self-reflection verification dramatically improves consistency: 2× better pass rate (60.0% vs. 29.8%)
- RMSD improvement is most pronounced for longer proteins (8× better on average)
- No-reflection variant shows higher diversity (32.6% vs. 18.2%), exploring more structural space (includes inconsistent outputs)
- TM-score gap widens for longer proteins, indicating consistency filtering is critical for complex structures
- Both variants generate foldable sequences (pLDDT 0.58-0.73)

#### Supplementary Table 3: Detailed Unconditional Generation Results (470M Without Self-Reflection)

Per-length performance breakdown for unconditional protein generation **with Gen-UME 470M without self-reflection** (100 samples per length):

| Length | Total | RMSD<2Å | Pass% | Clusters | Div.% | Avg TM | Avg RMSD (Å) | Avg pLDDT |
|--------|-------|---------|-------|----------|-------|--------|--------------|-----------|
| 100 | 100 | 68 | 68.0 | 14 | 20.59 | 0.7775 | 2.4655 | 0.7015 |
| 200 | 100 | 44 | 44.0 | 14 | 31.82 | 0.7487 | 4.5745 | 0.6542 |
| 300 | 100 | 55 | 55.0 | 14 | 25.45 | 0.7925 | 4.1185 | 0.7047 |
| 400 | 96 | 22 | 22.92 | 7 | 31.82 | 0.705 | 10.3069 | 0.6672 |
| 500 | 100 | 39 | 39.0 | 5 | 12.82 | 0.7491 | 18.7476 | 0.7103 |

**Averaged metrics**: 45.8% pass rate, 0.75 TM-score, 8.04 Å RMSD, 24.5% diversity.

**Observations**:
- The 470M model without self-reflection substantially outperforms 90M without self-reflection (45.8% vs. 29.8% pass rate)
- 470M achieves better TM-scores (0.75 vs. 0.62) and much lower RMSD (8.04 Å vs. 18.08 Å), demonstrating scaling benefits
- Diversity is similar between models (24.5% vs. 32.6% for 90M), suggesting both explore structural space effectively
- Performance degrades for longer proteins (>300 residues), consistent with the 90M model behavior
- The 470M model maintains reasonable pLDDT scores (0.65-0.71) across all lengths, indicating foldable sequences

#### Supplementary Table 4: Detailed Unconditional Generation Results (750M Without Self-Reflection, Default Parameters)

Per-length performance breakdown for unconditional protein generation **with Gen-UME 750M without self-reflection using default parameters** (100 samples per length):

| Length | Total | RMSD<2Å | Pass% | Clusters | Div.% | Avg TM | Avg RMSD (Å) | Avg pLDDT |
|--------|-------|---------|-------|----------|-------|--------|--------------|-----------|
| 100 | 100 | 78 | 78.0 | 14 | 17.95 | 0.7901 | 2.6766 | 0.7217 |
| 200 | 100 | 30 | 30.0 | 14 | 46.67 | 0.6632 | 16.3038 | 0.6568 |
| 300 | 100 | 30 | 30.0 | 12 | 40.00 | 0.7078 | 9.3493 | 0.6696 |
| 400 | 100 | 8 | 8.0 | 2 | 25.00 | 0.4667 | 47.4624 | 0.6465 |
| 500 | 100 | 2 | 2.0 | 2 | 100.00 | 0.3905 | 56.7924 | 0.6224 |

**Averaged metrics**: 29.6% pass rate, 0.60 TM-score, 26.52 Å RMSD, 44 total clusters.

**Observations**:
- The 750M model with default parameters shows lower pass rate (29.6%) than 470M (45.8%) and 90M with self-reflection (59.0%)
- Performance at length 100 is strong (78% pass rate, 0.79 TM-score) but degrades dramatically for longer proteins
- Length 500 shows very poor performance (2% pass rate, 0.39 TM-score, 56.8 Å RMSD), indicating severe challenges with long-range dependencies
- These results motivated hyperparameter optimization via wandb sweeps (see Supplementary Table 4a)

#### Supplementary Table 4a: Detailed Unconditional Generation Results (750M Sweep-Optimized, Without Self-Reflection)

Per-length performance breakdown for unconditional protein generation **with Gen-UME 750M without self-reflection using sweep-optimized hyperparameters** (100 samples per length). Hyperparameters were optimized via wandb sweeps for each target length independently (see Supplementary Table 6 for optimized parameters).

| Length | Total | RMSD<2Å | Pass% | Clusters | Div.% | Avg TM | Avg RMSD (Å) | Avg pLDDT |
|--------|-------|---------|-------|----------|-------|--------|--------------|-----------|
| 100 | 100 | 85 | 85.0 | 72 | 84.71 | 0.8070 | 2.0880 | 0.7482 |
| 200 | 100 | 46 | 46.0 | 26 | 56.52 | 0.7460 | 4.9913 | 0.6536 |
| 300 | 100 | 35 | 35.0 | 17 | 48.57 | 0.7205 | 9.5506 | 0.6617 |
| 400 | 100 | 33 | 33.0 | 7 | 21.21 | 0.6962 | 22.2082 | 0.6852 |
| 500 | 100 | 10 | 10.0 | 4 | 40.00 | 0.5328 | 33.2670 | 0.6153 |

**Averaged metrics**: 41.8% pass rate, 0.70 TM-score, 14.42 Å RMSD, 126 total clusters, 50.20% average diversity.

**Comparison with Default Parameters (Supplementary Table 4)**:
- **Pass rate improvement**: 29.6% → 41.8% (+41.2% relative improvement)
- **TM-score improvement**: 0.60 → 0.70 (+16.7% relative improvement)
- **RMSD improvement**: 26.52 Å → 14.42 Å (45.6% reduction)
- **Cluster diversity**: 44 → 126 total clusters (+186% increase)

**Per-length improvements from sweep optimization**:
- **L100**: 78% → 85% pass (+9%), 0.79 → 0.81 TM (+2.5%), 2.68 → 2.09 Å RMSD (-22%), 14 → 72 clusters (+414%)
- **L200**: 30% → 46% pass (+53%), 0.66 → 0.75 TM (+13.6%), 16.30 → 4.99 Å RMSD (-69%), 14 → 26 clusters (+86%)
- **L300**: 30% → 35% pass (+17%), 0.71 → 0.72 TM (+1.4%), 9.35 → 9.55 Å RMSD (~same), 12 → 17 clusters (+42%)
- **L400**: 8% → 33% pass (+313%), 0.47 → 0.70 TM (+49%), 47.46 → 22.21 Å RMSD (-53%), 2 → 7 clusters (+250%)
- **L500**: 2% → 10% pass (+400%), 0.39 → 0.53 TM (+36%), 56.79 → 33.27 Å RMSD (-41%), 2 → 4 clusters (+100%)

**Observations**:
- Sweep-optimized hyperparameters dramatically improve performance, especially for longer proteins (L400, L500)
- The most significant gains are in pass rate for L400 (+313%) and L500 (+400%), demonstrating that length-specific tuning is critical
- Structural diversity improves substantially with sweep optimization, with L100 achieving 72 unique clusters (84.7% diversity)
- These results demonstrate that careful hyperparameter selection per target length can significantly improve unconditional generation quality
- Even with sweep optimization, performance degrades for longer proteins, suggesting architectural improvements or self-reflection are needed for L500+

#### Supplementary Table 5: Generation Parameters (90M and 470M Models)

Optimized generation parameters used for Gen-UME 90M and 470M models across different generation modes (see Supplementary Table 6 for 750M length-specific parameters):

| Mode | Model | N (steps) | τ_seq | τ_struct | stochasticity_seq | stochasticity_struc | Schedule (seq) | Schedule (struc) | Notes |
|------|-------|-----------|-------|----------|-------------------|---------------------|----------------|------------------|-------|
| Unconditional | 90M | 1000 | 0.46 | 0.36 | 30 | 70 | - | - | Default parameters |
| Inverse Folding | 90M | 200 | 0.16 | 1.0 | 20 | 10 | - | - | Default parameters |
| Forward Folding | 90M | 100 | 0.30 | 0.11 | 10 | 30 | - | - | Default parameters |
| **Unconditional** | **470M** | **400** | **0.273** | **0.316** | **20** | **60** | **Log** | **Power** | Optimized for 470M model |
| **Forward Folding** | **470M** | **200** | **0.361** | **0.220** | **1** | **20** | - | - | Optimized for 470M model |

**Parameter Descriptions**:
- **N (steps)**: Number of discrete time steps in the flow matching process
- **τ_seq**: Temperature for sequence token sampling (controls diversity)
- **τ_struct**: Temperature for structure token sampling (controls diversity)
- **stochasticity_seq**: Stochasticity level for sequence transitions (0 to N)
- **stochasticity_struc**: Stochasticity level for structure transitions (0 to N)
- **Schedule (seq)**: Inference schedule for sequence unmasking (Log = logarithmic, Power = power-law, - = linear)
- **Schedule (struc)**: Inference schedule for structure unmasking

**Notes**:
- The 470M unconditional generation parameters use custom inference schedules for improved generation quality
  - **LogInferenceSchedule** for sequences: Logarithmic unmasking provides gradual refinement of amino acid sequences
  - **PowerInferenceSchedule** for structures: Power-law unmasking allows hierarchical structure generation
  - 400 steps (vs 1000 for 90M) with adjusted stochasticity maintains quality while improving efficiency
- The 470M forward folding parameters are optimized for accuracy on the Campbell et al. benchmark
  - Higher number of steps (200 vs 100) allows for more refined structure generation
  - Lower stochasticity values (1, 20) provide more deterministic sampling for better accuracy
- Temperature values are tuned to balance diversity and accuracy across both generation modes

#### Supplementary Table 6: Sweep-Optimized Generation Parameters for 750M Model (Per-Length)

Optimized generation parameters for Gen-UME 750M unconditional generation, obtained via wandb sweeps for each target length independently:

| Length | N (steps) | τ_seq | τ_struct | stochasticity_seq | stochasticity_struc | Schedule (seq) | Schedule (struc) |
|--------|-----------|-------|----------|-------------------|---------------------|----------------|------------------|
| 100 | 800 | 0.220 | 0.534 | 70 | 20 | Log | Linear |
| 200 | 900 | 0.515 | 0.290 | 50 | 90 | Log | Power |
| 300 | 600 | 0.472 | 0.272 | 20 | 60 | Log | Power |
| 400 | 700 | 0.151 | 0.205 | 10 | 70 | Log | Linear |
| 500 | 500 | 0.138 | 0.497 | 40 | 50 | Power | Linear |

**Hyperparameter Sweep Configuration**:
- **Search method**: Bayesian optimization via wandb sweeps
- **Objective**: Maximize pass rate (RMSD < 2.0 Å) while maintaining TM-score
- **Search space**: nsteps ∈ [100, 1000], temperatures ∈ [0.1, 1.0], stochasticity ∈ [10, 100], schedules ∈ {Linear, Log, Power}
- **Samples per trial**: 100 proteins per hyperparameter configuration

**Observations**:
- **Temperature patterns**: Lower sequence temperatures (τ_seq) for longer proteins (0.22 → 0.14 for L100 → L500), suggesting more deterministic sequence generation is beneficial for longer chains
- **Stochasticity patterns**: Higher structure stochasticity (stochasticity_struc) for L200-L400 (60-90), lower for L100 and L500 (20-50)
- **Inference schedules**: Log schedule for sequences works well across most lengths; Power schedule for structures benefits L200-L300
- **Step count**: Longer proteins don't necessarily require more steps (L500 uses 500 steps vs. L200 uses 900 steps)
- **Length-specific tuning is critical**: Default parameters severely underperform compared to sweep-optimized parameters (see Supplementary Table 4 vs 4a)

### B. Data Availability

**Reference Data**: All experimental results reported in this paper are sourced from the `gen_ume_paper_data/` directory, which contains:
- Unconditional generation results (with and without self-reflection)
- LatentGenerator reconstruction results (CASP15)
- Inverse folding benchmarks
- Forward folding benchmarks

See `gen_ume_paper_data/README.md` for complete data provenance and calculation methods.

### C. Additional Implementation Details

**Repository**: [Anonymous - will be released upon publication]

**Model checkpoint**: [Anonymous - will be released upon publication]

**Key modules**:
- `lobster.model.gen_ume`: Model implementation
- `lobster.cmdline.generate`: Generation interface
- `lobster.model.latent_generator`: Structure encoder-decoder

---

**License**: Apache 2.0  
**Contact**: Anonymous

