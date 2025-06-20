#!/usr/bin/env python3
"""
Script to generate UME (Universal Molecular Encoder) architecture diagram.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch


def create_ume_architecture_diagram():
    """Create a comprehensive UME architecture diagram."""

    # Modern color palette
    colors = {
        "input": "#667EEA",  # Modern purple-blue
        "input_light": "#E8EDFF",  # Light version for backgrounds
        "tokenizer": "#F093FB",  # Modern pink-purple
        "tokenizer_light": "#FDEBFF",  # Light version
        "backbone": "#4FACFE",  # Modern bright blue
        "backbone_light": "#E1F3FF",  # Light version
        "output": "#43E97B",  # Modern bright green
        "output_light": "#E8FFF0",  # Light version
        "arrow": "#2D3748",  # Modern dark gray
        "text": "#2D3748",  # Modern dark gray
        "special": "#FF6B6B",  # Modern coral red
        "special_light": "#FFE8E8",  # Light version
        "accent": "#FEC84B",  # Modern yellow accent
        "background": "#F7FAFC",  # Light gray background
    }

    # Create figure with high DPI for clarity
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor(colors["background"])

    # Title
    ax.text(
        8,
        11.5,
        "UME (Universal Molecular Encoder) Architecture",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=colors["text"],
    )

    # Input Modalities (Top Row)
    input_boxes = []
    input_labels = [
        'Amino Acid\nSequences\n"MKTVRQ..."',
        'SMILES\nStrings\n"CC(=O)OC1..."',
        'Nucleotide\nSequences\n"ATGCAT..."',
        "3D Coordinates\n[x,y,z]",
    ]

    for i, label in enumerate(input_labels):
        x = 2 + i * 3
        box = FancyBboxPatch(
            (x - 0.8, 9.5),
            1.6,
            1.2,
            boxstyle="round,pad=0.1",
            facecolor=colors["input_light"],
            edgecolor=colors["input"],
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x, 10.1, label, ha="center", va="center", fontsize=10, fontweight="bold", color=colors["text"])
        input_boxes.append((x, 9.5))

    # Tokenization Layer
    tokenizer_y = 7.8
    tokenizer_box = FancyBboxPatch(
        (1, tokenizer_y - 0.4),
        14,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor=colors["tokenizer_light"],
        edgecolor=colors["tokenizer"],
        linewidth=2,
    )
    ax.add_patch(tokenizer_box)
    ax.text(
        8,
        tokenizer_y,
        "Unified Tokenization System",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=colors["text"],
    )

    # Tokenizer details (moved below box to avoid overlap)
    tokenizer_details = [
        "Modality-Specific CLS: <cls_amino_acid>, <cls_smiles>, <cls_nucleotide>",
        "Shared Special Tokens: <cls>, <eos>, <mask>, <pad>, <unk>, <sep>",
        "Unified Vocabulary: ~1,280 tokens (aligned across modalities)",
    ]

    for i, detail in enumerate(tokenizer_details):
        ax.text(
            8,
            tokenizer_y - 0.7 - i * 0.25,
            detail,
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            color=colors["text"],
        )

    # Arrows from inputs to tokenizer (improved positioning)
    for x, y in input_boxes:
        arrow = ConnectionPatch(
            (x, y),
            (x, tokenizer_y + 0.4),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=15,
            fc=colors["arrow"],
            ec=colors["arrow"],
            linewidth=2,
        )
        ax.add_artist(arrow)

    # FlexBERT Backbone
    backbone_y = 5.2
    backbone_box = FancyBboxPatch(
        (3, backbone_y - 1),
        10,
        2,
        boxstyle="round,pad=0.15",
        facecolor=colors["backbone_light"],
        edgecolor=colors["backbone"],
        linewidth=3,
    )
    ax.add_patch(backbone_box)

    ax.text(
        8,
        backbone_y + 0.5,
        "FlexBERT Transformer Backbone",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=colors["text"],
    )

    # Backbone components
    backbone_components = [
        "• Shared Parameters Across All Modalities",
        "• Multi-Head Self-Attention with Flash Attention 2",
        "• Model Sizes: UME_mini (12M) → UME_large (740M)",
        "• Embedding Dimension: 768+ (model-dependent)",
        "• Max Sequence Length: 512-8192 tokens",
    ]

    for i, comp in enumerate(backbone_components):
        ax.text(8, backbone_y - 0.1 - i * 0.2, comp, ha="center", va="center", fontsize=10, color=colors["text"])

    # Arrow from tokenizer to backbone (better positioning)
    arrow = ConnectionPatch(
        (8, tokenizer_y - 1.15),
        (8, backbone_y + 1),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc=colors["arrow"],
        ec=colors["arrow"],
        linewidth=2,
    )
    ax.add_artist(arrow)

    # Training Objectives (Side boxes with better positioning)
    # MLM Box
    mlm_box = FancyBboxPatch(
        (0.5, 4.2),
        2,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["special_light"],
        edgecolor=colors["special"],
        linewidth=2,
    )
    ax.add_patch(mlm_box)
    ax.text(
        1.5,
        5.5,
        "Masked Language\nModeling (MLM)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=colors["text"],
    )
    ax.text(
        1.5,
        4.8,
        "• 25% Token Masking\n• Cross-Entropy Loss\n• Per-Modality Metrics",
        ha="center",
        va="center",
        fontsize=9,
        color=colors["text"],
    )

    # Contrastive Box
    contrast_box = FancyBboxPatch(
        (13.5, 4.2),
        2,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["special_light"],
        edgecolor=colors["special"],
        linewidth=2,
    )
    ax.add_patch(contrast_box)
    ax.text(
        14.5,
        5.5,
        "Contrastive\nLearning",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=colors["text"],
    )
    ax.text(
        14.5,
        4.8,
        "• InfoNCE Loss\n• Symile Loss\n• DisCo Loss",
        ha="center",
        va="center",
        fontsize=9,
        color=colors["text"],
    )

    # Arrows to training objectives (better positioning)
    arrow_mlm = ConnectionPatch(
        (3, backbone_y),
        (2.5, 5.2),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=12,
        fc=colors["arrow"],
        ec=colors["arrow"],
        linewidth=1.5,
    )
    ax.add_artist(arrow_mlm)

    arrow_contrast = ConnectionPatch(
        (13, backbone_y),
        (13.5, 5.2),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=12,
        fc=colors["arrow"],
        ec=colors["arrow"],
        linewidth=1.5,
    )
    ax.add_artist(arrow_contrast)

    # Output Embeddings
    output_y = 2.8
    output_box = FancyBboxPatch(
        (4, output_y - 0.5),
        8,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["output_light"],
        edgecolor=colors["output"],
        linewidth=2,
    )
    ax.add_patch(output_box)
    ax.text(
        8,
        output_y,
        "Unified Embedding Space",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=colors["text"],
    )

    # Arrow from backbone to output (better positioning)
    arrow = ConnectionPatch(
        (8, backbone_y - 1),
        (8, output_y + 0.5),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc=colors["arrow"],
        ec=colors["arrow"],
        linewidth=2,
    )
    ax.add_artist(arrow)

    # Output details (moved above the output box to avoid overlap)
    output_details = [
        "Sequence-Level: (batch_size, embedding_dim)",
        "Token-Level: (batch_size, seq_len, embedding_dim)",
        "Cross-Modal Compatibility: All modalities in same space",
    ]

    for i, detail in enumerate(output_details):
        ax.text(
            8,
            output_y - 0.75 - i * 0.15,
            detail,
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            color=colors["text"],
        )

    # Dataset Information (Bottom)
    dataset_y = 1.4
    dataset_box = FancyBboxPatch(
        (1, dataset_y - 0.3),
        14,
        0.6,
        boxstyle="round,pad=0.1",
        facecolor=colors["accent"],
        edgecolor=colors["text"],
        linewidth=1.5,
        alpha=0.3,
    )
    ax.add_patch(dataset_box)
    ax.text(
        8,
        dataset_y + 0.1,
        "Training Datasets",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=colors["text"],
    )

    datasets = [
        "M320M (19.4M SMILES) • CALM (7.9M nucleotides) • AMPLIFY (448M amino acids)",
        "Pinder (267K 3D coords) • OpenGenome2 (28.8B nucleotides) • ZINC (1.54B SMILES)",
    ]

    for i, dataset in enumerate(datasets):
        ax.text(8, dataset_y - 0.15 - i * 0.18, dataset, ha="center", va="center", fontsize=9, color=colors["text"])

    # Add legend/key features box
    legend_box = FancyBboxPatch(
        (0.5, 0.2),
        15,
        0.4,
        boxstyle="round,pad=0.1",
        facecolor=colors["background"],
        edgecolor=colors["accent"],
        linewidth=2,
    )
    ax.add_patch(legend_box)
    ax.text(
        8,
        0.4,
        "Key Features: Unified vocabulary • Modality-aware tokenization • Shared transformer parameters • Intra- and inter-modality embedding alignment",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=colors["text"],
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Create the diagram
    fig = create_ume_architecture_diagram()

    # Save the diagram
    output_path = "ume_architecture_diagram.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#F7FAFC", edgecolor="none")

    print(f"UME architecture diagram saved as: {output_path}")

    # Show the diagram
    plt.show()
