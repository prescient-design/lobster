from enum import Enum


class PEERTaskCategory(str, Enum):
    FUNCTION_PREDICTION = "function_prediction"
    LOCALIZATION_PREDICTION = "localization_prediction"
    PROTEIN_LIGAND_INTERACTION = "protein_ligand_interaction"
    PROTEIN_PROTEIN_INTERACTION = "protein_protein_interaction"
    STRUCTURE_PREDICTION = "structure_prediction"


class PEERTask(str, Enum):
    # Function prediction tasks
    AAV = "aav"
    BETALACTAMASE = "betalactamase"
    FLUORESCENCE = "fluorescence"
    GB1 = "gb1"
    SOLUBILITY = "solubility"
    STABILITY = "stability"
    THERMOSTABILITY = "thermostability"

    # Localization prediction tasks
    BINARY_LOCALIZATION = "binarylocalization"
    SUBCELLULAR_LOCALIZATION = "subcellularlocalization"

    # Protein-ligand interaction tasks
    BINDINGDB = "bindingdb"
    PDBBIND = "pdbbind"

    # Protein-protein interaction tasks
    HUMANPPI = "humanppi"
    PPIAFFINITY = "ppiaffinity"
    YEASTPPI = "yeastppi"
    # Structure prediction tasks
    FOLD = "fold"
    PROTEINNET = "proteinnet"
    SECONDARY_STRUCTURE = "secondarystructure"


# Map tasks to their categories
PEER_TASK_CATEGORIES = {
    # Function prediction
    PEERTask.AAV: PEERTaskCategory.FUNCTION_PREDICTION,
    PEERTask.BETALACTAMASE: PEERTaskCategory.FUNCTION_PREDICTION,
    PEERTask.FLUORESCENCE: PEERTaskCategory.FUNCTION_PREDICTION,
    PEERTask.GB1: PEERTaskCategory.FUNCTION_PREDICTION,
    PEERTask.SOLUBILITY: PEERTaskCategory.FUNCTION_PREDICTION,
    PEERTask.STABILITY: PEERTaskCategory.FUNCTION_PREDICTION,
    PEERTask.THERMOSTABILITY: PEERTaskCategory.FUNCTION_PREDICTION,
    # Localization prediction
    PEERTask.BINARY_LOCALIZATION: PEERTaskCategory.LOCALIZATION_PREDICTION,
    PEERTask.SUBCELLULAR_LOCALIZATION: PEERTaskCategory.LOCALIZATION_PREDICTION,
    # Protein-ligand interaction
    PEERTask.BINDINGDB: PEERTaskCategory.PROTEIN_LIGAND_INTERACTION,
    PEERTask.PDBBIND: PEERTaskCategory.PROTEIN_LIGAND_INTERACTION,
    # Protein-protein interaction
    PEERTask.PPIAFFINITY: PEERTaskCategory.PROTEIN_PROTEIN_INTERACTION,
    PEERTask.HUMANPPI: PEERTaskCategory.PROTEIN_PROTEIN_INTERACTION,
    PEERTask.YEASTPPI: PEERTaskCategory.PROTEIN_PROTEIN_INTERACTION,
    # Structure prediction
    PEERTask.FOLD: PEERTaskCategory.STRUCTURE_PREDICTION,
    PEERTask.PROTEINNET: PEERTaskCategory.STRUCTURE_PREDICTION,
    PEERTask.SECONDARY_STRUCTURE: PEERTaskCategory.STRUCTURE_PREDICTION,
}


# Define task types and num_classes for each task
# Using more specific task types: "regression", "binary", "multiclass", "multilabel"
# Based on the paper and error logs
PEER_TASKS = {
    # Function Prediction - All regression except Solubility which is binary classification
    PEERTask.GB1: ("regression", None),  # Protein-wise Regression
    PEERTask.AAV: ("regression", None),  # Protein-wise Regression
    PEERTask.THERMOSTABILITY: ("regression", None),  # Protein-wise Regression
    PEERTask.FLUORESCENCE: ("regression", None),  # Protein-wise Regression
    PEERTask.STABILITY: ("regression", None),  # Protein-wise Regression
    PEERTask.BETALACTAMASE: ("regression", None),  # Protein-wise Regression
    PEERTask.SOLUBILITY: ("binary", 2),  # Protein-wise Binary Classification (soluble or not)
    # Localization Prediction - All classification tasks
    PEERTask.SUBCELLULAR_LOCALIZATION: ("multiclass", 10),  # Protein-wise Multiclass (10 classes)
    PEERTask.BINARY_LOCALIZATION: ("binary", 2),  # Protein-wise Binary Classification (membrane-bound or soluble)
    # Structure Prediction
    PEERTask.PROTEINNET: ("binary", 2),  # Residue-pair Binary Classification (Contact prediction)
    PEERTask.FOLD: ("multiclass", 1195),  # Protein-wise Multiclass Classification (1195 fold classes as per paper)
    PEERTask.SECONDARY_STRUCTURE: (
        "multiclass",
        3,
    ),  # Residue-wise Multiclass Classification (3 classes: coil, strand, helix)
    # Protein-Protein Interaction
    PEERTask.YEASTPPI: ("binary", 2),  # Protein-pair Binary Classification (interact or not)
    PEERTask.HUMANPPI: ("binary", 2),  # Protein-pair Binary Classification (interact or not)
    PEERTask.PPIAFFINITY: ("regression", None),  # Protein-pair Regression
    # Protein-Ligand Interaction
    PEERTask.PDBBIND: ("regression", None),  # Protein-ligand Regression
    PEERTask.BINDINGDB: ("regression", None),  # Protein-ligand Regression
}

# Define available splits for each task
PEER_TASK_SPLITS = {
    # Standard tasks with train/valid/test
    PEERTask.AAV: ["train", "valid", "test"],
    PEERTask.BETALACTAMASE: ["train", "valid", "test"],
    PEERTask.FLUORESCENCE: ["train", "valid", "test"],
    PEERTask.GB1: ["train", "valid", "test"],
    PEERTask.SOLUBILITY: ["train", "valid", "test"],
    PEERTask.STABILITY: ["train", "valid", "test"],
    PEERTask.THERMOSTABILITY: ["train", "valid", "test"],
    PEERTask.BINARY_LOCALIZATION: ["train", "valid", "test"],
    PEERTask.SUBCELLULAR_LOCALIZATION: ["train", "valid", "test"],
    PEERTask.PDBBIND: ["train", "valid", "test"],
    PEERTask.PPIAFFINITY: ["train", "valid", "test"],
    PEERTask.PROTEINNET: ["train", "valid", "test"],
    # Special cases
    PEERTask.BINDINGDB: ["train", "valid", "random_test", "holdout_test"],
    PEERTask.HUMANPPI: ["train", "valid", "test", "cross_species_test"],
    PEERTask.YEASTPPI: ["train", "valid", "test", "cross_species_test"],
    PEERTask.FOLD: ["train", "valid", "test_family_holdout", "test_fold_holdout", "test_superfamily_holdout"],
    PEERTask.SECONDARY_STRUCTURE: ["train", "valid", "casp12", "cb513", "ts115"],
}

# Define expected column names for each task
# Format: (input_columns, target_columns)
PEER_TASK_COLUMNS = {
    # Standard protein tasks with single sequence input and target
    PEERTask.BETALACTAMASE: (["protein_sequence"], ["scaled_effect1"]),
    PEERTask.BINARY_LOCALIZATION: (["protein_sequence"], ["localization"]),
    PEERTask.FLUORESCENCE: (["protein_sequence"], ["log_fluorescence"]),
    PEERTask.GB1: (["protein_sequence"], ["target"]),
    PEERTask.SOLUBILITY: (["protein_sequence"], ["solubility"]),
    PEERTask.STABILITY: (["protein_sequence"], ["stability_score"]),
    PEERTask.THERMOSTABILITY: (["protein_sequence"], ["target"]),
    PEERTask.AAV: (["protein_sequence"], ["target"]),
    PEERTask.SUBCELLULAR_LOCALIZATION: (["protein_sequence"], ["localization"]),
    PEERTask.FOLD: (["protein_sequence"], ["fold_label"]),
    # Tasks with multiple targets
    PEERTask.SECONDARY_STRUCTURE: (["protein_sequence"], ["ss3"]),
    PEERTask.PROTEINNET: (["protein_sequence"], ["tertiary", "valid_mask"]),
    # Protein-ligand interaction tasks
    PEERTask.BINDINGDB: (["protein_sequence", "ligand_smiles"], ["affinity"]),
    PEERTask.PDBBIND: (["protein_sequence", "ligand_smiles"], ["affinity"]),
    # Protein-protein interaction tasks
    PEERTask.PPIAFFINITY: (["protein1_sequence", "protein2_sequence"], ["interaction"]),
    PEERTask.YEASTPPI: (["protein1_sequence", "protein2_sequence"], ["interaction"]),
    PEERTask.HUMANPPI: (["protein1_sequence", "protein2_sequence"], ["interaction"]),
}
