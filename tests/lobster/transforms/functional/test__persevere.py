import edlib
import numpy as np
import torch
from lobster.transforms.functional._persevere import (
    get_compatible_amino_acids,
    highlight_mutations,
    hydro_charge_transition_matrix,
    persevere,
)


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test__persevere():
    set_random_seeds()
    fv_heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAITWNSGHIDYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAREGYYGSSHWYFDVWGQGTLVTVSS"
    fv_light = (
        "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYDASSRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPPTFGQGTKVEIK"
    )

    designs = persevere(
        fv_heavy,
        total_num_designs=4,
        redesign_regions=("H1", "H2", "H3"),
        mutations_per_redesign_region=2,
    )

    highlighted_sequences = highlight_mutations(fv_heavy, designs)
    print("\n")
    print(f"Seed: {fv_heavy}")
    for _, (seq, indicator) in enumerate(highlighted_sequences):
        print(f"      {seq}")
        print(f"      {indicator}")
        print()

    for d in designs:
        assert len(d) == len(fv_heavy)
        edit_distance = edlib.align(fv_heavy, d, mode="NW", task="path")["editDistance"]
        assert edit_distance == 6
        print(edlib.align(fv_heavy, d, mode="NW", task="path")["editDistance"])

    # designs = persevere(
    #     fv_heavy,
    #     total_num_designs=4,
    #     redesign_regions=("H3",),
    #     mutations_per_redesign_region=4,
    # )

    # highlighted_sequences = highlight_mutations(fv_heavy, designs)
    # print(f"Seed: {fv_heavy}")
    # for _, (seq, indicator) in enumerate(highlighted_sequences):
    #     print(f"      {seq}")
    #     print(f"      {indicator}")
    #     print()

    # for d in designs:
    #     assert len(d) == len(fv_heavy)
    #     edit_distance = edlib.align(fv_heavy, d, mode="NW", task="path")["editDistance"]
    #     assert edit_distance == 4
    #     print(edlib.align(fv_heavy, d, mode="NW", task="path")["editDistance"])

    designs = persevere(
        fv_light,
        total_num_designs=4,
        redesign_regions=("L1", "L2"),
        mutations_per_redesign_region=2,
    )

    for d in designs:
        assert len(d) == len(fv_light)
        edit_distance = edlib.align(fv_light, d, mode="NW", task="path")["editDistance"]
        assert edit_distance == 4
        print(edlib.align(fv_light, d, mode="NW", task="path")["editDistance"])

    transition_matrix, amino_acids = hydro_charge_transition_matrix(delta_gravy=2.0)
    compatible_aas = get_compatible_amino_acids("A", transition_matrix, amino_acids)
    assert compatible_aas == ["R", "N", "D", "E", "Q", "G", "H", "I", "K", "P", "S", "T", "W", "Y", "V"]

    compatible_aas = get_compatible_amino_acids("W", transition_matrix, amino_acids)
    assert compatible_aas == ["A", "R", "N", "D", "C", "E", "Q", "H", "I", "L", "K", "M", "F", "V"]


# def test__mutation_mixer():
#     fv_heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAITWNSGHIDYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAREGYYGSSHWYFDVWGQGTLVTVSS"
#     fv_light = (
#         "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYDASSRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPPTFGQGTKVEIK"
#     )

#     designs = mutation_mixer(fv_heavy, total_num_designs=5, chain="H", mutations_per_design=4)

#     assert len(designs) == 5

#     designs = [d.replace("-", "") for d in designs]

#     edit_distances = [edlib.align(fv_heavy, d, mode="NW", task="path")["editDistance"] for d in designs]
#     print(edit_distances)
#     assert all([ed <= 4 for ed in edit_distances])

#     highlighted_sequences = highlight_mutations(fv_heavy, designs)
#     print(f"Seed: {fv_heavy}")
#     for i, (seq, indicator) in enumerate(highlighted_sequences):
#         print(f"      {seq}")
#         print(f"      {indicator}")
#         print()

#     designs = mutation_mixer(fv_light, total_num_designs=5, chain="H", mutations_per_design=4)

#     assert len(designs) == 5

#     designs = [d.replace("-", "") for d in designs]

#     edit_distances = [edlib.align(fv_light, d, mode="NW", task="path")["editDistance"] for d in designs]
#     print(edit_distances)
#     assert all([ed <= 4 for ed in edit_distances])

#     highlighted_sequences = highlight_mutations(fv_light, designs)
#     print(f"Seed: {fv_light}")
#     for i, (seq, indicator) in enumerate(highlighted_sequences):
#         print(f"      {seq}")
#         print(f"      {indicator}")
#         print()
