import torch
from lobster.model.ume_previous import UMEPrevious
from lobster.model import UME

sequence_length = 16
batch = {
    "input_ids": torch.randint(0, 400, (4,2,sequence_length)), # shape  (batch size, num views, sequence length)
    "attention_mask": torch.ones((4,2,sequence_length)),
    "modality": [("amino_acid", "amino_acid", "amino_acid"), ("amino_acid", "amino_acid", "amino_acid")],
}

for temperature in [0.07, 0.1, 0.2, 0.5, 1.0]:
    model = UMEPrevious(contrastive_loss_weight=1.0, contrastive_temperature=temperature)
    loss = model.training_step(batch, 0)
    print(f"Temperature: {temperature}, Loss: {loss}")

for temperature in [0.07, 0.1, 0.2, 0.5, 1.0]:
    model = UME(contrastive_loss_type="clip", contrastive_loss_weight=1.0, contrastive_temperature=temperature)
    loss = model.training_step(batch, 0)
    print(f"Temperature: {temperature}, Loss: {loss}")
