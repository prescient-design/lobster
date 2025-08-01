import torch
from transformers import AutoModel

if __name__ == "__main__":
    # Load the model
    model = AutoModel.from_pretrained("karina-zadorozhny/ume-mini-base-12M-test", trust_remote_code=True)

    # Example usage
    input_ids = torch.randint(0, 100, (1, 1, 10))
    attention_mask = torch.ones_like(input_ids)

    print(f"Shape of input_ids: {input_ids.shape}")
    print(f"Shape of attention_mask: {attention_mask.shape}")

    output = model(input_ids, attention_mask)
    print(f"Shape of output: {output.shape}")
