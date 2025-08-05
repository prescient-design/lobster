import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

if __name__ == "__main__":
    # Load the model
    config = AutoConfig.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True)
    model = AutoModel.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True, config=config)
    
    # Load the amino acid tokenizer from its specific subfolder
    tokenizer = AutoTokenizer.from_pretrained(
        "karina-zadorozhny/ume", 
        subfolder="tokenizer_amino_acid",  
        trust_remote_code=True
    )
    
    # Example amino acid sequence
    sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"] * 2 
    
    # Tokenize the sequence
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    

    print(f"Original sequences: {sequences}")
    print(f"Tokenized sequences: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    print(f"Shape of input_ids: {input_ids.shape}")
    print(f"Shape of attention_mask: {attention_mask.shape}")

    # Run inference
    with torch.no_grad():
        output = model(input_ids.unsqueeze(1), attention_mask.unsqueeze(1))
        
    print(f"Shape of output: {output.shape}")
