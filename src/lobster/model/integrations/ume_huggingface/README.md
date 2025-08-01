# UME HuggingFace Integration

## Step 1: Export UME to ONNX and Upload to Hub
```bash
uv run src/lobster/model/integrations/ume_hugginface/export_to_onnx.py
uv run src/upload_to_hub.py
```

## Step 2: Upload tokenizer vocabularies 
```bash
uv run src/lobster/model/integrations/ume_huggingface/upload_vocabs.py
```

## Step 3: Upload UME config, models, and tokenization code
```bash
uv run src/lobster/model/integrations/ume_huggingface/register_model.py
```

## Step 4: Verify that everything worked
```bash
uv run examples/ume_hf_example.py
```

## Step 5: Verify that we get the same outputs
```bash
uv run examples/ume_hf_sanity_check.py
```


