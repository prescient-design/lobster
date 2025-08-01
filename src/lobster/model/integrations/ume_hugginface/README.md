# UME HuggingFace Integration

## Step 1: Export UME to ONNX and Upload to Hub
```bash
uv run src/lobster/model/integrations/ume_hugginface/export_to_onnx.py
```


## Step 2: Upload UME modeling and tokenization code to Hub
```bash
uv run src/lobster/model/integrations/ume_hugginface/register_model.py
```