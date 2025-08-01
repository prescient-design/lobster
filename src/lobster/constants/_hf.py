import importlib.resources

HF_UME_REPO_ID = "karina-zadorozhny/ume-mini-base-12M-test"
HF_UME_MODEL_FILEPATH = str(importlib.resources.files("lobster") / "model/integrations/ume_huggingface/model")
