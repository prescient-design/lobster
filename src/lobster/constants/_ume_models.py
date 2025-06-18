# TODO: currently, these will work for internal users
# Support for external users will be added soon

UME_PRETRAINED_CHECKPOINTS = {
    "ume-mini-base-12M": "s3://prescient-lobster/ume/runs/2025-06-17T13-45-59/epoch=0-step=2500-val_loss=0.8203.ckpt",
    "ume-small-base-90M": None,  # Add when available
    "ume-medium-base-480M": "s3://prescient-lobster/ume/runs/2025-06-12T16-47-59/epoch=0-step=39000-val_loss=0.5387.ckpt",  # Add when available
    "ume-large-base-740M": "s3://prescient-lobster/ume/runs/2025-06-14T17-01-52/epoch=0-step=24500-val_loss=0.7146.ckpt",
}
