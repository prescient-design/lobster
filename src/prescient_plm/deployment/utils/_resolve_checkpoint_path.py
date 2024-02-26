from prescient.io import download
from upath import UPath
from upath.implementations.cloud import S3Path


def resolve_s3_checkpoint_path(fpath, logger=None):
    fpath = UPath(fpath)

    if isinstance(fpath, S3Path):
        if logger is not None:
            logger.info(f"Downloading {fpath} to ./tmp/checkpoints/{fpath.name}")

        download(str(fpath), "./tmp/checkpoints", fpath.name)
        return str(UPath("./tmp/checkpoints") / fpath.name)

    else:
        return str(fpath)
