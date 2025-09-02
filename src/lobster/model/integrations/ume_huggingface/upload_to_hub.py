from huggingface_hub import HfApi

from lobster.constants import HF_UME_MODEL_DIRPATH, HF_UME_REPO_ID


def create_repo():
    api = HfApi()
    api.create_repo(HF_UME_REPO_ID)


def upload_to_hub():
    api = HfApi()
    api.upload_folder(
        folder_path=HF_UME_MODEL_DIRPATH,
        repo_id=HF_UME_REPO_ID,
    )


if __name__ == "__main__":
    # create_repo() # only need to run this once
    upload_to_hub()
