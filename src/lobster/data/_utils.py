import pickle
from urllib.parse import urlparse

import boto3


def load_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def get_s3_bucket_and_key(s3_uri: str) -> tuple[str, str]:
    """Extract the bucket name and object key from the S3 path."""
    parsed_url = urlparse(str(s3_uri))
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip("/")

    return bucket, key


def upload_to_s3(s3_uri: str, local_filepath: str) -> None:
    """Upload a file to S3."""
    bucket, key = get_s3_bucket_and_key(s3_uri)
    s3_client = boto3.client("s3")

    with open(local_filepath, "rb") as data:
        s3_client.upload_fileobj(data, bucket, key)


def download_from_s3(s3_uri: str, local_filepath: str) -> None:
    """Download a file from S3."""
    bucket, key = get_s3_bucket_and_key(s3_uri)
    s3_client = boto3.client("s3")

    with open(local_filepath, "wb") as data:
        s3_client.download_fileobj(bucket, key, data)
