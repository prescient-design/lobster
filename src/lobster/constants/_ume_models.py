# TODO: currently, these will work for internal users
# Support for external users will be added soon
import json

import boto3

UME_CHECKPOINT_DICT_S3_BUCKET = "prescient-lobster"
UME_CHECKPOINT_DICT_S3_KEY = "ume/checkpoints.json"
UME_CHECKPOINT_DICT_S3_URI = f"s3://{UME_CHECKPOINT_DICT_S3_BUCKET}/{UME_CHECKPOINT_DICT_S3_KEY}"


def get_ume_checkpoints() -> dict[str, str]:
    """Get the UME checkpoints from S3."""
    client = boto3.client("s3")

    response = client.get_object(Bucket=UME_CHECKPOINT_DICT_S3_BUCKET, Key=UME_CHECKPOINT_DICT_S3_KEY)
    decoded_body = response["Body"].read().decode("utf-8")

    return json.loads(decoded_body)


UME_PRETRAINED_CHECKPOINTS = get_ume_checkpoints()
