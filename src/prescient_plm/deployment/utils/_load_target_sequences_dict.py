import pandas as pd

ANTIGEN_SEQS_PATH = "s3://prescient-data-dev/raw/affinity/target_sequences.csv"


def load_target_sequences_dict() -> pd.DataFrame:
    df = pd.read_csv(ANTIGEN_SEQS_PATH).rename(columns={"target_seq": "affinity_antigen"})
    df = df.set_index("target")
    return {k: v[0] for k, v in df.T.to_dict(orient="list").items()}
