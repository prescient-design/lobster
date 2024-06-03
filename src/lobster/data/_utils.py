import pickle

# from beignet.io import load_parquet_to_pandas


# def antibody_parquet_to_fasta(path_to_parquet: str, path_to_fasta: str = "output.fasta"):
#     df = load_parquet_to_pandas(path_to_parquet)
#     if HEAVY_COLUMN in df.columns:
#         df.dropna(subset=[HEAVY_COLUMN, LIGHT_COLUMN], inplace=True)
#     if "heavy_aho" in df.columns:
#         df.dropna(subset=["heavy_aho", "light_aho"], inplace=True)
#     if "heavy_aho" in df.columns:
#         df[HEAVY_COLUMN] = df["heavy_aho"].apply(lambda x: x.replace("-", ""))
#         df[LIGHT_COLUMN] = df["light_aho"].apply(lambda x: x.replace("-", ""))

#     idx_start = 0
#     heavy_records = [
#         SeqRecord(seq=Seq(seq), id=str(idx)) for idx, seq in enumerate(df[HEAVY_COLUMN].values)
#     ]
#     idx_start += len(heavy_records)
#     light_records = [
#         SeqRecord(seq=Seq(seq), id=str(idx + idx_start))
#         for idx, seq in enumerate(df[LIGHT_COLUMN].values)
#     ]

#     records = heavy_records + light_records
#     SeqIO.write(records, path_to_fasta, "fasta")


# def antibody_parquet_to_aho_fasta(path_to_parquet: str, path_to_fasta: str = "output.fasta"):
#     df = load_parquet_to_pandas(path_to_parquet)
#     if HEAVY_COLUMN in df.columns:
#         df.dropna(subset=[HEAVY_COLUMN, LIGHT_COLUMN], inplace=True)
#     if "heavy_aho" in df.columns:
#         df.dropna(subset=["heavy_aho", "light_aho"], inplace=True)
#     if "heavy_aho" in df.columns:
#         df[HEAVY_COLUMN] = df["heavy_aho"].apply(lambda x: x.replace("-", ""))
#         df[LIGHT_COLUMN] = df["light_aho"].apply(lambda x: x.replace("-", ""))

#     idx_start = 0
#     heavy_records = [
#         SeqRecord(seq=Seq(seq), id=str(idx)) for idx, seq in enumerate(df[HEAVY_COLUMN].values)
#     ]
#     idx_start += len(heavy_records)
#     light_records = [
#         SeqRecord(seq=Seq(seq), id=str(idx + idx_start))
#         for idx, seq in enumerate(df[LIGHT_COLUMN].values)
#     ]

#     records = heavy_records + light_records
#     SeqIO.write(records, path_to_fasta, "fasta")


def load_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data
