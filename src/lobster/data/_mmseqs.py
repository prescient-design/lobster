"""Code for MMseqs2. Install directions in README."""

import shutil
import subprocess
from typing import Optional, Sequence, Union

import pandas as pd
from Bio import SeqIO


class MMSeqsRunner:
    def __init__(self) -> None:
        """Includes code adapted from Ed Wagstaff"""
        assert self._check_tool_installed()  # check for mmseqs
        self.mmseqs_cmd = "mmseqs"

    def cluster_sequences(
        self,
        inputs: Union[str, pd.DataFrame],
        output_fasta_file: str,
        sequence_identity: float = 0.85,
        cov: float = 0.8,
        cov_mode: int = 1,
    ):
        """Adapted from https://www.biostars.org/p/9556498/
        https://github.com/soedinglab/MMseqs2/wiki#clustering
        """

        # Generate fasta file if input is pd.DataFrame instance, else copy to working path
        input_fasta_file = "input.fasta"
        if isinstance(inputs, pd.DataFrame):
            with open(input_fasta_file, "w") as f:
                for id_, sequence in inputs["sequence"].to_dict().items():
                    f.write(f">{id_}\n{sequence}\n")
        else:
            shutil.copyfile(inputs, input_fasta_file)

        # Create a database from the temporary FASTA file
        subprocess.run(["mmseqs", "createdb", input_fasta_file, "combined_db"])

        # Cluster sequences based on the specified identity threshold
        # target coverage (alignment covers at least 0.8 of target member sequence)
        subprocess.run(
            [
                "mmseqs",
                "linclust",
                "combined_db",
                "clusters",
                "tmp",
                "--min-seq-id",
                str(sequence_identity),
                "-c",
                str(cov),
                "--cov-mode",
                str(cov_mode),
                "--threads",
                "8",
                "--split-memory-limit",
                "70G",
                "--remove-tmp-files",
            ]
        )

        # Extract the representatives from the clusters
        subprocess.run(["mmseqs", "result2repseq", "combined_db", "clusters", "cluster_seq"])

        # # Extract sequences of the representatives
        # subprocess.run(["mmseqs", "convert2fasta", "combined_db", "cluster_representatives.tsv", output_fasta_file])
        subprocess.run(
            [
                "mmseqs",
                "result2flat",
                "combined_db",
                "combined_db",
                "cluster_seq",
                output_fasta_file,
                "--use-fasta-header",
            ]
        )

        # Rename the output file to the desired name
        # os.rename("cluster_representatives.fasta", output_fasta_file)

        # Remove temporary files (optional)
        subprocess.run(["rm -f cluster*"], shell=True)
        subprocess.run(["rm -f combined_db*"], shell=True)
        subprocess.run(["rm", "-rf", "tmp"])

        print(f"Clustering completed. Results saved to {output_fasta_file}")

    def run_linclust(
        self,
        input_fasta_file: str,
        file_prefix: str = "linclust_",
        sequence_identity: float = 0.85,
    ):
        """FILE_PREFIX=tenx_linclust_seqid70c80
        mmseqs easy-linclust tenx.fasta $FILE_PREFIX /tmp/ --min-seq-id 0.70 -c 0.8 --cov-mode 1
        """
        subprocess.run(
            [
                "mmseqs",
                "easy-linclust",
                input_fasta_file,
                file_prefix,
                "tmp",
                "--min-seq-id",
                str(sequence_identity),
                "-c",
                "0.8",
                "--cov-mode",
                "1",
                "--threads",
                "8",
                "--split-memory-limit",
                "70G",
                "--remove-tmp-files",
            ]
        )

    def linclust_to_train_val_test_split(
        self,
        cluster_csv: str,
        input_fasta: str,
        lengths: Optional[Sequence[float]] = (0.9, 0.05, 0.05),
    ):
        """
        Linclust min-seq-id: centroids of the clusters should be less than X% identical

        cluster_csv: csv from mmseqs easy-linclust
        input_fasta: fasta with cluster labels from mmseqs easy-linclust
        lengths: train/val/test splits
        """

        clusters_df = pd.read_csv(cluster_csv, sep="\t", header=None, names=["Cluster", "Sequence"])
        sequences = [record for record in SeqIO.parse(input_fasta, "fasta") if len(record.seq) > 0]

        num_train, num_val, num_test = [ll * len(clusters_df) for ll in lengths]
        print(num_train, num_val, num_test)
        train_data, val_data, test_data = (
            pd.DataFrame(columns=clusters_df.columns),
            pd.DataFrame(columns=clusters_df.columns),
            pd.DataFrame(columns=clusters_df.columns),
        )

        num_assigned = 0
        for _, group_df in clusters_df.groupby("Cluster"):
            num_samples = len(group_df)
            if num_assigned < round(num_train):
                train_data = pd.concat([train_data, group_df])
            elif round(num_train + num_val) < num_assigned < round(num_train + num_val + num_test):
                val_data = pd.concat([val_data, group_df])
            else:
                test_data = pd.concat([test_data, group_df])
            num_assigned += num_samples

        training_sequences = [seq for seq in sequences if int(seq.id) in train_data["Sequence"]]
        validation_sequences = [seq for seq in sequences if int(seq.id) in val_data["Sequence"]]
        test_sequences = [seq for seq in sequences if int(seq.id) in test_data["Sequence"]]

        for split, seqs in zip(
            ["train", "val", "test"],
            [training_sequences, validation_sequences, test_sequences],
        ):
            prefix = input_fasta.split(".")[0]  # remove .fasta suffix
            SeqIO.write(seqs, f"{prefix}_{split}.fasta", "fasta")

    def _check_tool_installed(self, name="mmseqs"):
        # Checks whether `name` is on PATH and marked as executable.
        if shutil.which(name) is None:
            raise EnvironmentError(
                f"{name} is not found. Please ensure the executable is installed and is in your PATH."
            )
        else:
            print(f"{name} is installed and available.")
            return True
