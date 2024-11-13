from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice

import numpy as np
from Bio import SeqIO
from datasketch import WeightedMinHashGenerator


class LobsterMinHasher:
    """Use MinHash to deduplicate sequence datasets.

    Current implementation converts sequences to k-mers and performs a
    WeightedMinHash on the k-mer frequency vectors.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        k: int = 4,
        is_protein: bool = True,
    ):
        """Initialize LobsterMinHasher.

        Parameters
        ----------
        num_perm : int
            Number of permutations to use in the MinHash.
        threshold : float
            Jaccard similarity threshold for considering sequences as duplicates.
        k : int
            Length of k-mers to use.
        is_protein : bool
            Whether the sequences are protein sequences.
        """

        self._num_perm = num_perm
        self._threshold = threshold
        self._is_protein = is_protein
        self._k = k
        self._counter = 0

        if self._is_protein:
            self._dim = 20**self._k
        else:
            self._dim = 4**self._k

        self._wmh_gen = WeightedMinHashGenerator(self._dim, sample_size=self._num_perm)
        # self._kmer_to_int = defaultdict(lambda: len(self._kmer_to_int))
        self._kmer_to_int = defaultdict(int)

    def deduplicate_sequences(self, fasta_file: str, output_fasta_file: str, n: int = None):
        """Deduplicate sequences in a FASTA file using MinHash.

        Parameters
        ----------
        fasta_file : str
            Path to the input FASTA file.
        output_fasta_file : str
            Path to the output FASTA file.
        n : int, optional
            Number of sequences to read from the FASTA file. If None, all sequences
            will be read.
        """

        if n is not None:
            sequences = self._take_from_fasta(SeqIO.parse(fasta_file, "fasta"), n)
        else:
            sequences = list(SeqIO.parse(fasta_file, "fasta"))

        minhashes = {}
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    partial(
                        LobsterMinHasher._process_sequence,
                        self._k,
                        self._counter,
                        self._dim,
                        self._kmer_to_int,
                        self._wmh_gen,
                    ),
                    sequences,
                )
            )

        minhashes = {
            record_id: minhash
            for record_id, minhash in results
            if not any(stored_minhash.jaccard(minhash) > self._threshold for stored_minhash in minhashes.values())
        }

        unique_seq_records = [record for record in sequences if record.id in minhashes]

        SeqIO.write(unique_seq_records, output_fasta_file, "fasta")

    @staticmethod
    def _process_sequence(k: int, counter: int, dim: int, kmer_to_int, wmh_gen, record: SeqIO.SeqRecord):
        # Create k-mers of the protein sequence and map them to unique integers
        kmers = [str(record.seq[i : i + k]) for i in range(len(record.seq) - k + 1)]
        # weights = np.bincount([kmer_to_int[kmer] for kmer in kmers])
        weights = np.bincount(
            [kmer_to_int.setdefault(kmer, counter + 1) for kmer in kmers if not kmer_to_int.get(kmer, counter)],
            minlength=dim,
        )
        minhash = wmh_gen.minhash(weights)
        return record.id, minhash

    def _take_from_fasta(iterable, n: int = 100_000):
        """Take n records from a FASTA file rather than reading into memory."""
        return list(islice(iterable, n))
