from shutil import which

from lobster.transforms import FoldseekTransform


class TestFoldseekTransform:
    """Fasta input -> PPLMFold -> foldseek -> AA+3Di output"""

    def test_foldseek_transform(self):
        seq = "MRLIPL"  # 6-mer peptide for testing

        foldseek = which("foldseek")
        foldseek_transform = FoldseekTransform(
            foldseek=foldseek, lobster_fold_model_name="PPLM", linker_length=3
        )

        if foldseek is not None:
            seq_dict = foldseek_transform.transform(sequences=[seq])
        else:
            seq_dict = {"A": ("MRLIPL", "DVVVVD", "MdRvLvIvPvLd")}  # mock

        assert len(seq_dict["A"]) == 3

        seqs = ["MRLIPL", "MRLIPL"]  # homo-dimer

        if foldseek is not None:
            seq_dict = foldseek_transform.transform(
                sequences=[seqs],
            )
        else:
            seq_dict = {"A": ("MRLIPLMRLIPL", "DVNCVCVVVVVD", "MdRvLnIcPvLcMvRvLvIvPvLd")}  # mock

        assert len(seq_dict["A"]) == 3
