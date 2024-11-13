from shutil import which

import pytest
from lobster.model import FoldseekTransform


def is_executable_available(executable_name):
    return which(executable_name) is not None


@pytest.mark.skipif(not is_executable_available("foldseek"), reason="No foldseek executable found")
class TestFoldseekTransform:
    """Fasta input -> LobsterFold -> foldseek -> AA+3Di output"""

    def test_foldseek_transform(self):
        seq = "MRLIPL"  # 6-mer peptide for testing

        foldseek = which("foldseek")
        assert foldseek is not None
        foldseek_transform = FoldseekTransform(foldseek=foldseek, lobster_fold_model_name="esmfold_v1", linker_length=3)

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
