import pandas as pd
from lobster.transforms.functional import fix_liabilities


def test_fix_liabilities():
    fv_heavy = "EVQLVESGGGLVQPGGSLRNGCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYYYGSSHWVFDVWGQGTLVTVSS"
    fv_light = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNCLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLNGSSLQPEDFATYYCQQYNIYPLTFGQGTKVEIK"

    df = fix_liabilities(fv_heavy, fv_light, return_sequences=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert df.shape[1] == 23
    assert df.iloc[0]["liability"] == "heavy_ng_motif"

    fv_heavy_fixed, fv_light_fixed = fix_liabilities(
        fv_heavy, fv_light, return_sequences=True
    )
    assert "NG" not in fv_heavy_fixed
    assert fv_heavy_fixed != fv_heavy
    assert fv_light_fixed != fv_light
