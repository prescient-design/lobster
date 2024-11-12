import pytest
from lobster.tokenization import MgmTokenizerTransform
from lobster.transforms import uniform_sample
from torch import Size


class TestMgmTokenizerTransform:
    def test_preprocess_seq(self):
        # aa seq
        transform_fn = MgmTokenizerTransform(
            tokenizer_dir="mgm_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=64,
            codon_sampling_strategy=uniform_sample,
            input_modality="aa",
            p_aa=0.111,
            p_nt=0.111,
            p_sf=0.111,
            p_aa_nt=0.111,
            p_nt_aa=0.111,
            p_aa_sf=0.111,
            p_sf_aa=0.111,
            p_nt_sf=0.111,
            p_sf_nt=0.111,
        )

        seq = "evqlvesgggl"
        preprocessed_seq = transform_fn.preprocess_seq(seq)
        print(preprocessed_seq)
        assert preprocessed_seq.isupper()

        # nt seq
        transform_fn = MgmTokenizerTransform(
            tokenizer_dir="mgm_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=64,
            codon_sampling_strategy=uniform_sample,
            input_modality="nt",
            p_aa=0.111,
            p_nt=0.111,
            p_sf=0.111,
            p_aa_nt=0.111,
            p_nt_aa=0.111,
            p_aa_sf=0.111,
            p_sf_aa=0.111,
            p_nt_sf=0.111,
            p_sf_nt=0.111,
        )

        # nt seq has 3n+2 characters
        seq = "atcgtacgatcgtacgatcg"
        preprocessed_seq = transform_fn.preprocess_seq(seq)
        assert preprocessed_seq == seq[:-2].upper()

        # early stop codon
        seq = "ATCGTACGATAGTCGTACGATCG"
        preprocessed_seq = transform_fn.preprocess_seq(seq)
        assert preprocessed_seq == seq[:12]

    def test_prep_input(self):
        # Test for 'aa' input modality
        transform_fn = MgmTokenizerTransform(
            tokenizer_dir="mgm_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=64,
            codon_sampling_strategy=uniform_sample,
            input_modality="aa",
            p_aa=0.111,
            p_nt=0.111,
            p_sf=0.111,
            p_aa_nt=0.111,
            p_nt_aa=0.111,
            p_aa_sf=0.111,
            p_sf_aa=0.111,
            p_nt_sf=0.111,
            p_sf_nt=0.111,
        )

        seq = "evimjl"
        # transform_fn._sampled_modalities = "aa_only"
        # transform_fn._modalities_no = 1
        out = transform_fn.prep_input(seq, modality_combination="aa", n_modalities=1, max_len=64, crop_left=False)
        assert len(out) == 1
        assert out[0] == "<cls_aa>EVIMJL"

        # transform_fn._sampled_modalities = "nt_sf"
        # transform_fn._modalities_no = 2
        out = transform_fn.prep_input(seq, modality_combination="nt_sf", n_modalities=2, max_len=64, crop_left=False)
        assert len(out) == 1
        assert out[0][:8] == "<cls_nt>"

        # Test for 'nt' input modality
        transform_fn._input_modality = "nt"
        seq = ["ATCGTACGATCGTACGATCGT", "ATCGTACGATCGTT"]
        # transform_fn._sampled_modalities = "aa_only"
        # transform_fn._modalities_no = 1
        out = transform_fn.prep_input(seq, modality_combination="aa", n_modalities=1, max_len=64, crop_left=False)
        assert len(out) == 2
        assert out == ["<cls_aa>IVRSYDR", "<cls_aa>IVRS"]
        # assert out[0][0].isupper()
        # assert out[0][1].isupper()
        # assert len(out[0][0]) == 7
        # assert len(out[0][1]) == 4

        # transform_fn._sampled_modalities = "nt_aa"
        # transform_fn._modalities_no = 2
        out = transform_fn.prep_input(seq, modality_combination="nt_aa", n_modalities=2, max_len=64, crop_left=False)
        assert len(out) == 2
        assert out[0][:8] == "<cls_nt>"
        assert out[0][:8] == "<cls_nt>"
        # assert out[0][0].islower()
        # assert out[0][1].islower()
        # assert len(out[0][0]) == 21
        # assert len(out[0][1]) == 12

        # assert out[1][0].isupper()
        # assert out[1][1].isupper()
        # assert len(out[1][0]) == 7
        # assert len(out[1][1]) == 4

    @pytest.mark.skip(reason="Skip test_transform")
    def test_transform(self):
        transform_fn = MgmTokenizerTransform(
            tokenizer_dir="mgm_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=64,
            codon_sampling_strategy=uniform_sample,
            input_modality="nt",
            p_aa=0.111,
            p_nt=0.111,
            p_sf=0.111,
            p_aa_nt=0.111,
            p_nt_aa=0.111,
            p_aa_sf=0.111,
            p_sf_aa=0.111,
            p_nt_sf=0.111,
            p_sf_nt=0.111,
            mask_percentage=0.25,
        )

        inputs = 4 * ["ATCGTACGATCGTACGATCGU"]
        tokenized = transform_fn.transform(inputs, {})

        assert tokenized["input_ids"].shape == Size([4, 64])
        assert tokenized["attention_mask"].shape == Size([4, 64])
        # assert tokenized["labels"].shape == Size([4, 64])

        inputs = "ATCGTACGATCGTACGATCGU"
        tokenized = transform_fn.transform(inputs, {})

        assert tokenized["input_ids"].shape == Size([1, 64])
        assert tokenized["attention_mask"].shape == Size([1, 64])
        # assert tokenized["labels"].shape == Size([1, 64])
