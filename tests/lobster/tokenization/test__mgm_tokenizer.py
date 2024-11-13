from lobster.tokenization import MgmTokenizerTransform
from lobster.transforms import uniform_sample
from torch import Size


class TestMgmTokenizer:
    def test_mgm_tokenizer(self):
        transform_fn = MgmTokenizerTransform(
            tokenizer_dir="mgm_tokenizer",
            truncation=True,
            padding="max_length",
            max_length=32,
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

        cls_id = transform_fn._auto_tokenizer.cls_token_id
        sep_id = transform_fn._auto_tokenizer.sep_token_id
        eos_id = transform_fn._auto_tokenizer.eos_token_id
        cls_nt_id = transform_fn._auto_tokenizer.cls_nt_token_id
        cls_aa_id = transform_fn._auto_tokenizer.cls_aa_token_id
        pad_id = transform_fn._auto_tokenizer.pad_token_id

        inputs = 4 * ["atcgtacgatcgtacgatcgun"]
        transform_fn._auto_tokenizer.modalities = "nt"
        tokenized = transform_fn._auto_tokenizer(
            inputs,
            padding=transform_fn._padding,
            truncation=transform_fn._truncation,
            max_length=transform_fn._max_length,
            return_tensors="pt",
            return_token_type_ids=transform_fn._return_token_type_ids,
            return_attention_mask=transform_fn._return_attention_mask,
            return_overflowing_tokens=transform_fn._return_overflowing_tokens,
            return_special_tokens_mask=transform_fn._return_special_tokens_mask,
            return_offsets_mapping=transform_fn._return_offsets_mapping,
            return_length=transform_fn._return_length,
            verbose=transform_fn._verbose,
        )

        print(cls_id, cls_aa_id, sep_id, cls_nt_id, eos_id, pad_id)
        print(tokenized["input_ids"][0])
        assert tokenized["input_ids"].shape == Size([4, 32])
        assert tokenized["attention_mask"].shape == Size([4, 32])
        assert tokenized["input_ids"][0, 0] == cls_id
        # assert tokenized["input_ids"][0, 1] == cls_nt_id
        assert tokenized["input_ids"][0, 23] == eos_id

        inputs = [["EVIMTL"], ["atcgtacgatcg"]]
        transform_fn._auto_tokenizer.modalities = "aa_nt"
        tokenized = transform_fn._auto_tokenizer(
            inputs[0],
            inputs[1],
            padding=transform_fn._padding,
            truncation=transform_fn._truncation,
            max_length=transform_fn._max_length,
            return_tensors="pt",
            return_token_type_ids=transform_fn._return_token_type_ids,
            return_attention_mask=transform_fn._return_attention_mask,
            return_overflowing_tokens=transform_fn._return_overflowing_tokens,
            return_special_tokens_mask=transform_fn._return_special_tokens_mask,
            return_offsets_mapping=transform_fn._return_offsets_mapping,
            return_length=transform_fn._return_length,
            verbose=transform_fn._verbose,
        )

        print(cls_id, cls_aa_id, sep_id, cls_nt_id, eos_id, pad_id)
        print(tokenized["input_ids"][0])
        assert tokenized["input_ids"].shape == Size([1, 32])
        assert tokenized["attention_mask"].shape == Size([1, 32])
        assert tokenized["input_ids"][0, 0] == cls_id
        # assert tokenized["input_ids"][0, 1] == cls_aa_id
        assert tokenized["input_ids"][0, 7] == sep_id
        # assert tokenized["input_ids"][0, 8] == cls_nt_id
        assert tokenized["input_ids"][0, 20] == eos_id
        assert tokenized["input_ids"][0, 21] == pad_id
