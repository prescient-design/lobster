import importlib.resources

from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast

LG_VOCAB = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, '<mask>': 4, '.': 5, 'a': 6, 'b': 7, 'c': 8,
            'd': 9, 'e': 10, 'f': 11, 'g': 12, 'h': 13, 'i': 14, 'j': 15, 'k': 16, 'l': 17, 'm': 18, 'n': 19,
            'o': 20, 'p': 21, 'q': 22, 'r': 23, 's': 24, 't': 25, 'u': 26, 'v': 27, 'w': 28, 'x': 29, 'y': 30,
            'z': 31, 'aa': 32, 'ab': 33, 'ac': 34, 'ad': 35, 'ae': 36, 'af': 37, 'ag': 38, 'ah': 39, 'ai': 40,
            'aj': 41, 'ak': 42, 'al': 43, 'am': 44, 'an': 45, 'ao': 46, 'ap': 47, 'aq': 48, 'ar': 49, 'as': 50,
            'at': 51, 'au': 52, 'av': 53, 'aw': 54, 'ax': 55, 'ay': 56, 'az': 57, 'ba': 58, 'bb': 59, 'bc': 60,
            'bd': 61, 'be': 62, 'bf': 63, 'bg': 64, 'bh': 65, 'bi': 66, 'bj': 67, 'bk': 68, 'bl': 69, 'bm': 70,
            'bn': 71, 'bo': 72, 'bp': 73, 'bq': 74, 'br': 75, 'bs': 76, 'bt': 77, 'bu': 78, 'bv': 79, 'bw': 80,
            'bx': 81, 'by': 82, 'bz': 83, 'ca': 84, 'cb': 85, 'cc': 86, 'cd': 87, 'ce': 88, 'cf': 89, 'cg': 90,
            'ch': 91, 'ci': 92, 'cj': 93, 'ck': 94, 'cl': 95, 'cm': 96, 'cn': 97, 'co': 98, 'cp': 99, 'cq': 100,
            'cr': 101, 'cs': 102, 'ct': 103, 'cu': 104, 'cv': 105, 'cw': 106, 'cx': 107, 'cy': 108, 'cz': 109,
            'da': 110, 'db': 111, 'dc': 112, 'dd': 113, 'de': 114, 'df': 115, 'dg': 116, 'dh': 117, 'di': 118,
            'dj': 119, 'dk': 120, 'dl': 121, 'dm': 122, 'dn': 123, 'do': 124, 'dp': 125, 'dq': 126, 'dr': 127,
            'ds': 128, 'dt': 129, 'du': 130, 'dv': 131, 'dw': 132, 'dx': 133, 'dy': 134, 'dz': 135, 'ea': 136,
            'eb': 137, 'ec': 138, 'ed': 139, 'ee': 140, 'ef': 141, 'eg': 142, 'eh': 143, 'ei': 144, 'ej': 145,
            'ek': 146, 'el': 147, 'em': 148, 'en': 149, 'eo': 150, 'ep': 151, 'eq': 152, 'er': 153, 'es': 154,
            'et': 155, 'eu': 156, 'ev': 157, 'ew': 158, 'ex': 159, 'ey': 160, 'ez': 161, 'fa': 162, 'fb': 163,
            'fc': 164, 'fd': 165, 'fe': 166, 'ff': 167, 'fg': 168, 'fh': 169, 'fi': 170, 'fj': 171, 'fk': 172,
            'fl': 173, 'fm': 174, 'fn': 175, 'fo': 176, 'fp': 177, 'fq': 178, 'fr': 179, 'fs': 180, 'ft': 181,
            'fu': 182, 'fv': 183, 'fw': 184, 'fx': 185, 'fy': 186, 'fz': 187, 'ga': 188, 'gb': 189, 'gc': 190,
            'gd': 191, 'ge': 192, 'gf': 193, 'gg': 194, 'gh': 195, 'gi': 196, 'gj': 197, 'gk': 198, 'gl': 199,
            'gm': 200, 'gn': 201, 'go': 202, 'gp': 203, 'gq': 204, 'gr': 205, 'gs': 206, 'gt': 207, 'gu': 208,
            'gv': 209, 'gw': 210, 'gx': 211, 'gy': 212, 'gz': 213, 'ha': 214, 'hb': 215, 'hc': 216, 'hd': 217,
            'he': 218, 'hf': 219, 'hg': 220, 'hh': 221, 'hi': 222, 'hj': 223, 'hk': 224, 'hl': 225, 'hm': 226,
            'hn': 227, 'ho': 228, 'hp': 229, 'hq': 230, 'hr': 231, 'hs': 232, 'ht': 233, 'hu': 234, 'hv': 235,
            'hw': 236, 'hx': 237, 'hy': 238, 'hz': 239, 'ia': 240, 'ib': 241, 'ic': 242, 'id': 243, 'ie': 244,
            'if': 245, 'ig': 246, 'ih': 247, 'ii': 248, 'ij': 249, 'ik': 250, 'il': 251, 'im': 252, 'in': 253,
            'io': 254, 'ip': 255, 'iq': 256, 'ir': 257, 'is': 258, 'it': 259, 'iu': 260, 'iv': 261}

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("lobster") / "assets" / "latent_generator_tokenizer"


def _make_latent_generator_tokenizer() -> PreTrainedTokenizerFast:
    """Create a `PreTrainedTokenizerFast` object for tokenization of protein structure latent generator sequences.

    To create the tokenizer config stored under lobster/assets/latent_generator_tokenizer we run

    ```
    tokenizer = _make_latent_generator_tokenizer()
    tokenizer.save_pretrained("src/lobster/assets/latent_generator_tokenizer")
    ```

    This can now be loaded using
    `PreTrainedTokenizerFast.from_pretrained("src/lobster/assets/latent_generator_tokenizer")`
    """

    # BPE with no merges => just use input vocab
    tokenizer_model = BPE(LG_VOCAB, merges=[], unk_token="<unk>", ignore_merges=True)

    # bert style post processing
    post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<cls>", 0), ("<eos>", 2)],  # NOTE must match ids from AA_VOCAB
    )

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        post_processor=post_processor,
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
    )


class LatentGeneratorTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(PRETRAINED_TOKENIZER_PATH / "tokenizer.json"),
            bos_token=None,
            eos_token="<eos>",
            unk_token="<unk>",
            sep_token=None,
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
        )
