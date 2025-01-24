import Levenshtein
import torch
from lobster.model import LobsterCBMPMLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the LobsterCBMPMLM model
cb_lobster = LobsterCBMPMLM("asalam91/cb_lobster_24M").to(device)
cb_lobster.eval()
print(cb_lobster.list_supported_concept())

concept = "gravy"
test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"

[new_protien] = cb_lobster.intervene_on_sequences([test_protein], concept, edits=5, intervention_type="negative")


print(new_protien)
print(Levenshtein.distance(test_protein, new_protien))
