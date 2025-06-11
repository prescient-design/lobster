import importlib.resources

BASE_PATH = importlib.resources.files("lobster") / "assets" / "codon_tables"
CODON_TABLE_PATH = BASE_PATH / "codon_table.json"
CODON_TABLE_PATH_VENDOR = BASE_PATH / "vendor_codon_table.json"
