import re
import pandas as pd

def entrez_dict(map_loc):
    """
    This function constructs the map between protein symbol and its corresponding Entrez ID.
    It returns the Symbol->Entrez and Entrez->Symbol dictionaries.
    The map file is located at:
    /cluster/tufts/cowenlab/Projects/Denoising_Experiments/shared_data/dream_files/idmap.csv
    """

    df = pd.read_csv(map_loc, sep = "\t")
    s_e_dict = {}
    rev_dict = {}
    for i, row in df.iterrows():
        entrez = row["entrezgene"]
        symbol = row["symbol"]
        s_e_dict[symbol] = entrez
        rev_dict[entrez] = symbol
    return s_e_dict, rev_dict
