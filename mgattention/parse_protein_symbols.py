import re
def entrez_dict(map_loc):
    """
    This function constructs the map between protein symbol and its corresponding Entrez ID.
    It returns the Symbol->Entrez and Entrez->Symbol dictionaries.
    The map file is located at:
    /cluster/tufts/cowenlab/Projects/Denoising_Experiments/shared_data/dream_files/idmap.csv
    """
    s_e_dict = {}
    with open(map_loc, "r") as of:
        header = True
        for line in of:
            if header:
                header = False
                continue
            words = re.split("\t", line.strip())
            if len(words) >= 2 and words[1] != "":
                s_e_dict[words[0]] = int(words[1])
    rev_dict = {s_e_dict[k]: k for k in s_e_dict}
    return s_e_dict, rev_dict
