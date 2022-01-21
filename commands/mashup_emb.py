import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
import sys
import json
sys.path.append('../')
from mashup.compute_mashup import compute_mashup, generate_As

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_folder", help = "The network folder")
    parser.add_argument("--dimension", type = int, default = 1000, help = "The dimension")
    parser.add_argument("--verbose", default = False, action = "store_true")
    return parser.parse_args()



def main(args):
    def log(strng):
        if args.verbose:
            print(strng)
    netfiles = [join(args.network_folder, f) 
                for f in listdir(args.network_folder) 
                if isfile(join(args.network_folder, f)) 
                and f.endswith(".txt")]
    As, nodemap = generate_As(netfiles, verbose = True)
    mashup_emb  = compute_mashup(As, args.dimension)
    np.save(f"{args.network_folder}/dim_{args.dimension}.npy", mashup_emb)
    with open(f"{args.network_folder}/nodemap_dim_{args.dimension}.json", "w") as jf:
        json.dump(nodemap, jf)
        
if __name__ == "__main__":
    main(parse_arguments())
