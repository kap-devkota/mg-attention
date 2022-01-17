import numpy as np
import time
import sys
import mygene
from goatools.base import get_godag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_ncbi_associations

##################### Code for Getting GO labels ##################################

class GoTool:
    def __init__(self, obo_location):
        # gdagfile = pkg_resources.resource_filename('glide', 'data/go-basic.obo.dat')
        self.godag = get_godag(obo_location, optional_attrs='relationship')
        
    def get_labels(self, filters = None):
        if filters == None:
            return list(self.godag.keys())
        go_terms   = []
        for k in self.godag.keys():
            k_val = self.godag[k]
            if "max_level" in filters:
                if k_val.level > filters["max_level"]:
                    continue
            if "min_level" in filters:
                if k_val.level < filters["min_level"]:
                    continue
            if "namespace" in filters:
                if k_val.namespace != filters["namespace"]:
                    continue
            go_terms.append(k)
        return go_terms
    
    def get_parents(self, go_id, filters = None):    
        gosubdag = GoSubDag(go_id, self.godag, relationships=True, prt=False)
        nts      = gosubdag.get_vals("id")
        parents  = []
        for n in nts:
            nObj = gosubdag.go2nt[n]
            ns   = nObj.NS
            if ns == "BP":
                ns = "biological_process"
            elif ns == "MF":
                ns = "molecular_function"
            else:
                ns = "cellular_component"
            level= nObj.level
            
            if (("max_level" in filters and filters["max_level"] < level) or
                ("min_level" in filters and filters["min_level"] > level) or
                ("namespace" in filters and filters["namespace"] != ns)):
                continue
            parents.append(nObj.id)
        return parents

    
def get_go_labels(filter_protein,
                  filter_label,
                  entrez_labels,
                  anno_map = lambda x : x,
                  g2gofile="gene2go.dat",
                  obofile="go-basic.obo",
                  verbose = False):
    """
    filter_protein: {"namespace" : "molecular_function" | "biological_process" | "cellular_component",
                     "min_level": 5,
                     "max_level": 1000}
    fliter_label: {"lower_bound": 50}
    entrez_labels: [entrez_proteins]
    """
    def log(strng):
        if verbose:
            print(strng)
    
    # g2gofile = pkg_resources.resource_filename('glide', 'data/gene2go.dat')
    objanno = Gene2GoReader(g2gofile, taxids=[9606])
    go2geneids_human = objanno.get_id2gos(namespace=filter_protein["namespace"], 
                                          go2geneids=True)
    gt          = GoTool(obofile)
    labels      = gt.get_labels(filter_label)
    labels_dict = {}
    f_labels    = []
    for key in labels:
        if key not in go2geneids_human:
            continue
        assoc_genes   = go2geneids_human[key]
        f_assoc_genes = [] 
        for a in assoc_genes:
            if a in entrez_labels:
                f_assoc_genes.append(anno_map(a))
        if len(f_assoc_genes) > filter_protein["lower_bound"]:
            labels_dict[key] = f_assoc_genes
            f_labels.append(key)
    log(f"Labels Obtained! The number of labels obtained is {len(f_labels)}")
    return f_labels, labels_dict
