import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import variable
from mgattention.models.layers import MGAT
from mgattention.parse_protein_symbols import entrez_dict
from mgattention.process_labels.get_labels import get_go_labels
import argparse
import pandas as pd
from sklearn.metrics import precision_score
import networkx as nx
import time
import json
import numpy as np

def get_labels(proteins,
               obofile,
               g2gofile,
               entrez_map,
               go_type = "molecular_function",
               min_level = 5,
               min_protein_annotation = 50,
               symbol_to_id = lambda x:x):
    """
    Get the protein GO labels corresponding to a given protein.
    """
    filter_protein = {"namespace": go_type, "lower_bound": min_level}
    filter_labels  = {"namespace": go_type, "min_level": min_level}
    symbol_to_entrez, entrez_to_symbol = entrez_dict(entrez_map)
    entrez_prots   = [symbol_to_entrez[p] for p in proteins]

    f_labels, labels_dict = get_go_labels(filter_protein,
                                          filter_label,
                                          entrez_prots,
                                          lambda x: symbol_to_id(entrez_to_symbol[x]),
                                          g2gofile,
                                          obofile,
                                          verbose = True)
    return f_labels, labels_dict


def _create_adj_from_df(df, n_nodes):
    """
    Constructs the adjacency matrix from the dataframe.
    """
    adjmat = np.zeros((n_nodes, n_nodes))
    for row in df.iterrows():
        p  = row["p"]
        q  = row["q"]
        wt = row["weight"]
        adjmat[p, q] = wt
        adjmat[q, p] = wt
    return adjmat

    
def get_adjacency_matrices(files, nodemap_file):
    """
    All file locations are given in the files list.
    """
    nodemap = {}
    with open(nodemap_file, "r") as n_f:
        nodemap = json.load(n_f)
    
    dfs   = []
    for f in files:
        dfs.append(pd.read_csv(f, header = None, sep = " "))

    # Generate the adjacency matrices
    adjs    = []
    for i, _ in enumerate(dfs):
        dfs[i].replace({0: nodemap, 1: nodemap})
        dfs[i].columns = ["p", "q", "weight"]
        adjs.append(_create_adj_from_df(dfs, len(nodemap)))
    return adjs, nodemap

                                          
def get_args():
    """
    Get arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder",  help = "Folder containing all the networks, indexed with .net extension...")
    parser.add_argument("--feature_emb", help = "Feature embedding file")
    parser.add_argument("--label_emb", help = "Label Embedding File")
    parser.add_argument("--train_indices", help = "Train indices file")
    parser.add_argument("--validation_indices", help = "Validation indices file")
    parser.add_argument("--output_folder", help = "Folder where the outputs are saved...")
    parser.add_argument("--hidden_features", type = int,
                        default = 50,
                        help = "The size of the hidden features...")
    parser.add_argument("--cuda", default = False, action = "store_true", help = "Enable CUDA")
    parser.add_argument("--nhead", type = int, default = 5, action = "MULTI-HEAD attention")
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--weight_decay", type = float, default = 5e-4)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--dropout", type = float, default = 0.6)
    parser.add_argument("--alpha", type = float, default = 0.2, help = "Alpha for leaky relu")
    parser.add_argument("--hidden", type = int, default = 8)
    return parser.parse_args()

def evaluate(predictions, labels, n_items):
    """
    Evaluate the average AUPR score for the test datasets
    """
    auprs = []
    
    def compute_auprs(prediction, label):
        """
        Compute the individual AUPR score
        """
        pred_ = prediction.cpu().numpy()
        lab_  = label.cpu().numpy()
        return precision_score(pred_, lab_)

    loss_val         = F.binary_cross_entropy(predictions, labels)
    for i in range(n_items):
        auprs.append(compute_auprs(predictions[i], labels[i]))
    return np.average(auprs), loss_val
        
        

def train_model(model, adjs, features, labels, train_indices, test_indices, params = None):
    """
    Perform training
    """
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr = params.learning_rate, weight_decay = params.weight_decay)

    for epoch in range(params.num_epochs):    
        model.train()
        optimizer.zero_grad()
        # Compute the output
        output     = model(features, adjs)
        # Get the train indices
        loss_train = F.binary_cross_entropy(output[train_indices], labels[train_indices])
        loss_train.backward()
        optimizer.step()

        # Evaluation
        if not params.fast_mode:
            model.eval()
            output = model(features, adjs)
            aupr, loss_val = evaluate(output[test_indices], labels[test_indices], len(test_indices))
            print("Epoch:      {:04d}".format(epoch+1),
                  "Loss_train: {:.4f}".format(loss_train.data.item()),
                  "loss_val:   {:.4f}".format(loss_val.data.item()),
                  "aupr_val:   {:4f}".format(aupr))
        
    return aupr, loss_val

def main(args):
    features_ = np.load(args.input_folder + f"/{args.feature_emb}", dtype = np.float)
    labels_   = np.load(args.input_folder + f"/{args.label_emb}", dtype = np.int32)
    

    n_features = features_.shape[1]
    n_classes  = labels_.shape[1]

    features = torch.from_numpy(features_)
    labels   = torch.from_numpy(labels_)
    
    model = MGAT(n_features, args.hidden_features, n_classes, args.dropout, args.alpha, args.nheads)

    if args.cuda:
        features = features.cuda()
        labels   = labels.cuda()
        model    = model.cuda()

    # Adjacency files
    network_files = [f for f in os.listdir(args.input_folder) if f.endswith(".txt")]
    json_file     = args.input_folder + "/nodemap.json"

    adjs, _       = get_adjacency_matrices(network_files, json_file)

    train_indices = np.load(args.input_folder + f"{args.train_indices}")
    test_indices  = np.load(args.input_folder + f"{args.test_indices}")
    
    train_model(model, adjs, features, labels, train_indices, test_indices, params = args)

    # Saving
    model.cpu()
    torch.save(model, args.output_folder + f"/epoch_{args.epochs + 1}.sav")
    if args.cuda:
        model.cuda()
