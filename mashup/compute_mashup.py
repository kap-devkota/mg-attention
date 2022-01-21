import numpy as np
from numpy.linalg import pinv, eig
import pandas as pd

def compute_mashup(As, reduced_dim = 1000):
    """
    Returns the MASHUP embedding. As is a list of adjacency matrices with the same dimension; obtained using the function As[0].shape
    """
    def compute_rwr(P, restart_prob = 0.5):
        dim  = A.shape[0]
        return pinv(np.identity(dim) - restart_prob * P) * (1 - restart_prob)
    
    def compute_inv_diagonal(D):
        D_   = np.zeros((D.shape[0], D.shape[0]))
        for i in range(D.shape[0]):
            D_[i, i] = 0 if D[i] == 0 else 1 / D[i]
        return D_
    
    def compute_p(A):
        m, _ = A.shape
        D    = (A @ np.ones((m, 1))).flatten()
        D_   = compute_inv_diagonal(D)
        return D_ @ A
    
    n, _ = As[0].shape
    R_f  = np.zeros((n,n))
    for A in As:        
        P = compute_p(A)
        Q = compute_rwr(P)
        R = np.log(Q + 1 / n)
        R_f += R @ R.T
    e, W = eig(R_f)
    # sort in descending order
    ids  = np.argsort(-e)
    ids  = ids.astype(int)
    print(ids)
    e    = e[ids[:reduced_dim]]
    W    = W[:, ids[:reduced_dim]]
    e    = np.diag(np.sqrt(np.sqrt(e)))
    return W @ e.T


def _dataframe_to_mat(df, ndim):
    df.columns = ["p", "q", "weight"]
    A          = np.zeros((ndim, ndim))
    for index, row in df.iterrows():
        p = int(row["p"])
        q = int(row["q"])
        A[p, q] = row["weight"]
    D = (A @ np.ones((A.shape[0], 1))).flatten()
    for i in range(D.shape[0]):
        if D[i] == 0:
            A[i, i] = 1
    return A
    
def generate_As(filenames, verbose = False):
    def log(strng):
        if verbose:
            print(strng)
    nodes = set()
    As    = []
    dfs   = []
    for f in filenames:
        log(f"Generate panda dataframe for graph {f}...")
        df = pd.read_csv(f, delim_whitespace=True, header = None)
        nodes = nodes.union(set(df[0]).union(set(df[1])))
        dfs.append(df)
    nodemap = {k: i for i, k in enumerate(nodes)}
    log(f"Number of nodes {len(nodemap)}.")
    for i in range(len(dfs)):
        log(f"Processing graph {i}...")
        df_ret = dfs[i].replace({0:nodemap, 1:nodemap})
        As.append(_dataframe_to_mat(df_ret, len(nodes)))
    return As, nodemap

                           