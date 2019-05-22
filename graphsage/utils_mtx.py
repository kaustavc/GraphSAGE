import numpy as np
from scipy.io import mmread
from graphsage.tracer import Tracer


def walks_gen(walks_file):
    with open(walks_file) as wf:
        for line in wf:
            yield([int(n) for n in line.split()])


def load_data(prefix, **kwargs):
    mtx_file = prefix + "-G.mtx"
    walks_file = prefix + "-walks.txt"
    with Tracer("Loading adjacency matrix", mtx_file=mtx_file):
        g = mmread(prefix + "-G.mtx")

    # We expect the matrix to be read in to be an adjacency matrix, hence it must be square
    (r, c) = g.get_shape()
    print("Loaded adjacency matrix has shape: " + str((r, c)))
    assert r == c, "Error: Adjacency matrix is not square"

    with Tracer("Converting adjacency matrix to CSR"):
        g = g.tocsr()

    with Tracer("Creating dummy features for nodes"):
        feats = np.ones((r, 1))

    with Tracer("Loading Walks", walks_file=walks_file):
        walks = walks_gen(walks_file)

    # Return graph, features, idmap, walks, labels
    return g, feats, None, walks, None


if __name__ == '__main__':
    g, feats, walks = load_data('../example_mtx/medium')
    print("Got g with shape: " + str(g.get_shape()))
    print(feats)
    print("Got walks" + str([w for w in walks]))
