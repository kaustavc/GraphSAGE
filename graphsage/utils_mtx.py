import numpy as np
import pickle
from graphsage.tracer import Tracer


def walks_gen(walks_file):
    with open(walks_file) as wf:
        for line in wf:
            yield([int(n) for n in line.split()])


def load_data(mtx_file, walks_file):
    with Tracer("Loading adjacency matrix", mtx_file=mtx_file):
        fin = open(mtx_file, 'rb')
        g = pickle.load(fin)

    with Tracer("Creating dummy features for nodes"):
        feats = np.ones((r, 1))

    with Tracer("Loading Walks", walks_file=walks_file):
        walks = walks_gen(walks_file)

    # Return graph, features, idmap, walks, labels
    return g, feats, walks


if __name__ == '__main__':
    g, feats, walks = load_data('../example_mtx/medium-G.mtx', '../example_mtx/medium-walks.txt')
    print("Got g with shape: " + str(g.get_shape()))
    print(feats)
    print("Got walks" + str([w for w in walks]))
