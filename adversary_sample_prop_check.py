import os
import _mylib
import community
import pickle
import log
import numpy as np
import argparse
import networkx as nx
from scipy import stats
import adversary_sample_robustness_norm as adv_norm

def get_community(G, fname):
    com_fname = fname + '_com'
    if os.path.isfile(com_fname):
        partition = pickle.load(open(com_fname, 'rb'))
    else:
        partition = community.best_partition(G)
        pickle.dump(partition, open(com_fname, 'wb'))

    return partition

def _getCommunity(partition):
    com = {}

    for n in partition:
        p = partition[n]
        if p not in com:
            com[p] = set()
        com[p].update(set([n]))

    com = {c: com[c] for c in com if len(com[c]) > 1}

    # Make sure we do not consider the singleton nodes
    return com

def partitionDistance(part1, part2, nodes=None):
    """
    Compute the partiton distance between communities c1 and c2
    """

    c1 = _getCommunity(part1)
    c2 = _getCommunity(part2)

    if nodes is None:
        n1 = set([])
        n2 = set([])
        for c in c1:
            n1.update(c1[c])
        for c in c2:
            n2.update(c2[c])
        nodes = n1.intersection(n2)

    c1 = {c: c1[c].intersection(nodes) for c in c1}
    c2 = {c: c2[c].intersection(nodes) for c in c2}

    m = max(len(c1), len(c2))
    m = range(0, m)

    mat = {i: {j: 0 for j in c2} for i in c1}

    total = 0
    for i in c1:
        for j in c2:
            if i in c1 and j in c2:
                mat[i][j] = len(c1[i].intersection(c2[j]))
                total += mat[i][j]

    if total <= 1:
        return 1.0

    assignment = []
    rows = c1.keys()
    cols = c2.keys()

    while len(rows) > 0 and len(cols) > 0:
        mval = 0
        r = -1
        c = -1
        for i in rows:
            for j in cols:
                if mat[i][j] >= mval:
                    mval = mat[i][j]
                    r = i
                    c = j
        rows.remove(r)
        cols.remove(c)
        assignment.append(mval)

    dist = total - np.sum(assignment)

    if np.isnan(dist / total):
        return 0

    return 1.*dist / total

def calculate_community_sim(sample_graph_1, sample_graph_2, sample_fname_1, sample_fname_2, deleted_nodes_count=0):

    s1_communities = get_community(sample_graph_1, sample_fname_1)
    s2_communities = get_community(sample_graph_2, sample_fname_2)


    partition_sim = 1. - partitionDistance(s1_communities, s2_communities)

    return partition_sim


def compute_eigenvals(G, graph_fname, laplacian=False):

    if laplacian:
        graph_fname = graph_fname + '_eigvals_N'
    else:
        graph_fname = graph_fname + '_eigvals_A'

	if os.path.isfile(graph_fname):
		e = pickle.load(open(graph_fname, 'rb'))
	else:
		if laplacian:
			L = nx.normalized_laplacian_matrix(G)
		else:
			L = nx.adjacency_matrix(G)

		print('Computing eigenvalues .. {}'.format(graph_fname))
		e = np.linalg.eigvalsh(L.A)
        e.sort()
        pickle.dump(e, open(graph_fname, 'wb'))

	return e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', help='dataset name', type=str)
    parser.add_argument('fname', help='Edgelist file', type=str)
    parser.add_argument('-remove', help='node or edge', type=str, default='edge_incomp')
    parser.add_argument('-train', help='node or edge', type=str, default='train')
    args = parser.parse_args()

    fname = args.fname
    remove_type = args.remove
    training = args.train

    fname = fname.replace('\\', '/')
    dataset = fname.split('.')[1].split('/')[-1]

    if remove_type == 'edge':
        directory = './data-adv/rand_edge/'
        nodes_original = set()
    elif remove_type == 'node':
        directory = './data-adv/rand_node/'
        nodes_original = pickle.load(open(directory + dataset + '_nodes.pickle'))
    elif remove_type == 'target':
        directory = './data-adv/target/'
        nodes_original = pickle.load(open(directory + dataset + '_nodes.pickle'))
    elif remove_type == 'edge_incomp':
        directory = './data-adv/edge_incomp/'
        nodes_original = set()

    log_file = './log/' + dataset + '_' + remove_type

    MAX_GEN = 10
    # crawler_list = ['med','mod','rw', 'bfs', 'rand']
    crawler_list = ['mod', 'rw', 'bfs', 'rand']

    G_original = _mylib.read_file(fname)
    deg = dict(G_original.degree())
    avg_deg = np.average(deg.values())
    e = adv_norm.compute_eigenvals(G_original, dataset)
    max_e = e[-1]
    cc = nx.average_clustering(G_original)

    for crawler_type in crawler_list:
        for i in range(0, MAX_GEN):
            s1_fname = directory + '0' + '/' + \
                       dataset + '/' + crawler_type + '/' + \
                       dataset + '_' + str(i + 1) + '_1'
            sample_graph_1 = _mylib.read_file(s1_fname)
            deg = dict(sample_graph_1.degree())
            avg_deg_0 = np.average(deg.values())

            # e = compute_eigenvals(sample_graph_1, s1_fname)
            # max_e_0 = e[-1]
            # cc_0 = nx.average_clustering(sample_graph_1)
            #
            size_ = sample_graph_1.number_of_nodes()

            # for ad_p_remove in ["0.01", "0.05", "0.1", "0.2", "0.3"]:
            for ad_p_remove in ["0.1", "0.2", "0.3","0.4", "0.5"]:
                s2_fname = directory + ad_p_remove + '/' + \
                           dataset + '/' + crawler_type + '/' + \
                           dataset + '_' + str(i + 1) + '_1'
                sample_graph_2 = _mylib.read_file(s2_fname)
                deleted_nodes_count = int(float(ad_p_remove) * len(nodes_original))

                deg = dict(sample_graph_2.degree())
                avg_deg_p = np.average(deg.values())

                # e = compute_eigenvals(sample_graph_2, s2_fname)
                # max_e_p = e[-1]
                # cc_p = nx.average_clustering(sample_graph_2)

                size_2 = sample_graph_2.number_of_nodes()

                print(' c:{} \t p:{} \t e:{} {}'.format(crawler_type, ad_p_remove, max_e_0, max_e_p ))

                # out_fname = './log/' + remove_type + '_sample_eigval.txt'

                if training == 'train':
                    out_fname = './log/sample-traning-properties.txt'
                else:
                    out_fname = './log/sample-testing-properties.txt'

                if not os.path.isfile(out_fname):
                    log.save_to_file_line(out_fname,['p', 'crawler', 'dataset', 'd', 'd_samp_0', 'd_samp_p', 'e', 'e_samp_0', 'e_samp_p', 'cc', 'cc_samp_0', 'cc_samp_p'])


                # log.save_to_file_line(out_fname, [ad_p_remove, crawler_type, dataset,
                                                  avg_deg, avg_deg_0, avg_deg_p,
                                                  max_e, max_e_0, max_e_p,
                                                  cc, cc_0, cc_p])
