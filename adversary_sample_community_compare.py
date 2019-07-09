import os
import _mylib
import community
import pickle
import log
import numpy as np
import argparse
import networkx as nx
from scipy import stats

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset name', type=str)
    parser.add_argument('-remove', help='node or edge', type=str, default='edge_incomp')
    args = parser.parse_args()

    dataset = args.dataset
    remove_type = args.remove

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



    for crawler_type in crawler_list:
        for i in range(0, MAX_GEN):
            s1_fname = directory + '0' + '/' + \
                       dataset + '/' + crawler_type + '/' + \
                       dataset + '_' + str(i + 1) + '_1'
            sample_graph_1 = _mylib.read_file(s1_fname)

            # for ad_p_remove in ["0.01", "0.05", "0.1", "0.2", "0.3"]:
            for ad_p_remove in ["0.1", "0.2", "0.3","0.4", "0.5"]:
                s2_fname = directory + ad_p_remove + '/' + \
                           dataset + '/' + crawler_type + '/' + \
                           dataset + '_' + str(i + 1) + '_1'
                sample_graph_2 = _mylib.read_file(s2_fname)
                deleted_nodes_count = int(float(ad_p_remove) * len(nodes_original))
                # print(' Delete nodes count: {}/{} \t {}'.format(deleted_nodes_count, len(nodes_original), ad_p_remove))

                partition_sim = calculate_community_sim(sample_graph_1, sample_graph_2,
                                                        s1_fname, s2_fname, deleted_nodes_count)

                print("{} \t {} \t {}".format(s1_fname, s2_fname, partition_sim))

                log.save_to_file_line('./log/' + remove_type + '_com_sim.txt', [ad_p_remove, crawler_type, dataset, partition_sim])
                # log.save_to_file_line(log_file + '_robustness_norm.txt', [ad_p_remove, crawler_type, "Node Coverage", node_cov_sim])
                # log.save_to_file_line(log_file + '_robustness_norm.txt', [ad_p_remove, crawler_type, "Partition Sim", partition_sim])
                # log.save_to_file_line(log_file + '_robustness_norm.txt', [ad_p_remove, crawler_type, "Degree Dist.", degree_sim])

