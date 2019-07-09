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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset name', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    directory = './data-adv/rand_edge/'
    # dataset = 'socfb-Amherst41'
    log_file = './log/' + dataset

    MAX_GEN = 10
    ad_p_remove = "0"
    crawler_list = ['med', 'mod', 'rw', 'bfs', 'rand']
    # crawler_list = ['mod']

    for crawler_type in crawler_list:
        for i in range(0, MAX_GEN):
            fname = directory + str(ad_p_remove) + '/' + \
                    dataset + '/' + crawler_type + '/' + \
                    dataset + '_' + str(i+1) + '_1'
            G = _mylib.read_file(fname)
            node_count = G.number_of_nodes()

            com_G = get_community(G, fname)
            node_G = G.nodes()
            deg_hist_G = nx.degree_histogram(G)

            for j in range(0, MAX_GEN):
                if i == j:
                    continue

                fname = directory + str(ad_p_remove) + '/' + \
                        dataset + '/' + crawler_type + '/' + \
                        dataset + '_' + str(j + 1) + '_1'

                G = _mylib.read_file(fname)
                com_adv = get_community(G, fname)
                node_adv = G.nodes()
                deg_hist_adv = nx.degree_histogram(G)
                d, p = stats.ks_2samp(deg_hist_G, deg_hist_adv)

                int_ = len(set(node_G).intersection(set(node_adv)))
                union_ = len(set(node_G).union(set(node_adv)))

                jaccard = 1.*int_ / union_
                # node_cov_size = 1.*len(node_adv) / len(node_G)
                coverage_dist = 1.*(abs(len(node_adv) - len(node_G))) / (abs(len(node_adv)) + abs(len(node_G)))
                node_cov_sim = 1. - coverage_dist
                partition_sim = 1. - partitionDistance(com_G, com_adv)
                degree_dist, p_val = stats.ks_2samp(deg_hist_G, deg_hist_adv)

                print('Type {} \t i-j:{}-{} \t p:{} \t '
                      'Coverage:{} \t'
                      ' Node sim:{} \t '
                      'Partiton sim:{} \t '
                      'Deg: {}  p:{}'.format(
                    crawler_type, i, j,
                    ad_p_remove,
                    node_cov_sim,
                    jaccard,
                    partition_sim,
                    degree_dist,
                    p_val))

                log.save_to_file_line(log_file + '_0_measure.txt', [ad_p_remove, crawler_type, "Node Sim", jaccard])
                log.save_to_file_line(log_file + '_0_measure.txt', [ad_p_remove, crawler_type, "Node Coverage", node_cov_sim])
                log.save_to_file_line(log_file + '_0_measure.txt', [ad_p_remove, crawler_type, "Partition Sim", partition_sim])
                log.save_to_file_line(log_file + '_0_measure.txt', [ad_p_remove, crawler_type, "Degree Dist.", degree_dist])
                log.save_to_file_line(log_file + '_0_measure.txt', [ad_p_remove, crawler_type, "p", p_val])
