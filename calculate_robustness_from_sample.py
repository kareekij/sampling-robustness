import os
import _mylib
import community
import pickle
import log
import numpy as np
import argparse
import networkx as nx
from scipy import stats

def get_community(G, fname, com_path=None):
    if com_path == None:
        com_fname = fname + '_com'
    else:
        com_fname = com_path + 'com_' + fname

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

def calculateRobustness(sample_graph_1, sample_graph_2, sample_fname_1, sample_fname_2, deleted_nodes_count=0):
    s1_node_count = sample_graph_1.number_of_nodes()
    s1_communities = get_community(sample_graph_1, sample_fname_1)
    s1_nodes = sample_graph_1.nodes()
    s1_deg_hist = nx.degree_histogram(sample_graph_1)

    s2_node_count = sample_graph_2.number_of_nodes()
    s2_communities = get_community(sample_graph_2, sample_fname_2)
    s2_nodes = sample_graph_2.nodes()
    s2_deg_hist = nx.degree_histogram(sample_graph_2)


    s1_node_count = s1_node_count - deleted_nodes_count

    ret = dict()
    int_ = len(set(s1_nodes).intersection(set(s2_nodes)))
    union_ = len(set(s1_nodes).union(set(s2_nodes)))
    degree_dist, p_val = stats.ks_2samp(s1_deg_hist, s2_deg_hist)

    ret['node_sim'] = 1. * int_ / union_
    ret['node_coverage_sim'] = 1 - (1.*(abs(s1_node_count - s2_node_count)) / (abs(s1_node_count) + abs(s2_node_count)))
    ret['partition_sim'] = 1. - partitionDistance(s1_communities, s2_communities)
    ret['degree_sim'] = 1. - degree_dist
    ret['p_val'] = p_val

    return ret

def get_robustness_zero(dataset):
    for crawler_type in crawler_list:
        robustness_zero[crawler_type] = dict()
        for i in range(0, MAX_GEN):
            robustness_zero[crawler_type][i] = dict()
            ad_p_remove = "0"
            s1_fname = directory + '/' + str(ad_p_remove) + '/' + \
                       dataset + '/' + crawler_type + '/' + \
                       dataset + '_' + str(i + 1) + '_1'
            sample_graph_1 = _mylib.read_file(s1_fname)

            node_sim_list = []
            node_cov_sim_list = []
            partition_sim_list = []
            degree_sim_list = []

            for j in range(0, MAX_GEN):
                if i == j:
                    continue

                s2_fname = directory + '/' + str(ad_p_remove) + '/' + \
                           dataset + '/' + crawler_type + '/' + \
                           dataset + '_' + str(j + 1) + '_1'
                sample_graph_2 = _mylib.read_file(s2_fname)

                sample_robustness = calculateRobustness(sample_graph_1, sample_graph_2,
                                                        s1_fname, s2_fname)

                node_sim_list.append(sample_robustness['node_sim'])
                node_cov_sim_list.append(sample_robustness['node_coverage_sim'])
                partition_sim_list.append(sample_robustness['partition_sim'])
                degree_sim_list.append(sample_robustness['degree_sim'])

            robustness_zero[crawler_type][i]['node_sim'] = 1. * sum(node_sim_list) / len(node_sim_list)
            robustness_zero[crawler_type][i]['node_cov_sim'] = 1. * sum(node_cov_sim_list) / len(node_cov_sim_list)
            robustness_zero[crawler_type][i]['partition_sim'] = 1. * sum(partition_sim_list) / len(partition_sim_list)
            robustness_zero[crawler_type][i]['degree_sim'] = 1. * sum(degree_sim_list) / len(degree_sim_list)

    return robustness_zero

def compute_eigenvals(G, name, laplacian=False, folder = './eigvals/'):

    if not os.path.exists(folder):
        os.makedirs(folder)

    if laplacian:
        file_path = folder + 'N_eigvals_' + name + '.pickle'
    else:
        file_path = folder + 'A_eigvals_' + name + '.pickle'

    if os.path.isfile(file_path):
        print(' [Eigval] Read from {}'.format(file_path))
        e = pickle.load(open(file_path, 'rb'))
    else:
        if laplacian:
            L = nx.normalized_laplacian_matrix(G)
        else:
            L = nx.adjacency_matrix(G)

        print('Computing eigenvalues .. {}'.format(name))
        e = np.linalg.eigvalsh(L.A)
        e.sort()
        pickle.dump(e, open(file_path, 'wb'))
        print(' [Eigval] Save to {}'.format(file_path))

    return e

def get_properties(G, dataset, folder='./graph_properties/'):
    prop = dict()
    prop['n'] = G.number_of_nodes()
    prop['m'] = G.number_of_edges()

    deg = dict(G.degree())
    prop['deg_avg'] = np.mean(deg.values())
    prop['deg_max'] = max(deg.values())
    prop['deg_med'] = np.median(deg.values())

    e = compute_eigenvals(G, dataset, folder=folder)
    prop['e_largest'] = e[-1]
    prop['cc'] = nx.average_clustering(G)

    partition = get_community(G, dataset, com_path=folder)
    Q = community.modularity(partition, G)
    prop['q'] = Q

    return prop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', help='dataset name', type=str)
    parser.add_argument('fname', help='Edgelist file', type=str)
    parser.add_argument('-remove', help='node or edge', type=str, default='edge_incomp')
    parser.add_argument('-log', help='output folder', type=str, default='./log/')
    parser.add_argument('-output_fn', help='output filename', type=str, default='output_robustness.txt')
    parser.add_argument('-sample_dir', help='folder contains sample', type=str, default='./samples')
    args = parser.parse_args()

    fname = args.fname
    fname = fname.replace('\\', '/')
    dataset = fname.split('.')[1].split('/')[-1]

    remove_type = args.remove
    log_folder = args.log
    output_fname = args.output_fn
    output = log_folder + output_fname

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)


    # directory = './data-adv/edge_incomp/'
    directory = args.sample_dir
    nodes_original = set()
        # nodes_original = pickle.load(open(directory + dataset + '_nodes.pickle'))



    MAX_GEN = 10
    crawler_list = ['mod', 'rw', 'bfs', 'rand']
    robustness_zero = dict()

    ######################################################################
    # calculate robustness of different samples when p = 0
    # When calculated, it keeps the output in a pickle format for later uses.
    # ./robustness_zero.pickle
    ######################################################################
    pickle_zero_fname = directory + '/0/' + dataset + '/robustness_zero.pickle'
    if os.path.isfile(pickle_zero_fname):
        robustness_zero = pickle.load(open(pickle_zero_fname, 'rb'))
    else:
        robustness_zero = get_robustness_zero(dataset)
        pickle.dump(robustness_zero, open(pickle_zero_fname, 'wb'))
    print('-- Zero done -- ')

    ########################################################################
    # Calculate `original` graph properties and keep the output as a pickle file for later uses.
    # Avg/Median/Max degree, Clustering Coefficient, Largest Eigenvalue, Modularity
    ########################################################################

    if not os.path.exists('./graph_properties'):
        os.makedirs('./graph_properties')

    graph_properties_fname = './graph_properties/' + dataset + '.pickle'

    if os.path.isfile(graph_properties_fname):
        graph_prop = pickle.load(open(graph_properties_fname, 'rb'))
    else:
        graph = _mylib.read_file(fname)
        graph_prop = get_properties(graph, dataset)
        pickle.dump(graph_prop, open(graph_properties_fname, 'wb'))


    ########################################################################
    # Calculate Sampling Robustness from samples
    ########################################################################

    for crawler_type in crawler_list:
        for i in range(0, MAX_GEN):
            s1_fname = directory + '/0/' + '/' + \
                       dataset + '/' + crawler_type + '/' + \
                       dataset + '_' + str(i + 1) + '_1'
            sample_graph_1 = _mylib.read_file(s1_fname)

            for ad_p_remove in ["0.1", "0.2", "0.3", "0.4", "0.5"]:
                s2_fname = directory + '/' + ad_p_remove + '/' + \
                           dataset + '/' + crawler_type + '/' + \
                           dataset + '_' + str(i + 1) + '_1'
                sample_name = dataset + '_' + crawler_type + '_' + ad_p_remove + '_' + str(i + 1) + '_1'
                sample_graph_2 = _mylib.read_file(s2_fname)
                deleted_nodes_count = int(float(ad_p_remove) * len(nodes_original))

                ########################################################################
                # Calculate Sampling Robustness
                ########################################################################
                sampling_robustness = calculateRobustness(sample_graph_1, sample_graph_2,
                                                        s1_fname, s2_fname, deleted_nodes_count)
                # Normalize with S_0
                node_sim = sampling_robustness['node_sim'] / robustness_zero[crawler_type][i]['node_sim']
                node_cov_sim = sampling_robustness['node_coverage_sim'] / robustness_zero[crawler_type][i]['node_cov_sim']
                partition_sim = sampling_robustness['partition_sim'] / robustness_zero[crawler_type][i]['partition_sim']
                degree_sim = sampling_robustness['degree_sim'] / robustness_zero[crawler_type][i]['degree_sim']
                ###############################################################

                ###############################################################
                # Calculate properties of the sample
                ###############################################################
                sample_prop = get_properties(sample_graph_2, sample_name)


                print('Type {} \t i:{} \t p:{} \t '
                      'Coverage:{} \t'
                      ' Node sim:{} \t '
                      'Partiton sim:{} \t '
                      'Deg: {}'.format(
                    crawler_type, i,
                    ad_p_remove,
                    node_cov_sim,
                    node_sim,
                    partition_sim,
                    degree_sim))


                # out_fname = './log/agg_testing_' + remove_type + '.txt'

                if not os.path.isfile(output):
                    log.save_to_file_line(output,['dataset', 'p', 'crawler', 'measure', 'R',
                                                     'g_node', 'g_edges', 'g_deg_avg',
                                                  'g_deg_med', 'g_deg_max',
                                                     'g_cc', 'g_q', 'g_e',
                                                  's_node', 's_edges', 's_deg_avg',
                                                  's_deg_med', 's_deg_max',
                                                  's_cc', 's_q', 's_e'
                                                  ])

                log.save_to_file_line(output, [dataset, ad_p_remove, crawler_type, "Node Sim", node_sim,
                                                 graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                 graph_prop['deg_med'],
                                                 graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                 graph_prop['e_largest'],
                                               sample_prop['n'], sample_prop['m'], sample_prop['deg_avg'],
                                               sample_prop['deg_med'],
                                               sample_prop['deg_max'], sample_prop['cc'], sample_prop['q'],
                                               sample_prop['e_largest']
                                               ])
                log.save_to_file_line(output, [dataset, ad_p_remove, crawler_type, "Node Coverage", node_cov_sim,
                                                             graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                             graph_prop['deg_med'],
                                                             graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                             graph_prop['e_largest'],
                                               sample_prop['n'], sample_prop['m'], sample_prop['deg_avg'],
                                               sample_prop['deg_med'],
                                               sample_prop['deg_max'], sample_prop['cc'], sample_prop['q'],
                                               sample_prop['e_largest']
                                            ])

                log.save_to_file_line(output, [dataset, ad_p_remove, crawler_type, "Partition Sim", partition_sim,
                                                             graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                             graph_prop['deg_med'],
                                                             graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                             graph_prop['e_largest'],
                                               sample_prop['n'], sample_prop['m'], sample_prop['deg_avg'],
                                               sample_prop['deg_med'],
                                               sample_prop['deg_max'], sample_prop['cc'], sample_prop['q'],
                                               sample_prop['e_largest']

                                               ])
                log.save_to_file_line(output, [dataset, ad_p_remove, crawler_type, "Degree Dist.", degree_sim,
                                                             graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                             graph_prop['deg_med'],
                                                             graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                             graph_prop['e_largest'],
                                               sample_prop['n'], sample_prop['m'], sample_prop['deg_avg'],
                                               sample_prop['deg_med'],
                                               sample_prop['deg_max'], sample_prop['cc'], sample_prop['q'],
                                               sample_prop['e_largest']

                                               ])

