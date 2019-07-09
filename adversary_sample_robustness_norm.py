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
            s1_fname = directory + str(ad_p_remove) + '/' + \
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

                s2_fname = directory + str(ad_p_remove) + '/' + \
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

def compute_eigenvals(G, name, laplacian=False):
	folder = './eigvals/'

	if not os.path.exists(folder):
		os.makedirs(folder)

	if laplacian:
		file_path = folder + 'N_eigvals_' + name + '.pickle'
	else:
		file_path = folder + 'A_eigvals_' + name + '.pickle'

	if os.path.isfile(file_path):
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

	return e

def get_properties(G, dataset):
    prop = dict()
    prop['n'] = G.number_of_nodes()
    prop['m'] = G.number_of_edges()

    deg = dict(G.degree())
    prop['deg_avg'] = np.mean(deg.values())
    prop['deg_max'] = max(deg.values())
    prop['deg_med'] = np.median(deg.values())

    e = compute_eigenvals(G, dataset)
    prop['e_largest'] = e[-1]
    prop['cc'] = nx.average_clustering(G)

    partition = get_community(G, dataset, com_path='./graph_properties/')
    Q = community.modularity(partition, G)
    prop['q'] = Q

    return prop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', help='dataset name', type=str)
    parser.add_argument('fname', help='Edgelist file', type=str)
    parser.add_argument('-remove', help='node or edge', type=str, default='edge_incomp')
    args = parser.parse_args()

    fname = args.fname
    fname = fname.replace('\\', '/')
    dataset = fname.split('.')[1].split('/')[-1]

    # dataset = args.dataset
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
        # nodes_original = pickle.load(open(directory + dataset + '_nodes.pickle'))

    log_file = './log/' + dataset + '_' + remove_type

    MAX_GEN = 10
    # crawler_list = ['med','mod','rw', 'bfs', 'rand']
    crawler_list = ['mod', 'rw', 'bfs', 'rand']
    robustness_zero = dict()

    # calculate robustness of different samples without removing its edges.
    pickle_zero_fname = directory + '/0/' + dataset + '/robustness_zero.pickle'
    if os.path.isfile(pickle_zero_fname):
        robustness_zero = pickle.load(open(pickle_zero_fname, 'rb'))
    else:
        robustness_zero = get_robustness_zero(dataset)
        pickle.dump(robustness_zero, open(pickle_zero_fname, 'wb'))
    print('-- Zero done -- ')

    ##### Get graph properties #####
    graph_properties_fname = './graph_properties/' + dataset + '.pickle'
    if os.path.isfile(graph_properties_fname):
        graph_prop = pickle.load(open(graph_properties_fname, 'rb'))
    else:
        graph = _mylib.read_file(fname)
        graph_prop = get_properties(graph, dataset)
        pickle.dump(graph_prop, open(graph_properties_fname, 'wb'))


    # for crawler_type, v in robustness_zero.iteritems():
    #    for key in v.keys():
    #         log.save_to_file_line('./log/'+remove_type+'_robustness_zero.txt', [dataset, crawler_type, "Node Sim", v[key]['node_sim']])
    #         log.save_to_file_line('./log/'+remove_type+'_robustness_zero.txt', [dataset, crawler_type, "Node Coverage", v[key]['node_cov_sim']])
    #         log.save_to_file_line('./log/'+remove_type+'_robustness_zero.txt', [dataset, crawler_type, "Partition Sim", v[key]['partition_sim']])
    #         log.save_to_file_line('./log/'+remove_type+'_robustness_zero.txt', [dataset, crawler_type, "Degree Dist.", v[key]['degree_sim']])


    for crawler_type in crawler_list:
        for i in range(0, MAX_GEN):
            s1_fname = directory + '0' + '/' + \
                       dataset + '/' + crawler_type + '/' + \
                       dataset + '_' + str(i + 1) + '_1'
            sample_graph_1 = _mylib.read_file(s1_fname)

            # for ad_p_remove in ["0.01", "0.05", "0.1", "0.2", "0.3"]:
            for ad_p_remove in ["0.1", "0.2", "0.3", "0.4", "0.5"]:
                s2_fname = directory + ad_p_remove + '/' + \
                           dataset + '/' + crawler_type + '/' + \
                           dataset + '_' + str(i + 1) + '_1'
                sample_graph_2 = _mylib.read_file(s2_fname)
                deleted_nodes_count = int(float(ad_p_remove) * len(nodes_original))
                # print(' Delete nodes count: {}/{} \t {}'.format(deleted_nodes_count, len(nodes_original), ad_p_remove))

                sampling_robustness = calculateRobustness(sample_graph_1, sample_graph_2,
                                                        s1_fname, s2_fname, deleted_nodes_count)

                node_sim = sampling_robustness['node_sim'] / robustness_zero[crawler_type][i]['node_sim']
                node_cov_sim = sampling_robustness['node_coverage_sim'] / robustness_zero[crawler_type][i]['node_cov_sim']
                partition_sim = sampling_robustness['partition_sim'] / robustness_zero[crawler_type][i]['partition_sim']
                degree_sim = sampling_robustness['degree_sim'] / robustness_zero[crawler_type][i]['degree_sim']

                # print('type:{} i:{} \t Norm:{} \t unnorm: {} \t zero:{}'.format(crawler_type, i, node_sim, sampling_robustness['node_sim'] , robustness_zero[crawler_type][i]['node_sim']))

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


                out_fname = './log/agg_testing_' + remove_type + '.txt'

                if not os.path.isfile(out_fname):
                    log.save_to_file_line(out_fname,['dataset', 'p', 'crawler', 'measure', 'R',
                                                     'n', 'm', 'deg_avg', 'deg_med', 'deg_max',
                                                     'cc', 'q', 'e'])

                log.save_to_file_line(out_fname, [dataset, ad_p_remove, crawler_type, "Node Sim", node_sim,
                                                             graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                             graph_prop['deg_med'],
                                                             graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                             graph_prop['e_largest']])
                log.save_to_file_line(out_fname, [dataset, ad_p_remove, crawler_type, "Node Coverage", node_cov_sim,
                                                             graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                             graph_prop['deg_med'],
                                                             graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                             graph_prop['e_largest']])

                log.save_to_file_line(out_fname, [dataset, ad_p_remove, crawler_type, "Partition Sim", partition_sim,
                                                             graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                             graph_prop['deg_med'],
                                                             graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                             graph_prop['e_largest']])
                log.save_to_file_line(out_fname, [dataset, ad_p_remove, crawler_type, "Degree Dist.", degree_sim,
                                                             graph_prop['n'], graph_prop['m'], graph_prop['deg_avg'],
                                                             graph_prop['deg_med'],
                                                             graph_prop['deg_max'], graph_prop['cc'], graph_prop['q'],
                                                             graph_prop['e_largest']])

