from __future__ import print_function
import networkx as nx
import _mylib
import random
import argparse
import os
import collections
import matplotlib.pyplot as plt
import numpy as np


def gen_adversary_graph(G, type='hub', p=0.01):
    if type == 'hub':
        print(' [ADV] remove `hub` {} percent'.format(p))
        G_p = delete_nodes(G, percent_remove=p)
    elif type == "node":
        print(' [ADV] remove `random nodes` {} percent'.format(p))
        G_p = delete_random_nodes(G, percent_remove=p)
    elif type == "edge":
        print(' [ADV] remove `random edges` {} percent'.format(p))
        G_p = delete_random_edges(G, percent_remove=p)

    return G_p

def delete_nodes(G, type='deg', percent_remove = 0.01):

    total_nodes = G.number_of_nodes()
    remove_count = int(percent_remove * total_nodes)

    print(' Adversary removes {} nodes'.format(remove_count))

    if type == 'deg':
        deg = dict(G.degree())
        deg_sorted = _mylib.sortDictByValues(deg, reverse=True)
        l = deg_sorted
    elif type == 'pr':
        pr = dict(nx.pagerank(G))
        pr_sorted = _mylib.sortDictByValues(pr, reverse=True)
        l = pr_sorted
    elif type == 'betweeness':
        betweeness = dict(nx.betweenness_centrality(G))
        betweeness_sorted = _mylib.sortDictByValues(betweeness, reverse=True)
        l = betweeness_sorted

    G_copy = G.copy()

    for idx, i in enumerate(l):
        if idx == remove_count:
            break
        node = i[0]
        d = i[1]
        G_copy.remove_node(node)
        print(' {} Remove node {}'.format(idx, node))

    return G_copy

def delete_random_nodes(G, percent_remove=.01):
    total_nodes = G.number_of_nodes()
    remove_count = int(percent_remove * total_nodes)
    print(' Remove {} nodes'.format(remove_count))

    to_be_deleted = random.sample(list(G.nodes()), remove_count)
    G_copy = G.copy()
    for n in to_be_deleted:
        G_copy.remove_node(n)

    print(' New graph has {} nodes'.format(G_copy.number_of_nodes()))
    return G_copy

def delete_random_edges(G, percent_remove=.01):
    total_edges = G.number_of_edges()
    remove_count = int(percent_remove * total_edges)
    print(remove_count)
    to_be_deleted = random.sample(list(G.edges()), remove_count)
    G_copy = G.copy()

    G_copy.remove_edges_from(to_be_deleted)

    return G_copy

def delete_random_edges_connect(G, percent_remove=.01):
    total_edges = G.number_of_edges()
    remove_count = int(percent_remove * total_edges)
    print('Remove total:' + str(remove_count))
    G_copy = G.copy()

    i=0
    while i < remove_count:
        e = random.choice(list(G_copy.edges()))
        node_a = e[0]
        node_b = e[1]

        deg_a = G_copy.degree(node_a)
        deg_b = G_copy.degree(node_b)

        if deg_a == 1 or deg_b == 1:
           # print('Cannot delete this edge - deg')
           continue

        G_copy.remove_edge(node_a, node_b)
        isConnected = nx.is_connected(G_copy)

        if not isConnected:
            G_copy.add_edge(node_a, node_b)
            print('Cannot delete this edge ' + str(i))
        else:
            i+=1

    return G_copy

def delete_random_edges_connect_optimized(G, percent_remove=.01):
    total_edges = G.number_of_edges()
    remove_count = int(percent_remove * total_edges)
    print('Remove total:' + str(remove_count))
    G_copy = G.copy()

    degree_ = dict(nx.degree(G_copy))
    node_degree_one = list()

    for node, deg in degree_.iteritems():
        if deg == 1:
            node_degree_one.append(node)

    i=0
    one_degree_edges = set()
    while i < remove_count:
        all_edge = set(G_copy.edges())
        candidate_e = list(all_edge - one_degree_edges)

        if len(candidate_e) == 0:
            print('No more edge to remove')
            break

        e = random.choice(candidate_e)
        node_a = e[0]
        node_b = e[1]

        if (node_a in node_degree_one) or (node_b in node_degree_one):
            one_degree_edges.add(e)
            print(' [DEG1] Cannot delete this edge {} / {}'.format(i, remove_count))
            continue

        G_tmp = G_copy.copy()
        G_tmp.remove_edge(node_a, node_b)
        G_tmp.remove_nodes_from(node_degree_one)

        # print(' check connected {}'.format(i))
        isConnected = nx.is_connected(G_tmp)
        # print(' connected {}'.format(isConnected))

        if isConnected:
            G_copy.remove_edge(node_a, node_b)

            degree_[node_a] = degree_.get(node_a) - 1
            degree_[node_b] = degree_.get(node_b) - 1

            if degree_[node_a] == 1:
                node_degree_one.append(node_a)
            if degree_[node_b] == 1:
                node_degree_one.append(node_b)

            i += 1
        else:
            print(' Cannot delete this edge {} / {}'.format(i, remove_count))



    return G_copy

def delete_random_nodes_connect(G, percent_remove=.01):
    total_node = G.number_of_nodes()
    remove_count = int(percent_remove * total_node)
    print('Remove total:' + str(remove_count))
    G_copy = G.copy()
    i = 0

    while i < remove_count:
        all_nodes = set(G_copy.nodes())
        candidates = list(all_nodes)

        selected_node = random.choice(candidates)

        G_tmp = G_copy.copy()
        G_tmp.remove_node(selected_node)

        isConnected = nx.is_connected(G_tmp)

        if isConnected:
            G_copy.remove_node(selected_node)
            i+=1
        else:
            print('Cannot remove this node \t {} / {}'.format(i, remove_count))

    return G_copy

def delete_hub_nodes_connect(G, percent_remove=.01):
    total_node = G.number_of_nodes()
    remove_count = int(percent_remove * total_node)
    print('Remove total:' + str(remove_count))
    G_copy = G.copy()
    i = 0

    possible_nodes = set(G_copy.nodes())

    while i < remove_count:
        deg = dict(G_copy.degree(possible_nodes))
        deg_sorted = _mylib.sortDictByValues(deg, reverse=True)
        selected_node = deg_sorted[0][0]


        G_tmp = G_copy.copy()
        G_tmp.remove_node(selected_node)

        isConnected = nx.is_connected(G_tmp)

        if isConnected:
            G_copy.remove_node(selected_node)
            i+=1
        else:
            print('Cannot remove this node \t {} / {}'.format(i, remove_count))
        possible_nodes.remove(selected_node)


    return G_copy

def genGraph(type):
    if type == 'edge':
        G_p = delete_random_edges_connect_optimized(G, percent_remove=ad_p_remove)
    elif type == 'node':
        G_p = delete_random_nodes_connect(G, percent_remove=ad_p_remove)
    elif type == 'target':
        G_p = delete_hub_nodes_connect(G, percent_remove=ad_p_remove)

    return G_p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='Edgelist file', type=str)
    parser.add_argument('-remove', help='remove node or edge', type=str, default='edge')

    args = parser.parse_args()
    fname = args.fname
    remove_type = args.remove

    fname = fname.replace('\\', '/')
    network_name = fname.split('.')[1].split('/')[-1]

    G = _mylib.read_file(fname, isDirected=False)
    print('-----' * 10)
    deg = dict(G.degree())

    med_deg = np.median(deg.values())
    avg_deg = np.average(deg.values())
    print(med_deg, avg_deg)


    # cc_count = nx.number_connected_components(G)
    # print('Network: {} \t CC: {} \t E: {}'.format(network_name, cc_count, G.number_of_edges()))
    # print(nx.info(G))
    # # deg = G.degree()
    #
    # low = min(dict(deg).values())
    # hi = max(dict(deg).values())
    # med = np.median(np.array(dict(deg).values()))
    #
    # print(low, hi, med)
    #

    # print(nx.average_clustering(G))


    # degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print(max(degree_sequence))
    # # print "Degree sequence", degree_sequence
    # degreeCount = collections.Counter(degree_sequence)
    # deg, cnt = zip(*degreeCount.items())
    #
    # fig, ax = plt.subplots()
    # plt.bar(deg, cnt, width=0.80, color='b')
    #
    # plt.title("Degree Histogram")
    # plt.ylabel("Count")
    # plt.xlabel("Degree")
    # ax.set_xticks([d + 0.4 for d in deg])
    # ax.set_xticklabels(deg)
    #
    # plt.savefig('hist_' + network_name +'.png')

    # for ad_p_remove in [0.01, 0.05, 0.1, 0.2, 0.3]:
    #     for i in range(0, 10):
    #         if remove_type == 'edge':
    #             directory = './data-adv/rand_edge/' + str(ad_p_remove) + '/'
    #         elif remove_type == 'node':
    #             directory = './data-adv/rand_node/' + str(ad_p_remove) + '/'
    #         elif remove_type == 'target':
    #             directory = './data-adv/target/' + str(ad_p_remove) + '/'
    #
    #         output_fname = directory + network_name + '_' + str(i + 1)
    #         if os.path.isfile(output_fname):
    #             print(output_fname + ' exists')
    #             continue
    #
    #         G_p = genGraph(remove_type)
    #
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #
    #         nx.write_edgelist(G_p, output_fname)