from __future__ import print_function
import argparse

import networkx as nx
import numpy.linalg

import math
import numpy as np


import community
import pickle
import os
import _mylib
import log


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
		e = numpy.linalg.eigvalsh(L.A)
        e.sort()
        pickle.dump(e, open(file_path, 'wb'))

	return e



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='Edgelist file', type=str)
    parser.add_argument('-remove', help='remove node or edge', type=str, default='edge')

    args = parser.parse_args()
    fname = args.fname
    remove_type = args.remove

    fname = fname.replace('\\', '/')
    network_name = fname.split('.')[1].split('/')[-1]

    log_file = './log/eigen_' + remove_type + '.txt'

    G = _mylib.read_file(fname, isDirected=False)
    e = compute_eigenvals(G, network_name)
    max_e = e[-1]

    print(max_e)

    # for ad_p_remove in [0.01, 0.05, 0.1, 0.2, 0.3]:
    #     e_list = list()
    #     for i in range(0, 10):
    #         if remove_type == 'edge':
    #             directory = './data-adv/rand_edge/' + str(ad_p_remove) + '/'
    #         elif remove_type == 'node':
    #             directory = './data-adv/rand_node/' + str(ad_p_remove) + '/'
    #         elif remove_type == 'target':
    #             directory = './data-adv/target/' + str(ad_p_remove) + '/'
    #
    #         network_disrt = network_name + '_' + str(ad_p_remove) + '_' + str(i + 1)
    #         G_disrt_fname = directory +  network_name + '_' + str(i + 1)
    #         G_disrt = _mylib.read_file(G_disrt_fname, isDirected=False)
    #         e = compute_eigenvals(G_disrt, network_disrt + '_' + remove_type)
    #         max_e = e[-1]
    #         e_list.append(max_e)
    #         print(max_e)
    #
    #     log.save_to_file_line(log_file, [network_name, ad_p_remove, 1.*sum(e_list) / len(e_list)])

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