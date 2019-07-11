# -*- coding: utf-8 -*-

"""
Simulate the API queries
"""

import networkx as nx
import random
import _mylib
import pickle
import community
import os
import math
import random

class SingleLayer(object):
	"""
	Class to simulate API queries for undirected single layer graphs
	"""
	def __init__(self, graph, cost=1, deg_cost=0.1, nodes_limit=0, p=0.1):
		super(SingleLayer, self).__init__()
		self._graph = graph 				# The complete graph
		self._cost_neighbor = cost 			# Cost of each query (default: 1)
		self._cost_num_neighbor = deg_cost		# Cost if query degree
		self._nodes_limit = nodes_limit		# The number of neighbors that the API can returns
		self._p = p

	def neighbors(self, node, isPage=False, pageNo=0):
		"""
		Return the neighbors of a node

		Args:
			node (str) -- The node id whose neighbors are needed
		Return:
			list[str] -- List of node ids which are neighbors of node
		"""
		is_lastPage = True
		nodes = list(self._graph.neighbors(node))

		return_nodes = list()
		for n in nodes:
			r = random.uniform(0, 1)
			if r >= self._p:
				return_nodes.append(n)

		# print(' [Query] return {} / {}'.format(len(return_nodes), len(nodes)))
		# get all edges
		edges = [(node, n) for n in return_nodes]

		if len(return_nodes) == 0:
			print(' [Query] return {} / {}'.format(len(return_nodes), len(nodes)))

		return set(return_nodes), set(edges), self._cost_neighbor, is_lastPage

	def in_neighbors(self, node):
		in_nodes = list(self._graph.predecessors(node))

		edges = [(node, n) for n in in_nodes]
		return set(in_nodes), set(edges), self._cost_neighbor

	def out_neighbors(self, node):
		out_nodes =  list(self._graph.successors(node))

		edges = [(node, n) for n in out_nodes]
		return set(out_nodes), set(edges), self._cost_neighbor

	def neighbors_directed(self, node, isBoth):
		if isBoth:
			in_nodes = list(self._graph.predecessors(node))
			out_nodes = list(self._graph.successors(node))
			all_nodes = list(set(in_nodes) | set(out_nodes))
			cost = (self._cost_neighbor * 2)
		else:
			all_nodes = list(self._graph.successors(node))
			cost = self._cost_neighbor

		edges = [(node, n) for n in all_nodes]
		return set(all_nodes), set(edges), cost

	def number_of_neighbors(self, nodes):
		deg = self._graph.degree(nodes)
		cost = self._cost_num_neighbor * len(deg.keys())
		return deg, cost

	def randomNode(self):
		"""
		Return a random node from the graph

		Args:
			None
		Return:
			str -- The node id of a random node in the graph
		"""
		nodes = self._graph.nodes()

		return random.choice(list(nodes))

	def randomNodeFromLCC(self):
		"""
		Return a random node from the graph

		Args:
			None
		Return:
			str -- The node id of a random node in the graph
		"""
		g_lcc = max(nx.connected_component_subgraphs(self._graph), key=len)
		nodes = g_lcc.nodes()

		return random.choice(list(nodes))

	def randomNode4Directed(self):
		nodes = self._graph.nodes()
		sel_node = random.choice(list(nodes))

		in_deg = self._graph.in_degree(sel_node)
		out_deg = self._graph.out_degree(sel_node)

		while in_deg <= 2 or out_deg <=2:
			sel_node = random.choice(list(nodes))

			in_deg = self._graph.in_degree(sel_node)
			out_deg = self._graph.out_degree(sel_node)


		print('[Random node] {} : In {} \t Out {}'.format(sel_node, in_deg, out_deg))

		return sel_node

	def randomHighDegreeNode(self):
		degree = self._graph.degree()
		degree_sorted = _mylib.sortDictByValues(degree,reverse=True)
		size = int(.3 * len(degree))
		degree_sorted = degree_sorted[:size]
		return random.choice(degree_sorted)[0]

	def randomFarNode(self):
		degree = self._graph.degree()
		deg_one_nodes = _mylib.get_members_from_com(2,degree)
		cc = nx.clustering(self._graph)

		for n in deg_one_nodes:
			print(cc[n])

	def randomFromLargeCommunity(self, G, dataset):
		com_fname = './data/pickle/communities_{}.pickle'.format(dataset)
		if os.path.isfile(com_fname):
			partition = pickle.load(open(com_fname, 'rb'))
		else:
			partition = community.best_partition(G)
			pickle.dump(partition, open(com_fname, 'wb'))

		count_members = {}
		for p in set(partition.values()):
			members = _mylib.get_members_from_com(p, partition)
			count_members[p] = len(members)

		selected_p, i = _mylib.get_max_values_from_dict(count_members, count_members.keys())
		members = _mylib.get_members_from_com(selected_p, partition)

		degree = self._graph.degree(members)
		degree_sorted = _mylib.sortDictByValues(degree, reverse=True)
		size = int(.5 * len(degree))
		degree_sorted = degree_sorted[:size]
		return random.choice(degree_sorted)[0]




