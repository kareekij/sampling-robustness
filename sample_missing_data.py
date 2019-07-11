# -*- coding: utf-8 -*-

"""
Script for the sampling algorithm
"""

from __future__ import division, print_function
import numpy as np
import networkx as nx
import csv
import sys
import random
import operator
import time
import math
import argparse

import query
import query_incomplete
import log
import os

import community
import _mylib
import networkx_helper as nx_helper
import Queue
from scipy import stats
import pickle

#from sklearn import linear_model
from collections import defaultdict, Counter

from scipy import stats


import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt


import adversary

starting_node = -1

class UndirectedSingleLayer(object):
	"""
	Class for Expansion-Densification sammling
	"""


	def __init__(self, query, budget=100, bfs_count=10, exp_type='oracle', dataset=None, logfile=None,k=5,cost=True, log_int=10):
		super(UndirectedSingleLayer, self).__init__()
		self._budget = budget 			# Total budget for sampling
		self._bfs_count = bfs_count 	# Portion of the budget to be used for initial bfs
		self._query = query 			# Query object
		self._dataset = dataset 		# Name of the dataset used; Used for logging and caching
		self._logfile = logfile 		# Name of the file to write log to
		self._exp_type = exp_type
		self._k = k
		self._isCost = cost
		self._log_interval = log_int
		self._stage = None
		self._one_cost = 0
		self._one_gain = 0

		self._cost = 0 					# Keep track of the cost spent
		self._sample = {'edges': set(), 'nodes':\
		 {'close':set(), 'open':set()}}

		self._starting_wt = 0
		self._wt_den = 3 #0.90 # 3 				# Weight of the densification score
		self._wt_exp = 1 #(1 - self._wt_den)			# Weight of the expansion score

		self._score_den_list = [] 			# List to store densification scores; used only for logging
		self._score_exp_list = [] 			# List to store expansion scores; used only for logging
		self._new_nodes = []			# List of the new nodes observed at each densification
		self._ext_nodes = []
		self._cumulative_new_nodes = [] # Cummulated new nodes
		self._exp_cut_off = 15			# Numbor of expansion candidates
		self._den_cut_off = 100 			# Number of densificaiton candidates
		self._score_den_threshold = 0.1
		self._sample_graph = nx.Graph() # The sample graph
		self._nodes_observed_count = [] # Number of nodes observed in each iteration
		self._avg_deg = 0.
		self._med_deg = 0.
		self._cost_spent = []
		self._nodes_return = []
		self._exp_count = 1
		self._densi_count = 1
		self._track_obs_nodes = []
		self._track_cost = []
		self._track = {}
		self._track_edges = {}
		self._track_cc = {}
		self._track_new_nodes = []
		self._track_selected_node = []
		self._track_rank_selected = []
		self._track_score_den = []

		self._track_hyb_samp = []
		self._track_hyb_samp_nn = []
		self._track_hyb_samp_threshold = []

		self._track_k = []
		self._track_open= []

		self._tmp = 0
		self._avg = {'unobs':[], 'close':[], 'open':[]}

		self._percentile = 90
		self._sd = []

		self._line_1 = []
		self._line_2 = []

		self._X = []
		self._Y = []

		self._WINDOW_SIZE = 10
		self._PREV = 100
		self._SAMP = 1

		self._data_to_plot = []
		self._chose_deg = []

		self._bandit = {'arms': dict.fromkeys([1,2,3,4]),
				  'score': dict.fromkeys([1,2,3,4], float('inf')),
				  'count': dict.fromkeys([1,2,3,4], 0),
				  'created_at': dict.fromkeys([1,2,3,4], 0),
				  'rewards': defaultdict(list)}

		self._sel_obs_deg = dict()
		self._sel_act_deg = dict()
		self._deg_correlation = []
		self._p_value = []

		self._expected_avg_deg = 0
		self._wt_increment = 10

	def _expansion_random(self, candidate_list):
		degree_observed = dict(self._sample_graph.degree(candidate_list))
		degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
		top_20 = int(.2 * len(degree_observed_sorted))
		sel = random.choice(degree_observed_sorted[top_20:])
		node = sel[0]
		deg = sel[1]
		current_node = node

		return current_node

	def _query_max_unobs(self, candidate_list):

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].update(candidate_list)
		obs_nodes = set(self._sample['nodes']['open']).union(set(self._sample['nodes']['close']))

		score_close = {}
		score_open = {}
		score_unobs = {}

		for node in set(candidate_list):
			nodes, edges, c2 = self._query.neighbors(node)

			score_close[node] = 1. * len(set(self._sample['nodes']['close']).intersection(set(nodes))) / len(nodes)
			score_open[node] = 1. * len(set(self._sample['nodes']['open']).intersection(set(nodes))) / len(nodes)
			score_unobs[node] = len(set(nodes) - obs_nodes)

			print(' \t\tNode: {} \tclose: {},\topen {},\tunobs {} \t {} \t {}'.format(node, score_close[node], score_open[node], score_unobs[node], self._cost, c2))

			self._increment_cost(c2)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			if self._cost > self._budget:
				break

		self._updateSample(sub_sample)

		#avg_new_nodes = np.mean(np.array(score_unobs.values()))
		idx = np.where(np.array(score_unobs.values()) == 0)[0]

		# TODO: Just use same variable name, it is a prob of not being 0
		avg_new_nodes = 1 - (1.*len(idx) / len(score_unobs))
		avg_close = np.mean(np.array(score_close.values()))
		avg_open = np.mean(np.array(score_open.values()))

		self._avg['unobs'].append(avg_new_nodes)
		self._avg['close'].append(avg_close)
		self._avg['open'].append(avg_open)

		best_node = _mylib.sortDictByValues(score_unobs,reverse=True)[0][0]
		best_val = _mylib.sortDictByValues(score_unobs,reverse=True)[0][1]

		if best_val != 0:
			nodes, edges, c2 = self._query.neighbors(best_node)
			tmp = set(nodes) - obs_nodes
		else:
			print('		-- switch -- ')
			best_node = _mylib.sortDictByValues(score_open, reverse=True)[0][0]
			nodes, edges, c2 = self._query.neighbors(best_node)
			tmp = (set(self._sample['nodes']['open']).intersection(set(nodes)))

		print('		==> Pick node {}: {} {} <== AVG new nodes: {} close : {} open : {}'.format(best_node, best_val, len(tmp),
																							   avg_new_nodes, avg_close,
																							   avg_open))

		ret = random.choice(list(tmp))

		return ret

	def _pick_nodes(self, d, graph, partition,k=5):
		ret = []

		score = {}
		for n in d.keys():
			nbs = graph.neighbors(n)
			p_set = set()

			for nb in nbs:
				p = partition[nb]
				p_set.add(p)
			score[n] = 1.*len(p_set) / len(nbs)

		score = _mylib.sortDictByValues(score, reverse=True)

		ret = [t[0] for t in score[:k]]
		return ret

	def _expansion_oracle(self, candidate_list):
		"""
		Run the expansion step

		Args:
			candidate_list (list[str]) -- List of expansion candidates
		Return:
			str -- The id of the node to perform densification on
		"""
		sample_g = self._sample_graph
		sample_g_in = sample_g.subgraph(self._sample['nodes']['close'])



		obs_nodes = list(self._sample['nodes']['close'].union(self._sample['nodes']['open']))
		# TODO: if want to balance distance, call this
		#center_node = nx.center(sample_g_in)[0]
		#current, c = self._oracle.expansion(candidate_list, obs_nodes, center_node)
		current, c = self._oracle.expansion(candidate_list, obs_nodes, "")

		self._increment_cost(c)

		return current

	def _densification_oracle(self, candidate):
		"""
		Run the densification steps

		Args:
			candidate (str) -- The id of the node to start densification on
		Return:
			list[str] -- List of candidate nodes for expansion
		"""

		# If the candidate is not in the sample, add it
		if candidate not in self._sample_graph.nodes():
			self._sample_graph.add_node(candidate)

		# Initialize a new sub sample
		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}
		sub_sample['nodes']['open'].add(candidate)

		# Initialize densification and expansion scores
		# score_den = self._scoreDen(sub_sample)
		score_den = self._scoreDen_test(sub_sample)
		score_exp = self._scoreExp(sub_sample)

		prev_score = score_den
		score_change = 1.
		score_list = []
		RANGE = 3
		THRESHOLD = 1.
		isConverge = False

		# Perform densification until one of the conditions is met:
		# 	1. Densification score is less than the expansion score
		# 	2. Budget allocated has run out
		# 	3. There are no more open nodes
		# TODO: Densification switch criteria
		# while score_exp < score_den and self._cost < self._budget\
		#  and len(sub_sample['nodes']['open']) > 0:

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			# Get the list of nodes to perform densification on
			den_nodes = self._getDenNodes(sub_sample['nodes']['open'])
			#den_nodes = sub_sample['nodes']['open']

			# Get the node to densify on from the Oracle
			observed_nodes = list(self._sample_graph.nodes())

			current, c1 = self._oracle.densification(den_nodes, observed_nodes)
			current_node_obs_deg = self._sample_graph.degree(current)

			# Query the neighbors of current
			nodes, edges, c2 = self._query.neighbors(current)

			# Update the densification and expansion scores
			score_den = self._scoreDen_test(sub_sample, nodes, current_node_obs_deg, score_den)
			score_exp = self._scoreExp(sub_sample)
			score_list.append(score_den)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			self._increment_cost(c2)

			print(' [oracle] score_den: {} current_deg: {} '.format(score_den, current_node_obs_deg))

			if score_den < 0.1:
				break

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

		# Return list of potential expansion nodes
		return self._getExpNodes()

	def _densification(self, candidate):
		"""
		Run the densification steps

		Args:
			candidate (str) -- The id of the node to start densification on
		Return:
			list[str] -- List of candidate nodes for expansion
		"""

		# If the candidate is not in the sample, add it
		if candidate not in self._sample_graph.nodes():
			self._sample_graph.add_node(candidate)

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(candidate)
		#sub_sample['nodes']['open'].update(self._sample['nodes']['open'])
		#sub_sample['nodes']['close'].update(self._sample['nodes']['close'])
		#sub_sample['edges'].update(self._sample['edges'])


		# Initialize densification and expansion scores
		#score_den = self._scoreDen(sub_sample)
		score_den = self._scoreDen_test(sub_sample)
		score_exp = self._scoreExp(sub_sample)

		# # Initialize unobs list
		# self._den_unobs = []

		prev_score = score_den
		#score_change = 1.
		#score_list = []
		#THRESHOLD = 1.
		#isConverge = False

		# starting_w_den = self._wt_den
		# starting_w_exp = self._wt_exp

		# Perform densification until one of the conditions is met:
		# 	1. Densification score is less than the expansion score
		# 	2. Budget allocated has run out
		# 	3. There are no more open nodes
		# TODO: Densification switch criteria
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			# TODO: Nodes for densify should be filter out ?

			den_nodes = self._getDenNodes(list(sub_sample['nodes']['open'])) #sub_sample['nodes']['open']
			degree_observed = dict(self._sample_graph.degree(den_nodes))

			c_nodes = set(self._sample['nodes']['close']).union(set(sub_sample['nodes']['close']))
			o_nodes = set(self._sample['nodes']['open']).union(set(sub_sample['nodes']['open']))
			selected_arm = -1

			if len(den_nodes) > 10:
				#current, current_node_obs_deg = self._node_selection(den_nodes)
				current, current_node_obs_deg  = self._node_selection(den_nodes)
			else:
				current = list(den_nodes)[0]
				current_node_obs_deg = degree_observed[current]

			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			self._count_new_nodes(nodes, current, deg_obs=current_node_obs_deg, score=score_den)

			# All neighbors that are not closed
			not_close_nbs = set(nodes) - c_nodes

			self._sel_obs_deg[current] = current_node_obs_deg
			self._sel_act_deg[current] = len(nodes)

			# TODO: Densification score, to be changed.
			# Update the densification and expansion scores
			deg_close = dict(self._sample_graph.degree(list(c_nodes))).values()
			avg_deg = np.median(np.array(deg_close))
			sd_deg = np.std(np.array(deg_close))


			#k = len(self._sel_act_deg)
			#tau, p_value = _mylib.get_rank_correlation(self._sel_act_deg, self._sel_obs_deg, k=k)
			#self._deg_correlation.append(tau)
			#self._p_value.append(p_value)

			score_den, deg_new = self._scoreDen_test(sub_sample, avg_deg, nodes,current_node_obs_deg,score_den)
			score_exp = self._scoreExp(sub_sample, avg_deg, nodes, current_node_obs_deg, score_exp)

			#score_exp = self._scoreExp(sub_sample, score_exp)

			# Store the densification and expansion scores
			self._score_den_list.append(score_den)
			self._score_exp_list.append(score_exp)

			self._densi_count += 1

			# ### For checking
			# deg_ori = self._query._graph.degree(den_nodes)
			# deg_ori_sort = _mylib.sortDictByValues(deg_ori, reverse=True)
			# best = deg_ori_sort[0][1]
			# best_nodes = deg_ori_sort[0][0]
			# best_obs = self._sample_graph.degree(best_nodes)
			# ### For checking


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			# TODO: just a dummy statement for MOD method
			#T = self._score_den_threshold * math.exp(-(0.05 * self._exp_count))
			#T = 0.
			# c_nodes = set(sub_sample['nodes']['close'])
			# deg = self._sample_graph.degree(list(c_nodes))
			# mean_deg = np.average(np.array(deg.values()))
			# sd_deg = np.std(np.array(deg.values()))
			# T = mean_deg + sd_deg

			#nn_mean = np.mean(np.array(self._new_nodes))

			c_nodes.add(current)
			deg_c = dict(self._sample_graph.degree(list(c_nodes))).values()
			avg_deg_c = np.average(np.array(deg_c))

			print('{}:{} weight {} \t score_den: {}	\t score_exp: {} \t cost: {} \t exp.deg:{} \t cur.deg:{} \t obs:{} act:{} new: {} | {}/{}'.format(self._densi_count, self._exp_count, self._wt_den, round(score_den,2),
																																  round(score_exp,2), self._cost, round(self._expected_avg_deg, 0), round(avg_deg_c,0),current_node_obs_deg, len(nodes), deg_new,
																															  len(den_nodes), len(list(sub_sample['nodes']['open'])) ))
			#score_exp = 0
			#if score_den <= T:
			if score_den < score_exp:
				print(' \t \t Switch!')
				break

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

		# Reset weights
		# starting_w_den = self._wt_den
		# starting_w_exp = self._wt_exp

		# Return list of potential expansion nodes
		return self._getExpNodes()

	def random_sampling(self):

		current = starting_node

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		# TODO: Densification switch criteria
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			den_nodes = sub_sample['nodes']['open']

			# Randomly pick node
			current = random.choice(list(den_nodes))

			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			self._count_new_nodes(nodes, current)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			if debug:
				print('			Number of nodes \t Close: {} \t Open: {}'.format( \
					len(sub_sample['nodes']['close']), \
					len(sub_sample['nodes']['open'])))

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _score_R(self, sub_sample):
		g = self._sample_graph
		closed_nodes = self._sample['nodes']['close'].union(sub_sample['nodes']['close'])
		o = self._sample['nodes']['open'].union(sub_sample['nodes']['open'])
		open_nodes = set(o).difference(closed_nodes)

		b_nodes = self._get_boundary_nodes(closed_nodes, open_nodes)

		t_nodes = set()
		b_edges = list(g.edges_iter(list(b_nodes)))
		for e in b_edges:
			t_nodes.add(e[1])

		try:
			r_score = len(set(t_nodes).intersection(set(closed_nodes))) / len(b_edges)
		except ZeroDivisionError:
			r_score = 1.

		#print(' Border: {} {} Closed: {} Open: {} T: {} score: {}'.format(len(b_nodes), len(b_edges), len(closed_nodes), len(open_nodes), len(t_nodes), r_score))

		return r_score

	def _get_boundary_nodes(self,closed_nodes, open_nodes):
		g = self._sample_graph
		b_nodes = set()
		for c_n in closed_nodes:
			nbs = g.neighbors(c_n)
			if len(set(nbs).intersection(set(open_nodes))) != 0:
				b_nodes.add(c_n)
		return b_nodes

	def _score(self, sub_sample, nodes=None, prev=0):
		# This should be for only the start of the densification
		if nodes is None:
			return np.inf

		# The new nodes; Nodes neithen in sample and sub sample
		new_nodes = nodes.difference(sub_sample['nodes']['close']) \
			.difference(sub_sample['nodes']['open']) \
			.difference(self._sample['nodes']['open']).difference(self._sample['nodes']['close'])

		#print('	Unobs {} / {} '.format(len(new_nodes), len(nodes)))

		med = np.median(np.array(self._new_nodes))

		# Calculate the densification score
		#score = 1.*len(new_nodes) / (len(sub_sample['nodes']['close']) + 1)
		score = 1.*len(new_nodes) / med

		print('score {} \t unobs {} \t med {} \t '.format(score,len(new_nodes),med))

		# Store number of new nodes for logging later
		self._new_nodes.append(len(new_nodes))

		if np.isfinite(prev):
			return (0.5 * prev) + score
		else:
			return score

	def _scoreDen_test(self, sub_sample, avg_deg=0, nodes=None, deg_obs=0, prev=0):
		"""
		Calculate the densification score

		Args:
			sub_sample(dict) -- Dict of the subsample
			nodes (list[str]) -- Unobserved nodes from last densification step (default: None)
			prev (float) -- The previous score (default: 0)
		Return:
			float -- The densification score
		"""

		# This should be for only the start of the densification
		if nodes is None:
			#return 1.
			return np.inf

		# The new nodes; Nodes neithen in sample and sub sample
		# new_nodes = nodes.difference(sub_sample['nodes']['close']) \
		# 	.difference(sub_sample['nodes']['open']) \
		# 	.difference(self._sample['nodes']['open']).difference(self._sample['nodes']['close'])
		observed_nodes = self._sample_graph.nodes()
		new_nodes = set(nodes) - set(observed_nodes)
		deg_true = len(nodes)
		deg_new_nodes = len(new_nodes)
		deg_existing_open = len(nodes) - deg_new_nodes - deg_obs
		deg_in = deg_existing_open + deg_new_nodes
		try:
			ratio = (deg_new_nodes / deg_in) * (deg_true / deg_obs)
		except ZeroDivisionError:
			ratio = 0.

		# TODO: score function
		# Calculate the densification score
		try:
			score = deg_new_nodes / deg_in
		except ZeroDivisionError:
			score = 0.

		# Store number of new nodes for logging later
		self._new_nodes.append(deg_new_nodes)
		self._ext_nodes.append(deg_existing_open)


		#self._wt_den = self._exp_count + self._cost
		#t = (2*math.log10(self._cost+1) / self._densi_count)
		#t = ((self._cost + 1) / self._densi_count) * (1/avg_deg)


		# if len(self._deg_correlation) > 1:
		# 	tau_cur = round(self._deg_correlation[-1],4)
		# 	tau_pre = round(self._deg_correlation[-2],4)
		#
		# 	if tau_cur - tau_pre > 0.01:
		# 		self._wt_den = self._wt_den - 0.2
		# 	else:
		# 		self._wt_den = self._wt_den + 0.2


		#if self._cost % 5 == 0:
		#if (deg_obs - self._expected_avg_deg) > 10 or (deg_obs - self._expected_avg_deg) < -10:

		bound = int(.20*self._expected_avg_deg)

		if deg_obs >= (self._expected_avg_deg):

			self._wt_den = self._wt_den + self._wt_increment

		# if lower_b < deg_obs and deg_obs < upper_b:
		# 		self._wt_den = self._wt_den + 10
		# else:
		# 	self._wt_den = self._wt_den - 10
		#
		# 	if self._wt_den < self._starting_wt:
		# 		self._wt_den = self._starting_wt



		if np.isfinite(prev):
			s = (0.5 * prev) + (self._wt_den * score)
		else:
			s = self._wt_den * score

		return s, deg_new_nodes

	def _scoreDen(self, sub_sample, nodes=None, prev=0):
		"""
		Calculate the densification score

		Args:
			sub_sample(dict) -- Dict of the subsample
			nodes (list[str]) -- Unobserved nodes from last densification step (default: None)
			prev (float) -- The previous score (default: 0)
		Return:
			float -- The densification score
		"""

		# This should be for only the start of the densification
		if nodes is None:
			print(nodes)
			return 1#self._sample_graph.number_of_nodes()
			#return np.inf

		# The new nodes; Nodes neithen in sample and sub sample
		new_nodes = nodes.difference(sub_sample['nodes']['close'])\
		 .difference(sub_sample['nodes']['open'])\
		 .difference(self._sample['nodes']['open']).difference(self._sample['nodes']['close'])

		#print('	Unobs {} / {} '.format(len(new_nodes), len(nodes)))

		# TODO: score function
		# Calculate the densification score
		score = len(new_nodes)/(len(sub_sample['nodes']['close']) + 1)
		#new_node_norm = len(new_nodes) / (len(nodes) - self._node_picked_obs_deg[-1:][0])

		#score = new_node_norm


		# Store number of new nodes for logging later
		self._new_nodes.append(len(new_nodes))

		if len(self._cumulative_new_nodes) == 0:
			self._cumulative_new_nodes.append(len(new_nodes))
		else:
			c_new_nodes = self._cumulative_new_nodes[-1]
			self._cumulative_new_nodes.append(c_new_nodes + len(new_nodes))

		if np.isfinite(prev):
			return 0.5 * prev + self._wt_den * score
		else:
			return self._wt_den * score

	def _scoreExp(self, sub_sample, avg_deg=0, nodes=None, deg_obs=0, prev=0):
		"""
		Calculate the expansion score

		Args:
			sub_sample(dict) -- The sub sample from current densification
			prev (float) -- The previous expansion score
		"""

		# Get the edges between open and close nodes in the current sub sample
		# edges = set()
		# for e in sub_sample['edges']:
		# 	if (e[0] in sub_sample['nodes']['close']\
		# 	 and e[1] in sub_sample['nodes']['open'])\
		# 	 or (e[1] in sub_sample['nodes']['close']\
		# 	 and e[0] in sub_sample['nodes']['open']):
		# 		edges.add(e)
		#
		# # Calculate the expansion score
		# score = len(edges)/(len(sub_sample['nodes']['open']) + 1)
		if nodes is None:
			return 0.

		observed_nodes = self._sample_graph.nodes()
		new_nodes = set(nodes) - set(observed_nodes)
		deg_true = len(nodes)
		deg_new_nodes = len(new_nodes)
		deg_existing_open = len(nodes) - deg_new_nodes - deg_obs
		deg_in = deg_existing_open + deg_new_nodes

		#self._wt_exp = (1. - self._wt_den)
		# t = self._wt_exp / self._exp_count
		t = self._wt_exp

		# TODO: score function
		# Calculate the densification score
		try:
			score = deg_existing_open / deg_in
		except ZeroDivisionError:
			score = 0.

		#t = (2 * math.log10(self._cost+1) / self._exp_count)
		#t = (self._cost+1) / self._exp_count

		return 0.5 * prev + (t * score)

	def _updateSample(self, sub_sample):
		"""
		Update the sample with the sub sample

		Args:
			sub_sample (dict) -- The sub sample dict
		"""
		self._sample['edges'].update(sub_sample['edges'])
		self._sample['nodes']['close'].update(sub_sample['nodes']['close'])
		self._sample['nodes']['open'] = self._sample['nodes']['open'].difference(\
		 sub_sample['nodes']['close'])
		self._sample['nodes']['open'].update(sub_sample['nodes']['open'])


		nodes_count = self._sample_graph.number_of_nodes()
		# TODO: Calculate mean and median of close nodes degree, MIGHT CHANGE !
		#degree = self._sample_graph.degree().values()

		# self._avg_deg, self._med_deg = nx_helper.get_graph_average_deg(self._sample_graph)


		# degree = self._sample_graph.degree(self._sample['nodes']['close']).values()
		# self._avg_deg = np.mean(np.array(degree))
		# self._med_deg = np.median(np.array(degree))
		# print( " Degree avg: {} , med: {}".format(self._avg_deg, self._med_deg))

	def _updateSubSample(self, sub_sample, nodes, edges, candidate):
		"""
		Update the sub sample with new nodes aned edges

		Args:
			sub_sample (dict) -- The sub sample to update
			nodes (list[str]) -- The open nodes
			edges (list[(str,str)]) -- The new edges
			candidate (str) -- The new open node
		Return:
			dict -- The updated sub sample
		"""
		try:
			sub_sample['edges'].update(edges)
			sub_sample['nodes']['close'].add(candidate)
			sub_sample['nodes']['open'].remove(candidate)
			sub_sample['nodes']['open'].update(\
				nodes.difference(sub_sample['nodes']['close'])\
				.difference(self._sample['nodes']['close']))
		except KeyError as e:
			print('		Error update Subsample:', e, candidate)
			nodes = self._query._graph.nodes()
			print(candidate in set(nodes))


		return sub_sample

	def _bfs_init(self):

		"""
		Collect the initial nodes through bfs

		Args:
			None
		Return:
			None
		"""

		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}

		current = starting_node

		sub_sample['nodes']['open'].add(current)
		queue = [current]

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._bfs_count and len(queue) > 0:
			# Select the first node from queue
			current = queue[0]

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c, _ = self._query.neighbors(current)
			self._count_new_nodes(nodes, current)

			self._increment_cost(c)

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Remove the current node from queue
			queue.remove(current)

			# Update queue
			nodes = nodes.difference(sub_sample['nodes']['close'])
			nodes = nodes.difference(sub_sample['nodes']['open'])
			queue += list(nodes)
			queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

	def _smooth_init(self):
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		current_node = starting_node

		sub_sample['nodes']['open'].add(current_node)

		sample_nodes = set()
		rank = 0
		deg_observed = 0

		while self._cost < self._bfs_count:
			closed_nodes = sub_sample['nodes']['close']
			nodes, edges, c, _ = self._query.neighbors(current_node)

			if current_node not in closed_nodes:
				# For tracking
				self._count_new_nodes(nodes, current_node, rank, deg_observed)

				# Add edges to sub_graph
				for e in edges:
					self._sample_graph.add_edge(e[0], e[1])

				# Update the cost
				self._increment_cost(c)
				#
				# Update the sub sample
				sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			next_node = random.choice(list(nodes))
			next_node_deg = self._sample_graph.degree(next_node)

			a = (next_node_deg / len(nodes))
			prob = min(1, a)
			pp = random.uniform(0, 1)

			if pp < prob:
				current_node = next_node
				deg_observed = self._sample_graph.degree(current_node)

			if self._cost > 0:
				sample_nodes.add(current_node)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

		degree_sampled = dict(self._sample_graph).degree(sample_nodes)
		print('Sampled nodes', len(sample_nodes))

		a = 0.
		b = 0.
		c = 0
		for node, deg in degree_sampled.iteritems():
			a += deg / (deg + c)
			b += 1 / (deg + c)
		self._expected_avg_deg = a/b
		print(' Expected Avg. Deg', self._expected_avg_deg )

	def _after_init(self):
		true_g = self._query._graph
		act_deg = dict(true_g.degree())
		act_min_deg = min(act_deg.values())
		act_max_deg = max(act_deg.values())
		average_deg = np.average(np.array(act_deg.values()))

		current_g = self._sample_graph
		close_nodes = self._sample['nodes']['close']
		open_nodes = self._sample['nodes']['open']

		deg = dict(current_g.degree(list(close_nodes)))
		min_deg = min(deg.values())
		max_deg = max(deg.values())
		avg = np.average(np.array(deg.values()))
		med = np.median(np.array(deg.values()))
		#self._expected_avg_deg = med



		deg_open = dict(current_g.degree(open_nodes))
		s_deg = _mylib.sortDictByValues(deg_open, reverse=True)
		max_deg_open = s_deg[0][1]

		self._wt_den = max_deg / self._expected_avg_deg
		self._starting_wt = self._wt_den

		#print('	Weight set to {} \t Max deg open {}'.format(self._wt_den, max_deg_open))
		print(p_bfs_budget, average_deg, self._expected_avg_deg, avg)
		line = [p_bfs_budget, self._bfs_count, average_deg, med, avg]

	def _bfs(self):

		"""
		Collect the initial nodes through bfs

		Args:
			None
		Return:
			None
		"""

		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}

		current = starting_node

		sub_sample['nodes']['open'].add(current)
		queue = [current]

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._budget and len(queue) > 0:
			# Select the first node from queue
			current = queue[0]

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c, _ = self._query.neighbors(current)
			self._count_new_nodes(nodes, current)

			self._increment_cost(c)

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Remove the current node from queue
			queue.remove(current)

			# Update queue
			nodes = nodes.difference(sub_sample['nodes']['close'])
			nodes = nodes.difference(sub_sample['nodes']['open'])
			queue += list(nodes)
			queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			if debug:
				print('			Number of nodes \t Close: {} \t Open: {}'.format( \
					len(sub_sample['nodes']['close']), \
					len(sub_sample['nodes']['open'])))


		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

	def _snowball_sampling(self):

		if starting_node == -1:
			print('Please check the starting node')
			return

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}

		print('start at ', starting_node)
		current = starting_node

		sub_sample['nodes']['open'].add(current)
		queue = [current]

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._budget and len(queue) > 0:
			# Select the first node from queue
			current = queue[0]

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c, _ = self._query.neighbors(current)
			self._increment_cost(c)

			self._count_new_nodes(nodes, current)

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Remove the current node from queue
			queue.remove(current)

			# Update queue
			nodes = nodes.difference(sub_sample['nodes']['close'])
			nodes = nodes.difference(sub_sample['nodes']['open'])

			node_to_add = random.sample(nodes, int(.5*len(nodes)))
			queue += list(node_to_add)
			queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

	def _dfs(self):

		"""
		Collect the initial nodes through bfs

		Args:
			None
		Return:
			None
		"""

		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}

		current = starting_node

		sub_sample['nodes']['open'].add(current)
		queue = [current]

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._budget and len(queue) > 0:
			# Select the first node from queue
			current = queue.pop()

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c, _ = self._query.neighbors(current)
			self._count_new_nodes(nodes, current)

			self._increment_cost(c)

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])


			# Update queue
			nodes = nodes.difference(sub_sample['nodes']['close'])
			nodes = nodes.difference(sub_sample['nodes']['open'])
			queue += list(nodes)
			queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)


	def _random_walk(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)

		rank=0
		deg_observed=0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current_node)
			print(' return node {}'.format(len(nodes)))
			if current_node in sub_sample['nodes']['open']:
				# For tracking
				self._count_new_nodes(nodes, current_node, rank,deg_observed)

				# Update the sub sample
				sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

				# Add edges to sub_graph
				for e in edges:
					self._sample_graph.add_edge(e[0], e[1])
				# Update the cost
				self._increment_cost(c)
			else:
				print(' ** Existing nbs \t {}'.format(current_node))


			print('Sample size: {} \t open: {} close: {}'.format(self._sample_graph.number_of_nodes(),
																 len(sub_sample['nodes']['open']),
																 len(sub_sample['nodes']['close'])))

			# print(self._sample_graph.nodes())
			next_candidate_nodes = list(self._sample_graph.neighbors(current_node))
			current_node = random.choice(next_candidate_nodes)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _count_new_nodes(self, nodes, current, rank=0, deg_obs=0, score=0):
		current_nodes = self._sample_graph.nodes()
		new_nodes = set(nodes).difference(current_nodes)

		deg_true = len(nodes)
		deg_new_nodes = len(new_nodes)
		deg_existing_open = len(nodes) - deg_new_nodes - deg_obs
		deg_in = deg_existing_open + deg_new_nodes
		try:
			ratio = (deg_new_nodes / deg_in) * (deg_true / deg_obs)
		except ZeroDivisionError:
			ratio = 0.


		self._track_selected_node.append(current)
		self._track_rank_selected.append(deg_obs)
		self._track_new_nodes.append(score)
		self._track_score_den.append(score)

	def _bandit(self):
		# Initialize graph with Random Walk
		close_n, cash_count = self._init_bandit_g_rw()

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}

		# Start Bandit algorithm
		arms, node2arm = self._get_arms(close_n)

		bandit = {'arms': arms,
				  'score': dict.fromkeys(arms.keys(), float('inf')),
				  'count': dict.fromkeys(arms.keys(), 0),
				  'created_at': dict.fromkeys(arms.keys(), 0),
				  'rewards': defaultdict(list),
				  'node2arm': node2arm}

		# Initialize score for each arm
		for k,v in bandit['arms'].iteritems():
			count = len(k.split('.')) - 1
			bandit['score'][k] = count
			#bandit['score'][k] = (1 / count) * len(v)


		# Pick fist arm
		max_score = max(bandit['score'].values())
		candidate_arms = _mylib.get_keys_by_value(bandit['score'], max_score)
		current_arm = random.choice(candidate_arms)
		members = bandit['arms'][current_arm]
		current_node = random.choice(members)

		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])
		sub_sample['nodes']['close'].update(close_n)
		iter_count = 1

		#cash_count = {}


		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query on the selected node
			nodes, edges, c, _ = self._query.neighbors(current_node)
			new_nodes = set(nodes) - set(self._sample_graph.nodes())
			closed_nodes = self._sample['nodes']['close'] | sub_sample['nodes']['close']

			# Update bandit
			bandit = self._update_arms(bandit, current_node, current_arm, nodes, closed_nodes, iter_count)

			for node in nodes:
				cash_count[node] = cash_count.get(node, 1 ) + (cash_count.get(current_node,1) / len(nodes))


			# For tracking
			self._count_new_nodes(nodes, current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			bandit = self._update_score(bandit, iter_count)
			max_score = max(bandit['score'].values())

			candidate_arms = _mylib.get_keys_by_value(bandit['score'], max_score)
			current_arm = random.choice(candidate_arms)
			members = bandit['arms'][current_arm]
			#current_node = self._pick_next_node(members)#random.choice(members)
			current_node = self._pick_next_node_from_cash(members, cash_count)

			iter_count += 1

		self._updateSample(sub_sample)

	def _pick_next_node(self, candidates):
		return random.choice(candidates)

	def _pick_next_node_from_cash(self, candidates, cash_count):
		cash_count_sorted = _mylib.sortDictByValues(cash_count, reverse=True)

		for index, ccc in enumerate(cash_count_sorted):
			#print(cash_count_sorted[index][0])
			if cash_count_sorted[index][0] in candidates:
				c = cash_count_sorted[index][0]
				return c

	def _update_score(self, bandit, iter_count, CONSTANT_P = 150):


		next_iter = iter_count + 1
		CONSTANT_P = CONSTANT_P * math.exp(-(0.05 * next_iter))
		#print(CONSTANT_P)

		for arm_id, rewards in bandit['rewards'].iteritems():
			avg_rewards = np.average(np.array(rewards))
			#avg_rewards = np.average(np.array(rewards),weights=range(1, len(rewards) + 1))


			created_at = bandit['created_at'][arm_id]
			arm_picked_count = bandit['count'][arm_id] + 0.1

			score = avg_rewards + \
					(CONSTANT_P * math.sqrt( (2 * math.log(next_iter - created_at)) / arm_picked_count ))

			#print(score, arm_picked_count, avg_rewards, rewards)
			bandit['score'][arm_id] = score

		return bandit

	def _update_arms(self, bandit, current_node, current_arm, nbs, closed_nodes, iter_count):
		#print('Update arms', iter_count)
		C = 5
		open_nodes = set(self._sample_graph.nodes()) - closed_nodes

		# all nodes in current arm
		members_in_arms = bandit['arms'][current_arm]
		# nbs that are in the current arm
		nbs_cur_arm = set(nbs).intersection(set(members_in_arms))
		# nbs that are newly added nodes
		new_nodes = set(nbs) - set(self._sample_graph.nodes())
		# nbs that are not in the current arm
		nbs_out_arm = (set(nbs) - nbs_cur_arm) & open_nodes

		# nbs that are existing inside and outside current arm
		nbs_in_out_arm = nbs_cur_arm & nbs_out_arm

		#print('Nbs: {}, Current arm: {}/{}, Outside: {}, Newly added: {}'.format(len(nbs), len(nbs_cur_arm), len(members_in_arms), len(nbs_out_arm), len(new_nodes)))

		# Update info of the current arm
		# Remove current node from current arm
		bandit['arms'][current_arm].remove(current_node)
		bandit['node2arm'].pop(current_node, -1)
		bandit['rewards'][current_arm].append(len(new_nodes))
		bandit['count'][current_arm] += 1

		avg_cur_arm_reward = np.mean(np.array(bandit['rewards'][current_arm]))
		P = (len(new_nodes)+1) / len(nbs)
		reward_4_new_arm = (P * avg_cur_arm_reward)

		#print ('	Avg. Reward: {}, Score: {}, R4new: {}, P: {}'.format(avg_cur_arm_reward, bandit['score'][current_arm], reward_4_new_arm, P))

		# For the case that current arm has one node and it is selected.
		if len(bandit['arms'][current_arm]) == 0:
			bandit['arms'].pop(current_arm)
			bandit['score'].pop(current_arm)
			bandit['count'].pop(current_arm)
			bandit['created_at'].pop(current_arm)
			bandit['rewards'].pop(current_arm)

		# # CASE 1: New arm becuase current node has link with nodes in current arm
		# for index, node in enumerate(nbs_cur_arm):
		# 	# New arm id
		# 	new_arm = current_arm + str(current_node) + '.'
		# 	# Update arm id
		# 	bandit['node2arm'][node] = new_arm
		# 	# Add node to new arm
		# 	bandit['arms'][new_arm].append(node)
		# 	#bandit['score'][new_arm] = bandit['score'].get(new_arm, float('inf'))
		# 	bandit['count'][new_arm] = 0
		# 	bandit['created_at'][new_arm] = iter_count
		# 	if index == 0:
		# 		count = len(new_arm.split('.'))
		# 		reward = reward_4_new_arm + ((1 - P) * (1/(count-1) ))
		# 		bandit['rewards'].setdefault(new_arm, []).append(reward)
		# 		bandit['score'][new_arm] = bandit['score'].get(new_arm, reward)
		#
		# 		#avg_cur_arm_reward = np.mean(np.array(bandit['rewards'][new_arm]))
		# 		#bandit['score'][current_arm] = avg_cur_arm_reward
		#
		# 	# Remove node from current arm
		# 	bandit['arms'][current_arm].remove(node)
		#
		# 	# If current Arm is disappear
		# 	if len(bandit['arms'][current_arm]) == 0:
		# 		bandit['arms'].pop(current_arm)
		# 		bandit['score'].pop(current_arm)
		# 		bandit['count'].pop(current_arm)
		# 		bandit['created_at'].pop(current_arm)
		# 		bandit['rewards'].pop(current_arm)

		# CASE 2: New arm becuase of newly added nodes
		for index, node in enumerate(new_nodes):
			# New arm id
			new_arm = str(current_node) + '.'
			# Update arm id
			bandit['node2arm'][node] = new_arm
			# Update arm info
			bandit['arms'][new_arm].append(node)

			if index == len(new_nodes) - 1:
				close_count = len(new_arm.split('.')) - 1
				reward = reward_4_new_arm + ((1 - P) * ((close_count)))
				#reward = reward_4_new_arm + ((1 - P) * (len(new_nodes) / (close_count)))

				bandit['rewards'].setdefault(new_arm, []).append(reward)
				bandit['created_at'][new_arm] = bandit['created_at'].get(new_arm, iter_count)
				bandit['count'][new_arm] = bandit['count'].get(new_arm, 0)


		# CASE 3: New arm becuase current node connects with nodes outside.
		for index, node in enumerate(nbs_in_out_arm):
			# Current arm of particular node
			node_arm = bandit['node2arm'][node]
			# New arm id
			new_arm = node_arm + str(current_node) + '.'
			# Update arm id
			bandit['node2arm'][node] = new_arm
			# Add node to new arm
			bandit['arms'][new_arm].append(node)

			bandit['count'][new_arm] = bandit['count'].get(new_arm, 0)
			bandit['created_at'][new_arm] = bandit['created_at'].get(new_arm, iter_count)

			#bandit['rewards'].setdefault(new_arm, []).append(reward)

			close_count = len(new_arm.split('.')) - 1
			node_in_arms = len(bandit['arms'][new_arm])
			reward = reward_4_new_arm + ((1 - P) * (close_count))
			#reward = reward_4_new_arm + ((1 - P) * (node_in_arms/close_count) )
			bandit['rewards'][new_arm] = [reward]

			# Remove node from current arm
			bandit['arms'][node_arm].remove(node)

			# If current Arm is disappear
			if len(bandit['arms'][node_arm]) == 0:
				bandit['arms'].pop(node_arm)
				bandit['score'].pop(node_arm, -1)
				bandit['count'].pop(node_arm)
				bandit['created_at'].pop(node_arm)
				bandit['rewards'].pop(node_arm, -1)

		return bandit

	def _get_arms(self, close_n):
		sample_g = self._sample_graph
		open_nodes = set(sample_g.nodes()) - close_n

		arms = defaultdict(list)
		node2arm = dict()

		for o_n in open_nodes:
			nbs = sample_g.neighbors(o_n)
			key = ''
			for x in nbs:
				key += x +'.'

			arms[key].append(o_n)
			node2arm[o_n] = key

		print(len(node2arm), len(open_nodes))
		return arms, node2arm

	def _init_bandit_g_rw(self, START_BUDGET = 10):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)

		cash_count = {}

		while self._cost < START_BUDGET and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current_node)
			# For tracking
			self._count_new_nodes(nodes, current_node)

			for node in nodes:
				cash_count[node] = cash_count.get(node, 1 ) + (cash_count.get(current_node,1) / len(nodes))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost
			self._increment_cost(c)

			# Candidate nodes are the (open) neighbors of current node
			candidates = list(
				set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))

			while len(candidates) == 0:
				current_node = random.choice(list(nodes))
				# Query the neighbors of current
				nodes, edges, c, _ = self._query.neighbors(current_node)
				# Candidate nodes are the (open) neighbors of current node
				candidates = list(
					set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
				# print(' Walking.. {} neighbors'.format(len(nodes)))

			current_node = random.choice(candidates)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)
		close_n = set(sub_sample['nodes']['close']) | set(self._sample['nodes']['close'])
		return close_n, cash_count

	def _cash_spread(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		cash_count = {}

		C = dict()
		H = dict()
		G = 0
		P = dict()

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node)

			cost_of_current = C.get(current_node, 1.)
			H[current_node] = H.get(current_node, 0.) + cost_of_current

			G += cost_of_current
			for nb in nodes:
				C[nb] = C.get(nb, 1.) + (cost_of_current / len(nodes))
				P[nb] = (H.get(nb, 0.) + C.get(nb, 1.)) / (G + 1)

			P[current_node] = 1. * (H.get(current_node, 0) + C.get(current_node, 1)) / (G + 1)
			C[current_node] = 0.

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			closed_node = sub_sample['nodes']['close']

			# P_open = _mylib.remove_entries_from_dict(closed_node, P)
			# max_p = max(P_open.values())
			#
			# nodes_with_max_p = [k for k, v in P_open.iteritems() if v == max_p]
			# print('node with max p ', len(nodes_with_max_p), len(P), len(P_open))
			# current_node = random.choice(nodes_with_max_p)
			P_open = _mylib.remove_entries_from_dict(closed_node, C)
			max_p = max(P_open.values())

			nodes_with_max_p = [k for k, v in P_open.iteritems() if v == max_p]
			print('node with max p ', len(nodes_with_max_p), len(P), len(P_open))
			current_node = random.choice(nodes_with_max_p)

			if debug:
				print('			Number of nodes \t Close: {} \t Open: {}'.format( \
					len(sub_sample['nodes']['close']), \
					len(sub_sample['nodes']['open'])))

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_obs_deg(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		rank = 0
		deg_obs = 0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node, rank, deg_obs)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)


			candidates = sub_sample['nodes']['open']

			if len(candidates) != 0:
				degree_observed = dict(self._sample_graph.degree(candidates))
				max_observed_degree = max(degree_observed.values())
				nodes_with_max_deg = [k for k, v in degree_observed.iteritems() if v == max_observed_degree]

				current_node = random.choice(nodes_with_max_deg)
				deg_obs = degree_observed[current_node]

			if debug:
				print('			Number of nodes \t Close: {} \t Open: {}'.format( \
					len(sub_sample['nodes']['close']), \
					len(sub_sample['nodes']['open'])))

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _node_selection(self, candidates):
		graph = self._sample_graph

		obs_deg = dict(graph.degree(candidates))

		min_val = min(obs_deg.values())
		max_val = max(obs_deg.values())
		#avg_val = np.average(np.array(obs_deg.values()))
		score_d = dict()

		top_k_nodes = []
		p_members = dict()
		for k, v in obs_deg.iteritems():
			cc = nx.clustering(graph, k)
			try:
				score_d[k] = ((v - min_val) / (max_val - min_val)) * (1 - cc)
			except ZeroDivisionError:
				score_d[k] = v

			top_k_nodes.append(self._query._graph.degree(k))

		sorted_list = _mylib.sortDictByValues(score_d, reverse=True)
		selected_node = sorted_list[0][0]
		deg_node = obs_deg[selected_node]
		sel_deg = self._query._graph.degree(selected_node)

		return selected_node, deg_node

	def _node_selection_bandit(self, candidates, c_nodes, o_nodes):

		graph = self._sample_graph
		obs_deg = dict(graph.degree(candidates))

		min_val = min(obs_deg.values())
		max_val = max(obs_deg.values())
		score_d = dict()

		top_k_nodes = []
		p_members = dict()
		#current_p = community.best_partition(graph)
		for k,v in obs_deg.iteritems():
			cc = nx.clustering(graph, k)
			#global_score = self._nbs_score(k, c_nodes, o_nodes, current_p)
			try:
				score_d[k] = (((v - min_val) / (max_val - min_val)) * (1 - cc))
			except ZeroDivisionError:
				score_d[k] = (v) * (1 - cc)

		score_values = score_d.values()
		size = len(score_values) / 4



		cut_points = []
		for n in [1,2,3]:
			c = size * n
			cut_points.append(c)

		# Assign nodes to arm
		arms = dict()

		sorted_score = _mylib.sortDictByValues(score_d, reverse=True)

		for v, t in enumerate(sorted_score):
			k = t[0]
			if v < cut_points[0]:
				arms[1] = arms.get(1, list()) + [k]
			elif cut_points[0] <= v < cut_points[1]:
				arms[2] = arms.get(2, list()) + [k]
			elif cut_points[1] <= v < cut_points[2]:
				arms[3] = arms.get(3, list()) + [k]
			elif cut_points[2] <= v:
				arms[4] = arms.get(4, list()) + [k]


		# Select arm
		bandit = self._bandit




		#print('score', bandit['score'])
		if min(bandit['count'].values()) == 0:
			for k,v in bandit['count'].iteritems():
				if v == 0:
					selected_arm = k
					break
			selected_node = random.choice(arms[selected_arm])
		else:
			arm_score = bandit['score']

			sorted_arm_score = _mylib.sortDictByValues(arm_score,reverse=True)
			selected_arm = int(sorted_arm_score[0][0])
			#print(arms)
			selected_node = random.choice(arms[selected_arm])






		# shuffle_list = score_d.keys()
		# random.shuffle(shuffle_list)
		#
		# TIPPING_POINT = int(0.37 * len(shuffle_list))
		#
		# print("		Tipping Pt", TIPPING_POINT)
		#
		# best_so_far = 0
		# selected_node = shuffle_list[-1]
		# for i, node in enumerate(shuffle_list):
		# 	current = score_d[node]
		# 	if i <= TIPPING_POINT:
		# 		if current > best_so_far:
		# 			best_so_far = current
		# 		elif current == best_so_far:
		# 			rand = random.uniform(0,1)
		# 			if rand <= 0.5:
		# 				best_so_far = current
		# 	elif i > TIPPING_POINT:
		# 		if current > best_so_far:
		# 			best_so_far = current
		# 			selected_node = node
		# 			break


		deg_node = obs_deg[selected_node]

		return selected_node, deg_node, selected_arm

	def _pick_from_close(self, candidates):
		close_nodes = set(self._sample_graph.nodes()) - set(candidates)
		deg = dict(self._sample_graph.degree(close_nodes))
		print(' pick from ', len(deg), len(close_nodes))

		if len(deg) <= 10:
			return random.choice(list(candidates))

		sorted_deg = _mylib.sortDictByValues(deg,reverse=True)
		new_cand = set()

		for t in sorted_deg[:10]:
			n = t[0]
			nbs = self._sample_graph.neighbors(n)
			open_nb = set(nbs).intersection(candidates)
			print(len(open_nb))
			new_cand.update(open_nb)

		if len(new_cand) == 0:
			new_cand = candidates

		return random.choice(list(new_cand))

	def _cal_score(self, candidates):
		open_nodes = candidates
		close_nodes = set(self._sample_graph.nodes()) - set(candidates)

		degree_cand = dict(self._sample_graph.degree(candidates))
		degree_close = dict(self._sample_graph.degree(close_nodes))
		degree_avg_close = np.mean(np.array(degree_close.values()))

		pos_list = {}
		neg_list = {}
		for candidate in candidates:
			deg = degree_cand[candidate]
			deg_diff = deg - degree_avg_close
			print('			deg',deg)
			if deg_diff >= 0 :
				pos_list[candidate] = deg_diff
			else:
				neg_list[candidate] = deg_diff

		print('Total: {} -- Pos{} Neg{}'.format(len(candidates), len(pos_list),len(neg_list)))

		if len(pos_list) != 0:
			print('		[Cal S] Positive list', degree_avg_close)
			n = _mylib.pickMaxValueFromDict(pos_list)
		else:
			print('		[Cal S] Negative list', degree_avg_close)
			n = _mylib.pickMaxValueFromDict(neg_list)
			#n = random.choice(list(neg_list.keys()))


		return n

	def _pick_max_score(self, score):
		max_val = max(score.values())
		np_score = np.array(score.values())
		max_idx = np.argmax(np_score)
		node = score.keys()[max_idx]
		#print(np_score)
		print(' max-score pick:', score[node])
		return node

	def _neighbor_score(self, candidates, close_nodes):
		score = dict()
		cand_score = dict()
		for n in candidates:
			nbs = self._sample_graph.neighbors(n)
			s = []
			for nb in nbs:
				if nb not in score.keys():
					nbs_of_nb = self._sample_graph.neighbors(nb)
					nbs_close = set(nbs_of_nb).intersection(close_nodes)
					score[nb] = len(nbs_close) / len(nbs_of_nb)

				s.append(score[nb])
			cand_score[n] = np.mean(np.array(s))

		return cand_score

	def _getDenNodes(self, nodes):
		graph = self._sample_graph

		obs_deg = dict(graph.degree(nodes))
		sorted_list = _mylib.sortDictByValues(obs_deg, reverse=True)
		top_k = int(.20 * len(sorted_list))

		if top_k == 0:
			candidates = obs_deg.keys()
		else:
			# candidates = list()
			# tmp = list()
			# for x in sorted_list[:top_k]:
			# 	node = x[0]
			# 	deg = x[1]
			# 	tmp.append(node)
			# 	if deg > 1:
			# 		candidates.append(node)

			# if len(candidates) == 0: candidates = tmp
			candidates = [x[0] for x in sorted_list[:top_k]]
		return candidates

	def _getExpNodes(self):
		"""
		Generate a list of best expansion nodes based on clustering coeff

		Only the number of nubmer of nodes with count lowest clustering
		coefficient are to be considered for expansion

		Considers all the open nodes. Not just from latest subsample

		Args:
			None
		Return:
			list[str] -- The self._exp_cut_off nodes with lowest clustering coeff
		"""
		if len(self._sample['nodes']['open']) > 0:
			# Clustering coeff of the open nodes in the sample
			cc = nx.clustering(self._sample_graph, self._sample['nodes']['open'])
			# Sort the nodes by clustering coeff in ascending order
			cc = sorted(cc, key=cc.get, reverse=False)
			return cc[:self._exp_cut_off]
			#return cc
		else:
			print('	*No open nodes')
			return list(nodes)

	def _increment_cost(self, cost ):
		self._cost += cost


		if  int(self._cost) % self._log_interval == 0:
			obs_nodes = self._sample_graph.number_of_nodes()
			obs_edges = self._sample_graph.number_of_edges()

			c = int(self._cost)

			self._track[c] = obs_nodes
			self._track_edges[c] = obs_edges

			self._track_k.append(self._tmp)
			self._track_open.append(len(self._sample['nodes']['open']))




	def _max_excess_deg(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])
		rank = 0
		deg_observed = 0
		print(current_node in self._query._graph.nodes())

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node, rank=rank, deg_obs=deg_observed)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = set(sub_sample['nodes']['open'])

			if len(candidates) != 0:
				degree_excess = self._get_node_excess_degree(candidates)

				degree_excess_sorted = _mylib.sortDictByValues(degree_excess, reverse=True)
				current_node = degree_excess_sorted[0][0]

				#rank = self._get_node_rank_from_excess_degree(current_node, candidates)
				deg_observed = self._sample_graph.degree(current_node)

			if debug:
				print('			Number of nodes \t Close: {} \t Open: {}'.format( \
					len(sub_sample['nodes']['close']), \
					len(sub_sample['nodes']['open'])))

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _get_node_excess_degree(self, candidates):
		degree_true = dict(self._query._graph.degree(candidates))
		degree_observed = dict(self._sample_graph.degree(candidates))

		degree_excess = Counter(degree_true)
		degree_excess.subtract(Counter(degree_observed))

		return degree_excess

	def generate(self):
			"""
			The main method that calls all the other methods
			"""

			current_list = []

			if self._exp_type == 'exp-den' and self._bfs_count != 0:
				#self._bfs_init()
				self._smooth_init()

				self._after_init()

			# Sample until budget runs out or thero are no more open nodes
			while self._cost < self._budget:

				# If there are no more nodes in current list, use open nodes from sample
				if len(current_list) < 1:
					current_list = list(self._sample['nodes']['open'])

				# Perform expansion
				if len(current_list) != 0 and self._densi_count != 1:
					self._stage = 'exp'
					self._exp_count += 1
					if self._exp_type == 'oracle':
						current_list = self._getExpNodes()
						current = self._expansion_oracle(current_list)
					elif self._exp_type == 'random-exp':
						current = self._expansion_random(self._sample['nodes']['open'])
					elif self._exp_type == 'exp-den':
						current = self._expansion_random(self._sample['nodes']['open'])
						#current = self._expansion_cheat(self._sample['nodes']['open'])
				else:
					current = starting_node

				# Perform densification
				self._stage = 'den'

				if self._exp_type == 'rand':
					self.random_sampling()
				elif self._exp_type == 'rw':
					self._random_walk()
				elif self._exp_type == 'mod':
					self._max_obs_deg()
				elif self._exp_type == 'med':
					self._max_excess_deg()
				elif self._exp_type == 'oracle':
					current_list = self._densification_oracle(current)
				elif self._exp_type == 'sb':
					self._snowball_sampling()
				elif self._exp_type == 'bfs':
					self._bfs()
				elif self._exp_type == 'dfs':
					self._dfs()
				elif self._exp_type == 'hybrid':
					self._hybrid()
				elif self._exp_type == 'vmab':
					self._bandit()
				elif self._exp_type == 'opic':
					self._cash_spread()
				elif self._exp_type == 'exp-den':
					current_list = self._densification(current)


				print(" [Done] Sample size: {}".format(self._sample_graph.number_of_nodes()))

			"""
			repititions = 0
			for x in self._oracle._communities_selected:
				repititions += self._oracle._communities_selected[x]
			repititions = repititions - len(self._oracle._communities_selected)
			print(self._oracle._communities_selected, len(self._oracle._communities_selected), repititions)
			"""


def Logging(sample):
	# Keep the results in file
	track_sort = _mylib.sortDictByKeys(sample._track)
	cost_track = [x[0] for x in track_sort]
	obs_track = [x[1] for x in track_sort]


	log.log_new_nodes(log_file, dataset, type, obs_track, cost_track, budget, bfs_budget)

	print("---- DONE! ----")
	print("	# Exp: {}, # Den: {}".format(sample._exp_count, sample._densi_count))
	print('	Nodes in S: ', sample._sample_graph.number_of_nodes())
	#print('	Clustering Coeff =', nx.average_clustering(graph))
	print('-'*15)

def SaveToFile(results_nodes,results_edges, query_order, rank_order):
	log.save_to_file(log_file_node, results_nodes)
	log.save_to_file(log_file_edge, results_edges)
	log.save_to_file(log_file_order, query_order)
	#log.save_to_file(log_file_rank, rank_order)


def Append_Log(sample, type):
	track_sort = _mylib.sortDictByKeys(sample._track)
	track_edges_sort = _mylib.sortDictByKeys(sample._track_edges)
	cost_track = [x[0] for x in track_sort]
	obs_track = [x[1] for x in track_sort]
	obs_edges_track = [x[1] for x in track_edges_sort]

	Log_result[type] = Log_result.get(type, list()) + obs_track
	Log_result_edges[type] = Log_result_edges.get(type, list()) + obs_edges_track
	Log_result_step_sel_node[type] = Log_result_step_sel_node.get(type, list()) + sample._track_selected_node
	Log_result_step_sel_rank[type] = Log_result_step_sel_rank.get(type, list()) + sample._track_rank_selected
	Log_result_step_sel_rank[type+'.new'] = Log_result_step_sel_rank.get(type+'.new', list()) + sample._track_new_nodes


	return cost_track

def write_sample_network(sample_g, output_fname):
	print(' Write edgelist: ', out_fname)
	nx.write_edgelist(sample_g, output_fname)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-task', help='Type of sampling', default='undirected_single')
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-budget', help='Total budget', type=int, default=0)
	parser.add_argument('-percent_b', help='percent budget', type=int, default=0.1)
	parser.add_argument('-bfs_budget', help='Bfs budget', type=int, default=5)
	parser.add_argument('-p_bfs_budget', help='Percent Bfs budget', type=float, default=.015)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)
	parser.add_argument('-log', help='Log file', default='./log/')
	parser.add_argument('-experiment', help='# of experiment', default=10)
	parser.add_argument('-log_interval', help='# of budget interval for logging', type=int, default=10)
	parser.add_argument('-k', help='top k percent', type=int, default=5)
	parser.add_argument('-is_cost', help='take cost in account', type=bool, default=True)
	parser.add_argument('-mode', help='mode', type=int, default=1)
	parser.add_argument('-delimiter', help='csv delimiter', type=str, default=None)
	parser.add_argument('-debug', help='debug mode', type=bool, default=False)

	parser.add_argument('-remove', help='Adversary type', type=str, default='edge')
	parser.add_argument('-ad_p_remove', help='Adversary remove percentage', type=float, default=0.05)

	args = parser.parse_args()

	print(args)

	fname = args.fname
	budget = args.budget
	bfs_budget = args.bfs_budget
	p_bfs_budget = args.p_bfs_budget
	dataset = args.dataset
	log_file = args.log
	k = args.k
	is_cost = args.is_cost
	log_interval = args.log_interval
	mode = args.mode
	delimeter = args.delimiter
	debug = args.debug
	P_BUDGET = args.percent_b

	ad_type = args.remove
	ad_p_remove = args.ad_p_remove

	if mode == 1:
		exp_list = ['mod','rw', 'bfs', 'rand']
	elif mode == 2:
		exp_list = ['med', 'mod', 'rw', 'bfs', 'sb', 'dfs', 'rand', 'opic' ]
	elif mode == 3:
		exp_list = ['exp-den']


	print(exp_list)
	Log_result = {}
	Log_result_edges = {}
	Log_result_step_sel_node = {}
	Log_result_step_sel_rank = {}

	Log_result_hyb = {}

	if dataset == None:
		fname = fname.replace('\\', '/')
		f = fname.split('.')[1].split('/')[-1]
		dataset = f


	G = _mylib.read_file(fname)

	n = G.number_of_nodes()
	nodes_original = G.nodes()

	if budget == 0:
		budget = int(P_BUDGET*n)


	print('** {} Budget set to {}, n={}'.format(dataset, budget, n))


	print('-' * 10)


	# directory = './data-adv/edge_incomp/'
	directory = './samples/'


	for i in range(0, int(args.experiment)):
		starting_node = -1
		for ad_p_remove in ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]:

			if str(ad_p_remove) == "0":
				query_ = query.SingleLayer(G)
				starting_node = query_.randomHighDegreeNode()
			else:
				query_ = query_incomplete.SingleLayer(G, p=float(ad_p_remove))


			for type in exp_list:
				print('Crawler: {} \t p:{} \t i:{} \t start:{}'.format(type, ad_p_remove, i, starting_node))
				out_dir = directory + str(ad_p_remove) + '/' + dataset + '/' + type + '/'
				out_fname = out_dir + dataset + '_' + str(i + 1) + '_1'

				if os.path.isfile(out_fname):
					print(out_fname + ' exists')
					continue

				sample = UndirectedSingleLayer(query=query_, budget=budget, exp_type=type, dataset=dataset, k=k)

				sample.generate()
				cost_arr = Append_Log(sample, type)


				if not os.path.exists(out_dir):
					os.makedirs(out_dir)

				write_sample_network(sample._sample_graph, out_fname)


			print('--'*10)