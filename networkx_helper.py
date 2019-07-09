import networkx
import numpy as np


def get_graph_average_deg(graph):
	degree_sequence = [d for n, d in graph.degree()]
	avg_deg = np.average(np.array(degree_sequence))
	med_deg = np.median(np.array(degree_sequence))
	return avg_deg, med_deg