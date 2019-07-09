from __future__ import print_function
import csv
import time
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import collections
import community


import random
import pickle
import scipy.stats as stats
import os

# import networkx2gt

try:
    import networkx2gt
    from graph_tool.all import *
except ImportError:
    pass

def logToFileCSV(data,filename,isAppend=False):
    if isAppend: mode = 'a'
    elif not isAppend: mode = 'w'

    with open(filename, mode) as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data)

def sortDictByValues(dict,reverse=False):
    d = dict.copy()
    return sorted(d.items(), key=operator.itemgetter(1), reverse=reverse)

def sortDictByKeys(dict,reverse=False):
    d = dict.copy()
    od = collections.OrderedDict(sorted(d.items(), reverse=reverse))
    return od.items()

def pickMaxValueFromDict(d):
    max_val = max(d.values())
    np_score = np.array(d.values())
    max_idx = np.where(np_score == max_val)[0]
    sel = random.choice(max_idx)
    key = d.keys()[sel]
    return key

def pickMinValueFromDict(d):
    min_score = min(d.values())
    np_score = np.array(d.values())
    mix_idx = np.where(np_score == min_score)[0]

    sel = random.choice(mix_idx)

    key = d.keys()[sel]
    return key

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def plotLineGraph(lines,legend=None,title=None,x_axis_text=None, y_axis_text=None,save=True, log=True):

    for line in lines:
        if log:
            plt.semilogy(line)
        else:
            plt.plot(line, marker='o',linestyle='-',markersize=3)

    if legend != None:
        plt.legend(legend, loc='upper left')

    plt.title(title)
    if save: plt.savefig('./draw/plot/linegraph_'+str(time.time()) + '.png')
    else: plt.show()

    plt.clf()

def plotTwoLineGraph(lines,legend=None,title=None,x_axis_text=None, y_axis_text=None,save=True, log=True):

    fig, ax1 = plt.subplots()

    if log:
        ax1.semilogy(lines[0])
    else:
        ax1.plot(lines[0])

    ax2 = ax1.twinx()
    if log:
        ax2.semilogy(lines[1], 'r')
    else:
        ax2.plot(lines[1], 'r')

    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    if legend != None:
        ax1.set_ylabel(legend[0], color='b')
        ax2.set_ylabel(legend[1], color='r')
    # if legend != None:
    #     plt.legend(legend, loc='upper left')

    plt.title(title)
    if save: plt.savefig('./draw/plot/linegraph_'+str(time.time()) + '.png')
    else: plt.show()

    plt.clf()

def dummyPlot(x_data):
    plt.plot(x_data)
    plt.ylabel('some numbers')
    plt.show()

def multiple_scatter_plot(nrows, ncols, x,y,titles=None):
    plt.figure(1)

    for i, plot in enumerate(x):
        plt.subplot(nrows, ncols, (i + 1))
        plt.scatter(x[i],y[i])
        if titles != None:
            plt.title(titles[i])

    plt.show()
    plt.ion()

def plot_hist(plots, auto_bins=True, titles=None,save=True, nrows=1, ncols=1 ):
    plt.figure(1)

    for i, plot in enumerate(plots):
        dmax = max(plot)
        dmin = min(plot)
        bins = dmax - dmin
        #print ("bin max %s min %s -- %s").format(dmax, dmin,bins)
        if bins == 0:
            bins = 10
        # #print ' %s bin= %s' % (i, max_min)

        plt.subplot(nrows,ncols,(i+1))
        if auto_bins:
            plt.hist(plot,bins=dmax)
        else:
            plt.hist(plot)

        if titles != None:
            plt.title(titles[i])

    if not save:
        plt.show()
    else:
        plt.savefig('./draw/plot/hist_s' + str(time.time()) + '.png')

    plt.clf()


# def degreePlotHist(nrows, ncols, plots, auto_bins=True, titles=None,save=False,bins=10):
#     plt.figure(1)
#
#     for i, plot in enumerate(plots):
#         dmax = max(plot)
#         dmin = min(plot)
#         bins = dmax - dmin
#         #print "     bin max %s min %s -- %s" % (dmax, dmin,bins)
#         if bins == 0:
#             bins = 10
#         # #print ' %s bin= %s' % (i, max_min)
#
#         plt.subplot(nrows,ncols,(i+1))
#         if auto_bins:
#             plt.hist(plot,bins=dmax)
#         else:
#             plt.hist(plot)
#
#         if titles != None:
#             plt.title(titles[i])
#
#     if not save:
#         plt.show()
#     else:
#         plt.savefig('./draw/plot/hist_s' + str(time.time()) + '.png')
#
#     plt.clf()

def ccPlotHist(nrows, ncols, plot, titles=None,save=False,bins=10):

    plt.hist(plot, bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    plt.yscale('log', nonposy='clip')

    if titles != None:
        plt.title(titles)

    if not save:
        plt.show()
    else:
        plt.savefig('./draw/plot/hist_s' + str(time.time()) + '.png')

    plt.clf()



def scatterPlot(x, y, xlabels=None, ylabels=None, title=None,save=False):
    plt.scatter(x, y,  s=5, color='red')
   # plt.xscale('log', nonposy='clip')
    #plt.yscale('log', nonposy='clip')

    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title(title)
    if not save: plt.show()
    else: plt.savefig('./plot/scatter_'+ str(time.time()) + '.png')

    plt.clf()

def correlationXY(x,y,x_name=None,y_name=None, save=False):
    correlation = np.corrcoef(x, y)
    #print "Correlation = %s " % correlation[0][1]

    scatterPlot(x, y, ylabels=y_name, xlabels=x_name, save=save, title="Correlation: " + str(correlation[0][1]))

def pairwise_correlation_matrix(features_mat,column=None):
    if column is not None:
        frame = pd.DataFrame(features_matrix, columns=column)
    #['deg', 'deg_nb', 'cc', 'cc_nb', 'ego_edges', 'ego_edges_out', 'deg_cen', 'page_rank','betweenness']
    else:
        frame = pd.DataFrame(features_mat)
    #print frame.corr()

def draw_com(G,partition, with_labels=False, node_size=30,com_color=True,save_png=True):
    values = [partition.get(node) for node in G.nodes()]

    #cmap = 'Blues'
    cmap = plt.cm.RdYlBu
    if com_color == True:
        nx.draw_spring(G, cmap=plt.get_cmap(cmap), node_color=values, node_size=node_size, with_labels=with_labels)
        #nx.draw_spring(G, cmap=plt.get_cmap(cmap), node_color=values2, node_size=50, with_labels=True)
    else:
        nx.draw_spring(G, cmap=plt.get_cmap(cmap), node_size=node_size, with_labels=with_labels)
    if not save_png:
        plt.show()
    else:
        plt.savefig('./draw/com/densi_' + str(time.time()) + '.png')

    plt.clf()


# def PlotDegree(edges_set):
#     s = nx.Graph()
#     s.add_edges_from(edges_set)
#
#     degree_sequence = sorted(nx.degree(s).values(), reverse=True)  # degree sequence
#     dmax = max(degree_sequence)
#     dmin = min(degree_sequence)
#     max_min = dmax - dmin
#
#     n, bins, patches = plt.hist(degree_sequence, bins=max_min)
#     # plt.title("Degree rank plot")
#     # plt.ylabel("degree")
#     # plt.xlabel("rank")
#
#     # plt.savefig("degree_histogram.png")
#     plt.show()

def get_members_from_com(com_id, partition):
    valueList = np.array(partition.values())
    indices = np.argwhere(valueList == com_id)
    indices = indices.reshape(len(indices))

    keyList = np.array(partition.keys())

    nodes = keyList[indices]
    return nodes

def remove_entries_from_dict(entries, the_dict):
    d = the_dict.copy()
    for key in entries:
        if key in d:
            del d[key]
    return d

def remove_node_with_deg(g,degree_removed=1):
    deg = g.degree()
    filter_nodes = []

    # Remove n-degree nodes
    for node in g.nodes():
        if deg[node] > degree_removed:
            filter_nodes.append(node)

    # Graph with n-deg nodes removed.
    return g.subgraph(filter_nodes)

def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y)
    return z

def sort_tuple_list(t_l,index=1,reverse=True):
	return sorted(t_l, key=lambda tup: tup[index], reverse=reverse)

def remove_zero_values(the_dict):
    zero_l = []
    not_zero_l = []
    for t in the_dict:
        k = t[0]
        v = t[1]
        if v != 0.0:
            not_zero_l.append(t)
        else:
            zero_l.append(t)
    return not_zero_l, zero_l

def remove_one_values(the_dict):
    remain_dict = {}
    for k,v in the_dict.iteritems():
        if v < 0.9:
            remain_dict[k] = v
    return remain_dict

def plot_bar_chart(plot, texts, title=None, x_label=None, y_label=None):
    ind = np.arange(len(plot))
    print(len(plot))
    print(ind)
    rects = plt.bar(ind, plot, 0.1, color='b')
    autolabel(rects,texts)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.show()
    plt.savefig('./draw/plot/bar_' + str(time.time()) + '.png')
    plt.clf()

def autolabel(rects,texts):
    # attach some text labels
    for i, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%s' % texts[i],
                ha='center', va='bottom')

def degreeRankPlot(deg, save=True):
    degree_sequence = sorted(deg, reverse=True)
    dmax = max(degree_sequence)
    plt.plot(degree_sequence, 'b-', marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")

    if not save:
        plt.show()
    else:
        plt.savefig('./draw/plot/loglog_' + str(time.time()) + '.png')
    plt.clf()

def degreeHist(deg,log_log=True, save=True):
    values = (sorted(deg))
    hist = [deg.count(x) for x in (values)]

    plt.figure()
    plt.grid(True)
    if log_log: plt.loglog(values, hist ,'r', marker='o', markersize=3)
    else: plt.plot(values, hist, 'r-')

    # plt.legend(['In-degree','Out-degree']#)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Degree distribution')
   # plt.xlim([0, 2 * 10 ** 2])
    if not save:
        plt.show()
    else:
        plt.savefig('./draw/loglog_' + str(time.time()) + '.png')

    plt.clf()

def degreeHist_2(deg_l,log_log=True, save=True, legend=['In-degree','Out-degree'] ):
    values_1 = sorted(deg_l[0])
    hist_1 = [deg_l[0].count(x) for x in values_1]

    values_2 = sorted(deg_l[1])
    hist_2 = [deg_l[1].count(x) for x in values_2]

    plt.figure()
    plt.grid(True)
    if log_log:
        plt.loglog(values_1, hist_1, 'r', marker='o', markersize=3)
        plt.loglog(values_2, hist_2, 'b', marker='o', markersize=3)
    else:
        plt.plot(values_1, hist_1, 'r')
        plt.plot(values_2, hist_2, 'b')

    plt.legend(legend)

    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Degree distribution')

   # plt.xlim([0, 2 * 10 ** 2])
    if not save:
        plt.show()
    else:
        plt.savefig('./draw/plot/deg_2_' + str(time.time()) + '.png')

    plt.clf()

def distributionPlot(deg,log_log=True, save=True, y_label="Freq", x_label="value", title=""):
    values = (sorted(deg))
    hist = [deg.count(x) for x in (values)]

    plt.figure()
    plt.grid(True)
    if log_log: plt.loglog(values, hist ,'r', marker='o', markersize=2)
    else: plt.scatter(values, hist, s=2)


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('distribution '+ title)

    if not save:
        plt.show()
    else:
        plt.savefig('./draw/plot/distribution' + str(time.time()) + '.png')

    plt.clf()


def calculate_density(g):
    n = g.number_of_nodes()
    m = g.number_of_edges()

    density = m / ((n*(n-1.)) / 2.)

    return density

def calculate_density_ahn(g):
    n = g.number_of_nodes()
    m = g.number_of_edges()

    if n == 1:
        return 0

    a = m - (n-1)
    b = (((n * (n - 1.)) / 2.) - (n-1))
    try:
        density = a / b
    except ZeroDivisionError:
        density = 0.
    print(n, m, density)
    return density


def create_gt_graph(g_nx):
    edges = g_nx.edges()
    edges = list(set(edges))

    gt_graph = Graph(directed=0)
    gt_graph.add_edge_list(edges, hashed=True)
    return gt_graph

def draw_graph_tool_ori(g):
    gt_graph = networkx2gt.nx2gt(g)

    pos = sfdp_layout(gt_graph, max_iter=100, multilevel=True)

    graph_draw(gt_graph, pos, output_size=(500, 500), output='./draw/networks/network_' + str(time.time()) + '.png')
    # graph_draw(gt_graph, pos, output_size=(1000, 1000), output='./draw/networks/network_' + str(time.time()) + '.png')

def draw_graph_tool(g, first_l):
    d_t = {}
    for n in g.nodes():
        if n in set(first_l):
            d_t[n] = 50
        else:
            d_t[n] = 20
	#
	#
    nx.set_node_attributes(g, 'first', d_t)


    gt_graph = networkx2gt.nx2gt(g)


    #print gt_graph.list_properties()
    prop = gt_graph.vertex_properties["first"]


    print(len(set(first_l)), g.number_of_nodes())
    #print(set(first_l).difference(set(g.nodes())))

   # prop = gt_graph.new_vertex_property("int32_t")
    #deg = gt_graph.degree_property_map("in")

    # for n in set(first_l):
    #     print(n)
    #     prop[gt_graph.vertex(n)] = 100

    # for v in gt_graph.vertices():
    #     print(v)


    pos = sfdp_layout(gt_graph, max_iter=100, multilevel=True)

    graph_draw(gt_graph, pos,vertex_size=prop, vertex_fill_color=prop, output_size=(10000, 10000), output='./draw/networks/network_' + str(time.time()) + '.pdf')
    # graph_draw(gt_graph, pos, output_size=(1000, 1000), output='./draw/networks/network_' + str(time.time()) + '.png')

def read_mtx_file(fname):
    file = open(fname, "r")
    edges_list = []
    for i, line in enumerate(file.readlines()):
        s = (line[0])
        if s == "%":
            continue
        line = str.replace(line, '\n', '')
        line = str.replace(line, ',', ' ')
        tmp = line.split(' ')[:2]

        edges_list.append(tuple(tmp))

    return edges_list

def read_pajek_file(fname):
    G = nx.read_pajek(fname)
    return G

def read_file(fname, isDirected=False, printInfo=False):
    ext = fname.split('.')[-1]
    if printInfo:
        print('     < Reading.. \'{}\' format, {} isDirected: {} '.format(ext, fname, isDirected))

    if ext == 'mtx' or ext == 'edges':
        edges_list = read_mtx_file(fname)

        if not isDirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        G.add_edges_from(edges_list)
    elif ext == 'pickle':
        G = pickle.load(open(fname, 'rb'))
    elif ext == 'net':
        G = read_pajek_file(fname)
    else:
        if not isDirected:
            G = nx.read_edgelist(fname)
        else:
            G = nx.read_edgelist(fname, create_using=nx.DiGraph())


    # Select giant component and remove all self-loop edges
    if not isDirected:
        graph = max(nx.connected_component_subgraphs(G), key=len)
    else:
        graph = max(nx.weakly_connected_component_subgraphs(G), key=len)

    selfloop_edges = graph.selfloop_edges()
    graph.remove_edges_from(selfloop_edges)

    if printInfo:
        print(' ***** Before *****')
        print(nx.info(G))
        print(' ***** Final ***** ')
        print(nx.info(graph))
        print(' - ' * 10)


    return graph

def get_keys_by_value(d, target_val=0):
    keys = np.array(d.keys())
    vals = np.array(d.values())

    indices = np.where(vals == target_val)[0]
    return keys[indices].tolist()

def get_max_values_from_dict(d, candidates=list()):
    if len(candidates) == 0:
        candidates = list(d.keys())


    keys = np.array(d.keys())
    vals = np.array(d.values())

    # Index of all candidates
    ix = np.in1d(keys.ravel(), candidates).reshape(keys.shape)

    # Get all the values of candidates and find the max
    max_val = np.amax(vals[np.where(ix)])

    # Get all indices of max value
    max_val_index = np.where(vals == max_val)

    #print(type(ix), type(max_val_index))

    return random.choice(list(keys[max_val_index])), max_val_index


def get_rank_correlation(d_1, d_2, k=.5):
	cutoff = int(k * len(d_1))
	d_1_sorted = sortDictByValues(d_1,reverse=True)

	l_1 = []
	l_2 = []
	for count, d in enumerate(d_1_sorted):
		id = d[0]
		val = d[1]

		l_1.append(val)
		l_2.append(d_2[id])

		if count == cutoff:
			#print(count)
			break

	tau, p_value = stats.kendalltau(l_1, l_2)
	return tau, p_value

def get_community(G, dataset):
    com_fname = './data/pickle/communities_{}.pickle'.format(dataset)
    if os.path.isfile(com_fname):
        partition = pickle.load(open(com_fname, 'rb'))
    else:
        partition = community.best_partition(G)
        pickle.dump(partition, open(com_fname, 'wb'))

    return partition

def get_modularity(partition, graphfile):
    q = community.modularity(partition, graphfile)
    return q

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


#name = "as-caida20051205"
#file = open('C:\\Users\\Bebe\\Downloads\\caida\\'+ name + '.txt', "r")
#edges_list = []
#f_out = open('./data-pajek/' + name +'.txt', 'a')

#for i, line in enumerate(file.readlines()):
#    if line[0] != "#":
#        line = str.replace(line, '\n', '').split('\t')

#        new_line = line[0] + ' ' + line[1]
#        print(new_line, file=f_out)

#print(name)
