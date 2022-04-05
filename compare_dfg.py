import networkx as nx
import pygraphviz as pgv


# convert the graphviz object (model generated using Pm4Py DFG) into pygraphviz Agraph
# then convert to networkx agraph for using network functions
def get_nxgraph_from_gviz(gviz):
    graph = pgv.AGraph(gviz.source)
    nxgraph = nx.nx_agraph.from_agraph(graph)
    return nxgraph


# remove frequency information from activity name, returning a new process map
def remove_frequencies_from_labels(g):
    # remove frequency information from labels
    mapping = {}
    for node in g.nodes.data():
        old_label = node[1]['label']
        new_label = old_label.partition('(')[0]
        mapping[node[0]] = new_label
    g_new = nx.relabel_nodes(g, mapping)
    return g_new


# remove nodes not existent in both process maps
def remove_different_nodes(g1, g2, diff_nodes):
    for node in diff_nodes:
        if node in g1.nodes:
            g1.remove_node(node)
        if node in g2.nodes:
            g2.remove_node(node)
    return g1, g2


# return the set of labels (activity names) without frequency
def get_labels(g):
    nodes_g = [n[1]['label'] for n in g.nodes.data()]
    labels_g = [l.partition('(')[0] for l in nodes_g]
    return labels_g


# def calculate_nodes_similarity(model1, model2):
#     # convert the graphviz model to Agraph (from networkx)
#     agraph_model1 = get_nxgraph_from_gviz(model1)
#     agraph_model2 = get_nxgraph_from_gviz(model2)
#
#     labels_g1 = get_labels(agraph_model1)
#     labels_g2 = get_labels(agraph_model2)
#
#     diff_removed = set(labels_g1).difference(set(labels_g2))
#     diff_added = set(labels_g2).difference(set(labels_g1))
#
#     inter = set(labels_g1).intersection(set(labels_g2))
#     value = 2 * len(inter) / (len(labels_g1) + len(labels_g2))
#     return value, diff_added, diff_removed
#
#
# def calculate_edges_similarity(model1, model2):
#     # calulate the nodes similarity first
#     nodes_similarity_value, added_nodes, removed_nodes = calculate_nodes_similarity(model1, model2)
#
#     # convert the graphviz model to Agraph (from networkx)
#     agraph_model1 = get_nxgraph_from_gviz(model1)
#     agraph_model2 = get_nxgraph_from_gviz(model2)
#
#     # remove frequencies from the labels
#     new_g1 = remove_frequencies_from_labels(agraph_model1)
#     new_g2 = remove_frequencies_from_labels(agraph_model2)
#
#     # if the nodes similarity is different than 1
#     # IPDD removes the different nodes
#     # then it calculated the edges similarity metric
#     if nodes_similarity_value < 1:
#         new_g1, new_g2 = remove_different_nodes(new_g1, new_g2, set.union(added_nodes, removed_nodes))
#
#     # get the different edges
#     diff_removed = nx.difference(new_g1, new_g2)
#     for e in diff_removed.edges:
#         diff_removed.add(e)
#
#     diff_added = nx.difference(new_g2, new_g1)
#     for e in diff_added.edges:
#         diff_added.add(e)
#
#     # calculate the edges similarity metric
#     inter = set(new_g1.edges).intersection(set(new_g2.edges))
#     value = 2 * len(inter) / (len(new_g1.edges) + len(new_g2.edges))
#     return value, diff_added, diff_removed

# calculate nodes and edges similarity using the list of nodes and the list of edges instead of
# a graphviz object
def calculate_nodes_similarity(nodes1, nodes2):
    diff_removed = set(nodes1).difference(set(nodes2))
    diff_added = set(nodes2).difference(set(nodes1))

    inter = set(nodes1).intersection(set(nodes2))
    value = 2 * len(inter) / (len(nodes1) + len(nodes2))
    return value, diff_added, diff_removed


def calculate_edges_similarity(edges1, edges2):
    diff_removed = set(edges1).difference(set(edges2))
    diff_added = set(edges2).difference(set(edges1))

    inter = set(edges1).intersection(set(edges2))
    value = 2 * len(inter) / (len(edges1) + len(edges2))
    return value, diff_added, diff_removed
