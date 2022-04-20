# calculate nodes and edges similarity using the list of nodes and the list of edges instead of
# a graphviz object
def calculate_nodes_similarity(nodes1, nodes2):
    value, diff_added, diff_removed = calculate_list_similarity(nodes1, nodes2)
    return value, diff_added, diff_removed


def calculate_edges_similarity(edges1, edges2):
    value, diff_added, diff_removed = calculate_list_similarity(edges1, edges2)
    return value, diff_added, diff_removed


def calculate_list_similarity(list1, list2):
    diff_removed = set(list1).difference(set(list2))
    diff_added = set(list2).difference(set(list1))

    inter = set(list1).intersection(set(list2))
    value = 2 * len(inter) / (len(list1) + len(list2))
    return value, diff_added, diff_removed
