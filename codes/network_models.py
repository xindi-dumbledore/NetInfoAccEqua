import networkx as nx
import random


def randomNetwork(m, e0, e1, N):
    e_dict = {0: e0, 1: e1}
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        group = nx.get_node_attributes(G, "group")
        # random select nodes
        selected_nodes = random.choices(sorted(group), k=e_dict[g])
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target) for target in selected_nodes])
    return G


def randomHomophily(m, e0, e1, h0, h1, N):
    homo_dict = {0: h0, 1: h1}
    e_dict = {0: e0, 1: e1}
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        # calculate connection probability to each node
        group = nx.get_node_attributes(G, "group")
        info = []
        for node in sorted(group):
            g_n = group[node]
            h = homo_dict[g]
            if g_n == g:
                p = h
            else:
                p = 1 - h
            info.append(p)
        # select node based on the probability
        selected_nodes = random.choices(
            sorted(group), weights=info, k=e_dict[g])  # currently doing select with replacement
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target) for target in selected_nodes])
    return G


def BA(m, e0, e1, alpha, N):
    e_dict = {0: e0, 1: e1}
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        # calculate connection probability to each node
        degree = dict(G.degree())
        group = nx.get_node_attributes(G, "group")
        info = []
        for node in sorted(group):
            g_n = group[node]
            d_n = degree[node]
            if g_n == g:
                p = d_n**alpha
            else:
                p = d_n**alpha
            info.append(p)
        # select node based on the probability
        selected_nodes = random.choices(
            sorted(group), weights=info, k=e_dict[g])  # currently doing select with replacement
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target) for target in selected_nodes])
    return G


def homophilyBA(m, e0, e1, h0, h1, alpha, N):
    homo_dict = {0: h0, 1: h1}
    e_dict = {0: e0, 1: e1}
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        # calculate connection probability to each node
        degree = dict(G.degree())
        group = nx.get_node_attributes(G, "group")
        info = []
        for node in sorted(group):
            g_n = group[node]
            d_n = degree[node]
            h = homo_dict[g]
            if g_n == g:
                p = h * d_n**alpha
            else:
                p = (1 - h) * d_n**alpha
            info.append(p)
        # select node based on the probability
        selected_nodes = random.choices(
            sorted(group), weights=info, k=e_dict[g])  # currently doing select with replacement
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target) for target in selected_nodes])
    return G


def DiversifiedHomophilyBA(m, e0, e1, h0, h1, alpha, po, eo0, eo1, N, weighted=True):
    homo_dict = {0: h0, 1: h1}
    e_dict = {0: e0, 1: e1}
    eo_dict = {0: eo0, 1: eo1}
    portion = {0: po, 1: 1 - po}  # different group 1-po
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        # calculate connection probability to each node
        degree = dict(G.degree())
        group = nx.get_node_attributes(G, "group")
        info = []
        for node in sorted(group):
            g_n = group[node]
            d_n = degree[node]
            h = homo_dict[g]
            if g_n == g:
                p = h * d_n**alpha
            else:
                p = (1 - h) * d_n**alpha
            info.append(p)
        # select node based on the probability
        selected_nodes = random.choices(
            sorted(group), weights=info, k=e_dict[g] - eo_dict[g])
        # select friends of friends who are of opposite group of me based on
        # degree
        info_dict = {}
        for n in selected_nodes:
            degree_n = degree[n]
            neighbors = list(G.neighbors(n))
            for neighbor in neighbors:
                # if group[neighbor] != g:
                degree_neighbor = degree[neighbor]
                distance = abs(degree_neighbor - degree_n)
                if neighbor in info_dict:
                    if distance > info_dict[neighbor]:
                        info_dict[neighbor] = portion[
                            group[neighbor] == g] / (distance + 0.1)
                else:
                    info_dict[neighbor] = portion[
                        group[neighbor] == g] / (distance + 0.1)
        if len(info_dict) > 0:
            fof, info = list(zip(*info_dict.items()))
            fof, info = list(fof), list(info)
            if weighted:
                selected_fof = random.choices(
                    fof, weights=info, k=min(len(info), eo_dict[g]))
            else:
                selected_fof = random.choices(
                    fof, k=min(len(info), eo_dict[g]))
        else:
            selected_fof = []
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target)
                          for target in selected_nodes + selected_fof])
    return G
