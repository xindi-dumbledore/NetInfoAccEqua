import networkx as nx
import random
import numpy as np
from scipy.stats import ks_2samp, moment, wasserstein_distance
from collections import Counter
import time
from collections import defaultdict
timestr = time.strftime("%Y%m%d-%H%M%S")


def centrality_fairness(G, centrality="degree", key="group", group_info=[0, 1]):
    group = nx.get_node_attributes(G, key)
    if centrality == "degree":
        f = nx.degree
    elif centrality == "pagerank":
        f = nx.pagerank
    elif centrality == "betweenness":
        f = nx.betweenness_centrality
    value_dict = dict(f(G))
    value_gM = [value_dict[node]
                for node in group if group[node] == group_info[0]]
    value_gm = [value_dict[node]
                for node in group if group[node] == group_info[1]]
    emd = wasserstein_distance(value_gM, value_gm)
    power_inequality = np.mean(value_gm) / \
        np.mean(value_gM)  # power inequality
    moment_gc = moment(value_gm, 2) / moment(value_gM, 2)
    # numbers used to calculate dyadicity and heterophilicity
    edge_type = []
    for edge in G.edges():
        edge_type.append(tuple(sorted((group[edge[0]], group[edge[1]]))))
    edge_type_count = Counter(edge_type)
    group_count = Counter(group.values())
    results = {"group_count": group_count,
               "edge_type_count": edge_type_count,
               "value_gM": value_gM,
               "value_gm": value_gm,
               "emd": emd,
               "power_inequality": power_inequality,
               "moment_gc": moment_gc,
               }
    return results


def shortest_path(G, key="group"):
    # calculate within group and between group
    length = dict(nx.all_pairs_shortest_path_length(G))
    length_values = []
    groups = nx.get_node_attributes(G, key)
    shortest_path_group = defaultdict(list)
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if i == j:
                continue
            node_i, node_j = nodes[i], nodes[j]
            group_i, group_j = groups[node_i], groups[node_j]
            group_tuple = tuple(sorted([group_i, group_j]))
            shortest_path_group[group_tuple].append(length[node_i][node_j])
            length_values.append(length[node_i][node_j])
    shortest_path_info = {"shortest_path": {}, "diameter": {}}
    for group in shortest_path_group:
        shortest_path_info["shortest_path"][
            group] = np.mean(shortest_path_group[group])
        shortest_path_info["diameter"][group] = max(shortest_path_group[group])
    avg_path_length = np.mean(length_values)
    diameter = max(length_values)
    shortest_path_info["shortest_path"]["full"] = avg_path_length
    shortest_path_info["diameter"]["full"] = diameter
    return shortest_path_info


def structural_hole(G, key="group", group_info=[0, 1]):
    group = nx.get_node_attributes(G, key)
    esize = nx.effective_size(G)
    efficiency = {n: v / G.degree(n) for n, v in esize.items()}
    esize_maj = [esize[node] for node in group if group[node] == group_info[0]]
    esize_min = [esize[node] for node in group if group[node] == group_info[1]]
    efficiency_maj = [efficiency[node] for node in group if group[node] == 0]
    efficiency_min = [efficiency[node] for node in group if group[node] == 1]
    results = {"effective_size": {"majority": esize_maj, "minority": esize_min},
               "efficiency": {"majority": efficiency_maj, "minority": efficiency_min}}
    return results


def SIR_network(G, beta_array, threshold, gamma, seed_nodes, seed_num, key="group", group_info=[0, 1]):
    nodes = list(G.nodes())
    # print(len(nodes))
    node_mapping = dict(zip(nodes, range(len(nodes))))
    group_mapping = {group_info[0]: 0, group_info[1]: 1}
    seeds = random.sample(list(seed_nodes), seed_num)
    # initialze state vector: S:0, I:1, R:2
    state = np.array([0] * len(nodes))
    for i in seeds:
        state[node_mapping[i]] = 1
    # record number of S and I at each time step for each group
    group = nx.get_node_attributes(G, key)
    group_M_node = list(
        [node for node in group if group[node] == group_info[0]])
    group_m_node = list(
        [node for node in group if group[node] == group_info[1]])
    I_M_list, I_m_list = [], []
    # R_M_list, R_m_list = [], []
    transition_edge_count = []
    iteration = 0
    state_t = []
    while True:
        state_M, state_m = state[[node_mapping[i] for i in group_M_node]], state[
            [node_mapping[i] for i in group_m_node]]
        count_M, count_m = Counter(state_M), Counter(state_m)
        state_t.append(state)
        # + count_M[2]) # do we want I + R or only R or only I?
        I_M_list.append(count_M[1])
        I_m_list.append(count_m[1])  # + count_m[2])
        # R_M_list.append(count_M[2])
        # R_m_list.append(count_m[2])
        state_next = state.copy()
        transition_edge_step = []
        for i, node in enumerate(nodes):
            # if state[node_mapping[node]] == 1:
            #     if random.random() < gamma:  # I->R
            #         state_next[node_mapping[node]] = 2
            if state[node_mapping[node]] == 0:  # I infect S
                neighbors = [n for n in G[node]]
                infectious_neighbors = [
                    n for n in neighbors if state[node_mapping[n]] == 1]
                if threshold is None:
                    for j in infectious_neighbors:
                        if random.random() < beta_array[group_mapping[group[j]]][group_mapping[group[node]]]:
                            state_next[node_mapping[node]] = 1
                            transition_edge_step.append((
                                group[j], group[node]))
                            break
                else:
                    threshold_count = len(neighbors) * threshold
                    infect_count = 0
                    temp = []
                    for j in infectious_neighbors:
                        if random.random() < beta_array[group_mapping[group[j]]][group_mapping[group[node]]]:
                            infect_count += 1
                            temp.append((group[j], group[node]))
                        if infect_count >= threshold_count:
                            state_next[node_mapping[node]] = 1
                            transition_edge_step += temp
                            break
        transition_edge_count.append(Counter(transition_edge_step))
        if (state_next == state).all():
            break
        if threshold is not None:
            if iteration == 1000:
                break
        state = state_next
        iteration += 1
    I_list = list((np.array(I_M_list) + np.array(I_m_list)) / len(nodes))
    I_M_list, I_m_list = list(np.array(
        I_M_list) / len(group_M_node)), list(np.array(
            I_m_list) / len(group_m_node))
    # R_M_list, R_m_list = list(np.array(
    #     R_M_list) / len(group_M_node)), list(np.array(
    #         R_m_list) / len(group_m_node))
    return I_M_list, I_m_list, I_list
