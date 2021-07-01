import networkx as nx
from fairness_measure import *
from simulation_param import *
import random
import pickle


def run_exp_real(G, key="gender", group_info=["male", "female"]):
    G = G.to_undirected()
    groups = nx.get_node_attributes(G, key)
    nodes_maj = [n for n in groups if groups[n] == group_info[0]]
    nodes_min = [n for n in groups if groups[n] == group_info[1]]
    G = G.subgraph(nodes_maj + nodes_min)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc)
    nodes_maj = [n for n in nodes_maj if n in largest_cc]
    nodes_min = [n for n in nodes_min if n in largest_cc]
    print(len(G), len(nodes_maj), len(nodes_min))
    seed_num = int(len(G) * 0.002)
    if seed_num < 5:
        seed_num = 5
    # degree parity
    degree_parity_dict = centrality_fairness(
        G, centrality="degree", key=key, group_info=group_info)
    # structural hole
    information_access_dict = defaultdict(list)
    for beta_array_name, beta_array in zip(["asy", "sym"], [BETA_ARRAY_ASY, BETA_ARRAY_SYM]):
        for minority_seeding_portion_type in ["low", "mid", "high"]:
            for threshold in [None, 0.1]:  # 0.1 is activation threshold
                for trial in range(N_TRIALS):
                    low_end, high_end = MINORITY_SEEDING_PORTION_DICT[
                        minority_seeding_portion_type]
                    minority_seeding_portion = random.uniform(
                        low_end, high_end)
                    seed_min = int(minority_seeding_portion * seed_num)
                    seed_maj = seed_num - seed_min
                    seed_nodes = random.sample(
                        nodes_min, seed_min) + random.sample(nodes_maj, seed_maj)
                    R_M, R_m, R = SIR_network(
                        G, beta_array, threshold, GAMMA, seed_nodes, seed_num, key=key, group_info=group_info)
                    information_access_dict[(beta_array_name, minority_seeding_portion, threshold)].append(
                        (R_M, R_m, R))
    return degree_parity_dict, information_access_dict

if __name__ == '__main__':
    key_dict = {"Github": "gender",
                "DBLP": "gender",
                "APS": "pacs",
                }
    group_dict = {"Github": ["male", "female"],
                  "DBLP": ["m", "f"],
                  "APS": ['05.30.-d', '05.20.-y']}
    for G_name, G_file_name in zip(["Github", "DBLP", "APS"], ["github_mutual_follower_ntw", "DBLP_graph", "sampled_APS_pacs052030"]):
        G = nx.read_gexf("datasets/%s.gexf" % G_file_name)
        degree_parity_dict, information_access_dict = run_exp_real(
            G, key=key_dict[G_name], group_info=group_dict[G_name])
        pickle.dump(degree_parity_dict, open(
            "exp_results/exp_real/%s_degree_parity_dict.pickle" % G_name, "wb"))
        pickle.dump(information_access_dict, open(
            "exp_results/exp_real/%s_information_access_dict.pickle" % G_name, "wb"))
