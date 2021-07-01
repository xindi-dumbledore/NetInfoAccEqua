from network_models import *
from fairness_measure import *
from simulation_param import *
import pickle


def run_exp_param(attr_name, attr_list):
    degree_parity_dict = defaultdict(list)
    shortest_path_dict = defaultdict(list)
    information_access_dict = defaultdict(list)
    print(attr_name)
    for attr in attr_list:
        for i in range(N_GRAPHS):
            if attr_name == "h":
                G = homophilyBA(M, E, E, attr, attr, ALPHA, N)
            elif attr_name == "m":
                G = homophilyBA(attr, E, E, H, H, ALPHA, N)
            elif attr_name == "alpha":
                G = homophilyBA(M, E, E, H, H, attr, N)
            elif attr_name == "pd":
                G = DiversifiedHomophilyBA(
                    M, E, E, H, H, ALPHA, attr, ED, ED, N)
            groups = nx.get_node_attributes(G, "group")
            nodes_maj = [n for n in groups if groups[n] == 0]
            nodes_min = [n for n in groups if groups[n] == 1]
            # degree parity
            degree_parity_dict[attr].append(
                centrality_fairness(G, centrality="degree"))
            # shortest path
            shortest_path_dict[attr].append(shortest_path(G))
            # information access
            for beta_array_name, beta_array in zip(["sym", "asy"], [BETA_ARRAY_SYM, BETA_ARRAY_ASY]):
                for minority_seeding_portion_type in ["low", "mid", "high"]:
                    for threshold in [None, 0.1]:  # 0.1 is activation threshold
                        for trial in range(N_TRIALS):
                            low_end, high_end = MINORITY_SEEDING_PORTION_DICT[
                                minority_seeding_portion_type]
                            minority_seeding_portion = random.uniform(
                                low_end, high_end)
                            seed_min = int(minority_seeding_portion * SEED_NUM)
                            seed_maj = SEED_NUM - seed_min
                            seed_nodes = random.sample(
                                nodes_min, seed_min) + random.sample(nodes_maj, seed_maj)
                            R_M, R_m, R = SIR_network(
                                G, beta_array, threshold, GAMMA, seed_nodes, SEED_NUM)
                            information_access_dict[(attr, beta_array_name, minority_seeding_portion, threshold)].append(
                                (R_M, R_m, R))
    return degree_parity_dict, shortest_path_dict, information_access_dict

if __name__ == '__main__':
    h_list = np.linspace(0.5, 1, 6)
    m_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_list = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
    pd_list = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    for parameter_name, parameter_space in zip(["h", "m", "alpha", "pd"], [h_list, m_list, alpha_list, pd_list]):
        degree_parity_dict, shortest_path_dict, information_access_dict = run_exp_param(
            parameter_name, parameter_space)
        pickle.dump(degree_parity_dict, open(
            "exp_results/exp_param/%s_degree_parity_dict.pickle" % parameter_name, "wb"))
        pickle.dump(shortest_path_dict, open(
            "exp_results/exp_param/%s_shortest_path_dict.pickle" % parameter_name, "wb"))
        pickle.dump(information_access_dict, open(
            "exp_results/exp_param/%s_information_access_dict.pickle" % parameter_name, "wb"))
