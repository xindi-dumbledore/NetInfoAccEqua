from network_models import *
from fairness_measure import *
from simulation_param import *
import pickle


def run_exp_model(G_name):
    degree_parity_list = []
    shortest_path_list = []
    information_access_dict = defaultdict(list)
    print(G_name)
    for i in range(N_GRAPHS):
        if G_name == "Random":
            G = randomNetwork(M, E, E, N)
        elif G_name == "HomophilyBA":
            G = homophilyBA(M, E, E, H, H, ALPHA, N)
        elif G_name == "RandomHomophily":
            G = randomHomophily(M, E, E, H, H, N)
        elif G_name == "BA":
            G = BA(M, E, E, ALPHA, N)
        elif G_name == "DiversifiedHomophily":
            G = DiversifiedHomophilyBA(M, E, E, H, H, 0, PD, ED, ED, N)
        elif G_name == "DiversifiedHomophilyBA":
            G = DiversifiedHomophilyBA(M, E, E, 0.5, 0.5, ALPHA, PD, ED, ED, N)
        else:
            raise ValueError("Model name recognized!")
        groups = nx.get_node_attributes(G, "group")
        nodes_maj = [n for n in groups if groups[n] == 0]
        nodes_min = [n for n in groups if groups[n] == 1]
        # degree parity
        degree_parity_list.append(centrality_fairness(G, centrality="degree"))
        # shortest_path info
        shortest_path_list.append(shortest_path(G))
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
                        information_access_dict[(beta_array_name, minority_seeding_portion, threshold)].append(
                            (R_M, R_m, R))
    return degree_parity_list, shortest_path_list, information_access_dict


if __name__ == '__main__':
    for network_name in ["Random", "HomophilyBA", "RandomHomophily", "BA", "DiversifiedHomophilyBA", "DiversifiedHomophily"]:
        degree_parity_list, shortest_path_list, information_access_dict = run_exp_model(
            network_name)
        pickle.dump(degree_parity_list, open(
            "exp_results/exp_model/%s_degree_parity_list.pickle" % network_name, "wb"))
        pickle.dump(shortest_path_list, open(
            "exp_results/exp_model/%s_shortest_path_list.pickle" % network_name, "wb"))
        pickle.dump(information_access_dict, open(
            "exp_results/exp_model/%s_information_access_dict.pickle" % network_name, "wb"))
