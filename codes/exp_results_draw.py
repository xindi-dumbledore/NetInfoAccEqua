import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import numpy as np
import scipy.stats as st
MINORITY_SEEDING_PORTION_BAR = [0, 0.3, 0.7, 1]
sns.set_context("paper", font_scale=1.4)


def get_average_sem(trials):
    max_length = max([len(trial) for trial in trials])
    extended_trials = [trial + [trial[-1]] *
                       (max_length - len(trial)) for trial in trials]
    avg = [sum(x) / float(len(extended_trials))
           for x in itertools.zip_longest(*extended_trials)]
    sem = [stats.sem(x) for x in itertools.zip_longest(*extended_trials)]
    return avg, sem


def dyadicity_heterophilicity(group_count, edge_group_count, group_info=[0, 1]):
    # print(group_count, edge_group_count, group_info)
    number_of_nodes = sum(list(group_count.values()))
    number_of_edges = sum(list(edge_group_count.values()))
    p = 2 * number_of_edges / (number_of_nodes * (number_of_nodes - 1))
    m11_exp = group_count[group_info[1]] * \
        (group_count[group_info[1]] - 1) / 2 * p
    m01_exp = group_count[group_info[0]] * group_count[group_info[1]] * p
    m00_exp = group_count[group_info[0]] * \
        (group_count[group_info[0]] - 1) / 2 * p
    # print(m11_exp, m01_exp, m00_exp)
    H = edge_group_count[tuple(
        sorted([group_info[0], group_info[1]]))] / m01_exp
    D0 = edge_group_count[(group_info[0], group_info[0])] / m00_exp
    D1 = edge_group_count[(group_info[1], group_info[1])] / m11_exp
    results = {
        "Dyadicity_majority": D0,
        "Dyadicity_minority": D1,
        "Heterophilicity": H}
    return results


def degree_information_draw(degree_information_dict, figname, xlabel=""):
    degree_records = []
    for attr in degree_information_dict:
        if attr == "Github":
            group_info = ["male", "female"]
        elif attr == "DBLP":
            group_info = ["m", "f"]
        elif attr == "APS":
            group_info = ['05.30.-d', '05.20.-y']
        else:
            group_info = [0, 1]
        degree_information = degree_information_dict[attr]
        if type(degree_information) != list:
            degree_information = [degree_information]
        for item in degree_information:
            emd = item["emd"]
            power_inequality = item["power_inequality"]
            moment_gc = item["moment_gc"]
            D_H_results = dyadicity_heterophilicity(
                item["group_count"], item["edge_type_count"], group_info=group_info)
            D0, D1, H = D_H_results["Dyadicity_majority"], D_H_results[
                "Dyadicity_minority"], D_H_results["Heterophilicity"]
            degree_records.append({"attr": attr,
                                   "emd": emd,
                                   "power_inequality": power_inequality,
                                   "moment_gc": moment_gc,
                                   "D0": D0,
                                   "D1": D1,
                                   "H": H
                                   })
    degree_info_df = pd.DataFrame.from_records(degree_records)
    # set color palette
    if len(degree_information_dict) > 3:
        color_palette = {"Random": sns.color_palette()[0],
                         "HomophilyBA": sns.color_palette()[1],
                         "Random\nHomophily": sns.color_palette()[2],
                         "BA": sns.color_palette()[3],
                         "Diversified\nHomophily\nBA": sns.color_palette()[4],
                         "Diversified\nHomophily": sns.color_palette()[5]}
    else:
        color_palette = {"Github": sns.color_palette()[0],
                         "DBLP": sns.color_palette()[1],
                         "APS": sns.color_palette()[2]}
    # plot D and H
    D_H_pivot = pd.melt(degree_info_df[["attr", "D0", "D1", "H"]], id_vars=[
                        "attr"], value_vars=["D0", "D1", "H"])
    ax = sns.barplot(x="attr", y="value", hue="variable",
                     data=D_H_pivot)

    if type(attr) == str and len(set(degree_info_df["attr"])) > 3:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[r"$D_{maj}$", r"$D_{min}$", r"$H$"])
    plt.xlabel(xlabel)
    plt.ylabel("Dyadicity/Heterophilicity")
    plt.tight_layout()
    plt.savefig("../%s_DH.pdf" % figname)
    plt.close()
    if xlabel != "":
        sns.barplot(x="attr", y="emd", data=degree_info_df,
                    palette=sns.color_palette("dark:salmon_r", len(set(degree_info_df["attr"]))))
    else:
        order = degree_info_df.groupby("attr")["emd"].mean(
        ).reset_index().sort_values("emd")["attr"]
        sns.barplot(x="attr", y="emd", order=order,
                    data=degree_info_df, palette=color_palette)
    plt.xlabel(xlabel)
    plt.ylabel("Earth Mover Distance")
    plt.axhline(0, color="black", ls="--")
    plt.ylim(-0.1, 1)
    if type(attr) == str and len(set(degree_info_df["attr"])) > 3:
        plt.gca().set_xticklabels(plt.gca().get_xticklabels(), fontsize=10)
    plt.tight_layout()
    plt.savefig("../%s_emd.pdf" % figname)
    plt.close()
    # plot power inequality
    if xlabel != "":
        sns.barplot(x="attr", y="power_inequality", data=degree_info_df,
                    palette=sns.color_palette("dark:salmon_r", len(set(degree_info_df["attr"]))))
    else:
        order = list(degree_info_df.groupby("attr")["power_inequality"].mean(
        ).reset_index().sort_values("power_inequality")["attr"])[::-1]
        sns.barplot(x="attr", y="power_inequality",
                    order=order, data=degree_info_df, palette=color_palette)
    plt.xlabel(xlabel)
    plt.ylabel("Power Inequality")
    if type(attr) == str and len(set(degree_info_df["attr"])) > 3:
        plt.gca().set_xticklabels(plt.gca().get_xticklabels(), fontsize=10)
    plt.axhline(1, color="black", ls="--")
    plt.tight_layout()
    plt.savefig("../%s_power_inequality.pdf" % figname)
    plt.close()
    # plot moment gc
    if xlabel != "":
        sns.barplot(x="attr", y="moment_gc", data=degree_info_df,
                    palette=sns.color_palette("dark:salmon_r", len(set(degree_info_df["attr"]))))
    else:
        order = list(degree_info_df.groupby("attr")["moment_gc"].mean(
        ).reset_index().sort_values("moment_gc")["attr"])[::-1]
        sns.barplot(x="attr", y="moment_gc", order=order,
                    data=degree_info_df, palette=color_palette)
    plt.axhline(1, color="black", ls="--")
    plt.xlabel(xlabel)
    plt.ylabel("Moment Glass Ceiling")
    if type(attr) == str and len(set(degree_info_df["attr"])) > 3:
        plt.gca().set_xticklabels(plt.gca().get_xticklabels(), fontsize=10)
    plt.tight_layout()
    plt.savefig("../%s_moment_gc.pdf" % figname)
    plt.close()


def shortest_path_length_draw(shortest_path_dict, figname, xlabel=""):
    shortest_path_records = []
    ylabel = {"shortest_path": "Average Shortest Path Length",
              "diameter": "Diameter"}
    for attr in shortest_path_dict:
        for key in ["shortest_path", "diameter"]:
            shortest_path_information = shortest_path_dict[attr]
            for item in shortest_path_information:
                shortest_path_records.append({"attr": attr,
                                              "All": item[key]["full"],
                                              "Majority": item[key][(0, 0)],
                                              "Minority": item[key][(1, 1)],
                                              "Majority-Minority": item[key][(0, 1)]
                                              })
            shortest_path_info_df = pd.DataFrame.from_records(
                shortest_path_records)
            # plot shortest path info
            shortest_path_info_pivot = pd.melt(shortest_path_info_df, id_vars=[
                "attr"], value_vars=["All", "Majority", "Minority", "Majority-Minority"])
            ax = sns.barplot(x="attr", y="value",
                             data=shortest_path_info_pivot[shortest_path_info_pivot["variable"] == "All"], palette=sns.color_palette("dark:salmon_r", len(set(shortest_path_info_pivot["attr"]))))
            if type(attr) == str and len(set(shortest_path_info_df["attr"])) > 3:
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel[key])
            plt.tight_layout()
            plt.savefig("../%s_%s.pdf" % (figname, key))
            plt.close()


def truncate_process(growth_list):
    max_index = np.argmax(growth_list)
    growth_list_truncate = growth_list[:max_index + 1]
    return growth_list_truncate


def get_reach_diff(RM, Rm, R):
    max_length = max(len(RM), len(Rm), len(R))
    RM += [RM[-1]] * (max_length - len(RM))
    Rm += [Rm[-1]] * (max_length - len(Rm))
    R += [R[-1]] * (max_length - len(R))
    percent_diff = {}
    for percent in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        current_time = int(max_length * percent / 100)
        if current_time < 1:
            current_time = 1
        percent_diff[percent] = (RM[current_time - 1],
                                 Rm[current_time - 1], R[current_time - 1])
    return percent_diff


def get_reach_diff_raw(RM, Rm, R):
    # version 2, not doing percentage
    max_length = max(len(RM), len(Rm), len(R))
    RM += [RM[-1]] * (max_length - len(RM))
    Rm += [Rm[-1]] * (max_length - len(Rm))
    R += [R[-1]] * (max_length - len(R))
    percent_diff = {}
    for percent in range(0, max_length, 1):
        percent_diff[percent] = (RM[percent],
                                 Rm[percent], R[percent])
    return percent_diff


def information_access_draw(info_access_dict, rate_key, type_key, network_list, figname):
    portion_dict = {0: "Low", 1: "Mid", 2: "High"}
    records, records_raw = [], []
    for each_network in info_access_dict:
        keys_consider = []
        for key in info_access_dict[each_network]:
            if key[0] == rate_key and key[2] == type_key:
                keys_consider.append(key)
        low_keys, mid_keys, high_keys = [], [], []
        for key in keys_consider:
            min_portion = key[-2]
            min_portion_bin = np.digitize(
                [min_portion], MINORITY_SEEDING_PORTION_BAR)[0]
            if min_portion_bin == 4:
                min_portion_bin = 3
            if min_portion_bin == 1:
                low_keys.append(key)
            elif min_portion_bin == 2:
                mid_keys.append(key)
            else:
                high_keys.append(key)
        for i, keys in enumerate([low_keys, mid_keys, high_keys]):
            RM_list, Rm_list, R_list = [], [], []
            for key in keys:
                RM_list += [truncate_process(trial[0])
                            for trial in info_access_dict[each_network][key]]
                Rm_list += [truncate_process(trial[1])
                            for trial in info_access_dict[each_network][key]]
                R_list += [truncate_process(trial[2])
                           for trial in info_access_dict[each_network][key]]
            for each_RM, each_Rm, each_R in zip(RM_list, Rm_list, R_list):
                percent_diff = get_reach_diff(each_RM, each_Rm, each_R)
                percent_diff_raw = get_reach_diff_raw(each_RM, each_Rm, each_R)
                for key in percent_diff:
                    record = {"Network": each_network,
                              "Minority Seeding Portion": portion_dict[i],
                              "t/T": int(key),
                              "reach_M": percent_diff[key][0] + 10**(-4),
                              "reach_m": percent_diff[key][1] + 10**(-4),
                              "reached_total": percent_diff[key][2]}
                    records.append(record)
                for key in percent_diff_raw:
                    record_raw = {"Network": each_network,
                                  "Minority Seeding Portion": portion_dict[i],
                                  "t": int(key),
                                  "reach_M": percent_diff_raw[key][0],
                                  "reach_m": percent_diff_raw[key][1] + 10**(-4),
                                  "reached_total": percent_diff_raw[key][2]}
                    records_raw.append(record_raw)
    # calculate reach difference
    df_reach = pd.DataFrame.from_records(records)
    df_reach_raw = pd.DataFrame.from_records(records_raw)
    df_reach["reach_diff"] = (df_reach["reach_M"] -
                              df_reach["reach_m"]) / (df_reach["reach_M"] + df_reach["reach_m"])
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, minority_seeding_portion in enumerate(["Low", "High"]):
        df_select = df_reach[
            df_reach["Minority Seeding Portion"] == minority_seeding_portion]
        df_mean = df_select.groupby(
            ["Network", "t/T"])["reach_diff"].mean().reset_index()
        df_aggregate = df_mean.sort_values(
            "t/T").groupby(["Network"])['reach_diff'].apply(list).reset_index()
        reach_diff_dict = dict(
            zip(df_aggregate["Network"], df_aggregate["reach_diff"]))
        reach_diff_list = [reach_diff_dict[network]
                           for network in network_list]
        reach_diff_array = np.array(reach_diff_list)
        if i < 2:
            im = sns.heatmap(reach_diff_array, vmin=-1, vmax=1, cmap="vlag",
                             ax=ax[i], cbar=None)
        else:
            im = sns.heatmap(reach_diff_array, vmin=-1, vmax=1, cmap="vlag", ax=ax[
                i])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels(range(0, 110, 10), rotation=90)
        ax[i].set_xlabel("t/T")
        ax[i].set_title(minority_seeding_portion)
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.85])
    mappable = im.get_children()[0]
    cbar = fig.colorbar(mappable, cax=cb_ax)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.outline.set_visible(False)
    ax[0].set_yticklabels(network_list, rotation=0)
    plt.tight_layout()
    plt.savefig("../%s_reached_diff_heatmap.pdf" %
                figname, bbox_inches='tight')
    plt.close()
    # reach portion
    reach_select = "reached_total"  # "reached_total"
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, minority_seeding_portion in enumerate(["Low", "High"]):
        df_select = df_reach_raw[df_reach_raw[
            "Minority Seeding Portion"] == minority_seeding_portion]
        df_mean = df_select.groupby(["Network", "t"])[
            reach_select].mean().reset_index()
        df_aggregate = df_mean.sort_values("t").groupby(
            ["Network"])[reach_select].apply(list).reset_index()
        reach_diff_dict = dict(
            zip(df_aggregate["Network"], df_aggregate[reach_select]))
        reach_diff_list = [reach_diff_dict[network]
                           for network in network_list]
        pad = len(max(reach_diff_list, key=len))
        reach_diff_array = np.array(
            [i + [1] * (pad - len(i)) for i in reach_diff_list])
        reach_diff_array = reach_diff_array[:, :21]
        if i < 2:
            im = sns.heatmap(reach_diff_array, vmin=0, vmax=1,
                             cmap="Greens", ax=ax[i], cbar=None)
        else:
            im = sns.heatmap(reach_diff_array, vmin=0, vmax=1,
                             cmap="Greens", ax=ax[i])
        ax[i].set_yticklabels([])
        ax[i].set_xlabel("t")
        ax[i].set_title(minority_seeding_portion)
        if reach_diff_array.shape[1] > 30:
            ax[i].set_xticks(list(range(0, reach_diff_array.shape[1] + 1, 10)))
            ax[i].set_xticklabels(
                list(range(0, reach_diff_array.shape[1] + 1, 10)), rotation=90)
        else:
            ax[i].set_xticks(list(range(0, reach_diff_array.shape[1] + 1, 5)))
            ax[i].set_xticklabels(
                list(range(0, reach_diff_array.shape[1] + 1, 5)), rotation=90)
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.85])
    mappable = im.get_children()[0]
    cbar = fig.colorbar(mappable, cax=cb_ax)
    cbar.outline.set_visible(False)
    ax[0].set_yticklabels(network_list, rotation=0)
    plt.tight_layout()
    plt.savefig("../%s_reached_heatmap.pdf" %
                figname, bbox_inches='tight')
    plt.close()


def information_access_draw_attr(info_access_dict, rate_key, type_key, figname, attrname):
    portion_dict = {0: "Low", 1: "Mid", 2: "High"}
    records, records_raw = [], []
    attrs_keys = {}
    for key in info_access_dict:
        portion = key[2]
        portion_bin = np.digitize(
            [portion], MINORITY_SEEDING_PORTION_BAR)[0]
        if portion_bin == 4:
            portion_bin = 3
        if key[1] == rate_key and key[-1] == type_key:
            if key[0] in attrs_keys:
                attrs_keys[key[0]][portion_dict[portion_bin - 1]].append(key)
            else:
                attrs_keys[key[0]] = {"Low": [], "Mid": [], "High": []}
                attrs_keys[key[0]][portion_dict[portion_bin - 1]].append(key)
    for attr in attrs_keys:
        low_keys, mid_keys, high_keys = attrs_keys[attr][
            "Low"], attrs_keys[attr]["Mid"], attrs_keys[attr]["High"]
        for i, keys in enumerate([low_keys, mid_keys, high_keys]):
            RM_list, Rm_list, R_list = [], [], []
            for key in keys:
                RM_list += [truncate_process(trial[0])
                            for trial in info_access_dict[key]]
                Rm_list += [truncate_process(trial[1])
                            for trial in info_access_dict[key]]
                R_list += [truncate_process(trial[2])
                           for trial in info_access_dict[key]]
            RM_mean, RM_sem = get_average_sem(RM_list)
            Rm_mean, Rm_sem = get_average_sem(Rm_list)
            for each_RM, each_Rm, each_R in zip(RM_list, Rm_list, R_list):
                percent_diff = get_reach_diff(each_RM, each_Rm, each_R)
                percent_diff_raw = get_reach_diff_raw(each_RM, each_Rm, each_R)
                for key in percent_diff:
                    record = {attrname: attr,
                              "Minority Seeding Portion": portion_dict[i],
                              "t/T": int(key),
                              "reach_M": percent_diff[key][0] + 10**(-4),
                              "reach_m": percent_diff[key][1] + 10**(-4),
                              "reached_total": percent_diff[key][2]}
                    records.append(record)
                for key in percent_diff_raw:
                    record_raw = {attrname: attr,
                                  "Minority Seeding Portion": portion_dict[i],
                                  "t": int(key),
                                  "reach_M": percent_diff_raw[key][0],
                                  "reach_m": percent_diff_raw[key][1] + 10**(-4),
                                  "reached_total": percent_diff_raw[key][2]}
                    records_raw.append(record_raw)
    # # plot reach difference
    df_reach = pd.DataFrame.from_records(records)
    df_reach_raw = pd.DataFrame.from_records(records_raw)
    df_reach["reach_diff"] = (df_reach["reach_M"] -
                              df_reach["reach_m"]) / (df_reach["reach_M"] + df_reach["reach_m"])
    # heatmap
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, minority_seeding_portion in enumerate(["Low", "High"]):
        df_select = df_reach[
            df_reach["Minority Seeding Portion"] == minority_seeding_portion]
        df_mean = df_select.groupby(
            [attrname, "t/T"])["reach_diff"].mean().reset_index()
        df_aggregate = df_mean.sort_values(
            "t/T").groupby([attrname])['reach_diff'].apply(list).reset_index()
        reach_diff_dict = dict(
            zip(df_aggregate[attrname], df_aggregate["reach_diff"]))
        attr_list = sorted(list(df_aggregate[attrname]))
        reach_diff_list = [reach_diff_dict[attr] for attr in attr_list]
        reach_diff_array = np.array(reach_diff_list)
        if i < 2:
            im = sns.heatmap(reach_diff_array, vmin=-1, vmax=1, cmap="vlag",
                             ax=ax[i], cbar=None)
        else:
            im = sns.heatmap(reach_diff_array, vmin=12, vmax=1, cmap="vlag", ax=ax[
                i])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels(range(0, 110, 10), rotation=90)
        ax[i].set_xlabel("t/T")
        ax[i].set_title(minority_seeding_portion)
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.85])
    mappable = im.get_children()[0]
    cbar = fig.colorbar(mappable, cax=cb_ax)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.outline.set_visible(False)
    ax[0].set_yticklabels(attr_list, rotation=0)
    ax[0].set_ylabel(attrname, rotation=0, labelpad=10)
    plt.tight_layout()
    plt.savefig("../%s_reached_diff_heatmap.pdf" %
                figname, bbox_inches='tight')
    plt.close()
    # reach portion
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, minority_seeding_portion in enumerate(["Low", "High"]):
        df_select = df_reach_raw[df_reach_raw[
            "Minority Seeding Portion"] == minority_seeding_portion]
        df_mean = df_select.groupby([attrname, "t"])[
            "reached_total"].mean().reset_index()
        df_aggregate = df_mean.sort_values("t").groupby(
            [attrname])['reached_total'].apply(list).reset_index()
        reach_diff_dict = dict(
            zip(df_aggregate[attrname], df_aggregate["reached_total"]))
        attr_list = sorted(list(df_aggregate[attrname]))
        reach_diff_list = [reach_diff_dict[attr] for attr in attr_list]
        pad = len(max(reach_diff_list, key=len))
        reach_diff_array = np.array(
            [i + [1] * (pad - len(i)) for i in reach_diff_list])
        reach_diff_array = reach_diff_array[:, :21]
        if i < 2:
            im = sns.heatmap(reach_diff_array, vmin=0, vmax=1,
                             cmap="Greens", ax=ax[i], cbar=None)
        else:
            im = sns.heatmap(reach_diff_array, vmin=0,
                             vmax=1, cmap="Greens", ax=ax[i])
        ax[i].set_yticklabels([])
        ax[i].set_xlabel("t")
        ax[i].set_title(minority_seeding_portion)
        if reach_diff_array.shape[1] > 30:
            ax[i].set_xticks(list(range(0, reach_diff_array.shape[1] + 1, 10)))
            ax[i].set_xticklabels(
                list(range(0, reach_diff_array.shape[1] + 1, 10)), rotation=90)
        else:
            ax[i].set_xticks(list(range(0, reach_diff_array.shape[1] + 1, 5)))
            ax[i].set_xticklabels(
                list(range(0, reach_diff_array.shape[1] + 1, 5)), rotation=90)
    ax[0].set_yticklabels(attr_list, rotation=0)
    ax[0].set_ylabel(attrname, rotation=0, labelpad=10)
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.85])
    mappable = im.get_children()[0]
    cbar = fig.colorbar(mappable, cax=cb_ax)
    cbar.outline.set_visible(False)
    plt.tight_layout()
    plt.savefig("../%s_reached_heatmap.pdf" %
                figname, bbox_inches='tight')
    plt.close()


def exp_model_draw():
    network_name_list = ["Random", "HomophilyBA", "RandomHomophily",
                         "BA", "DiversifiedHomophilyBA", "DiversifiedHomophily"]
    pretty_network_name_list = ["Random", "HomophilyBA", "Random\nHomophily",
                                "BA", "Diversified\nHomophily\nBA", "Diversified\nHomophily"]
    network_degree_info, network_shortest_path_info, info_access_dict = {}, {}, {}
    for name, pretty_name in zip(network_name_list, pretty_network_name_list):
        network_degree_info[pretty_name] = pickle.load(
            open("../exp_results/exp_model/%s_degree_parity_list.pickle" % name, "rb"))
        network_shortest_path_info[pretty_name] = pickle.load(
            open("../exp_results/exp_model/%s_shortest_path_list.pickle" % name, "rb"))
        info_access_dict[pretty_name.replace("\n", " ")] = pickle.load(open(
            "../exp_results/exp_model/%s_information_access_dict.pickle" % name, "rb"))
    degree_information_draw(network_degree_info, "figures/exp_model/network")
    shortest_path_length_draw(
        network_shortest_path_info, "figures/exp_model/network")
    network_list = [item.replace("\n", " ")
                    for item in pretty_network_name_list]
    for rate_key in ["sym", "asy"]:
        for type_key in [None, 0.1]:
            information_access_draw(info_access_dict, rate_key, type_key, network_list, "figures/exp_model/%s_%s" % (
                rate_key, type_key))


def exp_param_draw():
    for param in ["alpha", "h", "m", "pd"]:
        # print(param)
        if param == "alpha":
            attrname = r"$\alpha$"
        elif param == "pd":
            attrname = r"$p_d$"
        else:
            attrname = r"$%s$" % param
        degree_info = pickle.load(
            open("../exp_results/exp_param/%s_degree_parity_dict.pickle" % param, "rb"))
        shortest_path_info = pickle.load(
            open("../exp_results/exp_param/%s_shortest_path_dict.pickle" % param, "rb"))
        spreading_info = pickle.load(
            open("../exp_results/exp_param/%s_information_access_dict.pickle" % param, "rb"))
        degree_information_draw(
            degree_info, "figures/exp_param/%s" % param, attrname)
        shortest_path_length_draw(shortest_path_info,
                                  "figures/exp_param/homoBA_%s" % param, attrname)
        for rate_key in ["sym", "asy"]:
            for type_key in [None, 0.1]:
                information_access_draw_attr(spreading_info, rate_key, type_key,
                                             "figures/exp_param/%s_%s_%s" % (param, rate_key, type_key), attrname)


def exp_real_draw():
    network_name_list = ["Github", "DBLP", "APS"]
    network_degree_info, info_access_dict = {}, {}
    for name in network_name_list:
        network_degree_info[name] = pickle.load(
            open("../exp_results/exp_real/%s_degree_parity_dict.pickle" % name, "rb"))
        info_access_dict[name] = pickle.load(open(
            "../exp_results/exp_real/%s_information_access_dict.pickle" % name, "rb"))
    degree_information_draw(network_degree_info, "figures/exp_real/real")
    for rate_key in ["sym", "asy"]:
        for type_key in [None, 0.1]:
            information_access_draw(info_access_dict, rate_key, type_key, network_name_list, "figures/exp_real/%s_%s" % (
                rate_key, type_key))


if __name__ == '__main__':
    exp_model_draw()
    # exp_param_draw()
    # exp_real_draw()
