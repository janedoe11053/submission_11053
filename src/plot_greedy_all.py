import os
import math
import json
import itertools

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import hyperx.pytorch as pp
import seaborn as sns
import matplotlib.pyplot as plt


from explain_all import load_model, load_instance, load_baseline, get_explanation_path, \
    load_importance_scores, get_instance_dir


def get_instance_generator():
    experiments_path = Path('experiments.csv')
    experiments = pd.read_csv(experiments_path, header=0, sep=';')

    instances_grouped = experiments.groupby(["test_x_path", "model_path", "baseline_num", "baseline_seed",
                                             "instance_id"])

    for name, instance_data in instances_grouped:
        yield instance_data


def get_subset_mask(idx_list, n_features):
    indices = torch.tensor(idx_list, dtype=torch.long)
    mask = torch.zeros(n_features, dtype=torch.bool)
    mask[indices] = True
    return mask


def get_im_data(instance_x, masked_f, mean_scores, n_features, n_highest=None):
    if len(mean_scores) == 0:
        return pd.DataFrame(), 0

    if n_highest is None:
        n_highest = n_features

    descending_importance = True
    keep_features = True

    n_disp_features = min(n_features, n_highest)

    columns = ['x'] + [explainer_id for explainer_id in mean_scores.keys()]
    x_vals = range(n_disp_features + 1)
    data = pd.DataFrame(index=x_vals, columns=columns)
    data['x'] = x_vals

    full_val = masked_f(instance_x, get_subset_mask([i for i in range(n_features)], n_features))

    for explainer_id, scores in mean_scores.items():

        abs_indices = torch.argsort(scores.abs(), descending=descending_importance)
        highest_abs_subset = abs_indices[:n_highest]

        highest_abs_scores = scores[highest_abs_subset]
        indices = torch.argsort(highest_abs_scores.abs(), descending=descending_importance)
        scores_truncared_sorted = scores[highest_abs_subset[indices]]
        scores_idx_truncated_sorted = highest_abs_subset[indices]
        row_idx = 0
        while row_idx < data.shape[0]:
            row = data.iloc[row_idx]
            x = row['x']
            # if x > 0 and abs(scores_truncared_sorted[x - 1]) == float("inf"):
            #     data.loc[data['x'] == x, explainer_id] = float("NaN")
            #     continue
            step = 1
            if x > 0:
                # add all features that have the almost the same score
                while (x + step - 1 < indices.shape[0]) and \
                        (scores_truncared_sorted[x + step - 1] - scores_truncared_sorted[x - 1]).abs() < 1.0E-15:
                    step += 1
                x += step - 1
            idx = scores_idx_truncated_sorted[:x]

            subset_mask = get_subset_mask(idx.tolist(), n_features)
            if not keep_features:
                subset_mask = ~subset_mask
            masked_val = masked_f(instance_x, subset_mask)

            val = masked_val - full_val
            if not keep_features:
                val = -val
            data.loc[(data['x'] >= row['x']) & (data['x'] <= x), explainer_id] = val
            row_idx += step
        data[explainer_id] = data[explainer_id].astype(float)

    return data, full_val


def get_masked_im(model, instance, baseline):
    instance_torch = [(torch.tensor(instance), torch.zeros(1, 1))]
    baseline_torch = [(torch.tensor(baseline), torch.zeros(*baseline.shape))]
    impact_measure = pp.ImpactMeasure.build(
        instance_torch, baseline_torch, model, loss=None
    )

    def f(instance_x, subset_mask):
        metric_val = next(impact_measure(subset_mask))
        return metric_val.item()

    return f


def get_budget(n, c):
    return math.comb(n, c) * n


def get_explainer_budget_from_key(explainer_key_str, n):
    explainer = None
    budget = None
    key_items = explainer_key_str.split('-')
    if explainer_key_str == "random_shap":
        explainer = "random_shap"
        budget = None
    elif "perm_shap" == key_items[0]:
        explainer = key_items[0]
        perm_budget = int(key_items[1])
        if perm_budget == 2:
            perm_budget = 1  # because permutation shap crashes when #perms == 1
        budget = perm_budget * n
    elif "shap" in key_items[0]:
        explainer = key_items[0]
        budget = int(key_items[1])
    elif "lime" in key_items[0]:
        explainer = key_items[0]
        budget = int(key_items[1])
    elif "deg" == key_items[0]:
        explainer = 'hyperx'
        l_item, u = key_items[1].split(',')
        l = int(l_item.split('=')[-1])
        u = int(u)
        budget = get_budget(n, u - l)
    elif "eig" == key_items[0]:
        explainer = 'hyperx'
        l_item, u = key_items[2].split(',')
        l = int(l_item.split('=')[-1])
        u = int(u)
        budget = get_budget(n, u - l)
    if explainer is None:
        raise ValueError(F"Unknown explainer type: '{explainer_key_str}'")

    return explainer, budget


def group_by_complexity(plot_data, complexity, n_features):
    if complexity > 0:
        budget_prev = get_budget(n_features, complexity - 1)
    else:
        budget_prev = 0
    max_budget = get_budget(n_features, complexity)

    grouped_data = pd.DataFrame(index=plot_data.index)

    for explainer_key in plot_data.columns:
        explainer, budget = get_explainer_budget_from_key(explainer_key, n_features)
        if (explainer != "random_shap") and (budget <= budget_prev or budget > max_budget):
            continue

        grouped_data[explainer_key] = plot_data[explainer_key]

    return grouped_data


def group_similar_functions(data):
    same_vals = []
    left_cols = [col for col in data.columns if col != 'x']
    for explainer in [col for col in data.columns if col != 'x']:
        if explainer not in left_cols:
            continue
        group = [explainer]
        left_cols.remove(explainer)
        for explainer_2 in left_cols:
            if ('eig' in explainer) and ('deg' in explainer_2) or ('eig' in explainer_2) and ('deg' in explainer):
                continue
            if (data.loc[:, explainer] - data.loc[:, explainer_2]).abs().sum() / data.shape[0] < 1.0e-4:
                group += [explainer_2]

        left_cols = [col for col in left_cols if col not in group]
        same_vals.append(group)

    grouped = pd.DataFrame(index=data.index)
    grouped['x'] = data['x']
    for group in same_vals:
        group_name = '\n'.join(group)
        grouped[group_name] = data[group[0]]

    return grouped


def get_styles_grouped(explainer_ids, ex_styles=None):
    styles = {}

    dashesset_eig = itertools.cycle(('dotted', 'dashed', 'dashdot'))
    markers_eig = itertools.cycle(('o', 'X'))
    dashesset_deg = itertools.cycle(('dotted', 'dashed', 'dashdot'))
    markers_deg = itertools.cycle(('o', 'X'))
    colors_eig = itertools.cycle(('teal', 'turquoise', 'mediumaquamarine', 'steelblue', 'deepskyblue', 'lightskyblue'))
    colors_deg = itertools.cycle(('indigo', 'darkviolet', 'mediumslateblue', 'mediumvioletred', 'purple', 'magenta'))
    for explainer_id in explainer_ids:
        styles[explainer_id] = {}

        if ex_styles is not None:
            found = False
            for st in ex_styles:
                if explainer_id in st:
                    styles[explainer_id] = st[explainer_id]
                    found = True
                    break
            if found:
                continue

        if 'eig' in explainer_id:
            styles[explainer_id]['color'] = next(colors_eig)
            styles[explainer_id]['dashes'] = next(dashesset_eig)
            styles[explainer_id]['marker'] = next(markers_eig)

        elif 'deg' in explainer_id:
            styles[explainer_id]['color'] = next(colors_deg)
            styles[explainer_id]['dashes'] = next(dashesset_deg)
            styles[explainer_id]['marker'] = next(markers_deg)

        elif 'random' in explainer_id:
            styles[explainer_id]['color'] = 'red'
            styles[explainer_id]['dashes'] = 'dotted'
            styles[explainer_id]['marker'] = 'X'
        elif 'shap' in explainer_id:
            styles[explainer_id]['dashes'] = 'dashdot'
            if 'kernel' in explainer_id:
                styles[explainer_id]['color'] = 'darkorange'
                styles[explainer_id]['marker'] = 'v'
            elif 'perm' in explainer_id:
                styles[explainer_id]['color'] = 'darkgoldenrod'
                styles[explainer_id]['marker'] = '<'
            elif 'sampling' in explainer_id:
                styles[explainer_id]['color'] = 'olive'
                styles[explainer_id]['marker'] = '^'
        elif 'lime' in explainer_id:
            styles[explainer_id]['color'] = 'darkolivegreen'
            styles[explainer_id]['marker'] = 'v'
            styles[explainer_id]['dashes'] = 'dotted'
    return styles


def plot_functions(data: pd.DataFrame, plot_title: str, y_label: str, x_label: str, plot_path: Path,
                   styles):

    f, axes = plt.subplots(1, 1)
    explainers = [explainer_id for explainer_id in data.columns if explainer_id != 'x']
    for idx, explainer_id in enumerate(explainers):

        color = styles[explainer_id]['color']
        marker = styles[explainer_id]['marker']
        dashes = styles[explainer_id]['dashes']

        explainer_lab = explainer_id

        data_pruned = data[['x', explainer_id]].dropna()

        sns.lineplot(data=data_pruned, x='x', y=explainer_id, ax=axes, legend='brief',
                     label=explainer_lab, marker=marker, color=color, sort=False)
        axes.lines[idx].set_linestyle(dashes)

    axes.legend()

    axes.set_title(plot_title)
    size = 15
    axes.set_xlabel(x_label, fontsize=size)
    axes.set_ylabel(y_label, fontsize=size)

    x_ticks = data['x'].to_list()
    axes.set_xticks(x_ticks)

    axes.tick_params(axis='y', labelsize=size)
    axes.tick_params(axis='x', labelsize=size)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={"size": size})

    f.tight_layout()
    if not os.path.exists(plot_path.parents[0]):
        os.makedirs(plot_path.parents[0])

    f.savefig(plot_path)
    plt.close()


def get_scores_gen(experiments_data):

    experiments_grouped = experiments_data.groupby(["explainer", "params"])

    for index, explainer_data in experiments_grouped:

        importance_scores = None
        feature_names = None
        explainer_0 = None
        explain_params_0 = None
        idx = 0
        for exp_idx, experiment in explainer_data.iterrows():

            explain_params_noseed = json.loads(experiment["params"])
            if not pd.isna(experiment["random_state"]):
                random_state = int(experiment["random_state"])
                explain_params = {**explain_params_noseed, 'random_state': random_state}
            else:
                explain_params = explain_params_noseed
            explanation_path = get_explanation_path(experiment, explain_params)
            scores, feature_names = load_importance_scores(experiment["explainer"], explanation_path)
            if importance_scores is None:
                importance_scores = torch.zeros(explainer_data.shape[0], scores.shape[0])
                explainer_0 = experiment["explainer"]
                explain_params_0 = json.loads(experiment["params"])
            importance_scores[idx] = scores
            idx += 1

        yield explainer_0, explain_params_0, importance_scores.mean(dim=0), feature_names


def get_explainer_key(explainer_str, explain_params):
    res = None
    if explainer_str == "random_shap":
        res = "random"
    elif "shap" in explainer_str:
        res = F"{explain_params['type']}"
        if "nsamples" in explain_params.keys():
            res += F"-{explain_params['nsamples']}"
    elif explainer_str == "lime":
        res = F"{explainer_str}-{explain_params['num_samples']}"
    elif explainer_str == "hyperx":
        if explain_params['centrality'] == 'eigenvector':
            res = F"eig-p={explain_params['p']}-" \
                  F"s={explain_params['lowest_s_card']},{int(explain_params['highest_s_card'])-1}"
        else:
            res = F"deg-" \
                  F"s={explain_params['lowest_s_card']},{int(explain_params['highest_s_card'])-1}"

    if res is None:
        raise ValueError(F"Unknown explainer type: '{explainer_str}'")

    return res


def plot_greedy_all(n_highest):

    plots_dir_path = Path('plots_greedy')

    complexity_class = [0, 1, 2]
    cur_model = None
    cur_dataset = None

    instance_gen = get_instance_generator()
    for experiment in instance_gen:

        experiment_0 = experiment.iloc[0]

        if cur_dataset != experiment_0["test_x_path"]:
            path_to_test_x = experiment_0["test_x_path"]
            x_test = pd.read_csv(Path(path_to_test_x), header=0)
            cur_dataset = experiment_0["test_x_path"]

        plot_data_dir = get_instance_dir(experiment_0)
        plot_data_path = plots_dir_path / plot_data_dir / F"plot_data_nhighest={n_highest}.csv"
        # if os.path.exists(plot_data_path):
        #     im_data = pd.read_csv(plot_data_path, header=0)
        # else:
        if cur_model != experiment_0["model_path"]:
            model = load_model(path=Path(experiment_0["model_path"]),
                               type=experiment_0["model_type"])
            cur_model = experiment_0["model_path"]

        # compute_mean_scores
        mean_scores = {}
        for explainer, explain_params, scores, f_names in get_scores_gen(experiment):
            explainer_key = get_explainer_key(explainer, explain_params)
            # prune nans
            if np.isnan(scores).sum() == 0:
                mean_scores[explainer_key] = scores

        # Get the impact measure values
        instance_x = load_instance(x_test, experiment_0['instance_id'])
        baseline = load_baseline(x_test, experiment_0['baseline_num'], experiment_0['baseline_seed'])
        masked_im = get_masked_im(model, instance_x, baseline)
        im_data, f_full_val = get_im_data(instance_x, masked_im, mean_scores, x_test.shape[1], n_highest=n_highest)

        if not os.path.exists(plot_data_path.parents[0]):
            os.makedirs(plot_data_path.parents[0])
        im_data.to_csv(plot_data_path, header=True, index=False)

        for c in complexity_class:
            x_label = "x"
            y_label = "y"

            plot_dir = plots_dir_path / plot_data_dir

            if not os.path.exists(plot_dir.parents[0]):
                os.makedirs(plot_dir.parents[0])

            plot_path = plot_dir / F"c={c}_nhighest={n_highest}.png"
            nonx_cols = [col for col in im_data.columns if col != 'x']
            grouped_by_complexity = group_by_complexity(im_data[nonx_cols], c, x_test.shape[1])
            grouped_by_complexity['x'] = im_data['x']

            grouped_by_sim = group_similar_functions(grouped_by_complexity)
            grouped = grouped_by_sim
            if grouped.shape[0] > 0 and grouped.shape[1] > 0:

                eig_sim_cols = [col for col in grouped.columns if 'eig' in col]
                deg_sim_cols = [col for col in grouped.columns if 'deg' in col]

                eig_styles = get_styles_grouped(eig_sim_cols)
                deg_styles = get_styles_grouped(deg_sim_cols)

                cols = [col for col in grouped.columns if col != 'x']
                plot_functions(grouped, "", y_label, x_label, plot_path,
                               get_styles_grouped(cols, [eig_styles, deg_styles]))


if __name__ == '__main__':
    n_highest = 10  # number of features with the highest absolute value to keep
    plot_greedy_all(n_highest)
