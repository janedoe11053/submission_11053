import itertools
import math
import os
import sys
import traceback
import random

from pathlib import Path
import pandas as pd
import json


def get_n_samples_budget(n):
    scard_dists = [0, 1, 2]
    budget = [math.comb(n, dist) * n for dist in scard_dists]
    return budget


def populate_shap_params(dataset_name, n_features):
    params = []
    random_state = 3129 if dataset_name == 'rossmann' else 2510
    params.append({'type': 'random_shap', 'random_state': random_state})

    n_samples = get_n_samples_budget(n_features)
    n_perms = [math.ceil(c / n_features) for c in n_samples]

    if dataset_name == 'rossmann':
        kernel_seeds = [[1078, 2980, 4264], [1078, 2980, 4264], [1078, 2980, 4264]]
        perm_seeds = [[4246, 4397, 4565], [4246, 4397, 4565], [4246, 4397, 4565]]
        sampling_seeds = [[2323, 2595, 4481], [2323, 2595, 4481], [2323, 2595, 4481]]
    else:
        kernel_seeds = [[1411, 2498, 3100], [1411, 2823, 3100], [2823, 3100, 1411]]
        perm_seeds = [[3932, 4305, 3956], [3932, 4305, 3956], [3932, 4305, 3956]]
        sampling_seeds = [[3932, 4305, 3956], [2498, 4309, 4918], [4918, 2498, 4309]]

    budgets = [0, 1, 2]
    for c in budgets:
        params += [{'type': 'kernel_shap', 'nsamples': n_samples[c], 'random_state': r}
                      for r in kernel_seeds[c]]
        params += [{'type': 'perm_shap', 'nsamples': n_perms[c], 'random_state': r}
                      for r in perm_seeds[c]]
        params += [{'type': 'sampling_shap', 'nsamples': n_samples[c], 'random_state': r}
                      for r in sampling_seeds[c]]
    return params


def populate_lime_params(dataset, n_features, cat_features):
    n_samples = get_n_samples_budget(n_features)

    if dataset == 'rossmann':
        seeds = [1411, 2823, 3100]
    else:
        seeds = [3914, 4050, 4578]

    return [{'categorical_features': cat_features,
             'random_state': r, 'num_samples': n}
            for r, n in itertools.product(seeds, n_samples)]


def populate_hyperx_params(n_features):
    s_cards = [(0, 1), (0, 2), (0, 3), (n_features - 1, n_features), (n_features - 2, n_features),
               (n_features - 3, n_features)]
    p_vals = [1, 2, 10]
    explain_params = []
    for s_card, p in itertools.product(s_cards, p_vals):
        param_set = {"lowest_s_card": s_card[0], "highest_s_card": s_card[1], "p": p,
                     "centrality": "eigenvector", "p_normalization": 1, "omega": "uniform"}
        explain_params.append(param_set)

    for s_card in s_cards:
        param_set = {"lowest_s_card": s_card[0],
                     "highest_s_card": s_card[1],
                     "centrality": "degree", "omega": "uniform", "p_normalization": 1}
        explain_params.append(param_set)

    return explain_params


def get_feature_subsets(n_features, dataset_name):
    return [[i for i in range(n_features)]]


def get_cat_features(name):
    if name == 'rossmann':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif name == 'census':
        return [1, 2, 4, 5, 6, 7, 8, 12]
    else:
        raise ValueError(F"Unknown dataset: {name}")


if __name__ == '__main__':

    instances = [{'test_x_path': Path('datasets', 'rossmann', 'x_test.csv'),
                  'dataset_name': 'rossmann',
                  'model_path': Path('datasets', 'rossmann', 'xgb.onnx'),
                  'model_type': 'XGBoostRegression',
                  'instance_id': 39494,
                  'baseline_num': 1000, 'baseline_seed': 2595},
                 {'test_x_path': Path('datasets', 'census', 'x_test.csv'),
                  'dataset_name': 'census',
                  'model_path': Path('datasets', 'census', 'model.pt'),
                  'model_type': 'TorchBlackBox',
                  'instance_id': 1072, 'baseline_num': 1000,
                  'baseline_seed': 3129}
                 ]

    experiments_cols = ["test_x_path", "dataset_name", "model_path", "model_type", "instance_id", "baseline_num", "baseline_seed",
                        "explainer", "params", "random_state"]
    experiments_df = pd.DataFrame(columns=experiments_cols)
    explainers = ['shap', 'lime', 'hyperx']
    for instance in instances:
        test_x = pd.read_csv(Path(instance['test_x_path']), header=0)
        for explainer in explainers:
            if explainer == "shap":
                explain_params = populate_shap_params(instance['dataset_name'], test_x.shape[1])
            elif explainer == "hyperx":
                explain_params = populate_hyperx_params(test_x.shape[1])
            elif explainer == "lime":
                cat_features = get_cat_features(instance['dataset_name'])
                explain_params = populate_lime_params(instance['dataset_name'], test_x.shape[1], cat_features)

            for params in explain_params:
                experiment = pd.Series(index=experiments_cols, data=instance, dtype="object")
                experiment['explainer'] = explainer
                params_noseed = {k: v for k, v in params.items() if k != "random_state"}
                experiment['params'] = json.dumps(params_noseed)
                if 'random_state' in params.keys():
                    experiment['random_state'] = params['random_state']
                experiments_df = experiments_df.append(experiment, ignore_index=True)

    experiments_df.to_csv(Path('experiments.csv'), sep=';', index=False,
                          header=True)
