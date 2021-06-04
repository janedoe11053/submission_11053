import os

import pandas as pd
import json
from pathlib import Path

from model import model_factory
from explainer import ShapleyPkgExplainer, LimePkgExplainer, HyperXExplainer
from explanation import ShapleyPkgExplanation, LimePkgExplanation, HyperXExplanation


def get_instance_dir(experiment):
    instance_id = experiment['instance_id']
    ds_name = experiment['dataset_name']
    return F'{ds_name}_instance={instance_id}'


def get_explanation_dir(experiment):
    path = Path(get_instance_dir(experiment), experiment['explainer'])
    return path


def get_explanation_path(experiment, explain_params):
    path = get_explanation_dir(experiment)
    explain_params_str = "_".join([F"{key}={val}" for key, val in explain_params.items()
                                   if key != "categorical_features"])
    fname = F"{explain_params_str}.csv"
    return Path('explanations' / path / fname)


def get_explainer(explainer_str):

    explainer = None
    if explainer_str == "shap":
        explainer = ShapleyPkgExplainer()
    if explainer_str == "lime":
        explainer = LimePkgExplainer()
    elif explainer_str == "hyperx":
        explainer = HyperXExplainer()

    if explainer is None:
        raise ValueError(F"Unknown explainer type: '{explainer_str}'")

    return explainer


def load_model(path, type):

    model = model_factory(type)
    if model is None:
        raise ValueError(F"Unknown model type: '{type}'")

    model.load(path)
    return model


def save_explanation(explainer, explanation_gen, explain_params, feature_names_dict, explanation_path):
    explanation = None
    if explainer == "shap":
        explanation = ShapleyPkgExplanation(explanation_gen, explain_params, feature_names_dict)
    elif explainer == "lime":
        explanation = LimePkgExplanation(explanation_gen, explain_params, feature_names_dict)
    elif explainer == "hyperx":
        explanation = HyperXExplanation(explanation_gen, explain_params, feature_names_dict)

    if explanation is None:
        raise ValueError(F"Unknown explainer type: '{explainer}'")

    if not os.path.exists(explanation_path.parents[0]):
        os.makedirs(explanation_path.parents[0])

    explanation.save(explanation_path)


def load_importance_scores(explainer_str, path):
    explanation = None
    if "shap" in explainer_str:
        explanation = ShapleyPkgExplanation()
    elif explainer_str == "lime":
        explanation = LimePkgExplanation()
    elif explainer_str == "hyperx":
        explanation = HyperXExplanation()
    if explanation is None:
        raise ValueError(F"Unknown explainer type: '{explainer_str}'")

    explanation.load(path)
    return explanation.importance_scores, explanation.feature_names


def load_instance(x_test, instance_id):
    return x_test.loc[instance_id].to_numpy().reshape(1, -1)


def load_baseline(x_test, num, seed):
    return x_test.sample(n=num, random_state=seed).to_numpy()


def run_all(experiments):
    cur_ds_path = None
    cur_model_path = None
    for index, experiment in experiments.iterrows():
        if cur_ds_path != experiment['test_x_path']:
            x_test = pd.read_csv(Path(experiment['test_x_path']), header=0)
            cur_ds_path = experiment['test_x_path']
            feature_names_dict = {x_test.columns.get_loc(feature): feature
                                  for feature in x_test.columns.tolist()}

        if cur_model_path != experiment['model_path']:
            model = load_model(Path(experiment['model_path']), experiment['model_type'])
            cur_model_path = experiment['model_path']
        instance = load_instance(x_test, experiment['instance_id'])
        baseline = load_baseline(x_test, experiment['baseline_num'], experiment['baseline_seed'])

        explain_params_noseed = json.loads(experiment["params"])
        if not pd.isna(experiment["random_state"]):
            random_state = int(experiment["random_state"])
            explain_params = {**explain_params_noseed, 'random_state': random_state}
        else:
            explain_params = explain_params_noseed
        explainer = get_explainer(experiment["explainer"])
        explanation_path = get_explanation_path(experiment, explain_params)
        explanation_generator = explainer.explain(explain_params, instance, baseline, model)
        save_explanation(experiment["explainer"], explanation_generator, explain_params, feature_names_dict,
                         explanation_path)


if __name__ == '__main__':
    experiments = pd.read_csv(Path('experiments.csv'), sep=';')
    run_all(experiments)
