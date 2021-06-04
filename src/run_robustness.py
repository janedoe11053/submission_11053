import itertools
import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from model import TorchHumXGBModel
from explainer_wrapper import HyperxExplainerWrapper, ShapExplainerWrapper, LimeExplainerWrapper

titanic_categorical = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title", "IsAlone"]
energy_categorical = ["is_weekday"]
census_categorical = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex",
                      "native_country"]


@click.command()
@click.option('-d', '--data-path', 'data_path', required=True)
@click.option('-m', '--model-path', 'model_path', required=True)
@click.option('-i', '--iter-num', 'iter_num', required=True, default=30)
@click.option('-n', '--num-points', 'num_points', required=True, default=50)
@click.option('-b', '--num-baseline', 'num_baseline', required=True, default=100)
@click.option('-o', '--output-path', 'output_path', required=True)
def run_continuous_experiments(data_path, model_path, iter_num, num_points, num_baseline, output_path):
    data_path = Path(data_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    x_test = pd.read_csv(data_path / "x_test.csv", header=0, index_col=None)
    y_test = pd.read_csv(data_path / "y_test.csv", header=0, index_col=None)

    model = TorchHumXGBModel()
    model.load(model_path)

    np.random.rand(84)
    random.seed(84)
    seed = random.randint(1, 11111)

    behaviours = np.random.choice(x_test.shape[0], num_points)
    features = x_test.columns.tolist()
    classes = None
    results_sl = pd.DataFrame(columns=["b_id", "expl", "L", "c", "seed"])
    results_op = pd.DataFrame(columns=["b_id", "expl", "L", "p", "cent", "s", "seed"])

    N = len(features)
    for idx in behaviours:

        x_baseline = x_test.sample(n=num_baseline, random_state=seed)

        x = x_test.iloc[idx]
        y = y_test.iloc[idx]

        x_b = x_baseline.copy(deep=True)
        x_b[energy_categorical] = x[energy_categorical]

        x_back = x_test.to_numpy(dtype=np.float32)
        y_back = y_test.to_numpy(dtype=np.float32)

        x_b = x_b.to_numpy(dtype=np.float32)
        y_b = model(x_b)

        x = np.expand_dims(x.to_numpy(dtype=np.float32), axis=0)
        y = np.expand_dims(y.to_numpy(dtype=np.float32), axis=0)

        for c in [N, N * N, 2 * N, 2 * N * N]:
            model.out_type = np.ndarray
            shap_explainer = ShapExplainerWrapper(model,
                                                  model_type='regression',
                                                  multiclass=False,
                                                  feature_names=features,
                                                  class_names=classes,
                                                  background_data=(x_back, y_back),
                                                  baseline=(x_b, y_b),
                                                  seed=seed,
                                                  nsamples=c)

            lip = shap_explainer.local_lipschitz_estimate(x, bound_type='box_std',
                                                          optim="gp",
                                                          eps=0.01,
                                                          n_calls=iter_num,
                                                          verbose=2)
            results_sl.loc[len(results_sl)] = [idx, "kernel_shap", lip, c, seed]

            lime_explainer = LimeExplainerWrapper(model,
                                                  model_type='regression',
                                                  multiclass=False,
                                                  feature_names=features,
                                                  class_names=classes,
                                                  background_data=(x_back, y_back),
                                                  baseline=(x_b, y_b),
                                                  seed=seed,
                                                  feature_selection='none',  # so that wwe get attribs for all features!
                                                  num_samples=c,
                                                  verbose=False)
            lip = lime_explainer.local_lipschitz_estimate(x, bound_type='box_std',
                                                          optim="gp",
                                                          eps=0.01,
                                                          n_calls=iter_num,
                                                          verbose=2)
            results_sl.loc[len(results_sl)] = [idx, "lime", lip, c, seed]
            results_sl.to_csv(output_path / "results_sl.csv")

        subs = [(0, 1), (N - 1, N), (0, 2), (N - 2, N)]
        ps = [10, 2]
        cs = ["degree", "eigenvector"]
        f_subset = list(range(0, len(features)))

        model.out_type = torch.Tensor
        for p, cent, s in itertools.product(ps, cs, subs):
            if p != 10 and cent == "degree":
                continue
            explainer_params = {
                "feature_subset": f_subset,
                "lowest_s_card": s[0],
                "highest_s_card": s[1],
                "p": p,
                "centrality": cent,
                "p_normalization": 1,
                "omega": "uniform"
            }
            hyperx = HyperxExplainerWrapper(model, model_type='regression', multiclass=False,
                                            feature_names=features, class_names=classes,
                                            background_data=(x_back, y_back),
                                            baseline=(x_b, y_b),
                                            seed=seed,
                                            explainer_params=explainer_params)

            lip = hyperx.local_lipschitz_estimate(x, y, bound_type='box_std',
                                                  optim="gp",
                                                  eps=0.01,
                                                  n_calls=iter_num,
                                                  verbose=2)
            results_op.loc[len(results_op)] = [idx, "hyperx", lip, p, cent, s, seed]
            results_op.to_csv(output_path / "results_hx.csv")


if __name__ == "__main__":
    run_continuous_experiments()
