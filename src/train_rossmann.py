import json
import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from onnxconverter_common import FloatTensorType
from onnxmltools import convert_xgboost
import click


def rmspe(predt: np.ndarray, dm: xgb.DMatrix) -> Tuple[str, float]:
    y_true = np.expm1(dm.get_label())
    y_pred = np.expm1(predt)
    return rmspe_true(y_pred, y_true)


def rmspe_true(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[str, float]:
    factor = np.zeros(y_true.shape, dtype=float)
    indices = y_true != 0
    factor[indices] = 1.0 / (y_true[indices])
    diff = y_true - y_pred
    diff_percentage = diff * factor
    diff_percentage_squared = diff_percentage ** 2
    rmspe_err = np.sqrt(np.mean(diff_percentage_squared))
    return 'rmspe', rmspe_err


@click.command()
@click.option('-i', '--data-dir', 'data_dir', required=True)
@click.option('-p', '--train-params-file', 'params_path', required=True)
def train(data_dir, params_path):

    data_root = Path(data_dir)
    res_root = data_root
    x_train = pd.read_csv(data_root / 'x_train.csv')
    y_train = pd.read_csv(data_root / 'y_train.csv')
    x_test = pd.read_csv(data_root / 'x_test.csv')
    y_test = pd.read_csv(data_root / 'y_test.csv')

    dm_train_log = xgb.DMatrix(x_train.values, y_train.values.reshape((x_train.shape[0],)))
    dm_test_log = xgb.DMatrix(x_test.values, y_test.values.reshape((x_test.shape[0],)))

    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    xgb_params = {k: v for k, v in params.items() if k != 'num_boost_round'}
    xgb_params['disable_default_eval_metric'] = 1

    results: Dict[str, Dict[str, List[float]]] = {}

    xgb_other = {'dtrain': dm_train_log, 'num_boost_round': params['num_boost_round'],
                 'feval': rmspe, 'evals': [(dm_train_log, 'train'), (dm_test_log, 'test')],
                 'evals_result': results
                 }

    regressor = xgb.train(xgb_params, **xgb_other)

    if not os.path.exists(res_root):
        os.mkdir(res_root)

    model_onnx = convert_xgboost(regressor, 'xgb',
                                 [('input', FloatTensorType([None, x_train.shape[1]]))])
    with open(res_root / 'xgb.onnx', "wb") as f:
        f.write(model_onnx.SerializeToString())


if __name__ == "__main__":
    train()
