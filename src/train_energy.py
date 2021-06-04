import json
from pathlib import Path

import click
import pandas as pd
import xgboost as xgb
from hummingbird.ml import convert
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


@click.command()
@click.option('-i', '--data-dir', 'data_dir', required=True)
@click.option('-p', '--train-params-file', 't_param', required=True)
def train(data_dir, t_param):
    input_path = Path(data_dir)
    try:
        x_test = pd.read_csv(input_path / 'x_test.csv').to_numpy()
        y_test = pd.read_csv(input_path / 'y_test.csv').to_numpy()
        x_train = pd.read_csv(input_path / 'x_train.csv').to_numpy()
        y_train = pd.read_csv(input_path / 'y_train.csv').to_numpy()
    except FileNotFoundError as nf_error:
        print(F"{Path(nf_error.filename).name} is expected at {input_path}")
        raise nf_error
    with open(t_param, 'r', encoding='utf-8') as f:
        tr_params = json.load(f)

    xgb_model = xgb.XGBRegressor(**tr_params)
    xgb_model.fit(x_train, y_train)
    y_pred_xgb = xgb_model.predict(x_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse = mean_squared_error(y_test, y_pred_xgb, squared=False)

    hb_model = convert(xgb_model, 'torch', x_train[0:1])
    hb_model.save(str(input_path / "model.pt"))

    print(f"MAE: {mae_xgb} RMSE: {rmse}")


if __name__ == "__main__":
    train()
