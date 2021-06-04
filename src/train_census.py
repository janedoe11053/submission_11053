import json
from pathlib import Path

import click
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nn_model import NetModel

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


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
    torch.manual_seed(tr_params['seed'])

    model = NetModel(x_train.shape[1], tr_params['neurons'])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tr_params["learning_rate"])
    train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_dl = DataLoader(train_data, batch_size=tr_params["batch_size"], shuffle=True)
    epochs = tr_params["epochs"]
    for e in range(epochs):
        for i, (x_batch, y_batch) in enumerate(train_dl):
            optimizer.zero_grad()
            x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    x_test = torch.FloatTensor(x_test).to(dev)
    y_pred = model(x_test).cpu().detach().numpy()
    y_pred = y_pred > 0.5
    accuracy = (y_test == y_pred).sum() / y_test.size
    print(F"Test accuracy: {accuracy}")
    torch.save(model, input_path / "model.pt")


if __name__ == "__main__":
    train()
