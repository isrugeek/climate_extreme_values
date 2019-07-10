#!/usr/bin/env python
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('agg')
from torch import nn
from utils import device
import argparse
import constants
import json
import mkdir
import models
import numpy as np
import os
import pandas as pd
import torch
import utils
import warnings
warnings.filterwarnings("ignore")


def pre_process (df, save_plots = True):
    df = df.loc[df.index > "1954"]
    df.index = pd.to_datetime(df.index)

    null_columns = df.columns[df.isnull().any()]
    for col_name in null_columns:
        null_count = df[col_name].isnull().sum()

        if null_count < 200:
            utils.fill_missing_one_day(df[[col_name]].values)
        elif null_count > len(df) * 0.8:
            df = df.drop([col_name], axis=1)
        elif null_count > 200:
            df[[col_name]] = df[[col_name]].fillna(np.mean(df[[col_name]]))

    plot_raw = df.loc[df.index > "2018", ["temp" in c for c in df.columns]]
    ax = plot_raw.plot(title='Full Time Series Data',fontsize=13, figsize = (16,3.5))
    ax.set_ylabel('Value',fontsize=13)
    ax.set_xlabel('Date/Time',fontsize=13)

    print("Shape of data {} Missing values {}".format(
        df.shape, df.isnull().sum().sum()))

    print(df.columns[df.isnull().any()])
    return df

def create_dataset(dataset, look_back=1, forecast_horizon=1, batch_size=1):
    batches = {"x": [], "y": [], "z": []}

    for i in range(0, len(dataset)-look_back-forecast_horizon-batch_size+1, batch_size):
        for n in range(batch_size):

            past, cur, future = i + n, i + n + look_back, i + n + look_back + forecast_horizon
            x = dataset[["temp" in c for c in df.columns]].values[past:cur, :]
            y = np.array([dataset['max_temperature'].values[cur:future].max()])
            batches["x"].append(np.array(x).reshape(look_back, -1))
            batch["y"].append(np.array(y))
            batch["z"].append(np.array(x[0, 0]))

        for var, value in batches.items():
            batches[var] = np.array(value)

        batches["x"][:, :, 0] -= batches["z"].reshape(-1, 1)
        batch["y"] -= batch["z"].reshape(-1, 1)
        yield batches["x"], batches["y"], batches["z"]
    batches = {"x": [], "y": [], "z": []}


def generic_train(model, df, param, loss_path, model_path):
    df = df[df["Year"] < 2018]
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=param["lr"])

    model.train()
    for epoch in range(1, param["n_epochs"] + 1):
        ep_loss = []

        cur_data = create_dataset(
            df,
            look_back=param["look_back"],
            forecast_horizon=param["forecast_horizon"],
            batch_size=param["batch_size"]
        )

        for i, batch in enumerate(cur_data):
            ticks = 20 * i // len(df) // param["batch_size"]
            print("[{}{}] Epoch {}: loss={:0.4f}".format("-" * ix, " " * 20 - ticks, epoch, np.mean(ep_loss), end="\r"))
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break

            x_batch = batch[0].cuda().float()
            y_batch = batch[1].cuda().float()
            out = model.forward(x_batch)
            loss = model.loss(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        print()

    outLoss = open(loss_path, "w")
    for line in ep_loss:
        outLoss.write("{}\n".format(str(line)))
    outLoss.close()
    utils.model_saver(model_path, model)


def LSTMTrainer (df):
    param = constants.lstm_param
    model = models.LSTM(param["input_dim"], param["hidden_dim"], param["layer_dim"], param["output_dim"])
    generic_train(model, df, param, "log/lstm.txt", "vanilla_lstm.pth")

def ENCDECTrainer (df):
    param = constants.enc_dec_param
    model = models.ENDELSTM(dict(features=param['n_features'], output_dim = param["output_dim"]))
    generic_train(model, df, param, "log/enddeclstm.txt", "auto_lstm.pth")


def GULSTMTrainer (dataset):
    n_epochs = constants.gu_lstm_param['n_epochs']
    input_dim = constants.gu_lstm_param['input_dim']
    hidden_dim = constants.gu_lstm_param['hidden_dim']
    layer_dim = constants.gu_lstm_param['layer_dim']
    output_dim = constants.gu_lstm_param['output_dim']
    forecast_horizon = constants.gu_lstm_param['forecast_horizon']
    batch_size = constants.gu_lstm_param['batch_size']
    look_back = constants.gu_lstm_param['look_back']
    model = models.GULSTM(input_dim, hidden_dim, layer_dim, output_dim)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Print model's state_dict
    print("Model's State Dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's State Dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    model.train()
    train_true_y = []
    train_pred_y = []
    for epoch in range(1, n_epochs + 1):
        ep_loss = []
        for i, batch in enumerate(create_dataset(df[df['Year'] < 2018], look_back=look_back, forecast_horizon=forecast_horizon, batch_size=batch_size)):
            print("[{}{}] Epoch {}: loss={:0.4f}".format("-"*(20*i//(len(df[df['Year'] < 2018])//batch_size)),
                                                         " "*(20-(20*i//(len(df[df['Year'] < 2018])//batch_size))), epoch, np.mean(ep_loss)), end="\r")
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break

            x_batch = batch[0].cuda().float()
            y_batch = batch[1].cuda().float()
            out = model.forward(x_batch)
            loss = model.loss(out, y_batch)
            #GU playing ground
            x_prev,x_t = batch[0].cuda().float(), batch[1].cuda().float()
            mu_hat = model.forward(x_prev)

            # negative l1
            l1 = x_t - mu_hat
            new_loss = l1 + torch.exp(-l1)
            new_loss.backward()
            optimizer.step()
            # print ("Loss components {} | {} | {}".format(l1,torch.exp(-l1)),new_loss)
            #  END
            if epoch == n_epochs - 1:
                train_true_y.append((batch[0]).detach().numpy().reshape(-1))
                train_pred_y.append((out.cpu()).detach().numpy().reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print ("Loss components {} | {} | {} | {}".format(l1,torch.exp(-l1)),new_loss,loss)
            ep_loss.append(loss.item())
        print()
    outLoss = open("log/gulstm.txt", "w")
    for line in ep_loss:
        outLoss.write(str(line))
        outLoss.write("\n")
    outLoss.close()
    utils.model_saver("gu_lstm.pth", "GULSTM", model)


if __name__ == "__main__":
    mkdir.mkdir()
    print ("Using device {} for training".format(device))
    save_plots = True
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument("train_mode")
    args = parser.parse_args()

    ### READ DATASET ###
    df = pd.read_csv('data/weatherstats_montreal_daily.csv',
                     index_col=0, nrows=100 if debug else None)
    df = pre_process(df)
    if args.train_mode == "LSTM":
        print (constants.lstm_param)
        LSTMTrainer(df)
    if args.train_mode == "ENCDEC":
        print (constants.enc_dec_param)
        ENCDECTrainer(df)
    if args.train_mode == "GULSTM":
        print (constants.gu_lstm_param)
        GULSTMTrainer(df)
