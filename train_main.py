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
            #x = dataset.loc[["temp" in c for c in dataset.columns]].values[past:cur, :]
            x = dataset[['max_temperature', 'avg_hourly_temperature',
                         'avg_temperature','min_temperature']].values[past:cur, :]
            y = np.array([dataset['max_temperature'].values[cur:future].max()])
            batches["x"].append(np.array(x).reshape(look_back, -1))
            batches["y"].append(np.array(y))
            batches["z"].append(np.array(x[0, 0]))
            

        for var, value in batches.items():
            batches[var] = np.array(value)

        batches["x"][:, :, 0] -= batches["z"].reshape(-1, 1)
        batches["y"] -= batches["z"].reshape(-1, 1)
        yield batches["x"], batches["y"], batches["z"]
        batches = {"x": [], "y": [], "z": []}


def generic_train(model, df, param, loss_path, model_path):
    #df = df[df["Year"] < 2018]
    df = df.loc[df.index < "2018"]
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
            print("[{}{}] Epoch {}: loss={:0.4f}".format("-" * ticks, (" " * (20 - ticks)), epoch, np.mean(ep_loss)), end="\r")
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
    param = constants.gu_lstm_param
    model = models.GULSTM(param["input_dim"], param["hidden_dim"], param["layer_dim"], param["output_dim"])
    generic_train(model,df,param,"log/gu_lstm.txt", "gu_lstm.pth")



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
