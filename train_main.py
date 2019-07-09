from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('agg')
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import json
import os
import utils
from utils import device
import models
import constants
import mkdir
import warnings
warnings.filterwarnings("ignore")


def pre_process (dataset, save_plots = True):
    save_plots = save_plots
    df = dataset 
    df = df.drop(['max_humidex', 'min_windchill', 'sunrise',
                  'sunset', 'sunlight', 'sunrise_f', 'sunset_f'], axis=1)
    non_null_columns = [
        col for col in df.columns if df.loc[:, col].notna().any()]
    df[non_null_columns]
    null_columns = df.columns[df.isnull().any()]

    df = (df[df['Year'] > 1954])
    #utils.fill_missing_one_day(df.values)
    # df = df.drop(['max_humidex', 'min_windchill'], axis=1)
    for col_name in null_columns:
        if (df[col_name].isnull().sum() < 200):
            #print ("Filling {}'s null data by previous day.".format(col_name))
            utils.fill_missing_one_day(df[[col_name]].values)
            null_columns = df.columns[df.isnull().any()]
        if (df[col_name].isnull().sum() > (int(int(df.shape[0])*0.8))):
            #print ("Dropping {}.".format(col_name))
            df = df.drop([col_name], axis=1)
            null_columns=df.columns[df.isnull().any()]
    for col_name in null_columns:
        if (df[col_name].isnull().sum() > 200):
            #print ("Filling {}'s null data by Mean.".format(col_name))
            df[[col_name]] = df[[col_name]].fillna(np.mean(df[[col_name]]))
            null_columns=df.columns[df.isnull().any()]

    # df['year'] = pd.DatetimeIndex(df['date']).year
    # df = df.set_index('year')
    year = pd.DatetimeIndex(df['date']).year
    min_year = year.min()
    max_year = year.max()
    # df = df.drop(['Year', 'date'], axis=1)
    print("The full dataset contains from {} - {} but we choose from {}".format(
        min_year, max_year, df.index.min()))
    plot_raw = df.copy()
    plot_raw = df.drop(
        ['max_visibility', 'min_visibility', 'avg_visibility', 'avg_hourly_visibility','Year'] , axis=1)
    plot_raw = plot_raw.sort_index(ascending=True)  # .tail(365)
    ax = plot_raw[['max_temperature', 'avg_hourly_temperature','avg_temperature','min_temperature']].plot(title='Full Time Series Data',fontsize=13, figsize = (16,3.5))
    ax.set_ylabel('Value',fontsize=13)
    ax.set_xlabel('Date/Time',fontsize=13)
    # utils.save_or_show_plot("raw_data.png",save_plots)

    print("Shape of data {} Missing values {}".format(
        df.shape, df.isnull().sum().sum()))

    # if debug:
    #     df = df.iloc[:, 0:8]

    print(df.columns[df.isnull().any()])
    return df

def create_dataset(dataset, look_back=1, forecast_horizon=1, batch_size=1):
    #dataset = dataset.drop(['Year', 'date', 'cooldegdays', 'snow'], axis=1)
    batch_x, batch_y, batch_z = [], [], []

    for i in range(0, len(dataset)-look_back-forecast_horizon-batch_size+1, batch_size):
        for n in range(batch_size):
            
            # print (dataset.head())
            #x = dataset.loc[:, dataset.columns != 'year'].values[i+n:(i + n + look_back), :]
            x = dataset[['max_temperature', 'avg_hourly_temperature',
                         'avg_temperature','min_temperature']].values[i+n:(i + n + look_back), :]
            n_features = x.shape[1]
            # print ("no of fratures {}".format(n_features))
            constants.lstm_param['input_dim'] = n_features
            constants.enc_dec_param['input_dim'] = n_features
            constants.gu_lstm_param['input_dim'] = n_features
            
            offset = x[0, 0]
            y = np.array([dataset['max_temperature'].values[i + n +
                                                           look_back:i + n + look_back + forecast_horizon].max()])

            batch_x.append(np.array(x).reshape(look_back, -1))
            batch_y.append(np.array(y))
            batch_z.append(np.array(offset))
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_z = np.array(batch_z)
        batch_x[:, :, 0] -= batch_z.reshape(-1, 1)
        batch_y -= batch_z.reshape(-1, 1)
        yield batch_x, batch_y, batch_z
        batch_x, batch_y, batch_z = [], [], []

def LSTMTrainer (dataset):
    df = dataset
    batch_size = constants.lstm_param['batch_size']
    forecast_horizon = constants.lstm_param['forecast_horizon']
    look_back = constants.lstm_param['look_back']
    input_dim = constants.lstm_param['input_dim']
    hidden_dim = constants.lstm_param['hidden_dim']
    layer_dim = constants.lstm_param['layer_dim']
    output_dim = constants.lstm_param['output_dim']
    seq_dim = constants.lstm_param['seq_dim']
    lr = constants.lstm_param['lr']
    n_epochs = constants.lstm_param['n_epochs']
    model = models.LSTM(input_dim, hidden_dim, layer_dim, output_dim)
    model = model.cuda()
    # opt = torch.optim.RMSprop(model.parameters(), lr=lr
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('Start Vanilla LSTM model training')
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
    #         opt.step()
            if epoch == n_epochs - 1:
                train_true_y.append((batch[0]).detach().numpy().reshape(-1))
                train_pred_y.append((out.cpu()).detach().numpy().reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        print()
    outLoss = open("log/lstm.txt", "w")
    for line in ep_loss:
        outLoss.write(str(line))
        outLoss.write("\n")
    outLoss.close()
    utils.model_saver("vanilla_lstm.pth","LSTM",model)
    #torch.save(model, "model/vanilla_lstm.pth")
def ENCDECTrainer (dataset):
    batch_size = constants.enc_dec_param['batch_size']
    forecast_horizon = constants.enc_dec_param['forecast_horizon']
    look_back = constants.enc_dec_param['look_back']
    lr = constants.enc_dec_param['lr']
    n_epochs = constants.enc_dec_param['n_epochs']
    output_dim = constants.enc_dec_param['output_dim']
    model = models.ENDELSTM(dict(features=constants.enc_dec_param['n_features'], output_dim = output_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Print model's state_dict
    print("Model's State Dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # Print optimizer's state_dict
    print("Optimizer's State Dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    print("Start ENCDEC LSTM model training")
    model.train()
    train_true_y = []
    train_pred_y = []
    for epoch in range(1, n_epochs+1):
        ep_loss = []
        for i, batch in enumerate(create_dataset(df[df['Year'] < 2018], look_back=look_back, forecast_horizon=forecast_horizon, batch_size=batch_size)):

            print("[{}{}] Epoch {}: loss={:0.4f}".format("-"*(20*i//(len(df[df['Year'] < 2018])//batch_size)),
                                                        " "*(20-(20*i//(len(df[df['Year'] < 2018])//batch_size))), epoch, np.mean(ep_loss)), end="\r")
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break
            out = model.forward(batch[0].float(), batch_size)
            loss = model.loss(out, batch[1].float())
            if epoch == n_epochs - 1:
                train_true_y.append(
                    (batch[1] + batch[2]).detach().numpy().reshape(-1))
                train_pred_y.append((out + batch[2]).detach().numpy().reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
      
        print()
    # torch.save(model, "model/auto_lstm.pth")
    outLoss = open("log/enddeclstm.txt", "w")
    for line in ep_loss:
        outLoss.write(str(line))
        outLoss.write("\n")
    outLoss.close()
    utils.model_saver("auto_lstm.pth", "ENCDEC", model)


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
    print ("Using dvice {} for training".format(device))
    save_plots = True
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument("train_mode")
    args = parser.parse_args()
    #train_mode = "LSTM"
    ### READ DATASET ###
    df = pd.read_csv('data/weatherstats_montreal_daily_inter.csv',
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
