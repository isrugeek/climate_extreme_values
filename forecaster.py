import pandas
import numpy as np
import constants
import torch
from train_main import create_dataset
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from torch.nn import functional as F


def LSTMGUForecaster(dataset):
    correct, total = 0, 0
    df = dataset
    model = torch.load("model/gu_lstm.pth")
    model.eval()
    batch_size = constants.lstm_param['batch_size']
    forecast_horizon = constants.lstm_param['forecast_horizon']
    look_back = constants.lstm_param['look_back']
    n_repeats = 1
    test_true_y = []
    test_pred_y = []
    for repeats in range(n_repeats):
        ep_loss = []
        preds = []
        for batch in create_dataset(df[df['Year'] >= 2018], look_back=look_back, forecast_horizon=forecast_horizon, batch_size=batch_size):
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break
            out = model.forward(batch[0].cuda().float())  # .cuda().float()
            ## ACC
            preds_a = F.log_softmax(out, dim=1).argmax(dim=1)
            total += batch[0].size(0)
            correct += (preds_a == batch[0].cuda().long()).sum().item()
            ## ACC
            loss = model.loss(out, batch[1].cuda().float())
            ep_loss.append(loss.item())
            if repeats == 0:
                test_true_y.append((batch[1]).detach().numpy().reshape(-1))
            preds.append((out.cpu()).detach().numpy().reshape(-1))
        print("{:0.4f}".format(100*np.mean(ep_loss)), end=", ")
        test_pred_y.append(preds)
    test_true_y = np.array(test_true_y)
    test_pred_y = np.array(test_pred_y)
    print(test_true_y.shape, test_pred_y[0].shape)
    acc = (correct / total)
    rms = sqrt(mean_squared_error(test_true_y, test_pred_y[0]))

    mean = np.mean(test_pred_y, axis=0).reshape(-1)
    std = np.std(test_pred_y, axis=0).reshape(-1)
    lower = np.percentile(test_pred_y, 5, axis=0).reshape(-1)
    upper = np.percentile(test_pred_y, 95, axis=0).reshape(-1)
    print("\n Shape of Prediction {} Shape of GT {} Mean Error {}".format(
        test_pred_y[0].shape, test_true_y.shape, np.mean(mean)))
    mae = mean_absolute_error(test_true_y, test_pred_y[0]).reshape(-1)
    print("\n MAE {} STD {}".format(mae, np.mean(std)))
    return ep_loss, mean, rms, lower, upper, mae, acc, test_true_y, test_pred_y





def LSTMForecaster (dataset):
    correct, total = 0, 0
    df = dataset
    model = torch.load("model/vanilla_lstm.pth")
    model.eval()
    batch_size = constants.lstm_param['batch_size']
    forecast_horizon = constants.lstm_param['forecast_horizon']
    look_back = constants.lstm_param['look_back']
    n_repeats = 1
    test_true_y = []
    test_pred_y = []
    for repeats in range(n_repeats):
        ep_loss = []
        preds = []
        for batch in create_dataset(df[df['Year'] >= 2018], look_back=look_back, forecast_horizon=forecast_horizon, batch_size=batch_size):
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break
            out = model.forward(batch[0].cuda().float())  # .cuda().float()
            ## ACC
            preds_a = F.log_softmax(out, dim=1).argmax(dim=1)
            total += batch[0].size(0)
            correct += (preds_a == batch[0].cuda().long()).sum().item()
            ## ACC
            loss = model.loss(out, batch[1].cuda().float())
            ep_loss.append(loss.item())
            if repeats == 0:
                test_true_y.append((batch[1]).detach().numpy().reshape(-1))
            preds.append((out.cpu()).detach().numpy().reshape(-1))
        print("{:0.4f}".format(100*np.mean(ep_loss)), end=", ")
        test_pred_y.append(preds)
    test_true_y = np.array(test_true_y)
    test_pred_y = np.array(test_pred_y)
    print (test_true_y.shape,test_pred_y[0].shape)
    acc = (correct / total)
    rms = sqrt(mean_squared_error(test_true_y, test_pred_y[0]))

    mean = np.mean(test_pred_y, axis=0).reshape(-1)
    std = np.std(test_pred_y, axis=0).reshape(-1)
    lower = np.percentile(test_pred_y, 5, axis=0).reshape(-1)
    upper = np.percentile(test_pred_y, 95, axis=0).reshape(-1)
    print("\n Shape of Prediction {} Shape of GT {} Mean Error {}".format(
        test_pred_y[0].shape, test_true_y.shape, np.mean(mean)))
    mae = mean_absolute_error(test_true_y, test_pred_y[0]).reshape(-1)
    print("\n MAE {} STD {}".format(mae, np.mean(std)))
    return ep_loss, mean, rms, lower, upper, mae, acc, test_true_y, test_pred_y

def ENCDECForecaster(dataset):
    correct, total = 0, 0
    df=dataset
    model = torch.load("model/auto_lstm.pth")
    model.eval()
    batch_size = constants.enc_dec_param['batch_size']
    forecast_horizon = constants.enc_dec_param['forecast_horizon']
    look_back = constants.enc_dec_param['look_back']

    n_repeats = 1
    test_true_y = []
    test_pred_y = []
    for repeats in range(n_repeats):
        ep_loss = []
        preds = []
        for batch in create_dataset(df[df['Year'] >= 2018], look_back=look_back, forecast_horizon=forecast_horizon, batch_size=batch_size):
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break
            out = model.forward(batch[0].float(), batch_size=batch_size)
            ## ACC
            preds_a = F.log_softmax(out, dim=1).argmax(dim=1)
            total += batch[1].size(0)
            correct += (preds_a == batch[1].long()).sum().item()
            loss = model.loss(out, batch[1].float())
            ep_loss.append(loss.item())
            if repeats == 0:
                test_true_y.append(
                    (batch[1]).detach().numpy().reshape(-1))
            preds.append((out).detach().numpy().reshape(-1))
        print("{:0.4f}".format(100*np.mean(ep_loss)), end=", ")
        test_pred_y.append(preds)
    test_true_y = np.array(test_true_y)
    test_pred_y = np.array(test_pred_y)
    acc = (correct / total)

    mean = np.mean(test_pred_y, axis=0).reshape(-1)
    std = np.std(test_pred_y, axis=0).reshape(-1)
    lower = np.percentile(test_pred_y, 5, axis=0).reshape(-1)
    upper = np.percentile(test_pred_y, 95, axis=0).reshape(-1)
    print("\n Shape of Prediction {} Shape of GT {} Mean Error {}".format(
        test_pred_y[0].shape, test_true_y.shape, np.mean(mean)))
    mae = mean_absolute_error(test_true_y, test_pred_y[0]).reshape(-1)
    rms = sqrt(mean_squared_error(test_true_y, test_pred_y[0]))
    print("\n MAE {} STD {}".format(mae, np.mean(std)))
    return ep_loss, mean, rms, lower, upper, mae, acc, test_true_y, test_pred_y
