import pandas
import numpy as np
import constants
import torch
from train_main import create_dataset
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from torch.nn import functional as F


def generic_test(df, param, model_path):
    df = df.loc[df.index >= "2018"]
    model = torch.load(model_path)
    model.eval()
    n_repeats = 1
    test_true_y = []
    test_pred_y = []
    for repeats in range(n_repeats):
        cur_data = create_dataset(df, batch_size=param["batch_size"],
                                      forecast_horizon=param["forecast_horizon"],
                                      look_back=param["look_back"]
                                      )
        ep_loss = []
        preds = []
        for batch in cur_data:
            try:
                batch = [torch.Tensor(x) for x in batch]
            except:
                break
            out = model.forward(batch[0].cuda().float())
            loss = model.loss(out, batch[1].cuda().float())
            ep_loss.append(loss.item())
            if repeats == 0:
                test_true_y.append((batch[1].detach().numpy().reshape(-1)))
            preds.append((out.cpu()).detach().numpy().reshape(-1))
        # print("{:0.4f}".format(np.mean(ep_loss)), end=", ")
        test_pred_y.append(preds)
    test_true_y = np.array(test_true_y)
    test_pred_y = np.array(test_pred_y)
    rms = sqrt(mean_squared_error(test_true_y, test_pred_y[0]))
    mean = np.mean(test_pred_y, axis=0).reshape(-1)
    mae = mean_absolute_error(test_true_y, test_pred_y[0]).reshape(-1)

    # return (ep_loss, mean)

    return ( ep_loss, mean, rms, mae, test_true_y, test_pred_y)
     




def LSTMGUForecaster(dataset):
    df = dataset
    param = constants.gu_lstm_param
    ep_loss, mean, rms, mae, test_true_y, test_pred_y = generic_test(
        df, param, "model/gu_lstm.pth")
    return (ep_loss, mean, rms, mae, test_true_y, test_pred_y)


def LSTMForecaster (dataset):
    df = dataset
    param = constants.lstm_param
    ep_loss, mean, rms, mae, test_true_y, test_pred_y = generic_test(
        df, param, "model/vanilla_lstm.pth")
    return (ep_loss, mean, rms, mae, test_true_y, test_pred_y)
    



def ENCDECForecaster(dataset):
    df = dataset
    param = constants.enc_dec_param
    ep_loss, mean, rms, mae, test_true_y, test_pred_y = generic_test(
        df,param,"model/auto_lstm.pth")
    return (ep_loss, mean, rms, mae, test_true_y, test_pred_y)


    
 
