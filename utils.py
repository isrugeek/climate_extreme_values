import numpy as np
from numpy import isnan
import os
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fill_missing_one_day(values):
    one_day = 1
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]
def fill_missing_mean(values):
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = np.mean(values)


def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
    else:
        plt.show()


def numpy_to_tensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor).to(device)

def model_saver (file_nm: str, model_type: bool, model):
    if model_type == 'LSTM':
        torch.save(model, os.path.join(os.path.dirname(__file__), "model", file_nm))
    if model_type == 'ENCDEC':
        torch.save(model,os.path.join(os.path.dirname(__file__),"model",file_nm))
    if model_type == 'GULSTM':
        torch.save(model,os.path.join(os.path.dirname(__file__),"model",file_nm))
