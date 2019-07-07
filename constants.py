import torch

lstm_param  = { 
    "batch_size" : 1,
    "forecast_horizon" : 15,
    "look_back" : 28,
    "input_dim" : 4,
    "hidden_dim" : 256,
    "layer_dim" : 3,
    "output_dim" : 1,
    "seq_dim" : 128,
    "lr" : 0.0005,
    "n_epochs" : 120
}
enc_dec_param = {
    "batch_size" : 1,
    "n_features" : 4,
    "forecast_horizon" : 15,
    "look_back" : 28,
    "lr" : 0.0005,
    "n_epochs" : 300,
    "output_dim" :1
}
gu_lstm_param = {
    "n_epochs" : 100,
    "input_dim" : 4,
    "hidden_dim" : 256,
    "layer_dim" : 3,
    "output_dim" : 1,
    "forecast_horizon" : 1,
    "look_back": 28,
    "forecast_horizon": 15,
    "batch_size": 1
}
