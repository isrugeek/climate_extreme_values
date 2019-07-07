#!/bin/bash
source $HOME/.bashrc
source activate ml
python train_main.py LSTM
#python train_lstm.py
#python train_enc_dec_lstm.py
