import numpy as np
import argparse
import tqdm
import pickle
import pandas as pd
import os
import datetime
import warnings
import seaborn as sns
from matplotlib import pyplot as plt
from data import StockData

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utility.utility import *
from utility.training import callback
import tensorflow as tf

import configparser

# turn off system warning 
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def main(config, args):
    start_time = datetime.datetime.now()

    datetime_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # basic setting
    make_dir(config)
    maybe_make_dir(f'logs/{args.mode}/{args.model_type}')
    maybe_make_dir(f'logs/csv_logger/{args.model_type}')

    # logging    
    print(f'Training Start')

    # reading data
    print(f'Load Data Begin')
    df = pickle.load(open('data.pk', 'rb'))
    print('df',type(df))
    # df = read_data(config, args.mode)
    if args.mode == 'train':
        train  = df.trainSet 
        valid  = df.validSet 
        # if args.model_type == 'mialstm':
        if args.model_type in ['mima-lstm', 'mia-lstm','ori-mialstm']:
            X_train, y_train = miinput(train)
            X_valid, y_valid = miinput(valid)
            model = load_model(X_valid, args.model_type)

        else:    
            X_train, y_train = input(train)
            X_valid, y_valid = input(valid)
            model = load_model(X_valid.shape, args.model_type)
        
    if args.mode == 'test':
        test  = df.testSet 
        if args.model_type in ['mima-lstm', 'mia-lstm','ori-mialstm']:
            X_test, y_test = miinput(test)
            model = load_model(X_test, args.model_type)
        else:    
            X_test, y_test = input(test)
            model = load_model(X_test.shape, args.model_type)

    print(f'Load Data Finish')

    if args.mode == 'train':
        history = model.fit(
            X_train, y_train, 
            epochs=int(config['MODEL']['epoch']),
            verbose = 1,
            batch_size=512,
            validation_data=(X_valid, y_valid), 
            # callbacks = callback(config, args, datetime_prefix)
        )
        y_pred = model.predict(X_valid)

        model.save_weights(f'model/{args.model_type}/{args.model_type}')

        valid_mse = mean_squared_error(y_valid.reshape(-1,1), y_pred)   
        print(args.model_type, 'mse',valid_mse)
        pd.DataFrame(history.history).to_csv(f'logs/csv_logger/{args.model_type}/{datetime_prefix}_{valid_mse}.csv')

    if args.mode == 'test':
        model.load_weights(f'model/{args.model_type}/{args.model_type}')
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test.reshape(-1,1), y_pred)   
        
        print(args.model_type,'mse',mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default = 'train', required=False, help='either "train" or "test"')
    parser.add_argument('-t', '--model_type', type=str, default='milstm', required=False, help='Multi Input Model "mima-lstm", "mia-lstm","ori-mialstm" or base model "dnn", "conv1d", "conv2d", "lstm" or "transformer"')
    parser.add_argument('-w', '--weight', type=str, required=False, help='stock portfolios')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')

    main(config, args)
    