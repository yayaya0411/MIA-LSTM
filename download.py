
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
# warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import data as d
import pickle

#parameter list

makedata = 1

timesize=20
timesize_for_calc_correlation=50
positive_correlation_stock_num=10
negative_correlation_sotck_num=10
train_test_rate=0.7
batch_size=512

#
if makedata:
    kospi=d.StockData(
        # 'Stock/CSI300','Stock/CSI300.csv',
        'Stock/TW0050','Stock/TWII.csv',
        # 'Stock/TWII','Stock/TWII.csv',
        # 'Stock/SAMPLE','Stock/SAMPLE.csv',
        timesize,
        timesize_for_calc_correlation,
        positive_correlation_stock_num,
        negative_correlation_sotck_num,
        train_test_rate,
        batch_size
        )
    pickle.dump(kospi, open('data.pk', 'wb'))
