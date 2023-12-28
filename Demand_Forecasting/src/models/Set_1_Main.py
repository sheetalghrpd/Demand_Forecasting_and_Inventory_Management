 #!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,GRU,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.linear_model import ElasticNet
import random 
from tqdm.auto import tqdm
from tensorflow.keras.optimizers import Adam
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape,rmse,mae
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
from darts.models import TFTModel, NBEATSModel
from xgboost import XGBRegressor
from pmdarima.arima import auto_arima
import random

n_lookback = 9               
n_forecast = 4            
split_threshold = -9
epoch_dl = 100
epoch_t = 25


# In[2]:


from src.models.Set_1_Models import *


# In[ ]:


def set_1_final_results(cluster_df,final_run_repea_items_list):    
    
    final_df = pd.DataFrame()

    for item_code in tqdm(final_run_repea_items_list):
#         try:
        imputed_df = impute_outlier(item_code,cluster_df,1.5)
        multi_data = data_preperation(imputed_df,[4,6,8,10,12,15,16,5,9],[5,6,10,14,15,19,20],[3,5,7,9],[4,6,7,9])
        multi_model_data = output_feat_df(multi_data,item_code,0.80)

        STACKED_BIGRU_01_df = STACKED_BIGRU_01(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold)

        STACKED_GRU_03_df = STACKED_GRU_03(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold)

        STACKED_LSTM_01_df = STACKED_LSTM_01(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold)

        RF_df = RandomForestRegressor_Model(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold)        

        RF_df_01= RandomForestRegressor_Model_01(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold)

        RF_df_02= RandomForestRegressor_Model_02(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold)

        XGBOOST_df_03 = XGBOOST_03(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold)

        NBEATS_01_df = NBEATS_01(cluster_df,item_code,multi_model_data)

        final_df = pd.concat([final_df,NBEATS_01_df,XGBOOST_df_03,RF_df_02,RF_df_01,RF_df,STACKED_LSTM_01_df,STACKED_BIGRU_01_df,
                            STACKED_GRU_03_df])
#         except: pass
    
    final_df.reset_index(inplace=True)
    
    return final_df

