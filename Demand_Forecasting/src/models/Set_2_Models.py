#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd 
import numpy as np 


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


def set_2_final_run_repea_items_list(cluster_df,df3,less_than_25_IC):
    
    executable_item_codes = set(df3.Item_Code.unique())-set(less_than_25_IC)
    
    final_run_repea_items_list = []    

    for i in executable_item_codes:
        if cluster_df[cluster_df.index==i].values[0][-6:].mean() > 0:
            final_run_repea_items_list.append(i)
    
    return final_run_repea_items_list


# In[ ]:


def impute_outlier(item_code,cluster_df,k):

    data_df = cluster_df[cluster_df.index==item_code].T

    data_df = data_df.rename(columns={item_code:'Sales_Qty'})

    # print(len(data_df))

    val=list(data_df['Sales_Qty'])

    q1=np.quantile(val,0.25)

    q3=np.quantile(val,0.75)

    q2=q3-q1

    #print(q2)

 
        

    upper_bound=q3+(k*(q2))
 
    lower_bound=q1-(k*(q2))
 
    
 
    temp_df=data_df[(data_df['Sales_Qty']<upper_bound) & (data_df['Sales_Qty']>lower_bound)]

    #print(len(temp_df))
 
    val1=list(temp_df['Sales_Qty'])
 
    q1_t=np.quantile(val1,0.25)
 
    q3_t=np.quantile(val1,0.75)
 
    q2_t=q3_t-q1_t

    #print(q2_t)
 
    upper_bound_t=q3_t+(k*(q2_t))
 
    lower_bound_t=q1_t-(k*(q2_t))
 
    if upper_bound_t!=0:
 
        data_df['Sales_Qty']=data_df['Sales_Qty'].apply(lambda x:upper_bound_t if x>upper_bound_t else lower_bound_t if x<lower_bound_t else x)
 
    else:

        pass
 
 
    return data_df
 

def data_preperation(cluster_df,item_code,window_sizes,exp_window_sizes,lead_periods, lag_periods):
    
    data_df = cluster_df[cluster_df.index==item_code].T

    data_df = data_df.rename(columns={item_code:'Sales_Qty'})
    
    for window_size in window_sizes:
        ma_column = f'MA_{window_size}'  
        data_df[ma_column] = data_df['Sales_Qty'].rolling(window=window_size).mean()

    for exp_window_size in exp_window_sizes:
        ema_column = f'EMA_{exp_window_size}'  
        data_df[ema_column] = data_df['Sales_Qty'].ewm(span=exp_window_size,adjust=False).mean()

    # Lead Columns
    for lead_period in lead_periods:
        data_df[f'lead_{lead_period}'] = data_df['Sales_Qty'].shift(-lead_period)

    # Lag Columns
    for lag_period in lag_periods:
        data_df[f'lag_{lag_period}'] = data_df['Sales_Qty'].shift(-lead_period)

    
    start = 1
    end = len(data_df)
    recency_range = list(range(start, end+1))
    current_month = len(data_df)
    data_df['current_month'] = current_month
    data_df['observed_month'] = recency_range
    x_values = (data_df['current_month'] - data_df['observed_month']).tolist()
    recency = list(map(lambda x:pow(2,-x),x_values))
    data_df['recency_factor'] = recency
    data_df.drop(["current_month","observed_month"],axis=1,inplace=True)

    df = data_df.fillna(0)

    return df


def output_feat_df(multi_data,item_code,threshold):

    # model_data = data_preperation(item_code,cluster_df, window_sizes=[2, 3, 4,5,6], lead_periods=[2, 3, 4,5,6], lag_periods=[2, 3, 4,5,6])
    # fea_imp_rf=RandomForestRegressor()

    df_x=multi_data.iloc[:,1:]
    df_y=multi_data.iloc[:,0]
    fea_imp_rf=RandomForestRegressor()

    fea_imp_rf.fit(df_x,df_y)

    fea_data=pd.DataFrame(list(zip(df_x.columns,list(fea_imp_rf.feature_importances_)))).sort_values(1,ascending=False)
    #print(fea_data)
    a=0
    sum=0
    for i in list(fea_data[1]):
        if sum< threshold:
            sum=sum+i
            a=a+1
        else:
            break
    #print(a)
    fea_data=fea_data.iloc[:a,:]
    #print(fea_data)

    lst=list(fea_data[0])
    final_lst=['Sales_Qty']+lst
    
    final_data=multi_data[final_lst]

    #final_data=final_data.rename(columns={'Sales_Qty':'Sales_Qty_y'})

    return final_data

def RandomForestRegressor_Model_01(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold):
    
    final_col=['Sales_Qty']

    for i in multi_model_data.columns:

        if 'MA' in i:

            final_col.append(i)
    
    
    multi_model_data=multi_model_data[final_col]
    
    rob_sca = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
    scaled_values = rob_sca.fit_transform(multi_model_data)
    
    sequences = []
    targets = []
    
    for i in range(len(multi_model_data) - n_lookback):
        seq = scaled_values[i:i + n_lookback,::]
        target = scaled_values[i + n_lookback,::]
        sequences.append(seq)
        targets.append(target)
    
    x = np.array(sequences)
    y = np.array(targets)
    #print(y.shape)
    
    multi_x_train = x[:split_threshold]
    multi_x_test = x[split_threshold:]
    
    multi_y_train = y[:split_threshold]
    multi_y_test = y[split_threshold:]
    #print(multi_x_train)
    multi_x_train = multi_x_train.reshape(multi_x_train.shape[0],multi_x_train.shape[1]*multi_x_train.shape[2])
    multi_x_test = multi_x_test.reshape(multi_x_test.shape[0],multi_x_test.shape[1]*multi_x_test.shape[2])
    #print(multi_x_train)
    model = RandomForestRegressor( n_estimators=500,
    criterion='squared_error',
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=3)
    
    
    model.fit(multi_x_train,multi_y_train)
    
    multi_df = pd.DataFrame({'Actual_Sales_Qty':(multi_model_data.Sales_Qty)})
    multi_df['Test_Predicted_Sales_Qty'] = np.nan
    multi_df['Forecast_Sales_Qty'] = np.nan
    
    
    pred = rob_sca.inverse_transform(np.array(model.predict(multi_x_test)))
    
    for i in range(len(multi_y_test)):
        multi_df['Test_Predicted_Sales_Qty'].iloc[len(multi_df)-len(multi_y_test)+i] = int(pred[i][0])
    
    window = scaled_values[-n_lookback:]
    
    for i in range(n_forecast):
        y_pred = model.predict(np.array(window[-n_lookback:]).reshape(1,multi_x_train.shape[1]))
        window = np.append(window,y_pred,axis=0)
    
    window = rob_sca.inverse_transform(window)
    forecast = []
    
    for i in window[-n_forecast:]:
        forecast.append(int(i[0]))
    
    results = pd.concat([multi_df,pd.DataFrame({'Forecast_Sales_Qty':forecast})],ignore_index=True)
    
    time_index = []
    
    for i in range(len(cluster_df.columns)):
    
        time_index.append(str(cluster_df.columns[i][0])+'-'+str(cluster_df.columns[i][1]))
    
    
    future_dates=pd.date_range(freq='MS',start=max(pd.to_datetime(time_index,dayfirst=True)),periods=n_forecast+1,inclusive='right')
    
    for i in future_dates:
        time_index.append(str(i)[0:7])
    
    results['Date']=time_index
    results.set_index(results['Date'],inplace=True)
    results= results.drop(['Date'],axis=1) 
    results.iloc[len(results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values)-1,-1] = results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-1]
    
    
    rmse = np.sqrt(mean_squared_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values))
    
    
    mape = mean_absolute_percentage_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
    
    mae = mean_absolute_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
    
    r2_scr = r2_score(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

    
    ### fitting on whole data
    multi_x_train_final= x.reshape(x.shape[0],x.shape[1]*x.shape[2])
   
    model.fit(multi_x_train_final,y)
    window1 = scaled_values[-n_lookback:]
    
    for i in range(n_forecast):
        y_pred_1= model.predict(np.array(window1[-n_lookback:]).reshape(1,multi_x_train_final.shape[1]))
        window1 = np.append(window1,y_pred_1,axis=0)
    
    window = rob_sca.inverse_transform(window1)
    forecast_1 = []
    
    for i in window[-n_forecast:]:
        forecast_1.append(int(i[0]))
    
    Main_Fianl_forecast=[]
    for i in range(len(multi_model_data)):
        Main_Fianl_forecast.append(np.nan)
    
    Main_Fianl_forecast.extend(forecast_1)
    results['Fianl_forecast']=Main_Fianl_forecast
    results['r2_score'] = r2_scr
    
    results['Model'] = 'RF_Model_01'
    results['Item_Code'] = item_code
    results['RMSE'] = rmse
    results['MAE'] = mae
    results['MAPE'] = mape
    #print(results)
    return results

def XGBOOST_03(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold):

    
    final_col=['Sales_Qty']
 
    for i in multi_model_data.columns:
    
        if 'MA' in i:
    
            final_col.append(i)
    
    
    multi_model_data=multi_model_data[final_col]

    
    rob_sca = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
    scaled_values = rob_sca.fit_transform(multi_model_data)

    sequences = []
    targets = []

    for i in range(len(multi_model_data) - n_lookback):
        seq = scaled_values[i:i + n_lookback,::]
        target = scaled_values[i + n_lookback,::]
        sequences.append(seq)
        targets.append(target)

    x = np.array(sequences)
    y = np.array(targets)

    multi_x_train = x[:split_threshold]
    multi_x_test = x[split_threshold:]

    multi_y_train = y[:split_threshold]
    multi_y_test = y[split_threshold:]

    multi_x_train = multi_x_train.reshape(multi_x_train.shape[0],multi_x_train.shape[1]*multi_x_train.shape[2])
    multi_x_test = multi_x_test.reshape(multi_x_test.shape[0],multi_x_test.shape[1]*multi_x_test.shape[2])

    model = XGBRegressor(n_estimators=1500,max_depth=7,learning_rate=0.1)

    model.fit(multi_x_train,multi_y_train)
    
    multi_df = pd.DataFrame({'Actual_Sales_Qty':(multi_model_data.Sales_Qty)})
    multi_df['Test_Predicted_Sales_Qty'] = np.nan
    multi_df['Forecast_Sales_Qty'] = np.nan

    pred = rob_sca.inverse_transform(np.array(model.predict(multi_x_test)))

    for i in range(len(multi_y_test)):
        multi_df['Test_Predicted_Sales_Qty'].iloc[len(multi_df)-len(multi_y_test)+i] = int(pred[i][0])

    window = scaled_values[-n_lookback:]

    for i in range(n_forecast):
        y_pred = model.predict(np.array(window[-n_lookback:]).reshape(1,multi_x_train.shape[1]))
        window = np.append(window,y_pred,axis=0)

    window = rob_sca.inverse_transform(window)
    forecast = []

    for i in window[-n_forecast:]:
        forecast.append(int(i[0]))

    results = pd.concat([multi_df,pd.DataFrame({'Forecast_Sales_Qty':forecast})],ignore_index=True)

    time_index = []

    for i in range(len(cluster_df.columns)):

        time_index.append(str(cluster_df.columns[i][0])+'-'+str(cluster_df.columns[i][1]))


    future_dates=pd.date_range(freq='MS',start=max(pd.to_datetime(time_index,dayfirst=True)),periods=n_forecast+1,inclusive='right')

    for i in future_dates:
        time_index.append(str(i)[0:7])

    results['Date']=time_index
    results.set_index(results['Date'],inplace=True)
    results= results.drop(['Date'],axis=1) 
    results.iloc[len(results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values)-1,-1] = results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-1]


    rmse = np.sqrt(mean_squared_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values))


    mape = mean_absolute_percentage_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

    mae = mean_absolute_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
    
    r2_scr = r2_score(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

     ### fitting on whole data
    multi_x_train_final= x.reshape(x.shape[0],x.shape[1]*x.shape[2])
   
    model.fit(multi_x_train_final,y)
    window1 = scaled_values[-n_lookback:]
    
    for i in range(n_forecast):
        y_pred_1= model.predict(np.array(window1[-n_lookback:]).reshape(1,multi_x_train_final.shape[1]))
        window1 = np.append(window1,y_pred_1,axis=0)
    
    window = rob_sca.inverse_transform(window1)
    forecast_1 = []
    
    for i in window[-n_forecast:]:
        forecast_1.append(int(i[0]))
    
    Main_Fianl_forecast=[]
    for i in range(len(multi_model_data)):
        Main_Fianl_forecast.append(np.nan)
    
    Main_Fianl_forecast.extend(forecast_1)
    results['Fianl_forecast']=Main_Fianl_forecast

    
    
    
    results['r2_score'] = r2_scr
    
    results['Model'] = 'XGBOOST_03'
    results['Item_Code'] = item_code
    results['RMSE'] = rmse
    results['MAE'] = mae
    results['MAPE'] = mape
    
    results = results.reset_index()
    return results

def STACKED_LSTM_01(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold):
    
    rob_sca = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
    scaled_values = rob_sca.fit_transform(multi_model_data)
    
    sequences = []
    targets = []
    
    for i in range(len(multi_model_data) - n_lookback):
        seq = scaled_values[i:i + n_lookback,::]
        target = scaled_values[i + n_lookback,::]
        sequences.append(seq)
        targets.append(target)
    
    x = np.array(sequences)
    y = np.array(targets)
    
    multi_x_train = x[:split_threshold]
    multi_x_test = x[split_threshold:]
    
    multi_y_train = y[:split_threshold]
    multi_y_test = y[split_threshold:]
    
    np.random.seed(1)
    random.seed(123)
    tf.random.set_seed(1234)
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_lookback, multi_x_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(28, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
#     model.add(LSTM(28, activation='relu'))
#     model.add(Dropout(0.1))
    model.add(Dense(multi_x_train.shape[2]))
    model.compile(optimizer='adam', loss='mse')
        
    model.fit(multi_x_train,multi_y_train,epochs=60,batch_size=5,verbose=False)
    
    multi_df = pd.DataFrame({'Actual_Sales_Qty':(multi_model_data.Sales_Qty)})
    multi_df['Test_Predicted_Sales_Qty'] = np.nan
    multi_df['Forecast_Sales_Qty'] = np.nan
    
    pred = rob_sca.inverse_transform(np.array(model.predict(multi_x_test)))
    
    for i in range(len(multi_y_test)):
            multi_df['Test_Predicted_Sales_Qty'].iloc[len(multi_df)-len(multi_y_test)+i] = int(pred[i][0])
    
    window = scaled_values[-n_lookback:]
    
    for i in range(n_forecast):
        y_pred = model.predict(np.array(window[-n_lookback:]).reshape(1,n_lookback,multi_x_train.shape[2]))
        window = np.append(window,y_pred,axis=0)
    
    window = rob_sca.inverse_transform(window)
    
    forecast = []
    
    for i in window[-n_forecast:]:
        forecast.append(int(i[0]))
    
    results = pd.concat([multi_df,pd.DataFrame({'Forecast_Sales_Qty':forecast})],ignore_index=True)
    
    time_index = []
    
    for i in range(len(cluster_df.columns)):
    
        time_index.append(str(cluster_df.columns[i][0])+'-'+str(cluster_df.columns[i][1]))
    
    
    future_dates=pd.date_range(freq='MS',start=max(pd.to_datetime(time_index,dayfirst=True)),periods=n_forecast+1,inclusive='right')
    
    for i in future_dates:
        time_index.append(str(i)[0:7])
    
    results['Date']=time_index
    results.set_index(results['Date'],inplace=True)
    results= results.drop(['Date'],axis=1) 
    
    results.iloc[len(results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values)-1,-1] = results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-1]
    
    
    rmse = np.sqrt(mean_squared_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values))
    
    
    mape = mean_absolute_percentage_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
    
    mae = mean_absolute_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
    
    r2_scr = r2_score(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
    
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
    
    ### training on full data
    model.fit(x,y,epochs=80,batch_size=5,verbose=False)
    window_1 = scaled_values[-n_lookback:]
    for i in range(n_forecast):
        y_pred = model.predict(np.array(window_1[-n_lookback:]).reshape(1,n_lookback,x.shape[2]))
        window_1 = np.append(window_1,y_pred,axis=0)
    
    window = rob_sca.inverse_transform(window_1)
    forecast_1 = []
    for i in window[-n_forecast:]:
        forecast_1.append(int(i[0]))
    
    Main_Fianl_forecast=[]
    for i in range(len(multi_model_data)):
        Main_Fianl_forecast.append(np.nan)
    
    Main_Fianl_forecast.extend(forecast_1)
    results['Fianl_forecast']=Main_Fianl_forecast
    
    results['r2_score'] = r2_scr
    
    results['Model'] = 'STACKED_LSTM_01'
    results['Item_Code'] = item_code
    results['RMSE'] = rmse
    results['MAE'] = mae
    results['MAPE'] = mape
    
    
    return results


def NBEATS_01(cluster_df,item_code,multi_model_data):
 
    cluster_df1 = cluster_df.copy()
 
    multi_model_data1 = multi_model_data.copy()

    lst=[]

    for i in cluster_df1.columns:

        if i[0]>2010:

            lst.append((i[0],i[1]))

        else:

            lst.append((i[0]-1,i[1]))
 
    cluster_df1.columns=lst
 
    QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
 
    dates=[]
 
    for i in cluster_df1.columns:
 
        dates.append(str(i[1])+'-'+str(i[0]))
 
    split_boundary = (len(cluster_df1.columns) + split_threshold-1)
 
    multi_model_data1['dates']=dates
 
    multi_model_data1['dates']=pd.to_datetime(multi_model_data1['dates'])
 
    multi_model_data1=multi_model_data1.set_index(multi_model_data1['dates'])
 
    multi_model_data1.drop(['dates'],axis=1,inplace=True)
 
    ts = TimeSeries.from_series(multi_model_data1,fill_missing_dates=True,freq=None,fillna_value = 0)
 
    if isinstance(split_boundary, str):

        split = pd.Timestamp(split_boundary)

    else:

        split = split_boundary

    ts_train, ts_test = ts.split_after(split)
 
 
    transformer = Scaler()

    ts_ttrain = transformer.fit_transform(ts_train)

    ts_ttest = transformer.transform(ts_test)

    ts_t = transformer.transform(ts)

    model = NBEATSModel(input_chunk_length = 9, output_chunk_length = n_forecast,n_epochs = 20

                        ,num_layers = 6,batch_size = 10,random_state=42)

    model.fit(ts_ttrain,verbose=False)
 
    ts_tpred = model.predict(n=len(ts_test) + n_forecast)

    ts_pred = transformer.inverse_transform(ts_tpred)
 
    multi_df = pd.DataFrame({'Actual_Sales_Qty':multi_model_data.Sales_Qty})
 
    multi_df['Test_Predicted_Sales_Qty'] = np.nan
 
    multi_df['Forecast_Sales_Qty'] = np.nan
 
 
    for i in range(len(ts_test)):
 
            multi_df['Test_Predicted_Sales_Qty'].iloc[len(multi_model_data1)-len(ts_test)+i] = int(ts_pred[i].values()[0][0])
 
    forecast_values = []

    for i in ts_pred[-n_forecast:].values():

        forecast_values.append(int(i[0]))
 
 
    results = pd.concat([multi_df,pd.DataFrame({'Forecast_Sales_Qty':forecast_values})],ignore_index=True)
 
    time_index = []

    for i in range(len(cluster_df1.columns)):

        time_index.append(str(cluster_df.columns[i][0])+'-'+str(cluster_df.columns[i][1]))
 
    future_dates=pd.date_range(freq='MS',start=max(pd.to_datetime(time_index,dayfirst=True)),periods=n_forecast+1,inclusive='right')
 
    for i in future_dates:

        time_index.append(str(i)[0:7])
 
    results['Date']=time_index

    results.set_index(results['Date'],inplace=True)

    results.drop(['Date'],axis=1,inplace=True)

    results.iloc[len(results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values)-1,-1] = results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-1]
 
 
    rmse = np.sqrt(mean_squared_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values))
 
    mape = mean_absolute_percentage_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

    mae = mean_absolute_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

    r2_scr = r2_score(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
 
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
 
    #####
 
 
    model.fit(ts_t)
 
    ts_tpred_1 = model.predict(n= n_forecast)

    ts_pred_final = transformer.inverse_transform(ts_tpred_1)
 
 
    forecast_values_1 = []

    for i in ts_pred_final[-n_forecast:].values():

        forecast_values_1.append(int(i[0]))
 
    Main_Fianl_forecast=[]

    for i in range(len(multi_model_data)):

        Main_Fianl_forecast.append(np.nan)

    Main_Fianl_forecast.extend(forecast_values_1)

    results['Fianl_forecast']=Main_Fianl_forecast
 
    results['r2_score'] = r2_scr
 
    results['Model'] = 'NBEATS_01'

    results['Item_Code'] = item_code

    results['RMSE'] = rmse

    results['MAE'] = mae

    results['MAPE'] = mape

    return results


def STACKED_GRU_03(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold):

    rob_sca = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)

    scaled_values = rob_sca.fit_transform(multi_model_data)

    sequences = []

    targets = []

    for i in range(len(multi_model_data) - n_lookback):

        seq = scaled_values[i:i + n_lookback,::]

        target = scaled_values[i + n_lookback,::]

        sequences.append(seq)

        targets.append(target)
 
    x = np.array(sequences)

    y = np.array(targets)

    multi_x_train = x[:split_threshold]

    multi_x_test = x[split_threshold:]
 
    multi_y_train = y[:split_threshold]

    multi_y_test = y[split_threshold:]

    np.random.seed(1)

    random.seed(123)

    tf.random.set_seed(1234)

    model = Sequential()

    model.add(GRU(256, activation='relu', input_shape=(n_lookback, multi_x_train.shape[2]), return_sequences=True))

    model.add(Dropout(0.3))

    model.add(GRU(128, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(multi_x_train.shape[2]))

    model.compile(optimizer='adam', loss='mse')

    model.fit(multi_x_train,multi_y_train,epochs=100,batch_size=2,verbose=False)

    multi_df = pd.DataFrame({'Actual_Sales_Qty':(multi_model_data.Sales_Qty)})

    multi_df['Test_Predicted_Sales_Qty'] = np.nan

    multi_df['Forecast_Sales_Qty'] = np.nan
 
    pred = rob_sca.inverse_transform(np.array(model.predict(multi_x_test)))
 
    for i in range(len(multi_y_test)):

            multi_df['Test_Predicted_Sales_Qty'].iloc[len(multi_df)-len(multi_y_test)+i] = int(pred[i][0])
 
    window = scaled_values[-n_lookback:]
 
    for i in range(n_forecast):

        y_pred = model.predict(np.array(window[-n_lookback:]).reshape(1,n_lookback,multi_x_train.shape[2]))

        window = np.append(window,y_pred,axis=0)
 
    window = rob_sca.inverse_transform(window)

    forecast = []
 
    for i in window[-n_forecast:]:

        forecast.append(int(i[0]))
 
    results = pd.concat([multi_df,pd.DataFrame({'Forecast_Sales_Qty':forecast})],ignore_index=True)
 
    time_index = []
 
    for i in range(len(cluster_df.columns)):
 
        time_index.append(str(cluster_df.columns[i][0])+'-'+str(cluster_df.columns[i][1]))
 
 
    future_dates=pd.date_range(freq='MS',start=max(pd.to_datetime(time_index,dayfirst=True)),periods=n_forecast+1,inclusive='right')
 
    for i in future_dates:

        time_index.append(str(i)[0:7])
 
    results['Date']=time_index

    results.set_index(results['Date'],inplace=True)

    results= results.drop(['Date'],axis=1) 

    results.iloc[len(results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values)-1,-1] = results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-1]
 
 
    rmse = np.sqrt(mean_squared_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
 
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values))
 
 
    mape = mean_absolute_percentage_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
 
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
 
    mae = mean_absolute_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
 
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

    r2_scr = r2_score(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,
 
            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
 
    ###
 
     ### training on full data

    model.fit(x,y,epochs=100,batch_size=2,verbose=False)

    window_1 = scaled_values[-n_lookback:]

    for i in range(n_forecast):

        y_pred = model.predict(np.array(window_1[-n_lookback:]).reshape(1,n_lookback,x.shape[2]))

        window_1 = np.append(window_1,y_pred,axis=0)

    window = rob_sca.inverse_transform(window_1)

    forecast_1 = []

    for i in window[-n_forecast:]:

        forecast_1.append(int(i[0]))

    Main_Fianl_forecast=[]

    for i in range(len(multi_model_data)):

        Main_Fianl_forecast.append(np.nan)

    Main_Fianl_forecast.extend(forecast_1)

    results['Fianl_forecast']=Main_Fianl_forecast

    results['r2_score'] = r2_scr

    results['Model'] = 'STACKED_GRU_03'

    results['Item_Code'] = item_code

    results['RMSE'] = rmse

    results['MAE'] = mae

    results['MAPE'] = mape

 
    return results

def STACKED_BIGRU_01(cluster_df,multi_model_data,item_code,n_lookback,n_forecast,split_threshold):

    rob_sca = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)

    scaled_values = rob_sca.fit_transform(multi_model_data)

    sequences = []

    targets = []

    for i in range(len(multi_model_data) - n_lookback):

        seq = scaled_values[i:i + n_lookback,::]

        target = scaled_values[i + n_lookback,::]

        sequences.append(seq)

        targets.append(target)

    x = np.array(sequences)

    y = np.array(targets)

    multi_x_train = x[:split_threshold]

    multi_x_test = x[split_threshold:]

    multi_y_train = y[:split_threshold]

    multi_y_test = y[split_threshold:]

    np.random.seed(1)

    random.seed(123)

    tf.random.set_seed(1234)

    model = Sequential()

    model.add(Bidirectional(GRU(50, activation='relu', input_shape=(n_lookback, multi_x_train.shape[2]), return_sequences=True)))

    model.add(Dropout(0.3))

    model.add(Bidirectional(GRU(28, activation='relu', return_sequences=False)))

    model.add(Dropout(0.2))

#     model.add(Bidirectional(GRU(32, activation='relu')))

#     model.add(Dropout(0.1))

    model.add(Dense(multi_x_train.shape[2]))

    model.compile(optimizer='adam', loss='mse') 


    model.fit(multi_x_train,multi_y_train,epochs=60,batch_size=2,verbose=False)

    multi_df = pd.DataFrame({'Actual_Sales_Qty':(multi_model_data.Sales_Qty)})

    multi_df['Test_Predicted_Sales_Qty'] = np.nan

    multi_df['Forecast_Sales_Qty'] = np.nan

    pred = rob_sca.inverse_transform(np.array(model.predict(multi_x_test)))

    for i in range(len(multi_y_test)):

            multi_df['Test_Predicted_Sales_Qty'].iloc[len(multi_df)-len(multi_y_test)+i] = int(pred[i][0])

    window = scaled_values[-n_lookback:]

    for i in range(n_forecast):

        y_pred = model.predict(np.array(window[-n_lookback:]).reshape(1,n_lookback,multi_x_train.shape[2]))

        window = np.append(window,y_pred,axis=0)

    window = rob_sca.inverse_transform(window)

    forecast = []

    for i in window[-n_forecast:]:

        forecast.append(int(i[0]))

    results = pd.concat([multi_df,pd.DataFrame({'Forecast_Sales_Qty':forecast})],ignore_index=True)

    time_index = []

    for i in range(len(cluster_df.columns)):

        time_index.append(str(cluster_df.columns[i][0])+'-'+str(cluster_df.columns[i][1]))


    future_dates=pd.date_range(freq='MS',start=max(pd.to_datetime(time_index,dayfirst=True)),periods=n_forecast+1,inclusive='right')

    for i in future_dates:

        time_index.append(str(i)[0:7])

    results['Date']=time_index

    results.set_index(results['Date'],inplace=True)

    results= results.drop(['Date'],axis=1) 

    results.iloc[len(results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values)-1,-1] = results[results['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-1]


    rmse = np.sqrt(mean_squared_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values))


    mape = mean_absolute_percentage_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

    mae = mean_absolute_error(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)

    r2_scr = r2_score(results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values,

            results[results['Test_Predicted_Sales_Qty'].isnull()==False]['Test_Predicted_Sales_Qty'].values)
 
 
      ### training on full data

    model.fit(x,y,epochs=75,batch_size=5,verbose=False)

    window_1 = scaled_values[-n_lookback:]

    for i in range(n_forecast):

        y_pred = model.predict(np.array(window_1[-n_lookback:]).reshape(1,n_lookback,x.shape[2]))

        window_1 = np.append(window_1,y_pred,axis=0)

    window = rob_sca.inverse_transform(window_1)

    forecast_1 = []

    for i in window[-n_forecast:]:

        forecast_1.append(int(i[0]))

    Main_Fianl_forecast=[]

    for i in range(len(multi_model_data)):

        Main_Fianl_forecast.append(np.nan)

    Main_Fianl_forecast.extend(forecast_1)

    results['Fianl_forecast']=Main_Fianl_forecast


    results['r2_score'] = r2_scr

    results['Model'] = 'STACKED_BIGRU_01'

    results['Item_Code'] = item_code

    results['RMSE'] = rmse

    results['MAE'] = mae

    results['MAPE'] = mape


    return results

