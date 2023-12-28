#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def Quarter_Formation(master_data):
    quar = []
    for i in master_data.Month:
        if i>3 and i<7:
            quar.append('Q1')
        elif i>6 and i<10:
            quar.append('Q2')
        elif i>9 and i<13:
            quar.append('Q3')
        else: quar.append('Q4')

    quarter = []
    for i in range(len(quar)):
        if quar[i] == 'Q4':
            quarter.append(str(master_data['Year'].iloc[i]-1)+'-'+str(master_data['Year'].iloc[i])+'-'+quar[i])
        else:
            quarter.append(str(master_data['Year'].iloc[i])+'-'+str(master_data['Year'].iloc[i]+1)+'-'+quar[i])
    
    master_data['Quarter'] = quarter
   
#     master_data.reset_index(inplace=True)
    return master_data        


# In[ ]:


def feature_elimination(master_data):

    null_value_features = []
    for i in master_data.columns:
        if master_data[i].isnull().sum()==len(master_data):
            null_value_features.append(i)
    
    one_unique_features = []
    for i in master_data.columns:
        if master_data[i].nunique()==1:
            one_unique_features.append(i)
    
    master_data.drop(columns=one_unique_features +  null_value_features,axis=1,inplace=True)
    null_value_threshold = 70
    x = []
    for i in master_data.columns:
        m = round(master_data[i].isnull().sum()/len(master_data)*100,2)
        x.append([i,m])
    
    feature_null_percentage = pd.DataFrame(data=x,columns=(['Features','Percent_null_value']))
    y = []
    imbalanced_feature_threshold = 95
    for i in master_data.columns:
        n = master_data[i].value_counts().sort_values(ascending=False).head(1).values/len(master_data)
        y.append([i,round(float(n*100),2)])
            
    feature_class_dominant = pd.DataFrame(data=y,columns=(['Features','Percent_class_dominant']))
    
    x = list(feature_class_dominant[feature_class_dominant.Percent_class_dominant>imbalanced_feature_threshold].Features.values)
    x.pop(2)
    
    y = list(feature_null_percentage[feature_null_percentage.Percent_null_value>null_value_threshold]['Features'])
    
    optional_drop_columns = x + y
    
    master_data.drop(columns=optional_drop_columns,axis=1,inplace=True)
#     master_data.reset_index(inplace=True)
    return master_data

