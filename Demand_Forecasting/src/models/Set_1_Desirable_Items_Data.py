#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd 
import numpy as np 


# In[ ]:

def greater_than_range(df3,k):
    modified_df = pd.DataFrame()
    posi_ic_df = pd.DataFrame()
    neg_ic = df3[df3['Forecast_Sales_Qty']>k]['Item_Code'].unique()
    for i in neg_ic:
        copy_df = df3[(df3['Item_Code']==i)]
        mean_cal_val = copy_df[copy_df['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-9:]
        mean_sales = 0.5 * mean_cal_val[-3:].sum() + 0.3 * mean_cal_val[-6:-3].sum() + 0.2 * mean_cal_val[:3].sum()
        copy_df['Forecast_Sales_Qty'] = np.where(copy_df['Forecast_Sales_Qty']>k, mean_sales, copy_df['Forecast_Sales_Qty'])
        modified_df = pd.concat([modified_df,copy_df])
    for i in df3['Item_Code'].unique():
        if i not in neg_ic:
            posi_df = df3[(df3['Item_Code']==i)]
            posi_ic_df = pd.concat([posi_ic_df,posi_df])
    return pd.concat([modified_df,posi_ic_df])


def impute_negative_sales(df3):

    modified_df = pd.DataFrame()

    posi_ic_df = pd.DataFrame()

    neg_ic = df3[df3['Forecast_Sales_Qty']<0]['Item_Code'].unique()

    for i in neg_ic:

        copy_df = df3[(df3['Item_Code']==i)]

        mean_cal_val = copy_df[copy_df['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-9:]

        mean_sales = 0.5 * mean_cal_val[-3:].sum() + 0.3 * mean_cal_val[-6:-3].sum() + 0.2 * mean_cal_val[:3].sum()

        copy_df['Forecast_Sales_Qty'] = np.where(copy_df['Forecast_Sales_Qty'] < 0, mean_sales, copy_df['Forecast_Sales_Qty'])

        modified_df = pd.concat([modified_df,copy_df])

    for i in df3['Item_Code'].unique():

        if i not in neg_ic:

            posi_df = df3[(df3['Item_Code']==i)]

            posi_ic_df = pd.concat([posi_ic_df,posi_df])

    return pd.concat([modified_df,posi_ic_df])


# In[ ]:


def impute_negative_sales_for_final(df3):
    modified_df = pd.DataFrame()
    posi_ic_df = pd.DataFrame()
    neg_ic = df3[df3['Fianl_forecast']<0]['Item_Code'].unique()
    for i in neg_ic:
        copy_df = df3[(df3['Item_Code']==i)]
        mean_cal_val = copy_df[copy_df['Actual_Sales_Qty'].isnull()==False]['Actual_Sales_Qty'].values[-9:]
        mean_sales = 0.5 * mean_cal_val[-3:].sum() + 0.3 * mean_cal_val[-6:-3].sum() + 0.2 * mean_cal_val[:3].sum()
        copy_df['Fianl_forecast'] = np.where(copy_df['Fianl_forecast'] < 0, mean_sales, copy_df['Fianl_forecast'])
        modified_df = pd.concat([modified_df,copy_df])
    for i in df3['Item_Code'].unique():
        if i not in neg_ic:
            posi_df = df3[(df3['Item_Code']==i)]
            posi_ic_df = pd.concat([posi_ic_df,posi_df])
    return pd.concat([modified_df,posi_ic_df])


# In[ ]:


def set_1_final_data(df1):
    
    forecast_negative_handled_df = impute_negative_sales(df1)

    handled_final = impute_negative_sales_for_final(forecast_negative_handled_df)

    df_runner = handled_final.copy()

    Actual_data = pd.read_excel(r'C:\Users\HP\Desktop\rubiscape\Project_1_FIBRO\github\cookiecutter\data\raw\FIBRO_20232024_Sales_Register_Itemwise_07_11_2023_10_09_55.xlsx')
    
    oct_df=pd.pivot_table(Actual_data,columns='Month Year',index='Item Code',values='Quantity',aggfunc='sum',fill_value=0)

    val_df=pd.DataFrame()

    for i in df_runner['Item_Code'].unique():
        try:
            try_df=df_runner[df_runner['Item_Code']==i]
            a=oct_df[oct_df.index==i]['2023-10'].values[0]
            #print(a)
            try_df['Oct_Actual']=a
            val_df=pd.concat([val_df,try_df])
        except:
            pass

    Final_runner_df=val_df[val_df['Date']=='2023-10']
    Final_runner_df['New_Varience']=(Final_runner_df['Fianl_forecast']-Final_runner_df['Oct_Actual'])/(Final_runner_df['Fianl_forecast'])*100
    Final_runner_df['Old_Varience']=(Final_runner_df['Forecast_Sales_Qty']-Final_runner_df['Oct_Actual'])/(Final_runner_df['Forecast_Sales_Qty'])*100


    Final_runner_df.rename(columns={'Varience':'New_Varience'},inplace=True)
    Final_runner_df['Old_Varience']=(Final_runner_df['Forecast_Sales_Qty']-Final_runner_df['Oct_Actual'])/(Final_runner_df['Forecast_Sales_Qty'])*100
    Final_runner_df=Final_runner_df.drop(['Actual_Sales_Qty','Test_Predicted_Sales_Qty'],axis=1)
    Final_runner_df.rename(columns={'Fianl_forecast':'New_Forecast'},inplace=True)
    Final_runner_df['Forecast_Sales_Qty']=round(Final_runner_df['Forecast_Sales_Qty'],0)
    Final_runner_df['New_Forecast']=round(Final_runner_df['New_Forecast'],0)

    sorted_df=pd.DataFrame()
    for i in Final_runner_df['Item_Code'].unique():
        new_df=Final_runner_df[Final_runner_df['Item_Code']==i]
        new_df['New_Varience']=abs(new_df['New_Varience'])
        s_df=new_df.sort_values('New_Varience').iloc[0:1,:]
        #print(s_df)
        sorted_df=pd.concat([sorted_df,s_df])

    less_than_25 = list(sorted_df[sorted_df['New_Varience']<26]['Item_Code'])


    set_1_final_df = pd.DataFrame()

    for i in less_than_25:
        mdl_nm = sorted_df[sorted_df['Item_Code']==i]['Model'].values[0]
        temp_df = df1[(df1['Item_Code']==i) & (df1['Model']==mdl_nm)]

        set_1_final_df = pd.concat([set_1_final_df,temp_df])
        
    return set_1_final_df,less_than_25

