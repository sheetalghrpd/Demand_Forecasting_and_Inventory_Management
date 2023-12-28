#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from src.models.Set_1_Desirable_Items_Data import *

def set_2_desirable_items_data(df1):   
    
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
            try_df['Oct_Actual']=a
            val_df=pd.concat([val_df,try_df])
        except:
            try_df=df_runner[df_runner['Item_Code']==i]
            try_df['Oct_Actual']=0
            val_df=pd.concat([val_df,try_df])

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
        sorted_df=pd.concat([sorted_df,s_df])


    set_1_final_df = pd.DataFrame()

    for i in df1.Item_Code.unique():
        mdl_nm = sorted_df[sorted_df['Item_Code']==i]['Model'].values[0]
        temp_df = df1[(df1['Item_Code']==i) & (df1['Model']==mdl_nm)]

        set_1_final_df = pd.concat([set_1_final_df,temp_df])
    
    return set_1_final_df

