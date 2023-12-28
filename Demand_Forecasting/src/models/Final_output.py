import pandas as pd 
import numpy as np 


from src.models.Set_1_Desirable_Items_Data import *

def impute_zero_forecast_sales(df,n_forecast , weights=(0.5, 0.3, 0.2)):
    
    for item_code in df['Item_Code'].unique():
        subset_df = df[df['Item_Code'] == item_code]

        if subset_df['Forecast_Sales_Qty'][-n_forecast:].mean() == 0:
            mean_cal_val = subset_df.loc[subset_df['Actual_Sales_Qty'].notnull(), 'Actual_Sales_Qty'].values[-9:]
            mean_sales = np.dot(weights, [mean_cal_val[-3:].sum(), mean_cal_val[-6:-3].sum(), mean_cal_val[:3].sum()])
            subset_df.loc[subset_df['Forecast_Sales_Qty'] == 0, 'Forecast_Sales_Qty'] = mean_sales

        df.update(subset_df)

    return df


# In[ ]:


def both_sets_data(df1,df2):
        
    all_df =  pd.concat([df1,df2])

    all_df = all_df.drop('Forecast_Sales_Qty',axis=1)

    all_df = all_df.rename(columns={'Fianl_forecast':'Forecast_Sales_Qty'})

    set_1_final_df = impute_zero_forecast_sales(all_df, n_forecast = 4, weights=(0.5, 0.3, 0.2))
    
    return set_1_final_df


# In[ ]:


def Final_Data(set_1_IC_Data,set_2_IC_Data,grand_data):
    
    df2 = impute_zero_forecast_sales(set_2_IC_Data, 4, weights=(0.5, 0.3, 0.2))
    
    set_1_final_df = both_sets_data(set_1_IC_Data,df2)    
    
    
    final_df = pd.DataFrame()

    for i in set_1_final_df.Item_Code.unique():
        results = set_1_final_df[set_1_final_df['Item_Code'] == i]
        last_actual_sales_qty = results[results['Actual_Sales_Qty'].notnull()]['Actual_Sales_Qty'].values[-1]
        results = results.reset_index()
        results.loc[len(results) - 5, 'Forecast_Sales_Qty'] = last_actual_sales_qty
        final_df = pd.concat([final_df, results])

    new_df=pd.DataFrame()

    for i in final_df['Item_Code'].unique():
        try_df=final_df[final_df['Item_Code']==i]
        try_df['Product_Group_Name'] = list(grand_data[grand_data['Item_Code']==i]['Product_Group_Name'])[-1]
        new_df=pd.concat([new_df,try_df])

    new_df = new_df[new_df['Product_Group_Name']!=' ']
    new_df = new_df[new_df['Product_Group_Name'].isnull()==False]

    trial_df= pd.DataFrame()
    for i in new_df.Item_Code.unique():
        if new_df[(new_df['Item_Code']==i) & (new_df['Actual_Sales_Qty'].isnull()==False)]['Actual_Sales_Qty'].values.mean()!=0:
            trial_df = pd.concat([trial_df,new_df[new_df['Item_Code']==i]]) 

    use_df = grand_data[['Item_Code','Group_Labels']].drop_duplicates()
    fd_df = pd.merge(trial_df,use_df,on='Item_Code',how='left')
    fd_df = fd_df[(fd_df['Group_Labels']=='Runner_Items') | (fd_df['Group_Labels']=='Repeater_Items')]

    forecast_negative_handled_df = impute_negative_sales(fd_df)

    final_handled_df = greater_than_range(forecast_negative_handled_df,1500)
    
    return final_handled_df

