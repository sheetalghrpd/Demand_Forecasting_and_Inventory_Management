#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


def obsolete_features(master_data):  
    
    obsolete_features = ['Month_Year','Party_Address1','Party_Address2','Party_Address3','Party_Address4','Party_Phone_No',
                             'Party_Fax_No','Party_CST_No','Party_SST_No','Party_ECC_No','Party_IT_PAN','Transporter_Name',
                             'Transporter','Year_Id','Excise_Tariff_No','Excise_Description','Sales_Voucher_MonthYear','CGST_Value','SGST_Value','IGST_Value','CGST_Perc','SGST_Perc',
                             'IGST_Perc','GST_Perc','Customer_GSTIN','Indent_No','Indent_Series','Indent_Date','Indenting_Time',
                            'Invoicing_Time','Tonnage','GST_Value','Finished_Weight','LR_No','LR_Date']
        
        
    duplicate_features = ['Party_Region','Customer_State_Code','D_Tonnage','D_Amount','D_Net_Amount','Party_Area','D_Sales_Qty',
                         'Responsibility_Code','Invoice_No','Sales_Person_Name','Description_Short','Description_Long',
                         'Product_SubGroup_Code','Delivery_Location','Customer_State_Name']

    master_data.drop(columns=obsolete_features+duplicate_features,axis=1,inplace=True)

    master_data['Item_Name'] = master_data['Item_Name'].str.strip()
    master_data['Party_Name'] = master_data['Party_Name'].str.strip()

#     master_data.reset_index(inplace=True)

    return master_data


# In[ ]:


def clustering(master_data):
    
    cluster_df = pd.pivot_table(master_data,columns=['Year','Month'],index='Item_Code',values='Sales_Qty',fill_value=0,aggfunc=sum)
    
    runner_items = []
    for j in range(len(cluster_df)):
        counter = 0
        for i in range(len(cluster_df.columns)):
            if cluster_df.iloc[j].values[i] > 0:
                counter+=1
        if counter >= (len(cluster_df.columns)-12):
            runner_items.append(cluster_df.iloc[j].name)

    rare_items = []
    for j in range(len(cluster_df)):
        counter = 0
        for i in cluster_df.iloc[j].values:
            if i > 0:
                counter+=1
        if counter <= 5:
                rare_items.append(cluster_df.iloc[j].name)
    
    repeater_items = []
    for j in range(len(cluster_df)):
        counter = 0
        for i in range(0,len(cluster_df.columns),3):
            if cluster_df.iloc[j].values[i:i+3].any() > 0:
                counter+=1
        if counter >= ((len(cluster_df.columns)-12)//3):
            if (cluster_df.iloc[j].name not in runner_items) and (cluster_df.iloc[j].name not in rare_items):
                repeater_items.append(cluster_df.iloc[j].name)
    
    strenger_items = []
    for j in range(len(cluster_df)):
        counter = 0
        for i in range(0,len(cluster_df.columns),6):
            if cluster_df.iloc[j].values[i:i+6].any() > 0:
                counter+=1
        if counter >= ((len(cluster_df.columns)-12)//6):
            if (cluster_df.iloc[j].name not in runner_items) and (cluster_df.iloc[j].name not in repeater_items) and (cluster_df.iloc[j].name not in rare_items):
                strenger_items.append(cluster_df.iloc[j].name)
    
    yearly_items = []
    for j in range(len(cluster_df)):
        counter = 0
        for i in range(0,len(cluster_df.columns),12):
            if cluster_df.iloc[j].values[i:i+12].any() > 0:
                counter+=1
        if counter >= ((len(cluster_df.columns)-12)//12):
            if (cluster_df.iloc[j].name not in runner_items) and (cluster_df.iloc[j].name not in repeater_items) and (cluster_df.iloc[j].name not in strenger_items) and (cluster_df.iloc[j].name not in rare_items):
                yearly_items.append(cluster_df.iloc[j].name)
    
    master_data['Group_Labels'] = np.nan
    for i in range(len(master_data['Item_Code'])):
        if master_data['Item_Code'].iloc[i] in runner_items:
            master_data['Group_Labels'].iloc[i] = 'Runner_Items'
            
        elif master_data['Item_Code'].iloc[i] in rare_items:
            master_data['Group_Labels'].iloc[i] = 'Rare_Items'
            
        elif master_data['Item_Code'].iloc[i] in repeater_items:
            master_data['Group_Labels'].iloc[i] = 'Repeater_Items'
            
        elif master_data['Item_Code'].iloc[i] in strenger_items:
            master_data['Group_Labels'].iloc[i] = 'Strenger_Items'
            
        else:
            master_data['Group_Labels'].iloc[i] = 'Yearly_Items'
  
#     master_data.reset_index(inplace=True)

    master_data['Year'] = master_data['Year'].astype('int')
    master_data['Month'] = master_data['Month'].astype('int')
    
    return master_data

