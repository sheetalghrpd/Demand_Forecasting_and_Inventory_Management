#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np 
import pandas as pd


# In[2]:


def read_files(folder_path):
 
# Specify the folder containing the CSV files
    folder_path = folder_path
     
    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
     
    # Initialize an empty list to store individual DataFrames
    dfs = []
     
    # Loop through each CSV file and read it into a DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)
     
    # Concatenate all DataFrames in the list into a single DataFrame
    merged_data = pd.concat(dfs, ignore_index=True)

    merged_df=merged_data[merged_data['Sales_Qty']>=0]
    merged_df.replace(' ',np.nan,inplace=True)
    merged_df.replace('-',np.nan,inplace=True)
    
    return merged_df

