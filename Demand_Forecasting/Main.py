import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from src.features.Get_data import *
from src.features.Data_preprocessing import *
from src.features.clustered_data import *

from src.models.Set_1_Main import *
from src.models.Set_1_Desirable_Items_Data import *
from src.models.Set_2_Main import *
from src.models.Set_2_Desirable_Items_Data import *
from src.models.Final_output import *


master_data = read_files(r'File_Path')

master_data = Quarter_Formation(master_data)

master_data = feature_elimination(master_data)

master_data = obsolete_features(master_data)

master_data = clustering(master_data)

cluster_df,final_run_repea_items_list = cluster_df_formation(master_data)

set_1_Data = set_1_final_results(cluster_df,final_run_repea_items_list)

set_1_IC_Data,less_than_25_IC = set_1_final_data(set_1_Data)

set_2_Data = set_2_final_results(cluster_df,set_1_Data,less_than_25_IC)

set_2_IC_Data = set_2_desirable_items_data(set_2_Data)

db_data = Final_Data(set_1_IC_Data,set_2_IC_Data,grand_data = master_data.copy())

