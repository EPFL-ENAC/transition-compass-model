
import pandas as pd
import os
import numpy as np

def get_emissions_data(current_file_directory, ghg):
    filepath = os.path.join(current_file_directory,  "../data/emissions/Evolution_GHG_since_1990_2025-04.xlsx")
    df = pd.read_excel(filepath, sheet_name=ghg)
    df.columns = df.iloc[3,:]
    df = df.iloc[31:32,:]
    df = pd.melt(df, id_vars = ['Cat. (1)', np.nan], var_name='Years')
    df = df.loc[:,["Years","value"]]
    df["Years"] = df["Years"].astype(int)
    df["value"] = df["value"].astype(float)
    df["ghg"] = ghg
    return df

