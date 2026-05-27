# packages
import os
import pickle
import warnings

warnings.simplefilter("ignore")
import plotly.io as pio

pio.renderers.default = "browser"
import numpy as np

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_calib_energy_demand import (
    get_calib_energy_demand_data,
)
from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def make_calib_energy_demand_dm(
    current_file_directory, lever_file, years_ots, years_fts
):
    ###########################################################
    ############## CATEGORIES OF ENERGY CARRIERS ##############
    ###########################################################

    df = get_calib_energy_demand_data(current_file_directory)

    # make data matrix
    dm = DataMatrix.create_from_df(df, 1)

    # # create united kingdom
    # idx = dm.idx
    # arr_temp = dm.array[idx["Germany"],...]
    # dm.add(arr_temp, "Country", "United Kinddom")
    # dm.sort("Country")

    # add hydrogen
    dm.add(0, "Categories1", "hydrogen", dummy=True)
    dm.sort("Categories1")

    # rename
    dm.rename_col("energy-demand", "calib-energy-demand", "Variables")

    # add missing years as nan
    missing = np.array(years_ots)[
        [y not in dm.col_labels["Years"] for y in years_ots]
    ].tolist()
    dm.add(np.nan, "Years", missing, dummy=True)
    dm.sort("Years")
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.sort("Years")

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    with open(f, "wb") as handle:
        pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm


def run(years_ots, years_fts):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_file = "calibration_energy-demand.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            dm = pickle.load(handle)
    else:
        dm = make_calib_energy_demand_dm(
            current_file_directory, lever_file, years_ots, years_fts
        )

    return dm


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)


# df = dm.write_df()
# df = df.loc[df["Country"] == "EU27",:]
# df_temp = pd.melt(df, id_vars = ['Country','Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Years"] == 2021,:]
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)
