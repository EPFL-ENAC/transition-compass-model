# materials: ['aluminium', 'ammonia, 'cement', 'chem', 'copper', 'glass', 'lime', 'other', 'paper', 'steel', 'timber']

# packages
import os
import pickle
import warnings

import numpy as np

warnings.simplefilter("ignore")
import plotly.io as pio

pio.renderers.default = "browser"

from transition_compass_model._database.pre_processing.fix_jumps import fix_jumps_in_dm

# from transition_compass_model._database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_material_net_import import (
    get_prodcom_data,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    linear_fitting,
)


def make_material_net_import_dm(current_file_directory, years_ots, years_fts):
    #########################################
    ##### GET CLEAN DATAFRAME WITH DATA #####
    #########################################

    dm_mat = get_prodcom_data("ds-056120", current_file_directory)

    ###################
    ##### FIX OTS #####
    ###################

    # years_all = years_ots + years_fts

    # set years before 2006 as missing
    idx = dm_mat.idx
    for y in range(1995, 2006 + 1):
        dm_mat.array[:, idx[y], :, :] = np.nan

    # make zeroes as nans
    dm_mat.array[dm_mat.array == 0] = np.nan

    # make timber before 2022 as nan for EU27 (for some reason there is a big jump in 2022 for the checked countries)
    idx = dm_mat.idx
    for y in range(1995, 2022):
        dm_mat.array[idx["EU27"], idx[y], idx["material-demand"], idx["timber"]] = (
            np.nan
        )
    for y in range(1995, 2022):
        dm_mat.array[idx["EU27"], idx[y], idx["material-export"], idx["timber"]] = (
            np.nan
        )
    dm_mat.array[idx["EU27"], idx[2021], idx["material-export"], idx["timber"]] = (
        20000000000
    )

    # for wwp, before 2016 as nan for EU27 (same principle of timber)
    idx = dm_mat.idx
    for y in range(1995, 2016):
        dm_mat.array[idx["EU27"], idx[y], idx["material-demand"], idx["wwp"]] = np.nan
    for y in range(1995, 2016):
        dm_mat.array[idx["EU27"], idx[y], idx["material-export"], idx["wwp"]] = np.nan
    dm_mat.array[idx["EU27"], idx[2022], idx["material-export"], idx["wwp"]] = np.nan
    for y in range(1995, 2016):
        dm_mat.array[idx["EU27"], idx[y], idx["material-import"], idx["wwp"]] = np.nan
    dm_mat.array[idx["EU27"], idx[2022], idx["material-import"], idx["wwp"]] = np.nan

    # for tra equipment put missing before 2009
    idx = dm_mat.idx
    for y in range(1995, 2009):
        dm_mat.array[idx["EU27"], idx[y], :, idx["tra-equip"]] = np.nan

    # for ois put missing before 2010
    idx = dm_mat.idx
    for y in range(1995, 2010):
        dm_mat.array[idx["EU27"], idx[y], :, idx["ois"]] = np.nan

    # for other, put 2022-2023 as missing
    idx = dm_mat.idx
    for y in range(2022, 2023 + 1):
        dm_mat.array[idx["EU27"], idx[y], :, idx["other"]] = np.nan

    # check
    # dm_mat.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

    # fix jumps
    dm_mat = fix_jumps_in_dm(dm_mat)

    # check
    # dm_mat.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

    # # put nas for 2008 crisis when needed
    # idx = dm_mat.idx
    # for y in range(2007,2011+1):
    #     dm_mat.array[idx["EU27"],idx[y],:,idx["copper"]] = np.nan
    #     dm_mat.array[idx["EU27"],idx[y],:,idx["cement"]] = np.nan
    #     dm_mat.array[idx["EU27"],idx[y],:,idx["lime"]] = np.nan
    # for y in range(2007,2009+1):
    #     dm_mat.array[idx["EU27"],idx[y],:,idx["steel"]] = np.nan
    # dm_mat.array[idx["EU27"],idx[2018],:,idx["paper"]] = np.nan
    # dm_mat.array[idx["EU27"],idx[2008],:,idx["paper"]] = np.nan
    # dm_mat.array[idx["EU27"],idx[2022],:,idx["lime"]] = np.nan
    # dm_mat.array[idx["EU27"],idx[2023],:,idx["lime"]] = np.nan

    # flatten
    dm_mat = dm_mat.flatten()

    # check
    # dm_mat.filter({"Country" : ["EU27"]}).datamatrix_plot()

    # new variabs list
    dict_new = {}

    # function to adjust ots
    def make_ots(dm, variable, based_on):
        dm_temp = dm.filter({"Variables": [variable]})
        dm_temp = linear_fitting(
            dm_temp, years_ots, based_on=based_on, min_t0=0.1, min_tb=0.1
        )
        return dm_temp

    dict_call = {
        "material-demand_aluminium": None,
        "material-demand_ammonia": None,
        "material-demand_cement": None,
        "material-demand_chem": None,
        "material-demand_copper": None,
        "material-demand_fbt": None,
        "material-demand_glass": None,
        "material-demand_lime": None,
        "material-demand_mae": None,
        "material-demand_ois": None,
        "material-demand_other": range(2010, 2018 + 1),
        "material-demand_paper": range(2007, 2017 + 1),
        "material-demand_steel": range(2010, 2018 + 1),
        "material-demand_textiles": None,
        "material-demand_timber": None,
        "material-demand_tra-equip": None,
        "material-demand_wwp": None,
        "material-export_aluminium": None,
        "material-export_cement": range(2012, 2014 + 1),
        "material-export_chem": None,
        "material-export_copper": None,
        "material-export_fbt": None,
        "material-export_glass": None,
        "material-export_lime": range(2012, 2018 + 1),
        "material-export_mae": range(2007, 2017 + 1),
        "material-export_ois": range(2010, 2018 + 1),
        "material-export_other": None,
        "material-export_paper": None,
        "material-export_steel": None,
        "material-export_textiles": None,
        "material-export_timber": None,
        "material-export_tra-equip": None,
        "material-export_wwp": None,
        "material-import_aluminium": None,
        "material-import_cement": None,
        "material-import_chem": None,
        "material-import_copper": None,
        "material-import_fbt": None,
        "material-import_glass": None,
        "material-import_lime": range(2014, 2023 + 1),
        "material-import_mae": None,
        "material-import_ois": None,
        "material-import_other": None,
        "material-import_paper": None,
        "material-import_steel": None,
        "material-import_textiles": None,
        "material-import_timber": None,
        "material-import_tra-equip": None,
        "material-import_wwp": None,
    }

    for key in dict_call.keys():
        dict_new[key] = make_ots(dm_mat, key, based_on=dict_call[key])

    # append
    dm_mat_temp = dict_new["material-demand_aluminium"].copy()
    mylist = list(dict_call.keys())
    mylist.remove("material-demand_aluminium")
    for v in mylist:
        dm_mat_temp.append(dict_new[v], "Variables")
    dm_mat_temp.sort("Variables")
    dm_mat = dm_mat_temp.copy()
    dm_mat.deepen()

    # check
    # dm_mat_temp.filter({"Country" : ["EU27"]}).datamatrix_plot()

    # # fix jumps
    # dm_mat = fix_jumps_in_dm(dm_mat)

    # check
    # dm_mat_temp.filter({"Country" : ["EU27"]}).datamatrix_plot()

    # ####################
    # ##### MAKE FTS #####
    # ####################

    # # make function to fill in missing years fts for EU27 with linear fitting
    # def make_fts(
    #     dm,
    #     variable,
    #     year_start,
    #     year_end,
    #     country="EU27",
    #     dim="Categories1",
    #     min_t0=0,
    #     min_tb=0,
    #     years_fts=years_fts,
    # ):  # I put minimum to 1 so it does not go to zero
    #     dm = dm.copy()
    #     idx = dm.idx
    #     based_on_yars = list(range(year_start, year_end + 1, 1))
    #     dm_temp = linear_fitting(
    #         dm.filter({"Country": [country], dim: [variable]}),
    #         years_ots=years_fts,
    #         min_t0=min_t0,
    #         min_tb=min_tb,
    #         based_on=based_on_yars,
    #     )
    #     idx_temp = dm_temp.idx
    #     if dim == "Variables":
    #         dm.array[idx[country], :, idx[variable], ...] = np.round(
    #             dm_temp.array[idx_temp[country], :, idx_temp[variable], ...], 0
    #         )
    #     if dim == "Categories1":
    #         dm.array[idx[country], :, :, idx[variable]] = np.round(
    #             dm_temp.array[idx_temp[country], :, :, idx_temp[variable]], 0
    #         )
    #     if dim == "Categories2":
    #         dm.array[idx[country], :, :, :, idx[variable]] = np.round(
    #             dm_temp.array[idx_temp[country], :, :, :, idx_temp[variable]], 0
    #         )
    #     if dim == "Categories3":
    #         dm.array[idx[country], :, :, :, :, idx[variable]] = np.round(
    #             dm_temp.array[idx_temp[country], :, :, :, :, idx_temp[variable]], 0
    #         )

    #     return dm

    # # add missing years fts
    # dm_mat.add(np.nan, col_label=years_fts, dummy=True, dim="Years")

    # # set default time window for linear trend
    # # assumption: best is taking longer trend possible to make predictions to 2050 (even if earlier data is generated)
    # baseyear_start = 1990
    # baseyear_end = 2023

    # # fill in
    # dm_mat = make_fts(dm_mat, "aluminium", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "ammonia", baseyear_start, baseyear_end)
    # dm_mat = make_fts(
    #     dm_mat, "cement", 2014, 2023
    # )  # import on upward trend and export on downward trend since 2014 (demand predictions dont change much if we start from 2014)
    # dm_mat = make_fts(dm_mat, "chem", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "copper", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "fbt", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "glass", 2020, 2023)
    # dm_mat = make_fts(dm_mat, "lime", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "mae", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "ois", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "other", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "paper", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "steel", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "textiles", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "timber", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "tra-equip", baseyear_start, baseyear_end)
    # dm_mat = make_fts(dm_mat, "wwp", baseyear_start, baseyear_end)

    # # check
    # # dm_mat.filter({"Country" : ["EU27"]}).datamatrix_plot()

    ####################################
    ##### MAKE MATERIAL NET IMPORT #####
    ####################################

    # material-net-import[%] = (material-import - material-export)/material-demand

    # subset for main materials
    materials = [
        "aluminium",
        "ammonia",
        "cement",
        "chem",
        "copper",
        "glass",
        "lime",
        "other",
        "paper",
        "steel",
        "timber",
    ]
    dm_temp = dm_mat.filter({"Categories1": materials})

    # make material-net-import[%] = (material-import - material-export)/material-demand
    idx = dm_temp.idx
    arr_temp = dm_temp.array
    arr_net = (
        arr_temp[:, :, idx["material-import"], :]
        - arr_temp[:, :, idx["material-export"], :]
    ) / arr_temp[:, :, idx["material-demand"], :]

    # when both import and export are zero, assign a zero
    arr_net[
        (arr_temp[:, :, idx["material-import"], :] == 0)
        & (arr_temp[:, :, idx["material-export"], :] == 0)
    ] = 0
    dm_temp.add(
        arr_net[:, :, np.newaxis, :], "Variables", "material-net-import", unit="%"
    )

    # drop
    dm_temp.drop("Variables", ["material-import", "material-export", "material-demand"])

    # store
    dm_trade_netshare = dm_temp.copy()
    dm_trade_netshare.sort("Categories1")

    # fill in missing values for material-net-import (coming from dividing by zero)
    idx = dm_trade_netshare.idx
    dm_trade_netshare.array[dm_trade_netshare.array == np.inf] = np.nan
    years_fitting = dm_trade_netshare.col_labels["Years"]
    dm_trade_netshare = linear_fitting(dm_trade_netshare, years_fitting)

    # # fix jumps in material-net-import
    # dm_trade_netshare = fix_jumps_in_dm(dm_trade_netshare)

    # make ammonia as missing
    dm_trade_netshare.drop("Categories1", "ammonia")
    dm_trade_netshare.add(np.nan, col_label="ammonia", dummy=True, dim="Categories1")
    dm_trade_netshare.sort("Categories1")

    # let's cap everything to 1
    dm_trade_netshare.array[dm_trade_netshare.array > 1] = 1

    # check
    # dm_trade_netshare.filter({"Country" : ["EU27"]}).datamatrix_plot()

    ####################################
    ##### MAKE MATERIAL PRODUCTION #####
    ####################################

    # material-production[kg] = material-demand[kg] + material-export[kg] - material-import[kg]

    dm_temp = dm_mat.copy()

    # make material-production
    idx = dm_temp.idx
    arr_temp = dm_temp.array
    arr_net = (
        arr_temp[:, :, idx["material-demand"], :]
        + arr_temp[:, :, idx["material-export"], :]
        - arr_temp[:, :, idx["material-import"], :]
    )

    # assign zero when production is negative
    # material production < 0 when material import > demand + export
    # when this happens, I assume that material production is zero (a country that imports a lot to the point
    # that the material net import is larger than domestic demand)
    # whatever people do not consume of all this import, can be added to a measure of material stock in case
    arr_net[arr_net < 0] = 0

    # make dm with material production
    dm_temp.add(
        arr_net[:, :, np.newaxis, :], "Variables", "material-production", unit="kg"
    )
    dm_temp.drop("Variables", ["material-import", "material-export", "material-demand"])
    dm_matprod = dm_temp.copy()

    # # fix jumps in material-production
    # dm_matprod = fix_jumps_in_dm(dm_matprod)

    # # make ammonia as demand
    # idx = dm_mat.idx
    # arr_temp = dm_mat.array[:,:,idx["material-demand"],idx["ammonia"]]
    # dm_matprod.add(arr_temp[:,:,np.newaxis,np.newaxis], col_label="ammonia", dim='Categories1', unit="kg")
    # dm_matprod.sort("Categories1")

    # make it in kilo tonnes
    dm_matprod.array = dm_matprod.array / 1000000
    dm_matprod.units["material-production"] = "kt"

    # make fxa for non-modelled sectors
    dm_matprod_fxa = dm_matprod.filter(
        {"Categories1": ["fbt", "mae", "ois", "textiles", "tra-equip", "wwp"]}
    )
    dm_matprod_fxa = linear_fitting(dm_matprod_fxa, years_fts)

    # # make calibration data
    # dm_matprod_calib = dm_matprod.filter({"Years" : years_ots})
    # years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
    # missing = np.array(years)[[y not in dm_matprod_calib.col_labels["Years"] for y in years]].tolist()
    # dm_matprod_calib.add(np.nan, "Years", missing, dummy=True)
    # dm_matprod_calib.sort("Years")

    # check
    # dm_matprod_fxa.filter({"Country" : ["EU27"]}).datamatrix_plot()
    # dm_matprod_calib.filter({"Country" : ["EU27"]}).datamatrix_plot()

    # #########################################################
    # ##### MAKE CALIBRATION DATA FOR MATERIAL PRODUCTION #####
    # #########################################################

    # # TODO: need to change this data with data from material flow analysis

    # dm_matprod_calib = get_prodcom_data("ds-056121")
    # dm_matprod_calib.rename_col("material-demand", "material-production", "Variables")
    # dm_matprod_calib.change_unit("material-production", 1e-6, "kg", "kt")
    # materials = dm_matprod.col_labels["Categories1"]
    # current_materials = dm_matprod_calib.col_labels["Categories1"]
    # for m in materials:
    #     if m not in current_materials:
    #         dm_matprod_calib.add(np.nan, "Categories1", m, dummy=True)
    # dm_matprod_calib.sort("Categories1")
    # dm_matprod_calib.drop("Years", 2024)
    # current_years = dm_matprod_calib.col_labels["Years"]
    # for y in years_ots + years_fts:
    #     if y not in current_years:
    #         dm_matprod_calib.add(np.nan, "Years", [y], dummy=True)
    # dm_matprod_calib.sort("Years")

    # dm_matprod_calib.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
    # df_temp = dm_matprod_calib.filter({"Country" : ["EU27"],"Years" : [2023]}).write_df()
    # df_temp = df_temp.melt(["Country","Years"])
    # df_temp
    # note: it could be that aluminium data is too high, but also after filtering for non primary materials it stays the same, so not sure how to change it further
    # also other probably has an issue (spike post 2022), but not too sure how to deal with it
    # in general, material production data seems bad, and not sure if calibrating on it is a good idea

    #######################################
    ##### MAKE MATERIAL DEMAND OF WWP #####
    #######################################

    dm_temp = dm_mat.filter({"Variables": ["material-demand"], "Categories1": ["wwp"]})
    dm_temp.change_unit("material-demand", factor=1e-3, old_unit="kg", new_unit="t")
    dm_matdem_fxa = dm_temp.copy()
    dm_matdem_fxa = linear_fitting(dm_matdem_fxa, years_fts)

    ################
    ##### SAVE #####
    ################

    # # lever: trade net share
    # years_ots = list(range(1990, 2023 + 1))
    # years_fts = list(range(2025, 2055, 5))
    # dm_ots = dm_trade_netshare.filter({"Years": years_ots})
    # dm_fts = dm_trade_netshare.filter({"Years": years_fts})
    # DM_fts = {
    #     1: dm_fts.copy(),
    #     2: dm_fts.copy(),
    #     3: dm_fts.copy(),
    #     4: dm_fts.copy(),
    # }  # for now we set all levels to be the same
    # DM = {"ots": dm_ots, "fts": DM_fts}
    f = os.path.join(
        current_file_directory, "../data/datamatrix/lever_material-net-import.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(dm_trade_netshare, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # fxa material production
    f = os.path.join(
        current_file_directory, "../data/datamatrix/fxa_material-production.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(dm_matprod_fxa, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # fxa material demand
    f = os.path.join(
        current_file_directory, "../data/datamatrix/fxa_material-demand.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(dm_matdem_fxa, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # calib material production
    # f = os.path.join(
    #     current_file_directory, "../data/datamatrix/calibration_material-production.pickle"
    # )
    # with open(f, "wb") as handle:
    #     pickle.dump(dm_matprod_calib, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_trade_netshare, dm_matprod_fxa, dm_matdem_fxa


def run(years_ots, years_fts):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_files = [
        "lever_material-net-import.pickle",
        "fxa_material-production.pickle",
        "fxa_material-demand.pickle",
    ]
    filepaths = [
        os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
        for lever_file in lever_files
    ]
    true_condition = all([os.path.exists(filepath) for filepath in filepaths])
    if true_condition:
        with open(filepaths[0], "rb") as handle:
            dm_trade_netshare = pickle.load(handle)
        with open(filepaths[1], "rb") as handle:
            dm_matprod_fxa = pickle.load(handle)
        with open(filepaths[2], "rb") as handle:
            dm_matdem_fxa = pickle.load(handle)
    else:
        dm_trade_netshare, dm_matprod_fxa, dm_matdem_fxa = make_material_net_import_dm(
            current_file_directory, years_ots, years_fts
        )

    return dm_trade_netshare, dm_matprod_fxa, dm_matdem_fxa


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)
