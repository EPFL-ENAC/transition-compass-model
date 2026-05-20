# packages
import os
import pickle
import warnings

warnings.simplefilter("ignore")

from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_const_emission_factors import (
    get_emission_factors_data,
    get_process_emission_factors_data,
)
from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)


def make_const_emission_factors_dm(current_file_directory, lever_file):
    ###################################################################################
    ############################### COMBUSTION EMISSIONS ##############################
    ###################################################################################

    # get data
    df = get_emission_factors_data(current_file_directory)

    # make cdm
    def create_constant(df, variables):
        df_temp = df.loc[df["variable"].isin(variables), :]

        # put unit
        df_temp["unit"] = [i.split("[")[1].split("]")[0] for i in df_temp["variable"]]

        const = {
            "name": list(df_temp["variable"]),
            "value": list(df_temp["value"]),
            "idx": dict(
                zip(list(df_temp["variable"]), range(len(df_temp["variable"])))
            ),
            "units": dict(zip(list(df_temp["variable"]), list(df_temp["unit"]))),
        }

        # return
        return const

    tmp = create_constant(df, df["variable"])
    cdm = ConstantDataMatrix.create_from_constant(tmp, 0)
    cdm.deepen()

    # store
    CDM_emissions = {"combustion-emissions": cdm}

    # df = cdm.write_df()
    # df["country"] = "all"
    # df_temp = pd.melt(df, id_vars = ['country'], var_name='variable')
    # name = "temp.xlsx"
    # df_temp.to_excel("~/Desktop/" + name)

    ###################################################################################
    ################################ PROCESS EMISSIONS ################################
    ###################################################################################

    # get data
    df = get_process_emission_factors_data(current_file_directory)

    ###########################################################
    ############# CONVERT TO CONSTANT DATA MATRIX #############
    ###########################################################

    # create dms
    def create_constant(df, variables):
        df_temp = df.loc[df["variable"].isin(variables), :]

        # put unit
        df_temp["unit"] = [i.split("[")[1].split("]")[0] for i in df_temp["variable"]]

        const = {
            "name": list(df_temp["variable"]),
            "value": list(df_temp["value"]),
            "idx": dict(
                zip(list(df_temp["variable"]), range(len(df_temp["variable"])))
            ),
            "units": dict(zip(list(df_temp["variable"]), list(df_temp["unit"]))),
        }

        # return
        return const

    # excluding feedstock
    tmp = create_constant(df, df["variable"])
    cdm = ConstantDataMatrix.create_from_constant(tmp, 0)
    cdm.deepen_twice()

    # store
    CDM_emissions["process-emissions"] = cdm

    # save
    f = os.path.join(
        current_file_directory, "../data/datamatrix/const_emissions-factors.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(CDM_emissions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cdm_combustion_emission_factors = CDM_emissions["combustion-emissions"].copy()
    cdm_process_emission_factors = CDM_emissions["process-emissions"].copy()

    return cdm_combustion_emission_factors, cdm_process_emission_factors


def run():
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_file = "const_emissions-factors.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            CDM = pickle.load(handle)
            dm_combustion_emission_factors = CDM["combustion-emissions"].copy()
            dm_process_emission_factors = CDM["process-emissions"].copy()
    else:
        dm_combustion_emission_factors, dm_process_emission_factors = (
            make_const_emission_factors_dm(current_file_directory, lever_file)
        )

    return dm_combustion_emission_factors, dm_process_emission_factors


if __name__ == "__main__":
    run()


# df = cdm.write_df()
# df["country"] = "all"
# df_temp = pd.melt(df, id_vars = ['country'], var_name='variable')
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)
