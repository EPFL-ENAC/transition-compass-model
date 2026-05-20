# packages
import os
import pickle
import warnings

warnings.simplefilter("ignore")
import numpy as np

from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)


def make_material_switch_dm(current_file_directory, lever_file):
    # make constant
    names = [
        "tec_material-switch-ratios_cement-to-timber",
        "tec_material-switch-ratios_chem-to-natfibers",
        "tec_material-switch-ratios_chem-to-paper",
        "tec_material-switch-ratios_steel-to-aluminium",
        "tec_material-switch-ratios_steel-to-chem",
        "tec_material-switch-ratios_steel-to-timber",
    ]
    names = [i + "[kg/kg]" for i in names]
    values = [3.87, 1.0, 1.0, 0.55, 0.4, 1.04]
    units = np.repeat("kg/kg", len(values)).tolist()
    const = {
        "name": names,
        "value": values,
        "idx": dict(zip(names, range(len(values)))),
        "units": dict(zip(names, units)),
    }

    # make cdm
    cdm = ConstantDataMatrix.create_from_constant(const, 0)

    # rename
    cdm.rename_col_regex("tec_", "", "Variables")

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    with open(f, "wb") as handle:
        pickle.dump(cdm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cdm


def run():
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_file = "const_material-switch-ratios.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            cdm = pickle.load(handle)
    else:
        cdm = make_material_switch_dm(current_file_directory, lever_file)

    return cdm
