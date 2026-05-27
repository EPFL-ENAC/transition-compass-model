import os
import pickle

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def make_calib_material_production_dms(
    current_file_directory, lever_files, years_ots, years_fts
):
    countries = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Poland",
        "Portugal",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "EU27",
    ]
    years = years_ots + years_fts
    materials_industry = [
        "aluminium",
        "cement",
        "chem",
        "copper",
        "fbt",
        "glass",
        "lime",
        "mae",
        "ois",
        "other",
        "paper",
        "steel",
        "textiles",
        "timber",
        "tra-equip",
        "wwp",
    ]
    materials_ammonia = ["ammonia"]

    def make_nan_dm(countries, years, materials):
        dm = DataMatrix(
            col_labels={
                "Country": countries,
                "Years": years,
                "Variables": ["material-production"],
                "Categories1": materials,
            },
            units={"material-production": "kt"},
            empty=True,
        )
        dm.array = np.full((len(countries), len(years), 1, len(materials)), np.nan)
        return dm

    dm_industry = make_nan_dm(countries, years, materials_industry)
    dm_ammonia = make_nan_dm(countries, years, materials_ammonia)

    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_files[0])
    with open(f, "wb") as handle:
        pickle.dump(dm_industry, handle, protocol=pickle.HIGHEST_PROTOCOL)

    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_files[1])
    with open(f, "wb") as handle:
        pickle.dump(dm_ammonia, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_industry, dm_ammonia


def run(years_ots, years_fts):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    lever_files = [
        "calibration_material-production.pickle",
        "calibration_material-production_ammonia.pickle",
    ]
    filepaths = [
        os.path.join(current_file_directory, "../data/datamatrix/" + f)
        for f in lever_files
    ]
    if all(os.path.exists(p) for p in filepaths):
        with open(filepaths[0], "rb") as handle:
            dm_industry = pickle.load(handle)
        with open(filepaths[1], "rb") as handle:
            dm_ammonia = pickle.load(handle)
    else:
        dm_industry, dm_ammonia = make_calib_material_production_dms(
            current_file_directory, lever_files, years_ots, years_fts
        )

    return dm_industry, dm_ammonia


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)
