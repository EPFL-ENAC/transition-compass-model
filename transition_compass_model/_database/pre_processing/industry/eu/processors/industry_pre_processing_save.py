import os
import pickle


def save_industry_pre_processing_run(DM_input):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # save
    f = os.path.join(
        current_file_directory, "../data/datamatrix/industry_pre_processing.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(DM_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Ammonia

    # dm_trade_netshare_prod_amm = DM_input[""].filter({"Categories1" : ["fertilizer"]})
    # DM_input[""].drop("Categories1", "fertilizer")

    # dm_trade_netshare_amm = dm_trade_netshare.filter({"Categories1" : ["ammonia"]})
    # dm_trade_netshare.drop("Categories1", "ammonia")

    # dm_mat_eff_amm = dm_mat_eff.filter({"Categories1" : ["ammonia"]})
    # dm_mat_eff.drop("Categories1", "ammonia")

    # dm_material_recovery_amm = dm_material_recovery.filter({"Categories2" : ["ammonia"]})
    # dm_material_recovery.drop("Categories1", "ammonia")

    # dm_tech_dev_amm = dm_tech_dev.filter({"Categories1" : ["ammonia-tech"]})
    # dm_tech_dev.drop("Categories1", "ammonia-tech")

    # dm_cc_amm = dm_cc.filter({"Categories1" : ["ammonia-tech"]})
    # dm_cc.drop("Categories1", "ammonia-tech")

    # dm_ene_switch_amm = dm_ene_switch.filter({"Categories1" : ["ammonia-tech"]})
    # dm_ene_switch.drop("Categories1", "ammonia-tech")

    return
