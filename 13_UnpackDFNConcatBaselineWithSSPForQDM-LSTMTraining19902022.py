# Unpacking the Dataframes
# Define all your original variable name lists (as strings)
name_lists = {
    "DFds_base_Pr_Org": ["ACCESS_CM2histMnthPrec1990_2014_regriddedF", "CanESM5histMnthPrec1990_2014_regriddedF", "GFDL_ESM4histMnthPrec1990_2014_regriddedF"],
    "DFds_base_Pr_ERA_Corr": ["HistACCPrcERAcorr", "HistCanPrcERAcorr", "HistGFDLPrcERAcorr"],
    "DFds_Pr_Org": ["SelFutPrecAcc_SSP126_Rgdd", "SelFutPrecAcc_SSP245_Rgdd", "SelFutPrecAcc_SSP585_Rgdd",
                    "SelFutPrecCan_SSP126_Rgdd", "SelFutPrecCan_SSP245_Rgdd", "SelFutPrecCan_SSP585_Rgdd",
                    "SelFutPrecGFDL_SSP126_Rgdd", "SelFutPrecGFDL_SSP245_Rgdd", "SelFutPrecGFDL_SSP585_Rgdd"],
    "DFds_Pr_ERA_Corr": ["Tr_ERA_SSP126AccPrec_corr", "Tr_ERA_SSP245AccPrec_corr", "Tr_ERA_SSP585AccPrec_corr",
                         "Tr_ERA_SSP126CanPrc_corr", "Tr_ERA_SSP245CanPrc_corr", "Tr_ERA_SSP585CanPrc_corr",
                         "Tr_ERA_SSP126GFDLPrc_corr", "Tr_ERA_SSP245GFDLPrc_corr", "Tr_ERA_SSP585GFDLPrc_corr"],
    "DFds_Pr_target": ["AccPrcSSP126_26_80", "AccPrcSSP245_26_80", "AccPrcSSP585_26_80",
                       "CanSSP126_26_80", "CanSSP245_26_80", "CanSSP585_26_80",
                       "GFDLSSP126_26_80", "GFDLSSP245_26_80", "GFDLSSP585_26_80"],

    "DFds_base_Tp_Org": ["ACCESS_CM2histMnthTemp1990_2014_regriddedF", "CanESM5histMnthTemp1990_2014_regriddedF", "GFDL_ESM4histMnthTemp1990_2014_regriddedF"],
    "DFds_base_Tp_ERA_Corr": ["HistACCTmpERAcorr", "HistCanTmpERAcorr", "HistGFDLTmpERAcorr"],
    "DFds_Tp_Org": ["SelFutTempAcc_SSP126_Rgdd", "SelFutTempAcc_SSP245_Rgdd", "SelFutTempAcc_SSP585_Rgdd",
                    "SelFutTempCan_SSP126_Rgdd", "SelFutTempCan_SSP245_Rgdd", "SelFutTempCan_SSP585_Rgdd",
                    "SelFutTempGFDL_SSP126_Rgdd", "SelFutTempGFDL_SSP245_Rgdd", "SelFutTempGFDL_SSP585_Rgdd"],
    "DFds_Tp_ERA_Corr": ["Tr_ERA_SSP126AccTmp_corr", "Tr_ERA_SSP245AccTmp_corr", "Tr_ERA_SSP585AccTmp_corr",
                         "Tr_ERA_SSP126CanTmp_corr", "Tr_ERA_SSP245CanTmp_corr", "Tr_ERA_SSP585CanTmp_corr",
                         "Tr_ERA_SSP126GFDLTmp_corr", "Tr_ERA_SSP245GFDLTmp_corr", "Tr_ERA_SSP585GFDLTmp_corr"],
    "DFds_Tp_target": ["AccTmpSSP126_26_80", "AccTmpSSP245_26_80", "AccTmpSSP585_26_80",
                       "CanTmpSSP126_26_80", "CanTmpSSP245_26_80", "CanTmpSSP585_26_80",
                       "GFDLTmpSSP126_26_80", "GFDLTmpSSP245_26_80", "GFDLTmpSSP585_26_80"]
}

# Now assign each element in the DFds_* lists to a new variable name with prefix DF
for df_list_name, var_names in name_lists.items():
    df_list = globals().get(df_list_name)
    if df_list is not None:
        for name, df in zip(var_names, df_list):
            globals()["DF" + name] = df


  #Concatenate hist and Sel to make a full training data for the LSTM
import pandas as pd

# Labels
models = ['Acc', 'Can', 'GFDL']
ssps = ['SSP126', 'SSP245', 'SSP585']

# ============================ #
#   Precipitation Datasets     #
# ============================ #

# Historical base
DFds_base_Pr_Org = [
    DFACCESS_CM2histMnthPrec1990_2014_regriddedF,
    DFCanESM5histMnthPrec1990_2014_regriddedF,
    DFGFDL_ESM4histMnthPrec1990_2014_regriddedF
]

DFds_base_Pr_ERA_Corr = [
    DFHistACCPrcERAcorr, DFHistCanPrcERAcorr, DFHistGFDLPrcERAcorr
]

# Future
DFds_Pr_Org = [
    DFSelFutPrecAcc_SSP126_Rgdd, DFSelFutPrecAcc_SSP245_Rgdd, DFSelFutPrecAcc_SSP585_Rgdd,
    DFSelFutPrecCan_SSP126_Rgdd, DFSelFutPrecCan_SSP245_Rgdd, DFSelFutPrecCan_SSP585_Rgdd,
    DFSelFutPrecGFDL_SSP126_Rgdd, DFSelFutPrecGFDL_SSP245_Rgdd, DFSelFutPrecGFDL_SSP585_Rgdd
]

DFds_Pr_ERA_Corr = [
    DFTr_ERA_SSP126AccPrec_corr, DFTr_ERA_SSP245AccPrec_corr, DFTr_ERA_SSP585AccPrec_corr,
    DFTr_ERA_SSP126CanPrc_corr, DFTr_ERA_SSP245CanPrc_corr, DFTr_ERA_SSP585CanPrc_corr,
    DFTr_ERA_SSP126GFDLPrc_corr, DFTr_ERA_SSP245GFDLPrc_corr, DFTr_ERA_SSP585GFDLPrc_corr
]

# ============================ #
#     Temperature Datasets     #
# ============================ #

DFds_base_Tp_Org = [
    DFACCESS_CM2histMnthTemp1990_2014_regriddedF,
    DFCanESM5histMnthTemp1990_2014_regriddedF,
    DFGFDL_ESM4histMnthTemp1990_2014_regriddedF
]

DFds_base_Tp_ERA_Corr = [
    DFHistACCTmpERAcorr, DFHistCanTmpERAcorr, DFHistGFDLTmpERAcorr
]

DFds_Tp_Org = [
    DFSelFutTempAcc_SSP126_Rgdd, DFSelFutTempAcc_SSP245_Rgdd, DFSelFutTempAcc_SSP585_Rgdd,
    DFSelFutTempCan_SSP126_Rgdd, DFSelFutTempCan_SSP245_Rgdd, DFSelFutTempCan_SSP585_Rgdd,
    DFSelFutTempGFDL_SSP126_Rgdd, DFSelFutTempGFDL_SSP245_Rgdd, DFSelFutTempGFDL_SSP585_Rgdd
]

DFds_Tp_ERA_Corr = [
    DFTr_ERA_SSP126AccTmp_corr, DFTr_ERA_SSP245AccTmp_corr, DFTr_ERA_SSP585AccTmp_corr,
    DFTr_ERA_SSP126CanTmp_corr, DFTr_ERA_SSP245CanTmp_corr, DFTr_ERA_SSP585CanTmp_corr,
    DFTr_ERA_SSP126GFDLTmp_corr, DFTr_ERA_SSP245GFDLTmp_corr, DFTr_ERA_SSP585GFDLTmp_corr
]

# =================================== #
#        General Concatenation        # Only for 1990 - 2022 data
# =================================== #

def concatenate_model_runs(base_list, future_list, label_prefix, variable):
    """
    Concatenates base and future DataFrames based on model and SSP.

    Args:
        base_list (list): List of 3 historical DataFrames.
        future_list (list): List of 9 future DataFrames.
        label_prefix (str): "Org" or "ERA"
        variable (str): "Pr" or "Tp"

    Returns:
        dict: Dictionary of concatenated DataFrames.
    """
    concatenated = {}
    for i, model in enumerate(models):
        base_df = base_list[i]
        for j, ssp in enumerate(ssps):
            idx = i * 3 + j
            future_df = future_list[idx]
            combined_df = pd.concat([base_df.copy(), future_df])
            key = f"DF_{label_prefix}_{model}_{ssp}_{variable}_90_22"
            concatenated[key] = combined_df
    return concatenated

Org_combined_Pr = concatenate_model_runs(DFds_base_Pr_Org, DFds_Pr_Org, "Org", "Pr")  
ERA_combined_Pr = concatenate_model_runs(DFds_base_Pr_ERA_Corr, DFds_Pr_ERA_Corr, "ERA", "Pr")

Org_combined_Tp = concatenate_model_runs(DFds_base_Tp_Org, DFds_Tp_Org, "Org", "Tp")
ERA_combined_Tp = concatenate_model_runs(DFds_base_Tp_ERA_Corr, DFds_Tp_ERA_Corr, "ERA", "Tp")

all_combined_climate = {
    **Org_combined_Pr, **ERA_combined_Pr,
    **Org_combined_Tp, **ERA_combined_Tp
}

for var_name, df in all_combined_climate.items():
    globals()[var_name] = df

DFds_Pr_Org = [
    DFSelFutPrecAcc_SSP126_Rgdd, DFSelFutPrecAcc_SSP245_Rgdd, DFSelFutPrecAcc_SSP585_Rgdd,
    DFSelFutPrecCan_SSP126_Rgdd, DFSelFutPrecCan_SSP245_Rgdd, DFSelFutPrecCan_SSP585_Rgdd,
    DFSelFutPrecGFDL_SSP126_Rgdd, DFSelFutPrecGFDL_SSP245_Rgdd, DFSelFutPrecGFDL_SSP585_Rgdd
]

DFds_Pr_ERA_Corr = [
    DFTr_ERA_SSP126AccPrec_corr, DFTr_ERA_SSP245AccPrec_corr, DFTr_ERA_SSP585AccPrec_corr,
    DFTr_ERA_SSP126CanPrc_corr, DFTr_ERA_SSP245CanPrc_corr, DFTr_ERA_SSP585CanPrc_corr,
    DFTr_ERA_SSP126GFDLPrc_corr, DFTr_ERA_SSP245GFDLPrc_corr, DFTr_ERA_SSP585GFDLPrc_corr
]

# For Sel 2015 - 2022
DFSel_Pr_Org_named = {name: df for name, df in zip(name_lists["DFds_Pr_Org"], DFds_Pr_Org)}
DFSel_Pr_ERA_Corr_named = {name: df for name, df in zip(name_lists["DFds_Pr_ERA_Corr"], DFds_Pr_ERA_Corr)}

DFSel_Tp_Org_named = {name: df for name, df in zip(name_lists["DFds_Tp_Org"], DFds_Tp_Org)}
DFSel_Tp_ERA_Corr_named = {name: df for name, df in zip(name_lists["DFds_Tp_ERA_Corr"], DFds_Tp_ERA_Corr)}

#Name Future target too
# Create a dictionary with named DataFrames for future access
DF_Pr_target_named = {name: df for name, df in zip(name_lists["DFds_Pr_target"], DFds_Pr_target)}
DF_Tp_target_named = {name: df for name, df in zip(name_lists["DFds_Tp_target"], DFds_Tp_target)}

import pickle

# Dictionary of DataFrames to save
df_dicts = {
    "Org_combined_Pr": Org_combined_Pr,
    "ERA_combined_Pr": ERA_combined_Pr,
    "Org_combined_Tp": Org_combined_Tp,
    "ERA_combined_Tp": ERA_combined_Tp,
    "DFSel_Pr_Org_named": DFSel_Pr_Org_named,
    "DFSel_Pr_ERA_Corr_named": DFSel_Pr_ERA_Corr_named,
    "DFSel_Tp_Org_named": DFSel_Tp_Org_named,
    "DFSel_Tp_ERA_Corr_named": DFSel_Tp_ERA_Corr_named,
    "DF_Pr_target_named": DF_Pr_target_named
}

# Save to disk
with open(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputDictionaryPickles\all_processed_named_ERA_only_df_dicts.pkl", "wb") as f:
    pickle.dump(df_dicts, f)

print("✅ Saved as 'all_processed_named_ERA_only_df_dicts.pkl'")


#To load
#with open("all_processed_named_ERA_only_df_dicts.pkl'", "rb") as f:
#    loaded_df_dicts = pickle.load(f)

# Access like:
# loaded_df_dicts["Org_combined_Pr"]["Model1"]
