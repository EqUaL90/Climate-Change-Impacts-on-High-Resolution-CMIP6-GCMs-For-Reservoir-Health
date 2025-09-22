missing_fids = [15, 72, 167, 183, 257, 258]

# Create masks
missing_data_gdf = b[b['FID'].isin(missing_fids)]
valid_data_gdf = b[~b['FID'].isin(missing_fids)]

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
valid_data_gdf.plot(ax=ax, color='blue', markersize=20, label='Valid Stations')
missing_data_gdf.plot(ax=ax, color='red', markersize=30, label='Stations with Missing Data')

# Styling
ax.set_title("Grid Points and Missing Data Stations", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(fontsize = 10)
ax.grid(False)

plt.show()


missing_station = ["P15", "P72", "P167", "P183", "P257", "P258"]
replace_station = ["P3", "P73", "P168", "P182", "P229", "P229"]
missing_to_replace = dict(zip(missing_station, replace_station))
def impute_dfs(dirtydata, mapping):
    imputed_dict = {}
    for key, df in dirtydata.items():
        df_copy = df.copy()
        if key in mapping:
            replace_key = mapping[key]
            if replace_key in dirtydata:
                # Ensure the replacement column doesn't have NaNs
                df_copy[df_copy.isna()] = dirtydata[replace_key][df_copy.isna()]
        
        # Fill remaining NaNs using forward-fill or backward-fill
        df_copy = df_copy.ffill().bfill()
        
        # Optional: Drop columns that are still fully NaN after imputation
        df_copy = df_copy.dropna(axis=1, how='all')
        
        # Convert to numeric values (coercing errors to NaN)
        df_copy = df_copy.apply(pd.to_numeric, errors='coerce')
        
        imputed_dict[key] = df_copy
    return imputed_dict


Org_combined_Pr_imputD = impute_dfs(Org_combined_Pr, missing_to_replace)
ERA_combined_Pr_imputD = impute_dfs(ERA_combined_Pr, missing_to_replace)
Org_combined_Tp_imputD = impute_dfs(Org_combined_Tp, missing_to_replace)
ERA_combined_Tp_imputD = impute_dfs(ERA_combined_Tp, missing_to_replace)
DF_Pr_target_named_imputD = impute_dfs(DF_Pr_target_named, missing_to_replace)
DF_Tp_target_named_imputD = impute_dfs(DF_Tp_target_named, missing_to_replace)

print("✅ ERA NaNs after imputation:", ERA_combined_Pr_imputD['DF_ERA_GFDL_SSP126_Pr_90_22'].isnull().sum())
print("✅ ORG NaNs after imputation:", Org_combined_Pr_imputD['DF_Org_Can_SSP245_Pr_90_22'].isnull().sum())

#Imput Sel
DFSel_Pr_Org_imputD = impute_dfs(DFSel_Pr_Org_named, missing_to_replace)
DFSel_Pr_ERA_Corr_imputD = impute_dfs(DFSel_Pr_ERA_Corr_named, missing_to_replace)

DFSel_Tp_Org_imputD = impute_dfs(DFSel_Tp_Org_named, missing_to_replace)
DFSel_Tp_ERA_Corr_imputD = impute_dfs(DFSel_Tp_ERA_Corr_named, missing_to_replace
