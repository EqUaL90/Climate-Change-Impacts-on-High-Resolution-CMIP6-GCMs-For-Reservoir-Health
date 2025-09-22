from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    r, _ = pearsonr(y_true, y_pred)
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true)

    return {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'Bias': round(bias, 2),
        'r': round(r, 2),
        'PBIAS (%)': round(pbias, 2)
    }

# Define model names
model_names = ['ACCESS-CM2', 'CanESM5', 'GFDL-ESM4']

results_eragg = []
results_era = []

for org, eragg, era in zip(ds_base_Pr_Org, ds_base_Pr_ERAGg_Corr, ds_base_Pr_ERA_Corr):
    var_org = 'pr'
    var_corr = 'Precip'

    # Mean over lat-lon for time series
    org_mean = org[var_org].mean(dim=('latitude', 'longitude'), skipna=True)
    eragg_mean = eragg[var_corr].mean(dim=('latitude', 'longitude'), skipna=True)
    era_mean = era[var_corr].mean(dim=('latitude', 'longitude'), skipna=True)

    # Align all datasets to ensure they match in time
    org_mean, eragg_mean = xr.align(org_mean, eragg_mean, join='inner')
    org_mean, era_mean = xr.align(org_mean, era_mean, join='inner')

    # Apply masking for NaNs
    mask_eragg = ~np.isnan(org_mean) & ~np.isnan(eragg_mean)
    mask_era = ~np.isnan(org_mean) & ~np.isnan(era_mean)

    # Compute metrics
    metrics_eragg = compute_metrics(org_mean.values[mask_eragg], eragg_mean.values[mask_eragg])
    metrics_era = compute_metrics(org_mean.values[mask_era], era_mean.values[mask_era])

    results_eragg.append(metrics_eragg)
    results_era.append(metrics_era)

# Convert to DataFrame
Base_ERAGg_Prec_TrainMets = pd.DataFrame(results_eragg)
Base_ERAGg_Prec_TrainMets['GCMs'] = model_names
Base_ERAGg_Prec_TrainMets = Base_ERAGg_Prec_TrainMets.set_index('GCMs').T

Base_ERAOnly_Prec_TrainMets = pd.DataFrame(results_era)
Base_ERAOnly_Prec_TrainMets['GCMs'] = model_names
Base_ERAOnly_Prec_TrainMets = Base_ERAOnly_Prec_TrainMets.set_index('GCMs').T

print("Metrics for ERAGg vs Original:")
print(Base_ERAGg_Prec_TrainMets)

print("\nMetrics for ERA vs Original:")
print(Base_ERAOnly_Prec_TrainMets)
#Base_ERAGg_Prec_TrainMets.to_excel(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\ERA or ERAGg\1_Base_ERAGg_Prec_TrainMets.xlsx")
#Base_ERAOnly_Prec_TrainMets.to_excel(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\ERA or ERAGg\2_Base_ERAOnly_Prec_TrainMets.xlsx")
print("Saved")
