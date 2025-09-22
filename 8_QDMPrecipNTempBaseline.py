from scipy.stats import percentileofscore, scoreatpercentile
def QDM_Prc(hist_model_ds, hist_obs_ds, future_model_ds, var_model='pr', var_obs='Precip'):
    hist_model = hist_model_ds[var_model]
    hist_obs = hist_obs_ds[var_obs]
    future_model = future_model_ds[var_model]

    # Match shapes
    assert hist_model.shape[1:] == hist_obs.shape[1:] == future_model.shape[1:]

    corrected_future = np.full_like(future_model, np.nan)

    # Loop over each grid cell (lat, lon)
    for lat_idx in range(hist_model.shape[1]):
        for lon_idx in range(hist_model.shape[2]):
            model_hist_series = hist_model[:, lat_idx, lon_idx].values
            obs_hist_series = hist_obs[:, lat_idx, lon_idx].values
            model_future_series = future_model[:, lat_idx, lon_idx].values

            # Skip NaN cells
            if np.all(np.isnan(model_hist_series)) or np.all(np.isnan(obs_hist_series)):
                continue

            model_hist_sorted = np.sort(model_hist_series)
            obs_hist_sorted = np.sort(obs_hist_series)

            # Percentile mapping
            cdf_model_future = np.array([
                percentileofscore(model_hist_series, x, kind='mean') for x in model_future_series
            ]) / 100

            quantiles_obs = np.array([
                scoreatpercentile(obs_hist_sorted, p * 100) for p in cdf_model_future
            ])
            quantiles_model_hist = np.array([
                scoreatpercentile(model_hist_sorted, p * 100) for p in cdf_model_future
            ])
            delta = model_future_series - quantiles_model_hist
            corrected = quantiles_obs + delta
            corrected_future[:, lat_idx, lon_idx] = corrected

    corrected_da = xr.DataArray(
        corrected_future,
        dims=future_model.dims,
        coords=future_model.coords,
        name=f"{var_model}_QDM_corrected"
    )

    return corrected_da

# Bias correction using your QDM function
HistACCPrcM_ERAGag_corrNC = QDM_Prc(
    hist_model_ds=ACCESS_CM2histMnthPrec1990_2014_regriddedF,
    hist_obs_ds=HistNPrcM_ERAGag_corr,
    future_model_ds=ACCESS_CM2histMnthPrec1990_2014_regriddedF,
    var_model='pr',
    var_obs='Precip'
)

HistACCPrcM_ERAGag_corrNC['time'] = HistACCPrcM_ERAGag_corrNC['time'].dt.floor('D')
HistACCPrcM_ERAGag_corrNC = HistACCPrcM_ERAGag_corrNC.to_dataset(name='Precip')

HistCanPrcM_ERAGag_corrNC = QDM_Prc(
    hist_model_ds=CanESM5histMnthPrec1990_2014_regriddedF,
    hist_obs_ds=HistNPrcM_ERAGag_corr,
    future_model_ds=CanESM5histMnthPrec1990_2014_regriddedF,
    var_model='pr',
    var_obs='Precip'
)

HistCanPrcM_ERAGag_corrNC ['time'] = HistCanPrcM_ERAGag_corrNC ['time'].dt.floor('D')
HistCanPrcM_ERAGag_corrNC = HistCanPrcM_ERAGag_corrNC .to_dataset(name='Precip')

HistGFDLPrcM_ERAGag_corrNC = QDM_Prc(
    hist_model_ds=GFDL_ESM4histMnthPrec1990_2014_regriddedF,
    hist_obs_ds=HistNPrcM_ERAGag_corr,
    future_model_ds=GFDL_ESM4histMnthPrec1990_2014_regriddedF ,
    var_model='pr',
    var_obs='Precip'
)

HistGFDLPrcM_ERAGag_corrNC['time'] = HistGFDLPrcM_ERAGag_corrNC['time'].dt.floor('D')
HistGFDLPrcM_ERAGag_corrNC= HistGFDLPrcM_ERAGag_corrNC.to_dataset(name='Precip')

print("############################### Precip QDM Done #############################"
from scipy.stats import percentileofscore, scoreatpercentile
def QDM_Tmp(hist_model_ds, hist_obs_ds, future_model_ds, var_model='tas', var_obs='Temp'):
    hist_model = hist_model_ds[var_model]
    hist_obs = hist_obs_ds[var_obs]
    future_model = future_model_ds[var_model]

    # Match shapes
    assert hist_model.shape[1:] == hist_obs.shape[1:] == future_model.shape[1:]

    corrected_future = np.full_like(future_model, np.nan)

    # Loop over each grid cell (lat, lon)
    for lat_idx in range(hist_model.shape[1]):
        for lon_idx in range(hist_model.shape[2]):
            model_hist_series = hist_model[:, lat_idx, lon_idx].values
            obs_hist_series = hist_obs[:, lat_idx, lon_idx].values
            model_future_series = future_model[:, lat_idx, lon_idx].values

            # Skip NaN cells
            if np.all(np.isnan(model_hist_series)) or np.all(np.isnan(obs_hist_series)):
                continue

            model_hist_sorted = np.sort(model_hist_series)
            obs_hist_sorted = np.sort(obs_hist_series)

            # Percentile mapping
            cdf_model_future = np.array([
                percentileofscore(model_hist_series, x, kind='mean') for x in model_future_series
            ]) / 100

            quantiles_obs = np.array([
                scoreatpercentile(obs_hist_sorted, p * 100) for p in cdf_model_future
            ])
            quantiles_model_hist = np.array([
                scoreatpercentile(model_hist_sorted, p * 100) for p in cdf_model_future
            ])
            delta = model_future_series - quantiles_model_hist
            corrected = quantiles_obs + delta
            corrected_future[:, lat_idx, lon_idx] = corrected

    corrected_da = xr.DataArray(
        corrected_future,
        dims=future_model.dims,
        coords=future_model.coords,
        name=f"{var_model}_QDM_corrected"
    )

    return corrected_da

HistACCTmpM_ERAGag_corrNC = QDM_Tmp(
    hist_model_ds=ACCESS_CM2histMnthTemp1990_2014_regriddedF,
    hist_obs_ds=HistNTempM_ERAGag_corr,
    future_model_ds=ACCESS_CM2histMnthTemp1990_2014_regriddedF,
    var_model='tas',
    var_obs='Temp'
)

HistACCTmpM_ERAGag_corrNC['time'] = HistACCTmpM_ERAGag_corrNC['time'].dt.floor('D')
HistACCTmpM_ERAGag_corrNC = HistACCTmpM_ERAGag_corrNC.to_dataset(name='Temp')

HistCanTmpM_ERAGag_corrNC  = QDM_Tmp(
    hist_model_ds=CanESM5histMnthTemp1990_2014_regriddedF,
    hist_obs_ds=HistNTempM_ERAGag_corr,
    future_model_ds=CanESM5histMnthTemp1990_2014_regriddedF,
    var_model='tas',
    var_obs='Temp'
)

HistCanTmpM_ERAGag_corrNC ['time'] = HistCanTmpM_ERAGag_corrNC ['time'].dt.floor('D')
HistCanTmpM_ERAGag_corrNC  = HistCanTmpM_ERAGag_corrNC .to_dataset(name='Temp')

HistGFDLTmpM_ERAGag_corrNC = QDM_Tmp(
    hist_model_ds=GFDL_ESM4histMnthTemp1990_2014_regriddedF,
    hist_obs_ds=HistNTempM_ERAGag_corr,
    future_model_ds=GFDL_ESM4histMnthTemp1990_2014_regriddedF,
    var_model='tas',
    var_obs='Temp'
)

HistGFDLTmpM_ERAGag_corrNC['time'] = HistGFDLTmpM_ERAGag_corrNC['time'].dt.floor('D')
HistGFDLTmpM_ERAGag_corrNC = HistGFDLTmpM_ERAGag_corrNC.to_dataset(name='Temp')

print ("Temperature baselines corrected")

print ("######################## Temp QDM Done ####################################")
