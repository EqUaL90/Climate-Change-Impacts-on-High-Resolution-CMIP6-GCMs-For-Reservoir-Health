#Select subsets
HistERA5PrecipNCM = ERA5PrecipNCM.sel(time = slice("1990-01-01", "2014-12-31"))
HistERA5TempNCM = ERA5TempNCM.sel(time = slice("1990-01-01", "2014-12-31")) 

SelERA5PrecipNCM = ERA5PrecipNCM.sel(time = slice("2015-01-01", "2022-12-31"))
SelERA5TempNCM = ERA5TempNCM.sel(time = slice("2015-01-01", "2022-12-31"))  

from scipy.stats import percentileofscore, scoreatpercentile
def QDM_Pr(hist_model_ds, hist_obs_ds, future_model_ds, var_model='pr', var_obs='Precip'):
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
HistACCPrcERAcorr = QDM_Prc(
    hist_model_ds=ACCESS_CM2histMnthPrec1990_2014_regriddedF,
    hist_obs_ds=HistERA5PrecipNCM,
    future_model_ds=ACCESS_CM2histMnthPrec1990_2014_regriddedF,
    var_model='pr',
    var_obs='Precip'
)

HistACCPrcERAcorr['time'] = HistACCPrcERAcorr['time'].dt.floor('D')
HistACCPrcERAcorr = HistACCPrcERAcorr.to_dataset(name='Precip')

HistCanPrcERAcorr = QDM_Prc(
    hist_model_ds=CanESM5histMnthPrec1990_2014_regriddedF,
    hist_obs_ds=HistERA5PrecipNCM,
    future_model_ds=CanESM5histMnthPrec1990_2014_regriddedF,
    var_model='pr',
    var_obs='Precip'
)

HistCanPrcERAcorr['time'] = HistCanPrcERAcorr['time'].dt.floor('D')
HistCanPrcERAcorr = HistCanPrcERAcorr.to_dataset(name='Precip')

HistGFDLPrcERAcorr = QDM_Prc(
    hist_model_ds=GFDL_ESM4histMnthPrec1990_2014_regriddedF,
    hist_obs_ds=HistERA5PrecipNCM,
    future_model_ds=GFDL_ESM4histMnthPrec1990_2014_regriddedF,
    var_model='pr',
    var_obs='Precip'
)

HistGFDLPrcERAcorr['time'] = HistGFDLPrcERAcorr['time'].dt.floor('D')
HistGFDLPrcERAcorr = HistGFDLPrcERAcorr.to_dataset(name='Precip')

########Temp

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


HistACCTmpERAcorr = QDM_Tmp(
    hist_model_ds=ACCESS_CM2histMnthTemp1990_2014_regriddedF,
    hist_obs_ds=HistERA5TempNCM,
    future_model_ds=ACCESS_CM2histMnthTemp1990_2014_regriddedF,
    var_model='tas',
    var_obs='Temp'
)

HistACCTmpERAcorr['time'] = HistACCTmpERAcorr['time'].dt.floor('D')
HistACCTmpERAcorr = HistACCTmpERAcorr.to_dataset(name='Temp')

HistCanTmpERAcorr  = QDM_Tmp(
    hist_model_ds=CanESM5histMnthTemp1990_2014_regriddedF,
    hist_obs_ds=HistERA5TempNCM,
    future_model_ds=CanESM5histMnthTemp1990_2014_regriddedF,
    var_model='tas',
    var_obs='Temp'
)

HistCanTmpERAcorr['time'] = HistCanTmpERAcorr['time'].dt.floor('D')
HistCanTmpERAcorr = HistCanTmpERAcorr.to_dataset(name='Temp')

HistGFDLTmpERAcorr = QDM_Tmp(
    hist_model_ds=GFDL_ESM4histMnthTemp1990_2014_regriddedF,
    hist_obs_ds=HistERA5TempNCM,
    future_model_ds=GFDL_ESM4histMnthTemp1990_2014_regriddedF,
    var_model='tas',
    var_obs='Temp'
)

HistGFDLTmpERAcorr['time'] = HistGFDLTmpERAcorr['time'].dt.floor('D')
HistGFDLTmpERAcorr = HistGFDLTmpERAcorr.to_dataset(name='Temp')

print ("Temperature baselines corrected based on ERA5 only")

##SSP
#QDM training (2015 - 2022)
Tr_ERA_SSP126AccPrec_corr = QDM_Prc(
    hist_model_ds=SelFutPrecAcc_SSP126_Rgdd,
    hist_obs_ds=HistACCPrcERAcorr,
    future_model_ds=SelFutPrecAcc_SSP126_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_ERA_SSP126AccPrec_corr['time'] = Tr_ERA_SSP126AccPrec_corr['time'].dt.floor('D')
Tr_ERA_SSP126AccPrec_corr= Tr_ERA_SSP126AccPrec_corr.to_dataset(name='Precip')

Tr_ERA_SSP245AccPrec_corr = QDM_Prc(
    hist_model_ds=SelFutPrecAcc_SSP245_Rgdd,        
    hist_obs_ds=HistACCPrcERAcorr,
    future_model_ds=SelFutPrecAcc_SSP245_Rgdd,
    var_model='pr',
    var_obs='Precip'                        
)
Tr_ERA_SSP245AccPrec_corr['time'] = Tr_ERA_SSP245AccPrec_corr['time'].dt.floor('D')
Tr_ERA_SSP245AccPrec_corr = Tr_ERA_SSP245AccPrec_corr.to_dataset(name='Precip')

Tr_ERA_SSP585AccPrec_corr = QDM_Prc(
    hist_model_ds=SelFutPrecAcc_SSP585_Rgdd,
    hist_obs_ds=HistACCPrcERAcorr,
    future_model_ds=SelFutPrecAcc_SSP585_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_ERA_SSP585AccPrec_corr['time'] = Tr_ERA_SSP585AccPrec_corr['time'].dt.floor('D')
Tr_ERA_SSP585AccPrec_corr = Tr_ERA_SSP585AccPrec_corr.to_dataset(name='Precip')

#CanESM5 Prc
Tr_ERA_SSP126CanPrc_corr = QDM_Prc( 
    hist_model_ds=SelFutPrecCan_SSP126_Rgdd,
    hist_obs_ds=HistCanPrcERAcorr,
    future_model_ds=SelFutPrecCan_SSP126_Rgdd,
    var_model='pr',
    var_obs='Precip'    
)
Tr_ERA_SSP126CanPrc_corr['time'] = Tr_ERA_SSP126CanPrc_corr['time'].dt.floor('D')
Tr_ERA_SSP126CanPrc_corr = Tr_ERA_SSP126CanPrc_corr.to_dataset(name='Precip')

Tr_ERA_SSP245CanPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecCan_SSP245_Rgdd,
    hist_obs_ds=HistCanPrcERAcorr,
    future_model_ds=SelFutPrecCan_SSP245_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_ERA_SSP245CanPrc_corr['time'] = Tr_ERA_SSP245CanPrc_corr['time'].dt.floor('D')
Tr_ERA_SSP245CanPrc_corr = Tr_ERA_SSP245CanPrc_corr.to_dataset(name='Precip')

Tr_ERA_SSP585CanPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecCan_SSP585_Rgdd,
    hist_obs_ds=HistCanPrcERAcorr,
    future_model_ds=SelFutPrecCan_SSP585_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_ERA_SSP585CanPrc_corr['time'] = Tr_ERA_SSP585CanPrc_corr['time'].dt.floor('D')
Tr_ERA_SSP585CanPrc_corr = Tr_ERA_SSP585CanPrc_corr.to_dataset(name='Precip')

#GFDL_ESM4 Prc
Tr_ERA_SSP126GFDLPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecGFDL_SSP126_Rgdd,
    hist_obs_ds=HistGFDLPrcERAcorr,
    future_model_ds=SelFutPrecGFDL_SSP126_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_ERA_SSP126GFDLPrc_corr['time'] = Tr_ERA_SSP126GFDLPrc_corr['time'].dt.floor('D')
Tr_ERA_SSP126GFDLPrc_corr = Tr_ERA_SSP126GFDLPrc_corr.to_dataset(name='Precip')

Tr_ERA_SSP245GFDLPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecGFDL_SSP245_Rgdd,
    hist_obs_ds=HistGFDLPrcERAcorr,
    future_model_ds=SelFutPrecGFDL_SSP245_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_ERA_SSP245GFDLPrc_corr['time'] = Tr_ERA_SSP245GFDLPrc_corr['time'].dt.floor('D')
Tr_ERA_SSP245GFDLPrc_corr = Tr_ERA_SSP245GFDLPrc_corr.to_dataset(name='Precip')

Tr_ERA_SSP585GFDLPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecGFDL_SSP585_Rgdd,
    hist_obs_ds=HistGFDLPrcERAcorr,
    future_model_ds=SelFutPrecGFDL_SSP585_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_ERA_SSP585GFDLPrc_corr['time'] = Tr_ERA_SSP585GFDLPrc_corr['time'].dt.floor('D')
Tr_ERA_SSP585GFDLPrc_corr = Tr_ERA_SSP585GFDLPrc_corr.to_dataset(name='Precip')

#Temperature correction
Tr_ERA_SSP126AccTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempAcc_SSP126_Rgdd,
    hist_obs_ds=HistACCTmpERAcorr,
    future_model_ds=SelFutTempAcc_SSP126_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP126AccTmp_corr['time'] = Tr_ERA_SSP126AccTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP126AccTmp_corr = Tr_ERA_SSP126AccTmp_corr.to_dataset(name='Temp')

Tr_ERA_SSP245AccTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempAcc_SSP245_Rgdd,
    hist_obs_ds=HistACCTmpERAcorr,
    future_model_ds=SelFutTempAcc_SSP245_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP245AccTmp_corr['time'] = Tr_ERA_SSP245AccTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP245AccTmp_corr = Tr_ERA_SSP245AccTmp_corr.to_dataset(name='Temp')

Tr_ERA_SSP585AccTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempAcc_SSP585_Rgdd,
    hist_obs_ds=HistACCTmpERAcorr,
    future_model_ds=SelFutTempAcc_SSP585_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP585AccTmp_corr['time'] = Tr_ERA_SSP585AccTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP585AccTmp_corr = Tr_ERA_SSP585AccTmp_corr.to_dataset(name='Temp')

#CanESM5 Tmp
Tr_ERA_SSP126CanTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempCan_SSP126_Rgdd,
    hist_obs_ds=HistCanTmpERAcorr,
    future_model_ds=SelFutTempCan_SSP126_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP126CanTmp_corr['time'] = Tr_ERA_SSP126CanTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP126CanTmp_corr = Tr_ERA_SSP126CanTmp_corr.to_dataset(name='Temp')
Tr_ERA_SSP245CanTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempCan_SSP245_Rgdd,
    hist_obs_ds=HistCanTmpERAcorr,
    future_model_ds=SelFutTempCan_SSP245_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP245CanTmp_corr['time'] = Tr_ERA_SSP245CanTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP245CanTmp_corr = Tr_ERA_SSP245CanTmp_corr.to_dataset(name='Temp')

Tr_ERA_SSP585CanTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempCan_SSP585_Rgdd,
    hist_obs_ds=HistCanTmpERAcorr,
    future_model_ds=SelFutTempCan_SSP585_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP585CanTmp_corr['time'] = Tr_ERA_SSP585CanTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP585CanTmp_corr = Tr_ERA_SSP585CanTmp_corr.to_dataset(name='Temp')

#GFDL_ESM4 Tmp
Tr_ERA_SSP126GFDLTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempGFDL_SSP126_Rgdd,
    hist_obs_ds=HistGFDLTmpERAcorr,
    future_model_ds=SelFutTempGFDL_SSP126_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP126GFDLTmp_corr['time'] = Tr_ERA_SSP126GFDLTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP126GFDLTmp_corr = Tr_ERA_SSP126GFDLTmp_corr.to_dataset(name='Temp')

Tr_ERA_SSP245GFDLTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempGFDL_SSP245_Rgdd,
    hist_obs_ds=HistGFDLTmpERAcorr,
    future_model_ds=SelFutTempGFDL_SSP245_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP245GFDLTmp_corr['time'] = Tr_ERA_SSP245GFDLTmp_corr['time'].dt.floor('D') 
Tr_ERA_SSP245GFDLTmp_corr = Tr_ERA_SSP245GFDLTmp_corr.to_dataset(name='Temp')

Tr_ERA_SSP585GFDLTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempGFDL_SSP585_Rgdd,
    hist_obs_ds=HistGFDLTmpERAcorr,
    future_model_ds=SelFutTempGFDL_SSP585_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_ERA_SSP585GFDLTmp_corr['time'] = Tr_ERA_SSP585GFDLTmp_corr['time'].dt.floor('D')
Tr_ERA_SSP585GFDLTmp_corr = Tr_ERA_SSP585GFDLTmp_corr.to_dataset(name='Temp')

# Save the corrected datasets
output_dir = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\OutputNC\ERA_Org_BC"
Tr_ERA_SSP126AccPrec_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP126AccPrec_corr.nc")
Tr_ERA_SSP245AccPrec_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP245AccPrec_corr.nc")
Tr_ERA_SSP585AccPrec_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP585AccPrec_corr.nc")
Tr_ERA_SSP126CanPrc_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP126CanPrc_corr.nc")
Tr_ERA_SSP245CanPrc_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP245CanPrc_corr.nc")
Tr_ERA_SSP585CanPrc_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP585CanPrc_corr.nc")
Tr_ERA_SSP126GFDLPrc_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP126GFDLPrc_corr.nc")
Tr_ERA_SSP245GFDLPrc_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP245GFDLPrc_corr.nc")
Tr_ERA_SSP585GFDLPrc_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP585GFDLPrc_corr.nc")
Tr_ERA_SSP126AccTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP126AccTmp_corr.nc")
Tr_ERA_SSP245AccTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP245AccTmp_corr.nc")
Tr_ERA_SSP585AccTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP585AccTmp_corr.nc")
Tr_ERA_SSP126CanTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP126CanTmp_corr.nc")
Tr_ERA_SSP245CanTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP245CanTmp_corr.nc")
Tr_ERA_SSP585CanTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP585CanTmp_corr.nc")
Tr_ERA_SSP126GFDLTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP126GFDLTmp_corr.nc")
Tr_ERA_SSP245GFDLTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP245GFDLTmp_corr.nc")
Tr_ERA_SSP585GFDLTmp_corr.to_netcdf(f"{output_dir}\\Tr_ERA_SSP585GFDLTmp_corr.nc")

print("ERA_Original-Bias correction completed and saved to NetCDF files.")
