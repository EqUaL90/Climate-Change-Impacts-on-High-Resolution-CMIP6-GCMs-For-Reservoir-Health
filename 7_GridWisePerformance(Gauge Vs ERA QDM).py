import xarray as xr
import numpy as np
import os

# Load observed and corrected data
obs_pr = NObsPrcM["Precip"]             # (time, lat, lon)
cor_pr = NPrcM_ERAGag_corr["Precip"]

obs_tmp = NObsTmpM["Temp"]
cor_tmp = NTempM_ERAGag_corr["Temp"]

# Get lat/lon and time
lat = obs_pr.latitude
lon = obs_pr.longitude
time = obs_pr.time

def compute_metrics_per_pixel(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    o, s = obs[mask], sim[mask]
    rmse = np.sqrt(np.mean((s - o) ** 2))
    pbias = 100 * (np.sum(s - o) / np.sum(o)) if np.sum(o) != 0 else np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o) if np.std(o) != 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    nse = 1 - np.sum((s - o) ** 2) / np.sum((o - np.mean(o)) ** 2) if np.sum((o - np.mean(o)) ** 2) != 0 else np.nan
    mae = np.mean(np.abs(s - o))
    bias = np.mean(s - o)
    return rmse, pbias, kge, r, nse, mae, bias

def compute_grid_metrics(obs, cor):
    lat_size = obs.shape[1]
    lon_size = obs.shape[2]
    
    shape = (lat_size, lon_size)
    metrics = {
        "RMSE": np.full(shape, np.nan),
        "PBIAS": np.full(shape, np.nan),
        "KGE": np.full(shape, np.nan),
        "CORR": np.full(shape, np.nan),
        "NSE": np.full(shape, np.nan),
        "MAE": np.full(shape, np.nan),
        "BIAS": np.full(shape, np.nan),
    }

    for i in range(lat_size):
        for j in range(lon_size):
            obs_series = obs[:, i, j].values
            cor_series = cor[:, i, j].values
            results = compute_metrics_per_pixel(obs_series, cor_series)
            for key, val in zip(metrics.keys(), results):
                metrics[key][i, j] = val
                
    return metrics


metrics_pr = compute_grid_metrics(obs_pr, cor_pr)
metrics_tmp = compute_grid_metrics(obs_tmp, cor_tmp)
# Rename if coordinates are named differently
if 'latitude' in obs_pr.coords:
    obs_pr = obs_pr.rename({'latitude': 'lat'})
if 'longitude' in obs_pr.coords:
    obs_pr = obs_pr.rename({'longitude': 'lon'})
if 'latitude' in obs_tmp.coords:
    obs_tmp = obs_tmp.rename({'latitude': 'lat'})
if 'longitude' in obs_tmp.coords:
    obs_tmp = obs_tmp.rename({'longitude': 'lon'})

# Now extract coordinates using consistent names
lat = obs_pr.lat
lon = obs_pr.lon

def create_metrics_nc(metrics, lat, lon, out_path, var_prefix=""):
    ds = xr.Dataset()
    for key, data in metrics.items():
        ds[f"{var_prefix}_{key}"] = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": lat.values, "lon": lon.values},
            attrs={"description": f"{key} metric of {var_prefix}"}
        )
    ds.to_netcdf(out_path)
    print(f"Saved: {out_path}")

out_precip_file = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\OutputNC\BCERAGauge_Precip_Metrics.nc"
out_temp_file = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\OutputNC\\BCERAGauge_Temp_Metrics.nc"
create_metrics_nc(metrics_pr, lat, lon, out_precip_file, var_prefix="Prc")
create_metrics_nc(metrics_tmp, lat, lon, out_temp_file, var_prefix="Temp")
