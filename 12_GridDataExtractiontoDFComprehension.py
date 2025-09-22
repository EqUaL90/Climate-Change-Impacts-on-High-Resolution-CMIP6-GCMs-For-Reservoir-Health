
import geopandas as gpd
from tqdm import tqdm

def ExtCorrPrcPerGrid(ds, b):
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    points = [(row.geometry.y, row.geometry.x) for _, row in b.iterrows()]
    fids = b["FID"].values
    times = ds["time"].values
    all_data = []
    print("Extracting precipitation for each time step and location...")
    for time_step in tqdm(times):
        values = []
        for lat, lon in points:
            val = ds.sel(
                latitude=lat,
                longitude=lon,
                time=time_step,
                method="nearest"
            )["Precip"].values

            value = val.item() if hasattr(val, "item") else val
            if value < 0:
                value = 0.0
            values.append(value)

        all_data.append(values)

    df = pd.DataFrame(all_data, index=pd.to_datetime(times))
    df.columns = [f'P{i+1}' for i in range(len(fids))]
    df.index.name = "Date"
    df = df.round(2)

    return df


def ExtOrgPrcPerGrid(ds, b):
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    points = [(row.geometry.y, row.geometry.x) for _, row in b.iterrows()]
    fids = b["FID"].values
    times = ds["time"].values
    all_data = []
    print("Extracting precipitation for each time step and location...")
    for time_step in tqdm(times):
        values = []
        for lat, lon in points:
            val = ds.sel(
                latitude=lat,
                longitude=lon,
                time=time_step,
                method="nearest"
            )["pr"].values

            value = val.item() if hasattr(val, "item") else val
            if value < 0:
                value = 0.0
            values.append(value)

        all_data.append(values)

    df = pd.DataFrame(all_data, index=pd.to_datetime(times))
    df.columns = [f'P{i+1}' for i in range(len(fids))]
    df.index.name = "Date"
    df = df.round(2)

    return df
# Temp

def ExtCorrTmpPerGrid(ds, b):
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    points = [(row.geometry.y, row.geometry.x) for _, row in b.iterrows()]
    fids = b["FID"].values
    times = ds["time"].values
    all_data = []
    print("Extracting temperaturefor each time step and location...")
    for time_step in tqdm(times):
        values = []
        for lat, lon in points:
            val = ds.sel(
                latitude=lat,
                longitude=lon,
                time=time_step,
                method="nearest"
            )["Temp"].values

            value = val.item() if hasattr(val, "item") else val
            if value < 0:
                value = 0.0
            values.append(value)

        all_data.append(values)

    df = pd.DataFrame(all_data, index=pd.to_datetime(times))
    df.columns = [f'T{i+1}' for i in range(len(fids))]
    df.index.name = "Date"
    df = df.round(2)

    return df


def ExtOrgTmpPerGrid(ds, b):
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    points = [(row.geometry.y, row.geometry.x) for _, row in b.iterrows()]
    fids = b["FID"].values
    times = ds["time"].values
    all_data = []
    print("Extracting temperature for each time step and location...")
    for time_step in tqdm(times):
        values = []
        for lat, lon in points:
            val = ds.sel(
                latitude=lat,
                longitude=lon,
                time=time_step,
                method="nearest"
            )["tas"].values

            value = val.item() if hasattr(val, "item") else val
            if value < 0:
                value = 0.0
            values.append(value)

        all_data.append(values)

    df = pd.DataFrame(all_data, index=pd.to_datetime(times))
    df.columns = [f'T{i+1}' for i in range(len(fids))]
    df.index.name = "Date"
    df = df.round(2)

    return df
DFds_base_Pr_Org = [ExtOrgPrcPerGrid(ds, b) for ds in ds_base_Pr_Org] # Historical Original
DFds_base_Pr_ERA_Corr = [ExtCorrPrcPerGrid(ds, b) for ds in ds_base_Pr_ERA_Corr] # Historical Corrected

DFds_Pr_Org = [ExtOrgPrcPerGrid(ds, b) for ds in ds_Pr_Org] # Future Original (2015–2022)
DFds_Pr_ERA_Corr = [ExtCorrPrcPerGrid(ds, b) for ds in ds_Pr_ERA_Corr] # Future Corrected (ERA Quantile-mapped)

DFds_Pr_target = [ExtOrgPrcPerGrid(ds, b) for ds in ds_Pr_target] # Future Target (2026–2080)

#Temperature baselines based on ERA-adjusted datasets and original CMIP6 baselines
ds_base_Tp_Org  = [ACCESS_CM2histMnthTemp1990_2014_regriddedF, CanESM5histMnthTemp1990_2014_regriddedF, GFDL_ESM4histMnthTemp1990_2014_regriddedF]
ds_base_Tp_ERA_Corr =  [HistACCTmpERAcorr, HistCanTmpERAcorr, HistGFDLTmpERAcorr]

#Temp extraction 
DFds_base_Tp_Org = [ExtOrgTmpPerGrid(ds, b) for ds in ds_base_Tp_Org] # Historical Original
DFds_base_Tp_ERA_Corr = [ExtCorrTmpPerGrid(ds, b) for ds in ds_base_Tp_ERA_Corr] # Historical Corrected

#Temperature Scenario extraction
ds_Tp_Org = [SelFutTempAcc_SSP126_Rgdd, SelFutTempAcc_SSP245_Rgdd, SelFutTempAcc_SSP585_Rgdd,
             SelFutTempCan_SSP126_Rgdd, SelFutTempCan_SSP245_Rgdd, SelFutTempCan_SSP585_Rgdd,
             SelFutTempGFDL_SSP126_Rgdd, SelFutTempGFDL_SSP245_Rgdd,SelFutTempGFDL_SSP585_Rgdd]

ds_Tp_ERA_Corr = [Tr_ERA_SSP126AccTmp_corr, Tr_ERA_SSP245AccTmp_corr, Tr_ERA_SSP585AccTmp_corr,
                Tr_ERA_SSP126CanTmp_corr, Tr_ERA_SSP245CanTmp_corr, Tr_ERA_SSP585CanTmp_corr,
                Tr_ERA_SSP126GFDLTmp_corr, Tr_ERA_SSP245GFDLTmp_corr, Tr_ERA_SSP585GFDLTmp_corr]  

ds_Tp_target = [AccTmpSSP126_26_80, AccTmpSSP245_26_80, AccTmpSSP585_26_80,
                CanTmpSSP126_26_80, CanTmpSSP245_26_80, CanTmpSSP585_26_80,
                GFDLTmpSSP126_26_80, GFDLTmpSSP245_26_80, GFDLTmpSSP585_26_80]

#Temp
DFds_base_Tp_Org = [ExtOrgTmpPerGrid(ds, b) for ds in ds_base_Tp_Org] # Historical Original
DFds_base_Tp_ERA_Corr = [ExtCorrTmpPerGrid(ds, b) for ds in ds_base_Tp_ERA_Corr] # Historical Corrected
DFds_Tp_Org = [ExtOrgTmpPerGrid(ds, b) for ds in ds_Tp_Org] # Future Original (2015–2022)
DFds_Tp_ERA_Corr = [ExtCorrTmpPerGrid(ds, b) for ds in ds_Tp_ERA_Corr] # Future Corrected (ERA QuantileDelta-mapped)
DFds_Tp_target = [ExtOrgTmpPerGrid(ds, b) for ds in ds_Tp_target] # Future Target (2026–2080)

