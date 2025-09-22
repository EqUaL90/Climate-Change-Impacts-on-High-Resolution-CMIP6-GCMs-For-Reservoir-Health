#This worked!!!!!!!!!!!!!!!!!!  (Regrid only hist (1990 - 2014))
import xesmf as xe
import os
import rasterio as rio
niger5 = gpd.read_file(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\Niger 5\Subbasin_boundary.shp")
HistERA5PrecipNCM = ERA5PrecipNCM.sel(time = slice("1990-01-01", "2014-12-31"))
HistERA5TempNCM = ERA5TempNCM.sel(time = slice("1990-01-01", "2014-12-31")) 

def regrid_cmip6_to_era5(cmip6_files, hist_era5_precip, output_dir):
    for cmip6_file in cmip6_files:
        cmip6_ds = xr.open_dataset(cmip6_file)
        regridder = xe.Regridder(cmip6_ds, hist_era5_precip, "bilinear", extrap_method="nearest_s2d")
        cmip6_regridded = regridder(cmip6_ds)
        cmip6_regridded = cmip6_regridded.rio.write_crs("EPSG:4326", inplace=True)
        cmip6_regridded  = cmip6_regridded.rio.clip(niger5.geometry.values, niger5.crs, drop=True)
        cmip6_regridded = cmip6_regridded.ffill(dim='time')

        cmip6_name = os.path.basename(cmip6_file).replace('.nc', '_Rgdd.nc')
        output_path = os.path.join(output_dir, cmip6_name)

        cmip6_regridded.to_netcdf(output_path)
        print(regridder)
        print(f"✅ Saved regridded file: {output_path}")

cmip6_files = [
    r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Precip\AllStacksFull\Stack_pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_.nc",
    r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Precip\AllStacksFull\Stack_pr_day_CanESM5_historical_r1i1p1f1_gn_.nc",
    r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Precip\AllStacksFull\Stack_pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_.nc"
]

output_dir = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\RegrdOut"
os.makedirs(output_dir, exist_ok = True)

regrid_cmip6_to_era5(cmip6_files, HistERA5PrecipNCM, output_dir)

cmip6_files = [
    r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Temp\AllStacksFull\Stack_tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_.nc",
    r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Temp\AllStacksFull\Stack_tas_day_CanESM5_historical_r1i1p1f1_gn_.nc",
    r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Temp\AllStacksFull\Stack_tas_day_GFDL-ESM4_historical_r1i1p1f1_gr1_.nc"
]
##I remove the output_dir definition deliberately to avoid repetition
regrid_cmip6_to_era5(cmip6_files,HistERA5TempNCM, output_dir)


## Load regrided datasets 
ACCESS_CM2histMnthPrec1990_2014_regriddedF= xr.open_dataset(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\RegrdOut\Stack_pr_day_ACCESS-CM2_historical_r1i1p1f1_gn__Rgdd.nc")
ACCESS_CM2histMnthTemp1990_2014_regriddedF= xr.open_dataset(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\RegrdOut\Stack_tas_day_ACCESS-CM2_historical_r1i1p1f1_gn__Rgdd.nc")
CanESM5histMnthPrec1990_2014_regriddedF= xr.open_dataset(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\RegrdOut\Stack_pr_day_CanESM5_historical_r1i1p1f1_gn__Rgdd.nc")
CanESM5histMnthTemp1990_2014_regriddedF= xr.open_dataset(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\RegrdOut\Stack_tas_day_CanESM5_historical_r1i1p1f1_gn__Rgdd.nc")
GFDL_ESM4histMnthPrec1990_2014_regriddedF= xr.open_dataset(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\RegrdOut\Stack_pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1__Rgdd.nc")
GFDL_ESM4histMnthTemp1990_2014_regriddedF= xr.open_dataset(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\RegrdOut\Stack_tas_day_GFDL-ESM4_historical_r1i1p1f1_gr1__Rgdd.nc")
print("Loaded successfully by EqUaL")
