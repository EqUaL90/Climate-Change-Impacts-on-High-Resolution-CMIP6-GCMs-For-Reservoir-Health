ACC_Old_precip_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Precip\AllStacksFull\Stack_pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_.nc"
ACC_Old_temp_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Temp\AllStacksFull\Stack_tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_.nc"
Can_Old_precip_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Precip\AllStacksFull\Stack_pr_day_CanESM5_historical_r1i1p1f1_gn_.nc"
Can_Old_temp_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Temp\AllStacksFull\Stack_tas_day_CanESM5_historical_r1i1p1f1_gn_.nc"
GFDL_Old_precip_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Precip\AllStacksFull\Stack_pr_day_GFDL-ESM4_ssp126_r1i1p1f1_gr1_.nc"
GFDL_Old_temp_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\Temp\AllStacksFull\Stack_tas_day_GFDL-ESM4_historical_r1i1p1f1_gr1_.nc"

#PrecipVariables
ACC_Old_precip = xr.open_dataset(ACC_Old_precip_path)
RegACCPrecip = ACCESS_CM2histMnthPrec1990_2014_regriddedF
#era5_precip =  HistERA5PrecipNCM.rio.clip(niger5.geometry.values, niger5.crs, drop=True) 

Can_Old_precip = xr.open_dataset(Can_Old_precip_path)
RegCanPrecip = CanESM5histMnthPrec1990_2014_regriddedF

GFDL_Old_precip = xr.open_dataset(GFDL_Old_precip_path)
RegGFDLPrecip = GFDL_ESM4histMnthPrec1990_2014_regriddedF

#TempVariables
ACC_Old_temp = xr.open_dataset(ACC_Old_temp_path)
RegACCTemp = ACCESS_CM2histMnthTemp1990_2014_regriddedF
era5_temp = HistERA5TempNCM.rio.clip(niger5.geometry.values, niger5.crs, drop=True) 

Can_Old_temp = xr.open_dataset(Can_Old_temp_path)
RegCanTemp = CanESM5histMnthTemp1990_2014_regriddedF

GFDL_Old_temp = xr.open_dataset(GFDL_Old_temp_path)
RegGFDLTemp = GFDL_ESM4histMnthTemp1990_2014_regriddedF


ACC_Old_precip_ts = ACC_Old_precip.isel(time=5)
RegRegACCPrecip_ts = RegACCPrecip.isel(time=5)
#era5_precip_ts = era5_precip.isel(time=5)
Can_Old_precip_ts = Can_Old_precip.isel(time=5)
RegCanPrecip_ts = RegCanPrecip.isel(time=5)
GFDL_Old_precip_ts = GFDL_Old_precip.isel(time=5)
RegGFDLPrecip_ts = RegGFDLPrecip.isel(time=5)

ACC_Old_temp_ts = ACC_Old_temp.isel(time=5)
RegACCTemp_ts = RegACCTemp.isel(time=5)
#era5_temp_ts = era5_temp.isel(time=5)
Can_Old_temp_ts = Can_Old_temp.isel(time=5)
RegCanTemp_ts = RegCanTemp.isel(time=5)
GFDL_Old_temp_ts = GFDL_Old_temp.isel(time=5)
RegGFDLTemp_ts = RegGFDLTemp.isel(time=5)

#Use in paper too
fig, axs = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

titles = [
    "ACCESS Precip (Original)", "ACCESS Precip (Regridded)", "ACCESS Temp (Original)", "ACCESS Temp (Regridded)",
    "CanESM5 Precip (Original)", "CanESM5 Precip (Regridded)", "CanESM5 Temp (Original)", "CanESM5 Temp (Regridded)",
    "GFDL Precip (Original)", "GFDL Precip (Regridded)", "GFDL Temp (Original)", "GFDL Temp (Regridded)"
]

datasets = [
    ACC_Old_precip_ts['pr'], RegRegACCPrecip_ts['pr'], ACC_Old_temp_ts['tas'], RegACCTemp_ts['tas'],
    Can_Old_precip_ts['pr'], RegCanPrecip_ts['pr'], Can_Old_temp_ts['tas'], RegCanTemp_ts['tas'],
    GFDL_Old_precip_ts['pr'], RegGFDLPrecip_ts['pr'], GFDL_Old_temp_ts['tas'], RegGFDLTemp_ts['tas']
]

cmaps = ['jet_r', 'jet_r', 'inferno', 'inferno'] * 3
cbar_labels = ['Precipitation (mm/month)', 'Precipitation (mm/month)', 'Temperature (°C)', 'Temperature (°C)'] * 3

for i, ax in enumerate(axs.flat):
    im = datasets[i].plot(
        ax=ax,
        cmap=cmaps[i],
        add_colorbar=False
    )
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.67, pad=0.01)
    cbar.set_label(cbar_labels[i], fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    reservoir.plot(ax=ax, color='lightblue', edgecolor='lightgray', linewidth=1.5, alpha=0.3)
    river.plot(ax=ax, color='skyblue', linewidth=1.5, alpha=0.7)
    ax.set_title(titles[i], fontsize=18)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(''); ax.set_ylabel('')

#plt.subplots_adjust(wspace=0.3, hspace=0.2)
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\XXRegriddingProofALLMIP6OnlyWith ReservoirPaper.png", dpi=500)
plt.show()

