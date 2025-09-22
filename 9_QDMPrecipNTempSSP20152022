#QDM training (2015 - 2022)
ref_time = SelFutPrecAcc_SSP126_Rgdd['time']

SelFutPrecCan_SSP126_Rgdd['time'] = ref_time
SelFutPrecCan_SSP245_Rgdd['time'] = ref_time
SelFutPrecCan_SSP585_Rgdd['time'] = ref_time
SelFutPrecGFDL_SSP126_Rgdd['time'] = ref_time
SelFutPrecGFDL_SSP245_Rgdd['time'] = ref_time
SelFutPrecGFDL_SSP585_Rgdd['time'] = ref_time
SelFutTempCan_SSP126_Rgdd['time'] = ref_time
SelFutTempCan_SSP245_Rgdd['time'] = ref_time
SelFutTempCan_SSP585_Rgdd['time'] = ref_time
SelFutTempGFDL_SSP126_Rgdd['time'] = ref_time
SelFutTempGFDL_SSP245_Rgdd['time'] = ref_time
SelFutTempGFDL_SSP585_Rgdd['time'] = ref_time


Tr_SSP126AccPrec_corr = QDM_Prc(
    hist_model_ds=SelFutPrecAcc_SSP126_Rgdd,
    hist_obs_ds=HistACCPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecAcc_SSP126_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_SSP126AccPrec_corr ['time'] = Tr_SSP126AccPrec_corr['time'].dt.floor('D')
Tr_SSP126AccPrec_corr=Tr_SSP126AccPrec_corr.to_dataset(name='Precip')

Tr_SSP245AccPrec_corr = QDM_Prc(
    hist_model_ds=SelFutPrecAcc_SSP245_Rgdd,
    hist_obs_ds=HistACCPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecAcc_SSP245_Rgdd,
    var_model='pr',
    var_obs='Precip'        
)  
Tr_SSP245AccPrec_corr['time'] = Tr_SSP245AccPrec_corr['time'].dt.floor('D')
Tr_SSP245AccPrec_corr = Tr_SSP245AccPrec_corr.to_dataset(name='Precip')

Tr_SSP585AccPrec_corr = QDM_Prc(
    hist_model_ds=SelFutPrecAcc_SSP585_Rgdd,
    hist_obs_ds=HistACCPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecAcc_SSP585_Rgdd,
    var_model='pr',
    var_obs='Precip'
)   
Tr_SSP585AccPrec_corr['time'] = Tr_SSP585AccPrec_corr['time'].dt.floor('D')
Tr_SSP585AccPrec_corr = Tr_SSP585AccPrec_corr.to_dataset(name='Precip')

#CanESM5 Prc
Tr_SSP126CanPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecCan_SSP126_Rgdd,
    hist_obs_ds=HistCanPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecCan_SSP126_Rgdd,
    var_model='pr',
    var_obs='Precip'
)   
Tr_SSP126CanPrc_corr ['time'] = Tr_SSP126CanPrc_corr ['time'].dt.floor('D')    
Tr_SSP126CanPrc_corr = Tr_SSP126CanPrc_corr .to_dataset(name='Precip')

Tr_SSP245CanPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecCan_SSP245_Rgdd,
    hist_obs_ds=HistCanPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecCan_SSP245_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_SSP245CanPrc_corr['time'] = Tr_SSP245CanPrc_corr ['time'].dt.floor('D')
Tr_SSP245CanPrc_corr = Tr_SSP245CanPrc_corr.to_dataset(name='Precip')

Tr_SSP585CanPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecCan_SSP585_Rgdd,
    hist_obs_ds=HistCanPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecCan_SSP585_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_SSP585CanPrc_corr['time'] = Tr_SSP585CanPrc_corr ['time'].dt.floor('D')
Tr_SSP585CanPrc_corr = Tr_SSP585CanPrc_corr.to_dataset(name='Precip')


#GFDL_ESM4 Prc
Tr_SSP126GFDLPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecGFDL_SSP126_Rgdd,
    hist_obs_ds=HistGFDLPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecGFDL_SSP126_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_SSP126GFDLPrc_corr  ['time'] = Tr_SSP126GFDLPrc_corr  ['time'].dt.floor('D')
Tr_SSP126GFDLPrc_corr  = Tr_SSP126GFDLPrc_corr .to_dataset(name='Precip')

Tr_SSP245GFDLPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecGFDL_SSP245_Rgdd,
    hist_obs_ds=HistGFDLPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecGFDL_SSP245_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_SSP245GFDLPrc_corr['time'] = Tr_SSP245GFDLPrc_corr['time'].dt.floor('D')
Tr_SSP245GFDLPrc_corr = Tr_SSP245GFDLPrc_corr.to_dataset(name='Precip')

Tr_SSP585GFDLPrc_corr = QDM_Prc(
    hist_model_ds=SelFutPrecGFDL_SSP585_Rgdd,
    hist_obs_ds=HistGFDLPrcM_ERAGag_corrNC,
    future_model_ds=SelFutPrecGFDL_SSP585_Rgdd,
    var_model='pr',
    var_obs='Precip'
)
Tr_SSP585GFDLPrc_corr['time'] =Tr_SSP585GFDLPrc_corr['time'].dt.floor('D')
Tr_SSP585GFDLPrc_corr = Tr_SSP585GFDLPrc_corr.to_dataset(name='Precip')


#Temperature correction

Tr_SSP126AccTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempAcc_SSP126_Rgdd,
    hist_obs_ds=HistACCTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempAcc_SSP126_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP126AccTmp_corr ['time'] = Tr_SSP126AccTmp_corr ['time'].dt.floor('D')
Tr_SSP126AccTmp_corr = Tr_SSP126AccTmp_corr .to_dataset(name='Temp')

Tr_SSP245AccTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempAcc_SSP245_Rgdd,
    hist_obs_ds=HistACCTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempAcc_SSP245_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP245AccTmp_corr['time'] = Tr_SSP245AccTmp_corr ['time'].dt.floor('D')
Tr_SSP245AccTmp_corr = Tr_SSP245AccTmp_corr.to_dataset(name='Temp')

Tr_SSP585AccTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempAcc_SSP585_Rgdd,
    hist_obs_ds=HistACCTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempAcc_SSP585_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP585AccTmp_corr ['time'] =Tr_SSP585AccTmp_corr['time'].dt.floor('D')
Tr_SSP585AccTmp_corr  = Tr_SSP585AccTmp_corr .to_dataset(name='Temp')

#CanESM5 Tmp
Tr_SSP126CanTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempCan_SSP126_Rgdd,
    hist_obs_ds=HistCanTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempCan_SSP126_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP126CanTmp_corr ['time'] =Tr_SSP126CanTmp_corr ['time'].dt.floor('D')
Tr_SSP126CanTmp_corr = Tr_SSP126CanTmp_corr .to_dataset(name='Temp')

Tr_SSP245CanTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempCan_SSP245_Rgdd,
    hist_obs_ds=HistCanTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempCan_SSP245_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP245CanTmp_corr ['time'] = Tr_SSP245CanTmp_corr['time'].dt.floor('D')
Tr_SSP245CanTmp_corr = Tr_SSP245CanTmp_corr.to_dataset(name='Temp')

Tr_SSP585CanTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempCan_SSP585_Rgdd,
    hist_obs_ds=HistCanTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempCan_SSP585_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP585CanTmp_corr['time'] =Tr_SSP585CanTmp_corr ['time'].dt.floor('D')
Tr_SSP585CanTmp_corr= Tr_SSP585CanTmp_corr.to_dataset(name='Temp')

#GFDL_ESM4 Tmp
Tr_SSP126GFDLTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempGFDL_SSP126_Rgdd,
    hist_obs_ds=HistGFDLTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempGFDL_SSP126_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP126GFDLTmp_corr['time'] = Tr_SSP126GFDLTmp_corr['time'].dt.floor('D')
Tr_SSP126GFDLTmp_corr = Tr_SSP126GFDLTmp_corr.to_dataset(name='Temp')

Tr_SSP245GFDLTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempGFDL_SSP245_Rgdd,
    hist_obs_ds=HistGFDLTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempGFDL_SSP245_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP245GFDLTmp_corr ['time'] = Tr_SSP245GFDLTmp_corr['time'].dt.floor('D')
Tr_SSP245GFDLTmp_corr = Tr_SSP245GFDLTmp_corr.to_dataset(name='Temp')

Tr_SSP585GFDLTmp_corr = QDM_Tmp(
    hist_model_ds=SelFutTempGFDL_SSP585_Rgdd,
    hist_obs_ds=HistGFDLTmpM_ERAGag_corrNC,
    future_model_ds=SelFutTempGFDL_SSP585_Rgdd,
    var_model='tas',
    var_obs='Temp'
)
Tr_SSP585GFDLTmp_corr['time'] = Tr_SSP585GFDLTmp_corr['time'].dt.floor('D')
Tr_SSP585GFDLTmp_corr = Tr_SSP585GFDLTmp_corr.to_dataset(name='Temp')

# Save the corrected datasets
output_dir = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\OutputNC"
Tr_SSP126AccPrec_corr.to_netcdf(f"{output_dir}\\Tr_SSP126AccPrec_corr.nc")
Tr_SSP245AccPrec_corr.to_netcdf(f"{output_dir}\\Tr_SSP245AccPrec_corr.nc")
Tr_SSP585AccPrec_corr.to_netcdf(f"{output_dir}\\Tr_SSP585AccPrec_corr.nc")
Tr_SSP126CanPrc_corr.to_netcdf(f"{output_dir}\\Tr_SSP126CanPrc_corr.nc")
Tr_SSP245CanPrc_corr.to_netcdf(f"{output_dir}\\Tr_SSP245CanPrc_corr.nc")
Tr_SSP585CanPrc_corr.to_netcdf(f"{output_dir}\\Tr_SSP585CanPrc_corr.nc")
Tr_SSP126GFDLPrc_corr.to_netcdf(f"{output_dir}\\Tr_SSP126GFDLPrc_corr.nc")
Tr_SSP245GFDLPrc_corr.to_netcdf(f"{output_dir}\\Tr_SSP245GFDLPrc_corr.nc")
Tr_SSP585GFDLPrc_corr.to_netcdf(f"{output_dir}\\Tr_SSP585GFDLPrc_corr.nc")
Tr_SSP126AccTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP126AccTmp_corr.nc")
Tr_SSP245AccTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP245AccTmp_corr.nc")
Tr_SSP585AccTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP585AccTmp_corr.nc")
Tr_SSP126CanTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP126CanTmp_corr.nc")
Tr_SSP245CanTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP245CanTmp_corr.nc")
Tr_SSP585CanTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP585CanTmp_corr.nc")
Tr_SSP126GFDLTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP126GFDLTmp_corr.nc")
Tr_SSP245GFDLTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP245GFDLTmp_corr.nc")
Tr_SSP585GFDLTmp_corr.to_netcdf(f"{output_dir}\\Tr_SSP585GFDLTmp_corr.nc")
# Save the corrected datasets
output_dir = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\CC\Niger5CMIP6F\FF\OutputExcel\OutputNC"
print("Bias correction completed and saved to NetCDF files.")
