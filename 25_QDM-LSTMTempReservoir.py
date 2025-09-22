# Extract reservoir stations from basin prediction and target 
ResStnsTmpCol = ['T9', 'T12', 'T36', 'T39', 'T41', 'T43', 'T106', 'T111']
Res_predictions_by_sspTp = {}

for key, df in LSTMpredictions_by_sspTp.items():
    filtered_df = df[ResStnsTmpCol]
    filtered_df = filtered_df.map(lambda x: round(x, 2))
    filtered_df = filtered_df.clip(lower=0)
    Res_predictions_by_sspTp[key] = filtered_df

print(Res_predictions_by_sspTp)
for key in Res_predictions_by_sspTp:
    Res_predictions_by_sspTp[key] += 4.2

######################################
ResDFCanESM5histMnthTemp1990_2014_regriddedF = DFCanESM5histMnthTemp1990_2014_regriddedF[ResStnsTmpCol].copy()
ResDFACCESS_CM2histMnthTemp1990_2014_regriddedF = DFACCESS_CM2histMnthTemp1990_2014_regriddedF[ResStnsTmpCol].copy()
ResDFGFDL_ESM4histMnthTemp1990_2014_regriddedF = DFGFDL_ESM4histMnthTemp1990_2014_regriddedF[ResStnsTmpCol].copy()

mean_ACCResTp = ResDFACCESS_CM2histMnthTemp1990_2014_regriddedF.mean(axis=1)
mean_CANResTp =ResDFCanESM5histMnthTemp1990_2014_regriddedF.mean(axis=1)
mean_GFDLResTp = ResDFGFDL_ESM4histMnthTemp1990_2014_regriddedF.mean(axis=1)


BaselineEnsResTp = pd.concat([mean_ACCResTp, mean_CANResTp, mean_GFDLResTp], axis=1)
BaselineEnsResTp.columns = ['ACC', 'CAN', 'GFDL']

# Take the ensemble mean across the models
BaselineEnsResTp['Baseline'] = BaselineEnsResTp.mean(axis=1)
BaselineEnsFResTp = BaselineEnsResTp[["Baseline"]]
BaselineEnsFResTp += 4.2
print(BaselineEnsResTp)
print(BaselineEnsFResTp.head(2))

##################################\

#Harvest to df by computing mean ensemble across scenarios
res_pred_mean = {}
for ssp_key, df in Res_predictions_by_sspTp.items():
    pred_mean = df.mean(axis=1)

    date_index = pd.date_range(start='2026-01-01', periods=len(pred_mean), freq='ME')
    res_pred_mean[ssp_key] = pd.Series(pred_mean.values, index=date_index)

LSTM_FutTmpRes = pd.DataFrame(res_pred_mean)
LSTM_FutTmpRes.index.name = 'Date'

print("Processed LSTM Predictions:")
LSTM_FutTmpRes = LSTM_FutTmpRes.round(2)
LSTM_FutTmpRes.columns = LSTM_FutTmp.columns
LSTM_FutTmpRes.head()


# Resample to Annual Means
# LSTM Future Projections (Bias-Corrected)
LSTM_FutTmpRes = LSTM_FutTmpRes.iloc[12:]
ssp126_qdm_annual = LSTM_FutTmpRes[[col for col in LSTM_FutTmpRes.columns if 'SSP126' in col]].resample('YE').mean()
ssp245_qdm_annual = LSTM_FutTmpRes[[col for col in LSTM_FutTmpRes.columns if 'SSP245' in col]].resample('YE').mean()
ssp585_qdm_annual = LSTM_FutTmpRes[[col for col in LSTM_FutTmpRes.columns if 'SSP585' in col]].resample('YE').mean()

ssp126_qdm_total = ssp126_qdm_annual.mean(axis=1)
ssp245_qdm_total = ssp245_qdm_annual.mean(axis=1)
ssp585_qdm_total = ssp585_qdm_annual.mean(axis=1)

ssp126_qdm_std = ssp126_qdm_annual.std(axis=1)
ssp245_qdm_std = ssp245_qdm_annual.std(axis=1)
ssp585_qdm_std = ssp585_qdm_annual.std(axis=1)

BaselineEnsFResTp = BaselineEnsFResTp.round(2)
Baseline_annualTpRes = BaselineEnsFResTp.resample('YE').mean()

plt.figure(figsize=(14, 5))

# === Shaded Time Periods ===


plt.axvspan(pd.Timestamp("1990-12"), pd.Timestamp("2014-12"), color='teal', alpha=0.2)
plt.axvspan(pd.Timestamp("2027-12"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-1"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# === Historical Baseline ===
plt.plot(Baseline_annualTpRes.index, Baseline_annualTpRes['Baseline'], 
         label='Baseline (BC, Annual)', color='black', linewidth=2.5, linestyle='--')

# === SSP126 ===
plt.plot(ssp126_qdm_total.index, ssp126_qdm_total, label='SSP126 (QDM-LSTM)', color='tomato', linewidth=2.5)
plt.fill_between(ssp126_qdm_total.index, ssp126_qdm_total - ssp126_qdm_std, 
                 ssp126_qdm_total + ssp126_qdm_std, color='tomato', alpha=0.2)

# === SSP245 ===
plt.plot(ssp245_qdm_total.index, ssp245_qdm_total, label='SSP245 (QDM-LSTM)', color='fuchsia', linewidth=2.5)
plt.fill_between(ssp245_qdm_total.index, ssp245_qdm_total - ssp245_qdm_std, 
                 ssp245_qdm_total + ssp245_qdm_std, color='fuchsia', alpha=0.2)

# === SSP585 ===
plt.plot(ssp585_qdm_total.index, ssp585_qdm_total, label='SSP585 (QDM-LSTM)', color='indigo', linewidth=2.5)
plt.fill_between(ssp585_qdm_total.index, ssp585_qdm_total - ssp585_qdm_std, 
                 ssp585_qdm_total + ssp585_qdm_std, color='indigo', alpha=0.2)


plt.title("Annual Reservoir Temperature Projections by MME (QDM-LSTM)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Temperature (°C)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)

# === Legends ===
line_legend = plt.legend(loc='upper center', ncol=4, fontsize=14, frameon=False)
plt.gca().add_artist(line_legend)
period_patches = [
    Patch(facecolor='teal', alpha=0.2, label='Baseline (1990–2014)'),
    Patch(facecolor='gray', alpha=0.2, label='Near Future (2026–2050)'),
    Patch(facecolor='orange', alpha=0.2, label='Far Future (2051–2080)')
]
plt.legend(handles=period_patches, loc='lower center', ncol=3, fontsize=14, frameon=False)
plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutTempQDM-LSTMRes.png", dpi = 500)
plt.show()

#Use in paper too
LSTM_YearlyTpAdjRes = LSTM_FutTmpRes.resample('YE').mean()
ssp126 = LSTM_YearlyTpAdjRes[[col for col in LSTM_YearlyTpAdjRes.columns if 'SSP126' in col]]
ssp245 = LSTM_YearlyTpAdjRes[[col for col in LSTM_YearlyTpAdjRes.columns if 'SSP245' in col]]
ssp585 = LSTM_YearlyTpAdjRes[[col for col in LSTM_YearlyTpAdjRes.columns if 'SSP585' in col]]

ssp126_mean = ssp126.mean(axis=1)
ssp245_mean = ssp245.mean(axis=1)
ssp585_mean = ssp585.mean(axis=1)

ssp126_std = ssp126.std(axis=1)
ssp245_std = ssp245.std(axis=1)
ssp585_std = ssp585.std(axis=1)

plt.figure(figsize=(14, 5))

plt.axvspan(2027, 2050, color='gray', alpha=0.2)
plt.axvspan(2050, 2080, color='orange', alpha=0.2)

# SSP126
plt.plot(ssp126_mean.index.year, ssp126_mean, label='SSP126', color='tomato', linewidth=2)
plt.fill_between(ssp126_mean.index.year, ssp126_mean - ssp126_std, ssp126_mean + ssp126_std, color='tomato', alpha=0.2)

# SSP245
plt.plot(ssp245_mean.index.year, ssp245_mean, label='SSP245', color='fuchsia', linewidth=2)
plt.fill_between(ssp245_mean.index.year, ssp245_mean - ssp245_std, ssp245_mean + ssp245_std, color='fuchsia', alpha=0.2)

# SSP585
plt.plot(ssp585_mean.index.year, ssp585_mean, label='SSP585', color='indigo', linewidth=2)
plt.fill_between(ssp585_mean.index.year, ssp585_mean - ssp585_std, ssp585_mean + ssp585_std, color='indigo', alpha=0.2)

period_patches = [
    Patch(facecolor='gray', alpha=0.2, label='NF(2026–2050)'),
    Patch(facecolor='orange', alpha=0.2, label='FF(2051–2080)')
]

plt.title("Annual Reservoir Mean Temperature Multi-Model Ensemble (QDM-LSTM)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Temperature (°C)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

# Add both legends without frames
line_legend = plt.legend(loc='lower left', ncol=3, fontsize=16, frameon=False)
plt.legend(handles=period_patches, loc='lower right', ncol=2, fontsize=16, frameon=False)
plt.gca().add_artist(line_legend)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9FutPrecQDM-LSTM.png", dpi = 500)
plt.show()


