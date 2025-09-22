#Baseline BC: DFHistACCPrcERAcorr, DFHistCanPrcERAcorr, DFHistGFDLPrcERAcorr
#Baseline Org: DFACCESS_CM2histMnthPrec1990_2014_regriddedF,DFCanESM5histMnthPrec1990_2014_regriddedF, DFGFDL_ESM4histMnthPrec1990_2014_regriddedF
#Org
mean_ACCTp = DFACCESS_CM2histMnthTemp1990_2014_regriddedF.mean(axis=1)
mean_CANTp =DFCanESM5histMnthTemp1990_2014_regriddedF.mean(axis=1)
mean_GFDLTp = DFGFDL_ESM4histMnthTemp1990_2014_regriddedF.mean(axis=1)


BaselineEnsTp = pd.concat([mean_ACCTp, mean_CANTp, mean_GFDLTp], axis=1)
BaselineEnsTp.columns = ['ACC', 'CAN', 'GFDL']

# Take the ensemble mean across the models
BaselineEnsTp['Baseline'] = BaselineEnsTp.mean(axis=1)
BaselineEnsFTp = BaselineEnsTp[["Baseline"]]
print(BaselineEnsTp)
print(BaselineEnsFTp.head(2))

#BC
meanBC_ACCTp  = DFHistACCTmpERAcorr.mean(axis=1)
meanBC_CANTp  = DFHistCanTmpERAcorr.mean(axis=1)
meanBC_GFDLTp = DFHistGFDLTmpERAcorr.mean(axis=1)


BaselineBCEnsTp = pd.concat([meanBC_ACCTp, meanBC_CANTp, meanBC_GFDLTp], axis=1)
BaselineBCEnsTp.columns = ['ACC', 'CAN', 'GFDL']

BaselineBCEnsTp['Baseline'] = BaselineBCEnsTp.mean(axis=1)
BaselineBCEnsFTp = BaselineBCEnsTp[["Baseline"]]
print(BaselineBCEnsTp)
BaselineBCEnsFTp.head(3)
########################

# === Resample to Annual Totals ===
# LSTM Future Projections (Bias-Corrected)
ssp126_qdm_annual = LSTM_FutTmp[[col for col in LSTM_FutTmp.columns if 'SSP126' in col]].resample('YE').mean()
ssp245_qdm_annual = LSTM_FutTmp[[col for col in LSTM_FutTmp.columns if 'SSP245' in col]].resample('YE').mean()
ssp585_qdm_annual = LSTM_FutTmp[[col for col in LSTM_FutTmp.columns if 'SSP585' in col]].resample('YE').mean()

ssp126_qdm_total = ssp126_qdm_annual.mean(axis=1)
ssp245_qdm_total = ssp245_qdm_annual.mean(axis=1)
ssp585_qdm_total = ssp585_qdm_annual.mean(axis=1)

ssp126_qdm_std = ssp126_qdm_annual.std(axis=1)
ssp245_qdm_std = ssp245_qdm_annual.std(axis=1)
ssp585_qdm_std = ssp585_qdm_annual.std(axis=1)

# Historical Baseline (Bias-Corrected)
BaselineBC_annualTp = BaselineBCEnsFTp.resample('YE').mean()

# === Plotting ===
plt.figure(figsize=(14, 5))

# === Shaded Time Periods ===
plt.axvspan(pd.Timestamp("1990-12"), pd.Timestamp("2014-12"), color='teal', alpha=0.2)
plt.axvspan(pd.Timestamp("2027-12"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-1"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# === Historical Baseline ===
plt.plot(BaselineBC_annualTp.index, BaselineBC_annualTp['Baseline'], 
         label='Baseline (BC, Annual)', color='black', linewidth=2.2, linestyle='--')

# === SSP126 ===
plt.plot(ssp126_qdm_total.index, ssp126_qdm_total, label='SSP126 (QDM-LSTM)', color='gray', linewidth=1.5)
plt.fill_between(ssp126_qdm_total.index, ssp126_qdm_total - ssp126_qdm_std, 
                 ssp126_qdm_total + ssp126_qdm_std, color='blue', alpha=0.2)

# === SSP245 ===
plt.plot(ssp245_qdm_total.index, ssp245_qdm_total, label='SSP245 (QDM-LSTM)', color='pink', linewidth=1.5)
plt.fill_between(ssp245_qdm_total.index, ssp245_qdm_total - ssp245_qdm_std, 
                 ssp245_qdm_total + ssp245_qdm_std, color='green', alpha=0.2)

# === SSP585 ===
plt.plot(ssp585_qdm_total.index, ssp585_qdm_total, label='SSP585 (QDM-LSTM)', color='purple', linewidth=1.5)
plt.fill_between(ssp585_qdm_total.index, ssp585_qdm_total - ssp585_qdm_std, 
                 ssp585_qdm_total + ssp585_qdm_std, color='salmon', alpha=0.2)

# === Plot Formatting ===
plt.title("Annual Temperature Projections by MME (QDM-LSTM)", fontsize=24)
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
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutTempQDM-LSTM.png", dpi = 500)
plt.show()


# === Resample to Annual Totals ===
# Raw CMIP6 Future Projections
ssp126_raw_annual = CMIP6_OldFutTmp[[col for col in CMIP6_OldFutTmp.columns if 'SSP126' in col]].resample('YE').mean()
ssp245_raw_annual = CMIP6_OldFutTmp[[col for col in CMIP6_OldFutTmp.columns if 'SSP245' in col]].resample('YE').mean()
ssp585_raw_annual = CMIP6_OldFutTmp[[col for col in CMIP6_OldFutTmp.columns if 'SSP585' in col]].resample('YE').mean()

ssp126_raw_annual_total = ssp126_raw_annual.mean(axis=1)
ssp245_raw_annual_total = ssp245_raw_annual.mean(axis=1)
ssp585_raw_annual_total = ssp585_raw_annual.mean(axis=1)

ssp126_raw_annual_std = ssp126_raw_annual.std(axis=1)
ssp245_raw_annual_std = ssp245_raw_annual.std(axis=1)
ssp585_raw_annual_std = ssp585_raw_annual.std(axis=1)

# Historical Baseline (Bias-Corrected)
Baseline_annualTp = BaselineEnsFTp.resample('YE').mean()

# === Plotting ===
plt.figure(figsize=(14, 5))

# === Shaded Time Periods ===
plt.axvspan(pd.Timestamp("1990-12"), pd.Timestamp("2014-12"), color='teal', alpha=0.2)
plt.axvspan(pd.Timestamp("2027-12"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-1"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# === Historical Baseline ===
plt.plot(Baseline_annualTp.index, Baseline_annualTp['Baseline'], 
         label='Baseline (BC, Annual)', color='black', linewidth=2.2, linestyle='--')

# === SSP126 ===
plt.plot(ssp126_raw_annual_total.index, ssp126_raw_annual_total, label='SSP126', color='gray', linewidth=1.5)
plt.fill_between(ssp126_raw_annual_total.index, ssp126_raw_annual_total - ssp126_raw_annual_std, 
                 ssp126_raw_annual_total + ssp126_raw_annual_std, color='blue', alpha=0.2)

# === SSP245 ===
plt.plot(ssp245_raw_annual_total.index, ssp245_raw_annual_total, label='SSP245', color='pink', linewidth=1.5)
plt.fill_between(ssp245_raw_annual_total.index, ssp245_raw_annual_total - ssp245_raw_annual_std, 
                 ssp245_raw_annual_total + ssp245_raw_annual_std, color='green', alpha=0.2)

# === SSP585 ===
plt.plot(ssp585_raw_annual_total.index, ssp585_raw_annual_total, label='SSP585', color='purple', linewidth=1.5)
plt.fill_between(ssp585_raw_annual_total.index, ssp585_raw_annual_total - ssp585_raw_annual_std, 
                 ssp585_raw_annual_total + ssp585_raw_annual_std, color='salmon', alpha=0.2)

# === Plot Formatting ===
plt.title("Annual Temperature Projections by MME", fontsize=24)
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
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutTempOriginalCMIP6.png", dpi = 500)
plt.show()


###############################
from matplotlib.patches import Patch

# === LSTM-BASED PLOT ===

# Group columns by SSP
ssp126 = LSTM_YearlyTp[[col for col in LSTM_YearlyTp.columns if 'SSP126' in col]]
ssp245 = LSTM_YearlyTp[[col for col in LSTM_YearlyTp.columns if 'SSP245' in col]]
ssp585 = LSTM_YearlyTp[[col for col in LSTM_YearlyTp.columns if 'SSP585' in col]]

# Compute multi-model means and std
ssp126_mean = ssp126.mean(axis=1)
ssp245_mean = ssp245.mean(axis=1)
ssp585_mean = ssp585.mean(axis=1)

ssp126_std = ssp126.std(axis=1)
ssp245_std = ssp245.std(axis=1)
ssp585_std = ssp585.std(axis=1)

# Plot
plt.figure(figsize=(14, 5))

# Add shaded time periods
plt.axvspan(2027, 2050, color='gray', alpha=0.2)
plt.axvspan(2050, 2080, color='orange', alpha=0.2)

# SSP126
plt.plot(ssp126_mean.index.year, ssp126_mean, label='SSP126', color='blue', linewidth=2)
plt.fill_between(ssp126_mean.index.year, ssp126_mean - ssp126_std, ssp126_mean + ssp126_std, color='blue', alpha=0.2)

# SSP245
plt.plot(ssp245_mean.index.year, ssp245_mean, label='SSP245', color='green', linewidth=2)
plt.fill_between(ssp245_mean.index.year, ssp245_mean - ssp245_std, ssp245_mean + ssp245_std, color='green', alpha=0.2)

# SSP585
plt.plot(ssp585_mean.index.year, ssp585_mean, label='SSP585', color='red', linewidth=2)
plt.fill_between(ssp585_mean.index.year, ssp585_mean - ssp585_std, ssp585_mean + ssp585_std, color='salmon', alpha=0.2)

# Custom legend patches for periods
period_patches = [
    Patch(facecolor='gray', alpha=0.2, label='NF(2026–2050)'),
    Patch(facecolor='orange', alpha=0.2, label='FF(2051–2080)')
]

# Title and labels
plt.title("Annual Mean Temperature Multi-Model Ensemble (QDM-LSTM)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Temperature (°C)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

# Add both legends
line_legend = plt.legend(loc='lower left', ncol = 3, fontsize=16, frameon = False)
plt.legend(handles=period_patches, loc='lower right', ncol =2, fontsize=16, frameon = False)
ax = plt.gca()
ax.add_artist(line_legend)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9FutPrecQDM-LSTM.png", dpi = 500)
plt.show()


# === RAW CMIP6 PLOT ===

ssp126_raw = CMIP6_YearlyTp[[col for col in CMIP6_YearlyTp.columns if 'SSP126' in col]]
ssp245_raw = CMIP6_YearlyTp[[col for col in CMIP6_YearlyTp.columns if 'SSP245' in col]]
ssp585_raw = CMIP6_YearlyTp[[col for col in CMIP6_YearlyTp.columns if 'SSP585' in col]]

ssp126_raw_mean = ssp126_raw.mean(axis=1)
ssp245_raw_mean = ssp245_raw.mean(axis=1)
ssp585_raw_mean = ssp585_raw.mean(axis=1)

ssp126_raw_std = ssp126_raw.std(axis=1)
ssp245_raw_std = ssp245_raw.std(axis=1)
ssp585_raw_std = ssp585_raw.std(axis=1)

plt.figure(figsize=(14, 5))

# Add shaded time periods
plt.axvspan(2027, 2050, color='gray', alpha=0.2)
plt.axvspan(2050, 2080, color='orange', alpha=0.2)
# SSP126
plt.plot(ssp126_raw_mean.index.year, ssp126_raw_mean, label='SSP126', color='blue', linewidth=2)
plt.fill_between(ssp126_raw_mean.index.year,
                 ssp126_raw_mean - ssp126_raw_std,
                 ssp126_raw_mean + ssp126_raw_std,
                 color='blue', alpha=0.2)

# SSP245
plt.plot(ssp245_raw_mean.index.year, ssp245_raw_mean, label='SSP245', color='green', linewidth=2)
plt.fill_between(ssp245_raw_mean.index.year,
                 ssp245_raw_mean - ssp245_raw_std,
                 ssp245_raw_mean + ssp245_raw_std,
                 color='green', alpha=0.2)

# SSP585
plt.plot(ssp585_raw_mean.index.year, ssp585_raw_mean, label='SSP585', color='red', linewidth=2)
plt.fill_between(ssp585_raw_mean.index.year,
                 ssp585_raw_mean - ssp585_raw_std,
                 ssp585_raw_mean + ssp585_raw_std,
                 color='red', alpha=0.2)

# Custom legend patches reused
plt.title("Annual Mean Temperature Multi-Model Ensemble (Raw CMIP6)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Temperature (°C)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

# Add both legends
line_legend = plt.legend(loc='lower left', ncol = 3, fontsize=16, frameon = False)
plt.legend(handles=period_patches, loc='lower right', ncol =2, fontsize=16, frameon = False)
ax = plt.gca()
ax.add_artist(line_legend)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9FutPrecOriginalCMIP6.png", dpi = 500)
plt.show()
########################################

#####Use this in paper
LSTMadjCMIP6_OldFutTmp = CMIP6_OldFutTmp
BaselineEnsFTpadj = BaselineEnsFTp
ssp126_raw_annual = LSTMadjCMIP6_OldFutTmp[[col for col in LSTMadjCMIP6_OldFutTmp.columns if 'SSP126' in col]].resample('YE').mean()
ssp245_raw_annual = LSTMadjCMIP6_OldFutTmp[[col for col in LSTMadjCMIP6_OldFutTmp.columns if 'SSP245' in col]].resample('YE').mean()
ssp585_raw_annual = LSTMadjCMIP6_OldFutTmp[[col for col in LSTMadjCMIP6_OldFutTmp.columns if 'SSP585' in col]].resample('YE').mean()

ssp126_raw_annual_total = ssp126_raw_annual.mean(axis=1)
ssp245_raw_annual_total = ssp245_raw_annual.mean(axis=1)
ssp585_raw_annual_total = ssp585_raw_annual.mean(axis=1)

ssp126_raw_annual_std = ssp126_raw_annual.std(axis=1)
ssp245_raw_annual_std = ssp245_raw_annual.std(axis=1)
ssp585_raw_annual_std = ssp585_raw_annual.std(axis=1)

# Historical Baseline (Bias-Corrected)
Baseline_annualTp = BaselineEnsFTpadj.resample('YE').mean()

# === Plotting ===
plt.figure(figsize=(14, 5))

# === Shaded Time Periods ===
plt.axvspan(pd.Timestamp("1990-12"), pd.Timestamp("2014-12"), color='teal', alpha=0.2)
plt.axvspan(pd.Timestamp("2027-12"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-1"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# === Historical Baseline ===
plt.plot(Baseline_annualTp.index, Baseline_annualTp['Baseline'], 
         label='Baseline (BC, Annual)', color='black', linewidth=2.5, linestyle='--')

# === SSP126 ===
plt.plot(ssp126_raw_annual_total.index, ssp126_raw_annual_total, label='SSP126 (QDM-LSTM)', color='tomato', linewidth=2.5)
plt.fill_between(ssp126_raw_annual_total.index, ssp126_raw_annual_total - ssp126_raw_annual_std, 
                 ssp126_raw_annual_total + ssp126_raw_annual_std, color='tomato', alpha=0.2)

# === SSP245 ===
plt.plot(ssp245_raw_annual_total.index, ssp245_raw_annual_total, label='SSP245 (QDM-LSTM)', color='fuchsia', linewidth=2.5)
plt.fill_between(ssp245_raw_annual_total.index, ssp245_raw_annual_total - ssp245_raw_annual_std, 
                 ssp245_raw_annual_total + ssp245_raw_annual_std, color='fuchsia', alpha=0.2)

# === SSP585 ===
plt.plot(ssp585_raw_annual_total.index, ssp585_raw_annual_total, label='SSP585 (QDM-LSTM)', color='indigo', linewidth=2.5)
plt.fill_between(ssp585_raw_annual_total.index, ssp585_raw_annual_total - ssp585_raw_annual_std, 
                 ssp585_raw_annual_total + ssp585_raw_annual_std, color='indigo', alpha=0.2)

# === Plot Formatting ===
plt.title("Annual Temperature Projections by MME (QDM-LSTM)", fontsize=24)
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
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutTempQDM-LSTM.png", dpi = 500)
plt.show()#########################
