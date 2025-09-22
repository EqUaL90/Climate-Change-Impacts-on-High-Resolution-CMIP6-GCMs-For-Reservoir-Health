# Extract reservoir stations from basin prediction and target 
ResStnsPrcCol = ['P9', 'P12', 'P36', 'P39', 'P41', 'P43', 'P106', 'P111']
Res_predictions_by_ssp = {}

for key, df in predictions_by_ssp.items():
    filtered_df = df[ResStnsPrcCol]
    filtered_df = filtered_df.map(lambda x: round(x, 2))
    filtered_df = filtered_df.clip(lower=0)
    Res_predictions_by_ssp[key] = filtered_df

print(Res_predictions_by_ssp)
#Target
Res_DF_Pr_target_named_imputD = {}

for key, df in DF_Pr_target_named_imputD.items():
    filtered_df = df[ResStnsPrcCol]
    filtered_df = filtered_df.map(lambda x: round(x, 2))
    filtered_df = filtered_df.clip(lower=0)
    Res_DF_Pr_target_named_imputD[key] = filtered_df

print(Res_DF_Pr_target_named_imputD)

# === Resample to Annual Totals ===
# LSTM Future Projections (Bias-Corrected)
ssp126_qdm_annual = LSTM_FutPrcRes[[col for col in LSTM_FutPrcRes.columns if 'SSP126' in col]].resample('YE').sum()
ssp245_qdm_annual = LSTM_FutPrcRes[[col for col in LSTM_FutPrcRes.columns if 'SSP245' in col]].resample('YE').sum()
ssp585_qdm_annual = LSTM_FutPrcRes[[col for col in LSTM_FutPrcRes.columns if 'SSP585' in col]].resample('YE').sum()

ssp126_qdm_total = ssp126_qdm_annual.mean(axis=1)
ssp245_qdm_total = ssp245_qdm_annual.mean(axis=1)
ssp585_qdm_total = ssp585_qdm_annual.mean(axis=1)

ssp126_qdm_std = ssp126_qdm_annual.std(axis=1)
ssp245_qdm_std = ssp245_qdm_annual.std(axis=1)
ssp585_qdm_std = ssp585_qdm_annual.std(axis=1)

# Historical Baseline (Bias-Corrected)
BaselineBC_annual = BaselineBCEnsRes.resample('YE').sum()

# === Plotting ===
plt.figure(figsize=(14, 5))

# === Shaded Time Periods ===
plt.axvspan(pd.Timestamp("1990-12"), pd.Timestamp("2014-12"), color='teal', alpha=0.2)
plt.axvspan(pd.Timestamp("2027-12"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-1"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# === Historical Baseline ===
plt.plot(BaselineBC_annual.index, BaselineBC_annual['Baseline'], 
         label='Baseline (BC, Annual)', color='black', linewidth=2.5, linestyle='--')

# === SSP126 ===
plt.plot(ssp126_qdm_total.index, ssp126_qdm_total, label='SSP126 (QDM-LSTM)', color='blue', linewidth=2.5)
plt.fill_between(ssp126_qdm_total.index, ssp126_qdm_total - ssp126_qdm_std, 
                 ssp126_qdm_total + ssp126_qdm_std, color='blue', alpha=0.2)

# === SSP245 ===
plt.plot(ssp245_qdm_total.index, ssp245_qdm_total, label='SSP245 (QDM-LSTM)', color='green', linewidth=2.5)
plt.fill_between(ssp245_qdm_total.index, ssp245_qdm_total - ssp245_qdm_std, 
                 ssp245_qdm_total + ssp245_qdm_std, color='green', alpha=0.2)

# === SSP585 ===
plt.plot(ssp585_qdm_total.index, ssp585_qdm_total, label='SSP585 (QDM-LSTM)', color='red', linewidth=2.5)
plt.fill_between(ssp585_qdm_total.index, ssp585_qdm_total - ssp585_qdm_std, 
                 ssp585_qdm_total + ssp585_qdm_std, color='salmon', alpha=0.2)

# === Plot Formatting ===
plt.title("Annual Reservoir Precipitation Projections by MME (QDM-LSTM)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Precipitation (mm/year)", fontsize=18)
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
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutPrecQDM-LSTMRes.png", dpi = 500)
plt.show()

# === Resample to Annual Totals ===
# Raw CMIP6 Future Projections
ssp126_raw_annual = CMIP6_OldFutPrcRes[[col for col in CMIP6_OldFutPrcRes.columns if 'SSP126' in col]].resample('YE').sum()
ssp245_raw_annual = CMIP6_OldFutPrcRes[[col for col in CMIP6_OldFutPrcRes.columns if 'SSP245' in col]].resample('YE').sum()
ssp585_raw_annual = CMIP6_OldFutPrcRes[[col for col in CMIP6_OldFutPrcRes.columns if 'SSP585' in col]].resample('YE').sum()

ssp126_raw_annual_total = ssp126_raw_annual.mean(axis=1)
ssp245_raw_annual_total = ssp245_raw_annual.mean(axis=1)
ssp585_raw_annual_total = ssp585_raw_annual.mean(axis=1)

ssp126_raw_annual_std = ssp126_raw_annual.std(axis=1)
ssp245_raw_annual_std = ssp245_raw_annual.std(axis=1)
ssp585_raw_annual_std = ssp585_raw_annual.std(axis=1)

# Historical Baseline (Raw)
Baseline_annual = BaselineEnsRes.resample('YE').sum()

# === Plotting ===
plt.figure(figsize=(14, 5))

# === Shaded Time Periods ===
plt.axvspan(pd.Timestamp("1990-12"), pd.Timestamp("2014-12"), color='teal', alpha=0.2)
plt.axvspan(pd.Timestamp("2027-12"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-1"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# === Historical Baseline ===
plt.plot(Baseline_annual.index, Baseline_annual['Baseline'], 
         label='Baseline (BC, Annual)', color='black', linewidth=2.5, linestyle='--')

# === SSP126 ===
plt.plot(ssp126_raw_annual_total.index, ssp126_raw_annual_total, label='SSP126 (Raw)', color='blue', linewidth=2.5)
plt.fill_between(ssp126_raw_annual_total.index, ssp126_raw_annual_total - ssp126_raw_annual_std, 
                 ssp126_raw_annual_total + ssp126_raw_annual_std, color='blue', alpha=0.2)

# === SSP245 ===
plt.plot(ssp245_raw_annual_total.index, ssp245_raw_annual_total, label='SSP245 (Raw)', color='green', linewidth=2.5)
plt.fill_between(ssp245_raw_annual_total.index, ssp245_raw_annual_total - ssp245_raw_annual_std, 
                 ssp245_raw_annual_total + ssp245_raw_annual_std, color='green', alpha=0.2)

# === SSP585 ===
plt.plot(ssp585_raw_annual_total.index, ssp585_raw_annual_total, label='SSP585 (Raw)', color='red', linewidth=2.5)
plt.fill_between(ssp585_raw_annual_total.index, ssp585_raw_annual_total - ssp585_raw_annual_std, 
                 ssp585_raw_annual_total + ssp585_raw_annual_std, color='salmon', alpha=0.2)

# === Plot Formatting ===
plt.title("Annual Reservoir Precipitation Projections by MME (Raw)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Precipitation (mm/year)", fontsize=18)
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
#plt.legend(handles=period_patches, loc='lower center', ncol=3, fontsize=14, frameon=False)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutPrecOriginalCMIP6Res.png", dpi = 500)
plt.show()


from matplotlib.patches import Patch

# === LSTM-BASED PLOT ===

# Group columns by SSP
ssp126 = ResLSTM_Yearly[[col for col in ResLSTM_Yearly.columns if 'SSP126' in col]]
ssp245 = ResLSTM_Yearly[[col for col in ResLSTM_Yearly.columns if 'SSP245' in col]]
ssp585 = ResLSTM_Yearly[[col for col in ResLSTM_Yearly.columns if 'SSP585' in col]]

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
plt.title("Annual Reservoir Precipitation Multi-Model Ensemble (QDM-LSTM)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Precipitation (mm/year)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

# Add both legends
line_legend = plt.legend(loc='lower left', ncol = 3, fontsize=16, frameon = False)
plt.legend(handles=period_patches, loc='lower right', ncol =2, fontsize=16, frameon = False)
ax = plt.gca()
ax.add_artist(line_legend)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9FutPrecQDM-LSTMRes.png", dpi = 500)
plt.show()


# === RAW CMIP6 PLOT ===

ssp126_raw = ResCMIP6_Yearly[[col for col in ResCMIP6_Yearly.columns if 'SSP126' in col]]
ssp245_raw = ResCMIP6_Yearly[[col for col in ResCMIP6_Yearly.columns if 'SSP245' in col]]
ssp585_raw = ResCMIP6_Yearly[[col for col in ResCMIP6_Yearly.columns if 'SSP585' in col]]

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
plt.title("Annual Reservoir Precipitation Multi-Model Ensemble (Raw CMIP6)", fontsize=24)
plt.xlabel("Year", fontsize=22)
plt.ylabel("Precipitation (mm/year)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

# Add both legends
line_legend = plt.legend(loc='lower left', ncol = 3, fontsize=16, frameon = False)
plt.legend(handles=period_patches, loc='lower right', ncol =2, fontsize=16, frameon = False)
ax = plt.gca()
ax.add_artist(line_legend)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9FutPrecOriginalCMIP6Res.png", dpi = 500)
plt.show()
