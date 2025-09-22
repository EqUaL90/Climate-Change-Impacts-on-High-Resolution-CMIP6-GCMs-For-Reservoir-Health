#Baseline BC: DFHistACCPrcERAcorr, DFHistCanPrcERAcorr, DFHistGFDLPrcERAcorr
#Baseline Org: DFACCESS_CM2histMnthPrec1990_2014_regriddedF,DFCanESM5histMnthPrec1990_2014_regriddedF, DFGFDL_ESM4histMnthPrec1990_2014_regriddedF
#Org
mean_ACC = DFACCESS_CM2histMnthPrec1990_2014_regriddedF.mean(axis=1)
mean_CAN =DFCanESM5histMnthPrec1990_2014_regriddedF.mean(axis=1)
mean_GFDL = DFGFDL_ESM4histMnthPrec1990_2014_regriddedF.mean(axis=1)


BaselineEns = pd.concat([mean_ACC, mean_CAN, mean_GFDL], axis=1)
BaselineEns.columns = ['ACC', 'CAN', 'GFDL']

# Take the ensemble mean across the models
BaselineEns['Baseline'] = BaselineEns.mean(axis=1)
BaselineEnsF = BaselineEns[["Baseline"]]
print(BaselineEns)
print(BaselineEnsF.head(2))

#BC
meanBC_ACC = DFHistACCPrcERAcorr.mean(axis=1)
meanBC_CAN = DFHistCanPrcERAcorr.mean(axis=1)
meanBC_GFDL = DFHistGFDLPrcERAcorr.mean(axis=1)


BaselineBCEns = pd.concat([meanBC_ACC, meanBC_CAN, meanBC_GFDL], axis=1)
BaselineBCEns.columns = ['ACC', 'CAN', 'GFDL']

BaselineBCEns['Baseline'] = BaselineBCEns.mean(axis=1)
BaselineBCEnsF = BaselineBCEns[["Baseline"]]
print(BaselineBCEns)
BaselineBCEnsF.head(3)

#N
from matplotlib.patches import Patch
# === LSTM FUTURE PROJECTIONS (QDM-LSTM) ===

# Group columns by SSP scenario
ssp126 = LSTM_FutPrc[[col for col in LSTM_FutPrc.columns if 'SSP126' in col]]
ssp245 = LSTM_FutPrc[[col for col in LSTM_FutPrc.columns if 'SSP245' in col]]
ssp585 = LSTM_FutPrc[[col for col in LSTM_FutPrc.columns if 'SSP585' in col]]

# Calculate mean and standard deviation for monthly data
ssp126_mean = ssp126.mean(axis=1)
ssp245_mean = ssp245.mean(axis=1)
ssp585_mean = ssp585.mean(axis=1)

ssp126_std = ssp126.std(axis=1)
ssp245_std = ssp245.std(axis=1)
ssp585_std = ssp585.std(axis=1)

# Plot
plt.figure(figsize=(14, 5))

# Shaded periods
plt.axvspan(pd.Timestamp("2026-01"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-01"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# SSP126
plt.plot(ssp126_mean.index, ssp126_mean, label='SSP126', color='blue', linewidth=1.5)
plt.fill_between(ssp126_mean.index, ssp126_mean - ssp126_std, ssp126_mean + ssp126_std, color='blue', alpha=0.2)

# SSP245
plt.plot(ssp245_mean.index, ssp245_mean, label='SSP245', color='green', linewidth=1.5)
plt.fill_between(ssp245_mean.index, ssp245_mean - ssp245_std, ssp245_mean + ssp245_std, color='green', alpha=0.2)

# SSP585
plt.plot(ssp585_mean.index, ssp585_mean, label='SSP585', color='red', linewidth=1.5)
plt.fill_between(ssp585_mean.index, ssp585_mean - ssp585_std, ssp585_mean + ssp585_std, color='salmon', alpha=0.2)

# Custom patches
period_patches = [
    Patch(facecolor='gray', alpha=0.2, label='NF(2026–2050)'),
    Patch(facecolor='orange', alpha=0.2, label='FF(2051–2080)')
]

plt.title("Monthly Precipitation Multi-Model Ensemble (QDM-LSTM)", fontsize=24)
plt.xlabel("Date", fontsize=22)
plt.ylabel("Precipitation (mm/month)", fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

line_legend = plt.legend(loc='lower left', ncol = 3, fontsize=14)
plt.legend(handles=period_patches, loc='lower right', ncol =2, fontsize=14)

plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\8FutPrecQDM-LSTMMonthly.png", dpi = 500)
plt.show()


# === RAW CMIP6 MONTHLY PLOT ===

ssp126_raw = CMIP6_OldFutPrc[[col for col in CMIP6_OldFutPrc.columns if 'SSP126' in col]]
ssp245_raw = CMIP6_OldFutPrc[[col for col in CMIP6_OldFutPrc.columns if 'SSP245' in col]]
ssp585_raw = CMIP6_OldFutPrc[[col for col in CMIP6_OldFutPrc.columns if 'SSP585' in col]]

ssp126_raw_mean = ssp126_raw.mean(axis=1)
ssp245_raw_mean = ssp245_raw.mean(axis=1)
ssp585_raw_mean = ssp585_raw.mean(axis=1)

ssp126_raw_std = ssp126_raw.std(axis=1)
ssp245_raw_std = ssp245_raw.std(axis=1)
ssp585_raw_std = ssp585_raw.std(axis=1)

plt.figure(figsize=(14, 5))

plt.axvspan(pd.Timestamp("2026-01"), pd.Timestamp("2050-12"), color='gray', alpha=0.2)
plt.axvspan(pd.Timestamp("2051-01"), pd.Timestamp("2080-12"), color='orange', alpha=0.2)

# SSP126
plt.plot(ssp126_raw_mean.index, ssp126_raw_mean, label='SSP126', color='blue', linewidth=1.5)
plt.fill_between(ssp126_raw_mean.index,
                 ssp126_raw_mean - ssp126_raw_std,
                 ssp126_raw_mean + ssp126_raw_std,
                 color='blue', alpha=0.2)

# SSP245
plt.plot(ssp245_raw_mean.index, ssp245_raw_mean, label='SSP245', color='green', linewidth=1.5)
plt.fill_between(ssp245_raw_mean.index,
                 ssp245_raw_mean - ssp245_raw_std,
                 ssp245_raw_mean + ssp245_raw_std,
                 color='green', alpha=0.2)

# SSP585
plt.plot(ssp585_raw_mean.index, ssp585_raw_mean, label='SSP585', color='red', linewidth=1.5)
plt.fill_between(ssp585_raw_mean.index,
                 ssp585_raw_mean - ssp585_raw_std,
                 ssp585_raw_mean + ssp585_raw_std,
                 color='red', alpha=0.2)

plt.title("Monthly Precipitation Multi-Model Ensemble (Raw CMIP6)", fontsize=24)
plt.xlabel("Date", fontsize=22)
plt.ylabel("Precipitation (mm/month)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

line_legend = plt.legend(loc='lower left', ncol = 3, fontsize=14)
plt.legend(handles=period_patches, loc='lower right', ncol =2, fontsize=14)
plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\8FutPrecOriginalCMIP6Monthly.png", dpi = 500)
plt.show()

# Helper function to plot R² correlation per SSP
def plot_r2_by_ssp(true_df, pred_df, ssp_tag, ax):
    cols = [col for col in true_df.columns if ssp_tag in col]
    true_vals = true_df[cols].mean(axis=1)
    pred_vals = pred_df[cols].mean(axis=1)

    sns.regplot(x=true_vals, y=pred_vals, ax=ax, ci=95,
                line_kws={"color": "blue"}, scatter_kws={"alpha": 0.6, "s": 20})

    r2 = r2_score(true_vals, pred_vals)
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.2f}",
            transform=ax.transAxes, fontsize=21, verticalalignment='top')

    ax.set_title(f"{ssp_tag}", fontsize=22)
    ax.set_xlabel("Raw Precip. SSP", fontsize=20)
    ax.set_ylabel("QDM-LSTM_Corrected", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(False)

# === Create 1x3 subplot ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ssp_list = ['SSP126', 'SSP245', 'SSP585']

for i, ssp in enumerate(ssp_list):
    plot_r2_by_ssp(CMIP6_OldFutPrc, LSTM_FutPrc, ssp, axes[i])

#plt.suptitle("R² Correlation: Raw CMIP6 vs QDM-LSTM (Monthly Precip)", fontsize=18, y=1.02)
plt.tight_layout()
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\13Correlation Plot of FuturePrecipCorr.png", dpi = 500)
plt.show()


# === Resample to Annual Totals ===
# LSTM Future Projections (Bias-Corrected)
ssp126_qdm_annual = LSTM_FutPrc[[col for col in LSTM_FutPrc.columns if 'SSP126' in col]].resample('YE').sum()
ssp245_qdm_annual = LSTM_FutPrc[[col for col in LSTM_FutPrc.columns if 'SSP245' in col]].resample('YE').sum()
ssp585_qdm_annual = LSTM_FutPrc[[col for col in LSTM_FutPrc.columns if 'SSP585' in col]].resample('YE').sum()

ssp126_qdm_total = ssp126_qdm_annual.mean(axis=1)
ssp245_qdm_total = ssp245_qdm_annual.mean(axis=1)
ssp585_qdm_total = ssp585_qdm_annual.mean(axis=1)

ssp126_qdm_std = ssp126_qdm_annual.std(axis=1)
ssp245_qdm_std = ssp245_qdm_annual.std(axis=1)
ssp585_qdm_std = ssp585_qdm_annual.std(axis=1)

# Historical Baseline (Bias-Corrected)
BaselineBC_annual = BaselineBCEnsF.resample('YE').sum()

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
plt.title("Annual Precipitation Projections by MME (QDM-LSTM)", fontsize=24)
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
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutPrecQDM-LSTM.png", dpi = 500)
plt.show()


# === Resample to Annual Totals ===
# Raw CMIP6 Future Projections
ssp126_raw_annual = CMIP6_OldFutPrc[[col for col in CMIP6_OldFutPrc.columns if 'SSP126' in col]].resample('YE').sum()
ssp245_raw_annual = CMIP6_OldFutPrc[[col for col in CMIP6_OldFutPrc.columns if 'SSP245' in col]].resample('YE').sum()
ssp585_raw_annual = CMIP6_OldFutPrc[[col for col in CMIP6_OldFutPrc.columns if 'SSP585' in col]].resample('YE').sum()

ssp126_raw_annual_total = ssp126_raw_annual.mean(axis=1)
ssp245_raw_annual_total = ssp245_raw_annual.mean(axis=1)
ssp585_raw_annual_total = ssp585_raw_annual.mean(axis=1)

ssp126_raw_annual_std = ssp126_raw_annual.std(axis=1)
ssp245_raw_annual_std = ssp245_raw_annual.std(axis=1)
ssp585_raw_annual_std = ssp585_raw_annual.std(axis=1)

# Historical Baseline (Bias-Corrected)
Baseline_annual = BaselineEnsF.resample('YE').sum()

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
plt.title("Annual Precipitation Projections by MME (Raw)", fontsize=24)
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
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\9HistFutPrecOriginalCMIP6.png", dpi = 500)
plt.show()
