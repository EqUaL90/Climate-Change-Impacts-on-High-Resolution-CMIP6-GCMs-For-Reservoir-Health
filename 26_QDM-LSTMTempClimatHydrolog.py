#Monthly basin temperature predictions
ssp126_qdm_mon = LSTMadjCMIP6_OldFutTmp[[col for col in LSTMadjCMIP6_OldFutTmp.columns if 'SSP126' in col]]
ssp245_qdm_mon = LSTMadjCMIP6_OldFutTmp[[col for col in LSTMadjCMIP6_OldFutTmp.columns if 'SSP245' in col]]
ssp585_qdm_mon = LSTMadjCMIP6_OldFutTmp[[col for col in LSTMadjCMIP6_OldFutTmp.columns if 'SSP585' in col]]

ssp126_qdm_montotal = ssp126_qdm_mon.mean(axis=1)
ssp245_qdm_montotal = ssp245_qdm_mon.mean(axis=1)
ssp585_qdm_montotal = ssp585_qdm_mon.mean(axis=1)

ssp126_qdm_monstd = ssp126_qdm_mon.std(axis=1)
ssp245_qdm_monstd = ssp245_qdm_mon.std(axis=1)
ssp585_qdm_monstd = ssp585_qdm_mon.std(axis=1)

# === MME Mean DataFrame ===
Bas_MME_Mean_MonthlyTp = pd.DataFrame({
    'SSP126': ssp126_qdm_montotal,
    'SSP245': ssp245_qdm_montotal,
    'SSP585': ssp585_qdm_montotal
}, index=ssp126_qdm_montotal.index).round(1)

# === MME Std DataFrame ===
Bas_MME_Std_MonthlyTp = pd.DataFrame({
    'SSP126': ssp126_qdm_monstd,
    'SSP245': ssp245_qdm_monstd,
    'SSP585': ssp585_qdm_monstd
}, index=ssp126_qdm_monstd.index).round(1)


#Reservoir Temp basin predictions
ssp126_qdm_monRes = LSTM_FutTmpRes[[col for col in LSTM_FutTmpRes.columns if 'SSP126' in col]]
ssp245_qdm_monRes = LSTM_FutTmpRes[[col for col in LSTM_FutTmpRes.columns if 'SSP245' in col]]
ssp585_qdm_monRes = LSTM_FutTmpRes[[col for col in LSTM_FutTmpRes.columns if 'SSP585' in col]]

ssp126_qdm_montotalRes = ssp126_qdm_monRes.mean(axis=1)
ssp245_qdm_montotalRes = ssp245_qdm_monRes.mean(axis=1)
ssp585_qdm_montotalRes = ssp585_qdm_monRes.mean(axis=1)

ssp126_qdm_monstdRes = ssp126_qdm_monRes.std(axis=1)
ssp245_qdm_monstdRes = ssp245_qdm_monRes.std(axis=1)
ssp585_qdm_monstdRes = ssp585_qdm_monRes.std(axis=1)

# === MME Mean DataFrame ===
Bas_MME_Mean_MonthlyResTp = pd.DataFrame({
    'SSP126': ssp126_qdm_montotalRes,
    'SSP245': ssp245_qdm_montotalRes,
    'SSP585': ssp585_qdm_montotalRes
}, index=ssp126_qdm_montotalRes.index).round(1)

# === MME Std DataFrame ===
Bas_MME_Std_MonthlyResTp = pd.DataFrame({
    'SSP126': ssp126_qdm_monstdRes,
    'SSP245': ssp245_qdm_monstdRes,
    'SSP585': ssp585_qdm_monstdRes
}, index=ssp126_qdm_monstdRes.index).round(1)
print (Bas_MME_Mean_MonthlyTp.head(3))
print(Bas_MME_Mean_MonthlyResTp.head(3))


#Use in paper
# Use this
# With title
title_font = 22
label_font = 20
tick_font = 18
legend_font = 18

sns.set(style="whitegrid", rc={
    'axes.titlesize': title_font,
    'axes.labelsize': label_font,
    'xtick.labelsize': tick_font,
    'ytick.labelsize': tick_font,
    'legend.fontsize': tick_font,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.titlecolor': 'black',
    'legend.edgecolor': 'black'
})

cols_to_melt = ['SSP126', 'SSP245', 'SSP585']

df_basin_melt = Bas_MME_Mean_Monthly.melt(
    id_vars=['Season', 'Temporal Period'], 
    value_vars=cols_to_melt,
    var_name='Scenario', 
    value_name='Precipitation'
)
df_basin_melt['Source'] = 'Basin'

Bas_MME_Mean_MonthlyRes_renamed = Bas_MME_Mean_MonthlyRes.rename(
    columns={
        'SSP126_LSTM': 'SSP126',
        'SSP245_LSTM': 'SSP245',
        'SSP585_LSTM': 'SSP585'
    }
)
df_reservoir_melt = Bas_MME_Mean_MonthlyRes_renamed.melt(
    id_vars=['Season', 'Temporal Period'], 
    value_vars=cols_to_melt,
    var_name='Scenario', 
    value_name='Precipitation'
)
df_reservoir_melt['Source'] = 'Reservoir'

df_combined = pd.concat([df_basin_melt, df_reservoir_melt], ignore_index=True)

df_basin_melt_tp = Bas_MME_Mean_MonthlyTp.melt(
    id_vars=['Season', 'Temporal Period'], 
    value_vars=cols_to_melt,
    var_name='Scenario', 
    value_name='Temperature'
)
df_basin_melt_tp['Source'] = 'Basin'

df_reservoir_melt_tp= Bas_MME_Mean_MonthlyResTp.melt(
    id_vars=['Season', 'Temporal Period'], 
    value_vars=cols_to_melt,
    var_name='Scenario', 
    value_name='Temperature'
)
df_reservoir_melt_tp['Source'] = 'Reservoir'

df_combinedTp = pd.concat([df_basin_melt_tp, df_reservoir_melt_tp], ignore_index=True)

fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharey='col')

df_near_dry = df_combined[(df_combined['Temporal Period'] == 'Near Future') & (df_combined['Season'] == 'Dry')]
df_near_wet = df_combined[(df_combined['Temporal Period'] == 'Near Future') & (df_combined['Season'] == 'Wet')]
df_far_dry = df_combined[(df_combined['Temporal Period'] == 'Far Future') & (df_combined['Season'] == 'Dry')]
df_far_wet = df_combined[(df_combined['Temporal Period'] == 'Far Future') & (df_combined['Season'] == 'Wet')]

sns.boxplot(data=df_near_dry, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[0, 0], showfliers=False)
axes[0, 0].set_title('NF - Dry Season', fontsize=title_font)
axes[0, 0].set_xlabel('SSP Scenario', fontsize=label_font)
axes[0, 0].set_ylabel('Precipitation (mm)', fontsize=label_font)
axes[0, 0].tick_params(labelsize=tick_font, colors='black')
axes[0, 0].legend(loc='upper left', frameon=False)
axes[0, 0].grid(False)

sns.boxplot(data=df_near_wet, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[0, 1], showfliers=False)
axes[0, 1].set_title('NF - Wet Season', fontsize=title_font)
axes[0, 1].set_xlabel('SSP Scenario', fontsize=label_font)
axes[0, 1].set_ylabel('Precipitation (mm)', fontsize=label_font)
axes[0, 1].tick_params(labelsize=tick_font, colors='black')
axes[0, 1].get_legend().remove()
axes[0, 1].grid(False)

sns.boxplot(data=df_far_dry, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[1, 0], showfliers=False)
axes[1, 0].set_title('FF - Dry Season', fontsize=title_font)
axes[1, 0].set_xlabel('SSP Scenario', fontsize=label_font)
axes[1, 0].set_ylabel('Precipitation (mm)', fontsize=label_font)
axes[1, 0].tick_params(labelsize=tick_font, colors='black')
axes[1, 0].get_legend().remove()
axes[1, 0].grid(False)

sns.boxplot(data=df_far_wet, x='Scenario', y='Precipitation', hue='Source', palette='Paired',
            hue_order=['Basin', 'Reservoir'], ax=axes[1, 1], showfliers=False)
axes[1, 1].set_title('FF - Wet Season', fontsize=title_font)
axes[1, 1].set_xlabel('SSP Scenario', fontsize=label_font)
axes[1, 1].set_ylabel('Precipitation (mm)', fontsize=label_font)
axes[1, 1].tick_params(labelsize=tick_font, colors='black')
axes[1, 1].get_legend().remove()
axes[1, 1].grid(False)

df_near_dry_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Near Future') & (df_combinedTp['Season'] == 'Dry')]
df_near_wet_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Near Future') & (df_combinedTp['Season'] == 'Wet')]
df_far_dry_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Far Future') & (df_combinedTp['Season'] == 'Dry')]
df_far_wet_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Far Future') & (df_combinedTp['Season'] == 'Wet')]

sns.boxplot(data=df_near_dry_tp, x='Scenario', y='Temperature', hue='Source', palette='rocket',
            hue_order=['Basin', 'Reservoir'], ax=axes[0, 2], showfliers=False)
axes[0, 2].set_title('NF - Dry Season', fontsize=title_font)
axes[0, 2].set_xlabel('SSP Scenario', fontsize=label_font)
axes[0, 2].set_ylabel('Temperature (°C)', fontsize=label_font)
axes[0, 2].tick_params(labelsize=tick_font, colors='black')
axes[0, 2].legend(loc='upper left', frameon=False)
axes[0, 2].grid(False)

sns.boxplot(data=df_near_wet_tp, x='Scenario', y='Temperature', hue='Source', palette='rocket',
            hue_order=['Basin', 'Reservoir'], ax=axes[0, 3], showfliers=False)
axes[0, 3].set_title('NF - Wet Season', fontsize=title_font)
axes[0, 3].set_xlabel('SSP Scenario', fontsize=label_font)
axes[0, 3].set_ylabel('Temperature (°C)', fontsize=label_font)
axes[0, 3].tick_params(labelsize=tick_font, colors='black')
axes[0, 3].get_legend().remove()
axes[0, 3].grid(False)

sns.boxplot(data=df_far_dry_tp, x='Scenario', y='Temperature', hue='Source', palette='rocket',
            hue_order=['Basin', 'Reservoir'], ax=axes[1, 2], showfliers=False)
axes[1, 2].set_title('FF - Dry Season', fontsize=title_font)
axes[1, 2].set_xlabel('SSP Scenario', fontsize=label_font)
axes[1, 2].set_ylabel('Temperature (°C)', fontsize=label_font)
axes[1, 2].tick_params(labelsize=tick_font, colors='black')
axes[1, 2].get_legend().remove()
axes[1, 2].grid(False)

sns.boxplot(data=df_far_wet_tp, x='Scenario', y='Temperature', hue='Source', palette='rocket',
            hue_order=['Basin', 'Reservoir'], ax=axes[1, 3], showfliers=False)
axes[1, 3].set_title('FF - Wet Season', fontsize=title_font)
axes[1, 3].set_xlabel('SSP Scenario', fontsize=label_font)
axes[1, 3].set_ylabel('Temperature (°C)', fontsize=label_font)
axes[1, 3].tick_params(labelsize=tick_font, colors='black')
axes[1, 3].get_legend().remove()
axes[1, 3].grid(False)

fig.suptitle("(a) Climatological projections of annual precipitation and temperature", fontweight = 'bold', fontsize=26, color='black')
plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\10ClimSeasonalTempBoxPlotComboNewWithTitle.png", dpi = 600)
plt.show()

#Flow Regime
def assign_flow_regime(month):
    if month in [6, 7, 8, 9, 10, 11]:
        return 'WF'
    elif month in [12, 1, 2, 3]:
        return 'BF'
    else:
        return None  # April and May are not included in the regimes as they are transitional periods
    
Bas_MME_Mean_MonthlyTp['FlowRegime'] = Bas_MME_Mean_MonthlyTp.index.month.map(assign_flow_regime)
Bas_MME_Mean_MonthlyResTp['FlowRegime'] = Bas_MME_Mean_MonthlyResTp.index.month.map(assign_flow_regime)

df_basin_melt = Bas_MME_Mean_MonthlyTp.melt(id_vars=['Season', 'Temporal Period', 'FlowRegime'], 
                                           var_name='Scenario', value_name='Temperature')
df_basin_melt['Source'] = 'Basin'

df_reservoir_melt =Bas_MME_Mean_MonthlyResTp.melt(id_vars=['Season', 'Temporal Period', 'FlowRegime'], 
                                                         var_name='Scenario', value_name='Temperature')
df_reservoir_melt['Source'] = 'Reservoir'

df_combinedTp = pd.concat([df_basin_melt, df_reservoir_melt], ignore_index=True)
df_combinedTp.head(2)

import matplotlib.pyplot as plt
import seaborn as sns

# Use in paper
title_font = 22
label_font = 18
tick_font = 16
legend_font = 16

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

# Subgrids for precipitation and temperature (2x2 each)
gs_precip = gs[0].subgridspec(2, 2, hspace=0.4, wspace=0.35)
gs_temp = gs[1].subgridspec(2, 2, hspace=0.4, wspace=0.35)

# Filter data for each combination
df_nf_bf = df_combined[(df_combined['Temporal Period'] == 'Near Future') & (df_combined['FlowRegime'] == 'BF')]
df_nf_wf = df_combined[(df_combined['Temporal Period'] == 'Near Future') & (df_combined['FlowRegime'] == 'WF')]
df_ff_bf = df_combined[(df_combined['Temporal Period'] == 'Far Future') & (df_combined['FlowRegime'] == 'BF')]
df_ff_wf = df_combined[(df_combined['Temporal Period'] == 'Far Future') & (df_combined['FlowRegime'] == 'WF')]

df_nf_bf_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Near Future') & (df_combinedTp['FlowRegime'] == 'BF')]
df_nf_wf_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Near Future') & (df_combinedTp['FlowRegime'] == 'WF')]
df_ff_bf_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Far Future') & (df_combinedTp['FlowRegime'] == 'BF')]
df_ff_wf_tp = df_combinedTp[(df_combinedTp['Temporal Period'] == 'Far Future') & (df_combinedTp['FlowRegime'] == 'WF')]

# Plot precipitation
axes_precip = [fig.add_subplot(gs_precip[i, j]) for i in range(2) for j in range(2)]
precip_dfs = [df_nf_bf, df_nf_wf, df_ff_bf, df_ff_wf]
precip_titles = ['NF - BF Period', 'NF - WF Period', 'FF - BF Period', 'FF - WF Period']

for i, (ax, df, title) in enumerate(zip(axes_precip, precip_dfs, precip_titles)):
    sns.boxplot(data=df, x='Scenario', y='Precipitation', hue='Source', palette='husl',
                hue_order=['Basin', 'Reservoir'], ax=ax, showfliers=False)
    ax.set_title(title, fontsize=title_font)
    ax.set_xlabel('SSP Scenario', fontsize=label_font)
    ax.set_ylabel('Precipitation (mm)', fontsize=label_font)
    ax.tick_params(labelsize=tick_font)
    ax.legend_.remove()
    ax.grid(False)

# Precipitation legend
handles_precip, labels_precip = axes_precip[0].get_legend_handles_labels()
fig.legend(handles_precip, labels_precip, loc='upper left', bbox_to_anchor=(0.25, 0.96), frameon = False, ncol=2, fontsize=legend_font)

# Plot temperature
axes_temp = [fig.add_subplot(gs_temp[i, j]) for i in range(2) for j in range(2)]
temp_dfs = [df_nf_bf_tp, df_nf_wf_tp, df_ff_bf_tp, df_ff_wf_tp]
temp_titles = ['NF - BF Period', 'NF - WF Period', 'FF - BF Period', 'FF - WF Period']

for i, (ax, df, title) in enumerate(zip(axes_temp, temp_dfs, temp_titles)):
    sns.boxplot(data=df, x='Scenario', y='Temperature', hue='Source', palette='Spectral',
                hue_order=['Basin', 'Reservoir'], ax=ax, showfliers=False)
    ax.set_title(title, fontsize=title_font)
    ax.set_xlabel('SSP Scenario', fontsize=label_font)
    ax.set_ylabel('Temperature (°C)', fontsize=label_font)
    ax.tick_params(labelsize=tick_font)
    ax.legend_.remove()
    ax.grid(False)

# Temperature legend
handles_temp, labels_temp = axes_temp[0].get_legend_handles_labels()
fig.legend(handles_temp, labels_temp, loc='upper right', bbox_to_anchor=(0.75, 0.96), frameon = False, ncol=2, fontsize=legend_font)

# Main title
fig.suptitle("(b) Hydrological projections of annual precipitation and temperature", fontweight='bold', fontsize=26, color='black')

plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.savefig(r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\ResMod\C4_ResBCMLdata\0Fig\10HydroSeasonalTempBoxPlotComboNewWithTitleCorrected.png", dpi = 600)
plt.show()

